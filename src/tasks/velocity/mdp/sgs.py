"""Success-Guided Sampling and velocity tracking outcomes (arXiv:2610.12465).

Written locally from the paper's equations; no external project code is vendored.
The ring buffer preserves every simultaneous outcome, including duplicate IDs.
"""

from dataclasses import asdict, dataclass
import math

import torch


@dataclass
class SGSConfig:
  adaptive: bool = True
  bank_seed: int = 20261030
  velocity_subdivisions: int = 2
  history_length: int = 32
  group_prior_strength: float = 4.0
  target_success: float = 0.66
  concentration: float = 5.0
  temperature: float = 2.0
  epsilon: float = 1e-8
  uniform_mix: float = 0.1
  settle_s: float = 0.6
  min_segment_s: float = 1.0
  linear_rms: float = 0.2
  yaw_rms: float = 0.35
  relative_error: float = 0.5
  standing_linear_rms: float = 0.1
  standing_yaw_rms: float = 0.15

  def validate(self):
    if self.velocity_subdivisions < 1 or self.history_length < 1 or self.bank_seed < 0:
      raise ValueError("SGS requires positive subdivisions/history and a nonnegative seed")
    if not 0 <= self.target_success <= 1 or not 0 <= self.uniform_mix <= 1:
      raise ValueError("SGS target_success and uniform_mix must lie in [0, 1]")
    for name in ("concentration", "settle_s", "group_prior_strength"):
      if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
        raise ValueError(f"SGS {name} must be finite and nonnegative")
    for name in ("temperature", "epsilon", "min_segment_s", "linear_rms", "yaw_rms",
                 "relative_error", "standing_linear_rms", "standing_yaw_rms"):
      if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
        raise ValueError(f"SGS {name} must be finite and positive")
    if self.relative_error >= 1:
      raise ValueError("SGS relative_error must be below 1 to reject a stationary moving task")


class SuccessGuidedSampler:
  def __init__(self, size: int, cfg: SGSConfig, device="cpu", seed=0, *, group_ids=None):
    cfg.validate()
    if size < 1:
      raise ValueError("SGS bank cannot be empty")
    self.cfg, self.size = cfg, size
    self.history = torch.zeros(size, cfg.history_length, dtype=torch.bool, device=device)
    self.counts = torch.zeros(size, dtype=torch.long, device=device)
    self.generator = torch.Generator(device=device).manual_seed(seed)
    self.group_ids = None
    self.parent = None
    self.base_probs = torch.full((size,), 1.0 / size, device=device)
    if group_ids is not None:
      groups = torch.as_tensor(group_ids, device=device, dtype=torch.long)
      if groups.shape != (size,) or bool((groups < 0).any()):
        raise ValueError("SGS group IDs must identify every configuration")
      counts = torch.bincount(groups)
      if bool((counts == 0).any()):
        raise ValueError("SGS group IDs must be contiguous")
      self.group_ids = groups
      self.parent = SuccessGuidedSampler(counts.numel(), cfg, device, seed + 1)
      # Equal mass per terrain/category, then uniform within its speed sub-bins.
      # Standing has one bin, so it is not diluted by subdividing moving tasks.
      self.base_probs = 1.0 / (counts.numel() * counts[groups].float())

  def rates(self):
    n = self.counts.clamp(max=self.cfg.history_length)
    if self.parent is not None and self.cfg.group_prior_strength > 0:
      strength = self.cfg.group_prior_strength
      pn = self.parent.counts.clamp(max=self.cfg.history_length)
      prior = (self.parent.history.sum(1).float() + strength * self.cfg.target_success) / (pn + strength)
      return (self.history.sum(1).float() + strength * prior[self.group_ids]) / (n + strength)
    observed = self.history.sum(1).float() / n.clamp(min=1)
    return torch.where(n > 0, observed, self.cfg.target_success)

  def probabilities(self):
    if not self.cfg.adaptive:
      return self.base_probs.clone()
    p, c = self.rates(), self.cfg
    # Log-space evaluation avoids underflow for concentrated kernels.
    log_w = c.concentration * (c.target_success * torch.log(p + c.epsilon)
      + (1.0 - c.target_success) * torch.log(1.0 - p + c.epsilon))
    floor = torch.full_like(log_w, math.log(c.epsilon))
    logits = torch.logaddexp(torch.maximum(log_w, floor), floor)
    probs = torch.softmax(logits / max(c.temperature, 1.0) + self.base_probs.log(), dim=0)
    return (1.0 - c.uniform_mix) * probs + c.uniform_mix * self.base_probs

  def sample(self, count):
    return torch.multinomial(self.probabilities(), count, replacement=True, generator=self.generator)

  def update(self, config_ids, successes):
    ids = config_ids.to(device=self.counts.device, dtype=torch.long).flatten()
    outcomes = successes.to(device=self.counts.device, dtype=torch.bool).flatten()
    if ids.numel() != outcomes.numel():
      raise ValueError("SGS IDs and outcomes must have the same length")
    if ids.numel() == 0:
      return
    if bool(((ids < 0) | (ids >= self.size)).any()):
      raise ValueError("SGS configuration ID outside bank")
    if self.parent is not None:
      # Preserve input order across sibling bins before sorting fine IDs.
      self.parent.update(self.group_ids[ids], outcomes)
    order = torch.argsort(ids, stable=True)
    ids, outcomes = ids[order], outcomes[order]
    additions = torch.bincount(ids, minlength=self.size)
    starts = additions.cumsum(0) - additions
    rank = torch.arange(ids.numel(), device=ids.device) - starts[ids]
    slots = (self.counts[ids] + rank) % self.cfg.history_length
    # Only the newest H writes survive. The remaining (ID, slot) pairs are unique.
    keep = rank >= (additions[ids] - self.cfg.history_length)
    self.history[ids[keep], slots[keep]] = outcomes[keep]
    self.counts += additions

  def state_dict(self):
    return {"settings": asdict(self.cfg), "history": self.history.detach().cpu().clone(),
      "counts": self.counts.detach().cpu().clone(), "rng": self.generator.get_state().cpu().clone(),
      "groups": None if self.group_ids is None else self.group_ids.cpu().clone(),
      "parent": None if self.parent is None else self.parent.state_dict()}

  def load_state_dict(self, state):
    if state["settings"] != asdict(self.cfg) or state["history"].shape != self.history.shape:
      raise ValueError("SGS checkpoint settings/bank shape differ; use the original training configuration")
    groups = state.get("groups")
    if (groups is None) != (self.group_ids is None) or (
        groups is not None and not torch.equal(groups.cpu(), self.group_ids.cpu())):
      raise ValueError("SGS checkpoint groups differ")
    history, counts = state["history"], state["counts"]
    if counts.shape != self.counts.shape or bool((counts < 0).any()):
      raise ValueError("Invalid SGS checkpoint counters")
    if bool((history.sum(1) > counts.clamp(max=self.cfg.history_length)).any()):
      raise ValueError("Invalid SGS checkpoint history")
    self.history.copy_(history)
    self.counts.copy_(counts)
    self.generator.set_state(state["rng"].cpu())
    if self.parent is not None:
      self.parent.load_state_dict(state["parent"])


class VelocityTrackingOutcome:
  """One Boolean episode outcome, requiring every sufficiently long segment to pass.

  Inputs are effective body-frame commands held during the just-completed step.
  Heading feedback changing the command does not restart the settling timer.
  """
  def __init__(self, num_envs, cfg: SGSConfig, dt, device="cpu"):
    cfg.validate()
    self.cfg, self.dt = cfg, dt
    self.min_samples = math.ceil(cfg.min_segment_s / dt)
    self.settle_steps = math.ceil(cfg.settle_s / dt)
    self.age = torch.zeros(num_envs, dtype=torch.long, device=device)
    self.steps = torch.zeros_like(self.age)
    self.samples = torch.zeros_like(self.age)
    self.segments = torch.zeros_like(self.age)
    self.sums = torch.zeros(num_envs, 4, device=device)  # error xy, yaw; command xy, yaw
    self.total_error = torch.zeros(num_envs, 2, device=device)
    self.total_command = torch.zeros_like(self.total_error)
    self.total_samples = torch.zeros_like(self.age)
    self.passed = torch.ones(num_envs, dtype=torch.bool, device=device)
    self.finite = torch.ones_like(self.passed)

  def observe(self, command, actual_linear, actual_angular):
    errors = torch.stack(((command[:, :2] - actual_linear[:, :2]).square().sum(-1),
      (command[:, 2] - actual_angular[:, 2]).square(), command[:, :2].square().sum(-1),
      command[:, 2].square()), dim=1)
    finite = torch.isfinite(errors).all(-1)
    self.finite &= finite
    scored = (self.age >= self.settle_steps) & finite
    self.sums += torch.where(scored[:, None], errors, 0.0)
    self.samples += scored
    self.age += 1
    self.steps += 1

  def finish_segments(self, ids):
    n = self.samples[ids]
    rms = torch.sqrt(self.sums[ids] / n.clamp(min=1)[:, None])
    c = self.cfg
    lin_limit = torch.where(rms[:, 2] > 0,
      (c.relative_error * rms[:, 2]).clamp(max=c.linear_rms), c.standing_linear_rms)
    yaw_limit = torch.where(rms[:, 3] > 0,
      (c.relative_error * rms[:, 3]).clamp(max=c.yaw_rms), c.standing_yaw_rms)
    valid = n >= self.min_samples
    self.passed[ids] &= ~valid | ((rms[:, 0] < lin_limit) & (rms[:, 1] < yaw_limit))
    self.segments[ids] += valid
    self.total_error[ids] += self.sums[ids, :2]
    self.total_command[ids] += self.sums[ids, 2:]
    self.total_samples[ids] += n
    self.samples[ids] = 0
    self.sums[ids] = 0
    self.age[ids] = 0

  def success(self, ids):
    return self.passed[ids] & self.finite[ids] & (self.segments[ids] > 0)

  def reset(self, ids):
    self.age[ids] = self.steps[ids] = self.samples[ids] = self.segments[ids] = 0
    self.sums[ids] = self.total_error[ids] = 0
    self.total_command[ids] = 0
    self.total_samples[ids] = 0
    self.passed[ids] = self.finite[ids] = True
