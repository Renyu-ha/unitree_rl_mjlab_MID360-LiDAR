"""SGS over terrain and velocity intervals, with online command sampling."""

from dataclasses import asdict, dataclass, field, is_dataclass

import torch

from .deadzone_velocity_command import DeadzoneVelocityCommand, DeadzoneVelocityCommandCfg
from .sgs import SGSConfig, SuccessGuidedSampler, VelocityTrackingOutcome
from .velocity_buckets import make_velocity_buckets


def _signature(value):
  """A process-independent config signature (function reprs contain addresses)."""
  if is_dataclass(value):
    return _signature(asdict(value))
  if isinstance(value, dict):
    return {key: _signature(item) for key, item in value.items()}
  if isinstance(value, (tuple, list)):
    return tuple(_signature(item) for item in value)
  if callable(value):
    return f"{value.__module__}:{value.__qualname__}"
  return value


class SGSVelocityCommand(DeadzoneVelocityCommand):
  cfg: "SGSVelocityCommandCfg"

  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    c = cfg.sgs
    c.validate()
    terrain = env.scene.terrain
    if terrain is None or terrain.terrain_origins is None:
      raise ValueError("SGS requires generated terrain with tile origins")
    if cfg.resampling_time_range[0] < c.settle_s + c.min_segment_s:
      raise ValueError("SGS command intervals must allow settling and at least one scored segment")
    if cfg.resampling_time_range[1] < cfg.resampling_time_range[0]:
      raise ValueError("SGS command interval bounds are reversed")
    self.rows, self.cols = terrain.terrain_origins.shape[:2]
    self.buckets = make_velocity_buckets(cfg.ranges, c.velocity_subdivisions,
      heading_command=cfg.heading_command, device=env.device)
    self.bucket_count = self.buckets.size
    self.category_count = len(self.buckets.categories)
    self.bank_size = self.rows * self.cols * self.bucket_count
    tiles = torch.arange(self.rows * self.cols, device=env.device)
    groups = (tiles[:, None] * self.category_count + self.buckets.category_ids).flatten()
    self.sampler = SuccessGuidedSampler(self.bank_size, c, env.device, (env.cfg.seed or 0) + 9017,
      group_ids=groups)
    # Separate streams: choosing tasks must not consume command-sampling draws.
    self.command_generator = torch.Generator(device=env.device).manual_seed(c.bank_seed + (env.cfg.seed or 0))
    self.outcome = VelocityTrackingOutcome(env.num_envs, c, env.step_dt, env.device)
    self.config_ids = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
    self.profile_step = torch.zeros_like(self.config_ids)
    self._done = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    self._censored = torch.zeros_like(self._done)
    self.recorded = 0
    self.skipped = 0
    self.last_episodes = None

  def _resample(self, env_ids):
    if env_ids.numel() > 0:
      self.outcome.finish_segments(env_ids)
      self._resample_command(env_ids)
      self.command_counter[env_ids] += 1

  def _resample_command(self, env_ids):
    if bool((self.config_ids[env_ids] < 0).any()):
      raise RuntimeError("SGS curriculum must assign a configuration before command reset")
    p = self.config_ids[env_ids] % self.bucket_count
    phase = self.profile_step[env_ids] % 2
    lo, hi = self.buckets.lower[p, phase], self.buckets.upper[p, phase]
    random = torch.rand(env_ids.numel(), 5, device=self.device, generator=self.command_generator)
    command = lo + random[:, :3] * (hi - lo)
    mode = self.buckets.modes[p]
    standing = command.abs().sum(-1) == 0
    self.vel_command_b[env_ids] = command
    self.vel_command_w[env_ids] = command
    heading_range = self.cfg.ranges.heading or (-3.141592653589793, 3.141592653589793)
    self.heading_target[env_ids] = heading_range[0] + random[:, 3] * (heading_range[1] - heading_range[0])
    self.is_standing_env[env_ids] = standing
    self.is_heading_env[env_ids] = (mode == 2) & ~standing
    self.is_world_env[env_ids] = (mode == 1) & ~standing
    self.is_forward_env[env_ids] = False
    interval = self.cfg.resampling_time_range
    self.time_left[env_ids] = interval[0] + random[:, 4] * (interval[1] - interval[0])
    self.profile_step[env_ids] += 1
    self._apply_deadzone(env_ids)

  def observe_step(self):
    env = self._env
    self.last_episodes = None
    if self._joystick_enabled is not None and self._joystick_enabled.value:
      self._censored[self._joystick_get_env_idx()] = True
    # RSL-RL can shorten initial episodes by advancing only episode_length_buf.
    self._censored |= (self.outcome.steps == 0) & (env.episode_length_buf != 1)
    self.outcome.observe(self.command, self.robot.data.root_link_lin_vel_b,
                         self.robot.data.root_link_ang_vel_b)
    self.outcome.finite &= torch.isfinite(self.robot.data.root_link_pos_w).all(-1)
    self.outcome.finite &= torch.isfinite(self.robot.data.root_link_quat_w).all(-1)
    self._done.copy_(env.reset_buf)

  def on_reset(self, env_ids):
    env = self._env
    ids = torch.arange(self.num_envs, device=self.device)[env_ids] if isinstance(env_ids, slice) else env_ids
    if ids is None:
      ids = torch.arange(self.num_envs, device=self.device)
    if ids.numel() == 0:
      return {}
    self.outcome.finish_segments(ids)
    tm = env.termination_manager
    terminated = tm.terminated[ids]
    timed_out = tm.get_term("time_out")[ids]
    boundary = tm.get_term("terrain_out_of_bounds")[ids]
    full = self.outcome.steps[ids] >= env.max_episode_length
    manual = getattr(env, "_manual_reset_pending", torch.zeros_like(self._done))[ids]
    valid = ((self.config_ids[ids] >= 0) & self._done[ids] & ~self._censored[ids]
      & self.outcome.finite[ids] & ~boundary & ~manual & (terminated | (timed_out & full)))
    success = ~terminated & timed_out & full & self.outcome.success(ids)
    self.sampler.update(self.config_ids[ids][valid], success[valid])
    self.recorded += int(valid.sum().item())
    self.skipped += int(((self.config_ids[ids] >= 0) & ~valid).sum().item())
    denom = valid.sum().clamp(min=1)
    rms = torch.sqrt(self.outcome.total_error[ids] / self.outcome.total_samples[ids].clamp(min=1)[:, None])
    self.last_episodes = {"env_ids": ids.clone(), "config_ids": self.config_ids[ids].clone(), "valid": valid.clone(),
      "success": success.clone(), "terminated": terminated.clone(), "rms": rms.clone(),
      "command_rms": torch.sqrt(self.outcome.total_command[ids] /
        self.outcome.total_samples[ids].clamp(min=1)[:, None])}
    observed = self.sampler.counts > 0
    raw_rates = self.sampler.history.sum(1).float() / self.sampler.counts.clamp(max=self.cfg.sgs.history_length).clamp(min=1)
    stats = {"success_rate": (success & valid).sum().float() / denom,
      "valid_episodes": valid.sum(), "recorded_total": self.recorded, "skipped_total": self.skipped,
      "linear_rms": torch.where(valid, rms[:, 0], 0.0).sum() / denom,
      "yaw_rms": torch.where(valid, rms[:, 1], 0.0).sum() / denom,
      "bank_coverage": observed.float().mean(),
      "bank_success": torch.where(observed, raw_rates, 0.0).sum() / observed.sum().clamp(min=1),
      "group_coverage": (self.sampler.parent.counts > 0).float().mean(),
      "min_visits": self.sampler.counts.min(), "configurations": self.bank_size}
    self.config_ids[ids] = self.sampler.sample(ids.numel())
    tile = self.config_ids[ids] // self.bucket_count
    terrain = env.scene.terrain
    terrain.terrain_levels[ids] = tile // self.cols
    terrain.terrain_types[ids] = tile % self.cols
    terrain.env_origins[ids] = terrain.terrain_origins[tile // self.cols, tile % self.cols]
    self.outcome.reset(ids)
    self.profile_step[ids] = 0
    self._done[ids] = self._censored[ids] = False
    return stats

  def checkpoint_spec(self):
    events = getattr(self._env.cfg, "events", {})
    return {"version": 2, "distribution": "velocity_intervals", "shape": (self.rows, self.cols),
      "ranges": asdict(self.cfg.ranges), "buckets": self.buckets.description(),
      "interval": self.cfg.resampling_time_range, "episode_s": self._env.max_episode_length_s,
      "terrain": _signature(self._env.scene.terrain.cfg.terrain_generator),
      "reset": _signature(events["reset_base"].params) if "reset_base" in events else None}

  def sgs_state_dict(self):
    return {"spec": self.checkpoint_spec(), "sampler": self.sampler.state_dict(),
      "command_rng": self.command_generator.get_state().cpu().clone(),
      "recorded": self.recorded, "skipped": self.skipped}

  def load_sgs_state_dict(self, state):
    if state.get("spec") != self.checkpoint_spec():
      raise ValueError("SGS checkpoint task distribution differs (old fixed profiles are incompatible). "
        "Start a new run or use --env.commands.twist.warm-start=True to load weights with fresh statistics.")
    self.sampler.load_state_dict(state["sampler"])
    self.command_generator.set_state(state["command_rng"].cpu())
    self.recorded, self.skipped = state["recorded"], state["skipped"]
    # Simulator trajectories are not checkpointed by mjlab. Censor in-flight
    # episodes and assign fresh configurations at the next real env reset.
    self._censored.fill_(True)


@dataclass(kw_only=True)
class SGSVelocityCommandCfg(DeadzoneVelocityCommandCfg):
  sgs: SGSConfig = field(default_factory=SGSConfig)
  evaluation: bool = False
  warm_start: bool = False

  def build(self, env):
    return SGSVelocityCommand(self, env)


def sgs_tracking_step(env, env_ids, command_name="twist"):
  del env_ids
  env.command_manager.get_term(command_name).observe_step()


def sgs_reset_curriculum(env, env_ids, command_name="twist"):
  return env.command_manager.get_term(command_name).on_reset(env_ids)
