"""A fixed terrain/spawn/command-profile bank for Go2W SGS experiments."""

from dataclasses import asdict, dataclass, field
import math

import torch
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul

from .deadzone_velocity_command import DeadzoneVelocityCommand, DeadzoneVelocityCommandCfg
from .sgs import SGSConfig, SuccessGuidedSampler, VelocityTrackingOutcome


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
    if cfg.ranges.lin_vel_x[0] > -0.1 or cfg.ranges.lin_vel_x[1] < 0.1:
      raise ValueError("SGS forward/backward profiles require both signs with magnitude at least 0.1")
    self.rows, self.cols = terrain.terrain_origins.shape[:2]
    self.bank_size = self.rows * self.cols * c.profiles_per_tile
    self.profile_length = math.ceil(env.max_episode_length_s / cfg.resampling_time_range[0]) + 1
    self._make_bank()
    self.sampler = SuccessGuidedSampler(self.bank_size, c, env.device, (env.cfg.seed or 0) + 9017)
    self.outcome = VelocityTrackingOutcome(env.num_envs, c, env.step_dt, env.device)
    self.config_ids = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
    self.profile_step = torch.zeros_like(self.config_ids)
    self._done = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    self._censored = torch.zeros_like(self._done)
    self.recorded = 0
    self.skipped = 0
    self.last_episodes = None

  def _make_bank(self):
    c = self.cfg.sgs
    g = torch.Generator().manual_seed(c.bank_seed)
    shape = (c.profiles_per_tile, self.profile_length)
    ranges = self.cfg.ranges
    bounds = torch.tensor([ranges.lin_vel_x, ranges.lin_vel_y, ranges.ang_vel_z])
    commands = bounds[:, 0] + torch.rand(*shape, 3, generator=g) * (bounds[:, 1] - bounds[:, 0])
    heading_range = ranges.heading or (-math.pi, math.pi)
    heading = heading_range[0] + torch.rand(*shape, generator=g) * (heading_range[1] - heading_range[0])
    flags = torch.rand(*shape, 4, generator=g) < torch.tensor([
      self.cfg.rel_standing_envs, self.cfg.rel_heading_envs,
      self.cfg.rel_world_envs, self.cfg.rel_forward_envs])
    flags[..., 1] &= self.cfg.heading_command
    # Guarantee stand, forward, lateral, turn and backward profiles in every tile.
    # Other profiles retain the original command modes, including heading feedback.
    flags[:5] = False
    commands[0] = 0
    flags[0, :, 0] = True
    commands[1, :, 0] = 0.1 + torch.rand(self.profile_length, generator=g) * max(ranges.lin_vel_x[1] - 0.1, 0)
    commands[1, :, 1:] = 0
    commands[2, :, 0] = commands[2, :, 2] = 0
    commands[2, :, 1] = self._nonzero(commands[2, :, 1], ranges.lin_vel_y)
    commands[3, :, :2] = 0
    commands[3, :, 2] = self._nonzero(commands[3, :, 2], ranges.ang_vel_z)
    commands[4, :, 0] = -(0.1 + torch.rand(self.profile_length, generator=g) * max(-ranges.lin_vel_x[0] - 0.1, 0))
    commands[4, :, 1:] = 0
    durations = self.cfg.resampling_time_range[0] + torch.rand(*shape, generator=g) * (
      self.cfg.resampling_time_range[1] - self.cfg.resampling_time_range[0])
    spawn = torch.rand(self.bank_size, 3, generator=g) * 2 - 1
    spawn[:, :2] *= 0.5
    spawn[:, 2] *= math.pi
    self.bank = {"commands": commands.to(self.device), "heading": heading.to(self.device),
      "flags": flags.to(self.device), "durations": durations.to(self.device), "spawn": spawn.to(self.device)}

  @staticmethod
  def _nonzero(values, bounds):
    if bounds[0] > -0.1 or bounds[1] < 0.1:
      raise ValueError("SGS axis profiles require both signs with magnitude at least 0.1")
    return torch.where(values >= 0, values.clamp(min=0.1), values.clamp(max=-0.1))

  def _resample(self, env_ids):
    if env_ids.numel() > 0:
      self.outcome.finish_segments(env_ids)
      self._resample_command(env_ids)
      self.command_counter[env_ids] += 1

  def _resample_command(self, env_ids):
    if bool((self.config_ids[env_ids] < 0).any()):
      raise RuntimeError("SGS curriculum must assign a configuration before command reset")
    p = self.config_ids[env_ids] % self.cfg.sgs.profiles_per_tile
    s = self.profile_step[env_ids].clamp(max=self.profile_length - 1)
    command = self.bank["commands"][p, s].clone()
    flags = self.bank["flags"][p, s]
    standing, heading, world, forward = flags.unbind(-1)
    command[forward, 0] = command[forward, 0].abs().clamp(min=0.3)
    command[forward, 1:] = 0
    self.vel_command_b[env_ids] = command
    self.vel_command_w[env_ids] = command
    self.heading_target[env_ids] = self.bank["heading"][p, s]
    self.is_standing_env[env_ids] = standing
    self.is_heading_env[env_ids] = heading & ~forward
    self.is_world_env[env_ids] = world & ~forward
    self.is_forward_env[env_ids] = forward
    self.time_left[env_ids] = self.bank["durations"][p, s]
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
    self.last_episodes = {"config_ids": self.config_ids[ids].clone(), "valid": valid.clone(),
      "success": success.clone(), "terminated": terminated.clone(), "rms": rms.clone()}
    observed = self.sampler.counts > 0
    stats = {"success_rate": (success & valid).sum().float() / denom,
      "valid_episodes": valid.sum(), "recorded_total": self.recorded, "skipped_total": self.skipped,
      "linear_rms": torch.where(valid, rms[:, 0], 0.0).sum() / denom,
      "yaw_rms": torch.where(valid, rms[:, 1], 0.0).sum() / denom,
      "bank_coverage": observed.float().mean(),
      "bank_success": torch.where(observed, self.sampler.rates(), 0.0).sum() / observed.sum().clamp(min=1)}
    self.config_ids[ids] = self.sampler.sample(ids.numel())
    tile = self.config_ids[ids] // self.cfg.sgs.profiles_per_tile
    terrain = env.scene.terrain
    terrain.terrain_levels[ids] = tile // self.cols
    terrain.terrain_types[ids] = tile % self.cols
    terrain.env_origins[ids] = terrain.terrain_origins[tile // self.cols, tile % self.cols]
    self.outcome.reset(ids)
    self.profile_step[ids] = 0
    self._done[ids] = self._censored[ids] = False
    return stats

  def checkpoint_spec(self):
    return {"version": 1, "shape": (self.rows, self.cols), "ranges": asdict(self.cfg.ranges),
      "interval": self.cfg.resampling_time_range, "episode_s": self._env.max_episode_length_s,
      "terrain": repr(self._env.scene.terrain.cfg.terrain_generator)}

  def sgs_state_dict(self):
    return {"spec": self.checkpoint_spec(), "sampler": self.sampler.state_dict(),
      "bank": {k: v.detach().cpu().clone() for k, v in self.bank.items()},
      "recorded": self.recorded, "skipped": self.skipped}

  def load_sgs_state_dict(self, state):
    if state["spec"] != self.checkpoint_spec():
      raise ValueError("SGS checkpoint terrain/commands differ from this task")
    if set(state["bank"]) != set(self.bank):
      raise ValueError("Invalid SGS checkpoint bank")
    for name, value in self.bank.items():
      if not torch.equal(state["bank"][name].cpu(), value.cpu()):
        raise ValueError(f"SGS checkpoint bank differs: {name}")
    self.sampler.load_state_dict(state["sampler"])
    self.recorded, self.skipped = state["recorded"], state["skipped"]
    # Simulator trajectories are not checkpointed by mjlab. Censor in-flight
    # episodes and assign fresh configurations at the next real env reset.
    self._censored.fill_(True)


@dataclass(kw_only=True)
class SGSVelocityCommandCfg(DeadzoneVelocityCommandCfg):
  sgs: SGSConfig = field(default_factory=SGSConfig)
  evaluation: bool = False

  def build(self, env):
    return SGSVelocityCommand(self, env)


def sgs_tracking_step(env, env_ids, command_name="twist"):
  del env_ids
  env.command_manager.get_term(command_name).observe_step()


def sgs_reset_curriculum(env, env_ids, command_name="twist"):
  return env.command_manager.get_term(command_name).on_reset(env_ids)


def reset_sgs_root(env, env_ids, command_name="twist"):
  command = env.command_manager.get_term(command_name)
  ids = env_ids if env_ids is not None else torch.arange(env.num_envs, device=env.device)
  asset = command.robot
  state = asset.data.default_root_state[ids].clone()
  spawn = command.bank["spawn"][command.config_ids[ids]]
  state[:, :3] += env.scene.env_origins[ids]
  state[:, :2] += spawn[:, :2]
  zero = torch.zeros_like(spawn[:, 2])
  state[:, 3:7] = quat_mul(state[:, 3:7], quat_from_euler_xyz(zero, zero, spawn[:, 2]))
  state[:, 7:13] = 0
  asset.write_root_link_pose_to_sim(state[:, :7], env_ids=ids)
  asset.write_root_link_velocity_to_sim(state[:, 7:13], env_ids=ids)
