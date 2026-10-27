# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Wheel rewards adapted from robot_lab; see doc/go2w_perceptive_zh.md."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

import torch
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def wheel_lateral_slide(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  body_cfg: SceneEntityCfg,
  slip_deadband: float = 0.04,
  contact_threshold: float = 1.0,
) -> torch.Tensor:
  """Penalize contact-wheel velocity along the horizontal wheel axle.

  Adapt robot_lab's ``wheel_lateral_slide_switch``: local Y is the axle,
  projected onto the world horizontal plane to exclude vertical motion.
  Velocity is the wheel COM velocity, not a reconstructed contact-point
  velocity. This is a horizontal slip proxy, not a terrain-tangent tire model.
  Use the same current total-force contact gate and wheel ordering as
  ``wheel_spin_contact``. A detached wheel incurs no cost.
  """
  if slip_deadband < 0.0 or contact_threshold < 0.0:
    raise ValueError("slip_deadband and contact_threshold must be non-negative")
  asset = env.scene[body_cfg.name]
  velocity = asset.data.body_com_lin_vel_w[:, body_cfg.body_ids]
  quat = asset.data.body_link_quat_w[:, body_cfg.body_ids]
  force = env.scene[sensor_name].data.force
  if force is None or force.shape != velocity.shape:
    raise ValueError("Wheel-ground sensor must provide one net contact force per wheel")
  axle_b = torch.zeros_like(velocity)
  axle_b[..., 1] = 1.0
  axle_w = quat_apply(quat, axle_b)
  axle_w[..., 2] = 0.0
  axle_w = axle_w / axle_w.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
  lateral_speed = (velocity * axle_w).sum(dim=-1).abs()
  slip = (lateral_speed - slip_deadband).clamp_min(0.0)
  contact = force.norm(dim=-1) > contact_threshold
  return (slip * contact).sum(dim=-1)


def wheel_spin_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  wheel_radius: float,
  joint_cfg: SceneEntityCfg,
  body_cfg: SceneEntityCfg,
  contact_threshold: float = 1.0,
) -> torch.Tensor:
  """Adapt robot_lab's spin formula using the shared wheel-ground sensor.

  Cost per grounded wheel is max(0.8 * R * abs(joint_speed) - COM_speed -
  0.1, 0). Contact uses the current net contact force magnitude > threshold,
  including friction. Unlike robot_lab's normal-only gate, this only needs to
  detect contact; historical force does not keep a detached wheel active.

  The ordinary feet_ground_contact netforce sensor is also used by critic
  observations and landing rewards. Its primaries must have the same order as
  body_cfg and joint_cfg. No contact position or angular-velocity correction
  is used.
  """
  if wheel_radius <= 0.0 or contact_threshold < 0.0:
    raise ValueError("wheel_radius must be positive and contact_threshold non-negative")
  asset = env.scene[joint_cfg.name]
  wheel_ang_vel = asset.data.joint_vel[:, joint_cfg.joint_ids]
  wheel_lin_vel = env.scene[body_cfg.name].data.body_com_lin_vel_w[:, body_cfg.body_ids]
  if wheel_ang_vel.shape != wheel_lin_vel.shape[:2]:
    raise ValueError("Wheel joint and body selections must have matching dimensions")
  sensor = env.scene[sensor_name]
  net_force = sensor.data.force
  if net_force is None or net_force.shape != (*wheel_ang_vel.shape, 3):
    raise ValueError("Wheel-ground sensor must provide one net contact force per wheel")
  contact = net_force.norm(dim=-1) > contact_threshold
  wheel_speed = wheel_lin_vel.norm(dim=-1)
  spin = (0.8 * wheel_radius * wheel_ang_vel.abs() - wheel_speed - 0.1).clamp_min(0.0)
  return (spin * contact).sum(dim=-1)


def wheel_stand_still(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg,
  command_threshold: float = 0.1,
) -> torch.Tensor:
  """Brake the wheels when the commanded twist is zero."""
  command = env.command_manager.get_command(command_name)
  stopped = command.norm(dim=-1) < command_threshold
  velocity = env.scene[asset_cfg.name].data.joint_vel[:, asset_cfg.joint_ids]
  return velocity.square().sum(dim=-1) * stopped


def wheel_air_time_timeout(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  threshold: float = 0.5,
  ramp_time: float = 0.25,
) -> torch.Tensor:
  """Continuously penalize each wheel that stays airborne beyond the grace time.

  Use the contact sensor's current air time, which resets on contact and episode
  reset. No landing event or movement command is required. Each wheel ramps from
  zero to one over ``ramp_time`` seconds after ``threshold``; averaging keeps the
  total cost in [0, 1]. Configure a negative reward weight.
  """
  if threshold < 0.0 or ramp_time <= 0.0:
    raise ValueError("threshold must be non-negative and ramp_time must be positive")
  air_time = env.scene[sensor_name].data.current_air_time
  if air_time is None:
    raise ValueError("Wheel air-time timeout requires track_air_time=True")
  env.extras["log"]["Metrics/wheel_air_time_max_mean"] = air_time.max(dim=-1).values.mean()
  return ((air_time - threshold) / ramp_time).clamp(0.0, 1.0).mean(dim=-1)


class wheel_air_time_imbalance:
  """Penalize unequal recent unsupported-time ratios across the four wheels.

  Adapt the support-phase-ratio standard deviation from Irie et al. (2025),
  using force-qualified wheel support and a time constant in seconds. The EMA
  of unsupported time is equivalent to one minus the EMA of support. A light
  touch cannot clear this history. Update history even when the command gate
  is off; apply the absolute-command direction decay to the cost independently
  of the contact bonus's optional straight-only cutoffs.
  Equal airborne ratios incur no imbalance cost, so retain the contact bonus
  and the independent continuous-air-time timeout. Configure a negative weight.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    data = env.scene[cfg.params["sensor_name"]].data
    force, found = data.force, data.found
    if (force is None or found is None or found.ndim != 2 or found.shape[1] != 4
        or force.shape != (*found.shape, 3)):
      raise ValueError("Wheel air-time imbalance requires four net forces and contact flags")
    # Zero means no accumulated unsupported time yet, including after reset.
    self.air_ratio = torch.zeros_like(found, dtype=torch.float32)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self.air_ratio[slice(None) if env_ids is None else env_ids] = 0.0

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    time_constant: float = 0.5,
    force_threshold: float = 5.0,
    command_name: str = "twist",
    command_x_threshold: float = 0.1,
    command_y_scale: float = 0.1,
    command_yaw_scale: float = 0.2,
    direction_score_type: Literal["rational", "exponential"] = "rational",
  ) -> torch.Tensor:
    if time_constant <= 0.0 or env.step_dt <= 0.0:
      raise ValueError("time_constant and control step duration must be positive")
    if force_threshold < 0.0 or command_x_threshold < 0.0:
      raise ValueError("Force and command thresholds must be non-negative")
    if command_y_scale <= 0.0 or command_yaw_scale <= 0.0:
      raise ValueError("Direction scales must be positive")
    if direction_score_type not in ("rational", "exponential"):
      raise ValueError("direction_score_type must be 'rational' or 'exponential'")
    data = env.scene[sensor_name].data
    force, found = data.force, data.found
    if (force is None or found is None or found.shape != self.air_ratio.shape
        or force.shape != (*self.air_ratio.shape, 3)):
      raise ValueError("Wheel contact shape changed since air-time imbalance initialization")
    command = env.command_manager.get_command(command_name)
    if command is None:
      raise ValueError("Wheel air-time imbalance requires a velocity command")

    # Wheel-on-terrain force has the opposite sign to the upward support reaction.
    supported = (found > 0) & (-force[..., 2] > force_threshold)
    decay = math.exp(-env.step_dt / time_constant)
    self.air_ratio.mul_(decay).add_((~supported).float(), alpha=1.0 - decay)
    imbalance = self.air_ratio.std(dim=-1, correction=0)
    motion_weight = (command[:, 0].abs() > command_x_threshold).float()
    direction_magnitude = (
      (command[:, 1] / command_y_scale).abs()
      + (command[:, 2] / command_yaw_scale).abs()
    )
    direction_score = (
      (1.0 + direction_magnitude).reciprocal()
      if direction_score_type == "rational" else torch.exp(-direction_magnitude)
    )
    gate = motion_weight * direction_score
    env.extras["log"]["Metrics/wheel_air_ratio_mean"] = self.air_ratio.mean()
    env.extras["log"]["Metrics/wheel_air_ratio_max_mean"] = self.air_ratio.max(dim=-1).values.mean()
    env.extras["log"]["Metrics/wheel_air_ratio_std_mean"] = imbalance.mean()
    env.extras["log"]["Metrics/wheel_air_time_imbalance_gate_mean"] = gate.mean()
    return imbalance * gate


def wheel_contact_bonus(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 5.0,
  command_name: str = "twist",
  command_x_threshold: float = 0.1,
  command_y_scale: float = 0.1,
  command_yaw_scale: float = 0.2,
  direction_score_type: Literal["rational", "exponential"] = "rational",
  command_y_cutoff: float | None = None,
  command_yaw_cutoff: float | None = None,
) -> torch.Tensor:
  """Reward wheel support with a continuous preference for straight commands.

  Requires wheel primaries and terrain secondary, with one world-frame net
  force and contact flag per wheel. MuJoCo reports wheel-on-terrain force;
  its opposite is the reaction supporting the wheel. Horizontal contact,
  downward reaction and light touches do not count as support. A brief
  touchdown only earns reward while supporting the robot; it cannot erase the
  reward lost during suspension. The reward is enabled only when the absolute
  longitudinal command exceeds its threshold; a rational or exponential score
  reduces it as lateral/yaw commands grow.
  Both scores use the sum of absolute normalized commands, not squared commands.
  With rational decay, either direction scale alone gives a score of one half.
  Optional lateral/yaw cutoffs disable this bonus during maneuvers, freeing the
  policy to lift a wheel. Leaving both cutoffs unset preserves ungated decay.
  Use commands rather than actual motion, so yaw drift cannot switch it off.
  """
  if force_threshold < 0.0 or command_x_threshold < 0.0:
    raise ValueError("Force and command thresholds must be non-negative")
  if command_y_scale <= 0.0 or command_yaw_scale <= 0.0:
    raise ValueError("Direction scales must be positive")
  if direction_score_type not in ("rational", "exponential"):
    raise ValueError("direction_score_type must be 'rational' or 'exponential'")
  if ((command_y_cutoff is not None and command_y_cutoff <= 0.0)
      or (command_yaw_cutoff is not None and command_yaw_cutoff <= 0.0)):
    raise ValueError("Optional direction cutoffs must be positive")
  data = env.scene[sensor_name].data
  force, found = data.force, data.found
  if force is None or found is None or force.shape != (*found.shape, 3):
    raise ValueError("Wheel contact bonus requires one net force and contact flag per wheel")
  support_force_z = -force[..., 2]
  supported = (found > 0) & (support_force_z > force_threshold)
  fraction = supported.float().mean(dim=-1)
  command = env.command_manager.get_command(command_name)
  if command is None:
    raise ValueError("Wheel contact bonus requires a velocity command")
  motion_weight = (command[:, 0].abs() > command_x_threshold).float()
  direction_magnitude = (
    (command[:, 1] / command_y_scale).abs()
    + (command[:, 2] / command_yaw_scale).abs()
  )
  direction_score = (
    (1.0 + direction_magnitude).reciprocal()
    if direction_score_type == "rational" else torch.exp(-direction_magnitude)
  )
  direction_allowed = torch.ones_like(motion_weight, dtype=torch.bool)
  if command_y_cutoff is not None:
    direction_allowed &= command[:, 1].abs() < command_y_cutoff
  if command_yaw_cutoff is not None:
    direction_allowed &= command[:, 2].abs() < command_yaw_cutoff
  gate = motion_weight * direction_score * direction_allowed
  env.extras["log"]["Metrics/wheel_support_fraction"] = fraction.mean()
  env.extras["log"]["Metrics/all_wheels_grounded_ratio"] = supported.all(dim=-1).float().mean()
  env.extras["log"]["Metrics/wheel_contact_bonus_active_ratio"] = (gate > 0).float().mean()
  env.extras["log"]["Metrics/wheel_contact_bonus_gate_mean"] = gate.mean()
  return fraction * gate
