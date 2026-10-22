# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Wheel rewards adapted from robot_lab; see doc/go2w_perceptive_zh.md."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def wheel_spin_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  wheel_radius: float,
  joint_cfg: SceneEntityCfg,
  body_cfg: SceneEntityCfg,
  contact_threshold: float = 1.0,
) -> torch.Tensor:
  """Port robot_lab's wheel_spin_contact to mjlab's raw contact sensor.

  Cost per grounded wheel is max(0.8 * R * abs(joint_speed) - COM_speed -
  0.1, 0). Contact uses the current net normal reaction magnitude > threshold,
  matching robot_lab rather than gating on tangential friction or history.

  The sensor must use reduce='none', global_frame=False and fields force/normal,
  with its primaries in the same order as body_cfg and joint_cfg. Its force X
  component is the scalar normal reaction; normal is the actual world-space
  contact normal. No contact position or angular-velocity correction is used.
  """
  if wheel_radius <= 0.0 or contact_threshold < 0.0:
    raise ValueError("wheel_radius must be positive and contact_threshold non-negative")
  asset = env.scene[joint_cfg.name]
  wheel_ang_vel = asset.data.joint_vel[:, joint_cfg.joint_ids]
  wheel_lin_vel = env.scene[body_cfg.name].data.body_com_lin_vel_w[:, body_cfg.body_ids]
  if wheel_ang_vel.shape != wheel_lin_vel.shape[:2]:
    raise ValueError("Wheel joint and body selections must have matching dimensions")
  sensor = env.scene[sensor_name]
  if sensor.cfg.reduce != "none" or sensor.cfg.global_frame:
    raise ValueError("wheel_spin_contact requires raw contact-frame force slots")
  data = sensor.data
  assert data.force is not None and data.normal is not None
  num_envs, num_wheels = wheel_ang_vel.shape
  expected_slots = num_wheels * sensor.cfg.num_slots
  if data.force.shape != (num_envs, expected_slots, 3) or data.normal.shape != data.force.shape:
    raise ValueError("Contact primaries must match the selected wheel order and count")
  forces = data.force.reshape(num_envs, num_wheels, sensor.cfg.num_slots, 3)
  normals = data.normal.reshape_as(forces)
  net_normal_force = (forces[..., :1] * normals).sum(dim=2)
  contact = net_normal_force.norm(dim=-1) > contact_threshold
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
