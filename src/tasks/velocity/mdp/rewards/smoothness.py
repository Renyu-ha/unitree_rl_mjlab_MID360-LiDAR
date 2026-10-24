# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Optional action and GRF smoothness costs adapted from robot_lab.

These terms are available for experiments, but are not configured by default.
Differences are discrete increments per policy step, without division by dt.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.reward_manager import RewardTermCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def action_smoothness_second_order(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Squared second difference of raw policy actions, summed over all actions.

  The action manager shifts history before reward evaluation. Episode lengths
  1 and 2 do not yet have three actions from this episode and are ignored.
  Zero-valued actions are valid samples, not a marker for missing history.
  """
  manager = env.action_manager
  difference = manager.action - 2.0 * manager.prev_action + manager.prev_prev_action
  cost = difference.square().sum(dim=-1)
  return cost * (env.episode_length_buf > 2)


class grf_smoothness:
  """First/second differences of per-wheel world-frame total contact force.

  By default use only world Z, to avoid directly penalizing traction force.
  Maintain policy-step samples rather than sensor history: mjlab's history is
  sampled at physics substeps. Unlike robot_lab's normal-force input, this
  consumes the shared total-force sensor. No contact mask is applied, so both
  touchdown and liftoff transitions are included. Reset warmup is per env.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    sensor = env.scene[cfg.params["sensor_name"]]
    if sensor.cfg.reduce != "netforce" and not sensor.cfg.global_frame:
      raise ValueError("grf_smoothness requires world-frame contact forces")
    force = sensor.data.force
    if force is None or force.ndim != 3 or force.shape[-1] != 3:
      raise ValueError("grf_smoothness requires per-body contact forces [B, N, 3]")
    self.previous_force = torch.zeros_like(force)
    self.previous_previous_force = torch.zeros_like(force)
    self.samples = torch.zeros(force.shape[0], dtype=torch.long, device=force.device)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self.previous_force[env_ids] = 0.0
    self.previous_previous_force[env_ids] = 0.0
    self.samples[env_ids] = 0

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    first_difference_weight: float = 1.0,
    second_difference_weight: float = 0.5,
    vertical_only: bool = True,
  ) -> torch.Tensor:
    if first_difference_weight < 0.0 or second_difference_weight < 0.0:
      raise ValueError("GRF smoothness weights must be non-negative")
    force = env.scene[sensor_name].data.force
    if force is None or force.shape != self.previous_force.shape:
      raise ValueError("Contact force shape changed since grf_smoothness initialization")
    first = force - self.previous_force
    second = force - 2.0 * self.previous_force + self.previous_previous_force
    if vertical_only:
      first, second = first[..., 2], second[..., 2]
    cost = (
      first_difference_weight * first.flatten(start_dim=1).square().sum(dim=-1)
      * (self.samples > 0)
      + second_difference_weight * second.flatten(start_dim=1).square().sum(dim=-1)
      * (self.samples > 1)
    )
    self.previous_previous_force.copy_(self.previous_force)
    self.previous_force.copy_(force)
    self.samples.add_(1).clamp_max_(2)
    return cost
