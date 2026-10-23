"""Optional diagnostic episode boundary inside a generated terrain grid."""

import math
from typing import TYPE_CHECKING

import torch
from mjlab.managers.termination_manager import TerminationTermCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg


def terrain_grid_boundary(env: "ManagerBasedRlEnv", margin: float) -> torch.Tensor:
  """Time out before the robot's root reaches the grid's outer border.

  The generator centers the complete grid at world (0, 0). Per-environment
  spawn origins are tile centers, not centers of the complete grid.
  """
  generator = env.scene.terrain.cfg.terrain_generator
  half_x = generator.num_rows * generator.size[0] / 2.0
  half_y = generator.num_cols * generator.size[1] / 2.0
  position = env.scene["robot"].data.root_link_pos_w
  return (position[:, 0].abs() >= half_x - margin) | (
    position[:, 1].abs() >= half_y - margin
  )


def configure_terrain_boundary(cfg: "ManagerBasedRlEnvCfg", margin: float) -> None:
  terrain = cfg.scene.terrain
  generator = terrain.terrain_generator if terrain is not None else None
  if generator is None or terrain.terrain_type != "generator":
    raise ValueError("terrain_boundary_margin requires a generated terrain grid")
  half_extent = min(
    generator.num_rows * generator.size[0] / 2.0,
    generator.num_cols * generator.size[1] / 2.0,
  )
  if not math.isfinite(margin) or not 0.0 < margin < half_extent:
    raise ValueError("terrain_boundary_margin must be positive and smaller than the grid half-extent")
  cfg.terminations["terrain_grid_boundary"] = TerminationTermCfg(
    func=terrain_grid_boundary, time_out=True, params={"margin": margin},
  )
