from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def terrain_out_of_bounds(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  distance_buffer: float = 3.0,
) -> torch.Tensor:
  """Detect the outer map edge, including the flat border, as in Isaac Lab.

  The complete terrain is centred at world (0, 0). Tile spawn origins do
  not define its boundary. Register with time_out=True for truncations.
  """
  terrain = env.scene.terrain.cfg
  if terrain.terrain_type == "plane":
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  if terrain.terrain_type != "generator" or terrain.terrain_generator is None:
    raise ValueError("terrain_out_of_bounds requires plane or generated terrain")

  generator = terrain.terrain_generator
  half_x = generator.num_rows * generator.size[0] / 2.0 + generator.border_width
  half_y = generator.num_cols * generator.size[1] / 2.0 + generator.border_width
  position = env.scene[asset_cfg.name].data.root_link_pos_w
  return (position[:, 0].abs() > half_x - distance_buffer) | (
    position[:, 1].abs() > half_y - distance_buffer
  )


def illegal_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # force_history: [B, N, H, 3]
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    return (force_mag > force_threshold).any(dim=-1).any(dim=-1)  # [B]
  assert data.found is not None
  return torch.any(data.found, dim=-1)
