"""Terrain-gated Go2W rewards adapted from go2w-supermimic's v2/v5/v6 design.

Use measured wheel sites and the existing world-down wheel-height sensors rather
than the reference robot's analytical kinematics. All terrain queries use raw
privileged rays; actor noise, dropout and scan history never enter rewards.

Reference: uwvwko-zzz/go2w-supermimic at a299ed5 (MIT, Ziyue Zheng).
The reference license is retained in licenses/go2w-supermimic-MIT.txt.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

__all__ = [
  "leg_posture",
  "obstacle_progress",
  "rolling_leg_motion",
  "unnecessary_wheel_lift",
  "wheel_obstacle_clearance",
]

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.scene_entity_config import SceneEntityCfg


def _rotate(quat, vector, *, inverse=False):
  quat = torch.nn.functional.normalize(quat, dim=-1)
  xyz = -quat[..., 1:] if inverse else quat[..., 1:]
  xyz, vector = torch.broadcast_tensors(xyz, vector)
  cross = 2.0 * torch.cross(xyz, vector, dim=-1)
  return vector + quat[..., :1] * cross + torch.cross(xyz, cross, dim=-1)


def _yaw_xy(quat, vector):
  quat = torch.nn.functional.normalize(quat, dim=-1)
  w, x, y, z = quat.unbind(-1)
  yaw = torch.atan2(2.0 * (w*z + x*y), 1.0 - 2.0 * (y*y + z*z))
  while yaw.ndim < vector.ndim - 1:
    yaw = yaw.unsqueeze(-1)
  c, s = yaw.cos(), yaw.sin()
  return torch.stack((c*vector[..., 0] + s*vector[..., 1],
                      -s*vector[..., 0] + c*vector[..., 1]), dim=-1)


def _height_grid(env, sensor_name):
  sensor = env.scene[sensor_name]
  if sensor.cfg.ray_alignment != "yaw":
    raise ValueError("Terrain-aware rewards require a yaw-aligned raw height grid")
  pattern = sensor.cfg.pattern
  sx, sy = pattern.size
  resolution = pattern.resolution
  nx, ny = round(sx / resolution) + 1, round(sy / resolution) + 1
  if nx < 2 or ny < 2:
    raise ValueError("Terrain-aware rewards require a two-dimensional height grid")
  data = sensor.data
  height = data.hit_pos_w[..., 2].reshape(env.num_envs, ny, nx)
  valid = (data.distances.reshape_as(height) >= 0) & torch.isfinite(height)
  normals = getattr(data, "normals_w", None)
  if normals is not None:
    normal_z = normals[..., 2].reshape_as(height)
    valid &= torch.isfinite(normal_z) & (normal_z >= 0)
  height = torch.where(valid, height, 0.0)
  offset = getattr(pattern, "offset", (0.0, 0.0))
  x0 = -sx / 2 + getattr(pattern, "x_offset", offset[0])
  y0 = -sy / 2 + getattr(pattern, "y_offset", offset[1])
  env.extras["log"]["Metrics/reward_terrain_valid_fraction"] = valid.float().mean()
  return sensor, height, valid, x0, y0, resolution


def _sample(grid, valid, xy, x0, y0, resolution):
  """Bilinear terrain height; never clamp an outside query into a valid hit."""
  batch, ny, nx = grid.shape
  shape = xy.shape[:-1]
  gx, gy = (xy[..., 0] - x0) / resolution, (xy[..., 1] - y0) / resolution
  inside = (torch.isfinite(xy).all(-1) & (gx >= -1e-5) & (gx <= nx-1+1e-5)
            & (gy >= -1e-5) & (gy <= ny-1+1e-5))
  gx = torch.nan_to_num(gx).clamp(0, nx-1).reshape(batch, -1)
  gy = torch.nan_to_num(gy).clamp(0, ny-1).reshape(batch, -1)
  ix, iy = gx.floor().long().clamp(0, nx-2), gy.floor().long().clamp(0, ny-2)
  ax, ay = gx-ix, gy-iy
  values, usable = [], []
  for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
    index = (iy+dy)*nx + ix+dx
    values.append(grid.flatten(1).gather(1, index))
    usable.append(valid.flatten(1).gather(1, index))
  value = ((1-ax)*values[0] + ax*values[1])*(1-ay)
  value += ((1-ax)*values[2] + ax*values[3])*ay
  ok = inside & torch.stack(usable).all(0).reshape(shape)
  return torch.where(ok, value.reshape(shape), 0.0), ok


def _local_relief(env, terrain_sensor_name):
  _, grid, valid, x0, y0, res = _height_grid(env, terrain_sensor_name)
  x = torch.arange(grid.shape[2], device=grid.device)*res+x0
  y = torch.arange(grid.shape[1], device=grid.device)*res+y0
  local = (y[:, None].abs() <= .30001) & (x[None, :].abs() <= .50001)
  heights, hits = grid[:, local], valid[:, local]
  relief = heights.amax(-1) - heights.amin(-1)
  return relief, hits.all(-1)


def _wheel_patch(env, terrain_sensor_name, asset_cfg):
  sensor, grid, valid, x0, y0, res = _height_grid(env, terrain_sensor_name)
  data = env.scene[asset_cfg.name].data
  sites = data.site_pos_w[:, asset_cfg.site_ids]
  xy = _yaw_xy(sensor.data.quat_w[:, None], sites-sensor.data.pos_w[:, None])
  offsets = xy.new_tensor([[x, y] for x in (-.15, 0., .15) for y in (-.10, 0., .10)])
  samples, hits = _sample(grid, valid, xy[:, :, None]+offsets, x0, y0, res)
  relief = samples.amax(-1) - samples.amin(-1)
  return data, sites, xy, relief, hits.all(-1), (sensor, grid, valid, x0, y0, res)


def _rolling_gate(relief, valid, command):
  flat = (1.0-(relief-.02)/.04).clamp(0.0, 1.0)
  straight = torch.exp(-(command[:, 1]/.25).square()-command[:, 2].square())
  return flat * valid * straight[:, None]


def _wheel_ground(env, sensor_names, sites, fallback_context):
  scans = [env.scene[name].data for name in sensor_names]
  ground = torch.cat([scan.hit_pos_w[..., 2] for scan in scans], dim=1)
  valid = torch.cat([scan.distances >= 0 for scan in scans], dim=1) & torch.isfinite(ground)
  # A ray starting inside terrain may hit its underside. That is not the
  # supporting top surface and must use the elevated truth query as well.
  normal_valid = []
  for scan in scans:
    normals = getattr(scan, "normals_w", None)
    normal_valid.append(torch.ones_like(scan.distances, dtype=torch.bool) if normals is None
                        else torch.isfinite(normals[..., 2]) & (normals[..., 2] >= 0))
  valid &= torch.cat(normal_valid, dim=1)
  if ground.shape != sites.shape[:2]:
    raise ValueError("Wheel rewards require one height ray per ordered wheel site")
  # Prefer the dedicated per-wheel height sensor. If its low origin misses,
  # query the complete privileged terrain surface at that same world XY.
  sensor, grid, hits, x0, y0, res = fallback_context
  xy = _yaw_xy(sensor.data.quat_w[:, None], sites-sensor.data.pos_w[:, None])
  fallback, fallback_valid = _sample(grid, hits, xy, x0, y0, res)
  env.extras["log"]["Metrics/wheel_height_truth_fallback_fraction"] = ((~valid) & fallback_valid).float().mean()
  env.extras["log"]["Metrics/wheel_height_unknown_fraction"] = (~(valid | fallback_valid)).float().mean()
  return torch.where(valid, ground, fallback), valid | fallback_valid


def _vertical_radius(data, asset_cfg, wheel_radius, wheel_half_width):
  if wheel_radius <= 0 or wheel_half_width <= 0:
    raise ValueError("Wheel radius and half-width must be positive")
  quat = torch.nn.functional.normalize(data.body_link_quat_w[:, asset_cfg.body_ids], dim=-1)
  w, x, y, z = quat.unbind(-1)
  # Wheel axle is local Y, including after wheel spin around that same axis.
  axle_z = (2.0*(y*z+w*x)).clamp(-1.0, 1.0)
  return wheel_radius*(1.0-axle_z.square()).clamp_min(0.0).sqrt() + wheel_half_width*axle_z.abs()


def leg_posture(env: ManagerBasedRlEnv, terrain_sensor_name: str,
                asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """Keep hip abduction constrained; relax flexion for each wheel's local relief."""
  data, _, _, relief, valid, _ = _wheel_patch(env, terrain_sensor_name, asset_cfg)
  q = (data.joint_pos[:, asset_cfg.joint_ids]
       - data.default_joint_pos[:, asset_cfg.joint_ids]).reshape(env.num_envs, 4, 3)
  multiplier = torch.where(valid, 1.0-.875*(relief/.12).clamp(0.0, 1.0), .125)
  return 2.0*q[..., 0].square().sum(-1) + (multiplier*q[..., 1:].square().sum(-1)).sum(-1)


def unnecessary_wheel_lift(env: ManagerBasedRlEnv, terrain_sensor_name: str,
                            wheel_sensor_names: tuple[str, ...], command_name: str,
                            wheel_radius: float, wheel_half_width: float,
                            asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """Penalize clearance above each wheel's measured ground only on flat patches."""
  data, sites, _, relief, valid, context = _wheel_patch(env, terrain_sensor_name, asset_cfg)
  ground, grounded_ray = _wheel_ground(env, wheel_sensor_names, sites, context)
  bottom = sites[..., 2]-ground-_vertical_radius(data, asset_cfg, wheel_radius, wheel_half_width)
  gate = _rolling_gate(relief, valid & grounded_ray, env.command_manager.get_command(command_name))
  lift = ((bottom-.005).clamp_min(0.0)/.05).square()
  env.extras["log"]["Metrics/unnecessary_wheel_lift_gate"] = gate.mean()
  return (gate*lift).sum(-1)


def rolling_leg_motion(env: ManagerBasedRlEnv, terrain_sensor_name: str,
                       command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """Favor quiet legs on flat terrain, relaxing during maneuvers and recovery."""
  data, _, _, relief, valid, _ = _wheel_patch(env, terrain_sensor_name, asset_cfg)
  gate = _rolling_gate(relief, valid, env.command_manager.get_command(command_name))
  tilt = ((data.projected_gravity_b[:, :2].norm(dim=-1)-.08)/.22).clamp(0.0, 1.0)
  motion = data.joint_vel[:, asset_cfg.joint_ids].reshape(env.num_envs, 4, 3).square().mean(-1)
  env.extras["log"]["Metrics/rolling_leg_motion_gate"] = gate.mean()
  return (gate*motion).mean(-1)*(1.0-.8*tilt)


def obstacle_progress(env: ManagerBasedRlEnv, terrain_sensor_name: str,
                      command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """Signed, bounded actual body progress near uneven terrain; no wheel-spin bonus."""
  relief, valid = _local_relief(env, terrain_sensor_name)
  gate = ((relief-.02)/.08).clamp(0.0, 1.0)*valid
  command = env.command_manager.get_command(command_name)[:, :2]
  speed = command.norm(dim=-1)
  velocity = env.scene[asset_cfg.name].data.root_link_lin_vel_b[:, :2]
  along = (velocity*command).sum(-1)/speed.clamp_min(.1)
  env.extras["log"]["Metrics/obstacle_progress_gate"] = gate.mean()
  return gate*(speed > .1)*(along/speed.clamp_min(.2)).clamp(-1.0, 1.0)


def wheel_obstacle_clearance(env: ManagerBasedRlEnv, terrain_sensor_name: str,
                             wheel_sensor_names: tuple[str, ...], command_name: str,
                             wheel_radius: float, wheel_half_width: float,
                             asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """Bounded clearance deficit 16 cm ahead, using each wheel's measured ground."""
  data, sites, xy, _, patch_valid, context = _wheel_patch(env, terrain_sensor_name, asset_cfg)
  sensor, grid, valid, x0, y0, res = context
  command = env.command_manager.get_command(command_name)
  speed = command[:, :2].norm(dim=-1)
  body_vector = torch.cat((command[:, :2], torch.zeros_like(command[:, :1])), dim=-1)
  world_vector = _rotate(data.root_link_quat_w, body_vector)
  direction = _yaw_xy(sensor.data.quat_w, world_vector)
  direction = direction/direction.norm(dim=-1, keepdim=True).clamp_min(.1)
  ahead, ahead_valid = _sample(grid, valid, xy+.16*direction[:, None], x0, y0, res)
  ground, ground_valid = _wheel_ground(env, wheel_sensor_names, sites, context)
  usable = patch_valid & ahead_valid & ground_valid
  approaching = ((ahead-ground-.04)/.08).clamp(0.0, 1.0)*usable*(speed > .1)[:, None]
  radius = _vertical_radius(data, asset_cfg, wheel_radius, wheel_half_width)
  deficit = (ahead+radius+.015-sites[..., 2]).clamp_min(0.0)/.20
  env.extras["log"]["Metrics/wheel_obstacle_clearance_gate"] = approaching.mean()
  return (approaching*deficit/(1.0+deficit)).mean(-1)
