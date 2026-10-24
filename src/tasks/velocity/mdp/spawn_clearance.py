"""Conservative terrain clearance before the first collision solve after reset.

Each colliding robot geom is enclosed by its transformed local AABB. Boxes and
other finite terrain geoms use their world AABB top. Heightfields use the maximum
of all vertices belonging to cells overlapping that robot AABB, not sparse rays.
Raising the robot by the largest deficit separates every such pair vertically.
This protects reset placement, not subsequent dynamic motion or self collision.
"""

from contextlib import nullcontext

import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp

from mjlab.managers.manager_base import ManagerTermBase

_PLANE = int(mujoco.mjtGeom.mjGEOM_PLANE)
_HFIELD = int(mujoco.mjtGeom.mjGEOM_HFIELD)


@wp.func
def _bounds(pos: wp.vec3, rot: wp.mat33, center: wp.vec3, half: wp.vec3):
  midpoint = pos + rot @ center
  extent = wp.vec3()
  for axis in range(3):
    extent[axis] = wp.abs(rot[axis, 0]) * half[0] + wp.abs(rot[axis, 1]) * half[1] + wp.abs(rot[axis, 2]) * half[2]
  return midpoint - extent, midpoint + extent


@wp.kernel
def _required_lift(
  env_ids: wp.array[int], robot_ids: wp.array[int], terrain_ids: wp.array[int],
  collides: wp.array2d[int], geom_type: wp.array[int], geom_dataid: wp.array2d[int],
  aabb: wp.array3d[wp.vec3], margin: wp.array2d[float],
  xpos: wp.array2d[wp.vec3], xmat: wp.array2d[wp.mat33],
  hsize: wp.array[wp.vec4], nrow: wp.array[int], ncol: wp.array[int],
  hadr: wp.array[int], heights: wp.array[float], clearance: float,
  lift: wp.array[float],
):
  slot, robot_slot = wp.tid()
  world = env_ids[slot]
  robot = robot_ids[robot_slot]
  lo, hi = _bounds(xpos[world, robot], xmat[world, robot],
                   aabb[world % aabb.shape[0], robot, 0], aabb[world % aabb.shape[0], robot, 1])
  padding = margin[world % margin.shape[0], robot]
  lo = lo - wp.vec3(padding)
  hi = hi + wp.vec3(padding)
  deficit = float(0.0)
  for j in range(terrain_ids.shape[0]):
    if collides[robot_slot, j] == 0:
      continue
    terrain = terrain_ids[j]
    pos, rot = xpos[world, terrain], xmat[world, terrain]
    terrain_margin = margin[world % margin.shape[0], terrain]
    top = float(-1e9)
    if geom_type[terrain] == _PLANE:
      # Horizontal planes are validated once when the event is constructed.
      top = pos[2] + terrain_margin
    else:
      tlo, thi = _bounds(pos, rot, aabb[world % aabb.shape[0], terrain, 0],
                         aabb[world % aabb.shape[0], terrain, 1])
      if hi[0] < tlo[0] - terrain_margin or lo[0] > thi[0] + terrain_margin or hi[1] < tlo[1] - terrain_margin or lo[1] > thi[1] + terrain_margin:
        continue
      if geom_type[terrain] == _HFIELD:
        # Upright heightfields may be yaw-rotated. Project the robot AABB into
        # their local XY frame, then include all corners of overlapping cells.
        center = wp.transpose(rot) @ (0.5 * (lo + hi) - pos)
        half = 0.5 * (hi - lo)
        ex = wp.abs(rot[0, 0]) * half[0] + wp.abs(rot[1, 0]) * half[1] + terrain_margin
        ey = wp.abs(rot[0, 1]) * half[0] + wp.abs(rot[1, 1]) * half[1] + terrain_margin
        field = geom_dataid[world % geom_dataid.shape[0], terrain]
        size = hsize[field]
        cols, rows = ncol[field], nrow[field]
        c0 = wp.clamp(int(wp.floor((center[0] - ex + size[0]) * float(cols - 1) / (2.0 * size[0]))), 0, cols - 1)
        c1 = wp.clamp(int(wp.ceil((center[0] + ex + size[0]) * float(cols - 1) / (2.0 * size[0]))), 0, cols - 1)
        r0 = wp.clamp(int(wp.floor((center[1] - ey + size[1]) * float(rows - 1) / (2.0 * size[1]))), 0, rows - 1)
        r1 = wp.clamp(int(wp.ceil((center[1] + ey + size[1]) * float(rows - 1) / (2.0 * size[1]))), 0, rows - 1)
        peak = float(0.0)
        for row in range(r0, r1 + 1):
          for col in range(c0, c1 + 1):
            peak = wp.max(peak, heights[hadr[field] + row * cols + col])
        top = pos[2] + size[2] * peak + terrain_margin
      else:
        top = thi[2] + terrain_margin
    deficit = wp.max(deficit, top + clearance - lo[2])
  wp.atomic_max(lift, world, deficit)


class ensure_spawn_clearance(ManagerTermBase):
  """Last reset event: lift only reset worlds, without sampling or a collision solve."""

  def __init__(self, cfg, env):
    super().__init__(env)
    self.asset_name = cfg.params.get("asset_name", "robot")
    asset = env.scene[self.asset_name]
    terrain = env.scene.terrain
    if terrain is None or asset.is_fixed_base:
      raise ValueError("Spawn clearance requires a terrain and a floating-base robot")
    model = env.sim.mj_model
    robot_ids = asset.indexing.geom_ids.cpu().numpy()
    terrain_ids = terrain.indexing.geom_ids.cpu().numpy()
    active = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
    robot_ids, terrain_ids = robot_ids[active[robot_ids]], terrain_ids[active[terrain_ids]]
    collides = ((model.geom_contype[robot_ids, None] & model.geom_conaffinity[terrain_ids][None])
                | (model.geom_conaffinity[robot_ids, None] & model.geom_contype[terrain_ids][None])) != 0
    # A tilted heightfield needs a different vertical envelope. Fail explicitly
    # rather than silently claiming clearance for an unsupported terrain.
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    for geom in terrain_ids:
      if model.geom_type[geom] in (mujoco.mjtGeom.mjGEOM_PLANE, mujoco.mjtGeom.mjGEOM_HFIELD):
        rotation = data.geom_xmat[geom].reshape(3, 3)
        if not np.allclose(rotation[2], (0, 0, 1), atol=1e-6):
          raise ValueError("Spawn clearance requires upright planes/heightfields")
    device = env.sim.wp_device
    self.robot_ids = wp.array(robot_ids.astype(np.int32), dtype=int, device=device)
    self.terrain_ids = wp.array(terrain_ids.astype(np.int32), dtype=int, device=device)
    self.collides = wp.array(collides.astype(np.int32), dtype=int, device=device)
    self.last_lift = torch.zeros(env.num_envs, device=env.device)

  def required_lift(self, env, env_ids, clearance=.005):
    if not np.isfinite(clearance) or clearance <= 0:
      raise ValueError("Spawn clearance must be finite and positive")
    stream = (wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(env.device)))
              if str(env.device).startswith("cuda") else nullcontext())
    with stream:
      ids = env_ids.to(device=env.device, dtype=torch.int32)
      self.last_lift[ids.long()] = 0
      model, data = env.sim.wp_model, env.sim.wp_data
      with wp.ScopedDevice(env.sim.wp_device):
        mjw.kinematics(model, data)
        wp.launch(_required_lift, dim=(len(ids), len(self.robot_ids)), inputs=[
          wp.from_torch(ids), self.robot_ids, self.terrain_ids, self.collides,
          model.geom_type, model.geom_dataid, model.geom_aabb, model.geom_margin,
          data.geom_xpos, data.geom_xmat, model.hfield_size, model.hfield_nrow,
          model.hfield_ncol, model.hfield_adr, model.hfield_data, clearance,
        ], outputs=[wp.from_torch(self.last_lift)], device=env.sim.wp_device)
      return self.last_lift[ids.long()]

  def __call__(self, env, env_ids, clearance=.005, asset_name="robot"):
    if asset_name != self.asset_name:
      raise ValueError("Spawn-clearance asset cannot change after construction")
    if env_ids is None:
      env_ids = torch.arange(env.num_envs, device=env.device)
    stream = (wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(env.device)))
              if str(env.device).startswith("cuda") else nullcontext())
    with stream:
      lift = self.required_lift(env, env_ids, clearance)
      asset = env.scene[asset_name]
      pose = env.sim.data.qpos[env_ids[:, None], asset.indexing.free_joint_q_adr].clone()
      pose[:, 2] += lift
      asset.write_root_link_pose_to_sim(pose, env_ids=env_ids)
      with wp.ScopedDevice(env.sim.wp_device):
        mjw.kinematics(env.sim.wp_model, env.sim.wp_data)
