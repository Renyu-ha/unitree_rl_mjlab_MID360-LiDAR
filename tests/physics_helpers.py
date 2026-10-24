"""Collision/reset fixtures kept outside the training entry points."""

from importlib.metadata import version

import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp
from mjlab.tasks.registry import load_env_cfg


def make_coverage_cfg(task, seed=42, samples=2):
  cfg = load_env_cfg(task)
  cfg.seed = seed
  cfg.curriculum = {}
  cfg.events.pop("randomize_terrain", None)
  generator = cfg.scene.terrain.terrain_generator
  if generator is None:
    rows, names = 1, ["plane"]
  else:
    # Force one column per type so low proportions cannot hide a terrain.
    generator.curriculum = True
    rows, names = generator.num_rows, list(generator.sub_terrains)
  cfg.scene.num_envs = rows * len(names) * samples
  return cfg, rows, names

def assign_all_patches(env, rows, names):
  terrain = env.scene.terrain
  ids = torch.arange(env.num_envs, device=env.device)
  types = ids % len(names)
  levels = (ids // len(names)) % rows
  if terrain.terrain_origins is not None:
    terrain.terrain_types[:] = types
    terrain.terrain_levels[:] = levels
    terrain.env_origins[:] = terrain.terrain_origins[levels, types]
  return types.cpu().numpy()

def collision_stats(env):
  if str(env.device).startswith("cuda"):
    torch.cuda.synchronize(env.device)
  sim = env.sim
  with wp.ScopedDevice(sim.wp_device):
    sim.wp_data.overflow.zero_()
    mjw.kinematics(sim.wp_model, sim.wp_data)
    mjw.collision(sim.wp_model, sim.wp_data)
    wp.synchronize_device(sim.wp_device)
  count = min(int(sim.wp_data.nacon.numpy()[0]), sim.wp_data.naconmax)
  contact = sim.wp_data.contact
  pairs = contact.geom.numpy()[:count]
  robot = env.scene["robot"].indexing.geom_ids.cpu().numpy()
  terrain = env.scene.terrain.indexing.geom_ids.cpu().numpy()
  match = ((np.isin(pairs[:, 0], robot) & np.isin(pairs[:, 1], terrain))
           | (np.isin(pairs[:, 1], robot) & np.isin(pairs[:, 0], terrain)))
  worlds = contact.worldid.numpy()[:count][match]
  distances = contact.dist.numpy()[:count][match]
  penetration = np.zeros(env.num_envs)
  np.maximum.at(penetration, worlds, np.maximum(-distances, 0))
  flags = sim.wp_data.overflow.numpy()
  return {
    "penetration_m": penetration,
    "epa": (flags & int(mjw.OverflowType.EPA_HORIZON)) != 0,
    "other_overflow": (flags & ~int(mjw.OverflowType.EPA_HORIZON)) != 0,
    "finite": bool(np.isfinite(distances).all() and np.isfinite(contact.pos.numpy()[:count]).all()),
  }

def native_penetration(env):
  """Independent float64 collision check for the current Go2W DR profile.

Friction, COM and encoder DR do not change geom transforms or sizes. Reject
geometry DR here rather than compare against a native model with stale shapes.
"""
  geometric = {"body_pos", "body_quat", "geom_pos", "geom_quat", "geom_size", "geom_dataid"}
  if geometric & (set(env.sim.expanded_fields) | set(env.sim.per_world_default_fields)):
    raise ValueError("Native reference must be extended before testing geometry domain randomization")
  model = env.sim.mj_model
  data = mujoco.MjData(model)
  robot = set(env.scene["robot"].indexing.geom_ids.cpu().tolist())
  terrain = set(env.scene.terrain.indexing.geom_ids.cpu().tolist())
  result = np.zeros(env.num_envs)
  for world, qpos in enumerate(env.sim.data.qpos.detach().cpu().numpy()):
    data.qpos[:] = qpos
    mujoco.mj_fwdPosition(model, data)
    for contact in data.contact[:data.ncon]:
      a, b = int(contact.geom1), int(contact.geom2)
      if (a in robot and b in terrain) or (b in robot and a in terrain):
        if not np.isfinite(contact.dist):
          raise RuntimeError("Native collision reference returned a non-finite distance")
        result[world] = max(result[world], -float(contact.dist))
  return result

def audit_spawn_clearance(env, rows, names, resets=2, nominal_heights=(.4, .5, .6)):
  types = assign_all_patches(env, rows, names)
  guard = env.event_manager.get_term_cfg("ensure_spawn_clearance").func
  asset = env.scene["robot"]
  qadr = asset.indexing.free_joint_q_adr
  ids = torch.arange(env.num_envs, device=env.device)
  nominal = float(asset.cfg.init_state.pos[2])
  results = []
  passed = True
  for reset in range(resets):
    env.reset()
    # Reconstruct the exact raw reset before the guard lifted it. No integration
    # has happened, so all joints, heading, XY and DR remain frozen for this A/B.
    raw = env.sim.data.qpos[:, qadr].clone()
    raw[:, 2] -= guard.last_lift
    for height in nominal_heights:
      pose = raw.clone()
      pose[:, 2] += height - nominal
      asset.write_root_link_pose_to_sim(pose)
      before = collision_stats(env)
      native_before = native_penetration(env)
      guard(env, ids)
      lift = guard.last_lift.detach().cpu().numpy().copy()
      remaining = guard.required_lift(env, ids).detach().cpu().numpy().copy()
      after = collision_stats(env)
      native_after = native_penetration(env)
      for kind, name in enumerate(names):
        selected = types == kind
        entry = {
          "reset": reset, "terrain": name, "nominal_height_m": height,
          "samples": int(selected.sum()),
          "raw_penetrating_envs": int(np.count_nonzero(before["penetration_m"][selected] > 1e-5)),
          "raw_max_penetration_m": float(before["penetration_m"][selected].max()),
          "raw_epa_envs": int(np.count_nonzero(before["epa"][selected])),
          "native_raw_penetrating_envs": int(np.count_nonzero(native_before[selected] > 1e-5)),
          "native_raw_max_penetration_m": float(native_before[selected].max()),
          "lifted_envs": int(np.count_nonzero(lift[selected] > 1e-5)),
          "max_lift_m": float(lift[selected].max()),
          "protected_penetrating_envs": int(np.count_nonzero(after["penetration_m"][selected] > 1e-5)),
          "protected_max_penetration_m": float(after["penetration_m"][selected].max()),
          "protected_epa_envs": int(np.count_nonzero(after["epa"][selected])),
          "protected_other_overflow_envs": int(np.count_nonzero(after["other_overflow"][selected])),
          "native_protected_penetrating_envs": int(np.count_nonzero(native_after[selected] > 1e-5)),
          "native_protected_max_penetration_m": float(native_after[selected].max()),
          "remaining_lift_m": float(remaining[selected].max()),
        }
        entry["passed"] = (not entry["protected_penetrating_envs"] and not entry["protected_epa_envs"]
                           and not entry["protected_other_overflow_envs"] and entry["remaining_lift_m"] < 1e-5
                           and not entry["native_protected_penetrating_envs"] and after["finite"])
        passed &= entry["passed"]
        results.append(entry)
    print(f"[SpawnCheck] Reset {reset + 1}/{resets}: protected placement passed={passed}", flush=True)
  return {"passed": passed, "device": str(env.device), "rows": rows, "terrain_types": names,
          "seed": env.cfg.seed, "nominal_heights_m": list(nominal_heights),
          "versions": {name: version(name) for name in ("mjlab", "mujoco", "mujoco-warp", "warp-lang")},
          "num_envs": env.num_envs, "resets": resets, "results": results,
          "note": "All configured terrain types and difficulty rows are covered. Each height uses "
                  "the same frozen reset. This tests placement, not later dynamic penetration. "
                  "No observed EPA at a raw reset does not establish the origin of a training EPA warning."}
