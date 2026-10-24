"""Inspect a NaN archive or replay its saved inputs with the fixed backend.

Keep the NPZ and its referenced MJB together. Replay restores the recorded
controls, warmstart, randomization and solver; it does not run a policy.
"""

import argparse
from importlib.metadata import version
import json
from pathlib import Path

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

from mjlab.sim.randomization import expand_model_fields
from src.utils.physics_backend import physics_backend_info


def recorded_inputs(archive, state_key, dump_index):
  """Read a pre-operation input, never substitute a post-failure state."""
  metadata = json.loads(archive["metadata_json"].item()) if "metadata_json" in archive else {}
  if metadata.get("capture_status") != "complete":
    raise ValueError("Recorded-input replay requires a complete capture, including randomized model fields")
  prefix = "inputs_step_" + state_key.removeprefix("states_step_") + "_"
  fields = {key[len(prefix):]: archive[key][dump_index:dump_index + 1].copy()
            for key in archive.files if key.startswith(prefix)}
  if not {"qpos", "qvel", "ctrl", "qacc_warmstart"}.issubset(fields):
    raise ValueError("Selected frame lacks recorded inputs; select a pre-operation frame from a v2 dump")
  if not all(np.isfinite(value).all() for value in fields.values()):
    raise ValueError("Selected recorded inputs contain NaN/Inf")
  model_fields = {key[6:]: archive[key][dump_index:dump_index + 1].copy()
                  for key in archive.files if key.startswith("model_")}
  return fields, model_fields


def restore_inputs(data, fields):
  for name, values in fields.items():
    target = getattr(data, name)
    if hasattr(target, "assign"):
      target.assign(values)
    elif name == "time":
      data.time = float(values.reshape(-1)[0])
    else:
      target[:] = values[0]


def finite_summary(data, fields):
  result = {}
  for name in fields:
    array = getattr(data, name)
    values = array.numpy() if hasattr(array, "numpy") else np.asarray(array)
    finite = np.isfinite(values)
    result[name] = {
      "finite": bool(finite.all()),
      "nonfinite_count": int((~finite).sum()),
      "max_abs_finite": float(np.abs(values[finite]).max()) if finite.any() else None,
    }
  return result


def contact_detail(model, geoms, position, distance, frame, centers):
  """Report bad collision geometry separately from a later solver NaN."""
  values = np.linalg.norm(centers[geoms] - position, axis=1)
  normal = frame[0]
  sane = bool(np.isfinite(position).all() and np.isfinite(distance)
              and np.isfinite(frame).all()
              and np.isclose(np.linalg.norm(normal), 1, atol=1e-3, rtol=0))
  for geom, radius_distance in zip(geoms, values, strict=True):
    if model.geom_type[geom] in (mujoco.mjtGeom.mjGEOM_PLANE, mujoco.mjtGeom.mjGEOM_HFIELD):
      continue
    sane &= bool(radius_distance <= model.geom_rbound[geom] + abs(distance) + .01)
  return {
    "geoms": [model.geom(int(g)).name for g in geoms],
    "distance": float(distance) if np.isfinite(distance) else None,
    "position": np.where(np.isfinite(position), position, None).tolist(),
    "normal": np.where(np.isfinite(normal), normal, None).tolist(),
    "distance_to_geom_centers_m": [float(value) if np.isfinite(value) else None for value in values],
    "contact_sanity": bool(sane),
  }


def recorded_summary(archive):
  keys = sorted(key for key in archive.files if key.startswith("states_step_"))
  metadata = json.loads(str(archive["metadata_json"].item())) if "metadata_json" in archive else {}
  invalid = [key for key in keys if not np.isfinite(archive[key]).all()]
  fields = {}
  for key in archive.files:
    if not key.startswith("failure_"):
      continue
    value = archive[key]
    finite = np.isfinite(value)
    fields[key] = {
      "shape": list(value.shape), "nonfinite_count": int((~finite).sum()),
      "nonfinite_indices": np.argwhere(~finite)[:20].tolist(),
    }
  contacts = []
  if all(key in archive for key in ("failure_contact_geom", "failure_contact_pos", "failure_contact_geom_xpos")):
    for pair, point, centers in zip(archive["failure_contact_geom"], archive["failure_contact_pos"],
                                    archive["failure_contact_geom_xpos"], strict=True):
      contacts.append({"geom_ids": pair.tolist(),
                       "position": np.where(np.isfinite(point), point, None).tolist(),
                       "geom_centers": np.where(np.isfinite(centers), centers, None).tolist(),
                       "distance_to_geom_centers_m": [float(value) if np.isfinite(value) else None
                                                      for value in np.linalg.norm(centers - point, axis=1)]})
  return {
    "has_recorded_failure": bool(metadata.get("failure_state_key")),
    "failure_stage": metadata.get("failure_stage"),
    "failure_state_key": metadata.get("failure_state_key"),
    "model_file": metadata.get("model_file"),
    "versions": metadata.get("versions"),
    "dumped_env_ids": metadata.get("dumped_env_ids"),
    "nonfinite_fields": metadata.get("nonfinite_fields"),
    "capture_status": metadata.get("capture_status"),
    "capture_errors": metadata.get("capture_errors"),
    "skipped_model_fields": metadata.get("skipped_model_fields"),
    "num_frames": len(keys), "invalid_stored_state_keys": invalid,
    "fields": fields,
    "recorded_contacts": contacts,
    "note": metadata.get("note", "Legacy pre-step history only. The failure result, "
                         "acceleration, controls and original GPU contacts are absent."),
  }


def inspect(args):
  with np.load(args.dump, allow_pickle=False) as archive:
    recorded = recorded_summary(archive)
    keys = sorted(key for key in archive.files if key.startswith("states_step_"))
    if not args.summary_only:
      if not keys:
        raise ValueError("Archive has no recorded physics states")
      frame = args.frame
      if frame == -1 and keys[-1] == recorded["failure_state_key"] and len(keys) > 1:
        frame = -2
      key = keys[frame]
      if key != recorded["failure_state_key"]:
        state = np.asarray(archive[key][args.dump_index])
        inputs, model_fields = recorded_inputs(archive, key, args.dump_index)
  print("Recorded data:", json.dumps(recorded, indent=2), flush=True)
  if args.summary_only or key == recorded["failure_state_key"]:
    if args.output:
      args.output.parent.mkdir(parents=True, exist_ok=True)
      args.output.write_text(json.dumps(recorded, indent=2), encoding="utf-8")
    return recorded
  if not np.isfinite(state).all():
    raise ValueError("Selected input already contains NaN/Inf")
  model_path = args.model or args.dump.parent / recorded["model_file"]
  model = mujoco.MjModel.from_binary_path(str(model_path))
  if model.nflex:
    raise ValueError("Recorded-input replay currently supports rigid models only")
  for name, values in model_fields.items():
    getattr(model, name)[:] = values[0]

  native = mujoco.MjData(model)
  mujoco.mj_setState(model, native, state, mujoco.mjtState.mjSTATE_PHYSICS)
  mujoco.mj_fwdPosition(model, native)
  restore_inputs(native, inputs)
  mujoco.mj_forward(model, native)
  report = {
    **physics_backend_info(),
    "versions": {name: version(name) for name in ("mujoco", "mujoco-warp", "warp-lang", "mjlab")},
    "device": args.device, "dump": str(args.dump.resolve()), "model": str(model_path.resolve()),
    "state_key": key, "dump_index": args.dump_index, "input_source": "recorded",
    "recorded_data": recorded, "restored_model_fields": sorted(model_fields),
    "note": "Saved inputs with the recorded solver, controls, warmstart and randomized model fields. "
            "The fixed backend recomputes transforms on the replay device; this is not an exact GPU checkpoint.",
    "native": {
      "fields": finite_summary(native, ("qpos", "qvel", "qacc", "sensordata")),
      "contacts": [{"geoms": [model.geom(int(c.geom1)).name, model.geom(int(c.geom2)).name],
                    "distance": float(c.dist), "position": c.pos.tolist(), "normal": c.frame[:3].tolist()}
                   for c in native.contact],
    },
    "warp": [],
  }
  wp.init()
  with wp.ScopedDevice(args.device):
    wm = mjw.put_model(model)
    wd = mjw.make_data(model, nworld=1, nconmax=128, njmax=2048)
    expand_model_fields(wm, 1, model_fields)
    for name, values in model_fields.items():
      getattr(wm, name).assign(values)
    restore_inputs(wd, inputs)
    mjw.fwd_position(wm, wd, factorize=False)
    wp.synchronize()
    count = min(int(wd.nacon.numpy()[0]), wd.naconmax)
    geoms = wd.contact.geom.numpy()[:count].copy()
    distances = wd.contact.dist.numpy()[:count].copy()
    positions = wd.contact.pos.numpy()[:count].copy()
    frames = wd.contact.frame.numpy()[:count].copy()
    centers = wd.geom_xpos.numpy()[0].copy()
    result = {
      "solver": mujoco.mjtSolver(model.opt.solver).name,
      "overflow_before_solver": wd.overflow.numpy().tolist(),
      "contacts_before_solver": [contact_detail(model, geoms[i], positions[i], distances[i], frames[i], centers)
                                 for i in range(count)],
    }
    mjw.forward(wm, wd)
    wp.synchronize()
    result["after_forward"] = finite_summary(wd, ("qpos", "qvel", "qacc", "qacc_smooth", "sensordata"))
    result["overflow_after_forward"] = wd.overflow.numpy().tolist()
    result["nefc"] = int(wd.nefc.numpy()[0])
    restore_inputs(wd, inputs)
    mjw.step(wm, wd)
    wp.synchronize()
    result["after_step"] = finite_summary(wd, ("qpos", "qvel", "qacc", "sensordata"))
    result["overflow_after_step"] = wd.overflow.numpy().tolist()
    report["warp"].append(result)
  print(json.dumps(report, indent=2, allow_nan=False), flush=True)
  if args.output:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
  return report


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("dump", type=Path)
  parser.add_argument("--model", type=Path)
  parser.add_argument("--device", default="cpu")
  parser.add_argument("--frame", type=int, default=-1)
  parser.add_argument("--summary-only", action="store_true")
  parser.add_argument("--dump-index", type=int, default=0)
  parser.add_argument("--output", type=Path)
  inspect(parser.parse_args())
