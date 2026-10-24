"""Summarize recorded failures or probe a finite input state, without PPO.

Example:
  python scripts/inspect_nan_dump.py logs/go2w_nan_dump/nan_dump_latest.npz

The default replay probe uses synthetic finite hold targets. --recorded-inputs
instead restores the saved control, warmstart and randomized model fields.
Recorded failure arrays are reported separately from either replay.
"""

import argparse
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

from mjlab.sim.randomization import expand_model_fields


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
  use_recorded = getattr(args, "recorded_inputs", False)
  with np.load(args.dump, allow_pickle=False) as archive:
    recorded = recorded_summary(archive)
    keys = sorted(key for key in archive.files if key.startswith("states_step_"))
    # --recorded-inputs defaults to the last saved pre-operation frame.
    key = keys[-2] if use_recorded and args.frame == -1 and keys[-1] == recorded["failure_state_key"] else keys[args.frame]
    state = np.asarray(archive[key][args.dump_index])
    inputs, model_fields = recorded_inputs(archive, key, args.dump_index) if use_recorded else ({}, {})
  print("Recorded data:", json.dumps(recorded, indent=2), flush=True)
  if args.summary_only or key == recorded["failure_state_key"]:
    if args.output:
      args.output.parent.mkdir(parents=True, exist_ok=True)
      args.output.write_text(json.dumps(recorded, indent=2), encoding="utf-8")
    if not args.summary_only:
      print("Selected frame is the recorded failure. Use --frame=-2 for a synthetic pre-failure probe.")
    return recorded
  model_path = args.model
  if model_path is None:
    model_path = (args.dump.parent / recorded["model_file"] if recorded["model_file"] else
                  args.dump.with_name(args.dump.name.replace("nan_dump_", "model_")).with_suffix(".mjb"))
  model = mujoco.MjModel.from_binary_path(str(model_path))
  if not np.isfinite(state).all():
    raise ValueError("Selected input state already contains NaN/Inf.")
  if args.wheel_margin is not None:
    if use_recorded:
      raise ValueError("--wheel-margin changes the recorded model; use the synthetic probe for overrides")
    for geom_id in range(model.ngeom):
      name = model.geom(geom_id).name
      if "wheel" in name and "collision" in name:
        model.geom_margin[geom_id] = args.wheel_margin
  for name, values in model_fields.items():
    getattr(model, name)[:] = values[0]

  native = mujoco.MjData(model)
  mujoco.mj_setState(model, native, state, mujoco.mjtState.mjSTATE_PHYSICS)
  mujoco.mj_fwdPosition(model, native)
  if use_recorded:
    restore_inputs(native, inputs)
    control = inputs["ctrl"][0]
  else:
    # Synthetic hold targets are deliberately separate from original inputs.
    control = np.zeros(model.nu)
    for aid in range(model.nu):
      if model.actuator_trntype[aid] != mujoco.mjtTrn.mjTRN_JOINT:
        raise ValueError("This hold-control probe requires joint actuators.")
      joint_id = model.actuator_trnid[aid, 0]
      if model.actuator_biasprm[aid, 1] != 0:
        control[aid] = native.qpos[model.jnt_qposadr[joint_id]]
      else:
        control[aid] = np.clip(native.qvel[model.jnt_dofadr[joint_id]], -20.0, 20.0)
    native.ctrl[:] = control
  mujoco.mj_forward(model, native)
  native_contacts = [
    {
      "geoms": [model.geom(int(contact.geom1)).name, model.geom(int(contact.geom2)).name],
      "distance": float(contact.dist),
      "position": contact.pos.tolist(),
    }
    for contact in native.contact
  ]
  versions = {}
  for package in ("mujoco", "mujoco-warp", "warp-lang", "mjlab"):
    try:
      versions[package] = version(package)
    except PackageNotFoundError:
      versions[package] = None
  report = {
    "versions": versions,
    "device": args.device,
    "dump": str(args.dump.resolve()),
    "model": str(model_path.resolve()),
    "state_key": key,
    "dump_index": args.dump_index,
    "recorded_data": recorded,
    "wheel_margin_override": args.wheel_margin,
    "input_source": "recorded" if use_recorded else "synthetic_hold",
    "restored_model_fields": sorted(model_fields),
    "note": ("Saved pre-operation inputs and randomized model fields, with the recorded solver. "
             "Geometry transforms are recomputed on the replay device; this is not an exact GPU checkpoint. "
             "A single-world probe may not reproduce a multi-world GPU failure." if use_recorded else
             "Synthetic finite hold controls and zero warmstart; not the original training inputs or randomized world model. "
             "Native and Warp contact-distance margin conventions can differ."),
    "native": {
      "fields": finite_summary(native, ("qpos", "qvel", "qacc", "sensordata")),
      "contacts": native_contacts,
    },
    "warp": [],
  }
  wp.init()
  print("Versions:", json.dumps(versions), flush=True)
  print("State:", key, "root XYZ:", native.qpos[:3].tolist(), flush=True)
  with wp.ScopedDevice(args.device):
    solvers = (mujoco.mjtSolver(model.opt.solver),) if use_recorded else (mujoco.mjtSolver.mjSOL_NEWTON, mujoco.mjtSolver.mjSOL_CG)
    for solver in solvers:
      model.opt.solver = solver
      wm = mjw.put_model(model)
      wd = mjw.make_data(model, nworld=1, nconmax=128, njmax=2048)
      wd.qpos.assign(native.qpos[None].astype(np.float32))
      wd.qvel.assign(native.qvel[None].astype(np.float32))
      if model.na:
        wd.act.assign(native.act[None].astype(np.float32))
      wd.ctrl.assign(control[None].astype(np.float32))
      wd.qacc_warmstart.zero_()
      if use_recorded:
        expand_model_fields(wm, 1, model_fields)
        for name, values in model_fields.items():
          getattr(wm, name).assign(values)
        restore_inputs(wd, inputs)
      # Read contacts before running any constraint solver. This distinguishes
      # a narrowphase error from a failure introduced during Newton solving.
      mjw.fwd_position(wm, wd, factorize=False)
      wp.synchronize()
      overflow_before_solver = wd.overflow.numpy().tolist()
      count = min(int(wd.nacon.numpy()[0]), wd.naconmax)
      geoms = wd.contact.geom.numpy()[:count].copy()
      distances = wd.contact.dist.numpy()[:count].copy()
      positions = wd.contact.pos.numpy()[:count].copy()
      centers = wd.geom_xpos.numpy()[0].copy()
      mjw.forward(wm, wd)
      wp.synchronize()
      result = {
        "solver": solver.name,
        "overflow_before_solver": overflow_before_solver,
        "overflow_after_forward": wd.overflow.numpy().tolist(),
        "after_forward": finite_summary(wd, ("qpos", "qvel", "qacc", "qacc_smooth", "sensordata")),
        "nefc": int(wd.nefc.numpy()[0]),
        "contacts_before_solver": [
          {
            "geoms": [model.geom(int(g)).name for g in geoms[i]],
            "distance": float(distances[i]),
            "position": positions[i].tolist(),
            "distance_to_geom_centers_m": np.linalg.norm(centers[geoms[i]] - positions[i], axis=1).tolist(),
          }
          for i in range(count)
        ],
      }
      if use_recorded:
        # Probe step from the original input, not a warmstart changed by forward.
        restore_inputs(wd, inputs)
      mjw.step(wm, wd)
      wp.synchronize()
      result["after_step"] = finite_summary(wd, ("qpos", "qvel", "qacc", "sensordata"))
      result["overflow_after_step"] = wd.overflow.numpy().tolist()
      report["warp"].append(result)
      print(solver.name, "forward qacc finite:", result["after_forward"]["qacc"]["finite"],
            "step qvel finite:", result["after_step"]["qvel"]["finite"], flush=True)
      for contact in result["contacts_before_solver"]:
        print(" ", json.dumps(contact), flush=True)
  if args.output:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print("Report:", args.output.resolve(), flush=True)
  return report


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("dump", type=Path)
  parser.add_argument("--model", type=Path)
  parser.add_argument("--device", default="cpu", help="Warp device, e.g. cpu or cuda:0")
  parser.add_argument("--frame", type=int, default=-1)
  parser.add_argument("--recorded-inputs", action="store_true", help="Restore original inputs, randomized model fields and solver; defaults to the last pre-operation frame.")
  parser.add_argument("--summary-only", action="store_true", help="Report original stored fields without replay or requiring an MJB.")
  parser.add_argument("--dump-index", type=int, default=0, help="Index among the environments stored in the dump.")
  parser.add_argument("--wheel-margin", type=float, help="Optional in-memory probe; does not change model/config files.")
  parser.add_argument("--output", type=Path)
  inspect(parser.parse_args())
