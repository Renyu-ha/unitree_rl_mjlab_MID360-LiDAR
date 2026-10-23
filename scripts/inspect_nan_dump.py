"""Compare native MuJoCo and Warp at a saved pre-failure state, without PPO.

Example:
  python scripts/inspect_nan_dump.py logs/go2w_nan_dump/nan_dump_latest.npz

Controls are synthetic finite hold targets. The dump does not contain the
original controls, warmstarts, or the randomized per-environment model.
"""

import argparse
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp


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


def inspect(args):
  model_path = args.model
  if model_path is None:
    model_path = args.dump.with_name(args.dump.name.replace("nan_dump_", "model_")).with_suffix(".mjb")
  model = mujoco.MjModel.from_binary_path(str(model_path))
  with np.load(args.dump, allow_pickle=False) as archive:
    keys = sorted(key for key in archive.files if key.startswith("states_step_"))
    key = keys[args.frame]
    saved = archive[key]
    state = np.asarray(saved[args.dump_index])
  if not np.isfinite(state).all():
    raise ValueError("Selected input state already contains NaN/Inf.")
  if args.wheel_margin is not None:
    for geom_id in range(model.ngeom):
      name = model.geom(geom_id).name
      if "wheel" in name and "collision" in name:
        model.geom_margin[geom_id] = args.wheel_margin

  native = mujoco.MjData(model)
  mujoco.mj_setState(model, native, state, mujoco.mjtState.mjSTATE_PHYSICS)
  mujoco.mj_fwdPosition(model, native)
  # Go2W's position servos hold the saved joint positions; wheel velocity
  # servos track saved speeds clipped to the task's +/-20 rad/s range.
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
    "wheel_margin_override": args.wheel_margin,
    "note": "Synthetic finite hold controls and zero warmstart; not the original training inputs or randomized world model. Native and Warp contact-distance margin conventions can differ.",
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
    for solver in (mujoco.mjtSolver.mjSOL_NEWTON, mujoco.mjtSolver.mjSOL_CG):
      model.opt.solver = solver
      wm = mjw.put_model(model)
      wd = mjw.make_data(model, nworld=1, nconmax=128, njmax=2048)
      wd.qpos.assign(native.qpos[None].astype(np.float32))
      wd.qvel.assign(native.qvel[None].astype(np.float32))
      if model.na:
        wd.act.assign(native.act[None].astype(np.float32))
      wd.ctrl.assign(control[None].astype(np.float32))
      wd.qacc_warmstart.zero_()
      # Read contacts before running any constraint solver. This distinguishes
      # a narrowphase error from a failure introduced during Newton solving.
      mjw.fwd_position(wm, wd, factorize=False)
      wp.synchronize()
      count = int(wd.nacon.numpy()[0])
      geoms = wd.contact.geom.numpy()[:count].copy()
      distances = wd.contact.dist.numpy()[:count].copy()
      positions = wd.contact.pos.numpy()[:count].copy()
      mjw.forward(wm, wd)
      wp.synchronize()
      result = {
        "solver": solver.name,
        "after_forward": finite_summary(wd, ("qpos", "qvel", "qacc", "qacc_smooth", "sensordata")),
        "nefc": int(wd.nefc.numpy()[0]),
        "contacts_before_solver": [
          {
            "geoms": [model.geom(int(g)).name for g in geoms[i]],
            "distance": float(distances[i]),
            "position": positions[i].tolist(),
          }
          for i in range(count)
        ],
      }
      mjw.step(wm, wd)
      wp.synchronize()
      result["after_step"] = finite_summary(wd, ("qpos", "qvel", "qacc", "sensordata"))
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
  parser.add_argument("--dump-index", type=int, default=0, help="Index among the environments stored in the dump.")
  parser.add_argument("--wheel-margin", type=float, help="Optional in-memory probe; does not change model/config files.")
  parser.add_argument("--output", type=Path)
  inspect(parser.parse_args())
