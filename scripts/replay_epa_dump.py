"""Replay one captured rigid collision at several EPA horizon capacities.

No policy or integration is run. Original device geom transforms and randomized
model fields are restored directly. The capacity override is temporary, applies
only inside this diagnostic process and does not modify installed files.
"""

import argparse
from collections import Counter
from importlib.metadata import version
import json
from pathlib import Path

import mujoco
import mujoco_warp as mjw
import numpy as np
from scipy.optimize import linear_sum_assignment
import warp as wp

from mjlab.sim.randomization import expand_model_fields
from src.utils.collision_probe import temporary_epa_horizon


def _contact_sanity(model, wm, positions, contacts):
  finite = all(np.isfinite(value).all() for value in contacts.values())
  if not finite:
    return False
  if not len(contacts["dist"]):
    return True
  normals = contacts["frame"][:, 0]
  if not np.allclose(np.linalg.norm(normals, axis=1), 1, atol=1e-3, rtol=0):
    return False
  radii = wm.geom_rbound.numpy().reshape(-1, model.ngeom)
  for world, pair, point, depth in zip(contacts["worldid"], contacts["geom"],
                                     contacts["pos"], contacts["dist"], strict=True):
    for geom in pair:
      if model.geom_type[geom] in (mujoco.mjtGeom.mjGEOM_PLANE, mujoco.mjtGeom.mjGEOM_HFIELD):
        continue
      bound = radii[world % len(radii), geom]
      # Contact midpoints can be inside the other geom under penetration.
      if np.linalg.norm(point - positions[world, geom]) > bound + abs(depth) + .01:
        return False
  return True


def _run_collision(model_path, arrays, metadata, device, capacity):
  model = mujoco.MjModel.from_binary_path(str(model_path))
  if model.nflex:
    raise ValueError("This replay supports rigid collisions only, not flex")
  with wp.ScopedDevice(device), temporary_epa_horizon(capacity):
    wm = mjw.put_model(model)
    nworld = arrays["geom_xpos"].shape[0]
    fields = [key[6:] for key in arrays if key.startswith("model_")]
    expand_model_fields(wm, nworld, fields)
    for name in fields:
      getattr(wm, name).assign(arrays[f"model_{name}"])
    for name in ("broadphase", "broadphase_filter", "ccd_iterations"):
      setattr(wm.opt, name, metadata[name])
    wm.opt.ccd_tolerance.assign(arrays["option_ccd_tolerance"])
    recorded_count = len(arrays.get("recorded_contact_dist", []))
    data = mjw.make_data(model, nworld=nworld,
                         nconmax=max(64, metadata["nconmax_per_world"], recorded_count + 32),
                         njmax=1)
    mjw.kinematics(wm, data)
    data.geom_xpos.assign(arrays["geom_xpos"])
    data.geom_xmat.assign(arrays["geom_xmat"])
    mjw.collision(wm, data)
    wp.synchronize_device(device)
    count = min(int(data.nacon.numpy()[0]), data.naconmax)
    contacts = {name: getattr(data.contact, name).numpy()[:count].copy()
                for name in ("worldid", "geom", "pos", "dist", "frame")}
    flags = data.overflow.numpy().copy()
    summary = {
      "capacity": capacity, "overflow": flags.tolist(),
      "epa_overflow": bool(np.any(flags & int(mjw.OverflowType.EPA_HORIZON))),
      "other_overflow": bool(np.any(flags & ~int(mjw.OverflowType.EPA_HORIZON))),
      "contacts": count,
      "contact_sanity": _contact_sanity(model, wm, arrays["geom_xpos"], contacts),
      "horizon_buffer_bytes": data.naccdmax * capacity * 4,
    }
    return summary, contacts


def compare_contacts(actual, reference, position_atol=1e-5, distance_atol=1e-5, normal_atol=1e-4):
  """Match contacts per geom pair so packing/point order does not affect verdict."""
  def groups(contacts):
    result = {}
    for i, (world, pair) in enumerate(zip(contacts["worldid"], contacts["geom"], strict=True)):
      result.setdefault((int(world), int(pair[0]), int(pair[1])), []).append(i)
    return result

  a, b = groups(actual), groups(reference)
  errors = {"max_position_error_m": 0.0, "max_distance_error_m": 0.0, "max_normal_error": 0.0}
  if a.keys() != b.keys() or any(len(a[key]) != len(b[key]) for key in a):
    return {"matches": False, "reason": "different geom pairs or contact counts", **errors}
  for key in a:
    ai, bi = np.array(a[key]), np.array(b[key])
    cost = np.linalg.norm(actual["pos"][ai, None] - reference["pos"][bi][None], axis=-1)
    rows, cols = linear_sum_assignment(cost)
    ai, bi = ai[rows], bi[cols]
    errors["max_position_error_m"] = max(errors["max_position_error_m"], float(cost[rows, cols].max()))
    errors["max_distance_error_m"] = max(errors["max_distance_error_m"], float(np.abs(actual["dist"][ai] - reference["dist"][bi]).max()))
    errors["max_normal_error"] = max(errors["max_normal_error"], float(np.linalg.norm(actual["frame"][ai, 0] - reference["frame"][bi, 0], axis=-1).max()))
  matches = (errors["max_position_error_m"] <= position_atol
             and errors["max_distance_error_m"] <= distance_atol
             and errors["max_normal_error"] <= normal_atol)
  return {"matches": matches, **errors}


def replay_epa_dump(path, device="cpu", capacities=(64, 128, 256)):
  path = Path(path)
  with np.load(path, allow_pickle=False) as archive:
    metadata = json.loads(archive["metadata_json"].item())
    if metadata.get("format") != "epa_collision_snapshot_v1":
      raise ValueError("Expected an EPA collision snapshot")
    if metadata.get("capture_status") != "complete":
      raise ValueError(f"Capture is incomplete: {metadata.get('capture_errors')}")
    for package, captured_version in metadata["versions"].items():
      if version(package) != captured_version:
        raise ValueError(f"Replay requires {package}=={captured_version}, found {version(package)}")
    arrays = {key: archive[key].copy() for key in archive.files if key != "metadata_json"}
  baseline_capacity = metadata["epa_horizon"]
  if len(set(capacities)) < 2 or min(capacities) <= baseline_capacity:
    raise ValueError("Provide at least two capacities above the recorded baseline (candidate and reference)")
  if not all(np.isfinite(arrays[name]).all() for name in ("geom_xpos", "geom_xmat")):
    raise ValueError("Recorded collision transforms contain NaN/Inf")
  ordered = [baseline_capacity] + sorted(set(capacities))
  runs = [_run_collision(path.parent / metadata["model_file"], arrays, metadata, device, size)
          for size in ordered]
  baseline, reference = runs[0][0], runs[-1][0]
  # An overflowing local query need not change the final contact count (e.g.
  # heightfield prism contacts are reduced). Judge the actual contact data,
  # rather than requiring more contacts than the baseline.
  reference_ok = (not reference["epa_overflow"] and not reference["other_overflow"]
                  and reference["contact_sanity"] and reference["contacts"] > 0)
  candidates = []
  for result, contacts in runs[1:-1]:
    comparison = compare_contacts(contacts, runs[-1][1]) if result["contact_sanity"] and reference["contact_sanity"] else {"matches": False}
    result.update(comparison)
    result["passed"] = bool(baseline["epa_overflow"] and not baseline["other_overflow"] and reference_ok
                            and not result["epa_overflow"] and not result["other_overflow"]
                            and result["contact_sanity"] and comparison["matches"])
    candidates.append(result)
  def pair_counts(contacts):
    return Counter((int(world), int(pair[0]), int(pair[1]))
                   for world, pair in zip(contacts["worldid"], contacts["geom"], strict=True))

  recovered = pair_counts(runs[-1][1]) - pair_counts(runs[0][1])
  names = metadata["geom_names"]
  return {
    "dump": str(path), "device": device, "recorded_device": metadata["device"],
    "dumped_env_ids": metadata["dumped_env_ids"], "baseline_reproduced": baseline["epa_overflow"],
    "baseline": baseline, "reference": reference, "candidates": candidates,
    "recovered_geom_pairs": [
      {"env_id": metadata["dumped_env_ids"][world], "geom_ids": [a, b],
       "geom_names": [names[a], names[b]], "additional_contacts": count}
      for (world, a, b), count in recovered.items()],
    "passed": any(result["passed"] for result in candidates),
    "note": "Pass means this frozen collision reproduces the baseline overflow and a candidate "
            "clears overflow with sane, nonempty contacts consistent with the larger-capacity reference. "
            "Contact counts need not increase; recovered_geom_pairs may be empty. It is not independent "
            "ground truth or proof of long-run training stability. A GPU-only failure may not reproduce on CPU.",
  }


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("dump", type=Path)
  parser.add_argument("--device", default="cpu")
  parser.add_argument("--capacities", type=int, nargs="+", default=[64, 128, 256])
  parser.add_argument("--output", type=Path)
  args = parser.parse_args()
  report = replay_epa_dump(args.dump, args.device, args.capacities)
  print(json.dumps(report, indent=2), flush=True)
  output = args.output or args.dump.with_suffix(".replay.json")
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(json.dumps(report, indent=2), encoding="utf-8")
  if not report["passed"]:
    raise SystemExit(1)


if __name__ == "__main__":
  main()
