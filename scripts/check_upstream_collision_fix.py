"""Compare pinned upstream EPA fixes without installing or modifying packages.

Each fresh subprocess imports a private copy of the installed 3.16.0 package.
Only the exact production diffs from PRs 1771/1796 differ between variants. All
copies receive both upstream regression tests. A candidate must preserve native
contacts, pass upstream/project regressions and a short Go2W rollout/PPO check. This is not a
long-run stability guarantee or an automatic production dependency change.
"""

import argparse
from datetime import datetime
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
RESOURCES = Path(__file__).with_name("upstream_collision_patches")
VARIANTS = {"baseline": (), "pr1771": (1771,), "pr1796": (1796,), "both": (1771, 1796)}


def digest(text):
  return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path, data):
  temporary = path.with_suffix(".partial.json")
  temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
  temporary.replace(path)


def apply_exact_patch(text, patch, target):
  """Apply unique complete hunk contexts; never fuzz, skip or partially apply."""
  lines = patch.splitlines(keepends=True)
  if lines[:2] != [f"--- a/{target}\n", f"+++ b/{target}\n"]:
    raise ValueError(f"Unexpected patch target: {target}")
  i, applied = 2, 0
  while i < len(lines):
    if not re.match(r"^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@", lines[i]):
      raise ValueError("Invalid upstream patch hunk")
    i += 1
    old, new = [], []
    while i < len(lines) and not lines[i].startswith("@@"):
      line = lines[i]
      if line.startswith((" ", "-")):
        old.append(line[1:])
      if line.startswith((" ", "+")):
        new.append(line[1:])
      if line[:1] not in (" ", "-", "+"):
        raise ValueError("Unsupported upstream patch line")
      i += 1
    before, after = "".join(old), "".join(new)
    if not before or text.count(before) != 1:
      raise ValueError(f"Upstream hunk does not match uniquely in {target}; refusing to adapt it")
    text = text.replace(before, after, 1)
    applied += 1
  if not applied:
    raise ValueError("Empty upstream patch")
  return text


def prepare_variants(package, output, manifest):
  target = manifest["target"]
  baseline = (package.parent / target).read_text(encoding="utf-8")
  if digest(baseline) != manifest["baseline_sha256"]:
    raise ValueError("Installed collision_gjk.py differs from official v3.16.0; refusing to patch it")
  patches, test_patches = {}, {}
  for pr in (1771, 1796):
    info = manifest["pull_requests"][str(pr)]
    patches[pr] = (RESOURCES / f"{pr}.patch").read_text(encoding="utf-8")
    test_patches[pr] = (RESOURCES / f"{pr}_tests.patch").read_text(encoding="utf-8")
    if digest(patches[pr]) != info["patch_sha256"] or digest(test_patches[pr]) != info["test_patch_sha256"]:
      raise ValueError(f"Bundled upstream PR {pr} patch checksum mismatch")
  result = {}
  for name, prs in VARIANTS.items():
    directory = output / name
    shutil.copytree(package, directory / package.name,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"))
    text = baseline
    for pr in prs:
      text = apply_exact_patch(text, patches[pr], target)
    (directory / target).write_text(text, encoding="utf-8", newline="\n")
    # The same tests are installed in every private copy, including baseline.
    for patch in test_patches.values():
      test_target = patch.splitlines()[0].removeprefix("--- a/")
      if test_target not in ("mujoco_warp/_src/collision_gjk_test.py", "mujoco_warp/_src/collision_driver_test.py"):
        raise ValueError("Unexpected upstream test file")
      file = directory / test_target
      file.write_text(apply_exact_patch(file.read_text(encoding="utf-8"), patch, test_target),
                      encoding="utf-8", newline="\n")
    result[name] = {"directory": str(directory), "pull_requests": list(prs), "collision_sha256": digest(text)}
  return result


def child_environment(directory, output):
  env = os.environ.copy()
  env["PYTHONPATH"] = os.pathsep.join([str(directory), str(ROOT), env.get("PYTHONPATH", "")])
  env["PYTHONDONTWRITEBYTECODE"] = "1"
  env["PYTHONIOENCODING"] = "utf-8"
  env["WARP_CACHE_PATH"] = str(output / "warp_cache")
  return env


def run_child(arguments, env, log):
  with log.open("w", encoding="utf-8") as stream:
    return subprocess.run([sys.executable, *map(str, arguments)], cwd=ROOT, env=env,
                          stdout=stream, stderr=subprocess.STDOUT, check=False).returncode


def source_matches(report, variant):
  source = Path(report.get("engine_source", "")).resolve()
  expected = Path(variant["directory"]) / "mujoco_warp"
  return source.is_relative_to(expected.resolve()) and report.get("collision_source_sha256") == variant["collision_sha256"]


def assess_contacts(report, position_atol=.001, distance_atol=.00005, normal_angle_atol=5.0):
  """Reject finite-but-wrong and missing contacts, using the native oracle."""
  import numpy as np
  from scipy.optimize import linear_sum_assignment

  warp = report["warp"][0]
  contacts, native = warp["contacts_before_solver"], report["native"]["contacts"]
  finite = all(warp[stage][field]["finite"] for stage in ("after_forward", "after_step")
               for field in ("qpos", "qvel", "qacc", "sensordata"))
  zero_flags = all(flag == 0 for key in ("overflow_before_solver", "overflow_after_forward", "overflow_after_step")
                   for flag in warp[key])
  result = {"finite_physics": finite, "zero_overflow": zero_flags,
            "warp_contacts": len(contacts), "native_contacts": len(native), "matches_native": False,
            "oracle_valid": False,
            "max_position_error_m": None, "max_distance_error_m": None, "max_normal_angle_deg": None,
            "passed": False}
  if not native:
    return {**result, "reason": "Native oracle has no contacts; this scene cannot validate contact recovery"}
  if not all(report["native"]["fields"][field]["finite"] for field in ("qpos", "qvel", "qacc", "sensordata")):
    return {**result, "reason": "Native oracle physics is nonfinite"}
  for contact in native:
    if (not all(np.isfinite(contact[field]).all() for field in ("position", "distance", "normal"))
        or not np.isclose(np.linalg.norm(contact["normal"]), 1, atol=.001, rtol=0)):
      return {**result, "reason": "Native oracle contact geometry is invalid"}
  result["oracle_valid"] = True
  if not contacts or not all(c["contact_sanity"] for c in contacts):
    return {**result, "reason": "Missing or geometrically implausible Warp contacts"}

  def grouped(values):
    groups = {}
    for contact in values:
      pair = tuple(contact["geoms"])
      key = tuple(sorted(pair))
      normal = np.asarray(contact["normal"]) * (1 if pair == key else -1)
      groups.setdefault(key, []).append((contact, normal))
    return groups

  actual, expected = grouped(contacts), grouped(native)
  if actual.keys() != expected.keys() or any(len(actual[k]) != len(expected[k]) for k in actual):
    return {**result, "reason": "Native and Warp geom pairs/contact counts differ"}
  pos_error = dist_error = normal_angle = 0.0
  for key, values in actual.items():
    refs = expected[key]
    cost = np.array([[np.linalg.norm(np.asarray(a[0]["position"]) - b[0]["position"]) for b in refs] for a in values])
    rows, cols = linear_sum_assignment(cost)
    for row, col in zip(rows, cols, strict=True):
      a, na = values[row]
      b, nb = refs[col]
      pos_error = max(pos_error, float(cost[row, col]))
      dist_error = max(dist_error, abs(a["distance"] - b["distance"]))
      angle = np.degrees(np.arccos(np.clip(np.dot(na, nb) / (np.linalg.norm(na) * np.linalg.norm(nb)), -1, 1)))
      normal_angle = max(normal_angle, float(angle))
  matches = pos_error <= position_atol and dist_error <= distance_atol and normal_angle <= normal_angle_atol
  return {**result, "matches_native": bool(matches), "max_position_error_m": pos_error,
          "max_distance_error_m": dist_error, "max_normal_angle_deg": normal_angle,
          "passed": bool(finite and zero_flags and matches)}


def upstream_test_worker(device, output):
  import importlib
  import warp as wp
  from absl import flags
  import mujoco_warp as mjw

  wp.init()
  wp.set_device(device)
  if not flags.FLAGS.is_parsed():
    flags.FLAGS(["upstream-collision-tests"])
  suite = unittest.TestSuite()
  for module_name, class_name, prefix in (
      ("collision_gjk_test", "GJKTest", "test_cylinder_box_side_witness"),
      ("collision_driver_test", "CollisionTest", "test_cylinder_box_shallow_parity")):
    cls = getattr(importlib.import_module("mujoco_warp._src." + module_name), class_name)
    selected = [test for test in unittest.defaultTestLoader.loadTestsFromTestCase(cls)
                if test._testMethodName.startswith(prefix)]
    if not selected:
      raise RuntimeError(f"Missing upstream regression {prefix}")
    suite.addTests(selected)
  result = unittest.TextTestRunner(verbosity=2).run(suite)
  write_json(output, {"tests_run": result.testsRun, "passed": result.wasSuccessful(),
                     "engine_source": str(Path(mjw.__file__).resolve()),
                     "collision_source_sha256": digest((Path(mjw.__file__).parent / "_src/collision_gjk.py").read_text(encoding="utf-8")),
                     "failures": [str(test) for test, _ in result.failures],
                     "errors": [str(test) for test, _ in result.errors]})
  return 0 if result.wasSuccessful() else 1


def check(args):
  manifest = json.loads((RESOURCES / "manifest.json").read_text(encoding="utf-8"))
  for package, expected in (("mujoco-warp", manifest["mujoco_warp"]), ("mujoco", manifest["mujoco"]),
                            ("warp-lang", manifest["warp_lang"])):
    if metadata.version(package) != expected:
      raise ValueError(f"Expected {package}=={expected}; keep the captured environment unchanged")
  package = Path(metadata.distribution("mujoco-warp").locate_file("mujoco_warp")).resolve()
  output = (args.output_dir or ROOT / "logs/upstream_collision_check" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")).resolve()
  output.mkdir(parents=True, exist_ok=False)
  variants = prepare_variants(package, output, manifest)
  report = {"device": args.device, "dump": str(args.dump.resolve()), "output": str(output),
            "upstream_manifest": manifest, "variants": variants, "baseline_reproduced": False,
            "selected_candidate": None, "passed": False,
            "note": "Upstream patches on private 3.16.0 copies; no installation changes. Pass covers this frozen scene, "
                    "three upstream regressions, project regressions and a short Go2W PPO smoke check, "
                    "not long-run training stability."}
  for name, variant in variants.items():
    print(f"[{name}] Replaying original inputs and upstream regression tests...", flush=True)
    directory = Path(variant["directory"])
    env = child_environment(directory, output)
    probe_path = directory / "replay.json"
    code = run_child([ROOT / "scripts/inspect_nan_dump.py", args.dump.resolve(), "--recorded-inputs",
                      "--device", args.device, "--epa-horizon", "24", "--collision-fma", "on",
                      "--output", probe_path], env, directory / "replay.log")
    variant["replay_exit_code"] = code
    variant["source_verified"] = False
    if code == 0 and probe_path.exists():
      probe = json.loads(probe_path.read_text(encoding="utf-8"))
      variant["source_verified"] = source_matches(probe, variant)
      variant["assessment"] = assess_contacts(probe)
    test_path = directory / "upstream_tests.json"
    test_code = run_child([Path(__file__).resolve(), "--test-worker", "--device", args.device,
                           "--worker-output", test_path], env, directory / "upstream_tests.log")
    variant["upstream_tests_exit_code"] = test_code
    variant["upstream_tests"] = json.loads(test_path.read_text(encoding="utf-8")) if test_path.exists() else {"passed": False}
    variant["upstream_source_verified"] = source_matches(variant["upstream_tests"], variant)
    variant["frozen_and_upstream_passed"] = bool(variant["source_verified"] and variant.get("assessment", {}).get("passed")
                             and variant["upstream_source_verified"] and test_code == 0
                             and variant["upstream_tests"].get("passed") and variant["upstream_tests"].get("tests_run") == 3)
    print(f"[{name}] scene={variant.get('assessment', {})}; upstream_tests={variant['upstream_tests'].get('passed')}", flush=True)
    write_json(output / "summary.json", report)
  baseline = variants["baseline"]
  assessment = baseline.get("assessment", {})
  report["baseline_reproduced"] = bool(baseline["source_verified"] and assessment.get("oracle_valid")
      and (not assessment["finite_physics"] or not assessment["matches_native"]))
  if report["baseline_reproduced"]:
    for name in ("pr1771", "pr1796", "both"):
      if not variants[name]["frozen_and_upstream_passed"]:
        continue
      print(f"[{name}] Running the project's regression suite...", flush=True)
      env = child_environment(Path(variants[name]["directory"]), output)
      code = run_child(["-m", "unittest", "discover", "-s", ROOT / "tests", "-q"],
                       env, output / name / "project_tests.log")
      variants[name]["project_tests_exit_code"] = code
      if code != 0:
        write_json(output / "summary.json", report)
        continue
      print(f"[{name}] Running 32-world Go2W, 64 steps and two PPO iterations...", flush=True)
      code = run_child([ROOT / "scripts/check_go2w_upgrade.py", "--device", args.device, "--num-envs", "32",
                        "--steps", "64", "--learn-iterations", "2", "--output-dir", output / name / "go2w_smoke"],
                       env, output / name / "go2w_smoke.log")
      variants[name]["smoke_exit_code"] = code
      write_json(output / "summary.json", report)
      if code == 0:
        report["selected_candidate"] = name
        report["passed"] = True
        break
  report["status"] = "passed_frozen_and_smoke" if report["passed"] else (
      "baseline_not_reproduced_or_unverified" if not report["baseline_reproduced"] else "no_validated_candidate")
  report["installed_collision_unchanged"] = digest((package / "_src/collision_gjk.py").read_text(encoding="utf-8")) == manifest["baseline_sha256"]
  if not report["installed_collision_unchanged"]:
    report["passed"] = False
    report["status"] = "installation_changed_during_check"
  write_json(output / "summary.json", report)
  print(f"Result: {report['status']}; report: {output / 'summary.json'}", flush=True)
  return report


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--dump", type=Path)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--output-dir", type=Path, help="New directory only; existing results are never overwritten")
  parser.add_argument("--test-worker", action="store_true", help=argparse.SUPPRESS)
  parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
  args = parser.parse_args()
  if args.test_worker:
    raise SystemExit(upstream_test_worker(args.device, args.worker_output))
  if args.dump is None or not args.dump.is_file():
    parser.error("--dump must name the original NaN NPZ; keep its MJB beside it")
  report = check(args)
  if not report["passed"]:
    raise SystemExit(1)


if __name__ == "__main__":
  main()
