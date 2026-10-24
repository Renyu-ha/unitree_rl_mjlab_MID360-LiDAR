"""Provenance and verdict regressions for the isolated upstream comparison."""

import copy
from contextlib import redirect_stdout
from importlib import metadata
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest import mock

from scripts import check_upstream_collision_fix as checker
from scripts.check_upstream_collision_fix import (
  RESOURCES, apply_exact_patch, assess_contacts, digest, prepare_variants, source_matches,
)


class UpstreamCollisionFixTests(unittest.TestCase):
  def setUp(self):
    self.manifest = json.loads((RESOURCES / "manifest.json").read_text(encoding="utf-8"))

  def test_exact_1771_patch_reconstructs_pinned_upstream_head(self):
    package = Path(metadata.distribution("mujoco-warp").locate_file("mujoco_warp"))
    source = (package / "_src/collision_gjk.py").read_text(encoding="utf-8")
    patch = (RESOURCES / "1771.patch").read_text(encoding="utf-8")
    self.assertEqual(digest(source), self.manifest["baseline_sha256"])
    patched = apply_exact_patch(source, patch, self.manifest["target"])
    self.assertEqual(digest(patched), self.manifest["pull_requests"]["1771"]["head_file_sha256"])
    with self.assertRaisesRegex(ValueError, "does not match uniquely"):
      apply_exact_patch(patched, patch, self.manifest["target"])

  def test_private_variants_preserve_original_and_apply_upstream_tests(self):
    package = Path(metadata.distribution("mujoco-warp").locate_file("mujoco_warp"))
    with TemporaryDirectory() as temporary:
      original = Path(temporary, "site/mujoco_warp")
      original.mkdir(parents=True)
      paths = ["_src/collision_gjk.py", "_src/collision_gjk_test.py", "_src/collision_driver_test.py"]
      for relative in paths:
        target = original / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((package / relative).read_bytes())
      before = {path: (original / path).read_bytes() for path in paths}
      output = Path(temporary, "copies")
      variants = prepare_variants(original, output, self.manifest)
      self.assertEqual(set(variants), {"baseline", "pr1771", "pr1796", "both"})
      self.assertEqual(len({v["collision_sha256"] for v in variants.values()}), 4)
      self.assertEqual(variants["baseline"]["collision_sha256"], self.manifest["baseline_sha256"])
      self.assertEqual(variants["pr1771"]["collision_sha256"], self.manifest["pull_requests"]["1771"]["head_file_sha256"])
      for path in paths:
        self.assertEqual((original / path).read_bytes(), before[path])
      # Every variant has identical upstream test code, including baseline.
      for relative in paths[1:]:
        contents = [(Path(v["directory"]) / "mujoco_warp" / relative).read_bytes() for v in variants.values()]
        self.assertTrue(all(content == contents[0] for content in contents))
      original.joinpath(paths[0]).write_text("modified installation\n", encoding="utf-8")
      with self.assertRaisesRegex(ValueError, "differs from official"):
        prepare_variants(original, Path(temporary, "rejected"), self.manifest)

  def test_wrong_import_cannot_be_reported_as_tested_patch(self):
    with TemporaryDirectory() as temporary:
      variant = {"directory": str(Path(temporary, "pr1771")), "collision_sha256": "expected"}
      report = {"engine_source": str(Path(temporary, "installed/mujoco_warp/__init__.py")),
                "collision_source_sha256": "expected"}
      self.assertFalse(source_matches(report, variant))
      report["engine_source"] = str(Path(variant["directory"], "mujoco_warp/__init__.py"))
      self.assertTrue(source_matches(report, variant))

  def test_missing_or_far_contacts_do_not_pass_finite_physics(self):
    contact = {"geoms": ["wheel", "terrain"], "position": [0, 0, 0], "normal": [0, 0, -1],
               "distance": -.0000188, "contact_sanity": True}
    fields = {name: {"finite": True} for name in ("qpos", "qvel", "qacc", "sensordata")}
    report = {"native": {"contacts": [copy.deepcopy(contact)], "fields": fields}, "warp": [{
      "contacts_before_solver": [contact], "after_forward": fields, "after_step": fields,
      "overflow_before_solver": [0], "overflow_after_forward": [0], "overflow_after_step": [0]}]}
    self.assertTrue(assess_contacts(report)["passed"])
    for change in ("missing", "far", "normal", "depth"):
      with self.subTest(change=change):
        changed = copy.deepcopy(report)
        target = changed["warp"][0]["contacts_before_solver"][0]
        if change == "missing":
          changed["warp"][0]["contacts_before_solver"] = []
        elif change == "far":
          target["position"] = [0, 61.87, 0]
        elif change == "normal":
          target["normal"] = [1, 0, 0]
        else:
          target["distance"] = -.19
        self.assertFalse(assess_contacts(changed)["passed"])

  def test_overall_verdict_requires_reproduction_regressions_and_smoke(self):
    package = Path(metadata.distribution("mujoco-warp").locate_file("mujoco_warp"))
    versions = {"mujoco-warp": "3.16.0", "mujoco": "3.16.0", "warp-lang": "1.18.0"}
    cases = ((False, 0, 0), (True, 1, 0), (True, 0, 1), (True, 0, 0))
    for reproduce, regression_code, smoke_code in cases:
      with self.subTest(reproduce=reproduce, regressions=regression_code, smoke=smoke_code), TemporaryDirectory() as temporary:
        original = Path(temporary, "site/mujoco_warp")
        original.mkdir(parents=True)
        for relative in ("_src/collision_gjk.py", "_src/collision_gjk_test.py", "_src/collision_driver_test.py"):
          target = original / relative
          target.parent.mkdir(parents=True, exist_ok=True)
          target.write_bytes((package / relative).read_bytes())
        calls = []

        def child(arguments, env, log):
          calls.append(str(log.name))
          directory = log.parent
          source = {"engine_source": str(directory / "mujoco_warp/__init__.py"),
                    "collision_source_sha256": digest((directory / "mujoco_warp/_src/collision_gjk.py").read_text(encoding="utf-8"))}
          if log.name == "replay.log":
            contact = {"geoms": ["wheel", "terrain"], "position": [0, 0, 0], "normal": [0, 0, -1],
                       "distance": -.0000188, "contact_sanity": True}
            fields = {name: {"finite": True} for name in ("qpos", "qvel", "qacc", "sensordata")}
            native = copy.deepcopy(contact)
            if reproduce and directory.name == "baseline":
              contact["position"][1] = 61.87
            report = {**source, "native": {"contacts": [native], "fields": fields}, "warp": [{
              "contacts_before_solver": [contact], "after_forward": fields, "after_step": fields,
              "overflow_before_solver": [0], "overflow_after_forward": [0], "overflow_after_step": [0]}]}
            (directory / "replay.json").write_text(json.dumps(report), encoding="utf-8")
            return 0
          if log.name == "upstream_tests.log":
            passed = directory.name == "both"
            (directory / "upstream_tests.json").write_text(json.dumps({**source, "tests_run": 3, "passed": passed}), encoding="utf-8")
            return 0 if passed else 1
          return regression_code if log.name == "project_tests.log" else smoke_code

        args = SimpleNamespace(dump=Path(temporary, "input.npz"), device="cpu", output_dir=Path(temporary, "output"))
        with mock.patch.object(checker.metadata, "version", side_effect=versions.__getitem__), \
             mock.patch.object(checker.metadata, "distribution", return_value=SimpleNamespace(locate_file=lambda _: original)), \
             mock.patch.object(checker, "run_child", side_effect=child), redirect_stdout(io.StringIO()):
          result = checker.check(args)
        expected = reproduce and regression_code == 0 and smoke_code == 0
        self.assertEqual(result["passed"], expected)
        self.assertEqual(result["selected_candidate"], "both" if expected else None)
        self.assertTrue(result["installed_collision_unchanged"])
        self.assertEqual("project_tests.log" in calls, reproduce)
        self.assertEqual("go2w_smoke.log" in calls, reproduce and regression_code == 0)
        self.assertEqual(json.loads((args.output_dir / "summary.json").read_text(encoding="utf-8"))["passed"], expected)


if __name__ == "__main__":
  unittest.main()
