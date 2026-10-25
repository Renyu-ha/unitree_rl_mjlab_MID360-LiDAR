"""Reject version drift, duplicate packages and mixed RSL-RL import sources."""

from copy import deepcopy
import importlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import tomllib
import unittest
from unittest.mock import patch

import torch

from src.tasks.velocity.rl.go2w_runner import Go2WOnPolicyRunner
from src.utils.rl_backend import RSL_RL_VERSION, rl_backend_info


class RlBackendTests(unittest.TestCase):
  def test_runtime_matches_the_single_project_pin(self):
    project = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    pins = [item for item in project["project"]["dependencies"] if item.startswith("rsl-rl-lib")]
    self.assertEqual(pins, [f"rsl-rl-lib=={RSL_RL_VERSION}"])
    info = rl_backend_info()
    self.assertEqual(info["version"], RSL_RL_VERSION)
    self.assertEqual(len(info["source_sha256"]), 7)
    self.assertTrue(all(len(value) == 64 for value in info["source_sha256"].values()))

  def test_different_version_is_rejected(self):
    with patch("src.utils.rl_backend.distribution", return_value=SimpleNamespace(version="5.0.1")):
      with self.assertRaisesRegex(RuntimeError, "requires rsl-rl-lib==5.5.1"):
        rl_backend_info()

  def test_duplicate_distributions_are_rejected(self):
    entries = [SimpleNamespace(version=version, locate_file=lambda _: Path("/duplicate/rsl_rl"))
               for version in ("5.5.1", "5.0.1")]
    with patch("src.utils.rl_backend.distributions", return_value=entries):
      with self.assertRaisesRegex(RuntimeError, "Multiple RSL-RL"):
        rl_backend_info()

  def test_shadow_package_and_mixed_submodule_are_rejected(self):
    for module, pattern in (("rsl_rl", "shadowed"), ("rsl_rl.extensions.symmetry", "Mixed RSL-RL")):
      with self.subTest(module=module), patch.object(importlib.import_module(module), "__file__", "/other/rsl_rl/file.py"):
        with self.assertRaisesRegex(RuntimeError, pattern):
          rl_backend_info()

  def test_checkpoint_version_and_source_drift_are_rejected_before_loading_weights(self):
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "model.pt"
      for field in ("version", "source_sha256"):
        info = deepcopy(rl_backend_info())
        info[field] = "different" if field == "version" else {}
        torch.save({"infos": {"rsl_rl_backend": info}}, path)
        runner = Go2WOnPolicyRunner.__new__(Go2WOnPolicyRunner)
        with self.subTest(field=field), self.assertRaisesRegex(RuntimeError, "Checkpoint RSL-RL"):
          runner.load(str(path), map_location="cpu")


if __name__ == "__main__":
  unittest.main()
