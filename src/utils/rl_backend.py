"""Require the project's single RSL-RL runtime and record its actual sources."""

import hashlib
import importlib
from importlib.metadata import PackageNotFoundError, distribution, distributions
from pathlib import Path
import sys


RSL_RL_VERSION = "5.5.1"


def rl_backend_info() -> dict:
  try:
    installed = distribution("rsl-rl-lib")
  except PackageNotFoundError as error:
    raise RuntimeError(f"rsl-rl-lib=={RSL_RL_VERSION} is missing from {sys.executable}. "
                       "Run: python -m pip install -e .") from error
  if installed.version != RSL_RL_VERSION:
    raise RuntimeError(
      f"This project requires rsl-rl-lib=={RSL_RL_VERSION}, found {installed.version}. "
      "Use this Python interpreter to run: python -m pip install -e ."
    )
  candidates = {(item.version, str(Path(item.locate_file("rsl_rl")).resolve()))
                for item in distributions(name="rsl-rl-lib")}
  if len(candidates) != 1:
    raise RuntimeError(f"Multiple RSL-RL installations are visible to this Python: {sorted(candidates)}")
  root = Path(installed.locate_file("rsl_rl")).resolve()
  package = importlib.import_module("rsl_rl")
  if Path(package.__file__).resolve().parent != root:
    raise RuntimeError(f"RSL-RL import is shadowed: {package.__file__}; expected {root}")
  critical = {
    "ppo": "rsl_rl.algorithms.ppo",
    "symmetry": "rsl_rl.extensions.symmetry",
    "storage": "rsl_rl.storage.rollout_storage",
    "runner": "rsl_rl.runners.on_policy_runner",
    "model": "rsl_rl.models.mlp_model",
    "normalization": "rsl_rl.modules.normalization",
    "distribution": "rsl_rl.modules.distribution",
  }
  modules = {name: importlib.import_module(module) for name, module in critical.items()}
  for name, module in list(sys.modules.items()):
    if not name.startswith("rsl_rl."):
      continue
    source = getattr(module, "__file__", None)
    if source and not Path(source).resolve().is_relative_to(root):
      raise RuntimeError(f"Mixed RSL-RL sources: {name} loaded from {source}; expected {root}")
  hashes = {name: hashlib.sha256(Path(module.__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for name, module in modules.items()}
  return {"version": installed.version, "python": sys.executable,
          "package_source": str(Path(package.__file__).resolve()), "source_sha256": hashes}
