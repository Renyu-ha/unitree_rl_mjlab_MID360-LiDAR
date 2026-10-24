"""Identify the fixed backend installed through the project dependencies."""

import hashlib
from importlib.metadata import version
from pathlib import Path

import mujoco_warp as mjw
from mujoco_warp._src import collision_convex


def physics_backend_info():
  """Reject stale installs and return the actual engine used by this process."""
  source = Path(mjw.__file__).resolve()
  capacity = collision_convex.MJ_MAX_EPAHORIZON
  if version("mujoco-warp") != "3.16.0.post1" or capacity != 64:
    raise RuntimeError(
      "This project requires its pinned MuJoCo-Warp backport with EPA horizon 64. "
      "Remove old collision-test PYTHONPATH overrides and run pip install -e ."
    )
  return {
    "engine_source": str(source),
    "collision_source_sha256": hashlib.sha256(
      (source.parent / "_src/collision_gjk.py").read_bytes().replace(b"\r\n", b"\n")
    ).hexdigest(),
    "mujoco_warp": version("mujoco-warp"),
    "warp_lang": version("warp-lang"),
    "epa_horizon": capacity,
  }
