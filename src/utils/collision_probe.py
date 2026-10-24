"""Process-local collision experiments for the pinned diagnostic tools.

These contexts never change installed files or training defaults. FMA changes
are applied to the actual dynamically built CCD kernel modules before launch;
changing only collision_gjk's module options would not affect those kernels.
"""

from contextlib import contextmanager
from functools import wraps
from importlib.metadata import version

from mujoco_warp._src import collision_convex
import warp as wp


def _check_backend():
  if version("mujoco-warp") != "3.16.0" or version("warp-lang") != "1.18.0":
    raise RuntimeError("Collision experiments require mujoco-warp 3.16.0 and warp-lang 1.18.0")


@contextmanager
def temporary_epa_horizon(capacity):
  """Override the rigid CCD allocation constant only inside this context."""
  original = collision_convex.MJ_MAX_EPAHORIZON
  if capacity is not None:
    _check_backend()
    if capacity < 1:
      raise ValueError("EPA horizon capacity must be positive")
    collision_convex.MJ_MAX_EPAHORIZON = capacity
  try:
    yield collision_convex.MJ_MAX_EPAHORIZON
  finally:
    collision_convex.MJ_MAX_EPAHORIZON = original


@contextmanager
def collision_fma_mode(mode="default"):
  """Override only rigid CCD kernels, audit actual options, then restore them.

  For single-threaded replay without CUDA graphs. This is a diagnostic control,
  not evidence that FMA or a compiler is defective.
  """
  if mode not in ("default", "on", "off"):
    raise ValueError("Collision FMA mode must be default, on or off")
  audit = {}
  if mode == "default":
    yield audit
    return
  _check_backend()
  originals = {}
  modules = {}

  def wrap_builder(builder):
    @wraps(builder)
    def build(*args, **kwargs):
      kernel = builder(*args, **kwargs)
      module = kernel.module
      if module not in modules:
        modules[module] = wp.get_module_options(module)["fuse_fp"]
        wp.set_module_options({"fuse_fp": mode == "on"}, module=module)
      # All geom specializations share a kernel.key; the unique module name
      # keeps cylinder/box and other pairs from overwriting each other's audit.
      audit[module.name] = {"kernel": kernel.key, "module": module.name,
                            "geom_types": [int(args[0]), int(args[1])],
                            "fuse_fp": wp.get_module_options(module)["fuse_fp"]}
      return kernel
    return build

  try:
    for name in ("ccd_kernel_builder", "ccd_hfield_kernel_builder"):
      originals[name] = getattr(collision_convex, name)
      setattr(collision_convex, name, wrap_builder(originals[name]))
    yield audit
  finally:
    try:
      if modules:
        wp.synchronize()
    finally:
      for name, builder in originals.items():
        setattr(collision_convex, name, builder)
      for module, fuse_fp in modules.items():
        wp.set_module_options({"fuse_fp": fuse_fp}, module=module)
