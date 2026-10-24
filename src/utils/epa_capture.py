"""Opt-in capture of the first EPA overflow, before a reset erases geometry.

This records the actual float32 collision transforms after step/forward, not
transforms recomputed from the already integrated qpos. Checking overflow bits
synchronizes the device; enable it for a diagnostic run, not throughput tests.
"""

from datetime import datetime
from functools import wraps
from importlib.metadata import version
import json
from pathlib import Path
import shutil

import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp

from src.utils.nan_capture import _tensor


class EpaCaptureError(RuntimeError):
  """A collision overflow was recorded; stop before another physics operation."""


def _write_archive(path, payload, metadata):
  payload["metadata_json"] = np.array(json.dumps(metadata))
  temporary = path.with_name(f".{path.stem}.partial.npz")
  np.savez_compressed(temporary, **payload)
  temporary.replace(path)


def save_epa_snapshot(sim, env_ids, output_dir, operation, task=None):
  """Save rigid collision inputs/results; optional extras cannot discard the core."""
  from mujoco_warp._src import collision_convex

  output_dir = Path(output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
  path = output_dir / f"epa_dump_{stamp}.npz"
  model_path = output_dir / f"epa_model_{stamp}.mjb"
  # One world is sufficient to isolate the first failure and limits dump size.
  selected = env_ids[:1]
  indices = torch.tensor(selected, device=sim.device, dtype=torch.long)
  payload = {}
  for name in ("geom_xpos", "geom_xmat", "overflow"):
    payload[name] = _tensor(sim.data, name)[indices].detach().cpu().numpy().copy()
  metadata = {
    "format": "epa_collision_snapshot_v1", "timestamp": stamp,
    "task": task, "operation": operation, "device": str(sim.device),
    "versions": {name: version(name) for name in
                 ("mjlab", "mujoco", "mujoco-warp", "warp-lang")},
    "num_envs_total": sim.num_envs, "overflow_env_ids": env_ids,
    "dumped_env_ids": selected, "epa_horizon": collision_convex.MJ_MAX_EPAHORIZON,
    "model_file": model_path.name, "capture_status": "core_saved",
    "geom_names": [sim.mj_model.geom(i).name for i in range(sim.mj_model.ngeom)],
    "capture_errors": {}, "skipped_model_fields": [],
    "nconmax_per_world": max(1, sim.wp_data.naconmax // sim.num_envs),
    "note": "geom_xpos/geom_xmat are original device collision inputs, not CPU FK. "
            "result_* are post-operation context, not pre-step replay inputs. "
            "This is a rigid collision snapshot, not a deterministic training checkpoint.",
  }
  _write_archive(path, payload, metadata)

  def optional(label, collect):
    try:
      collect()
    except Exception as error:
      metadata["capture_errors"][label] = f"{type(error).__name__}: {error}"
      print(f"[EPACapture] Could not save {label}: {error}; core: {path}", flush=True)

  optional("model_file", lambda: mujoco.mj_saveModel(sim.mj_model, str(model_path), None))

  def model_fields():
    fields = set(sim.expanded_fields) | set(sim.per_world_default_fields)
    for name in sorted(fields):
      raw = getattr(sim.wp_model, name)
      if raw.size == 0:
        metadata["skipped_model_fields"].append(name)
        continue
      values = wp.to_torch(raw)
      payload[f"model_{name}"] = values[indices % values.shape[0]].cpu().numpy().copy()

  optional("model_fields", model_fields)

  def runtime_options():
    for name in ("ccd_tolerance",):
      raw = getattr(sim.wp_model.opt, name)
      values = wp.to_torch(raw)
      payload[f"option_{name}"] = values[indices % values.shape[0]].cpu().numpy().copy()
    metadata["broadphase"] = int(sim.wp_model.opt.broadphase)
    metadata["broadphase_filter"] = int(sim.wp_model.opt.broadphase_filter)
    metadata["ccd_iterations"] = sim.wp_model.opt.ccd_iterations

  optional("runtime_options", runtime_options)

  def context():
    for name in ("qpos", "qvel", "ctrl", "time"):
      value = _tensor(sim.data, name)
      if value is not None and value.numel():
        payload[f"result_{name}"] = value[indices].cpu().numpy().copy()

  optional("result_context", context)

  def contacts():
    count = min(int(sim.wp_data.nacon.numpy()[0]), sim.wp_data.naconmax)
    contact = sim.wp_data.contact
    worlds = wp.to_torch(contact.worldid)[:count]
    mask = worlds == selected[0]
    # The archive uses local world indices for direct comparison with replay.
    payload["recorded_contact_worldid"] = np.zeros(int(mask.sum().item()), dtype=np.int32)
    for name in ("geom", "pos", "dist", "frame"):
      payload[f"recorded_contact_{name}"] = wp.to_torch(getattr(contact, name))[:count][mask].cpu().numpy().copy()

  optional("contacts", contacts)
  metadata["capture_status"] = "partial" if metadata["capture_errors"] else "complete"
  try:
    _write_archive(path, payload, metadata)
  except Exception as error:
    print(f"[EPACapture] Could not enrich archive: {error}; core: {path}", flush=True)
  try:
    temporary = output_dir / ".epa_dump_latest.partial.npz"
    shutil.copyfile(path, temporary)
    temporary.replace(output_dir / "epa_dump_latest.npz")
  except Exception as error:
    print(f"[EPACapture] Could not update latest: {error}; use {path}", flush=True)
  return path


class EpaCapture:
  def __init__(self, sim, output_dir, task=None):
    self.sim = sim
    self.output_dir = Path(output_dir)
    self.task = task
    self.dump_path = None

  def check(self, operation, stop=True):
    if self.dump_path is not None:
      return
    wp.synchronize_device(self.sim.wp_device)
    flags = self.sim.wp_data.overflow.numpy()
    ids = np.flatnonzero(flags & int(mjw.OverflowType.EPA_HORIZON)).tolist()
    if not ids:
      return
    self.dump_path = save_epa_snapshot(self.sim, ids, self.output_dir, operation, self.task)
    print(f"[EPACapture] Original collision snapshot saved: {self.dump_path}", flush=True)
    if stop:
      raise EpaCaptureError(f"EPA horizon overflow in environments {ids}; dump: {self.dump_path}")


def install_epa_capture(sim, output_dir, task=None):
  """Watch step/forward only on this simulation; keep engine warnings enabled."""
  if hasattr(sim, "_epa_capture"):
    return sim._epa_capture
  if not sim.wp_model.opt.warn_overflow & int(mjw.OverflowType.EPA_HORIZON):
    raise ValueError("EPA warnings must remain enabled during diagnostic capture")
  wp.synchronize_device(sim.wp_device)
  # Constructor graph-capture calls precede installation. Start a fresh watch,
  # preserving every other diagnostic bit. This changes no physics input.
  wp.to_torch(sim.wp_data.overflow).bitwise_and_(~int(mjw.OverflowType.EPA_HORIZON))
  if str(sim.device).startswith("cuda"):
    torch.cuda.synchronize(sim.device)
  guard = EpaCapture(sim, output_dir, task)
  sim._epa_capture = guard

  def wrap(name, original):
    @wraps(original)
    def watched(*args, **kwargs):
      try:
        result = original(*args, **kwargs)
      except BaseException:
        # A simultaneous NaN or other physics exception remains the primary error.
        try:
          guard.check(name, stop=False)
        except Exception as error:
          print(f"[EPACapture] Additional capture failed: {error}", flush=True)
        raise
      guard.check(name)
      return result
    return watched

  for name in ("step", "forward"):
    setattr(sim, name, wrap(name, getattr(sim, name)))
  return guard
