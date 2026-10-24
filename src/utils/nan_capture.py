"""Capture the first invalid physics result before a reset can erase it.

Installed by the training entry point only when mjlab's NaN guard is enabled.
History remains compatible with mjlab's PHYSICS-state dumps; failure_* arrays
are the original device results, never values recomputed by CPU MuJoCo.
"""

from collections import deque
from contextlib import contextmanager
from datetime import datetime
from functools import wraps
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import shutil

import mujoco
import numpy as np
import torch
import warp as wp


CHECK_FIELDS = (
  "qpos", "qvel", "act", "ctrl", "qacc", "qacc_warmstart", "sensordata",
  "qacc_smooth", "qfrc_actuator", "qfrc_constraint",
)
INPUT_FIELDS = (
  "time", "qpos", "qvel", "act", "ctrl", "qacc_warmstart",
  "qfrc_applied", "xfrc_applied", "mocap_pos", "mocap_quat",
)
RESULT_FIELDS = tuple(dict.fromkeys(INPUT_FIELDS + CHECK_FIELDS + ("nefc", "actuator_force")))
CONTACT_FIELDS = ("geom", "pos", "dist", "frame", "includemargin", "dim", "efc_address")


def _tensor(data, name):
  # Empty Warp arrays (e.g. no mocap bodies or tendons) carry no values;
  # omit them before asking the bridge to create a Torch view.
  raw = getattr(data, "_struct", None)
  array = getattr(raw, name, None) if raw is not None else None
  if isinstance(array, wp.array) and array.size == 0:
    return None
  value = getattr(data, name, None)
  return None if value is None else value[:]


class PhysicsNanError(RuntimeError):
  """The original failure has been saved and rollout must stop."""


class PhysicsNanCapture:
  def __init__(self, sim):
    self.sim = sim
    cfg = sim.cfg.nan_guard
    self.enabled = cfg.enabled
    if cfg.buffer_size < 1 or cfg.max_envs_to_dump < 1:
      raise ValueError("NaN capture buffer_size and max_envs_to_dump must be positive")
    self.buffer = deque(maxlen=cfg.buffer_size)
    self.output_dir = Path(cfg.output_dir)
    self.max_envs_to_dump = cfg.max_envs_to_dump
    self.step_counter = 0
    self._dumped = False
    self.dump_path = None

  def _sync_warp(self):
    # Debug mode intentionally synchronizes: Torch clones must finish before
    # a Warp graph overwrites the shared state on another CUDA stream.
    wp.synchronize_device(self.sim.wp_device)

  def _sync_torch(self):
    if str(self.sim.device).startswith("cuda"):
      torch.cuda.synchronize(self.sim.device)

  def _snapshot(self, data, fields):
    return {name: value.detach().clone() for name in fields
            if (value := _tensor(data, name)) is not None}

  def capture(self, data, stage="step"):
    self._sync_warp()
    self.buffer.append({
      "step": self.step_counter, "stage": stage,
      "inputs": self._snapshot(data, INPUT_FIELDS),
    })
    self.step_counter += 1
    self._sync_torch()

  @contextmanager
  def watch(self, data, stage="step"):
    if not self.enabled:
      yield
      return
    self.capture(data, stage)
    self.check_and_dump(data, stage=f"before_{stage}")
    try:
      yield
    finally:
      self.check_and_dump(data, stage=f"after_{stage}")

  def check_and_dump(self, data, stage="after_step"):
    if not self.enabled or self._dumped:
      return False
    self._sync_warp()
    bad = {}
    for name in CHECK_FIELDS:
      value = _tensor(data, name)
      if value is not None and value.numel():
        mask = ~torch.isfinite(value).reshape(self.sim.num_envs, -1).all(dim=1)
        if mask.any().item():
          bad[name] = mask.nonzero(as_tuple=False).flatten().cpu().tolist()
    if not bad:
      return False
    env_ids = sorted({env_id for ids in bad.values() for env_id in ids})
    print(f"[NaNCapture] First invalid result: {stage}; fields/envs: {bad}", flush=True)
    self.dump_path = self._dump(data, env_ids, bad, stage)
    self._dumped = True
    print(f"[NaNCapture] Original failure saved: {self.dump_path}", flush=True)
    raise PhysicsNanError(
      f"NaN/Inf {stage}: {bad}. Rollout stopped before reset. Dump: {self.dump_path}"
    )

  def _dump(self, data, env_ids, bad, stage):
    self.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = self.output_dir / f"nan_dump_{stamp}.npz"
    model_path = self.output_dir / f"model_{stamp}.mjb"
    selected = env_ids[:self.max_envs_to_dump]
    model = self.sim.mj_model
    payload = {}
    phases = {}

    def save_fields(prefix, fields):
      for name, values in fields.items():
        payload[f"{prefix}_{name}"] = values[selected].detach().cpu().numpy().copy()

    def physics_state(fields):
      parts = [fields["qpos"], fields["qvel"]]
      if model.na:
        parts.append(fields["act"])
      return torch.cat(parts, dim=1)[selected].detach().cpu().numpy().copy()

    # Each pre-operation input is retained, including the original ctrl and
    # warmstart. The terminal state is appended separately, even for a full ring.
    for item in self.buffer:
      key = f"states_step_{item['step']:06d}"
      payload[key] = physics_state(item["inputs"])
      phases[key] = f"before_{item['stage']}"
      save_fields(f"inputs_step_{item['step']:06d}", item["inputs"])
    failure_key = f"states_step_{self.step_counter:06d}"
    result = self._snapshot(data, RESULT_FIELDS)
    payload[failure_key] = physics_state(result)
    phases[failure_key] = stage
    save_fields("failure", result)

    model_fields = sorted(self.sim.expanded_fields)
    versions = {}
    for package in ("mjlab", "mujoco", "mujoco-warp", "warp-lang", "torch"):
      try:
        versions[package] = version(package)
      except PackageNotFoundError:
        versions[package] = None
    metadata = {
      "format_version": 2,
      "versions": versions, "device": str(self.sim.device),
      "num_envs_total": self.sim.num_envs, "num_envs_dumped": len(selected),
      "nan_env_ids": env_ids, "dumped_env_ids": selected,
      "state_size": mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_PHYSICS),
      "state_spec": mujoco.mjtState.mjSTATE_PHYSICS.value,
      "buffer_size": len(phases), "detection_step": self.step_counter,
      "timestamp": stamp, "model_file": model_path.name,
      "failure_state_key": failure_key, "failure_stage": stage,
      "nonfinite_fields": bad, "frame_phases": phases,
      "capture_status": "core_saved", "capture_errors": {},
      "requested_model_fields": model_fields,
      "randomized_model_fields": [], "skipped_model_fields": {},
      "note": "states_step_* use mjSTATE_PHYSICS; the final state is the original "
              "invalid result. failure_* are recorded device arrays, not CPU replay. "
              "MJB is the base model; model_* store selected randomized fields. "
              "failure_geom_xpos/geom_xmat are original collision transforms. "
              "History counts step/forward calls, not only integration steps. "
              "This is a diagnostic capture, not a complete deterministic checkpoint.",
    }
    def write_archive():
      payload["metadata_json"] = np.array(json.dumps(metadata))
      # Retain compatibility with the original mjlab viewer.
      payload["_metadata"] = np.array(metadata, dtype=object)
      temporary = path.with_name(f".{path.stem}.partial.npz")
      np.savez_compressed(temporary, **payload)
      temporary.replace(path)

    def capture_optional(label, collect):
      # Diagnostic extraction must never replace the original physics error.
      # The core archive is already on disk if an optional array is unreadable.
      try:
        collect()
        return True
      except Exception as error:
        message = f"{type(error).__name__}: {error}"
        metadata["capture_errors"][label] = message
        print(f"[NaNCapture] Could not capture {label}: {message}; core saved at {path}", flush=True)
        return False

    # Commit essential states and raw failure results before touching model
    # fields or additional contact/constraint diagnostics. Atomic enrichment
    # below preserves this first archive if a later write fails.
    write_archive()
    capture_optional("model_file", lambda: mujoco.mj_saveModel(model, str(model_path), None))

    def capture_collision_inputs():
      # qpos may already have been integrated or invalidated. Retain the
      # actual device transforms used by collision, including GPU rounding.
      for name in ("geom_xpos", "geom_xmat", "overflow"):
        values = _tensor(data, name)
        if values is not None:
          save_fields("failure", {name: values})

    capture_optional("collision_inputs", capture_collision_inputs)

    def capture_contacts():
      # Contacts are globally packed; select by worldid rather than nconmax.
      contact = getattr(data, "contact", None)
      count = _tensor(data, "nacon")
      if contact is None or count is None:
        return
      n = int(count.flatten()[0].item())
      if n == 0:
        payload["failure_contact_worldid"] = np.empty(0, dtype=np.int32)
        return
      world = _tensor(contact, "worldid")[:n]
      mask = torch.zeros_like(world, dtype=torch.bool)
      for env_id in selected:
        mask |= world == env_id
      payload["failure_contact_worldid"] = world[mask].cpu().numpy().copy()
      for name in CONTACT_FIELDS:
        values = _tensor(contact, name)
        if values is not None:
          payload[f"failure_contact_{name}"] = values[:n][mask].cpu().numpy().copy()
      geoms = _tensor(contact, "geom")
      centers = _tensor(data, "geom_xpos")
      if geoms is not None and centers is not None:
        pairs = geoms[:n][mask].long()
        worlds = world[mask].long()
        payload["failure_contact_geom_xpos"] = centers[worlds[:, None], pairs].cpu().numpy().copy()

    capture_optional("contacts", capture_contacts)
    efc = getattr(data, "efc", None)
    if efc is not None:
      def capture_constraint(name):
        values = _tensor(efc, name)
        if values is not None:
          save_fields("failure_efc", {name: values})

      for name in ("J", "pos", "D", "aref", "vel", "force", "state"):
        capture_optional(f"efc.{name}", lambda name=name: capture_constraint(name))

    def capture_model_field(name):
      # COM recomputation registers tendon arrays even when ntendon == 0.
      # _tensor deliberately returns None for these empty Warp arrays.
      values = _tensor(self.sim.model, name)
      if values is None or values.numel() == 0:
        metadata["skipped_model_fields"][name] = "empty_or_unavailable"
        return
      if values.ndim == 0 or values.shape[0] not in (1, self.sim.num_envs):
        raise ValueError(f"Unexpected per-world model field shape: {tuple(values.shape)}")
      indices = selected if values.shape[0] == self.sim.num_envs else [0] * len(selected)
      payload[f"model_{name}"] = values[indices].cpu().numpy().copy()
      metadata["randomized_model_fields"].append(name)

    for name in model_fields:
      capture_optional(f"model.{name}", lambda name=name: capture_model_field(name))
    metadata["capture_status"] = "partial" if metadata["capture_errors"] else "complete"
    if not capture_optional("archive_enrichment", write_archive):
      metadata["capture_status"] = "core_saved"

    try:
      path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    except OSError as error:
      print(f"[NaNCapture] Could not write JSON sidecar: {error}; use {path}", flush=True)
    # Replace directory entries atomically. Copying directly to an old latest
    # symlink would follow it and overwrite a historical diagnostic file.
    for source, name in ((model_path, "model_latest.mjb"), (path, "nan_dump_latest.npz")):
      if not source.exists():
        continue
      try:
        temporary = self.output_dir / f".{stamp}_{name}"
        shutil.copyfile(source, temporary)
        temporary.replace(self.output_dir / name)
      except OSError as error:
        print(f"[NaNCapture] Could not update {name}: {error}; use {source}", flush=True)
    return path


def install_nan_capture(sim):
  """Replace only this simulation's guard; leave the installed mjlab untouched."""
  if not sim.cfg.nan_guard.enabled or isinstance(sim.nan_guard, PhysicsNanCapture):
    return sim.nan_guard
  guard = PhysicsNanCapture(sim)
  sim.nan_guard = guard
  original_forward = sim.forward

  @wraps(original_forward)
  def guarded_forward(*args, **kwargs):
    with guard.watch(sim.data, stage="forward"):
      return original_forward(*args, **kwargs)

  sim.forward = guarded_forward
  return guard
