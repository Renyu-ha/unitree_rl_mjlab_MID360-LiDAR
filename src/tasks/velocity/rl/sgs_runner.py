"""Go2W deployment export plus SGS checkpoint persistence."""

import os
import warnings

from .go2w_runner import Go2WOnPolicyRunner


class Go2WSGSRunner(Go2WOnPolicyRunner):
  def __init__(self, *args, **kwargs):
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
      raise ValueError("Go2W SGS currently requires one training process; use --gpu-ids 0. "
        "Per-rank sampler checkpointing is not implemented.")
    super().__init__(*args, **kwargs)

  def save(self, path, infos=None):
    command = self.env.unwrapped.command_manager.get_term("twist")
    if command.cfg.evaluation:
      raise ValueError("Evaluation must not overwrite SGS training checkpoints")
    cfg = getattr(self, "cfg", {})
    model_cfg = {name: {key: cfg.get(name, {})[key]
      for key in ("activation", "obs_normalization") if key in cfg.get(name, {})}
      for name in ("actor", "critic")}
    infos = {**(infos or {}), "sgs_state": command.sgs_state_dict(), "sgs_model_cfg": model_cfg}
    super().save(path, infos)

  def load(self, path, load_cfg=None, strict=True, map_location=None):
    infos = super().load(path, load_cfg, strict, map_location)
    command = self.env.unwrapped.command_manager.get_term("twist")
    # Play uses a fresh uniform bank. Actor-only warm starts also keep a fresh
    # sampler rather than mixing success histories from a different policy.
    if not command.cfg.evaluation and load_cfg is None:
      if infos and "sgs_state" in infos:
        command.load_sgs_state_dict(infos["sgs_state"])
      else:
        warnings.warn("Checkpoint has no SGS history; starting with an unvisited bank", stacklevel=2)
    return infos
