"""Go2W deployment export plus SGS checkpoint persistence."""

import os
import warnings

from .go2w_runner import Go2WOnPolicyRunner


class Go2WSGSRunner(Go2WOnPolicyRunner):
  def __init__(self, *args, **kwargs):
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
      raise ValueError("Go2W SGS currently requires one training process; use a single GPU, e.g. --gpu-ids '[0]'. "
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
    command = self.env.unwrapped.command_manager.get_term("twist")
    warm_start = command.cfg.warm_start and not command.cfg.evaluation and load_cfg is None
    effective_load = {"actor": True, "critic": True} if warm_start else load_cfg
    infos = super().load(path, effective_load, strict, map_location)
    if warm_start:
      self.current_learning_iteration = 0
      self.env.unwrapped.common_step_counter = 0
      if hasattr(self, "alg"):
        self.alg.optimizer.state.clear()
        initial_lr = self.cfg["algorithm"]["learning_rate"]
        self.alg.learning_rate = initial_lr
        for group in self.alg.optimizer.param_groups:
          group["lr"] = initial_lr
      command.sampler.history.zero_()
      command.sampler.counts.zero_()
      command.sampler.parent.history.zero_()
      command.sampler.parent.counts.zero_()
      command.recorded = command.skipped = 0
      command._censored.fill_(True)
      warnings.warn("Warm start: loaded actor/critic weights; optimizer, iteration, curricula and SGS history start fresh.",
        stacklevel=2)
      return infos
    # Play uses a fresh uniform bank. Actor-only warm starts also keep a fresh
    # sampler rather than mixing success histories from a different policy.
    if not command.cfg.evaluation and load_cfg is None:
      if infos and "sgs_state" in infos:
        command.load_sgs_state_dict(infos["sgs_state"])
      else:
        warnings.warn("Checkpoint has no SGS history; starting with an unvisited bank", stacklevel=2)
    return infos
