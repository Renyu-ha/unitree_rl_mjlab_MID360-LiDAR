"""Export mixed-action metadata instead of labelling wheel speeds as positions."""

import json
import os
import warnings

import torch
from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
from mjlab.rl.runner import MjlabOnPolicyRunner

from src.utils.rl_backend import rl_backend_info


class Go2WOnPolicyRunner(MjlabOnPolicyRunner):
  def __init__(self, env, train_cfg: dict, log_dir=None, device="cpu"):
    self.rl_backend = rl_backend_info()
    print(f"[INFO] RSL-RL {self.rl_backend['version']}: {self.rl_backend['package_source']}")
    super().__init__(env, train_cfg, log_dir, device)

  def save(self, path: str, infos=None):
    infos = {**(infos or {}), "rsl_rl_backend": self.rl_backend}
    super().save(path, infos)
    directory = os.path.dirname(path)
    self.export_policy_to_onnx(directory, "policy.onnx")
    env = self.env.unwrapped
    metadata = get_base_metadata(env, "local")
    action_names, action_types, scales, offsets = [], [], [], []
    for name, kind in (("joint_pos", "position"), ("wheel_vel", "velocity")):
      term = env.action_manager.get_term(name)
      action_names.extend(term.target_names)
      action_types.extend([kind] * term.action_dim)
      for value, output in ((term.scale, scales), (term.offset, offsets)):
        output.extend(
          value[0].detach().cpu().tolist()
          if isinstance(value, torch.Tensor) else [value] * term.action_dim
        )
    metadata.update({
      "action_joint_names": action_names,
      "action_types": action_types,
      "action_scale": scales,
      "action_offset": offsets,
      "action_clip": self.cfg["clip_actions"],
      "actor_history_length": env.cfg.observations["actor"].history_length,
      "observation_layout": "term_major_history",
      "control_dt": env.step_dt,
      "rsl_rl_version": self.rl_backend["version"],
      "rsl_rl_source_sha256": json.dumps(self.rl_backend["source_sha256"], sort_keys=True),
    })
    attach_metadata_to_onnx(os.path.join(directory, "policy.onnx"), metadata)

  def load(self, path: str, load_cfg=None, strict=True, map_location=None):
    current = rl_backend_info()
    # Validate metadata on CPU before the native loader applies any model state.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    recorded = (checkpoint.get("infos") or {}).get("rsl_rl_backend")
    del checkpoint
    if recorded is None:
      warnings.warn("Checkpoint has no RSL-RL version record; only the current runtime can be verified.",
                    RuntimeWarning, stacklevel=2)
    elif (recorded.get("version") != current["version"]
          or recorded.get("source_sha256") != current["source_sha256"]):
      raise RuntimeError("Checkpoint RSL-RL version or sources differ from the pinned runtime")
    return super().load(path, load_cfg=load_cfg, strict=strict, map_location=map_location)
