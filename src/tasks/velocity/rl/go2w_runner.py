"""Export mixed-action metadata instead of labelling wheel speeds as positions."""

import os
import json

import torch
from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
from mjlab.rl.runner import MjlabOnPolicyRunner


class Go2WOnPolicyRunner(MjlabOnPolicyRunner):
  def save(self, path: str, infos=None):
    super().save(path, infos)
    directory = os.path.dirname(path)
    self.export_deploy_policy(directory)

  def export_deploy_policy(self, directory: str):
    """Export the actor and its G1-style Go2W deployment configuration together."""
    from src.utils.go2w_deploy import deployment_config, write_deploy_yaml

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
      "policy_kind": "trained_actor",
    })
    metadata["go2w_deploy_contract"] = json.dumps(deployment_config(env, metadata), separators=(",", ":"))
    path = os.path.join(directory, "policy.onnx")
    attach_metadata_to_onnx(path, metadata)
    write_deploy_yaml(path)
