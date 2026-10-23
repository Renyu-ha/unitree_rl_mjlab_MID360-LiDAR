"""Run an exported Go2W actor against mjlab's original observation managers."""

from pathlib import Path

import numpy as np
import torch


class OnnxPolicy:
  def __init__(self, path: str | Path, env, *, clip_actions: float | None):
    import onnx
    from onnx.reference import ReferenceEvaluator

    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    self.metadata = {entry.key: entry.value for entry in model.metadata_props}
    self.env = env.unwrapped
    inputs, outputs = model.graph.input, model.graph.output
    if len(inputs) != 1 or len(outputs) != 1:
      raise ValueError("ONNX play supports one observation input and one action output")
    self.input_name, self.output_name = inputs[0].name, outputs[0].name
    self.input_shape = tuple(d.dim_value for d in inputs[0].type.tensor_type.shape.dim)
    self.output_shape = tuple(d.dim_value for d in outputs[0].type.tensor_type.shape.dim)
    observation = env.get_observations()["actor"]
    if self.input_shape != tuple(observation.shape):
      raise ValueError(f"ONNX input {self.input_shape} != actor observations {tuple(observation.shape)}")
    expected_output = (env.num_envs, self.env.action_manager.total_action_dim)
    if self.output_shape != expected_output:
      raise ValueError(f"ONNX output {self.output_shape} != environment actions {expected_output}")
    self._validate_metadata(clip_actions)
    # This CPU evaluator is bundled with ONNX; it preserves the exported
    # normalization graph and does not require a second inference dependency.
    self.evaluator = ReferenceEvaluator(model)

  def _validate_metadata(self, clip_actions):
    from mjlab.rl.exporter_utils import get_base_metadata

    def check_text(key, expected):
      if self.metadata.get(key) != expected:
        raise ValueError(f"ONNX {key} does not match this task: {self.metadata.get(key)!r} != {expected!r}")

    def check_numbers(key, expected):
      values = np.fromstring(self.metadata.get(key, ""), sep=",")
      target = np.asarray(expected).reshape(-1)
      if values.shape != target.shape or not np.allclose(values, target, atol=5e-4):
        raise ValueError(f"ONNX {key} does not match this task")

    actor = self.env.cfg.observations["actor"]
    check_text("observation_names", ",".join(actor.terms))
    check_text("actor_history_length", str(actor.history_length))
    check_text("observation_layout", "term_major_history")
    check_numbers("control_dt", [self.env.step_dt])
    check_numbers("action_clip", [clip_actions])
    base = get_base_metadata(self.env, "local")
    check_text("joint_names", ",".join(base["joint_names"]))
    check_text("command_names", ",".join(base["command_names"]))
    check_numbers("joint_stiffness", base["joint_stiffness"])
    check_numbers("joint_damping", base["joint_damping"])
    check_numbers("default_joint_pos", base["default_joint_pos"])
    names, types, scales, offsets = [], [], [], []
    for name, kind in (("joint_pos", "position"), ("wheel_vel", "velocity")):
      term = self.env.action_manager.get_term(name)
      names.extend(term.target_names)
      types.extend([kind] * term.action_dim)
      for value, result in ((term.scale, scales), (term.offset, offsets)):
        result.extend(value[0].detach().cpu().tolist() if isinstance(value, torch.Tensor)
          else [value] * term.action_dim)
    check_text("action_joint_names", ",".join(names))
    check_text("action_types", ",".join(types))
    check_numbers("action_scale", scales)
    check_numbers("action_offset", offsets)

  def __call__(self, obs) -> torch.Tensor:
    array = obs["actor"].detach().cpu().numpy().astype(np.float32, copy=False)
    if array.shape != self.input_shape or not np.isfinite(array).all():
      raise ValueError("ONNX actor input has an unexpected shape or NaN/Inf")
    action = self.evaluator.run([self.output_name], {self.input_name: array})[0]
    if action.shape != self.output_shape or not np.isfinite(action).all():
      raise ValueError("ONNX policy produced an unexpected shape or NaN/Inf")
    # Scale, offset and clipping belong to the existing environment/wrapper.
    return torch.as_tensor(action, device=self.env.device)
