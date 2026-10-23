"""Check exported normalization/action semantics and reference-scene validation."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mujoco
import numpy as np
import onnx
import torch
from onnx import TensorProto, helper, numpy_helper

from src.utils.border_play import verify_reference_geometry
from src.utils.onnx_policy import OnnxPolicy


class OnnxPlayTests(unittest.TestCase):
  def make_fixture(self, path, *, scale=0.5, clip_actions=None):
    # Embedded normalization is (obs - 1) / 2. It must run exactly once;
    # action scale/offset/clipping must remain outside the actor.
    graph = helper.make_graph([
      helper.make_node("Sub", ["obs", "mean"], ["centered"]),
      helper.make_node("Div", ["centered", "std"], ["actions"]),
    ], "actor", [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, 16])],
      [helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, 16])], [
        numpy_helper.from_array(np.ones((1, 16), np.float32), "mean"),
        numpy_helper.from_array(np.full((1, 16), 2, np.float32), "std"),
      ])
    model = helper.make_model(graph)
    joint_names = [f"joint_{i}" for i in range(16)]
    base = dict(joint_names=joint_names, command_names=["twist"], joint_stiffness=[20]*16,
      joint_damping=[1]*16, default_joint_pos=[0]*16)
    metadata = dict(observation_names="joint_vel", actor_history_length="1",
      observation_layout="term_major_history", control_dt="0.02", action_clip=str(clip_actions),
      action_joint_names=",".join(joint_names), action_types=",".join(["position"]*12+["velocity"]*4),
      action_scale=",".join(map(str, [scale]*12+[20]*4)), action_offset=",".join(["0"]*16))
    metadata.update({key: ",".join(map(str, value)) for key, value in base.items()})
    helper.set_model_props(model, metadata)
    onnx.save(model, path)
    leg = SimpleNamespace(target_names=joint_names[:12], action_dim=12, scale=.5, offset=0.)
    wheel = SimpleNamespace(target_names=joint_names[12:], action_dim=4, scale=20., offset=0.)
    env = SimpleNamespace(num_envs=1, device="cpu", step_dt=.02,
      cfg=SimpleNamespace(observations={"actor": SimpleNamespace(history_length=1, terms={"joint_vel": None})}),
      action_manager=SimpleNamespace(total_action_dim=16, get_term=lambda name: leg if name=="joint_pos" else wheel),
      get_observations=lambda: {"actor": torch.zeros(1, 16)})
    env.unwrapped = env
    return env, base

  def test_onnx_normalization_runs_once_without_action_transforms(self):
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "policy.onnx"
      env, base = self.make_fixture(path)
      with patch("mjlab.rl.exporter_utils.get_base_metadata", return_value=base):
        policy = OnnxPolicy(path, env, clip_actions=None)
      torch.testing.assert_close(policy({"actor": torch.full((1, 16), 9.)}), torch.full((1, 16), 4.))
      with self.assertRaisesRegex(ValueError, "NaN/Inf"):
        policy({"actor": torch.full((1, 16), float("nan"))})

  def test_rejects_incompatible_action_metadata(self):
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "policy.onnx"
      env, base = self.make_fixture(path, scale=1.)
      with patch("mjlab.rl.exporter_utils.get_base_metadata", return_value=base):
        with self.assertRaisesRegex(ValueError, "action_scale"):
          OnnxPolicy(path, env, clip_actions=None)

  def test_clipping_metadata_distinguishes_unclipped_and_legacy_policies(self):
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "policy.onnx"
      for exported_clip, configured_clip in ((None, 1.), (1., None)):
        with self.subTest(exported=exported_clip, configured=configured_clip):
          env, base = self.make_fixture(path, clip_actions=exported_clip)
          with patch("mjlab.rl.exporter_utils.get_base_metadata", return_value=base):
            with self.assertRaisesRegex(ValueError, "action_clip"):
              OnnxPolicy(path, env, clip_actions=configured_clip)
            policy = OnnxPolicy(path, env, clip_actions=exported_clip)
          torch.testing.assert_close(policy({"actor": torch.full((1, 16), 9.)}), torch.full((1, 16), 4.))

  def test_reference_check_rejects_changed_floor_height(self):
    model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom name="floor" type="box" size="4 4 .5" pos="0 0 -.5"/></worldbody></mujoco>')
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "scene.mjb"
      mujoco.mj_saveModel(model, str(path))
      verify_reference_geometry(model, path)
      model.geom_pos[0, 2] -= .01
      with self.assertRaisesRegex(ValueError, "geom_pos"):
        verify_reference_geometry(model, path)

  def test_reference_check_handles_removed_unused_meshes_but_rejects_changed_vertices(self):
    mesh = '<mesh name="{name}" vertex="0 0 0 1 0 0 0 1 0 0 0 1"/>'
    template = '<mujoco><asset>{assets}</asset><worldbody><geom name="wheel" type="mesh" mesh="wheel"/></worldbody></mujoco>'
    current = mujoco.MjModel.from_xml_string(template.format(assets=mesh.format(name="wheel")))
    saved = mujoco.MjModel.from_xml_string(template.format(assets=mesh.format(name="unused") + mesh.format(name="wheel")))
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "scene.mjb"
      mujoco.mj_saveModel(saved, str(path))
      verify_reference_geometry(current, path)
      current.mesh_vert[0, 0] += .01
      with self.assertRaisesRegex(ValueError, "mesh_vert"):
        verify_reference_geometry(current, path)


if __name__ == "__main__":
  unittest.main()
