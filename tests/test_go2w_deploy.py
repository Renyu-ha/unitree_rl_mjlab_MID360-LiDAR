"""Deployment interfaces compared with the real training environment."""
import copy
from dataclasses import asdict
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mujoco
import numpy as np
import onnx
import torch
import yaml

import src.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from scripts.prepare_go2w_sim import make_scene
from src.tasks.velocity.mdp.observations import bridged_mid360_height_scan_memory
from src.utils.go2w_deploy import OBSERVATIONS, validate_contract, write_deploy_yaml

ROOT = Path(__file__).resolve().parents[1]


class Go2WDeployTests(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cfg = load_env_cfg("Unitree-Go2W-Perceptive-Rough-v2", play=True)
    cfg.scene.num_envs = 1
    cfg.scene.terrain.terrain_generator.num_rows = 2
    cfg.scene.terrain.terrain_generator.num_cols = 2
    cls.env = ManagerBasedRlEnv(cfg, device="cpu")
    agent = load_rl_cfg("Unitree-Go2W-Perceptive-Rough-v2")
    wrapped = RslRlVecEnvWrapper(cls.env, clip_actions=agent.clip_actions)
    cls.runner = load_runner_cls("Unitree-Go2W-Perceptive-Rough-v2")(wrapped, asdict(agent), device="cpu")
    cls.wrapped = wrapped
    cls.temporary = tempfile.TemporaryDirectory()
    cls.path = Path(cls.temporary.name) / "policy.onnx"
    cls.runner.export_deploy_policy(cls.temporary.name)
    cls.model = onnx.load(cls.path)
    cls.metadata = {x.key: x.value for x in cls.model.metadata_props}
    cls.config = yaml.safe_load((cls.path.parent / "params/deploy.yaml").read_text())

  @classmethod
  def tearDownClass(cls):
    cls.env.close()
    cls.temporary.cleanup()

  def test_export_matches_resolved_training_observations_and_gains(self):
    cfg = self.config
    self.assertEqual(cfg["observation_dim"], self.env.observation_manager.group_obs_dim["actor"][0])
    self.assertEqual(cfg["observation_dim"], 720)
    self.assertEqual(list(cfg["observations"]), [OBSERVATIONS[x][0] for x in self.env.observation_manager.active_terms["actor"]])
    for term in cfg["observations"].values():
      self.assertEqual(term["history_length"], 3)
    wheel_scale = self.env.observation_manager.get_term_cfg("actor", "wheel_vel").scale
    np.testing.assert_allclose(cfg["observations"]["wheel_vel"]["scale"], np.asarray(wheel_scale).reshape(-1))
    self.assertEqual(cfg["observations"]["joint_pos_rel"]["params"]["asset_cfg"]["joint_ids"], list(range(12)))
    self.assertEqual(cfg["stiffness"][12:], [0]*4)
    self.assertEqual(cfg["damping"][12:], [2]*4)
    self.assertEqual(cfg["actions"]["JointPositionAction"]["scale"], [0.5]*12)
    self.assertEqual(cfg["actions"]["JointVelocityAction"]["scale"], [20]*4)
    self.assertEqual(cfg["height_scan"]["x_offset_m"], 0.35)
    self.assertEqual(cfg["height_scan"]["memory_max_age_s"], 0.0)
    self.assertTrue(cfg["height_scan"]["mask_visible"])
    self.assertEqual(len(cfg["height_scan"]["visible_mask"]), 187)
    self.assertTrue(any(value == 0 for value in cfg["height_scan"]["visible_mask"]))
    self.assertEqual(sorted(cfg["joint_ids_map"]), list(range(16)))

  def test_rejects_wrong_tensor_shape_or_history_layout(self):
    cfg = copy.deepcopy(self.config)
    cfg["observation_dim"] -= 1
    with self.assertRaisesRegex(ValueError, "dimensions"):
      validate_contract(self.model, cfg, self.metadata)
    model = copy.deepcopy(self.model)
    model.graph.output[0].type.tensor_type.elem_type = onnx.TensorProto.DOUBLE
    with self.assertRaisesRegex(ValueError, "float32"):
      validate_contract(model, self.config, self.metadata)
    meta = dict(self.metadata, observation_layout="frame_major")
    with self.assertRaisesRegex(ValueError, "history"):
      validate_contract(self.model, self.config, meta)
    cfg = copy.deepcopy(self.config)
    cfg["actions"]["JointVelocityAction"]["scale"][0] = 1
    with self.assertRaisesRegex(ValueError, "action_scale"):
      validate_contract(self.model, cfg, self.metadata)
    meta = dict(self.metadata, action_types=",".join(["position"]*16))
    with self.assertRaisesRegex(ValueError, "wheel velocity"):
      validate_contract(self.model, self.config, meta)

  def test_legacy_onnx_requires_explicit_reexport(self):
    model = copy.deepcopy(self.model)
    del model.metadata_props[:]
    with tempfile.TemporaryDirectory() as temporary:
      path = Path(temporary)/"policy.onnx"
      onnx.save(model, path)
      with self.assertRaisesRegex(ValueError, "re-export"):
        write_deploy_yaml(path)

  def test_cli_packages_policy_and_matching_yaml(self):
    with tempfile.TemporaryDirectory() as temporary:
      output = Path(temporary)/"bundle"
      command = [sys.executable, str(ROOT/"scripts/prepare_go2w_deploy.py"),
        "--onnx", str(self.path), "--output", str(output)]
      subprocess.run(command, check=True, capture_output=True, text=True, cwd=ROOT)
      loaded = onnx.load(output/"exported/policy.onnx")
      cfg = yaml.safe_load((output/"params/deploy.yaml").read_text())
      validate_contract(loaded, cfg, cfg["onnx_metadata"])
      again = subprocess.run(command, capture_output=True, text=True, cwd=ROOT)
      self.assertNotEqual(again.returncode, 0)

  @unittest.skipUnless(importlib.util.find_spec("onnxruntime"), "Optional ONNX Runtime parity check")
  def test_exported_normalization_matches_pytorch_actor(self):
    import onnxruntime as ort
    session = ort.InferenceSession(str(self.path), providers=["CPUExecutionProvider"])
    observation = self.wrapped.get_observations()
    observation["actor"] = torch.randn_like(observation["actor"])
    with torch.inference_mode():
      expected = self.runner.get_inference_policy(device="cpu")(observation).cpu().numpy()
    actual = session.run(None, {"obs": observation["actor"].cpu().numpy()})[0]
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=1e-5)

  def test_scene_has_named_motors_and_supports_standing_pd(self):
    with tempfile.TemporaryDirectory() as temporary:
      m = make_scene(Path(temporary)/"scene.xml", stairs=True)
      d = mujoco.MjData(m)
      mujoco.mj_resetDataKeyframe(m, d, 0)
      initial = d.qpos.copy()
      names = self.config["action_joint_names"]
      for _ in range(1000):
        for i, name in enumerate(names):
          j = m.joint(name).id
          a = m.actuator(f"sdk_{i}").id
          q, dq = d.qpos[m.jnt_qposadr[j]], d.qvel[m.jnt_dofadr[j]]
          kp = self.config["stiffness"][self.config["joint_ids_map"][i]]
          kd = self.config["damping"][self.config["joint_ids_map"][i]]
          target = initial[m.jnt_qposadr[j]]
          d.ctrl[a] = np.clip(kp*(target-q)-kd*dq, *m.actuator_ctrlrange[a])
        mujoco.mj_step(m, d)
      self.assertTrue(np.isfinite(d.qpos).all())
      self.assertGreater(d.qpos[2], 0.20)
      self.assertGreater(d.qpos[3], 0.95)

  @unittest.skipUnless(os.environ.get("GO2W_CORE_TEST"), "Set GO2W_CORE_TEST to the compiled C++ test executable")
  def test_cpp_memory_matches_actual_training_function(self):
    binary = os.environ["GO2W_CORE_TEST"]
    subprocess.run([binary], check=True, capture_output=True, text=True)
    trace = subprocess.run([binary,"--trace"], check=True, capture_output=True, text=True)
    cpp = np.loadtxt(trace.stdout.splitlines())
    memory = bridged_mid360_height_scan_memory()
    env = SimpleNamespace(num_envs=1, step_dt=0.02, common_step_counter=6000*24, extras={},
      scene={"robot": SimpleNamespace(data=SimpleNamespace(root_link_lin_vel_b=torch.zeros(1,3)))})
    python = []
    for t in range(60):
      env.scene["robot"].data.root_link_lin_vel_b[0,0] = 0.7 if t < 30 else 0
      scan = 0.3 + 0.001*torch.arange(187,dtype=torch.float32).unsqueeze(0) + 0.0001*t
      with patch("src.tasks.velocity.mdp.observations.envs_mdp.height_scan", return_value=scan):
        value = memory(env,"dense","mid360",x_offset_m=0.35,
          visible_x_min_m=0.25,visible_x_max_m=1.15,visible_y_abs_m=0.5,
          keep_prob_min=1.,keep_prob_max=1.,frame_dropout_prob=0.,fixed_curriculum_step=6000*24)
      python.append(value[0].numpy())
    np.testing.assert_allclose(cpp, python, atol=2e-6, rtol=0)


if __name__ == "__main__":
  unittest.main()
