"""CPU checks for Go2W morphology, task compatibility and rolling rewards.

Run from the repository root: python -m unittest discover -s tests -v
"""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import mujoco
import torch
from mjlab.entity import Entity
from mjlab.sensor import GridPatternCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg

import src.tasks  # noqa: F401 -- register repository tasks
from src.assets.robots.unitree_go2w.go2w_constants import (
  LEG_JOINTS, WHEEL_JOINTS, WHEEL_RADIUS, get_go2w_robot_cfg,
)
from src.tasks.velocity.mdp.observations import bridged_mid360_height_scan_memory
from src.tasks.velocity.mdp.wheel_rewards import wheel_spin_contact, wheel_stand_still
from src.tasks.velocity.config.go2w import env_cfgs as go2w_env_cfgs


class Go2WTests(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.robot = Entity(get_go2w_robot_cfg())
    cls.model = cls.robot.spec.compile()

  def test_go2w_configs_do_not_import_other_task_configs(self):
    config_dir = Path(go2w_env_cfgs.__file__).parent
    for path in config_dir.glob("*.py"):
      for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        modules = []
        if isinstance(node, ast.ImportFrom):
          modules = [node.module or ""]
        elif isinstance(node, ast.Import):
          modules = [alias.name for alias in node.names]
        for module in modules:
          if module.startswith("src.tasks.velocity.config."):
            self.assertTrue(module.startswith("src.tasks.velocity.config.go2w."), (path.name, module))
          self.assertFalse(module.startswith("mjlab.tasks.") and ".config." in module, (path.name, module))

  def test_registered_variants_do_not_chain_task_factories(self):
    factories = {
      name: value for name, value in vars(go2w_env_cfgs).items()
      if name.startswith("unitree_go2w_") and name.endswith("_env_cfg")
    }
    self.assertEqual(len(factories), 6)
    blocked = {name: Mock(side_effect=AssertionError(f"Task factory reused: {name}")) for name in factories}
    with patch.multiple(go2w_env_cfgs, **blocked):
      for name, factory in factories.items():
        for play in (False, True):
          with self.subTest(task=name, play=play):
            cfg = factory(play=play)
            self.assertIn("robot", cfg.scene.entities)
            self.assertEqual(tuple(cfg.actions), ("joint_pos", "wheel_vel"))

  def test_morphology_and_control_modes(self):
    model = self.model
    self.assertEqual((model.nq, model.nv, model.nu), (23, 22, 16))
    for name in self.robot.joint_names:
      joint = model.joint(name)
      actuator = next(a for a in self.robot.spec.actuators if a.target == name)
      idx = actuator.id
      if "wheel" in name:
        self.assertFalse(joint.limited[0])
        self.assertAlmostEqual(model.actuator_biasprm[idx, 1], 0.0)
        self.assertAlmostEqual(model.actuator_gainprm[idx, 0], 2.0)
        self.assertAlmostEqual(model.actuator_biasprm[idx, 2], -2.0)
        self.assertAlmostEqual(model.actuator_forcerange[idx, 1], 45.0)
        self.assertEqual(model.actuator_ctrlrange[idx].tolist(), [-20.0, 20.0])
        self.assertAlmostEqual(model.dof_armature[joint.dofadr[0]], 0.02)
      else:
        self.assertTrue(joint.limited[0])
        calf = "calf" in name
        kp, kd, effort, armature = (40.0, 2.0, 45.0, 0.02) if calf else (20.0, 1.0, 23.5, 0.01)
        self.assertAlmostEqual(model.actuator_gainprm[idx, 0], kp)
        self.assertAlmostEqual(model.actuator_biasprm[idx, 1], -kp)
        self.assertAlmostEqual(model.actuator_biasprm[idx, 2], -kd)
        self.assertEqual(model.actuator_forcerange[idx].tolist(), [-effort, effort])
        self.assertAlmostEqual(model.dof_armature[joint.dofadr[0]], armature)
      self.assertAlmostEqual(model.dof_frictionloss[joint.dofadr[0]], 0.0)
    self.assertEqual(len(self.robot.find_joints(LEG_JOINTS)[0]), 12)
    self.assertEqual(len(self.robot.find_joints(WHEEL_JOINTS)[0]), 4)

  def test_initial_keyframe_matches_zhongl1_reference(self):
    # ZhongL1/unitree_rl_mjlab_go2w @ fe0deea, asset INIT_STATE.
    # Inspect compiled qpos by joint name to catch sign/order mapping mistakes.
    qpos = self.model.key("init_state").qpos
    self.assertEqual(qpos[:7].tolist(), [0.0, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0])
    for leg in ("FR", "FL", "RR", "RL"):
      expected = {"hip": 0.1 if leg.endswith("R") else -0.1, "thigh": 0.9, "calf": -1.8, "wheel": 0.0}
      for kind, value in expected.items():
        name = f"{leg}_{kind}_joint"
        with self.subTest(joint=name):
          self.assertAlmostEqual(qpos[self.model.joint(name).qposadr[0]], value)

  def test_reference_collisions_keep_only_named_physics_geoms(self):
    # Reference FULL_COLLISION has condim=3, priority=0; no friction override.
    for geom_id in range(self.model.ngeom):
      geom = self.model.geom(geom_id)
      with self.subTest(geom=geom.name):
        if "_collision" in geom.name:
          self.assertEqual(int(geom.condim[0]), 3)
          self.assertEqual(int(geom.priority[0]), 0)
          self.assertEqual(int(geom.contype[0]), 1)
          self.assertEqual(int(geom.conaffinity[0]), 0)
          if "wheel_link" in geom.name:
            self.assertEqual(geom.friction.tolist(), [0.8, 0.02, 0.01])
        else:
          self.assertEqual(int(geom.contype[0]), 0)
          self.assertEqual(int(geom.conaffinity[0]), 0)

  def test_registered_tasks_and_compatible_stages(self):
    tasks = [name for name in list_tasks() if name.startswith("Unitree-Go2W-")]
    self.assertEqual(len(tasks), 6)
    schemas = []
    for task in tasks:
      cfg = load_env_cfg(task)
      self.assertEqual(set(cfg.actions), {"joint_pos", "wheel_vel"})
      self.assertFalse(cfg.commands["twist"].heading_command)
      self.assertIsNone(cfg.commands["twist"].ranges.heading)
      for group in cfg.observations.values():
        self.assertEqual(group.terms["joint_pos"].params["asset_cfg"].joint_names, LEG_JOINTS)
        self.assertEqual(group.terms["wheel_vel"].params["asset_cfg"].joint_names, WHEEL_JOINTS)
      self.assertNotIn("foot_slip", cfg.rewards)
      self.assertNotIn("foot_gait", cfg.rewards)
      self.assertNotIn("foot_clearance", cfg.rewards)
      self.assertNotIn("wheel_slip", cfg.rewards)
      self.assertIs(cfg.rewards["wheel_spin_contact"].func, wheel_spin_contact)
      for name in ("pose", "stand_still", "joint_pos_limits", "joint_acc_l2"):
        self.assertEqual(cfg.rewards[name].params["asset_cfg"].joint_names, LEG_JOINTS)
      self.assertEqual(load_rl_cfg(task).clip_actions, 1.0)
      if task != "Unitree-Go2W-Flat":
        schemas.append([(g.history_length, tuple(g.terms)) for g in cfg.observations.values()])
        self.assertEqual(load_rl_cfg(task).experiment_name, "go2w_perceptive")
      else:
        self.assertNotIn("height_scan", cfg.observations["actor"].terms)
    self.assertTrue(all(schema == schemas[0] for schema in schemas))

  def test_wheel_contacts_bodies_and_joints_have_same_order(self):
    cfg = load_env_cfg("Unitree-Go2W-Rough")
    sensor = next(s for s in cfg.scene.sensors if s.name == "feet_ground_contact")
    geoms = self.robot.find_geoms(sensor.primary.pattern)[1]
    sites = cfg.observations["critic"].terms["foot_height"].params["asset_cfg"].site_names
    self.assertEqual([name[:2] for name in geoms], list(sites))
    nonfoot = next(s for s in cfg.scene.sensors if s.name == "nonfoot_ground_touch")
    self.assertEqual(nonfoot.primary.exclude, sensor.primary.pattern)
    spin = cfg.rewards["wheel_spin_contact"]
    spin_sensor = next(s for s in cfg.scene.sensors if s.name == spin.params["sensor_name"])
    self.assertEqual(spin_sensor.primary.pattern, sensor.primary.pattern)
    self.assertEqual(spin_sensor.reduce, "none")
    self.assertFalse(spin_sensor.global_frame)
    self.assertEqual(spin_sensor.fields, ("force", "normal"))
    bodies = self.robot.find_bodies(spin.params["body_cfg"].body_names)[1]
    joints = self.robot.find_joints(spin.params["joint_cfg"].joint_names)[1]
    self.assertEqual([n[:2] for n in geoms], [n[:2] for n in bodies])
    self.assertEqual([n[:2] for n in geoms], [n[:2] for n in joints])

  def test_sparse_play_and_memory_are_independent(self):
    task = "Unitree-Go2W-Perceptive-Rough-v2"
    train, play = load_env_cfg(task), load_env_cfg(task, play=True)
    a = train.observations["actor"].terms["height_scan"]
    b = play.observations["actor"].terms["height_scan"]
    self.assertIsNot(a.func, b.func)
    self.assertEqual(a.params["frame_dropout_prob"], 0.02)
    self.assertEqual(b.params["fixed_curriculum_step"], 6000 * 24)
    self.assertEqual(b.params["frame_dropout_prob"], 0.0)
    self.assertEqual(b.delay_max_lag, 0)
    self.assertFalse(play.observations["actor"].enable_corruption)
    self.assertEqual(play.curriculum, {})
    for sensor in train.scene.sensors:
      if sensor.name.startswith("terrain_scan"):
        self.assertEqual(sensor.include_geom_groups, (0, 1))

  def test_step14_is_fixed_ascent(self):
    cfg = load_env_cfg("Unitree-Go2W-Perceptive-Rough-v2-Step14-Up", play=True)
    terrain = cfg.scene.terrain.terrain_generator
    enabled = {k: v for k, v in terrain.sub_terrains.items() if v.proportion > 0}
    self.assertEqual(list(enabled), ["pyramid_stairs_inv"])
    self.assertEqual(enabled["pyramid_stairs_inv"].step_height_range, (0.14, 0.14))
    self.assertFalse(terrain.curriculum)

  def test_offset_grid_geometry_and_type(self):
    cfg = load_env_cfg("Unitree-Go2W-Perceptive-Rough-v2")
    sensor = next(s for s in cfg.scene.sensors if s.name == "terrain_scan")
    self.assertIsInstance(sensor.pattern, GridPatternCfg)
    offsets, directions = sensor.pattern.generate_rays(None, "cpu")
    self.assertEqual(tuple(offsets.shape), (187, 3))
    self.assertAlmostEqual(offsets[:, 0].min().item(), -0.45, places=5)
    self.assertAlmostEqual(offsets[:, 0].max().item(), 1.15, places=5)
    self.assertTrue(torch.all(directions[:, 2] == -1.0))

  def _rolling_env(self, speed=1.0, spin=None, contact=True, num_wheels=4):
    spin = speed / WHEEL_RADIUS if spin is None else spin
    data = SimpleNamespace(
      body_com_lin_vel_w=torch.tensor([[[speed, 0.0, 0.0]]]).expand(2, num_wheels, 3).clone(),
      joint_vel=torch.full((2, num_wheels), float(spin)),
    )
    forces = torch.zeros(2, num_wheels, 8, 3)
    forces[:, :, 0, 0] = 2.0 if contact else 0.0
    normals = torch.zeros_like(forces)
    normals[:, :, 0, 2] = 1.0
    return SimpleNamespace(
      scene={"robot": SimpleNamespace(data=data), "contact": SimpleNamespace(
        cfg=SimpleNamespace(reduce="none", global_frame=False, num_slots=8),
        data=SimpleNamespace(force=forces.flatten(1, 2), normal=normals.flatten(1, 2)))},
      command_manager=SimpleNamespace(get_command=lambda name: torch.zeros(2, 3)),
    )

  def _spin_cost(self, env, radius=WHEEL_RADIUS):
    cfg = SimpleNamespace(name="robot", joint_ids=slice(None), body_ids=slice(None))
    return wheel_spin_contact(env, "contact", radius, cfg, cfg)

  def test_pure_rolling_has_no_spin_cost(self):
    torch.testing.assert_close(self._spin_cost(self._rolling_env()), torch.zeros(2))
    torch.testing.assert_close(self._spin_cost(self._rolling_env(spin=0.0)), torch.zeros(2))
    spinning = self._spin_cost(self._rolling_env(speed=0.0, spin=10.0))
    torch.testing.assert_close(spinning, torch.full((2,), 4 * (0.8 * WHEEL_RADIUS * 10 - 0.1)))
    torch.testing.assert_close(spinning, self._spin_cost(self._rolling_env(speed=0.0, spin=-10.0)))
    torch.testing.assert_close(
      self._spin_cost(self._rolling_env(speed=0.0, spin=10.0, contact=False)), torch.zeros(2)
    )

  def test_spin_matches_robot_lab_reference_fixture(self):
    # Same numerical case as robot_lab/tests/test_backend_configuration_update.py.
    env = self._rolling_env(speed=0.0, spin=0.0, contact=False, num_wheels=3)
    env.scene["robot"].data.joint_vel = torch.tensor([[1., 2., 3.], [4., 1., 0.]])
    env.scene["robot"].data.body_com_lin_vel_w[0, 2, 0] = 0.12
    forces = env.scene["contact"].data.force.reshape(2, 3, 8, 3)
    forces[:, [0, 2], 0, 0] = 2.0
    torch.testing.assert_close(self._spin_cost(env, radius=0.1), torch.tensor([0.02, 0.22]))

  def test_spin_gate_uses_current_normal_force_only(self):
    env = self._rolling_env(speed=0.0, spin=10.0, contact=False)
    data = env.scene["contact"].data
    forces = data.force.reshape(2, 4, 8, 3)
    normals = data.normal.reshape_as(forces)
    forces[..., 1:] = 100.0  # large friction alone must not count as support
    torch.testing.assert_close(self._spin_cost(env), torch.zeros(2))
    forces[:, :, 0, 0] = 1.0  # robot_lab uses a strict > 1 N comparison
    torch.testing.assert_close(self._spin_cost(env), torch.zeros(2))
    # Sum normal reactions from multiple actual contacts, including a stair face.
    forces[:, :, :2, 0] = 0.6
    normals[:, :, :2] = torch.tensor([1., 0., 0.])
    self.assertTrue(torch.all(self._spin_cost(env) > 0))
    # Historical contact cannot keep this reward active after detachment.
    data.force_history = torch.full((2, 32, 4, 3), 100.0)
    forces[..., 0] = 0.0
    torch.testing.assert_close(self._spin_cost(env), torch.zeros(2))

  def test_zero_command_brakes_wheels(self):
    env = self._rolling_env()
    cfg = SimpleNamespace(name="robot", joint_ids=slice(None))
    self.assertTrue(torch.all(wheel_stand_still(env, "twist", cfg) > 0))
    env.command_manager.get_command = lambda name: torch.ones(2, 3)
    self.assertTrue(torch.all(wheel_stand_still(env, "twist", cfg) == 0))

  def test_sparse_memory_resets_per_environment(self):
    memory = bridged_mid360_height_scan_memory()
    env = SimpleNamespace(
      num_envs=2, step_dt=0.02, common_step_counter=6000 * 24, extras={},
      scene={"robot": SimpleNamespace(data=SimpleNamespace(root_link_lin_vel_b=torch.zeros(2, 3)))},
    )
    kwargs = dict(
      sensor_name="dense", mid360_sensor_name="sparse", x_offset_m=0.35,
      keep_prob_min=1.0, keep_prob_max=1.0, frame_dropout_prob=0.0,
    )
    with patch("src.tasks.velocity.mdp.observations.envs_mdp.height_scan", return_value=torch.full((2, 187), 0.4)):
      scan = memory(env, **kwargs)
      self.assertTrue(torch.all((scan == 0.4) | (scan == 5.0)))
      self.assertTrue(torch.any(scan == 5.0))  # no dense ground truth leaks into blind cells
      memory.reset(torch.tensor([0]))
      self.assertTrue(torch.all(memory.age[0] == 10000))
      self.assertTrue(torch.any(memory.age[1] == 0))


if __name__ == "__main__":
  unittest.main()
