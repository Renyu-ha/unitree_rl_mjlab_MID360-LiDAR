"""CPU checks for Go2W morphology, task compatibility and rolling rewards.

Run from the repository root: python -m unittest discover -s tests -v
"""

import ast
from dataclasses import asdict
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import mujoco
import numpy as np
from mjlab.actuator import BuiltinVelocityActuatorCfg
from mjlab.entity import Entity
from mjlab.sensor import ContactSensorCfg, GridPatternCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg

import src.tasks  # noqa: F401 -- register repository tasks
from src.assets.robots.unitree_go2w.go2w_constants import (
  GO2W_XML, LEG_JOINTS, WHEEL_JOINTS, WHEEL_RADIUS, get_go2w_robot_cfg, get_spec,
)
from src.tasks.velocity.mdp.observations import bridged_mid360_height_scan_memory
from src.tasks.velocity.mdp.wheel_rewards import wheel_spin_contact, wheel_stand_still
from src.tasks.velocity.config.go2w import env_cfgs as go2w_env_cfgs


class Go2WTests(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.robot = Entity(get_go2w_robot_cfg())
    cls.model = cls.robot.spec.compile()

  def test_final_xml_is_self_contained_without_unused_assets(self):
    folder = GO2W_XML.parent
    root = ET.parse(GO2W_XML).getroot()
    meshdir = Path(root.find("compiler").get("meshdir", ""))
    mesh_paths = [(meshdir / mesh.get("file")).as_posix() for mesh in root.findall("asset/mesh")]
    self.assertEqual(len(mesh_paths), 14)
    self.assertTrue(all(Path(path).parent == Path("assets") for path in mesh_paths))
    self.assertEqual({(folder / path).name for path in mesh_paths}, {path.name for path in (folder / "assets").iterdir()})
    self.assertEqual({Path(path).stem for path in mesh_paths}, {geom.get("mesh") for geom in root.findall(".//worldbody//geom") if geom.get("mesh")})
    # Loading the active XML itself must work without the other robot folders
    # or the asset injection performed by Entity/get_spec().
    standalone = mujoco.MjModel.from_xml_path(str(GO2W_XML))
    self.assertEqual((standalone.nq, standalone.nv, standalone.nmesh, standalone.nsensor), (23, 22, 14, 3))

  def test_loader_does_not_modify_final_xml_definitions(self):
    before = GO2W_XML.read_bytes()
    raw = mujoco.MjModel.from_xml_path(str(GO2W_XML))
    loaded = get_spec().compile()
    # The loader only embeds files; all model interfaces are already in XML.
    fields = (
      "body_pos", "body_quat", "body_ipos", "body_iquat", "body_mass", "body_inertia",
      "jnt_type", "jnt_pos", "jnt_axis", "jnt_range", "jnt_limited",
      "dof_armature", "dof_damping", "dof_frictionloss",
      "geom_type", "geom_size", "geom_pos", "geom_quat", "geom_dataid",
      "geom_friction", "geom_condim", "geom_priority", "geom_margin", "geom_solref", "geom_solimp",
      "geom_contype", "geom_conaffinity", "geom_group", "site_pos", "site_size",
      "sensor_type", "sensor_objtype", "sensor_objid", "sensor_dim", "names",
      "mesh_vert", "mesh_face",
    )
    for field in fields:
      with self.subTest(field=field):
        np.testing.assert_array_equal(getattr(raw, field), getattr(loaded, field))
    self.assertEqual(loaded.nsensor, 3)
    self.assertEqual(GO2W_XML.read_bytes(), before)

  def test_task_sensor_references_exist_with_retained_telemetry(self):
    available = {self.model.sensor(index).name for index in range(self.model.nsensor)}
    referenced = set()
    for task in list_tasks():
      if not task.startswith("Unitree-Go2W-"):
        continue
      for play in (False, True):
        cfg = load_env_cfg(task, play=play)
        terms = [term for group in cfg.observations.values() for term in group.terms.values()]
        terms.extend(cfg.rewards.values())
        for term in terms:
          sensor = term.params.get("sensor_name", "")
          if sensor.startswith("robot/"):
            name = sensor.removeprefix("robot/")
            self.assertIn(name, available, (task, play, name))
            referenced.add(name)
    self.assertEqual(referenced, {"imu_ang_vel", "imu_lin_vel"})
    self.assertEqual(available, referenced | {"root_angmom"})

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

  def test_go2w_configurations_are_explicit_without_loops(self):
    paths = list(Path(go2w_env_cfgs.__file__).parent.glob("*.py"))
    paths.append(Path(__file__).parents[1] / "src/assets/robots/unitree_go2w/go2w_constants.py")
    loops = (ast.For, ast.AsyncFor, ast.While, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
    for path in paths:
      for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        self.assertNotIsInstance(node, loops, f"{path.name}:{getattr(node, 'lineno', 0)}")

  def test_morphology_and_control_modes(self):
    model = self.model
    self.assertEqual((model.nq, model.nv, model.nu), (23, 22, 16))
    self.assertIs(type(get_go2w_robot_cfg().articulation.actuators[-1]), BuiltinVelocityActuatorCfg)
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
        self.assertFalse(model.actuator_ctrllimited[idx])
        self.assertAlmostEqual(model.dof_armature[joint.dofadr[0]], 0.0)
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

  def test_collision_masks_enable_self_contacts_and_disable_visuals(self):
    # Retain reference condim/priority/friction, with self collision enabled.
    for geom_id in range(self.model.ngeom):
      geom = self.model.geom(geom_id)
      with self.subTest(geom=geom.name):
        if "_collision" in geom.name:
          self.assertEqual(int(geom.condim[0]), 3)
          self.assertEqual(int(geom.priority[0]), 0)
          self.assertEqual(int(geom.contype[0]), 1)
          self.assertEqual(int(geom.conaffinity[0]), 1)
          if "wheel_link" in geom.name:
            self.assertEqual(geom.friction.tolist(), [0.8, 0.02, 0.01])
        else:
          self.assertEqual(int(geom.contype[0]), 0)
          self.assertEqual(int(geom.conaffinity[0]), 0)

  def test_wheel_cylinders_match_visual_mesh_size_and_axle(self):
    model = self.model
    for leg in ("FL", "FR", "RL", "RR"):
      with self.subTest(leg=leg):
        collision = model.geom(f"{leg}_wheel_link_collision")
        body_id = model.body(f"{leg}_wheel_link").id
        visuals = [i for i in range(model.ngeom) if model.geom_bodyid[i] == body_id
          and model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH]
        self.assertEqual(len(visuals), 1)
        visual = model.geom(visuals[0])
        self.assertEqual(int(collision.type[0]), mujoco.mjtGeom.mjGEOM_CYLINDER)
        self.assertFalse(visual.contype[0] or visual.conaffinity[0])
        mesh_id = int(visual.dataid[0])
        start, count = int(model.mesh_vertadr[mesh_id]), int(model.mesh_vertnum[mesh_id])
        rotation = np.empty((3, 3))
        mujoco.mju_quat2Mat(rotation.ravel(), visual.quat)
        vertices = model.mesh_vert[start:start+count] @ rotation.T + visual.pos
        lower, upper = vertices.min(axis=0), vertices.max(axis=0)
        np.testing.assert_allclose(collision.pos, (lower + upper) / 2, atol=1e-8)
        self.assertAlmostEqual(collision.size[0], WHEEL_RADIUS)
        self.assertAlmostEqual(collision.size[0], np.linalg.norm(vertices[:, [0, 2]], axis=1).max())
        self.assertAlmostEqual(collision.size[1], (upper[1] - lower[1]) / 2)
        mujoco.mju_quat2Mat(rotation.ravel(), collision.quat)
        np.testing.assert_allclose(rotation[:, 2], model.joint(f"{leg}_wheel_joint").axis, atol=1e-12)
        self.assertAlmostEqual(model.body_mass[body_id], 0.98)
        np.testing.assert_array_equal(model.body_inertia[body_id], [0.003, 0.0016, 0.0016])

  def test_wheel_ground_contact_height_is_invariant_to_wheel_spin(self):
    spec = get_spec()
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=(1, 1, 0.1))
    model = spec.compile()
    wheel = model.geom("FL_wheel_link_collision")
    data = mujoco.MjData(model)
    joint = model.joint("FL_wheel_joint")
    distances = []
    for angle in np.linspace(0.0, 2 * np.pi, 17):
      data.qpos[:] = model.qpos0
      data.qpos[joint.qposadr[0]] = angle
      mujoco.mj_fwdPosition(model, data)
      axis = data.geom_xmat[wheel.id].reshape(3, 3)[:, 2]
      support_z = wheel.size[0] * np.sqrt(1.0 - axis[2]**2) + wheel.size[1] * abs(axis[2])
      data.qpos[2] -= data.geom_xpos[wheel.id, 2] - support_z + 0.0002
      mujoco.mj_fwdPosition(model, data)
      contacts = [c.dist for c in data.contact if wheel.id in (c.geom1, c.geom2)
        and model.geom("floor").id in (c.geom1, c.geom2)]
      self.assertTrue(contacts)
      distances.append(min(contacts))
    np.testing.assert_allclose(distances, -0.0002, atol=1e-7)

  def test_self_collision_is_clear_in_default_pose_and_active_when_folded(self):
    model = self.model
    self.assertFalse(model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_FILTERPARENT)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_fwdPosition(model, data)
    self.assertEqual(data.ncon, 0)
    # A legal folded front-leg pose makes the front-right wheel touch the
    # rear-right wheel/calf. These are non-adjacent bodies, not hinge neighbors.
    for leg, values in (("FL", (-.298800904, 2.388383992, -1.448528280)),
                        ("FR", (.260503085, 1.559224259, -.902612689))):
      for kind, value in zip(("hip", "thigh", "calf"), values):
        data.qpos[model.joint(f"{leg}_{kind}_joint").qposadr[0]] = value
    mujoco.mj_fwdPosition(model, data)
    pairs = {frozenset((model.geom(int(c.geom1)).name, model.geom(int(c.geom2)).name))
      for c in data.contact if c.dist < -.001}
    self.assertIn(frozenset(("FR_wheel_link_collision", "RR_wheel_link_collision")), pairs)
    affinity = model.geom_conaffinity.copy()
    try:
      model.geom_conaffinity[:] = 0
      mujoco.mj_fwdPosition(model, data)
      self.assertEqual(data.ncon, 0)
    finally:
      model.geom_conaffinity[:] = affinity

  def test_registered_tasks_and_compatible_stages(self):
    tasks = [name for name in list_tasks() if name.startswith("Unitree-Go2W-")]
    self.assertEqual(len(tasks), 6)
    schemas = []
    for task in tasks:
      cfg = load_env_cfg(task)
      self.assertEqual(set(cfg.actions), {"joint_pos", "wheel_vel"})
      self.assertTrue(cfg.commands["twist"].heading_command)
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
      self.assertIsNone(load_rl_cfg(task).clip_actions)
      if task != "Unitree-Go2W-Flat":
        if "v2" in task:
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
    self.assertIs(spin_sensor, sensor)
    self.assertEqual([s.name for s in cfg.scene.sensors], ["terrain_scan", "feet_ground_contact", "nonfoot_ground_touch", "self_collision"])
    self.assertIs(type(spin_sensor), ContactSensorCfg)
    self.assertEqual(spin_sensor.primary.pattern, sensor.primary.pattern)
    self.assertEqual(spin_sensor.reduce, "netforce")
    self.assertEqual(spin_sensor.num_slots, 1)
    self.assertFalse(spin_sensor.global_frame)
    self.assertTrue(spin_sensor.track_air_time)
    self.assertEqual(spin_sensor.fields, ("found", "force"))
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
    sensor = next(s for s in cfg.scene.sensors if s.name == "terrain_scan_mid360")
    self.assertIsInstance(sensor.pattern, GridPatternCfg)
    offsets, directions = sensor.pattern.generate_rays(None, "cpu")
    self.assertEqual(tuple(offsets.shape), (187, 3))
    self.assertAlmostEqual(offsets[:, 0].min().item(), -0.45, places=5)
    self.assertAlmostEqual(offsets[:, 0].max().item(), 1.15, places=5)
    self.assertTrue(torch.all(directions[:, 2] == -1.0))

  def test_generic_training_settings_match_corresponding_g1_tasks(self):
    pairs = (
      ("Unitree-Go2W-Flat", "Unitree-G1-Flat"),
      ("Unitree-Go2W-Rough", "Unitree-G1-Rough"),
      ("Unitree-Go2W-Perceptive-Rough-v1", "Unitree-G1-Perceptive-Rough-v1"),
      ("Unitree-Go2W-Perceptive-Rough-v2", "Unitree-G1-Perceptive-Rough-v2"),
      ("Unitree-Go2W-Perceptive-Rough-v2-StepUp-Finetune", "Unitree-G1-Perceptive-Rough-v2-StepUp-Finetune"),
      ("Unitree-Go2W-Perceptive-Rough-v2-Step14-Up", "Unitree-G1-Perceptive-Rough-v2-Step14-Up"),
    )
    for go2w_task, g1_task in pairs:
      with self.subTest(task=go2w_task):
        actual, expected = asdict(load_rl_cfg(go2w_task)), asdict(load_rl_cfg(g1_task))
        for key in ("experiment_name", "run_name"):
          actual.pop(key)
          expected.pop(key)
        self.assertEqual(actual, expected)
      for play in (False, True):
        with self.subTest(task=go2w_task, play=play):
          actual, expected = load_env_cfg(go2w_task, play=play), load_env_cfg(g1_task, play=play)
          self.assertEqual(asdict(actual.sim), asdict(expected.sim))
          terrain, reference_terrain = asdict(actual.scene.terrain), asdict(expected.scene.terrain)
          # TerrainEntityCfg creates a fresh bound loader for every instance.
          terrain.pop("spec_fn")
          reference_terrain.pop("spec_fn")
          self.assertEqual(terrain, reference_terrain)
          commands, reference = asdict(actual.commands["twist"]), asdict(expected.commands["twist"])
          commands.pop("viz")
          reference.pop("viz")
          self.assertEqual(commands, reference)
          self.assertEqual(actual.decimation, expected.decimation)
          for group in ("actor", "critic"):
            self.assertEqual(actual.observations[group].history_length, expected.observations[group].history_length)
            self.assertEqual(actual.observations[group].enable_corruption, expected.observations[group].enable_corruption)
            for term in expected.observations[group].terms:
              a, b = asdict(actual.observations[group].terms[term]), asdict(expected.observations[group].terms[term])
              a.pop("func")
              b.pop("func")
              a["params"].pop("asset_cfg", None)
              b["params"].pop("asset_cfg", None)
              self.assertEqual(a, b, (group, term))
          for event in actual.events.keys() & expected.events.keys():
            a, b = asdict(actual.events[event]), asdict(expected.events[event])
            a["params"].pop("asset_cfg", None)
            b["params"].pop("asset_cfg", None)
            self.assertEqual(a, b, event)
          for reward in actual.rewards.keys() & expected.rewards.keys() - {"pose", "body_orientation_l2", "track_angular_velocity"}:
            self.assertEqual(actual.rewards[reward].weight, expected.rewards[reward].weight, reward)
          self.assertEqual(actual.rewards["pose"].weight, 0.5)
          self.assertEqual(actual.rewards["body_orientation_l2"].weight, -0.7)
          self.assertEqual(actual.rewards["track_angular_velocity"].params["std"], 0.5)
          self.assertEqual(actual.rewards["track_angular_velocity"].weight, expected.rewards["track_angular_velocity"].weight)
          self.assertNotIn("illegal_contact", actual.terminations)
          for curriculum in ("command_vel", "track_linear_velocity_weight"):
            if curriculum in expected.curriculum:
              self.assertEqual(asdict(actual.curriculum[curriculum]), asdict(expected.curriculum[curriculum]))
          self.assertNotIn("pose_weight", actual.curriculum)
          self.assertNotIn("foot_clearance_weight", actual.curriculum)
          self.assertNotIn("foot_swing_height_weight", actual.curriculum)

  def _rolling_env(self, speed=1.0, spin=None, contact=True, num_wheels=4):
    spin = speed / WHEEL_RADIUS if spin is None else spin
    data = SimpleNamespace(
      body_com_lin_vel_w=torch.tensor([[[speed, 0.0, 0.0]]]).expand(2, num_wheels, 3).clone(),
      joint_vel=torch.full((2, num_wheels), float(spin)),
    )
    forces = torch.zeros(2, num_wheels, 3)
    forces[:, :, 2] = 2.0 if contact else 0.0
    return SimpleNamespace(
      scene={"robot": SimpleNamespace(data=data), "contact": SimpleNamespace(
        data=SimpleNamespace(force=forces))},
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
    # Same spin-formula fixture as robot_lab/tests/test_backend_configuration_update.py.
    # Contact gating here intentionally uses total force, including friction.
    env = self._rolling_env(speed=0.0, spin=0.0, contact=False, num_wheels=3)
    env.scene["robot"].data.joint_vel = torch.tensor([[1., 2., 3.], [4., 1., 0.]])
    env.scene["robot"].data.body_com_lin_vel_w[0, 2, 0] = 0.12
    env.scene["contact"].data.force[:, [0, 2], 2] = 2.0
    torch.testing.assert_close(self._spin_cost(env, radius=0.1), torch.tensor([0.02, 0.22]))

  def test_spin_gate_uses_current_net_contact_force(self):
    env = self._rolling_env(speed=0.0, spin=10.0, contact=False)
    data = env.scene["contact"].data
    torch.testing.assert_close(self._spin_cost(env), torch.zeros(2))
    data.force[..., 2] = 1.0  # Keep a strict > 1 N deadband for small forces.
    torch.testing.assert_close(self._spin_cost(env), torch.zeros(2))
    data.force[..., 2] = 1.2
    self.assertTrue(torch.all(self._spin_cost(env) > 0))
    data.force.zero_()
    data.force[..., 0] = 1.2  # Contact force in any direction can activate the gate.
    self.assertTrue(torch.all(self._spin_cost(env) > 0))
    data.force.zero_()
    data.force[..., :2] = 0.8  # Threshold applies to vector magnitude.
    self.assertTrue(torch.all(self._spin_cost(env) > 0))
    # Historical contact cannot keep this reward active after detachment.
    data.force_history = torch.full((2, 32, 4, 3), 100.0)
    data.force.zero_()
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
