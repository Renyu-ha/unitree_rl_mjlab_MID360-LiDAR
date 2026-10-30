"""Check reflection against real model indices and mjlab observation/reward managers."""

from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.reward_manager import RewardManager
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz

import src.tasks  # noqa: F401
from src.assets.robots.unitree_go2w import go2w_constants as robot
from src.tasks.velocity.mdp.observations import bridged_mid360_height_scan_memory
from src.tasks.velocity.rl.go2w_symmetry import Go2WSymmetry


LEGS = ("FL", "FR", "RL", "RR")
FOOT_PERM = [1, 0, 3, 2]
POLAR = torch.tensor([1., -1., 1.])
AXIAL = -POLAR


def reflected_joint_names(names):
  other = dict(zip(LEGS, ("FR", "FL", "RR", "RL"), strict=True))
  return [other[name[:2]] + name[2:] for name in names]


class Go2WMirrorRuntimeTests(unittest.TestCase):
  def test_scan_memory_reflection_survives_dropout_forward_shift_and_expiry(self):
    function = bridged_mid360_height_scan_memory()
    grid = torch.arange(187.).reshape(11, 17) * .002
    scan = torch.stack((grid.flatten(), grid.flip(0).flatten()))
    velocity = torch.tensor([[2., .3, 0.], [2., -.3, 0.]])
    env = SimpleNamespace(num_envs=2, device="cpu", step_dt=.1, extras={},
                          scene={"robot": SimpleNamespace(data=SimpleNamespace(root_link_lin_vel_b=velocity))})
    params = deepcopy(load_env_cfg("Unitree-Go2W-Perceptive-Rough-v2", play=True)
                      .observations["actor"].terms["height_scan"].params)
    params.update(visible_x_min_m=-10., visible_x_max_m=10.,
                  memory_max_age_s=.25, keep_prob_min=1., keep_prob_max=1.)
    with patch("src.tasks.velocity.mdp.observations.envs_mdp.height_scan", return_value=scan):
      first = function(env, **params)
      torch.testing.assert_close(first, scan)
      params.update(keep_prob_min=0., keep_prob_max=0.)
      second = function(env, **params)
      expected = grid.clone()
      expected[:, :-2] = grid[:, 2:]
      expected[:, -2:] = 5.
      torch.testing.assert_close(second[0].reshape(11, 17), expected)
      torch.testing.assert_close(second[1].reshape(11, 17), expected.flip(0))
      # Reverse motion currently leaves X memory stationary, but reflection must
      # still be valid. This deliberately records the existing forward-only model.
      velocity[:, 0] = -2.
      third = function(env, **params)
      torch.testing.assert_close(third, second)
      expired = function(env, **params)
      torch.testing.assert_close(expired, torch.full_like(scan, 5.))

  def test_every_resolved_selection_maps_names_to_matching_model_indices(self):
    for xml in (robot.GO2W_CYLINDER_XML, robot.GO2W_MESH_XML):
      with patch.object(robot, "GO2W_XML", xml):
        asset = Entity(robot.get_go2w_robot_cfg())
      for task in list_tasks():
        if not task.startswith("Unitree-Go2W-"):
          continue
        for play in (False, True):
          cfg = load_env_cfg(task, play=play)
          groups = [term for group in cfg.observations.values() for term in group.terms.values()]
          groups += list(cfg.rewards.values()) + list(cfg.events.values())
          groups += list(cfg.terminations.values()) + list(cfg.curriculum.values())
          with self.subTest(model=xml.name, task=task, play=play):
            for term in groups:
              for selection in term.params.values():
                if not isinstance(selection, SceneEntityCfg):
                  continue
                selection.resolve({"robot": asset})
                for kind in ("joint", "actuator", "body", "site", "geom"):
                  names = getattr(selection, f"{kind}_names")
                  if names is None:
                    continue
                  ids = getattr(selection, f"{kind}_ids")
                  actual = getattr(asset, f"{kind}_names")
                  actual = actual[ids] if isinstance(ids, slice) else [actual[i] for i in ids]
                  self.assertEqual(list(actual), list(names))
            for group in cfg.observations.values():
              for name, expected in (("joint_pos", robot.LEG_JOINTS),
                                     ("joint_vel", robot.LEG_JOINTS),
                                     ("wheel_vel", robot.WHEEL_JOINTS)):
                self.assertEqual(group.terms[name].params["asset_cfg"].joint_names, list(expected))
            sensor_cfg = next(s for s in cfg.scene.sensors if s.name == "feet_ground_contact")
            primaries = sensor_cfg.build()._resolve_primary_names({"robot": asset}, sensor_cfg.primary)
            self.assertEqual(primaries, list(robot.WHEEL_GEOMS))
            for name in ("wheel_lateral_slide", "wheel_spin_contact"):
              self.assertEqual(cfg.rewards[name].params["body_cfg"].body_names, list(robot.WHEEL_BODIES))
            self.assertEqual(cfg.rewards["wheel_spin_contact"].params["joint_cfg"].joint_names,
                             list(robot.WHEEL_JOINTS))

  def test_physical_reflection_matches_actual_actions_observation_history_and_every_reward(self):
    self._check_task_reflection("Unitree-Go2W-Perceptive-Rough-v2-StepUp-Finetune")

  def test_gait_physical_reflection_matches_shifted_phase_and_every_restored_reward(self):
    self._check_task_reflection("Unitree-Go2W-Perceptive-Rough-v2-Gait")

  def _check_task_reflection(self, task):
    for xml in (robot.GO2W_CYLINDER_XML, robot.GO2W_MESH_XML):
      with self.subTest(model=xml.name), patch.object(robot, "GO2W_XML", xml):
        cfg = load_env_cfg(task)
        cfg.seed, cfg.scene.num_envs = 7, 2
        # Isolate the coordinate mapping from random terrain/DR and sensor corruption.
        cfg.scene.terrain.terrain_type = "plane"
        cfg.scene.terrain.terrain_generator = None
        cfg.curriculum = {}
        cfg.events = {}
        cfg.observations["actor"].enable_corruption = False
        scan = cfg.observations["actor"].terms["height_scan"]
        scan.delay_max_lag = 0
        scan.params.update(keep_prob_min=1., keep_prob_max=1., frame_dropout_prob=0.,
                           fixed_curriculum_step=6000 * 24)
        env = ManagerBasedRlEnv(cfg, device="cpu")
        try:
          self._check_physical_pair(env)
        finally:
          env.close()

  def _check_physical_pair(self, env):
    asset = env.scene["robot"]
    names = list(asset.joint_names)
    perm = [names.index(name) for name in reflected_joint_names(names)]
    sign = torch.tensor([-1. if "_hip_joint" in name else 1. for name in names])
    mirror = Go2WSymmetry(env)
    env.reset()
    ids = torch.arange(2)
    env.observation_manager.reset(ids)
    defaults = asset.data.default_joint_pos
    torch.testing.assert_close(defaults[1], defaults[0, perm] * sign)
    actuator_names = list(asset.actuator_names)
    actuator_perm = [actuator_names.index(name) for name in reflected_joint_names(actuator_names)]
    actuator_sign = torch.tensor([-1. if "_hip_joint" in n else 1. for n in actuator_names])
    ctrl_ids = asset.indexing.ctrl_ids.tolist()
    for field in ("actuator_gainprm", "actuator_biasprm", "actuator_forcerange", "actuator_gear"):
      parameters = torch.tensor(getattr(env.sim.mj_model, field)[ctrl_ids])
      torch.testing.assert_close(parameters, parameters[actuator_perm])
    limits = asset.data.soft_joint_pos_limits[0]
    reflected_limits = (limits[perm] * sign[:, None]).sort(dim=-1).values
    torch.testing.assert_close(limits, reflected_limits)
    all_reward_cfgs = deepcopy(env.cfg.rewards)
    rewards = RewardManager(all_reward_cfgs, env)
    nonzero_rewards = set()
    for frame in range(3):
      angles = torch.tensor([[.13, -.21, .72 + frame * .1], [-.13, -.21, -.72 - frame * .1]])
      pose = torch.cat((torch.tensor([[.2, .3, .8], [.2, -.3, .8]]),
                        quat_from_euler_xyz(*angles.T)), dim=1)
      velocity = torch.tensor([[.08, -.03, .014, .31, -.17, .21]])
      velocity = torch.cat((velocity, velocity * torch.cat((POLAR, AXIAL))), dim=0)
      q = defaults[0] + torch.linspace(.015, .15, 16) * (frame + 1)
      if frame == 2:
        q[names.index("FL_thigh_joint")] = 3.6  # Also exercise a nonzero position-limit cost.
      qvel = torch.linspace(-3., 7., 16) * (frame + 1)
      qvel[[names.index(name) for name in robot.WHEEL_JOINTS]] = torch.tensor([13., 29., 47., 71.])
      asset.write_root_link_pose_to_sim(pose)
      asset.write_root_link_velocity_to_sim(velocity)
      asset.write_joint_state_to_sim(torch.stack((q, q[perm] * sign)),
                                    torch.stack((qvel, qvel[perm] * sign)))
      action = torch.linspace(-.3, .4, 16).unsqueeze(0) * (frame + 1)
      actions = torch.cat((action, mirror.mirror("actions", action)))
      bias = torch.linspace(-.02, .02, 16)
      asset.data.encoder_bias[:] = torch.stack((bias, bias[perm] * sign))
      env.action_manager.process_action(actions)
      env.action_manager.apply_action()
      env.scene.write_data_to_sim()
      # Check values in the real MuJoCo actuator slots, including default pose and bias.
      for term_name in ("joint_pos", "wheel_vel"):
        term = env.action_manager.get_term(term_name)
        for column, name in enumerate(term.target_names):
          local_joint = names.index(name)
          local_actuator = actuator_names.index(name)
          ctrl_id = asset.indexing.ctrl_ids[local_actuator]
          if term_name == "joint_pos":
            expected = (actions[:, column] * robot.LEG_ACTION_SCALE + defaults[:, local_joint]
                        - asset.data.encoder_bias[:, local_joint])
          else:
            expected = actions[:, 12 + column] * robot.WHEEL_ACTION_SCALE
          torch.testing.assert_close(env.sim.data.ctrl[:, ctrl_id], expected)
      controls = env.sim.data.ctrl[:, ctrl_ids]
      torch.testing.assert_close(controls[1], controls[0, actuator_perm] * actuator_sign)
      env.sim.forward()
      env.sim.sense()
      # Inject distinct per-wheel values at actual compiled sensor addresses. This
      # tests extraction/ordering even when there are no physical contacts in air.
      # Native contact sensors report the wheel-on-terrain force (negative Z).
      force = torch.tensor([[.3, .6, -3.], [0., 0., 0.], [-.2, .8, -5.], [1., -.4, -8.]])
      if frame == 2:
        # A horizontal wheel-wall contact must also preserve the stumble cost.
        force[0] = torch.tensor([30., 40., -5.])
      found = torch.tensor(([0., 0., 1., 1.], [1., 0., 1., 1.], [1., 0., 0., 1.])[frame])
      force = force * found[:, None]
      forces = torch.stack((force, force[FOOT_PERM] * POLAR))
      found = torch.stack((found, found[FOOT_PERM]))
      contact = env.scene["feet_ground_contact"]
      for slot in contact._slots:
        foot = LEGS.index(slot.primary_name[:2])
        value = forces[:, foot] if slot.field_name == "force" else found[:, foot, None]
        compiled = env.sim.mj_model.sensor(slot.sensor_name)
        start, width = int(compiled.adr[0]), int(compiled.dim[0])
        env.sim.data.sensordata[:, start:start + width] = value
      for slot in env.scene["nonfoot_ground_touch"]._slots:
        # A brief body-ground hit must be detected through the real history buffer.
        hit = float(frame == 1 and slot.primary_name == "base_link_collision0")
        test_force = env.cfg.rewards["undesired_contact"].params["force_threshold"] + 1.
        value = torch.tensor([[test_force, 0., 0.]]).repeat(2, 1) * hit if slot.field_name == "force" else hit
        compiled = env.sim.mj_model.sensor(slot.sensor_name)
        start, width = int(compiled.adr[0]), int(compiled.dim[0])
        env.sim.data.sensordata[:, start:start + width] = value
      for slot in env.scene["self_collision"]._slots:
        value = torch.tensor([[11., 12., 13.]]).repeat(2, 1) if slot.field_name == "force" else 1.
        compiled = env.sim.mj_model.sensor(slot.sensor_name)
        start, width = int(compiled.adr[0]), int(compiled.dim[0])
        env.sim.data.sensordata[:, start:start + width] = value
      if frame == 0:
        # Start with two already-airborne wheels so this short reflection check
        # also exercises a nonzero timeout, including the zero-command frame.
        air_time = torch.tensor([.8, .9, 0., 0.])
        contact.data.current_air_time[:] = torch.stack((air_time, air_time[FOOT_PERM]))
      env.scene.update(env.step_dt)
      torch.testing.assert_close(contact.data.force, forces)
      torch.testing.assert_close(contact.data.found, found)
      for sensor_name in ("terrain_scan", "terrain_scan_mid360"):
        data = env.scene[sensor_name].data
        grid = torch.arange(187.).reshape(11, 17) * .002 + frame * .01
        if frame == 1:
          grid[:, ::2] += .14  # Exercise landing near a terrain edge.
        data.distances.fill_(1.)
        data.hit_pos_w[..., 2] = torch.stack((grid.flatten(), grid.flip(0).flatten()))
      # Acceleration and torque marker values expose joint/actuator ID confusion.
      acc = torch.linspace(10., 25., 16) * (frame + 1)
      env.sim.data.qacc[:, asset.indexing.joint_v_adr] = torch.stack((acc, acc[perm] * sign))
      torque = torch.linspace(-2., 6., 16) * (frame + 1)
      env.sim.data.actuator_force[:, asset.indexing.ctrl_ids] = torch.stack(
        (torque, torque[actuator_perm] * actuator_sign))
      command = torch.tensor([[.4, .02 if frame == 1 else 0., .03 if frame == 1 else 0.]])
      if frame == 0:
        command.zero_()
      env.command_manager.get_term("twist").vel_command_b[:] = torch.cat(
        (command, command * torch.tensor([1., -1., -1.])))
      env.episode_length_buf[:] = frame + 1
      if "foot_gait" in env.cfg.rewards:
        period = env.cfg.rewards["foot_gait"].params["period"]
        env.episode_length_buf[1] += round(period / (2 * env.step_dt))
      env.termination_manager.terminated[:] = frame == 2
      observations = env.observation_manager.compute(update_history=True)
      for group, values in observations.items():
        torch.testing.assert_close(mirror.mirror(group, values[:1]), values[1:], atol=2e-5, rtol=2e-5,
                                   msg=lambda msg: f"{group}, frame {frame}: {msg}")
      for name in rewards.active_terms:
        term = rewards.get_term_cfg(name)
        value = term.func(env, **term.params)
        self.assertEqual(tuple(value.shape), (2,), name)
        self.assertTrue(torch.isfinite(value).all(), name)
        if torch.any(value != 0):
          nonzero_rewards.add(name)
        if name == "angular_momentum":
          # The supplied wheel inertial tensors are not exact left/right mirrors.
          # Check the real sensor calculation; algebraic reflection is tested separately.
          raw = env.scene[term.params["sensor_name"]].data
          torch.testing.assert_close(value, raw.square().sum(-1))
          print("GO2W_GAIT_ANGMOM_ASSET_DIFFERENCE", frame, value.tolist(), flush=True)
          continue
        torch.testing.assert_close(value[:1], value[1:], atol=2e-5, rtol=2e-5,
                                   msg=lambda msg: f"{name}, frame {frame}: {msg}")
    self.assertEqual(nonzero_rewards, set(rewards.active_terms), "Every reward must be exercised")


if __name__ == "__main__":
  unittest.main()
