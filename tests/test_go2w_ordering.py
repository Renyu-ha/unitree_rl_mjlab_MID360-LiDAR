"""Explicit name order must survive native model ordering and exclusions."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch
from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg

import src.tasks
from src.assets.robots.unitree_go2w.go2w_constants import get_go2w_robot_cfg, WHEEL_GEOMS
from src.tasks.velocity import mdp
from src.tasks.velocity.mdp.rewards.locomotion import variable_posture


class Go2WOrderingTests(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.asset = Entity(get_go2w_robot_cfg())

  def test_joint_actions_honor_reversed_query_order_and_write_matching_ids(self):
    requested = ("FR_hip_joint", "FL_hip_joint")
    for cfg_type, setter_name in (
      (mdp.OrderedJointPositionActionCfg, "set_joint_position_target"),
      (mdp.OrderedJointVelocityActionCfg, "set_joint_velocity_target"),
    ):
      for preserve in (True, False):
        with self.subTest(action=cfg_type.__name__, preserve=preserve):
          setter = Mock()
          asset = SimpleNamespace(
            find_joints_by_actuator_names=self.asset.find_joints_by_actuator_names,
            data=SimpleNamespace(encoder_bias=torch.zeros(1, 16)),
            **{setter_name: setter},
          )
          env = SimpleNamespace(scene={"robot": asset}, device="cpu", num_envs=1)
          cfg = cfg_type(entity_name="robot", actuator_names=requested,
            preserve_order=preserve, use_default_offset=False)
          action = cfg.build(env)
          expected_ids, expected_names = self.asset.find_joints(requested, preserve_order=preserve)
          self.assertEqual(action.target_names, expected_names)
          values = torch.tensor([[.2, .7]])
          action.process_actions(values)
          action.apply_actions()
          torch.testing.assert_close(setter.call_args.args[0], values)
          self.assertEqual(setter.call_args.kwargs["joint_ids"].tolist(), expected_ids)

  def test_contact_sensor_honors_query_order_after_native_exclusions(self):
    requested = tuple(reversed(WHEEL_GEOMS))
    for preserve in (True, False):
      cfg = mdp.OrderedContactSensorCfg(name="test_contact", preserve_order=preserve,
        primary=ContactMatch(mode="geom", entity="robot", pattern=requested, exclude=(WHEEL_GEOMS[2],)))
      sensor = cfg.build()
      names = sensor._resolve_primary_names({"robot": self.asset}, cfg.primary)
      expected = [name for name in (requested if preserve else WHEEL_GEOMS) if name != WHEEL_GEOMS[2]]
      self.assertEqual(names, expected)

  def test_posture_tolerances_follow_the_selected_joint_order(self):
    q = torch.zeros(1, 16)
    calf = self.asset.joint_names.index("FR_calf_joint")
    hip = self.asset.joint_names.index("FL_hip_joint")
    q[0, calf], q[0, hip] = .4, .03
    asset = SimpleNamespace(find_joints=self.asset.find_joints,
      data=SimpleNamespace(default_joint_pos=torch.zeros_like(q), joint_pos=q))
    env = SimpleNamespace(scene={"robot": asset}, device="cpu",
      command_manager=SimpleNamespace(get_command=lambda _: torch.zeros(1, 3)))
    std = {r".*_hip_joint": .1, r".*_calf_joint": .5}
    for preserve in (True, False):
      with self.subTest(preserve=preserve):
        selection = SceneEntityCfg("robot", joint_names=("FR_calf_joint", "FL_hip_joint"),
                                   preserve_order=preserve)
        selection.resolve({"robot": self.asset})
        cfg = RewardTermCfg(func=variable_posture, weight=1., params={
          "asset_cfg": selection, "command_name": "twist",
          "std_standing": std, "std_walking": std, "std_running": std,
        })
        reward = variable_posture(cfg, env)
        torch.testing.assert_close(reward.std_standing, torch.tensor([.5, .1] if preserve else [.1, .5]))
        expected = torch.exp(torch.tensor([-((.4 / .5) ** 2 + (.03 / .1) ** 2) / 2]))
        torch.testing.assert_close(reward(env, **cfg.params), expected)

  def test_reduced_joint_costs_are_invariant_to_selection_order(self):
    cfg = load_env_cfg("Unitree-Go2W-Perceptive-Rough-v2")
    values = torch.arange(32., dtype=torch.float32).reshape(2, 16) * .07
    data = SimpleNamespace(joint_pos=values, default_joint_pos=values * .1,
      joint_vel=values * 2., joint_acc=values * 3., actuator_force=values * 4.,
      soft_joint_pos_limits=torch.stack((torch.full_like(values, -.5), torch.full_like(values, .5)), dim=-1))
    env = SimpleNamespace(scene={"robot": SimpleNamespace(data=data)},
      command_manager=SimpleNamespace(get_command=lambda _: torch.zeros(2, 3)))
    for name in ("leg_torques_l2", "wheel_torques_l2", "joint_vel_l2", "joint_acc_l2",
                 "joint_pos_limits", "wheel_speed", "wheel_acc_l2", "stand_still", "wheel_stand_still"):
      with self.subTest(reward=name):
        term = cfg.rewards[name]
        selection = term.params["asset_cfg"]
        selection.resolve({"robot": self.asset})
        field = "actuator_names" if "torques" in name else "joint_names"
        reverse = SceneEntityCfg("robot", **{field: tuple(reversed(getattr(selection, field)))},
                                 preserve_order=True)
        reverse.resolve({"robot": self.asset})
        params = {**term.params, "asset_cfg": reverse}
        torch.testing.assert_close(term.func(env, **term.params), term.func(env, **params))

  def test_go2w_preserves_order_only_for_feature_columns_and_cross_wheel_pairs(self):
    for task in list_tasks():
      if not task.startswith("Unitree-Go2W-"):
        continue
      for play in (False, True):
        cfg = load_env_cfg(task, play=play)
        with self.subTest(task=task, play=play):
          self.assertTrue(all(term.preserve_order for term in cfg.actions.values()))
          for sensor in cfg.scene.sensors:
            if isinstance(sensor, ContactSensorCfg):
              if sensor.name == "feet_ground_contact":
                self.assertIsInstance(sensor, mdp.OrderedContactSensorCfg)
                self.assertTrue(sensor.preserve_order)
              else:
                self.assertNotIsInstance(sensor, mdp.OrderedContactSensorCfg)
          terms = [("observation", name, term) for group in cfg.observations.values()
                   for name, term in group.terms.items()]
          terms += [("reward", name, term) for name, term in cfg.rewards.items()]
          terms += [("other", name, term) for group in (cfg.events, cfg.terminations, cfg.curriculum)
                    for name, term in group.items()]
          for scope, name, term in terms:
            ordered = scope == "observation" or (
              scope == "reward" and name in ("wheel_lateral_slide", "wheel_spin_contact"))
            for selection in term.params.values():
              if isinstance(selection, SceneEntityCfg):
                self.assertEqual(selection.preserve_order, ordered, (task, scope, name, selection))


if __name__ == "__main__":
  unittest.main()
