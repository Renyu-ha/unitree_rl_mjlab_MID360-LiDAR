"""Explicit name order must survive native model ordering and exclusions."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch
from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch
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
    selection = SceneEntityCfg("robot", joint_names=("FR_calf_joint", "FL_hip_joint"), preserve_order=True)
    selection.resolve({"robot": self.asset})
    q = torch.zeros(1, 16)
    calf, hip = selection.joint_ids
    q[0, calf], q[0, hip] = .4, .03
    asset = SimpleNamespace(find_joints=self.asset.find_joints,
      data=SimpleNamespace(default_joint_pos=torch.zeros_like(q), joint_pos=q))
    env = SimpleNamespace(scene={"robot": asset}, device="cpu",
      command_manager=SimpleNamespace(get_command=lambda _: torch.zeros(1, 3)))
    std = {r".*_hip_joint": .1, r".*_calf_joint": .5}
    cfg = RewardTermCfg(func=variable_posture, weight=1., params={
      "asset_cfg": selection, "command_name": "twist",
      "std_standing": std, "std_walking": std, "std_running": std,
    })
    reward = variable_posture(cfg, env)
    torch.testing.assert_close(reward.std_standing, torch.tensor([.5, .1]))
    expected = torch.exp(torch.tensor([-((.4 / .5) ** 2 + (.03 / .1) ** 2) / 2]))
    torch.testing.assert_close(reward(env, **cfg.params), expected)

  def test_all_go2w_actions_sensors_and_explicit_entity_selections_preserve_order(self):
    for task in list_tasks():
      if not task.startswith("Unitree-Go2W-"):
        continue
      for play in (False, True):
        cfg = load_env_cfg(task, play=play)
        with self.subTest(task=task, play=play):
          self.assertTrue(all(term.preserve_order for term in cfg.actions.values()))
          self.assertTrue(all(sensor.preserve_order for sensor in cfg.scene.sensors
            if isinstance(sensor, mdp.OrderedContactSensorCfg)))
          groups = list(cfg.observations.values())
          terms = [term for group in groups for term in group.terms.values()]
          terms += list(cfg.rewards.values()) + list(cfg.events.values()) + list(cfg.terminations.values())
          for term in terms:
            for selection in term.params.values():
              if isinstance(selection, SceneEntityCfg):
                self.assertTrue(selection.preserve_order, (task, selection))


if __name__ == "__main__":
  unittest.main()
