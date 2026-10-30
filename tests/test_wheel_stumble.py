"""Wheel-wall contact detection, force conventions and V2 reward wiring."""

from types import SimpleNamespace
import unittest

import torch
from mjlab.managers.reward_manager import RewardManager, RewardTermCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg

import src.tasks  # noqa: F401 -- registration must also tolerate the absent legacy reward.
from src.tasks.velocity.mdp.rewards.wheel import wheel_stumble


class WheelStumbleTests(unittest.TestCase):
  def make_env(self, force, found=None):
    force = torch.as_tensor(force, dtype=torch.float32)
    found = torch.ones(force.shape[:2]) if found is None else torch.as_tensor(found)
    return SimpleNamespace(
      scene={"contact": SimpleNamespace(data=SimpleNamespace(force=force, found=found))},
      num_envs=force.shape[0], device="cpu", max_episode_length_s=20., extras={"log": {}},
    )

  def cost(self, env, **params):
    return wheel_stumble(env, "contact", **params)

  def test_ordinary_driving_forces_and_airborne_wheels_are_free(self):
    env = self.make_env([[[30., 40., -100.], [-30., -40., -100.],
                          [0., 0., -100.], [0., 0., 0.]]])
    torch.testing.assert_close(self.cost(env), torch.zeros(1))

  def test_horizontal_wall_contacts_trigger_for_either_direction(self):
    env = self.make_env([[[30., 40., -5.], [0., 0., 0.]],
                         [[-30., -40., -5.], [0., 0., 0.]],
                         [[0., 20., 0.], [0., 0., 0.]]])
    torch.testing.assert_close(self.cost(env), torch.ones(3))

  def test_ratio_is_strict_and_configurable(self):
    env = self.make_env([[[5., 0., -1.]], [[5.01, 0., -1.]], [[3., 0., -1.]]])
    torch.testing.assert_close(self.cost(env), torch.tensor([0., 1., 0.]))
    torch.testing.assert_close(self.cost(env, force_ratio=2.), torch.ones(3))

  def test_stale_forces_and_tiny_contacts_do_not_trigger(self):
    env = self.make_env([[[100., 0., 0.], [.9, 0., 0.], [1., 0., 0.]]],
                        found=[[0., 1., 1.]])
    torch.testing.assert_close(self.cost(env), torch.zeros(1))
    torch.testing.assert_close(self.cost(env, min_horizontal_force=.5), torch.ones(1))

  def test_cost_is_per_environment_and_not_multiplied_by_wheel_count(self):
    env = self.make_env([[[0., 0., -50.]] * 4,
                         [[50., 0., -1.]] + [[0., 0., -50.]] * 3,
                         [[50., 0., -1.]] * 4])
    torch.testing.assert_close(self.cost(env), torch.tensor([0., 1., 1.]))
    torch.testing.assert_close(env.extras["log"]["Metrics/wheel_stumble_ratio"],
                               torch.tensor(2 / 3))
    torch.testing.assert_close(env.extras["log"]["Metrics/wheel_stumble_wheel_fraction"],
                               torch.tensor(5 / 12))

  def test_force_sign_and_left_right_reflection_leave_cost_unchanged(self):
    force = torch.tensor([[[30., 40., -5.], [0., 0., -50.],
                           [-3., 4., -10.], [0., 0., 0.]]])
    found = torch.tensor([[1., 1., 1., 0.]])
    original = self.cost(self.make_env(force, found))
    reflected = force[:, [1, 0, 3, 2]] * torch.tensor([1., -1., 1.])
    torch.testing.assert_close(self.cost(self.make_env(reflected, found[:, [1, 0, 3, 2]])),
                               original)
    torch.testing.assert_close(self.cost(self.make_env(-force, found)), original)

  def test_reward_manager_applies_negative_weight_and_control_duration(self):
    env = self.make_env([[[30., 40., -5.]], [[0., 0., -50.]]])
    manager = RewardManager({
      "wheel_stumble": RewardTermCfg(func=wheel_stumble, weight=-.5,
                                     params={"sensor_name": "contact"}),
    }, env)
    torch.testing.assert_close(manager.compute(.02), torch.tensor([-.01, 0.]))

  def test_v2_variants_use_the_existing_world_force_sensor(self):
    for task in list_tasks():
      if not task.startswith("Unitree-Go2W-"):
        continue
      for play in (False, True):
        with self.subTest(task=task, play=play):
          cfg = load_env_cfg(task, play=play)
          if not task.startswith("Unitree-Go2W-Perceptive-Rough-v2"):
            self.assertNotIn("wheel_stumble", cfg.rewards)
            continue
          term = cfg.rewards["wheel_stumble"]
          self.assertIs(term.func, wheel_stumble)
          self.assertEqual(term.weight, -.5)
          self.assertEqual(term.params, {
            "sensor_name": "feet_ground_contact", "force_ratio": 5.,
            "min_horizontal_force": 1.,
          })
          sensor = next(s for s in cfg.scene.sensors if s.name == term.params["sensor_name"])
          self.assertEqual(sensor.fields, ("found", "force"))
          self.assertEqual(sensor.reduce, "netforce")
          self.assertEqual(sensor.num_slots, 1)

  def test_invalid_parameters_and_incompatible_sensor_data_are_rejected(self):
    env = self.make_env([[[10., 0., -1.]]])
    for params in ({"force_ratio": 0.}, {"force_ratio": -1.},
                   {"min_horizontal_force": -1.}):
      with self.subTest(params=params), self.assertRaises(ValueError):
        self.cost(env, **params)
    data = env.scene["contact"].data
    for name in ("force", "found"):
      original = getattr(data, name)
      setattr(data, name, None)
      with self.subTest(missing=name), self.assertRaises(ValueError):
        self.cost(env)
      setattr(data, name, original)
    data.force = torch.zeros(1, 2, 3)
    with self.assertRaises(ValueError):
      self.cost(env)


if __name__ == "__main__":
  unittest.main()
