"""Sustained four-wheel support should beat occasional toe taps."""

from types import SimpleNamespace
import unittest

import torch
from mjlab.tasks.registry import list_tasks, load_env_cfg

import src.tasks  # noqa: F401 -- register task variants
from src.tasks.velocity.mdp.rewards.wheel import wheel_contact_bonus


class WheelContactBonusTests(unittest.TestCase):
  def make_env(self, vertical_force):
    vertical_force = torch.tensor(vertical_force, dtype=torch.float32)
    force = torch.zeros((*vertical_force.shape, 3))
    # Native MuJoCo contact force is wheel-on-terrain, opposite the support.
    force[..., 2] = -vertical_force
    return SimpleNamespace(
      scene={"contact": SimpleNamespace(data=SimpleNamespace(
        force=force, found=torch.ones_like(vertical_force),
      ))},
      extras={"log": {}},
    )

  def reward(self, env, **params):
    return wheel_contact_bonus(env, "contact", **params)

  def test_more_supporting_wheels_earn_more_reward_even_without_a_command(self):
    env = self.make_env([[0., 0., 0., 0.], [30., 30., 0., 0.],
                         [30., 30., 30., 0.], [30., 30., 30., 30.]])
    torch.testing.assert_close(self.reward(env), torch.tensor([0., .5, .75, 1.]))
    torch.testing.assert_close(env.extras["log"]["Metrics/all_wheels_grounded_ratio"],
                               torch.tensor(.25))

  def test_force_threshold_and_current_contact_filter_light_or_stale_contacts(self):
    env = self.make_env([[4.9, 5., 5.1, 40.]])
    env.scene["contact"].data.found[0, 3] = 0.
    torch.testing.assert_close(self.reward(env), torch.tensor([.25]))

  def test_wall_contact_and_downward_force_do_not_count_as_upward_support(self):
    env = self.make_env([[0., -40., 4., 20.]])
    env.scene["contact"].data.force[0, 0, :2] = torch.tensor([30., 40.])
    torch.testing.assert_close(self.reward(env), torch.tensor([.25]))

  def test_short_touchdowns_cannot_recover_reward_lost_during_suspension(self):
    # A one-second rollout at 50 Hz: three wheels support continuously; the
    # fourth supports for two isolated steps, so it contributes only 2/50.
    env = self.make_env(torch.full((50, 4), 30.).tolist())
    force = env.scene["contact"].data.force
    force[:, 3, 2] = 0.
    force[[10, 30], 3, 2] = -30.
    integrated = self.reward(env).sum() * .02
    torch.testing.assert_close(integrated, torch.tensor(.76))
    force[:, 3, 2] = -30.
    torch.testing.assert_close(self.reward(env).sum() * .02, torch.tensor(1.))

  def test_rejects_invalid_force_or_contact_data(self):
    env = self.make_env([[30., 30., 30., 30.]])
    with self.assertRaises(ValueError):
      self.reward(env, force_threshold=-1.)
    data = env.scene["contact"].data
    for field in ("force", "found"):
      original = getattr(data, field)
      setattr(data, field, None)
      with self.subTest(field=field), self.assertRaises(ValueError):
        self.reward(env)
      setattr(data, field, original)
    data.force = torch.zeros(1, 3, 3)
    with self.assertRaises(ValueError):
      self.reward(env)

  def test_all_go2w_variants_reward_support_using_world_net_force(self):
    for task in list_tasks():
      if not task.startswith("Unitree-Go2W-"):
        continue
      for play in (False, True):
        with self.subTest(task=task, play=play):
          cfg = load_env_cfg(task, play=play)
          term = cfg.rewards["wheel_contact_bonus"]
          self.assertIs(term.func, wheel_contact_bonus)
          self.assertEqual(term.weight, 1.)
          self.assertEqual(term.params, {"sensor_name": "feet_ground_contact", "force_threshold": 5.})
          sensor = next(sensor for sensor in cfg.scene.sensors
                        if sensor.name == term.params["sensor_name"])
          self.assertEqual(sensor.reduce, "netforce")
          self.assertEqual(sensor.num_slots, 1)
          self.assertEqual(sensor.fields, ("found", "force"))
          self.assertEqual(len(sensor.primary.pattern), 4)


if __name__ == "__main__":
  unittest.main()
