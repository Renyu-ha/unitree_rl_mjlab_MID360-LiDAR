"""Long suspension must incur a cost before the wheel lands."""

from types import SimpleNamespace
import unittest

import torch
from mjlab.tasks.registry import list_tasks, load_env_cfg

import src.tasks  # noqa: F401 -- register task variants
from src.tasks.velocity.mdp.rewards.wheel import wheel_air_time_timeout


class WheelAirTimeTimeoutTests(unittest.TestCase):
  def make_env(self, air_time):
    return SimpleNamespace(
      scene={"contact": SimpleNamespace(data=SimpleNamespace(
        current_air_time=torch.tensor(air_time, dtype=torch.float32),
      ))},
      extras={"log": {}},
    )

  def cost(self, env, **params):
    return wheel_air_time_timeout(env, "contact", **params)

  def test_contact_and_normal_short_swing_are_free(self):
    env = self.make_env([[0., 0., 0., 0.], [.1, .3, .49, .5]])
    torch.testing.assert_close(self.cost(env), torch.zeros(2))

  def test_each_airborne_wheel_contributes_and_total_cost_is_bounded(self):
    env = self.make_env([
      [.625, 0., 0., 0.], [.75, 0., 0., 0.],
      [.75, .75, 0., 0.], [10., 10., 10., 10.],
    ])
    torch.testing.assert_close(self.cost(env), torch.tensor([.125, .25, .5, 1.]))

  def test_cost_continues_without_a_landing_or_command(self):
    # This environment intentionally has no landing-event API or command manager.
    env = self.make_env([[.6, 0., 0., 0.]])
    air_time = env.scene["contact"].data.current_air_time
    torch.testing.assert_close(self.cost(env), torch.tensor([.1]))
    air_time[0, 0] = .7
    torch.testing.assert_close(self.cost(env), torch.tensor([.2]))
    air_time[0, 0] = 1.5
    torch.testing.assert_close(self.cost(env), torch.tensor([.25]))
    torch.testing.assert_close(self.cost(env), torch.tensor([.25]))

  def test_landing_or_partial_episode_reset_clears_only_the_affected_timer(self):
    env = self.make_env([[1., 0., 0., 0.], [0., 0., 0., 1.]])
    torch.testing.assert_close(self.cost(env), torch.tensor([.25, .25]))
    env.scene["contact"].data.current_air_time[0].zero_()
    torch.testing.assert_close(self.cost(env), torch.tensor([0., .25]))
    env.scene["contact"].data.current_air_time[1].zero_()
    torch.testing.assert_close(self.cost(env), torch.zeros(2))

  def test_longer_stepup_grace_time_allows_a_normal_step(self):
    env = self.make_env([[.6, .7, .825, .95]])
    torch.testing.assert_close(self.cost(env, threshold=.7), torch.tensor([.375]))

  def test_logs_longest_current_air_time_averaged_over_environments(self):
    env = self.make_env([[.1, .2, .3, .4], [.9, .2, 0., 0.]])
    self.cost(env)
    torch.testing.assert_close(
      env.extras["log"]["Metrics/wheel_air_time_max_mean"], torch.tensor(.65),
    )

  def test_requires_air_time_tracking_and_valid_timing_parameters(self):
    env = self.make_env([[0., 0., 0., 0.]])
    for params in ({"threshold": -.1}, {"ramp_time": 0.}, {"ramp_time": -.1}):
      with self.subTest(params=params), self.assertRaises(ValueError):
        self.cost(env, **params)
    env.scene["contact"].data.current_air_time = None
    with self.assertRaisesRegex(ValueError, "track_air_time=True"):
      self.cost(env)

  def test_all_go2w_train_and_play_variants_enable_timeout(self):
    for task in list_tasks():
      if not task.startswith("Unitree-Go2W-"):
        continue
      for play in (False, True):
        with self.subTest(task=task, play=play):
          cfg = load_env_cfg(task, play=play)
          if task.endswith("-Gait"):
            self.assertNotIn("wheel_air_time_timeout", cfg.rewards)
            continue
          term = cfg.rewards["wheel_air_time_timeout"]
          self.assertIs(term.func, wheel_air_time_timeout)
          self.assertEqual(term.weight, -.5)
          self.assertEqual(term.params["ramp_time"], .25)
          self.assertEqual(term.params["threshold"], .7 if "StepUp" in task else .5)
          self.assertEqual(term.params["sensor_name"], "feet_ground_contact")
          sensor = next(sensor for sensor in cfg.scene.sensors
                        if sensor.name == term.params["sensor_name"])
          self.assertTrue(sensor.track_air_time)


if __name__ == "__main__":
  unittest.main()
