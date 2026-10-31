"""Command-directed anti-stall costs must respect slow and omnidirectional commands."""

from types import SimpleNamespace
import unittest

import torch
from mjlab.managers.reward_manager import RewardManager
from mjlab.tasks.registry import list_tasks, load_env_cfg

import src.tasks  # noqa: F401 -- register task variants
from src.tasks.velocity.mdp.rewards.locomotion import dont_wait


class DontWaitTests(unittest.TestCase):
  @staticmethod
  def make_env(commands, velocities):
    commands = torch.tensor(commands, dtype=torch.float32)
    velocities = torch.tensor(velocities, dtype=torch.float32)
    return SimpleNamespace(
      num_envs=len(commands), device="cpu",
      command_manager=SimpleNamespace(get_command=lambda name: commands),
      scene={"robot": SimpleNamespace(data=SimpleNamespace(root_link_lin_vel_b=velocities))},
    )

  def test_low_speed_standing_and_pure_yaw_commands_are_not_penalized(self):
    commands = [[0., 0., 0.], [0., 0., .8], [.1, 0., 0.], [.15, 0., 0.],
                [.2, 0., 0.], [-.2, 0., 0.], [0., .2, 0.], [.1, .1, .8]]
    env = self.make_env(commands, [[-1., -1., 0.]] * len(commands))
    torch.testing.assert_close(dont_wait(env, "twist"), torch.zeros(len(commands)))

  def test_forward_tiers_include_exact_speed_boundaries(self):
    speeds = [.3, .15, .149, 0., -.001, -.15, -.151]
    env = self.make_env([[.4, 0., 0.]] * len(speeds), [[v, 0., 0.] for v in speeds])
    torch.testing.assert_close(dont_wait(env, "twist"), torch.tensor([0., 0., 1., 1., 2., 2., 3.]))

  def test_same_tiers_apply_to_forward_reverse_lateral_and_diagonal_motion(self):
    directions = [(1., 0.), (-1., 0.), (0., 1.), (0., -1.),
                  (.6, .8), (-.6, .8), (.6, -.8), (-.6, -.8)]
    commands, velocities, expected = [], [], []
    for dx, dy in directions:
      for speed, cost in ((.3, 0.), (.1, 1.), (0., 1.), (-.1, 2.), (-.3, 3.)):
        commands.append([.5 * dx, .5 * dy, .4])
        velocities.append([speed * dx, speed * dy, 0.])
        expected.append(cost)
    env = self.make_env(commands, velocities)
    torch.testing.assert_close(dont_wait(env, "twist"), torch.tensor(expected))

  def test_large_perpendicular_and_vertical_velocities_do_not_count_as_progress(self):
    env = self.make_env(
      [[.4, 0., 0.], [0., -.4, 0.], [.4, 0., 0.]],
      [[0., 2., 0.], [2., 0., 0.], [0., 0., 2.]],
    )
    torch.testing.assert_close(dont_wait(env, "twist"), torch.ones(3))

  def test_exact_command_tracking_never_incurs_the_fixed_floor_conflict(self):
    commands = [[.1, 0., 0.], [.15, 0., 0.], [.21, 0., 0.], [-.21, 0., 0.],
                [0., .21, 0.], [.15, .15, .4], [-.3, -.4, -.5]]
    env = self.make_env(commands, [[x, y, 0.] for x, y, _ in commands])
    torch.testing.assert_close(dont_wait(env, "twist"), torch.zeros(len(commands)))

  def test_configured_reward_manager_applies_negative_weight_and_control_dt(self):
    cfg = load_env_cfg("Unitree-Go2W-Perceptive-Rough-v2")
    env = self.make_env([[.4, 0., 0.]] * 4, [[.3, 0., 0.], [0., 0., 0.],
                                               [-.1, 0., 0.], [-.3, 0., 0.]])
    manager = RewardManager({"dont_wait": cfg.rewards["dont_wait"]}, env)
    torch.testing.assert_close(manager.compute(dt=.02), torch.tensor([0., -.002, -.004, -.006]))


class Go2WDontWaitConfigTests(unittest.TestCase):
  def test_all_v2_variants_replace_the_floor_in_train_and_play(self):
    for task in list_tasks():
      if not task.startswith("Unitree-Go2W-Perceptive-Rough-v2"):
        continue
      for play in (False, True):
        with self.subTest(task=task, play=play):
          cfg = load_env_cfg(task, play=play)
          self.assertNotIn("forward_velocity_floor", cfg.rewards)
          reward = cfg.rewards["dont_wait"]
          self.assertIs(reward.func, dont_wait)
          self.assertEqual(reward.weight, -.1)
          self.assertEqual(reward.params, {
            "command_name": "twist", "command_threshold": .2, "slow_velocity_threshold": .15,
          })
          self.assertIn("track_linear_velocity", cfg.rewards)
          self.assertEqual(cfg.observations["actor"].history_length, 3)

  def test_other_go2w_stages_and_g1_stepup_also_replace_the_floor(self):
    for task in ("Unitree-Go2W-Flat", "Unitree-Go2W-Rough", "Unitree-Go2W-Perceptive-Rough-v1",
                 "Unitree-G1-Perceptive-Rough-v2-StepUp-Finetune"):
      for play in (False, True):
        with self.subTest(task=task, play=play):
          rewards = load_env_cfg(task, play=play).rewards
          self.assertNotIn("forward_velocity_floor", rewards)
          self.assertIs(rewards["dont_wait"].func, dont_wait)
          self.assertEqual(rewards["dont_wait"].weight, -.1)
          self.assertEqual(rewards["dont_wait"].params, {
            "command_name": "twist", "command_threshold": .2, "slow_velocity_threshold": .15,
          })


if __name__ == "__main__":
  unittest.main()
