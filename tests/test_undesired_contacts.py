"""Non-wheel terrain contact costs and substep-history coverage."""

import re
from types import SimpleNamespace
import unittest

import torch
from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardManager, RewardTermCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg

import src.tasks  # noqa: F401
from src.assets.robots.unitree_go2w.go2w_constants import WHEEL_GEOMS, get_go2w_robot_cfg
from src.tasks.velocity.mdp.rewards.locomotion import undesired_contacts


class UndesiredContactsTests(unittest.TestCase):
  def make_env(self, force, history=None):
    force = torch.as_tensor(force, dtype=torch.float32)
    history = None if history is None else torch.as_tensor(history, dtype=torch.float32)
    return SimpleNamespace(
      scene={"nonfoot": SimpleNamespace(data=SimpleNamespace(force=force, force_history=history))},
      num_envs=force.shape[0], device="cpu", max_episode_length_s=20., extras={"log": {}},
    )

  def cost(self, env, **params):
    return undesired_contacts(env, "nonfoot", **params)

  def test_no_contact_and_threshold_boundary_are_free(self):
    env = self.make_env([[[0., 0., 0.], [0., 0., -1.], [.6, .8, 0.]]])
    torch.testing.assert_close(self.cost(env), torch.zeros(1))
    torch.testing.assert_close(self.cost(env, force_threshold=.9), torch.tensor([2.]))

  def test_multiple_nonwheel_primaries_are_counted_separately(self):
    env = self.make_env([[[0., 0., 0.], [0., 0., 0.]],
                         [[0., 0., -2.], [0., 0., 0.]],
                         [[0., 0., -2.], [3., 0., 0.]]])
    torch.testing.assert_close(self.cost(env), torch.tensor([0., 1., 2.]))
    torch.testing.assert_close(env.extras["log"]["Metrics/undesired_contact_count"],
                               torch.tensor(1.))
    torch.testing.assert_close(env.extras["log"]["Metrics/undesired_contact_ratio"],
                               torch.tensor(2 / 3))

  def test_brief_contact_is_retained_after_current_force_returns_to_zero(self):
    history = torch.zeros(1, 2, 4, 3)
    history[0, 1, 2, 0] = 2.
    env = self.make_env(torch.zeros(1, 2, 3), history)
    torch.testing.assert_close(self.cost(env), torch.ones(1))
    history.zero_()
    torch.testing.assert_close(self.cost(env), torch.zeros(1))

  def test_repeated_substep_contacts_count_once_per_primary(self):
    history = torch.zeros(1, 2, 4, 3)
    history[0, 0, :, 0] = 2.
    env = self.make_env(history[:, :, 0], history)
    torch.testing.assert_close(self.cost(env), torch.ones(1))
    history[0, 1, 3, 2] = -3.
    torch.testing.assert_close(self.cost(env), torch.tensor([2.]))

  def test_force_sign_and_reflection_do_not_change_the_cost(self):
    force = torch.tensor([[[2., 3., -4.], [0., 0., -1.], [0., 0., 0.]]])
    expected = self.cost(self.make_env(force))
    torch.testing.assert_close(self.cost(self.make_env(-force)), expected)
    reflected = force[:, [2, 0, 1]] * torch.tensor([1., -1., 1.])
    torch.testing.assert_close(self.cost(self.make_env(reflected)), expected)

  def test_reward_manager_applies_weight_and_control_duration(self):
    env = self.make_env([[[0., 0., -2.], [3., 0., 0.]], [[0., 0., 0.]] * 2])
    manager = RewardManager({
      "undesired_contact": RewardTermCfg(func=undesired_contacts, weight=-1.,
        params={"sensor_name": "nonfoot", "force_threshold": 1.}),
    }, env)
    torch.testing.assert_close(manager.compute(.02), torch.tensor([-.04, 0.]))

  def test_all_go2w_variants_exclude_wheels_and_cover_each_control_step(self):
    robot = Entity(get_go2w_robot_cfg())
    expected = {name for name in robot.geom_names
                if re.fullmatch(r".*_collision\d*$", name) and name not in WHEEL_GEOMS}
    self.assertTrue(expected)
    for task in list_tasks():
      if not task.startswith("Unitree-Go2W-"):
        continue
      for play in (False, True):
        with self.subTest(task=task, play=play):
          cfg = load_env_cfg(task, play=play)
          term = cfg.rewards["undesired_contact"]
          self.assertIs(term.func, undesired_contacts)
          self.assertEqual(term.weight, -1.)
          self.assertEqual(term.params, {"sensor_name": "nonfoot_ground_touch", "force_threshold": 1.})
          sensor = next(s for s in cfg.scene.sensors if s.name == term.params["sensor_name"])
          actual = sensor.build()._resolve_primary_names({"robot": robot}, sensor.primary)
          self.assertEqual(set(actual), expected)
          self.assertTrue(set(actual).isdisjoint(WHEEL_GEOMS))
          self.assertEqual(sensor.secondary.mode, "body")
          self.assertEqual(sensor.secondary.pattern, "terrain")
          self.assertIn("force", sensor.fields)
          self.assertEqual(sensor.num_slots, 1)
          self.assertEqual(sensor.history_length, cfg.decimation)

  def test_missing_or_incompatible_forces_are_rejected(self):
    env = self.make_env([[[0., 0., 0.]]])
    with self.assertRaises(ValueError):
      self.cost(env, force_threshold=-1.)
    data = env.scene["nonfoot"].data
    for force in (None, torch.zeros(1, 3), torch.zeros(1, 1, 2)):
      data.force = force
      with self.subTest(force=force), self.assertRaises(ValueError):
        self.cost(env)
    for history in (torch.zeros(1, 1, 3), torch.zeros(1, 1, 0, 3), torch.zeros(1, 1, 4, 2)):
      data.force_history = history
      with self.subTest(shape=history.shape), self.assertRaises(ValueError):
        self.cost(env)


if __name__ == "__main__":
  unittest.main()
