"""Sustained four-wheel support should beat occasional toe taps."""

from types import SimpleNamespace
import unittest

import torch
from mjlab.tasks.registry import list_tasks, load_env_cfg

import src.tasks  # noqa: F401 -- register task variants
from src.tasks.velocity.mdp.rewards.wheel import wheel_contact_bonus


class WheelContactBonusTests(unittest.TestCase):
  def make_env(self, vertical_force, commands=None):
    vertical_force = torch.tensor(vertical_force, dtype=torch.float32)
    force = torch.zeros((*vertical_force.shape, 3))
    # Native MuJoCo contact force is wheel-on-terrain, opposite the support.
    force[..., 2] = -vertical_force
    if commands is None:
      commands = torch.zeros(vertical_force.shape[0], 3)
      commands[:, 0] = .4
    manager = SimpleNamespace(command=torch.as_tensor(commands, dtype=torch.float32))
    manager.get_command = lambda name: manager.command
    return SimpleNamespace(
      scene={"contact": SimpleNamespace(data=SimpleNamespace(
        force=force, found=torch.ones_like(vertical_force),
      ))},
      command_manager=manager,
      extras={"log": {}},
    )

  def reward(self, env, **params):
    return wheel_contact_bonus(env, "contact", **params)

  def test_more_supporting_wheels_earn_more_reward_during_straight_motion(self):
    env = self.make_env([[0., 0., 0., 0.], [30., 30., 0., 0.],
                         [30., 30., 30., 0.], [30., 30., 30., 30.]])
    torch.testing.assert_close(self.reward(env), torch.tensor([0., .5, .75, 1.]))
    torch.testing.assert_close(env.extras["log"]["Metrics/all_wheels_grounded_ratio"],
                               torch.tensor(.25))

  def test_forward_reverse_and_smooth_direction_scores_without_rewarding_in_place_motion(self):
    env = self.make_env([[30.] * 4] * 9, commands=[
      [.4, 0., 0.], [-.4, 0., 0.], [0., 0., 0.], [0., .3, 0.],
      [0., 0., .3], [.4, .2, 0.], [.4, 0., .2], [.4, .01, .01], [-.4, -.01, -.01],
    ])
    torch.testing.assert_close(self.reward(env), torch.tensor([
      1., 1., 0., 0., 0., .01831564, .36787944, .9875778, .9875778,
    ]))
    torch.testing.assert_close(env.extras["log"]["Metrics/wheel_contact_bonus_active_ratio"],
                               torch.tensor(6 / 9))
    torch.testing.assert_close(env.extras["log"]["Metrics/wheel_contact_bonus_gate_mean"],
                               self.reward(env).mean())

  def test_longitudinal_motion_ramps_on_in_both_directions(self):
    env = self.make_env([[30.] * 4] * 7, commands=[
      [.1, 0., 0.], [-.1, 0., 0.], [.125, 0., 0.], [-.15, 0., 0.],
      [.2, 0., 0.], [-.4, 0., 0.], [.4, 0., 0.],
    ])
    torch.testing.assert_close(self.reward(env), torch.tensor([0., 0., .25, .5, 1., 1., 1.]))

  def test_lateral_and_yaw_scores_decay_continuously_and_symmetrically(self):
    expected = torch.tensor([1., .93941306, .77880078, .36787944, .01831564])
    for column, commands in (
      (1, [0., .025, .05, .1, .2]), (2, [0., .05, .1, .2, .4]),
    ):
      env = self.make_env([[30.] * 4] * 5)
      env.command_manager.command[:, column] = torch.tensor(commands)
      with self.subTest(column=column):
        torch.testing.assert_close(self.reward(env), expected)
        env.command_manager.command[:, column].neg_()
        torch.testing.assert_close(self.reward(env), expected)

  def test_old_yaw_cutoff_no_longer_causes_a_reward_jump(self):
    env = self.make_env([[30.] * 4] * 3, commands=[
      [.4, 0., .03999], [.4, 0., .04], [.4, 0., .04001],
    ])
    values = self.reward(env)
    self.assertTrue((values > .95).all())
    self.assertLess(float((values[1:] - values[:-1]).abs().max()), .0001)

  def test_direction_scales_can_relax_the_same_command(self):
    env = self.make_env([[30.] * 4], commands=[[.2, .1, .07]])
    ordinary = self.reward(env)
    relaxed = self.reward(env, command_y_scale=.2, command_yaw_scale=.4)
    self.assertTrue((relaxed > ordinary).all())
    self.assertTrue((relaxed <= 1.).all())

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
    for name in ("force_threshold", "command_x_threshold", "command_x_ramp",
                 "command_y_scale", "command_yaw_scale"):
      with self.subTest(parameter=name), self.assertRaises(ValueError):
        self.reward(env, **{name: -1.})
    for name in ("command_x_ramp", "command_y_scale", "command_yaw_scale"):
      with self.subTest(zero_parameter=name), self.assertRaises(ValueError):
        self.reward(env, **{name: 0.})
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

  def test_missing_velocity_command_is_rejected(self):
    env = self.make_env([[30.] * 4])
    env.command_manager.command = None
    with self.assertRaisesRegex(ValueError, "velocity command"):
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
          self.assertEqual(term.params, {
            "sensor_name": "feet_ground_contact", "force_threshold": 5.,
            "command_name": "twist", "command_x_threshold": .1,
            "command_x_ramp": .1, "command_y_scale": .1, "command_yaw_scale": .2,
          })
          sensor = next(sensor for sensor in cfg.scene.sensors
                        if sensor.name == term.params["sensor_name"])
          self.assertEqual(sensor.reduce, "netforce")
          self.assertEqual(sensor.num_slots, 1)
          self.assertEqual(sensor.fields, ("found", "force"))
          self.assertEqual(len(sensor.primary.pattern), 4)


if __name__ == "__main__":
  unittest.main()
