"""CPU checks for lateral rolling slip and optional temporal costs."""

import math
import unittest
from types import SimpleNamespace

import torch
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz

import src.tasks  # noqa: F401 -- register tasks
from src.tasks.velocity.mdp.smoothness_rewards import (
  action_smoothness_second_order, grf_smoothness,
)
from src.tasks.velocity.mdp.wheel_rewards import wheel_lateral_slide


class WheelSlideTests(unittest.TestCase):
  def make_env(self, velocities, rolls=None, yaws=None):
    velocity = torch.tensor(velocities, dtype=torch.float32).unsqueeze(1)
    batch = velocity.shape[0]
    roll = torch.tensor(rolls) if rolls is not None else torch.zeros(batch)
    yaw = torch.tensor(yaws) if yaws is not None else torch.zeros(batch)
    quat = quat_from_euler_xyz(roll, torch.zeros(batch), yaw).unsqueeze(1)
    force = torch.zeros_like(velocity)
    force[..., 2] = 2.0
    return SimpleNamespace(scene={
      "robot": SimpleNamespace(data=SimpleNamespace(
        body_com_lin_vel_w=velocity, body_link_quat_w=quat,
      )),
      "contact": SimpleNamespace(data=SimpleNamespace(force=force)),
    })

  def cost(self, env, **kwargs):
    return wheel_lateral_slide(env, "contact", SceneEntityCfg("robot"), **kwargs)

  def test_rolling_is_free_but_both_lateral_directions_cost(self):
    env = self.make_env([[2., 0., 0.], [0., .2, 0.], [0., -.2, 0.], [0., .03, 0.]])
    torch.testing.assert_close(self.cost(env), torch.tensor([0., .16, .16, 0.]))

  def test_axle_follows_heading_and_projects_out_vertical_camber_motion(self):
    env = self.make_env(
      [[0., 2., 0.], [.2, 0., 0.], [0., 0., 3.], [0., .2, 3.]],
      rolls=[0., 0., .8, .8], yaws=[math.pi / 2, math.pi / 2, 0., 0.],
    )
    torch.testing.assert_close(self.cost(env), torch.tensor([0., .16, 0., .16]))

  def test_current_force_gate_excludes_detached_and_threshold_contacts(self):
    env = self.make_env([[0., .2, 0.]] * 4)
    data = env.scene["contact"].data
    data.force[:] = torch.tensor([[[0., 0., 0.]], [[0., 0., 1.]],
                                  [[1.2, 0., 0.]], [[.8, .8, 0.]]])
    data.force_history = torch.full((4, 1, 3, 3), 100.)
    torch.testing.assert_close(self.cost(env), torch.tensor([0., 0., .16, .16]))

  def test_spinning_about_local_axle_preserves_lateral_direction(self):
    env = self.make_env([[0., .2, 0.]] * 3)
    pitch = torch.tensor([0., 1., 2.])
    env.scene["robot"].data.body_link_quat_w = quat_from_euler_xyz(
      torch.zeros(3), pitch, torch.zeros(3),
    ).unsqueeze(1)
    torch.testing.assert_close(self.cost(env), torch.full((3,), .16))

  def test_vertical_axle_is_finite_and_invalid_inputs_are_rejected(self):
    env = self.make_env([[0., .2, 3.]], rolls=[math.pi / 2])
    self.assertTrue(torch.isfinite(self.cost(env)).all())
    with self.assertRaises(ValueError):
      self.cost(env, slip_deadband=-.1)
    with self.assertRaises(ValueError):
      self.cost(env, contact_threshold=-1.)
    env.scene["contact"].data.force = torch.zeros(1, 2, 3)
    with self.assertRaises(ValueError):
      self.cost(env)


class TemporalRewardTests(unittest.TestCase):
  def test_action_history_uses_episode_age_including_valid_zero_samples(self):
    env = SimpleNamespace(
      action_manager=SimpleNamespace(
        action=torch.tensor([[2., 0.], [2., 0.], [2., 0.], [3., 4.]]),
        prev_action=torch.tensor([[0., 0.], [0., 0.], [0., 0.], [2., 3.]]),
        prev_prev_action=torch.tensor([[1., 0.], [1., 0.], [1., 0.], [1., 2.]]),
      ),
      episode_length_buf=torch.tensor([1, 2, 3, 10]),
    )
    torch.testing.assert_close(action_smoothness_second_order(env), torch.tensor([0., 0., 9., 0.]))

  def make_grf(self):
    force = torch.zeros(2, 1, 3)
    sensor = SimpleNamespace(
      cfg=ContactSensorCfg(name="contact", primary=ContactMatch(
        mode="geom", pattern=".*wheel.*", entity="robot",
      ), reduce="netforce"),
      data=SimpleNamespace(force=force),
    )
    env = SimpleNamespace(scene={"contact": sensor})
    cfg = RewardTermCfg(func=grf_smoothness, weight=-1., params={"sensor_name": "contact"})
    return env, force, grf_smoothness(cfg, env)

  def test_grf_policy_step_warmup_linear_ramp_and_liftoff(self):
    env, force, term = self.make_grf()
    force[..., 2] = 10.
    torch.testing.assert_close(term(env, "contact"), torch.zeros(2))
    force[..., 2] = 12.
    torch.testing.assert_close(term(env, "contact"), torch.full((2,), 4.))
    force[..., 2] = 14.
    torch.testing.assert_close(term(env, "contact"), torch.full((2,), 4.))
    force.zero_()  # Liftoff is included, rather than masked by current contact.
    torch.testing.assert_close(term(env, "contact"), torch.full((2,), 324.))

  def test_grf_partial_reset_does_not_mix_episodes_or_other_envs(self):
    env, force, term = self.make_grf()
    force[..., 2] = 10.
    term(env, "contact")
    term(env, "contact")
    term.reset(torch.tensor([0]))
    force[..., 2] = 20.
    torch.testing.assert_close(term(env, "contact"), torch.tensor([0., 150.]))
    force[..., 2] = 22.
    torch.testing.assert_close(term(env, "contact"), torch.tensor([4., 36.]))
    term.reset()
    torch.testing.assert_close(term(env, "contact"), torch.zeros(2))

  def test_grf_vertical_default_and_optional_vector_difference(self):
    env, force, term = self.make_grf()
    term(env, "contact")
    force[..., 0] = 3.
    torch.testing.assert_close(term(env, "contact"), torch.zeros(2))
    term.reset()
    force.zero_()
    term(env, "contact", vertical_only=False)
    force[..., 0] = 3.
    torch.testing.assert_close(term(env, "contact", vertical_only=False), torch.full((2,), 9.))
    with self.assertRaises(ValueError):
      term(env, "contact", second_difference_weight=-1.)

  def test_grf_rejects_contact_frame_force(self):
    env, _, _ = self.make_grf()
    env.scene["contact"].cfg.reduce = "maxforce"
    cfg = RewardTermCfg(func=grf_smoothness, weight=-1., params={"sensor_name": "contact"})
    with self.assertRaises(ValueError):
      grf_smoothness(cfg, env)

  def test_only_slide_is_configured_in_every_go2w_variant(self):
    for task in list_tasks():
      if not task.startswith("Unitree-Go2W-"):
        continue
      for play in (False, True):
        with self.subTest(task=task, play=play):
          cfg = load_env_cfg(task, play=play)
          slide = cfg.rewards["wheel_lateral_slide"]
          self.assertIs(slide.func, wheel_lateral_slide)
          self.assertEqual(slide.weight, -.1)
          self.assertEqual(slide.params["sensor_name"], "feet_ground_contact")
          self.assertEqual(slide.params["body_cfg"].body_names,
                           cfg.rewards["wheel_spin_contact"].params["body_cfg"].body_names)
          self.assertFalse(any(term.func in (action_smoothness_second_order, grf_smoothness)
                               for term in cfg.rewards.values()))


if __name__ == "__main__":
  unittest.main()
