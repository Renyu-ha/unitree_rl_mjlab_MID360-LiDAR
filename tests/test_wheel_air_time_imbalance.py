"""A brief touchdown must not erase a wheel's recent lack of support."""

import math
from types import SimpleNamespace
import unittest

import torch
from mjlab.managers.reward_manager import RewardManager, RewardTermCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg

import src.tasks  # noqa: F401 -- register task variants
from src.tasks.velocity.mdp.rewards.wheel import (
  wheel_air_time_imbalance,
  wheel_air_time_timeout,
  wheel_contact_bonus,
)


class WheelAirTimeImbalanceTests(unittest.TestCase):
  def make_env(self, support_forces, commands=None, step_dt=.02):
    support = torch.tensor(support_forces, dtype=torch.float32)
    force = torch.zeros((*support.shape, 3))
    force[..., 2] = -support
    if commands is None:
      commands = torch.tensor([[.4, 0., 0.]]).repeat(support.shape[0], 1)
    command_manager = SimpleNamespace(command=torch.as_tensor(commands, dtype=torch.float32))
    command_manager.get_command = lambda name: command_manager.command
    return SimpleNamespace(
      num_envs=support.shape[0], device="cpu", step_dt=step_dt, max_episode_length_s=20.,
      scene={"contact": SimpleNamespace(data=SimpleNamespace(
        force=force, found=torch.ones_like(support), current_air_time=torch.zeros_like(support),
      ))},
      command_manager=command_manager, extras={"log": {}},
    )

  def make_term(self, env):
    cfg = RewardTermCfg(func=wheel_air_time_imbalance, weight=-.5,
                        params={"sensor_name": "contact"})
    return wheel_air_time_imbalance(cfg, env)

  def cost(self, term, env, **params):
    return term(env, "contact", **params)

  def test_population_std_penalizes_unequal_ratios_not_equal_flight(self):
    env = self.make_env([[30., 30., 30., 0.], [30., 30., 0., 0.],
                         [0., 0., 0., 0.], [30., 30., 30., 30.]])
    term = self.make_term(env)
    term.air_ratio[:] = torch.tensor([[0., 0., 0., 1.], [0., 0., 1., 1.],
                                     [1., 1., 1., 1.], [0., 0., 0., 0.]])
    torch.testing.assert_close(self.cost(term, env), torch.tensor([math.sqrt(3) / 4, .5, 0., 0.]))

  def test_history_builds_gradually_and_one_leg_stays_penalized(self):
    env = self.make_env([[30., 30., 30., 0.]])
    term = self.make_term(env)
    first = self.cost(term, env)
    torch.testing.assert_close(first, torch.tensor([math.sqrt(3) / 4 * (1 - math.exp(-.04))]))
    for _ in range(149):
      settled = self.cost(term, env)
    self.assertGreater(settled.item(), .43)
    self.assertLess(first.item(), .02)
    self.assertTrue(((term.air_ratio >= 0.) & (term.air_ratio <= 1.)).all())

  def test_single_touchdown_only_reduces_history_by_one_ema_update(self):
    env = self.make_env([[30., 30., 30., 0.]])
    term = self.make_term(env)
    term.air_ratio[0, 3] = 1.
    before = self.cost(term, env)
    env.scene["contact"].data.force[0, 3, 2] = -30.
    after = self.cost(term, env)
    torch.testing.assert_close(after, before * math.exp(-.04))
    self.assertGreater(after.item(), .4)
    env.scene["contact"].data.force[0, 3, 2] = 0.
    self.assertGreater(self.cost(term, env).item(), after.item())

  def test_repeated_rare_touches_remain_worse_than_continuous_support(self):
    env = self.make_env([[30.] * 4, [30.] * 4])
    term = self.make_term(env)
    for step in range(200):
      env.scene["contact"].data.force[0, 3, 2] = -30. if step % 25 == 0 else 0.
      values = self.cost(term, env)
    self.assertGreater(values[0].item(), .4)
    self.assertEqual(values[1].item(), 0.)

  def test_alternating_balanced_support_has_little_imbalance(self):
    env = self.make_env([[30., 0., 0., 30.]])
    term = self.make_term(env)
    for step in range(200):
      env.scene["contact"].data.force[0, :, 2] = torch.tensor(
        [-30., 0., 0., -30.] if step % 2 == 0 else [0., -30., -30., 0.])
      value = self.cost(term, env)
    self.assertLess(value.item(), .011)
    self.assertGreater(term.air_ratio.min().item(), .48)

  def test_time_constant_is_independent_of_control_frequency(self):
    results = []
    for dt in (.01, .02, .05):
      env = self.make_env([[30., 30., 30., 0.]], step_dt=dt)
      term = self.make_term(env)
      for _ in range(round(1. / dt)):
        value = self.cost(term, env, time_constant=.5)
      results.append(value.item())
    expected = math.sqrt(3) / 4 * (1 - math.exp(-2.))
    torch.testing.assert_close(torch.tensor(results), torch.full((3,), expected))

  def test_force_qualified_support_ignores_light_wall_downward_and_stale_contacts(self):
    env = self.make_env([[5., 5.1, -40., 40.], [0., 30., 30., 30.]])
    env.scene["contact"].data.found[0, 3] = 0.
    env.scene["contact"].data.force[1, 0, :2] = torch.tensor([30., 40.])
    term = self.make_term(env)
    self.cost(term, env)
    expected = torch.tensor([[1., 0., 1., 1.], [1., 0., 0., 0.]]) * (1 - math.exp(-.04))
    torch.testing.assert_close(term.air_ratio, expected)

  def test_binary_motion_gate_and_direction_scores_match_contact_bonus(self):
    env = self.make_env([[30., 30., 30., 0.]] * 9, commands=[
      [.4, 0., 0.], [-.4, 0., 0.], [.1, 0., 0.], [-.1, 0., 0.],
      [0., 0., 0.], [0., .3, 0.], [0., 0., .3], [.4, .05, .1], [-.4, -.05, -.1],
    ])
    term = self.make_term(env)
    term.air_ratio[:, 3] = 1.
    raw = math.sqrt(3) / 4
    gate = torch.tensor([1., 1., 0., 0., 0., 0., 0., .5, .5])
    torch.testing.assert_close(self.cost(term, env), gate * raw)
    torch.testing.assert_close(wheel_contact_bonus(env, "contact"), gate * .75)
    torch.testing.assert_close(self.cost(term, env, direction_score_type="exponential")[-2:],
                               torch.full((2,), raw * math.exp(-1.)))

  def test_history_keeps_updating_while_command_gate_is_off(self):
    env = self.make_env([[30., 30., 30., 0.]], commands=[[0., 0., 0.]])
    term = self.make_term(env)
    for _ in range(50):
      torch.testing.assert_close(self.cost(term, env), torch.zeros(1))
    self.assertGreater(term.air_ratio[0, 3].item(), .86)
    self.assertEqual(env.extras["log"]["Metrics/wheel_air_time_imbalance_gate_mean"].item(), 0.)
    self.assertGreater(env.extras["log"]["Metrics/wheel_air_ratio_std_mean"].item(), .37)
    env.command_manager.command[0, 0] = .4
    self.assertGreater(self.cost(term, env).item(), .37)

  def test_partial_and_full_reset_only_clear_selected_history(self):
    env = self.make_env([[30., 30., 30., 0.], [30., 30., 0., 30.]])
    term = self.make_term(env)
    for _ in range(10):
      self.cost(term, env)
    before = term.air_ratio.clone()
    term.reset(torch.tensor([0]))
    torch.testing.assert_close(term.air_ratio[0], torch.zeros(4))
    torch.testing.assert_close(term.air_ratio[1], before[1])
    term.reset()
    torch.testing.assert_close(term.air_ratio, torch.zeros(2, 4))

  def test_reward_manager_scales_cost_and_resets_it_independently_of_legacy_timeout(self):
    env = self.make_env([[30., 30., 30., 0.]] * 2)
    manager = RewardManager({
      "legacy": RewardTermCfg(func=wheel_air_time_timeout, weight=-.5,
                              params={"sensor_name": "contact"}),
      "new": RewardTermCfg(func=wheel_air_time_imbalance, weight=-.5,
                           params={"sensor_name": "contact", "time_constant": .5}),
    }, env)
    new = manager.get_term_cfg("new").func
    for _ in range(10):
      value = manager.compute(env.step_dt)
    expected = new.air_ratio.std(dim=-1, correction=0) * -.5 * env.step_dt
    torch.testing.assert_close(value, expected)
    torch.testing.assert_close(manager.get_term("legacy"), torch.zeros(2))
    before = new.air_ratio.clone()
    manager.reset(env_ids=torch.tensor([0]))
    torch.testing.assert_close(new.air_ratio[0], torch.zeros(4))
    torch.testing.assert_close(new.air_ratio[1], before[1])
    self.assertIs(manager.get_term_cfg("legacy").func, wheel_air_time_timeout)

  def test_configured_support_terms_allow_normal_turning_steps_but_penalize_idle_leg(self):
    env = self.make_env([[30.] * 4] * 2, commands=[[.4, 0., .2]] * 2)
    env.scene["feet_ground_contact"] = env.scene["contact"]
    rewards = load_env_cfg("Unitree-Go2W-Perceptive-Rough-v2-StepUp-Finetune").rewards
    balance_cfg = rewards["wheel_air_time_imbalance"]
    balance = wheel_air_time_imbalance(balance_cfg, env)
    timeout_cfg, bonus_cfg = rewards["wheel_air_time_timeout"], rewards["wheel_contact_bonus"]
    data = env.scene["contact"].data
    # Normal diagonal stepping: each swing lasts .3 s. The second environment
    # leaves the same leg permanently unsupported throughout the six seconds.
    for step in range(300):
      support = torch.tensor([
        [1., 0., 0., 1.] if (step // 15) % 2 == 0 else [0., 1., 1., 0.],
        [1., 1., 1., 0.],
      ])
      data.force[..., 2] = -30. * support
      data.found.copy_(support)
      data.current_air_time.copy_(torch.where(
        support.bool(), 0., data.current_air_time + env.step_dt))
      balance_reward = balance(env, **balance_cfg.params) * balance_cfg.weight
    timeout_reward = timeout_cfg.func(env, **timeout_cfg.params) * timeout_cfg.weight
    bonus_reward = bonus_cfg.func(env, **bonus_cfg.params) * bonus_cfg.weight
    torch.testing.assert_close(bonus_reward, torch.zeros(2))
    torch.testing.assert_close(timeout_reward, torch.tensor([0., -.125]))
    self.assertLess(abs(balance_reward[0].item()), .02)
    self.assertGreater(abs(balance_reward[1].item()), 4 * abs(balance_reward[0].item()))
    self.assertLess((balance_reward + timeout_reward)[1].item(), -.17)

  def test_invalid_parameters_and_missing_command_do_not_mutate_history(self):
    env = self.make_env([[30.] * 4])
    term = self.make_term(env)
    for params in ({"time_constant": 0.}, {"time_constant": -.1}, {"force_threshold": -1.},
                   {"command_x_threshold": -1.}, {"command_y_scale": 0.},
                   {"command_yaw_scale": -1.}, {"direction_score_type": "unknown"}):
      with self.subTest(params=params), self.assertRaises(ValueError):
        self.cost(term, env, **params)
    env.command_manager.command = None
    with self.assertRaisesRegex(ValueError, "velocity command"):
      self.cost(term, env)
    torch.testing.assert_close(term.air_ratio, torch.zeros(1, 4))

  def test_missing_or_changed_contact_data_is_rejected(self):
    env = self.make_env([[30.] * 4])
    term = self.make_term(env)
    data = env.scene["contact"].data
    for field in ("force", "found"):
      original = getattr(data, field)
      setattr(data, field, None)
      with self.subTest(field=field), self.assertRaises(ValueError):
        self.make_term(env)
      with self.subTest(changed_field=field), self.assertRaises(ValueError):
        self.cost(term, env)
      setattr(data, field, original)
    data.force, data.found = torch.zeros(1, 3, 3), torch.ones(1, 3)
    with self.assertRaises(ValueError):
      self.make_term(env)
    with self.assertRaises(ValueError):
      self.cost(term, env)

  def test_all_go2w_variants_add_imbalance_without_replacing_existing_terms(self):
    for task in list_tasks():
      if not task.startswith("Unitree-Go2W-"):
        continue
      for play in (False, True):
        with self.subTest(task=task, play=play):
          cfg = load_env_cfg(task, play=play)
          term = cfg.rewards["wheel_air_time_imbalance"]
          self.assertIs(term.func, wheel_air_time_imbalance)
          self.assertEqual(term.weight, -.25)
          self.assertEqual(term.params, {
            "sensor_name": "feet_ground_contact", "time_constant": 1.,
            "force_threshold": 5., "command_name": "twist", "command_x_threshold": .1,
            "command_y_scale": .1, "command_yaw_scale": .2, "direction_score_type": "rational",
          })
          self.assertIs(cfg.rewards["wheel_air_time_timeout"].func, wheel_air_time_timeout)
          self.assertIs(cfg.rewards["wheel_contact_bonus"].func, wheel_contact_bonus)


if __name__ == "__main__":
  unittest.main()
