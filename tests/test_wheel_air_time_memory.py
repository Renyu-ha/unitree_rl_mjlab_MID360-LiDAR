"""The original air-time timeout with a record that survives brief contacts."""

from types import SimpleNamespace
import unittest

import torch
from mjlab.managers.reward_manager import RewardManager, RewardTermCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg

import src.tasks  # noqa: F401
from src.tasks.velocity.mdp.rewards.wheel import wheel_air_time_memory, wheel_air_time_timeout


class WheelAirTimeMemoryTests(unittest.TestCase):
  def make_env(self, air_time, step_dt=.02):
    current = torch.tensor(air_time, dtype=torch.float32)
    return SimpleNamespace(
      scene={"contact": SimpleNamespace(data=SimpleNamespace(current_air_time=current))},
      num_envs=current.shape[0], device="cpu", step_dt=step_dt,
      max_episode_length_s=20., extras={"log": {}},
    )

  def make_term(self, env):
    cfg = RewardTermCfg(func=wheel_air_time_memory, weight=-.25,
                        params={"sensor_name": "contact"})
    return wheel_air_time_memory(cfg, env)

  def cost(self, term, env, **params):
    return term(env, "contact", **params)

  def test_uninterrupted_suspension_matches_original_timeout(self):
    env = self.make_env([[.625, 0., 0., 0.], [.75, 0., 0., 0.], [10.] * 4])
    term = self.make_term(env)
    before = env.scene["contact"].data.current_air_time.clone()
    torch.testing.assert_close(self.cost(term, env), wheel_air_time_timeout(env, "contact"))
    torch.testing.assert_close(env.scene["contact"].data.current_air_time, before)
    self.assertLessEqual(term.air_time.max().item(), .75)

  def test_short_swing_is_free_and_sustained_contact_recovers_history(self):
    env = self.make_env([[0.] * 4])
    term = self.make_term(env)
    current = env.scene["contact"].data.current_air_time
    for _ in range(20):
      current[0, 0] += env.step_dt
      torch.testing.assert_close(self.cost(term, env), torch.zeros(1))
    torch.testing.assert_close(term.air_time[0, 0], torch.tensor(.4))
    current.zero_()
    for _ in range(11):
      self.cost(term, env)
    torch.testing.assert_close(term.air_time, torch.zeros(1, 4))

  def test_one_contact_step_does_not_clear_history_or_penalty(self):
    env = self.make_env([[1., 0., 0., 0.]])
    term = self.make_term(env)
    torch.testing.assert_close(self.cost(term, env), torch.tensor([.25]))
    env.scene["contact"].data.current_air_time.zero_()
    torch.testing.assert_close(self.cost(term, env), torch.tensor([.21]))
    torch.testing.assert_close(term.air_time[0, 0], torch.tensor(.71))
    torch.testing.assert_close(wheel_air_time_timeout(env, "contact"), torch.zeros(1))
    env.scene["contact"].data.current_air_time[0, 0] = .02
    torch.testing.assert_close(self.cost(term, env), torch.tensor([.23]))

  def test_rare_touches_can_reset_original_timer_but_cannot_erase_memory(self):
    env = self.make_env([[0.] * 4])
    term = self.make_term(env)
    current = env.scene["contact"].data.current_air_time
    for step in range(200):
      current[0, 0] = 0. if step % 25 == 0 else current[0, 0] + env.step_dt
      value = self.cost(term, env)
      torch.testing.assert_close(wheel_air_time_timeout(env, "contact"), torch.zeros(1))
    torch.testing.assert_close(value, torch.tensor([.25]))

  def test_all_wheels_airborne_are_penalized(self):
    env = self.make_env([[2.] * 4])
    torch.testing.assert_close(self.cost(self.make_term(env), env), torch.ones(1))

  def test_contact_recovery_depends_on_elapsed_time_not_frequency(self):
    for dt in (.01, .02, .05):
      with self.subTest(step_dt=dt):
        env = self.make_env([[1., 0., 0., 0.]], step_dt=dt)
        term = self.make_term(env)
        self.cost(term, env)
        env.scene["contact"].data.current_air_time.zero_()
        for _ in range(round(.1 / dt)):
          value = self.cost(term, env)
        torch.testing.assert_close(term.air_time[0, 0], torch.tensor(.55))
        torch.testing.assert_close(value, torch.tensor([.05]))

  def test_record_is_bounded_so_long_flight_does_not_require_long_recovery(self):
    env = self.make_env([[100., 0., 0., 0.]])
    term = self.make_term(env)
    self.cost(term, env)
    env.scene["contact"].data.current_air_time.zero_()
    for _ in range(20):
      value = self.cost(term, env)
    torch.testing.assert_close(value, torch.zeros(1))
    torch.testing.assert_close(term.air_time, torch.zeros(1, 4))

  def test_stepup_grace_and_configurable_recovery_rate(self):
    env = self.make_env([[.6, .7, .825, .95]])
    term = self.make_term(env)
    torch.testing.assert_close(self.cost(term, env, threshold=.7), torch.tensor([.375]))
    env.scene["contact"].data.current_air_time.zero_()
    before = term.air_time.clone()
    self.cost(term, env, threshold=.7, recovery_rate=1.)
    torch.testing.assert_close(term.air_time, before - .02)

  def test_partial_and_full_reset_only_clear_selected_record(self):
    env = self.make_env([[1., 0., 0., 0.], [0., 0., 0., 1.]])
    term = self.make_term(env)
    self.cost(term, env)
    before = term.air_time.clone()
    term.reset(torch.tensor([0]))
    torch.testing.assert_close(term.air_time[0], torch.zeros(4))
    torch.testing.assert_close(term.air_time[1], before[1])
    term.reset()
    torch.testing.assert_close(term.air_time, torch.zeros(2, 4))

  def test_manager_keeps_original_term_and_resets_memory_per_environment(self):
    env = self.make_env([[1., 0., 0., 0.]] * 2)
    manager = RewardManager({
      "original": RewardTermCfg(func=wheel_air_time_timeout, weight=-.5,
                                params={"sensor_name": "contact"}),
      "memory": RewardTermCfg(func=wheel_air_time_memory, weight=-.25,
                              params={"sensor_name": "contact"}),
    }, env)
    torch.testing.assert_close(manager.compute(env.step_dt), torch.full((2,), -.00375))
    term = manager.get_term_cfg("memory").func
    before = term.air_time.clone()
    manager.reset(env_ids=torch.tensor([0]))
    torch.testing.assert_close(term.air_time[0], torch.zeros(4))
    torch.testing.assert_close(term.air_time[1], before[1])
    self.assertIs(manager.get_term_cfg("original").func, wheel_air_time_timeout)

  def test_invalid_timing_or_missing_tracking_is_rejected(self):
    env = self.make_env([[0.] * 4])
    term = self.make_term(env)
    for params in ({"threshold": -1.}, {"ramp_time": 0.}, {"recovery_rate": 0.}):
      with self.subTest(params=params), self.assertRaises(ValueError):
        self.cost(term, env, **params)
    env.scene["contact"].data.current_air_time = None
    with self.assertRaises(ValueError):
      self.make_term(env)
    with self.assertRaises(ValueError):
      self.cost(term, env)
    env.scene["contact"].data.current_air_time = torch.zeros(1, 3)
    with self.assertRaises(ValueError):
      self.cost(term, env)

  def test_all_go2w_variants_keep_original_timeout_and_add_only_time_record_parameters(self):
    for task in list_tasks():
      if not task.startswith("Unitree-Go2W-"):
        continue
      for play in (False, True):
        with self.subTest(task=task, play=play):
          cfg = load_env_cfg(task, play=play)
          term = cfg.rewards["wheel_air_time_memory"]
          self.assertIs(term.func, wheel_air_time_memory)
          self.assertEqual(term.weight, -.25)
          self.assertEqual(term.params, {
            "sensor_name": "feet_ground_contact", "threshold": .7 if "StepUp" in task else .5,
            "ramp_time": .25, "recovery_rate": 2.,
          })
          self.assertIs(cfg.rewards["wheel_air_time_timeout"].func, wheel_air_time_timeout)
          self.assertNotIn("wheel_air_time_imbalance", cfg.rewards)


if __name__ == "__main__":
  unittest.main()
