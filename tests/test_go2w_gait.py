"""G1 reward coverage, isolated configuration and phase-consistent Go2W mirrors."""

from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace
import unittest

import torch
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

import src.tasks  # noqa: F401
from src.tasks.velocity import mdp
from src.tasks.velocity.config.go2w.gait_env_cfg import unitree_go2w_perceptive_gait_env_cfg
from src.tasks.velocity.rl.go2w_symmetry import Go2WSymmetry
from tests.test_go2w_symmetry import make_symmetry_env, term_slice


TASK = "Unitree-Go2W-Perceptive-Rough-v2-Gait"
BASE = "Unitree-Go2W-Perceptive-Rough-v2"
RESTORED = {"angular_momentum", "foot_gait", "foot_clearance", "foot_swing_height"}
REMOVED = {"wheel_air_time_timeout", "wheel_air_time_memory", "wheel_contact_bonus"}


class Go2WGaitTests(unittest.TestCase):
  def test_g1_v2_reward_types_are_all_present_except_foot_slip(self):
    for play in (False, True):
      with self.subTest(play=play):
        cfg, g1, baseline = (load_env_cfg(name, play=play) for name in
                             (TASK, "Unitree-G1-Perceptive-Rough-v2", BASE))
        self.assertEqual(set(cfg.rewards), (set(baseline.rewards) - REMOVED) | RESTORED)
        self.assertTrue((set(g1.rewards) - {"foot_slip"}) <= set(cfg.rewards))
        self.assertNotIn("foot_slip", cfg.rewards)
        for name in g1.rewards.keys() & cfg.rewards.keys():
          self.assertIs(cfg.rewards[name].func, g1.rewards[name].func, name)
        for name in RESTORED:
          self.assertEqual(cfg.rewards[name].weight, g1.rewards[name].weight)
          actual, expected = deepcopy(cfg.rewards[name].params), deepcopy(g1.rewards[name].params)
          actual.pop("asset_cfg", None)
          expected.pop("asset_cfg", None)
          if name == "foot_gait":
            self.assertEqual(actual.pop("offset"), [0., .5, .5, 0.])
            self.assertEqual(expected.pop("offset"), [0., .5])
          self.assertEqual(actual, expected)
        if play:
          self.assertEqual(cfg.curriculum, {})
        for name in ("foot_clearance_weight", "foot_swing_height_weight"):
          if not play:
            self.assertEqual(asdict(cfg.curriculum[name]), asdict(g1.curriculum[name]))

  def test_only_the_requested_observations_rewards_and_curricula_differ_from_baseline(self):
    for play in (False, True):
      cfg = unitree_go2w_perceptive_gait_env_cfg(play)
      baseline = load_env_cfg(BASE, play=play)
      for group in ("actor", "critic"):
        phase = cfg.observations[group].terms.pop("phase")
        self.assertIs(phase.func, mdp.phase)
        self.assertIsNone(phase.noise)
        self.assertEqual(phase.params, {"period": .6, "command_name": "twist"})
        self.assertNotIn("phase", baseline.observations[group].terms)
      for name in RESTORED:
        cfg.rewards.pop(name)
      for name in REMOVED:
        cfg.rewards[name] = deepcopy(baseline.rewards[name])
      for name in ("foot_clearance_weight", "foot_swing_height_weight"):
        cfg.curriculum.pop(name, None)
      # Ignore freshly bound loaders and the sensing callable instance identity.
      for variant in (cfg, baseline):
        variant.scene.entities["robot"].spec_fn = None
        variant.scene.terrain.spec_fn = None
        variant.observations["actor"].terms["height_scan"].func = None
      self.assertEqual(asdict(cfg), asdict(baseline))
      untouched = load_env_cfg(BASE, play=play)
      self.assertTrue(REMOVED <= set(untouched.rewards))
      self.assertFalse(RESTORED & set(untouched.rewards))
    rl, base_rl = asdict(load_rl_cfg(TASK)), asdict(load_rl_cfg(BASE))
    self.assertEqual(rl.pop("run_name"), "v2_gait")
    base_rl.pop("run_name")
    self.assertEqual(rl, base_rl)

  @staticmethod
  def gait_env(ticks, contacts, commands=None):
    ticks = torch.as_tensor(ticks)
    commands = torch.tensor([[.4, 0., 0.]]).expand(len(ticks), -1).clone() if commands is None else commands
    return SimpleNamespace(
      num_envs=len(ticks), device="cpu", step_dt=.02, episode_length_buf=ticks,
      command_manager=SimpleNamespace(get_command=lambda name: commands),
      scene={"feet_ground_contact": SimpleNamespace(data=SimpleNamespace(
        current_contact_time=torch.as_tensor(contacts, dtype=torch.float32) * .02,
      ))},
    )

  def test_gait_scores_the_four_correct_wheels_over_two_periods_and_gates_standing(self):
    cfg = load_env_cfg(TASK)
    reward = cfg.rewards["foot_gait"]
    ticks = torch.arange(60)
    # Independent oracle in control ticks: period 30, diagonal offset 15, duty 16.8.
    stance = ((ticks[:, None] + torch.tensor([0, 15, 15, 0])) % 30) < 16.8
    env = self.gait_env(ticks, stance)
    torch.testing.assert_close(reward.func(env, **reward.params), torch.ones(60))
    env.scene["feet_ground_contact"].data.current_contact_time = (~stance).float() * .02
    torch.testing.assert_close(reward.func(env, **reward.params), torch.zeros(60))
    env.command_manager.get_command = lambda name: torch.zeros(60, 3)
    torch.testing.assert_close(reward.func(env, **reward.params), torch.zeros(60))
    torch.testing.assert_close(mdp.phase(env, .6, "twist"), torch.zeros(60, 2))

  def test_mirrored_contacts_require_a_half_period_shift_of_the_phase_clock(self):
    env = self.gait_env([3, 18], [[1, 0, 0, 1], [0, 1, 1, 0]])
    reward = load_env_cfg(TASK).rewards["foot_gait"]
    torch.testing.assert_close(reward.func(env, **reward.params), torch.ones(2))
    phase = mdp.phase(env, .6, "twist")
    torch.testing.assert_close(phase[1], -phase[0], atol=1.e-6, rtol=1.e-6)
    env.episode_length_buf[1] = 3
    torch.testing.assert_close(reward.func(env, **reward.params), torch.tensor([1., 0.]))

  def test_actor_and_critic_mirror_every_phase_history_frame(self):
    env = make_symmetry_env(TASK)
    mirror, baseline = Go2WSymmetry(env), Go2WSymmetry(make_symmetry_env(BASE))
    for group, width, base_width in (("actor", 726, 720), ("critic", 265, 263)):
      values = torch.arange(float(width)).view(1, -1)
      result = mirror.mirror(group, values)
      torch.testing.assert_close(result[:, :base_width], baseline.mirror(group, values[:, :base_width]))
      phase = term_slice(env, group, "phase")
      torch.testing.assert_close(result[:, phase], -values[:, phase])
      torch.testing.assert_close(mirror.mirror(group, result), values)

  def test_mismatched_clock_or_unsupported_offset_symmetry_is_rejected(self):
    for change in ("clock", "offset", "contact_order"):
      env = make_symmetry_env(TASK)
      if change == "clock":
        env.cfg.rewards["foot_gait"].params["period"] = .8
      elif change == "offset":
        env.cfg.rewards["foot_gait"].params["offset"] = [0., .25, .5, .75]
      else:
        env.scene["feet_ground_contact"].primary_names.reverse()
      with self.subTest(change=change), self.assertRaises(ValueError):
        Go2WSymmetry(env)

  def test_swing_peak_history_clears_only_reset_environments(self):
    cfg = load_env_cfg(TASK).rewards["foot_swing_height"]
    env = SimpleNamespace(num_envs=2, device="cpu", step_dt=.02)
    reward = cfg.func(cfg=cfg, env=env)
    reward.peak_heights[:] = torch.tensor([[1., 2., 3., 4.], [5., 6., 7., 8.]])
    reward.reset(torch.tensor([0]))
    torch.testing.assert_close(reward.peak_heights[0], torch.zeros(4))
    torch.testing.assert_close(reward.peak_heights[1], torch.tensor([5., 6., 7., 8.]))
    reward.reset()
    torch.testing.assert_close(reward.peak_heights, torch.zeros(2, 4))

  def test_angular_momentum_cost_is_invariant_to_axial_vector_reflection(self):
    value = torch.tensor([[.3, -.5, .7], [1.4, .6, -.2]])
    env = SimpleNamespace(scene={"robot/root_angmom": SimpleNamespace(data=value)}, extras={"log": {}})
    reward = load_env_cfg(TASK).rewards["angular_momentum"]
    actual = reward.func(env, **reward.params)
    env.scene["robot/root_angmom"].data = value * torch.tensor([-1., 1., -1.])
    torch.testing.assert_close(reward.func(env, **reward.params), actual)


if __name__ == "__main__":
  unittest.main()
