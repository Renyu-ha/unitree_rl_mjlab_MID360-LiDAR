"""Behavioral checks for SGS, tracking labels, reset attribution and checkpoints."""

from dataclasses import asdict
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from src.tasks.velocity.mdp.sgs import SGSConfig, SuccessGuidedSampler, VelocityTrackingOutcome
from src.tasks.velocity.mdp.sgs_velocity_command import SGSVelocityCommandCfg, reset_sgs_root


class SamplerTests(unittest.TestCase):
  def test_beta_kernel_prefers_frontier_and_keeps_uniform_floor(self):
    cfg = SGSConfig(history_length=100, target_success=.5, uniform_mix=.2)
    sampler = SuccessGuidedSampler(5, cfg)
    for i, n in enumerate((0, 10, 50, 90, 100)):
      sampler.update(torch.full((100,), i), torch.arange(100) < n)
    p = sampler.probabilities()
    self.assertGreater(p[2], p[1])
    self.assertGreater(p[1], p[0])
    torch.testing.assert_close(p[0], p[4])
    self.assertTrue(bool((p >= .2 / 5).all()))
    self.assertAlmostEqual(float(p.sum()), 1.)

  def test_duplicate_updates_match_a_sequential_ring_reference(self):
    sampler = SuccessGuidedSampler(7, SGSConfig(history_length=4))
    ref = [[] for _ in range(7)]
    generator = torch.Generator().manual_seed(8)
    for n in (1, 20, 70, 3, 100):
      ids = torch.randint(7, (n,), generator=generator)
      values = torch.rand(n, generator=generator) > .4
      sampler.update(ids, values)
      for i, value in zip(ids.tolist(), values.tolist()):
        ref[i].append(value)
      expected = torch.tensor([sum(v[-4:]) / len(v[-4:]) if v else .66 for v in ref])
      torch.testing.assert_close(sampler.rates(), expected)
      self.assertEqual(sampler.counts.tolist(), [len(v) for v in ref])

  def test_checkpoint_preserves_history_and_next_draw(self):
    sampler = SuccessGuidedSampler(3, SGSConfig(), seed=42)
    sampler.update(torch.tensor([0, 1, 1]), torch.tensor([True, False, True]))
    saved = sampler.state_dict()
    clone = SuccessGuidedSampler(3, SGSConfig(), seed=99)
    clone.load_state_dict(saved)
    torch.testing.assert_close(sampler.sample(200), clone.sample(200))
    sampler.update(torch.tensor([0]), torch.tensor([False]))
    self.assertEqual(int(saved["counts"][0]), 1)  # a snapshot, not aliased storage

  def test_uniform_control_and_invalid_checkpoint(self):
    sampler = SuccessGuidedSampler(3, SGSConfig(adaptive=False))
    sampler.update(torch.tensor([0]), torch.tensor([True]))
    torch.testing.assert_close(sampler.probabilities(), torch.full((3,), 1 / 3))
    state = sampler.state_dict()
    state["counts"][0] = -1
    with self.assertRaises(ValueError):
      sampler.load_state_dict(state)
    with self.assertRaises(ValueError):
      sampler.update(torch.tensor([3]), torch.tensor([False]))


class OutcomeTests(unittest.TestCase):
  def make_tracker(self, n=1, **kwargs):
    return VelocityTrackingOutcome(n, SGSConfig(settle_s=.2, min_segment_s=.2, **kwargs), .1)

  def test_good_tracking_passes_and_stationary_low_speed_does_not(self):
    t = self.make_tracker(4)
    command = torch.tensor([[.15, 0., 0.], [.15, 0., 0.], [0., 0., .1], [0., 0., 0.]])
    actual = torch.tensor([[.15, 0., 0.], [0., 0., 0.], [0., 0., 0.], [.11, 0., 0.]])
    for _ in range(10):
      t.observe(command, actual, torch.zeros(4, 3))
    t.finish_segments(torch.arange(4))
    self.assertEqual(t.success(torch.arange(4)).tolist(), [True, False, False, False])

  def test_settling_and_heading_changes_do_not_restart_timer(self):
    t = self.make_tracker()
    for i in range(8):
      command = torch.tensor([[.3, 0., .2 + .01 * i]])
      velocity = command.clone() if i >= 2 else torch.full((1, 3), 9.)
      angular = torch.zeros(1, 3)
      angular[:, 2] = velocity[:, 2]
      t.observe(command, velocity, angular)
    self.assertEqual(int(t.samples[0]), 6)
    t.finish_segments(torch.tensor([0]))
    self.assertTrue(bool(t.success(torch.tensor([0]))[0]))

  def test_failed_segment_cannot_be_hidden_by_later_success(self):
    t = self.make_tracker()
    cmd = torch.tensor([[.3, 0., 0.]])
    for _ in range(6):
      t.observe(cmd, torch.zeros(1, 3), torch.zeros(1, 3))
    t.finish_segments(torch.tensor([0]))
    for _ in range(15):
      t.observe(cmd, cmd, torch.zeros(1, 3))
    t.finish_segments(torch.tensor([0]))
    self.assertFalse(bool(t.success(torch.tensor([0]))[0]))

  def test_no_scored_segment_and_nonfinite_states_are_not_successes(self):
    t = self.make_tracker(2)
    t.observe(torch.zeros(2, 3), torch.full((2, 3), float("nan")), torch.zeros(2, 3))
    t.finish_segments(torch.arange(2))
    self.assertFalse(bool(t.success(torch.arange(2)).any()))
    t.reset(torch.tensor([0]))
    self.assertTrue(bool(t.finite[0]))
    self.assertFalse(bool(t.finite[1]))


class Scene(dict):
  pass


def make_command(n=5):
  origins = torch.tensor([[[0., 0., 0.], [0., 8., 0.]], [[8., 0., 0.], [8., 8., 0.]]])
  terrain = SimpleNamespace(terrain_origins=origins, env_origins=torch.zeros(n, 3),
    terrain_levels=torch.zeros(n, dtype=torch.long), terrain_types=torch.zeros(n, dtype=torch.long),
    cfg=SimpleNamespace(terrain_generator="seeded-test-map"))
  data = SimpleNamespace(heading_w=torch.zeros(n), root_link_pos_w=torch.zeros(n, 3),
    root_link_quat_w=torch.tensor([[1., 0., 0., 0.]]).repeat(n, 1),
    root_link_lin_vel_b=torch.zeros(n, 3), root_link_ang_vel_b=torch.zeros(n, 3),
    default_root_state=torch.zeros(n, 13))
  data.default_root_state[:, 3] = 1
  robot = SimpleNamespace(data=data)
  robot.write_root_link_pose_to_sim = lambda p, env_ids: setattr(robot, "pose", p.clone())
  robot.write_root_link_velocity_to_sim = lambda v, env_ids: setattr(robot, "velocity", v.clone())
  scene = Scene(robot=robot)
  scene.terrain, scene.env_origins = terrain, terrain.env_origins
  done = {key: torch.zeros(n, dtype=torch.bool) for key in ("time_out", "terrain_out_of_bounds")}
  tm = SimpleNamespace(terminated=torch.zeros(n, dtype=torch.bool), get_term=done.__getitem__)
  env = SimpleNamespace(num_envs=n, device="cpu", step_dt=.1, max_episode_length_s=1., max_episode_length=10,
    cfg=SimpleNamespace(seed=5), scene=scene, termination_manager=tm,
    episode_length_buf=torch.zeros(n, dtype=torch.long), reset_buf=torch.zeros(n, dtype=torch.bool))
  cfg = SGSVelocityCommandCfg(entity_name="robot", resampling_time_range=(2., 3.),
    ranges=SGSVelocityCommandCfg.Ranges(lin_vel_x=(-.3, .5), lin_vel_y=(-.3, .3), ang_vel_z=(-.5, .5)),
    sgs=SGSConfig(profiles_per_tile=5, settle_s=.2, min_segment_s=.2))
  command = cfg.build(env)
  env.command_manager = SimpleNamespace(get_term=lambda _: command)
  command.on_reset(torch.arange(n))
  command.reset(torch.arange(n))
  command.compute(0.)
  return command, env


class IntegrationTests(unittest.TestCase):
  def step(self, command, env, *, terminal=False, timeout=False, boundary=False):
    env.episode_length_buf += 1
    env.termination_manager.terminated.fill_(terminal)
    env.termination_manager.get_term("time_out").fill_(timeout)
    env.termination_manager.get_term("terrain_out_of_bounds").fill_(boundary)
    env.reset_buf.fill_(terminal or timeout or boundary)
    command.robot.data.root_link_lin_vel_b.copy_(command.command)
    command.robot.data.root_link_ang_vel_b[:, 2] = command.command[:, 2]
    command.observe_step()

  def test_full_episode_is_credited_before_partial_reset_relocates(self):
    c, e = make_command()
    old_ids = c.config_ids.clone()
    for i in range(10):
      self.step(c, e, timeout=i == 9)
    c.on_reset(torch.tensor([1, 3]))
    self.assertEqual(c.recorded, 2)
    self.assertEqual(int(c.sampler.counts.sum()), 2)
    expected = torch.bincount(old_ids[[1, 3]], minlength=c.bank_size)
    torch.testing.assert_close(c.sampler.counts, expected)
    self.assertTrue(bool(c.sampler.history.any()))
    self.assertEqual(c.outcome.steps.tolist(), [10, 0, 10, 0, 10])
    tile = c.config_ids[[1, 3]] // c.cfg.sgs.profiles_per_tile
    torch.testing.assert_close(e.scene.env_origins[[1, 3]], e.scene.terrain.terrain_origins[tile // 2, tile % 2])
    reset_sgs_root(e, torch.tensor([1, 3]))
    torch.testing.assert_close(c.robot.pose[:, :2], e.scene.env_origins[[1, 3], :2] + c.bank["spawn"][c.config_ids[[1, 3]], :2])

  def test_fall_during_settling_is_a_failure(self):
    c, e = make_command(1)
    self.step(c, e, terminal=True)
    c.on_reset(torch.tensor([0]))
    self.assertEqual(c.recorded, 1)
    self.assertFalse(bool(c.sampler.history.any()))

  def test_boundary_manual_and_short_initial_timeouts_are_excluded(self):
    for reason in ("boundary", "manual", "short", "nan"):
      with self.subTest(reason=reason):
        c, e = make_command(1)
        if reason == "short":
          e.episode_length_buf.fill_(7)
        self.step(c, e, timeout=reason != "manual", boundary=reason == "boundary")
        if reason == "nan":
          c.outcome.finite.fill_(False)
        c.on_reset(torch.tensor([0]))
        self.assertEqual(c.recorded, 0)
        self.assertEqual(c.skipped, 1)

  def test_bank_and_rng_round_trip_and_changed_bank_rejected(self):
    c, _ = make_command()
    saved = c.sgs_state_dict()
    other, _ = make_command()
    other.load_sgs_state_dict(saved)
    torch.testing.assert_close(c.sampler.sample(100), other.sampler.sample(100))
    self.assertTrue(bool(other._censored.all()))
    saved["bank"]["commands"][0, 0, 0] = .1
    with self.assertRaises(ValueError):
      other.load_sgs_state_dict(saved)

  def test_resampling_closes_segment_and_replays_the_fixed_profile(self):
    c, e = make_command(1)
    c.config_ids[0] = 1
    c.profile_step[0] = 0
    c._resample(torch.tensor([0]))
    first = c.command.clone()
    duration = c.time_left.clone()
    for _ in range(6):
      self.step(c, e)
    c._resample(torch.tensor([0]))
    self.assertEqual(int(c.outcome.segments[0]), 1)
    self.assertEqual(int(c.outcome.age[0]), 0)
    c.profile_step[0] = 0
    c._resample(torch.tensor([0]))
    torch.testing.assert_close(c.command, first)
    torch.testing.assert_close(c.time_left, duration)

  def test_manual_joystick_censors_only_the_selected_environment(self):
    c, e = make_command(2)
    c._joystick_enabled = SimpleNamespace(value=True)
    c._joystick_get_env_idx = lambda: 1
    self.step(c, e)
    self.assertEqual(c._censored.tolist(), [False, True])

  def test_runner_persists_history_but_evaluation_and_actor_warm_starts_do_not_load_it(self):
    from src.tasks.velocity.rl.sgs_runner import Go2WSGSRunner
    from src.tasks.velocity.rl.go2w_runner import Go2WOnPolicyRunner
    command, env = make_command()
    runner = object.__new__(Go2WSGSRunner)
    runner.env = SimpleNamespace(unwrapped=env)
    with patch.object(Go2WOnPolicyRunner, "save") as parent:
      runner.save("unused.pt", {"other": 3})
      info = parent.call_args.args[1]
      self.assertIn("sgs_state", info)
      self.assertEqual(info["other"], 3)
    with patch.object(Go2WOnPolicyRunner, "load", return_value=info):
      runner.load("unused.pt")
      self.assertTrue(bool(command._censored.all()))
    with patch.object(Go2WOnPolicyRunner, "load", return_value={"sgs_state": {}}):
      runner.load("unused.pt", load_cfg={"actor": True})
      command.cfg.evaluation = True
      runner.load("unused.pt")
    with self.assertRaises(ValueError):
      runner.save("unused.pt")

  def test_task_registration_preserves_v2_and_uniform_control(self):
    import src.tasks  # noqa: F401
    from mjlab.tasks.registry import load_env_cfg, load_runner_cls
    from src.tasks.velocity.rl.sgs_runner import Go2WSGSRunner
    base = load_env_cfg("Unitree-Go2W-Perceptive-Rough-v2")
    sgs = load_env_cfg("Unitree-Go2W-Perceptive-Rough-v2-SGS")
    control = load_env_cfg("Unitree-Go2W-Perceptive-Rough-v2-SGS-Uniform")
    play = load_env_cfg("Unitree-Go2W-Perceptive-Rough-v2-SGS", play=True)
    self.assertIn("terrain_levels", base.curriculum)
    self.assertIn("command_vel", base.curriculum)
    self.assertNotIn("terrain_levels", sgs.curriculum)
    self.assertNotIn("command_vel", sgs.curriculum)
    self.assertEqual(asdict(sgs.rewards["track_linear_velocity"]), asdict(base.rewards["track_linear_velocity"]))
    self.assertEqual(asdict(sgs.actions["wheel_vel"]), asdict(base.actions["wheel_vel"]))
    self.assertEqual(sgs.observations["actor"].history_length, base.observations["actor"].history_length)
    self.assertEqual(sgs.commands["twist"].ranges, control.commands["twist"].ranges)
    self.assertTrue(sgs.commands["twist"].sgs.adaptive)
    self.assertFalse(control.commands["twist"].sgs.adaptive)
    self.assertFalse(play.commands["twist"].sgs.adaptive)
    self.assertNotEqual(sgs.commands["twist"].sgs.bank_seed, play.commands["twist"].sgs.bank_seed)
    self.assertTrue(play.commands["twist"].evaluation)
    self.assertEqual(play.episode_length_s, 20.)
    self.assertIs(load_runner_cls("Unitree-Go2W-Perceptive-Rough-v2-SGS"), Go2WSGSRunner)

  def test_multi_process_use_fails_explicitly(self):
    from src.tasks.velocity.rl.sgs_runner import Go2WSGSRunner
    with patch.dict("os.environ", {"WORLD_SIZE": "2"}):
      with self.assertRaisesRegex(ValueError, "one training process"):
        Go2WSGSRunner()


if __name__ == "__main__":
  unittest.main()
