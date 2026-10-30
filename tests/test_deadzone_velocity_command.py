"""The project-local command preserves mjlab behavior except the per-axis dead zone."""

from dataclasses import fields
import math
from types import SimpleNamespace
import unittest

import torch
from mjlab.tasks.registry import list_tasks, load_env_cfg
from mjlab.tasks.velocity.mdp.velocity_command import (
  UniformVelocityCommandCfg as NativeCommandCfg,
)

import src.tasks  # noqa: F401
from src.tasks.velocity.mdp.deadzone_velocity_command import (
  DeadzoneVelocityCommand, DeadzoneVelocityCommandCfg,
)


class DeadzoneVelocityCommandTests(unittest.TestCase):
  def make_command(self, cfg_type=DeadzoneVelocityCommandCfg, num_envs=8, **options):
    heading = options.get("heading_command", False)
    ranges = options.pop("ranges", NativeCommandCfg.Ranges(
      lin_vel_x=(-.3, .3), lin_vel_y=(-.3, .3), ang_vel_z=(-.3, .3),
      heading=(-math.pi, math.pi) if heading else None,
    ))
    cfg = cfg_type(entity_name="robot", resampling_time_range=(3., 8.), ranges=ranges, **options)
    robot = SimpleNamespace(data=SimpleNamespace(
      heading_w=torch.zeros(num_envs), root_link_pos_w=torch.zeros(num_envs, 3),
      root_link_lin_vel_b=torch.zeros(num_envs, 3), root_link_ang_vel_b=torch.zeros(num_envs, 3),
      forward_vec_b=torch.tensor([[1., 0., 0.]]).expand(num_envs, -1),
    ), indexing=SimpleNamespace(free_joint_q_adr=list(range(7))))
    qpos = torch.zeros(num_envs, 7)
    qpos[:, 3] = 1.
    env = SimpleNamespace(num_envs=num_envs, device="cpu", step_dt=.02,
      scene={"robot": robot}, sim=SimpleNamespace(data=SimpleNamespace(qpos=qpos)),
      episode_length_buf=torch.ones(num_envs, dtype=torch.long), writes=[])
    robot.write_root_link_velocity_b_to_sim = lambda velocity, env_ids: env.writes.append(
      (velocity.clone(), env_ids.clone()))
    return cfg.build(env)

  def test_independent_axis_dead_zone_preserves_both_threshold_boundaries(self):
    command = self.make_command(num_envs=4)
    command.vel_command_b[:] = torch.tensor([
      [-.1001, .1, .0999], [.1, -.1, -.0999], [.09, .09, .09], [0., .4, -.4],
    ])
    command._apply_deadzone(slice(None))
    torch.testing.assert_close(command.command, torch.tensor([
      [-.1001, .1, 0.], [.1, -.1, 0.], [0., 0., 0.], [0., .4, -.4],
    ]))

  def test_resampling_matches_native_random_state_and_all_mode_assignments(self):
    options = dict(heading_command=True, rel_heading_envs=.7, rel_standing_envs=.2,
                   rel_world_envs=.4, rel_forward_envs=.2)
    native = self.make_command(NativeCommandCfg, num_envs=64, **options)
    local = self.make_command(num_envs=64, **options)
    ids = torch.arange(64)
    torch.manual_seed(123)
    native._resample_command(ids)
    rng_state = torch.get_rng_state()
    torch.manual_seed(123)
    local._resample_command(ids)
    torch.testing.assert_close(torch.get_rng_state(), rng_state)
    expected = native.command * (native.command.abs() >= .1)
    torch.testing.assert_close(local.command, expected)
    for field in ("heading_target", "is_heading_env", "is_standing_env", "is_world_env",
                  "is_forward_env", "vel_command_w"):
      torch.testing.assert_close(getattr(local, field), getattr(native, field), msg=field)
    # World references retain the raw sample, before rotation into the body frame.
    world_reference = native.vel_command_w.clone()
    for tick in range(5):
      heading = torch.linspace(-1., 1., 64) * tick
      native._resolve_command(heading)
      local._resolve_command(heading)
      torch.testing.assert_close(local.command, native.command * (native.command.abs() >= .1))
    torch.testing.assert_close(local.vel_command_w, native.vel_command_w)
    torch.testing.assert_close(local.vel_command_w[~local.is_standing_env],
                               world_reference[~local.is_standing_env])

  def test_partial_resample_and_resolve_preserve_unselected_environments(self):
    command = self.make_command(heading_command=True)
    command.vel_command_b.fill_(.05)
    command.vel_command_w.fill_(.05)
    command.is_heading_env.fill_(True)
    command.heading_target.fill_(.1)
    before = command.command.clone()
    command._resample_command(torch.tensor([2, 5]))
    untouched = torch.tensor([0, 1, 3, 4, 6, 7])
    torch.testing.assert_close(command.command[untouched], before[untouched])
    command._resolve_command(torch.zeros(8), torch.tensor([2, 5]))
    torch.testing.assert_close(command.command[untouched], before[untouched])

  def test_heading_updates_cannot_recreate_subthreshold_yaw_commands(self):
    command = self.make_command(num_envs=4, heading_command=True, heading_control_stiffness=.5)
    command.is_heading_env.fill_(True)
    command.heading_target[:] = torch.tensor([.1, -.1, .4, -.4])
    command._update_command()
    torch.testing.assert_close(command.command[:, 2], torch.tensor([0., 0., .2, -.2]))
    command.heading_target[:] = torch.tensor([.4, -.4, .1, -.1])
    command._update_command()
    torch.testing.assert_close(command.command[:, 2], torch.tensor([.2, -.2, 0., 0.]))

  def test_world_rotation_keeps_raw_reference_and_filters_final_body_components(self):
    command = self.make_command(num_envs=1)
    command.is_world_env.fill_(True)
    command.vel_command_w[0] = torch.tensor([.2, 0., 0.])
    command._resolve_command(torch.tensor([math.pi / 8]))
    torch.testing.assert_close(command.command, torch.tensor([[.2 * math.cos(math.pi / 8), 0., 0.]]))
    command._resolve_command(torch.tensor([math.pi / 2]))
    torch.testing.assert_close(command.command, torch.tensor([[0., -.2, 0.]]))
    torch.testing.assert_close(command.vel_command_w, torch.tensor([[.2, 0., 0.]]))

  def test_forward_mode_preserves_native_minimum_and_heading_world_exclusion(self):
    command = self.make_command(heading_command=True, rel_heading_envs=1.,
      rel_world_envs=1., rel_forward_envs=1., ranges=NativeCommandCfg.Ranges(
        lin_vel_x=(-.05, -.05), lin_vel_y=(.2, .2), ang_vel_z=(.2, .2), heading=(-1., 1.)))
    command._resample_command(torch.arange(8))
    command._update_command()
    torch.testing.assert_close(command.command, torch.tensor([[.3, 0., 0.]]).expand(8, -1))
    self.assertTrue(command.is_forward_env.all())
    self.assertFalse(command.is_heading_env.any())
    self.assertFalse(command.is_world_env.any())

  def test_standing_still_overrides_every_command_component(self):
    command = self.make_command(heading_command=True)
    command.is_standing_env.fill_(True)
    command.is_heading_env.fill_(True)
    command.is_world_env.fill_(True)
    command.vel_command_b.fill_(.4)
    command.vel_command_w.fill_(.4)
    command.heading_target.fill_(1.)
    command._update_command()
    torch.testing.assert_close(command.command, torch.zeros(8, 3))
    torch.testing.assert_close(command.vel_command_w, torch.zeros(8, 3))

  def test_reset_initial_velocity_uses_the_filtered_command(self):
    command = self.make_command(init_velocity_prob=1., ranges=NativeCommandCfg.Ranges(
      lin_vel_x=(.05, .05), lin_vel_y=(-.2, -.2), ang_vel_z=(-.09, -.09)))
    command.reset(torch.tensor([1, 4]))
    velocity, ids = command._env.writes[0]
    torch.testing.assert_close(ids, torch.tensor([1, 4]))
    expected = torch.tensor([[0., -.2, 0., 0., 0., 0.]]).expand(2, -1)
    torch.testing.assert_close(velocity, expected)
    torch.testing.assert_close(command.command[ids], expected[:, [0, 1, 5]])

  def test_viewer_commands_obey_the_same_dead_zone(self):
    command = self.make_command()
    command.time_left.fill_(10.)
    command._joystick_enabled = SimpleNamespace(value=True)
    command._joystick_get_env_idx = lambda: 3
    command._joystick_sliders = [SimpleNamespace(value=value) for value in (.05, -.1, .4)]
    command.compute(.02)
    torch.testing.assert_close(command.command[3], torch.tensor([0., -.1, .4]))

  def test_command_integral_accumulates_the_filtered_previous_command(self):
    command = self.make_command(num_envs=2)
    command.time_left.fill_(10.)
    command.vel_command_b[:] = torch.tensor([[.05, -.2, .09], [-.3, .09, -.05]])
    command._apply_deadzone(slice(None))
    command.compute(.02)
    torch.testing.assert_close(command.commanded_displacement_w,
                               torch.tensor([[0., -.004], [-.006, 0.]]))

  def test_dead_zone_preserves_left_right_command_symmetry(self):
    command = self.make_command(num_envs=2)
    value = torch.tensor([[.099, -.1001, .1]])
    command.vel_command_b[:] = torch.cat((value, value * torch.tensor([1., -1., -1.])))
    command._apply_deadzone(slice(None))
    torch.testing.assert_close(command.command[1], command.command[0] * torch.tensor([1., -1., -1.]))

  def test_all_go2w_variants_use_local_command_and_g1_keeps_native_command(self):
    self.assertEqual({f.name for f in fields(DeadzoneVelocityCommandCfg)},
                     {f.name for f in fields(NativeCommandCfg)})
    for task in list_tasks():
      if not task.startswith("Unitree-Go2W-"):
        continue
      for play in (False, True):
        with self.subTest(task=task, play=play):
          cfg = load_env_cfg(task, play=play).commands["twist"]
          self.assertIsInstance(cfg, DeadzoneVelocityCommandCfg)
          self.assertIs(type(cfg.ranges), DeadzoneVelocityCommandCfg.Ranges)
          self.assertIs(type(cfg.viz), DeadzoneVelocityCommandCfg.VizCfg)
          # SGS extends this command with a terrain/profile bank; its real reset
          # lifecycle is covered in test_sgs rather than this command-only stub.
          if type(cfg) is DeadzoneVelocityCommandCfg:
            self.assertIs(type(cfg.build(self.make_command()._env)), DeadzoneVelocityCommand)
    self.assertIs(type(load_env_cfg("Unitree-G1-Perceptive-Rough-v2").commands["twist"]),
                  NativeCommandCfg)


if __name__ == "__main__":
  unittest.main()
