"""Wheel landings must query the aligned, valid MID360 height grid."""

from types import SimpleNamespace
import unittest

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg
from mjlab.utils.lab_api.math import quat_apply, quat_from_euler_xyz, yaw_quat

import src.tasks  # noqa: F401
from src.assets.robots.unitree_go2w.go2w_constants import WHEEL_SITES, get_go2w_robot_cfg
from src.tasks.velocity.config.go2w.env_cfgs import SCAN
from src.tasks.velocity.mdp.rewards.locomotion import feet_landing_edge_penalty


class Go2WLandingEdgeTests(unittest.TestCase):
  def make_env(self, landings=None, angles=(0., 0., .8), commands=None):
    landings = torch.tensor([[1., 0., 0., 0.]] if landings is None else landings)
    batch = landings.shape[0]
    x = torch.arange(17) * .1 - .45
    terrain_z = torch.zeros(batch, 11, 17)
    terrain_z[:, :, x >= .249] = .14
    origin = torch.tensor([[5., -3., .6]]).expand(batch, -1).clone()
    quat = quat_from_euler_xyz(*(torch.full((batch,), value) for value in angles))
    feet = torch.tensor([[[.25, .2, -.3], [.25, -.2, -.3],
                          [-.25, .2, -.3], [-.25, -.2, -.3]]]).expand(batch, -1, -1)
    rotation = yaw_quat(quat)[:, None].expand(-1, 4, -1)
    feet_w = quat_apply(rotation.reshape(-1, 4), feet.reshape(-1, 3)).reshape(batch, 4, 3)
    hit = torch.zeros(batch, 187, 3)
    hit[..., 2] = terrain_z.flatten(1)
    ray = SimpleNamespace(cfg=SimpleNamespace(ray_alignment="yaw"), data=SimpleNamespace(
      pos_w=origin, quat_w=quat, hit_pos_w=hit, distances=torch.ones(batch, 187),
    ))
    contact = SimpleNamespace(compute_first_contact=lambda dt: landings)
    command = torch.tensor([[.4, 0., 0.]]).expand(batch, -1).clone() if commands is None else torch.tensor(commands)
    return SimpleNamespace(
      num_envs=batch, step_dt=.02, extras={"log": {}},
      command_manager=SimpleNamespace(get_command=lambda name: command),
      scene={"scan": ray, "contact": contact, "robot": SimpleNamespace(data=SimpleNamespace(
        site_pos_w=feet_w + origin[:, None], root_link_pos_w=origin, root_link_quat_w=quat,
      ))},
    )

  def cost(self, env, **params):
    return feet_landing_edge_penalty(
      env, "contact", "scan", "twist", **SCAN,
      asset_cfg=SceneEntityCfg("robot", site_ids=[0, 1, 2, 3]), **params,
    )

  def test_first_landing_penalizes_height_change_above_threshold(self):
    env = self.make_env()
    torch.testing.assert_close(self.cost(env), torch.tensor([(.14 - .06)**2]))
    torch.testing.assert_close(env.extras["log"]["Metrics/landing_edge_score_mean"], torch.tensor(.14))
    torch.testing.assert_close(env.extras["log"]["Metrics/landing_edge_valid_ratio"], torch.tensor(1.))
    torch.testing.assert_close(self.cost(env, edge_threshold=.15), torch.zeros(1))

  def test_flat_ground_and_continuous_rolling_have_no_landing_penalty(self):
    env = self.make_env()
    env.scene["scan"].data.hit_pos_w[..., 2] = 0.
    torch.testing.assert_close(self.cost(env), torch.zeros(1))
    torch.testing.assert_close(self.cost(self.make_env(landings=[[0., 0., 0., 0.]])), torch.zeros(1))

  def test_contact_columns_match_fl_fr_rl_rr_sites(self):
    env = self.make_env(landings=torch.eye(4).tolist())
    torch.testing.assert_close(self.cost(env), torch.tensor([.0064, .0064, 0., 0.]))

  def test_pitch_and_roll_do_not_shift_the_yaw_aligned_grid_query(self):
    tilted = self.make_env(angles=(.35, .6, .8))
    torch.testing.assert_close(self.cost(tilted), self.cost(self.make_env()))

  def test_scan_origin_can_differ_from_the_robot_root(self):
    env = self.make_env()
    env.scene["robot"].data.root_link_pos_w = torch.tensor([[50., 20., 5.]])
    torch.testing.assert_close(self.cost(env), torch.tensor([.0064]))

  def test_forward_command_gate_excludes_reverse_stationary_and_turning_in_place(self):
    commands = [[0., 0., 0.], [.1, 0., 0.], [.11, 0., 0.], [-.4, 0., 0.], [0., 0., .5]]
    env = self.make_env(landings=[[1., 0., 0., 0.]] * 5, commands=commands)
    torch.testing.assert_close(self.cost(env), torch.tensor([0., 0., .0064, 0., 0.]))

  def test_missing_center_or_neighbors_do_not_invent_edges(self):
    env = self.make_env()
    data = env.scene["scan"].data
    # Front-left wheel is at row 7, column 7. The left neighbor is the low tread.
    data.distances.reshape(1, 11, 17)[0, 7, 6] = -1.
    torch.testing.assert_close(self.cost(env), torch.zeros(1))
    data.distances.reshape(1, 11, 17)[0, 7, 7] = -1.
    torch.testing.assert_close(self.cost(env), torch.zeros(1))
    torch.testing.assert_close(env.extras["log"]["Metrics/landing_edge_valid_ratio"], torch.tensor(0.))

  def test_sites_outside_the_grid_are_not_clamped_to_an_edge(self):
    env = self.make_env()
    ray = env.scene["scan"]
    yaw = yaw_quat(ray.data.quat_w)
    outside = torch.tensor([[2., .2, -.3]])
    env.scene["robot"].data.site_pos_w[0, 0] = quat_apply(yaw, outside)[0] + ray.data.pos_w[0]
    ray.data.hit_pos_w.reshape(1, 11, 17, 3)[:, :, -1, 2] = .5
    torch.testing.assert_close(self.cost(env), torch.zeros(1))

  def test_mirrored_sites_contacts_and_scan_have_equal_cost(self):
    env = self.make_env(landings=[[1., 0., 0., 0.], [0., 1., 0., 0.]])
    force_grid = env.scene["scan"].data.hit_pos_w[..., 2].reshape(2, 11, 17)
    force_grid[0, 7, 7] = .25
    force_grid[1] = force_grid[0].flip(0)
    value = self.cost(env)
    torch.testing.assert_close(value[:1], value[1:])

  def test_v2_configuration_uses_wheel_sites_and_matching_scan_parameters(self):
    robot = Entity(get_go2w_robot_cfg())
    for task in list_tasks():
      if not task.startswith("Unitree-Go2W-"):
        continue
      for play in (False, True):
        with self.subTest(task=task, play=play):
          cfg = load_env_cfg(task, play=play)
          if not task.startswith("Unitree-Go2W-Perceptive-Rough-v2"):
            self.assertNotIn("landing_edge_penalty", cfg.rewards)
            continue
          term = cfg.rewards["landing_edge_penalty"]
          self.assertIs(term.func, feet_landing_edge_penalty)
          self.assertEqual(term.weight, -1.)
          params = term.params
          self.assertEqual({name: params[name] for name in SCAN}, SCAN)
          sites = params["asset_cfg"]
          self.assertTrue(sites.preserve_order)
          sites.resolve({"robot": robot})
          self.assertEqual(tuple(sites.site_names), WHEEL_SITES)
          contact = next(s for s in cfg.scene.sensors if s.name == params["contact_sensor_name"])
          self.assertTrue(contact.track_air_time)
          primaries = contact.build()._resolve_primary_names({"robot": robot}, contact.primary)
          self.assertEqual(tuple(name[:2] for name in primaries), WHEEL_SITES)
          ray = next(s for s in cfg.scene.sensors if s.name == params["terrain_sensor_name"])
          self.assertEqual(ray.ray_alignment, "yaw")
          self.assertEqual(ray.pattern.size, (params["size_x_m"], params["size_y_m"]))
          self.assertEqual(ray.pattern.resolution, params["resolution_m"])
          self.assertEqual(ray.pattern.x_offset, params["x_offset_m"])
          self.assertEqual(ray.pattern.y_offset, params["y_offset_m"])


if __name__ == "__main__":
  unittest.main()
