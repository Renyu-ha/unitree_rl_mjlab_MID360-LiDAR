"""CPU numerical checks without importing the GPU simulation stack."""

import importlib.util
import math
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
  "go2w_terrain_rewards", ROOT / "src/tasks/velocity/mdp/rewards/terrain_aware.py",
)
rewards = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rewards)

WHEEL_RAYS = tuple(f"height_{leg}" for leg in ("FL", "FR", "RL", "RR"))
SELECTION = SimpleNamespace(name="robot", site_ids=[0, 1, 2, 3],
                            body_ids=[0, 1, 2, 3], joint_ids=list(range(12)))


class TerrainAwareRewardTests(unittest.TestCase):
  def make_env(self, batch=1):
    sites = torch.tensor([[[.25, .2, .086], [.25, -.2, .086],
                           [-.25, .2, .086], [-.25, -.2, .086]]]).repeat(batch, 1, 1)
    quat = torch.tensor([[1., 0., 0., 0.]]).repeat(batch, 1)
    data = SimpleNamespace(
      site_pos_w=sites, body_link_quat_w=quat[:, None].repeat(1, 4, 1),
      root_link_pos_w=torch.tensor([[0., 0., .4]]).repeat(batch, 1),
      root_link_quat_w=quat, projected_gravity_b=torch.tensor([[0., 0., -1.]]).repeat(batch, 1),
      root_link_lin_vel_b=torch.zeros(batch, 3), joint_pos=torch.zeros(batch, 12),
      default_joint_pos=torch.zeros(batch, 12), joint_vel=torch.zeros(batch, 12),
    )
    scene = {"robot": SimpleNamespace(data=data), "terrain": SimpleNamespace(
      cfg=SimpleNamespace(ray_alignment="yaw", pattern=SimpleNamespace(size=(1.6, 1.), resolution=.1)),
      data=SimpleNamespace(pos_w=data.root_link_pos_w.clone(), quat_w=quat.clone(),
                           hit_pos_w=torch.zeros(batch, 187, 3), distances=torch.ones(batch, 187)),
    )}
    for name in WHEEL_RAYS:
      scene[name] = SimpleNamespace(data=SimpleNamespace(
        hit_pos_w=torch.zeros(batch, 1, 3), distances=torch.full((batch, 1), .086),
      ))
    command = torch.tensor([[.4, 0., 0.]]).repeat(batch, 1)
    return SimpleNamespace(num_envs=batch, scene=scene, command=command, extras={"log": {}},
                           command_manager=SimpleNamespace(get_command=lambda name: command))

  def call(self, name, env):
    params = {"terrain_sensor_name": "terrain", "asset_cfg": SELECTION}
    if name != "leg_posture":
      params["command_name"] = "twist"
    if name in ("unnecessary_wheel_lift", "wheel_obstacle_clearance"):
      params.update(wheel_sensor_names=WHEEL_RAYS, wheel_radius=.086, wheel_half_width=.0225)
    return getattr(rewards, name)(env, **params)

  def step(self, env, height=.14, x=.35):
    xs = torch.arange(17)*.1-.8
    env.scene["terrain"].data.hit_pos_w[..., 2].reshape(env.num_envs, 11, 17)[:, :, xs >= x] = height

  def test_default_pose_rolling_and_flat_progress_are_free(self):
    env = self.make_env()
    for name in rewards.__all__:
      torch.testing.assert_close(self.call(name, env), torch.zeros(1))

  def test_posture_constrains_hip_but_relaxes_obstacle_side_flexion(self):
    env = self.make_env()
    env.scene["robot"].data.joint_pos.fill_(1.)
    torch.testing.assert_close(self.call("leg_posture", env), torch.tensor([16.]))
    self.step(env)
    # Only front wheel patches see the edge: front flexion x .125; rear x 1.
    torch.testing.assert_close(self.call("leg_posture", env), torch.tensor([12.5]))
    env.scene["robot"].data.joint_pos[:, 1::3] = 0.
    env.scene["robot"].data.joint_pos[:, 2::3] = 0.
    torch.testing.assert_close(self.call("leg_posture", env), torch.tensor([8.]))

  def test_each_lift_uses_its_own_ground_sensor_and_five_mm_tolerance(self):
    env = self.make_env()
    sites = env.scene["robot"].data.site_pos_w
    sites[0, :, 2] += torch.tensor([.055, .005, .3, .4])
    env.scene[WHEEL_RAYS[2]].data.hit_pos_w[..., 2] = .3
    env.scene[WHEEL_RAYS[3]].data.hit_pos_w[..., 2] = .4
    torch.testing.assert_close(self.call("unnecessary_wheel_lift", env), torch.ones(1))

  def test_missing_wheel_ray_uses_terrain_truth_instead_of_disabling_lift(self):
    env = self.make_env()
    env.scene["robot"].data.site_pos_w[..., 2] += .055
    env.scene[WHEEL_RAYS[1]].data.distances.fill_(-1.)
    env.scene[WHEEL_RAYS[1]].data.hit_pos_w[..., 2] = float("nan")
    torch.testing.assert_close(self.call("unnecessary_wheel_lift", env), torch.tensor([4.]))
    torch.testing.assert_close(env.extras["log"]["Metrics/wheel_height_truth_fallback_fraction"], torch.tensor(.25))

  def test_motion_is_mean_squared_leg_speed_and_relaxes_per_wheel(self):
    env = self.make_env()
    env.scene["robot"].data.joint_vel.fill_(2.)
    torch.testing.assert_close(self.call("rolling_leg_motion", env), torch.tensor([4.]))
    self.step(env)
    torch.testing.assert_close(self.call("rolling_leg_motion", env), torch.tensor([2.]))

  def test_wheel_ray_backface_uses_top_surface_truth(self):
    env = self.make_env()
    env.scene["robot"].data.site_pos_w[0, 1, 2] += .055
    ray = env.scene[WHEEL_RAYS[1]].data
    ray.hit_pos_w[..., 2] = -1.
    ray.normals_w = torch.tensor([[[0., 0., -1.]]])
    torch.testing.assert_close(self.call("unnecessary_wheel_lift", env), torch.ones(1))
    torch.testing.assert_close(env.extras["log"]["Metrics/wheel_height_truth_fallback_fraction"], torch.tensor(.25))

  def test_lateral_yaw_and_tilt_relax_motion_without_reversing_sign(self):
    env = self.make_env(batch=4)
    env.scene["robot"].data.joint_vel.fill_(1.)
    env.command[1, 1] = .25
    env.command[2, 2] = 1.
    env.scene["robot"].data.projected_gravity_b[3] = torch.tensor([.3, 0., -math.sqrt(.91)])
    torch.testing.assert_close(self.call("rolling_leg_motion", env),
                               torch.tensor([1., math.exp(-1), math.exp(-1), .2]))

  def test_progress_is_signed_capped_and_ignores_perpendicular_motion(self):
    env = self.make_env(batch=6)
    self.step(env)
    env.scene["robot"].data.root_link_lin_vel_b[:, :2] = torch.tensor(
      [[0., 0.], [.2, 0.], [.4, 0.], [.8, 0.], [-.4, 0.], [0., 1.]],
    )
    torch.testing.assert_close(self.call("obstacle_progress", env), torch.tensor([0., .5, 1., 1., -1., 0.]))

  def test_reverse_lateral_small_commands_and_pure_yaw(self):
    env = self.make_env(batch=5)
    self.step(env)
    env.command[:] = torch.tensor([[-.4, 0., 0.], [0., .4, 0.], [.15, 0., 0.], [.1, 0., 0.], [0., 0., 1.]])
    env.scene["robot"].data.root_link_lin_vel_b[:, :2] = env.command[:, :2]
    torch.testing.assert_close(self.call("obstacle_progress", env), torch.tensor([1., 1., .75, 0., 0.]))

  def test_clearance_penalizes_only_approaching_up_step_and_saturates(self):
    env = self.make_env()
    self.step(env)
    value = self.call("wheel_obstacle_clearance", env)
    self.assertGreater(value.item(), 0.)
    self.assertLess(value.item(), .5)
    env.command[:, 0] = -.4
    torch.testing.assert_close(self.call("wheel_obstacle_clearance", env), torch.zeros(1))
    env.command[:, 0] = .4
    env.scene["robot"].data.site_pos_w[:, :2, 2] += .155
    torch.testing.assert_close(self.call("wheel_obstacle_clearance", env), torch.zeros(1))

  def test_down_step_and_pure_yaw_do_not_request_extra_clearance(self):
    env = self.make_env()
    self.step(env, height=-.14)
    torch.testing.assert_close(self.call("wheel_obstacle_clearance", env), torch.zeros(1))
    self.step(env)
    env.command[:] = torch.tensor([[0., 0., 1.]])
    torch.testing.assert_close(self.call("wheel_obstacle_clearance", env), torch.zeros(1))

  def test_invalid_scan_is_finite_and_creates_no_fake_lift_or_obstacle(self):
    env = self.make_env()
    env.scene["robot"].data.site_pos_w[..., 2] += .2
    env.scene["robot"].data.joint_vel.fill_(2.)
    scan = env.scene["terrain"].data
    scan.hit_pos_w.fill_(float("nan"))
    scan.distances.fill_(-1.)
    for name in rewards.__all__:
      torch.testing.assert_close(self.call(name, env), torch.zeros(1))

  def test_outside_wheel_is_not_clamped_into_an_obstacle(self):
    env = self.make_env()
    self.step(env)
    env.scene["robot"].data.site_pos_w[:, :, 0] = 3.
    for name in ("unnecessary_wheel_lift", "rolling_leg_motion", "wheel_obstacle_clearance"):
      env.scene["robot"].data.joint_vel.fill_(2.)
      torch.testing.assert_close(self.call(name, env), torch.zeros(1))

  def test_world_translation_and_yaw_leave_all_rewards_invariant(self):
    env = self.make_env()
    self.step(env)
    env.scene["robot"].data.joint_pos.fill_(.1)
    env.scene["robot"].data.joint_vel.fill_(1.)
    env.scene["robot"].data.root_link_lin_vel_b[:, 0] = .2
    expected = {name: self.call(name, env).clone() for name in rewards.__all__}
    q = torch.tensor([[math.cos(.7/2), 0., 0., math.sin(.7/2)]])
    shift = torch.tensor([[20., -10., 5.]])
    data = env.scene["robot"].data
    data.site_pos_w = rewards._rotate(q[:, None], data.site_pos_w)+shift[:, None]
    data.root_link_quat_w = q
    data.body_link_quat_w = q[:, None].repeat(1, 4, 1)
    scan = env.scene["terrain"].data
    scan.pos_w = rewards._rotate(q, scan.pos_w)+shift
    scan.quat_w = q
    scan.hit_pos_w[..., 2] += shift[:, None, 2]
    for name in WHEEL_RAYS:
      env.scene[name].data.hit_pos_w[..., 2] += shift[:, None, 2]
    for name in rewards.__all__:
      torch.testing.assert_close(self.call(name, env), expected[name], atol=1e-4, rtol=1e-4)

  def test_mirroring_preserves_values_and_wheel_order(self):
    env = self.make_env()
    self.step(env)
    data = env.scene["robot"].data
    data.joint_pos[0] = torch.arange(12)/20.
    data.joint_vel[0] = torch.arange(12)/10.
    env.command[0, 1:] = torch.tensor([.2, .3])
    data.root_link_lin_vel_b[0] = torch.tensor([.3, .1, 0.])
    expected = {name: self.call(name, env).clone() for name in rewards.__all__}
    order = [1, 0, 3, 2]
    for name in ("joint_pos", "joint_vel"):
      value = getattr(data, name).reshape(1, 4, 3)[:, order].clone()
      value[..., 0] *= -1
      setattr(data, name, value.reshape(1, 12))
    data.site_pos_w = data.site_pos_w[:, order].clone()
    data.site_pos_w[..., 1] *= -1
    env.command[:, 1:] *= -1
    data.root_link_lin_vel_b[:, 1] *= -1
    scan = env.scene["terrain"].data
    scan.hit_pos_w[..., 2] = scan.hit_pos_w[..., 2].reshape(1, 11, 17).flip(1).flatten(1)
    for name in rewards.__all__:
      torch.testing.assert_close(self.call(name, env), expected[name])

  def test_tilted_wheel_radius_uses_our_cylinder_width_and_ignores_spin(self):
    env = self.make_env()
    data = env.scene["robot"].data
    q = torch.tensor([[[math.sqrt(.5), math.sqrt(.5), 0., 0.]]]).repeat(1, 4, 1)
    data.body_link_quat_w = q
    radius = rewards._vertical_radius(data, SELECTION, .086, .0225)
    torch.testing.assert_close(radius, torch.full((1, 4), .0225), atol=5e-5, rtol=1e-4)
    data.body_link_quat_w = torch.tensor([[[math.cos(.8), 0., math.sin(.8), 0.]]]).repeat(1, 4, 1)
    torch.testing.assert_close(rewards._vertical_radius(data, SELECTION, .086, .0225), torch.full((1, 4), .086))


if __name__ == "__main__":
  unittest.main()
