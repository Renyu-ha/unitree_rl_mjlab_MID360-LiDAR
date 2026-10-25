"""Wheel axle height must follow each wheel's terrain, not world elevation."""

from types import SimpleNamespace
import unittest

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import RayCastSensorCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg

import src.tasks  # noqa: F401 -- register task variants
from src.tasks.velocity.mdp.observations import foot_height_above_terrain


SENSOR_NAMES = (
  "wheel_height_FL", "wheel_height_FR", "wheel_height_RL", "wheel_height_RR",
)


class FootHeightTests(unittest.TestCase):
  def setUp(self):
    self.ground = torch.tensor([[0., .14, -.2, .4], [2., 1.5, 3., -.1]])
    self.expected = torch.tensor([[.086, .086, .086, .086], [.1, .2, .3, .4]])
    self.sites = torch.zeros(2, 4, 3)
    self.sites[..., 2] = self.ground + self.expected
    scene = {"robot": SimpleNamespace(data=SimpleNamespace(site_pos_w=self.sites))}
    for index, name in enumerate(SENSOR_NAMES):
      hits = torch.zeros(2, 1, 3)
      hits[..., 2] = self.ground[:, index:index + 1]
      scene[name] = SimpleNamespace(data=SimpleNamespace(
        hit_pos_w=hits, distances=self.expected[:, index:index + 1].clone(),
      ))
    self.env = SimpleNamespace(scene=scene)
    self.selection = SceneEntityCfg("robot", site_ids=[0, 1, 2, 3])

  def observe(self, **params):
    return foot_height_above_terrain(
      self.env, SENSOR_NAMES, asset_cfg=self.selection, **params,
    )

  def test_each_wheel_uses_its_own_ground_and_keeps_the_radius(self):
    torch.testing.assert_close(self.observe(), self.expected)

  def test_common_world_height_translation_does_not_change_observation(self):
    shift = torch.tensor([10., -7.])
    self.sites[..., 2] += shift[:, None]
    for name in SENSOR_NAMES:
      self.env.scene[name].data.hit_pos_w[..., 2] += shift[:, None]
    torch.testing.assert_close(self.observe(), self.expected)

  def test_missed_ray_uses_sentinel_only_for_the_affected_wheel(self):
    scan = self.env.scene["wheel_height_FR"].data
    scan.distances[1, 0] = -1.
    scan.hit_pos_w[1, 0, 2] = float("nan")
    expected = self.expected.clone()
    expected[1, 1] = 4.5
    torch.testing.assert_close(self.observe(miss_value=4.5), expected)

  def test_multiple_rays_per_wheel_are_rejected_instead_of_broadcast(self):
    scan = self.env.scene["wheel_height_FL"].data
    scan.hit_pos_w = scan.hit_pos_w.repeat(1, 2, 1)
    scan.distances = scan.distances.repeat(1, 2)
    with self.assertRaisesRegex(ValueError, "one terrain ray"):
      self.observe()

  def test_all_go2w_train_and_play_variants_keep_four_world_down_rays(self):
    for task in list_tasks():
      if not task.startswith("Unitree-Go2W-"):
        continue
      for play in (False, True):
        with self.subTest(task=task, play=play):
          cfg = load_env_cfg(task, play=play)
          term = cfg.observations["critic"].terms["foot_height"]
          self.assertIs(term.func, foot_height_above_terrain)
          self.assertEqual(term.params["sensor_names"], SENSOR_NAMES)
          self.assertEqual(tuple(term.params["asset_cfg"].site_names), ("FL", "FR", "RL", "RR"))
          sensors = {sensor.name: sensor for sensor in cfg.scene.sensors}
          self.assertEqual(len(sensors), len(cfg.scene.sensors))
          for site, name in zip(("FL", "FR", "RL", "RR"), SENSOR_NAMES):
            sensor = sensors[name]
            self.assertIsInstance(sensor, RayCastSensorCfg)
            self.assertEqual((sensor.frame.type, sensor.frame.name, sensor.frame.entity),
                             ("site", site, "robot"))
            self.assertEqual(sensor.ray_alignment, "world")
            self.assertEqual(sensor.include_geom_groups, (0, 1))
            origins, directions = sensor.pattern.generate_rays(None, device="cpu")
            torch.testing.assert_close(origins, torch.zeros(1, 3))
            torch.testing.assert_close(directions, torch.tensor([[0., 0., -1.]]))


if __name__ == "__main__":
  unittest.main()
