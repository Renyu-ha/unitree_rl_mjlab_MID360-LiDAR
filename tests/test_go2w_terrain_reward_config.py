"""Native registry and entity-selection contracts for terrain-aware rewards."""

import inspect
import unittest

from mjlab.entity import Entity
from mjlab.tasks.registry import list_tasks, load_env_cfg

import src.tasks  # noqa: F401 -- register all variants
from src.assets.robots.unitree_go2w.go2w_constants import (
  LEG_JOINTS,
  WHEEL_BODIES,
  WHEEL_SITES,
  get_go2w_robot_cfg,
)
from src.tasks.velocity.config.go2w.env_cfgs import WHEEL_HEIGHT_SENSORS
from src.tasks.velocity.mdp.rewards import terrain_aware


class TerrainRewardConfigurationTests(unittest.TestCase):
  def test_every_go2w_variant_wires_all_five_rewards_and_no_old_constraints(self):
    expected = {"leg_posture": -.15, "unnecessary_wheel_lift": -.2,
                "rolling_leg_motion": -.12, "obstacle_progress": .5, "wheel_obstacle_clearance": -.25}
    tasks = [name for name in list_tasks() if name.startswith("Unitree-Go2W-")]
    self.assertTrue(tasks)
    for task in tasks:
      for play in (False, True):
        with self.subTest(task=task, play=play):
          cfg = load_env_cfg(task, play=play)
          for name in ("pose", "wheel_air_time_memory", "wheel_air_time_timeout", "wheel_contact_bonus"):
            self.assertNotIn(name, cfg.rewards)
          self.assertNotIn("dont_wait", cfg.rewards)
          self.assertNotIn("straight_yaw_rate_l2", cfg.rewards)
          self.assertNotIn("pose_weight", cfg.curriculum)
          sensors = {sensor.name: sensor for sensor in cfg.scene.sensors}
          self.assertIn("terrain_scan", sensors)
          self.assertIn("reward_terrain_scan", sensors)
          truth = sensors["reward_terrain_scan"]
          self.assertEqual(truth.ray_alignment, "yaw")
          offsets, directions = truth.pattern.generate_rays(None, device="cpu")
          self.assertTrue((offsets[:, 2] == 10.).all())
          self.assertTrue((directions[:, 2] == -1.).all())
          self.assertEqual(offsets.shape[0], 273)
          for name, weight in expected.items():
            term = cfg.rewards[name]
            self.assertIs(term.func, getattr(terrain_aware, name))
            self.assertEqual(term.weight, weight)
            inspect.signature(term.func).bind(object(), **term.params)
            self.assertEqual(term.params["terrain_sensor_name"], "reward_terrain_scan")
          for name in ("unnecessary_wheel_lift", "wheel_obstacle_clearance"):
            term = cfg.rewards[name]
            self.assertEqual(term.params["wheel_sensor_names"], WHEEL_HEIGHT_SENSORS)
            self.assertEqual(term.params["wheel_half_width"], .0225)
            for sensor_name, site_name in zip(WHEEL_HEIGHT_SENSORS, WHEEL_SITES, strict=True):
              self.assertEqual(sensors[sensor_name].frame.name, site_name)
          # Every remaining scheduled reward must still exist after replacing pose.
          for term in cfg.curriculum.values():
            if "reward_name" in term.params:
              self.assertIn(term.params["reward_name"], cfg.rewards)
          if "Flat" in task:
            self.assertNotIn("height_scan", cfg.observations["actor"].terms)

  def test_native_selection_keeps_leg_triplets_and_wheel_sites_in_sensor_order(self):
    robot = Entity(get_go2w_robot_cfg())
    cfg = load_env_cfg("Unitree-Go2W-Perceptive-Rough-v2")
    for name in terrain_aware.__all__:
      with self.subTest(reward=name):
        selection = cfg.rewards[name].params["asset_cfg"]
        self.assertTrue(selection.preserve_order)
        selection.resolve({"robot": robot})
        self.assertEqual(tuple(selection.joint_names), LEG_JOINTS)
        self.assertEqual(tuple(selection.site_names), WHEEL_SITES)
        self.assertEqual(tuple(selection.body_names), WHEEL_BODIES)


if __name__ == "__main__":
  unittest.main()
