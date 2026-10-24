"""Reset clearance on every configured terrain row, plus partial-reset isolation."""

import unittest
from unittest.mock import patch

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import list_tasks, load_env_cfg

import src.tasks  # noqa: F401
from tests.physics_helpers import (
  assign_all_patches, audit_spawn_clearance, collision_stats, make_coverage_cfg,
  native_penetration,
)
from src.assets.robots.unitree_go2w import go2w_constants


class SpawnClearanceTests(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cfg, cls.rows, cls.names = make_coverage_cfg("Unitree-Go2W-Perceptive-Rough-v2", samples=1)
    cls.env = ManagerBasedRlEnv(cfg, device="cpu")

  @classmethod
  def tearDownClass(cls):
    cls.env.close()

  def test_all_terrain_types_and_difficulty_rows_clear_even_a_low_raw_reset(self):
    report = audit_spawn_clearance(self.env, self.rows, self.names,
                                   resets=2, nominal_heights=(.2, .4, .6))
    self.assertEqual(set(report["terrain_types"]), {
      "flat", "pyramid_stairs", "pyramid_stairs_inv", "hf_pyramid_slope",
      "hf_pyramid_slope_inv", "random_rough", "wave_terrain",
    })
    self.assertTrue(report["passed"], [row for row in report["results"] if not row["passed"]])
    self.assertGreater(sum(row["raw_penetrating_envs"] for row in report["results"]
                           if row["nominal_height_m"] == .2), 0)
    self.assertTrue(all(row["samples"] == self.rows for row in report["results"]))

  def test_partial_reset_lifts_only_the_selected_world(self):
    env = self.env
    assign_all_patches(env, self.rows, self.names)
    env.reset()
    qpos, qvel = env.sim.data.qpos[1:].clone(), env.sim.data.qvel[1:].clone()
    # A bad preceding state must be replaced without moving live environments.
    env.sim.data.qpos[0, 2] -= 1
    env.reset(env_ids=torch.tensor([0]))
    torch.testing.assert_close(env.sim.data.qpos[1:], qpos, atol=0, rtol=0)
    torch.testing.assert_close(env.sim.data.qvel[1:], qvel, atol=0, rtol=0)
    stats = collision_stats(env)
    self.assertLessEqual(stats["penetration_m"][0], 1e-5)
    self.assertFalse(stats["epa"][0])
    self.assertTrue(np.isfinite(env.sim.data.qpos.numpy()).all())

  def test_mesh_wheels_clear_every_terrain_type_and_difficulty_row(self):
    cfg, rows, names = make_coverage_cfg("Unitree-Go2W-Perceptive-Rough-v2", samples=1)
    with patch.object(go2w_constants, "GO2W_XML", go2w_constants.GO2W_MESH_XML):
      env = ManagerBasedRlEnv(cfg, device="cpu")
    try:
      report = audit_spawn_clearance(env, rows, names, resets=1, nominal_heights=(.2, .4))
      self.assertTrue(report["passed"], [row for row in report["results"] if not row["passed"]])
      qpos = env.sim.data.qpos[1:].clone()
      env.reset(env_ids=torch.tensor([0]))
      torch.testing.assert_close(env.sim.data.qpos[1:], qpos, atol=0, rtol=0)
      self.assertLessEqual(native_penetration(env)[0], 1e-5)
    finally:
      env.close()

  def test_placement_away_from_platform_centers_clears_steps_and_slopes(self):
    env = self.env
    assign_all_patches(env, self.rows, self.names)
    env.reset()
    asset = env.scene["robot"]
    pose = env.sim.data.qpos[:, asset.indexing.free_joint_q_adr].clone()
    pose[:, 0] += 1.5
    pose[:, 1] += .8
    pose[:, 2] -= 1
    asset.write_root_link_pose_to_sim(pose)
    guard = env.event_manager.get_term_cfg("ensure_spawn_clearance").func
    guard(env, torch.arange(env.num_envs))
    stats = collision_stats(env)
    self.assertTrue(np.all(stats["penetration_m"] <= 1e-5), stats["penetration_m"])
    self.assertFalse(stats["epa"].any())
    self.assertTrue(np.all(native_penetration(env) <= 1e-5))

  def test_every_go2w_variant_runs_clearance_last_and_relocates_before_root_reset(self):
    for task in list_tasks():
      if not task.startswith("Unitree-Go2W-"):
        continue
      for play in (False, True):
        with self.subTest(task=task, play=play):
          cfg = load_env_cfg(task, play=play)
          reset_terms = [name for name, term in cfg.events.items() if term.mode == "reset"]
          self.assertEqual(reset_terms[-1], "ensure_spawn_clearance")
          if "randomize_terrain" in reset_terms:
            self.assertLess(reset_terms.index("randomize_terrain"), reset_terms.index("reset_base"))

  def test_play_reset_places_robot_at_the_new_terrain_origin(self):
    cfg = load_env_cfg("Unitree-Go2W-Perceptive-Rough-v2", play=True)
    cfg.seed = 7
    cfg.episode_length_s = 20
    cfg.scene.num_envs = 2
    cfg.scene.terrain.terrain_generator.num_rows = 2
    cfg.scene.terrain.terrain_generator.num_cols = 2
    env = ManagerBasedRlEnv(cfg, device="cpu")
    try:
      for _ in range(6):
        env.reset()
        delta = env.scene["robot"].data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]
        self.assertTrue(torch.all(delta.abs() <= .50001))
        self.assertTrue(np.all(native_penetration(env) <= 1e-5))
    finally:
      env.close()


if __name__ == "__main__":
  unittest.main()
