"""Exercise reward rays against actual MuJoCo-Warp terrain and height sensors."""

import unittest
from dataclasses import dataclass

import mujoco
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.terrains.terrain_generator import (
  SubTerrainCfg,
  TerrainGeneratorCfg,
  TerrainGeometry,
  TerrainOutput,
)

import src.tasks  # noqa: F401


@dataclass
class TallStepTerrain(SubTerrainCfg):
  def function(self, difficulty, spec, rng):
    body = spec.body("terrain")
    floor = body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,
                          pos=(2., 2., -.1), size=(2., 2., .1))
    step = body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,
                         pos=(2.6, 2., .4), size=(.15, .6, .4))
    return TerrainOutput(origin=np.array([2., 2., 0.]),
                         geometries=[TerrainGeometry(geom=floor), TerrainGeometry(geom=step)])


class TerrainRewardRuntimeTests(unittest.TestCase):
  def test_ground_truth_queries_above_body_and_reward_manager_steps(self):
    cfg = load_env_cfg("Unitree-Go2W-Flat", play=True)
    cfg.scene.num_envs = 1
    cfg.events, cfg.curriculum = {}, {}
    cfg.scene.terrain.terrain_type = "generator"
    cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
      seed=0, size=(4., 4.), num_rows=1, num_cols=1,
      sub_terrains={"tall_step": TallStepTerrain()},
    )
    cfg.scene.terrain.max_init_terrain_level = 0
    env = ManagerBasedRlEnv(cfg, device="cpu")
    try:
      env.reset()
      truth = env.scene["reward_terrain_scan"].data
      # The selected step top is 80 cm, above the 40 cm body frame. A ray
      # originating at body height cannot provide this top surface.
      tops = truth.hit_pos_w[..., 2][truth.distances >= 0]
      self.assertTrue((tops > .79).any())
      self.assertTrue(torch.isfinite(tops).all())
      for name in ("leg_posture", "unnecessary_wheel_lift", "rolling_leg_motion",
                   "obstacle_progress", "wheel_obstacle_clearance"):
        term = env.reward_manager.get_term_cfg(name)
        values = term.func(env, **term.params)
        self.assertEqual(values.shape, (1,))
        self.assertTrue(torch.isfinite(values).all(), name)
      _, reward, _, _, _ = env.step(torch.zeros(1, 16))
      self.assertTrue(torch.isfinite(reward).all())
    finally:
      env.close()


if __name__ == "__main__":
  unittest.main()
