"""Boundary diagnostics must use the whole grid and preserve default tasks."""

import unittest
from types import SimpleNamespace

import mujoco
import torch
from mjlab.tasks.registry import load_env_cfg
from mjlab.terrains.utils import make_border

import src.tasks  # noqa: F401
from src.utils.border_inspection import find_terrain_border
from src.utils.terrain_boundary import configure_terrain_boundary, terrain_grid_boundary


class TerrainBoundaryTests(unittest.TestCase):
  def test_boundary_uses_global_grid_not_spawn_tile(self):
    generator = SimpleNamespace(num_rows=10, num_cols=6, size=(8.0, 8.0))
    positions = torch.tensor([
      [-36.0, -20.0, 0.4], [38.9, 0.0, 0.4], [39.0, 0.0, 0.4],
      [0.0, -22.9, 0.4], [0.0, -23.0, 0.4], [-24.15454, -24.00259, 0.41844],
    ])
    class Scene(dict):
      terrain = SimpleNamespace(cfg=SimpleNamespace(terrain_generator=generator))
      env_origins = positions.clone()
    env = SimpleNamespace(scene=Scene(robot=SimpleNamespace(data=SimpleNamespace(root_link_pos_w=positions))))
    self.assertEqual(terrain_grid_boundary(env, 1.0).tolist(), [False, False, True, False, True, True])

  def test_optional_timeout_preserves_the_terrain(self):
    cfg = load_env_cfg("Unitree-Go2W-Perceptive-Rough-v2")
    terrain = cfg.scene.terrain.terrain_generator
    self.assertNotIn("terrain_grid_boundary", cfg.terminations)
    configure_terrain_boundary(cfg, 1.0)
    self.assertIs(cfg.scene.terrain.terrain_generator, terrain)
    self.assertEqual(terrain.border_width, 10.0)
    term = cfg.terminations["terrain_grid_boundary"]
    self.assertTrue(term.time_out)
    self.assertIs(term.func, terrain_grid_boundary)

  def test_rejects_invalid_margin_and_plane(self):
    cfg = load_env_cfg("Unitree-Go2W-Perceptive-Rough-v2")
    for margin in (0.0, -1.0, 24.0, float("nan")):
      with self.subTest(margin=margin), self.assertRaises(ValueError):
        configure_terrain_boundary(cfg, margin)
    with self.assertRaises(ValueError):
      configure_terrain_boundary(load_env_cfg("Unitree-Go2W-Flat"), 1.0)

  def test_recognizes_generated_border_without_hardcoded_geom_ids(self):
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="terrain")
    body.add_geom(name="terrain_0", type=mujoco.mjtGeom.mjGEOM_BOX, size=(4, 4, 0.5))
    boxes = make_border(body, size=(100.0, 68.0), inner_size=(80.0, 48.0), height=1.0, position=(0, 0, -0.5))
    for i, box in enumerate(boxes, start=1):
      box.name = f"terrain_{i}"
    self.assertEqual(find_terrain_border(spec.compile()), (1, 2, 3, 4))

  def test_does_not_mistake_ordinary_tiles_for_border(self):
    spec = mujoco.MjSpec()
    for i in range(4):
      spec.worldbody.add_geom(name=f"terrain_{i}", type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(4, 4, 0.5), pos=(8*i, 0, -0.5))
    self.assertEqual(find_terrain_border(spec.compile()), ())


if __name__ == "__main__":
  unittest.main()
