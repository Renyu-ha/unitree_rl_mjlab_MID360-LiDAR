"""Map boundaries truncate episodes without applying failure penalties."""

import unittest
from types import SimpleNamespace

import mujoco
import torch
from mjlab.managers.reward_manager import RewardManager
from mjlab.managers.termination_manager import TerminationManager
from mjlab.tasks.registry import list_tasks, load_env_cfg
from mjlab.terrains.utils import make_border

import src.tasks  # noqa: F401
from src.tasks.velocity.mdp.terminations import terrain_out_of_bounds
from src.utils.border_inspection import find_terrain_border


class TerrainBoundaryTests(unittest.TestCase):
  @staticmethod
  def make_env(positions, *, plane=False):
    positions = torch.tensor(positions, dtype=torch.float32)
    generator = SimpleNamespace(num_rows=10, num_cols=6, size=(8.0, 8.0), border_width=10.0)
    class Scene(dict):
      terrain = SimpleNamespace(cfg=SimpleNamespace(
        terrain_type="plane" if plane else "generator",
        terrain_generator=None if plane else generator,
      ))
      # Deliberately different tile origins: bounds must use world positions.
      env_origins = positions.clone()
    return SimpleNamespace(num_envs=len(positions), device="cpu", scene=Scene(
      robot=SimpleNamespace(data=SimpleNamespace(root_link_pos_w=positions)),
    ))

  def test_go2w_variants_enable_boundary_timeout_for_train_and_play(self):
    tasks = [task for task in list_tasks() if task.startswith("Unitree-Go2W-")]
    self.assertEqual(len(tasks), 6)
    for task in tasks:
      for play in (False, True):
        with self.subTest(task=task, play=play):
          cfg = load_env_cfg(task, play=play)
          term = cfg.terminations["terrain_out_of_bounds"]
          self.assertIs(term.func, terrain_out_of_bounds)
          self.assertTrue(term.time_out)
          self.assertEqual(term.params["distance_buffer"], 3.0)
          self.assertFalse(cfg.is_finite_horizon)

  def test_outer_boundary_includes_border_on_both_axes_and_sides(self):
    env = self.make_env([
      [-36.0, -20.0, 0.4], [-24.15, -24.0026, 0.4],
      [47.0, 0.0, 0.4], [47.01, 0.0, 0.4],
      [-47.0, 0.0, 0.4], [-47.01, 0.0, 0.4],
      [0.0, 31.0, 0.4], [0.0, 31.01, 0.4],
      [0.0, -31.0, 0.4], [0.0, -31.01, 0.4],
    ])
    self.assertEqual(terrain_out_of_bounds(env).tolist(),
      [False, False, False, True, False, True, False, True, False, True])
    # A different buffer changes the threshold, including on the border.
    env.scene["robot"].data.root_link_pos_w[:] = torch.tensor([48.0, 0.0, 0.4])
    self.assertTrue(terrain_out_of_bounds(env).all())
    self.assertFalse(terrain_out_of_bounds(env, distance_buffer=1.0).any())

  def test_plane_has_no_boundary(self):
    env = self.make_env([[1e6, -1e6, 0.4], [-1e6, 1e6, 0.4]], plane=True)
    result = terrain_out_of_bounds(env)
    self.assertEqual(result.dtype, torch.bool)
    self.assertEqual(result.tolist(), [False, False])

  def test_boundary_truncates_without_failure_reward_but_preserves_falls(self):
    cfg = load_env_cfg("Unitree-Go2W-Perceptive-Rough-v2")
    env = self.make_env([
      [0.0, 0.0, 0.4], [48.0, 0.0, 0.4],
      [48.0, 0.0, 0.4], [0.0, 0.0, 0.4],
    ])
    env.scene["robot"].data.projected_gravity_b = torch.tensor([
      [0.0, 0.0, -1.0], [0.0, 0.0, -1.0],
      [1.0, 0.0, 0.0], [1.0, 0.0, 0.0],
    ])
    env.termination_manager = TerminationManager({
      "terrain_out_of_bounds": cfg.terminations["terrain_out_of_bounds"],
      "fell_over": cfg.terminations["fell_over"],
    }, env)
    self.assertEqual(env.termination_manager.compute().tolist(), [False, True, True, True])
    self.assertEqual(env.termination_manager.time_outs.tolist(), [False, True, True, False])
    self.assertEqual(env.termination_manager.terminated.tolist(), [False, False, True, True])
    rewards = RewardManager({"is_terminated": cfg.rewards["is_terminated"]}, env)
    torch.testing.assert_close(rewards.compute(dt=0.02), torch.tensor([0.0, 0.0, -4.0, -4.0]))
    # The next step must clear timeout/failure flags once the robot is reset.
    env.scene["robot"].data.root_link_pos_w.zero_()
    env.scene["robot"].data.projected_gravity_b[:] = torch.tensor([0.0, 0.0, -1.0])
    self.assertFalse(env.termination_manager.compute().any())
    self.assertFalse(rewards.compute(dt=0.02).any())

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
