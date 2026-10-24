"""Replay finite contact states that exhausted the old 10/20 solver budget."""

from pathlib import Path
import unittest

import mujoco_warp as mjw
import numpy as np

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

import src.tasks  # noqa: F401
from scripts.check_solver_convergence import SolverMonitor


class SolverConvergenceTests(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cfg = load_env_cfg("Unitree-Go2W-Perceptive-Rough-v2")
    cfg.seed = 42
    cfg.scene.num_envs = 1
    cls.env = ManagerBasedRlEnv(cfg, device="cpu")
    cls.env.reset()

  @classmethod
  def tearDownClass(cls):
    cls.env.close()

  def replay(self, case, iterations, ls_iterations):
    sim = self.env.sim
    sim.wp_model.opt.iterations = iterations
    sim.wp_model.opt.ls_iterations = ls_iterations
    with np.load(Path(__file__).parent / "fixtures" / f"go2w_{case}_limit.npz",
                 allow_pickle=False) as frame:
      for key in frame.files:
        if key.startswith("model_"):
          getattr(sim.wp_model, key[6:]).assign(frame[key])
        elif key.startswith("input_"):
          getattr(sim.wp_data, key[6:]).assign(frame[key])
      method = frame["method"].item()
    monitor = SolverMonitor(sim)
    monitor.install()
    try:
      getattr(sim, method)()
      return monitor, sim.wp_data.qacc.numpy().copy()
    finally:
      monitor.restore()

  def test_captured_limits_converge_without_disabling_diagnostics(self):
    cfg = load_env_cfg("Unitree-Go2W-Perceptive-Rough-v2")
    for case, counter in (("linesearch", "linesearch_limit"), ("solver", "solver_limit")):
      with self.subTest(case=case):
        old, _ = self.replay(case, 10, 20)
        self.assertFalse(old.passed)
        self.assertGreater(old.counts[counter], 0)
        actual, qacc = self.replay(case, cfg.sim.mujoco.iterations, cfg.sim.mujoco.ls_iterations)
        reference, reference_qacc = self.replay(case, 100, 200)
        self.assertTrue(actual.passed, actual.counts)
        self.assertTrue(reference.passed, reference.counts)
        np.testing.assert_allclose(qacc, reference_qacc, rtol=1e-6, atol=1e-5)
        self.assertTrue(np.isfinite(qacc).all())
    warning_bits = int(mjw.OverflowType.ITERATIONS | mjw.OverflowType.LS_ITERATIONS)
    self.assertEqual(self.env.sim.wp_model.opt.warn_overflow & warning_bits, warning_bits)


if __name__ == "__main__":
  unittest.main()
