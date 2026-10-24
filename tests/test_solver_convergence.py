"""Replay finite contact states that exhausted the old 10/20 solver budget."""

from pathlib import Path
import unittest

import mujoco_warp as mjw
import numpy as np

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

import src.tasks  # noqa: F401


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
    sim.wp_data.overflow.zero_()
    getattr(sim, method)()
    return sim.wp_data.overflow.numpy().copy(), sim.wp_data.qacc.numpy().copy()

  def test_captured_limits_converge_without_disabling_diagnostics(self):
    cfg = load_env_cfg("Unitree-Go2W-Perceptive-Rough-v2")
    for case in ("linesearch", "solver"):
      with self.subTest(case=case):
        old, _ = self.replay(case, 10, 20)
        bit = mjw.OverflowType.LS_ITERATIONS if case == "linesearch" else mjw.OverflowType.ITERATIONS
        self.assertTrue(np.any(old & int(bit)))
        actual, qacc = self.replay(case, cfg.sim.mujoco.iterations, cfg.sim.mujoco.ls_iterations)
        reference, reference_qacc = self.replay(case, 100, 200)
        self.assertTrue(np.all(actual == 0), actual)
        self.assertTrue(np.all(reference == 0), reference)
        np.testing.assert_allclose(qacc, reference_qacc, rtol=1e-6, atol=1e-5)
        self.assertTrue(np.isfinite(qacc).all())
    warning_bits = int(mjw.OverflowType.ITERATIONS | mjw.OverflowType.LS_ITERATIONS)
    self.assertEqual(self.env.sim.wp_model.opt.warn_overflow & warning_bits, warning_bits)


if __name__ == "__main__":
  unittest.main()
