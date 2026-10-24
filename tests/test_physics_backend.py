"""Verify the installed backend and the buffer used by real simulations."""

import unittest
from unittest.mock import patch

import mujoco
import mujoco_warp as mjw
from mujoco_warp._src import collision_convex
from mjlab.sim import Simulation, SimulationCfg
import numpy as np
import warp as wp

from src.utils.physics_backend import physics_backend_info


class PhysicsBackendTests(unittest.TestCase):
  def test_real_simulation_uses_fixed_capacity_and_keeps_diagnostics(self):
    info = physics_backend_info()
    self.assertEqual(info["mujoco_warp"], "3.16.0.post1")
    self.assertEqual(info["epa_horizon"], 64)
    model = mujoco.MjModel.from_xml_string("""
      <mujoco><worldbody>
        <geom type="box" size="1.8 1.8 .02" pos="0 0 -.02"/>
        <body pos=".2 .3 .08"><freejoint/>
          <geom type="cylinder" size=".086 .0225" quat="1 1 0 0"/>
        </body>
      </worldbody></mujoco>
    """)
    with wp.ScopedDevice("cpu"), patch.object(wp, "empty", wraps=wp.empty) as allocate:
      sim = Simulation(num_envs=2, cfg=SimulationCfg(nconmax=64, njmax=128),
                       model=model, device="cpu")
      sim.forward()
      sim.step()
    shapes = [call.kwargs.get("shape") for call in allocate.call_args_list]
    self.assertIn((sim.wp_data.naccdmax, 64), shapes)
    self.assertTrue(np.isfinite(sim.wp_data.qacc.numpy()).all())
    self.assertTrue(sim.wp_model.opt.warn_overflow & int(mjw.OverflowType.EPA_HORIZON))

  def test_stale_backend_is_rejected(self):
    with patch.object(collision_convex, "MJ_MAX_EPAHORIZON", 24):
      with self.assertRaisesRegex(RuntimeError, "pip install -e"):
        physics_backend_info()


if __name__ == "__main__":
  unittest.main()
