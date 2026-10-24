"""Verify worker capacity is applied before real simulation buffer allocation."""

from contextlib import redirect_stdout
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mujoco
import mujoco_warp as mjw
from mujoco_warp._src import collision_convex
from mjlab.sim import Simulation, SimulationCfg
import warp as wp

from scripts.train import run_train


class TrainCollisionSettingsTests(unittest.TestCase):
  def test_worker_allocates_selected_capacity_and_restores_after_training_error(self):
    original = collision_convex.MJ_MAX_EPAHORIZON
    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <geom type="box" size="1.8 1.8 .02" pos="0 0 -.02"/>
      <body pos=".2 .3 .08"><freejoint/>
        <geom type="cylinder" size=".086 .0225" quat="1 1 0 0"/>
      </body></worldbody></mujoco>""")
    for requested in (None, 30):
      expected = original if requested is None else requested
      with self.subTest(requested=requested), wp.ScopedDevice("cpu"):
        cfg = SimpleNamespace(epa_horizon=requested)

        def train(task_id, config, log_dir, settings):
          self.assertEqual(settings["epa_horizon"], expected)
          self.assertEqual(settings["epa_horizon_override"], requested)
          self.assertEqual(Path(settings["engine_source"]), Path(mjw.__file__).resolve())
          # Exercise real allocation and collision, not just the global constant.
          with patch.object(wp, "empty", wraps=wp.empty) as allocate:
            sim = Simulation(num_envs=2, cfg=SimulationCfg(nconmax=64, njmax=128), model=model, device="cpu")
            sim.forward()
            sim.step()
          wp.synchronize()
          shapes = [call.kwargs.get("shape") for call in allocate.call_args_list]
          self.assertIn((sim.wp_data.naccdmax, expected), shapes)
          self.assertTrue(sim.wp_model.opt.warn_overflow & int(mjw.OverflowType.EPA_HORIZON))
          if requested is not None:
            raise RuntimeError("training stopped")

        with patch("scripts.train._run_train", side_effect=train), redirect_stdout(io.StringIO()):
          if requested is None:
            run_train("test", cfg, Path("unused"))
          else:
            with self.assertRaisesRegex(RuntimeError, "training stopped"):
              run_train("test", cfg, Path("unused"))
        self.assertEqual(collision_convex.MJ_MAX_EPAHORIZON, original)


if __name__ == "__main__":
  unittest.main()
