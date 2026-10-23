"""CPU regression tests for the exact failure/result distinction."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import mujoco
import numpy as np
from mjlab.sim import Simulation, SimulationCfg

from scripts.inspect_nan_dump import recorded_summary
from src.utils.nan_capture import PhysicsNanError, install_nan_capture
from src.utils.nan_viewer import PhysicsNanDumpViewer


class NanCaptureTests(unittest.TestCase):
  def setUp(self):
    self.directory = TemporaryDirectory()
    self.addCleanup(self.directory.cleanup)
    self.model = mujoco.MjModel.from_xml_string("""
      <mujoco><worldbody><body><joint name="angle" type="hinge"/>
      <geom type="sphere" size=".1"/></body></worldbody>
      <actuator><motor joint="angle"/></actuator>
      <sensor><jointpos joint="angle"/></sensor></mujoco>
    """)
    cfg = SimulationCfg(nconmax=4, njmax=8)
    cfg.nan_guard.enabled = True
    cfg.nan_guard.buffer_size = 1
    cfg.nan_guard.output_dir = self.directory.name
    self.sim = Simulation(num_envs=2, cfg=cfg, model=self.model, device="cpu")

  def load_dump(self, guard):
    return np.load(guard.dump_path, allow_pickle=False)

  def test_step_saves_failure_even_when_history_ring_is_full(self):
    sim = self.sim
    sim.expand_model_fields(("body_mass",))
    sim.model.body_mass[1, 1] = 3.0
    guard = install_nan_capture(sim)
    sim.data.ctrl[1, 0] = 2.5
    sim.data.qacc_warmstart[1, 0] = 4.0
    # Populate beyond the ring limit. Only the final pre-input may survive,
    # but the invalid post-result must still be appended.
    guard.capture(sim.data)
    guard.capture(sim.data)

    def invalid_step(*_args):
      sim.data.qpos[1, 0] = float("nan")
      sim.data.qvel[1, 0] = float("inf")
      sim.data.qacc[1, 0] = float("nan")
      sim.data.nacon[0] = 2
      sim.data.contact.worldid[:2] = 1
      sim.data.contact.pos[0] = -243.0
      sim.data.contact.dist[0] = -1.0

    with patch("mjlab.sim.sim.mjwarp.step", side_effect=invalid_step):
      with self.assertRaises(PhysicsNanError):
        sim.step()
    with self.load_dump(guard) as archive:
      metadata = json.loads(archive["metadata_json"].item())
      self.assertEqual(metadata["failure_stage"], "after_step")
      self.assertEqual(metadata["dumped_env_ids"], [1])
      keys = sorted(k for k in archive.files if k.startswith("states_step_"))
      self.assertEqual(len(keys), 2)
      self.assertTrue(np.isfinite(archive[keys[0]]).all())
      self.assertFalse(np.isfinite(archive[keys[-1]]).all())
      np.testing.assert_equal(archive["inputs_step_000002_ctrl"], [[2.5]])
      np.testing.assert_equal(archive["inputs_step_000002_qacc_warmstart"], [[4.0]])
      self.assertEqual(archive["failure_contact_worldid"].tolist(), [1, 1])
      self.assertEqual(float(archive["failure_contact_dist"][0]), -1.0)
      self.assertEqual(float(archive["model_body_mass"][0, 1]), 3.0)
      self.assertEqual(recorded_summary(archive)["fields"]["failure_qpos"]["nonfinite_count"], 1)
    self.assertTrue(Path(self.directory.name, "nan_dump_latest.npz").is_file())
    self.assertTrue(Path(self.directory.name, "model_latest.mjb").is_file())
    self.assertTrue(guard.dump_path.with_suffix(".json").is_file())

  def test_forward_catches_sensor_failure_while_joint_angle_is_finite(self):
    sim = self.sim

    def invalid_forward():
      sim.data.sensordata[0, 0] = float("nan")

    sim.forward = invalid_forward
    guard = install_nan_capture(sim)
    with self.assertRaisesRegex(PhysicsNanError, "after_forward"):
      sim.forward()
    with self.load_dump(guard) as archive:
      metadata = json.loads(archive["metadata_json"].item())
      self.assertEqual(metadata["nonfinite_fields"], {"sensordata": [0]})
      self.assertTrue(np.isfinite(archive[metadata["failure_state_key"]]).all())
      self.assertTrue(np.isnan(archive["failure_sensordata"][0, 0]))

  def test_invalid_control_is_captured_before_physics_runs(self):
    sim = self.sim
    original = Mock()
    sim.forward = original
    guard = install_nan_capture(sim)
    sim.data.ctrl[1, 0] = float("inf")
    with self.assertRaisesRegex(PhysicsNanError, "before_forward"):
      sim.forward()
    original.assert_not_called()
    with self.load_dump(guard) as archive:
      self.assertTrue(np.isinf(archive["failure_ctrl"][0, 0]))

  def test_disabled_guard_does_not_wrap_forward(self):
    sim = self.sim
    sim.cfg.nan_guard.enabled = False
    original = sim.forward
    install_nan_capture(sim)
    self.assertEqual(sim.forward, original)

  def test_viewer_keeps_raw_nan_but_renders_same_world_predecessor(self):
    viewer = PhysicsNanDumpViewer.__new__(PhysicsNanDumpViewer)
    viewer.model = self.model
    viewer.data = mujoco.MjData(self.model)
    viewer.state_keys = ["states_step_000000", "states_step_000001"]
    viewer.dump = {
      viewer.state_keys[0]: np.array([[0.1, 0.0], [0.7, 0.0]]),
      viewer.state_keys[1]: np.array([[0.2, 0.0], [np.nan, np.inf]]),
      "failure_qpos": np.array([[0.2], [np.nan]]),
    }
    viewer.current_step, viewer.current_env = 1, 1
    viewer.dumped_env_ids = [7, 42]
    viewer.metadata = {"failure_state_key": viewer.state_keys[-1],
                       "frame_phases": {viewer.state_keys[-1]: "after_step"}}
    viewer.info_html = SimpleNamespace(content="")
    viewer.scene = Mock()
    viewer._update_state()
    self.assertAlmostEqual(viewer.data.qpos[0], 0.7)
    self.assertTrue(np.isnan(viewer.dump[viewer.state_keys[1]][1, 0]))
    self.assertIn("RECORDED FAILURE", viewer.info_html.content)
    self.assertIn("qpos: 1 nonfinite", viewer.info_html.content)
    self.assertIn("previous renderable pose", viewer.info_html.content)
    viewer.metadata = {}
    self.assertIn("Legacy dump", viewer._get_info_html())


if __name__ == "__main__":
  unittest.main()
