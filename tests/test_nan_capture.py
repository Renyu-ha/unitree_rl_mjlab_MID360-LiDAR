"""CPU regression tests for the exact failure/result distinction."""

import json
import contextlib
import io
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import mujoco
import numpy as np
import torch
from mjlab.sim import Simulation, SimulationCfg
from mjlab.envs.mdp.dr.body import body_com_offset

from scripts.inspect_nan_dump import inspect, recorded_inputs, recorded_summary
from src.utils.nan_capture import PhysicsNanError, install_nan_capture
from src.utils import nan_capture
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
      np.testing.assert_equal(archive["failure_geom_xpos"], sim.data.geom_xpos[1:2].numpy())
      np.testing.assert_equal(archive["failure_geom_xmat"], sim.data.geom_xmat[1:2].numpy())
      self.assertIn("failure_overflow", archive)
      self.assertEqual(float(archive["model_body_mass"][0, 1]), 3.0)
      self.assertEqual(recorded_summary(archive)["fields"]["failure_qpos"]["nonfinite_count"], 1)
      inputs, fields = recorded_inputs(archive, keys[0], 0)
      np.testing.assert_equal(inputs["ctrl"], [[2.5]])
      np.testing.assert_equal(inputs["qacc_warmstart"], [[4.0]])
      self.assertEqual(float(fields["body_mass"][0, 1]), 3.0)
      with self.assertRaisesRegex(ValueError, "lacks recorded inputs"):
        recorded_inputs(archive, keys[-1], 0)
    self.assertTrue(Path(self.directory.name, "nan_dump_latest.npz").is_file())
    self.assertTrue(Path(self.directory.name, "model_latest.mjb").is_file())
    self.assertTrue(guard.dump_path.with_suffix(".json").is_file())
    args = SimpleNamespace(dump=guard.dump_path, model=None, device="cpu", frame=-1,
                           summary_only=False, dump_index=0, output=None)
    with contextlib.redirect_stdout(io.StringIO()):
      report = inspect(args)
    self.assertEqual(report["input_source"], "recorded")
    self.assertEqual(report["state_key"], keys[0])
    self.assertIn("body_mass", report["restored_model_fields"])
    self.assertEqual(len(report["warp"]), 1)
    self.assertTrue(report["warp"][0]["after_step"]["qvel"]["finite"])

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

  def test_empty_tendon_fields_from_com_randomization_do_not_discard_failure(self):
    sim = self.sim
    # COM randomization registers derived tendon fields even on a robot with
    # no tendons, exactly as the Go2W training environment does.
    sim.expand_model_fields(body_com_offset.model_fields)

    def invalid_forward():
      sim.data.sensordata[1, 0] = float("nan")

    sim.forward = invalid_forward
    guard = install_nan_capture(sim)
    with self.assertRaisesRegex(PhysicsNanError, "after_forward"):
      sim.forward()
    with self.load_dump(guard) as archive:
      metadata = json.loads(archive["metadata_json"].item())
      self.assertEqual(metadata["capture_status"], "complete")
      self.assertEqual(metadata["dumped_env_ids"], [1])
      self.assertTrue(np.isnan(archive["failure_sensordata"][0, 0]))
      self.assertIn("model_body_ipos", archive)
      for name in ("tendon_length0", "tendon_invweight0"):
        self.assertIn(name, metadata["requested_model_fields"])
        self.assertIn(name, metadata["skipped_model_fields"])
        self.assertNotIn(name, metadata["randomized_model_fields"])
        self.assertNotIn(f"model_{name}", archive)

  def test_core_archive_survives_optional_model_read_error(self):
    sim = self.sim
    sim.expand_model_fields(("body_mass", "geom_friction"))
    original_tensor = nan_capture._tensor
    saw_core_archive = []

    def unavailable_model_field(data, name):
      if data is sim.model and name == "body_mass":
        # The essential data must already exist on disk when optional
        # diagnostics are first accessed, including before an extraction error.
        paths = list(Path(self.directory.name).glob("nan_dump_*.npz"))
        self.assertEqual(len(paths), 1)
        with np.load(paths[0], allow_pickle=False) as archive:
          metadata = json.loads(archive["metadata_json"].item())
          saw_core_archive.append(metadata["capture_status"])
          self.assertTrue(np.isnan(archive["failure_sensordata"][0, 0]))
        raise RuntimeError("optional model read failed")
      return original_tensor(data, name)

    def invalid_forward():
      sim.data.sensordata[0, 0] = float("nan")

    sim.forward = invalid_forward
    guard = install_nan_capture(sim)
    with patch.object(nan_capture, "_tensor", side_effect=unavailable_model_field):
      with self.assertRaisesRegex(PhysicsNanError, "after_forward"):
        sim.forward()
    self.assertEqual(saw_core_archive, ["core_saved"])
    with self.load_dump(guard) as archive:
      metadata = json.loads(archive["metadata_json"].item())
      self.assertEqual(metadata["capture_status"], "partial")
      self.assertIn("optional model read failed", metadata["capture_errors"]["model.body_mass"])
      summary = recorded_summary(archive)
      self.assertEqual(summary["capture_status"], "partial")
      self.assertEqual(summary["capture_errors"], metadata["capture_errors"])
      self.assertIn("model_geom_friction", archive)
      self.assertTrue(np.isnan(archive["failure_sensordata"][0, 0]))

  def test_latest_copy_failure_does_not_mask_saved_physics_failure(self):
    sim = self.sim

    def invalid_forward():
      sim.data.sensordata[0, 0] = float("nan")

    sim.forward = invalid_forward
    guard = install_nan_capture(sim)
    with patch.object(nan_capture.shutil, "copyfile", side_effect=OSError("latest copy failed")):
      with self.assertRaisesRegex(PhysicsNanError, "after_forward"):
        sim.forward()
    with self.load_dump(guard) as archive:
      self.assertTrue(np.isnan(archive["failure_sensordata"][0, 0]))

  def test_failed_archive_enrichment_keeps_initial_core_readable(self):
    sim = self.sim
    original_save = nan_capture.np.savez_compressed
    calls = []

    def fail_second_write(path, **payload):
      calls.append(path)
      if len(calls) == 2:
        Path(path).write_bytes(b"interrupted archive write")
        raise OSError("archive enrichment failed")
      original_save(path, **payload)

    def invalid_forward():
      sim.data.sensordata[0, 0] = float("nan")

    sim.forward = invalid_forward
    guard = install_nan_capture(sim)
    with patch.object(nan_capture.np, "savez_compressed", side_effect=fail_second_write):
      with self.assertRaisesRegex(PhysicsNanError, "after_forward"):
        sim.forward()
    self.assertEqual(len(calls), 2)
    with self.load_dump(guard) as archive:
      self.assertTrue(np.isnan(archive["failure_sensordata"][0, 0]))
      metadata = json.loads(archive["metadata_json"].item())
      self.assertEqual(metadata["capture_status"], "core_saved")
    metadata = json.loads(guard.dump_path.with_suffix(".json").read_text(encoding="utf-8"))
    self.assertIn("archive enrichment failed", metadata["capture_errors"]["archive_enrichment"])

  def test_go2w_training_randomization_can_dump_after_step_failure(self):
    import src.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    cfg = load_env_cfg("Unitree-Go2W-Perceptive-Rough-v2")
    cfg.scene.num_envs = 2
    cfg.scene.terrain.terrain_generator.num_rows = 1
    cfg.scene.terrain.terrain_generator.num_cols = 2
    cfg.scene.terrain.max_init_terrain_level = 0
    cfg.sim.nan_guard.enabled = True
    cfg.sim.nan_guard.output_dir = self.directory.name
    env = ManagerBasedRlEnv(cfg, device="cpu")
    try:
      guard = install_nan_capture(env.sim)
      env.reset()
      self.assertIn("tendon_length0", env.sim.expanded_fields)
      import mujoco_warp
      step = mujoco_warp.step

      def invalid_step(*args, **kwargs):
        step(*args, **kwargs)
        env.sim.data.sensordata[1, 0] = float("nan")

      with patch("mjlab.sim.sim.mjwarp.step", side_effect=invalid_step):
        with self.assertRaisesRegex(PhysicsNanError, "after_step"):
          env.step(torch.zeros(2, 16))
      with self.load_dump(guard) as archive:
        metadata = json.loads(archive["metadata_json"].item())
        self.assertEqual(metadata["failure_stage"], "after_step")
        self.assertEqual(metadata["capture_status"], "complete")
        self.assertEqual(metadata["capture_errors"], {})
        self.assertIn("tendon_length0", metadata["skipped_model_fields"])
        self.assertIn("tendon_invweight0", metadata["skipped_model_fields"])
        self.assertIn("model_geom_friction", archive)
        self.assertIn("model_body_ipos", archive)
        self.assertTrue(np.isnan(archive["failure_sensordata"][0, 0]))
        self.assertTrue(np.isfinite(archive["failure_qpos"]).all())
    finally:
      env.close()

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
