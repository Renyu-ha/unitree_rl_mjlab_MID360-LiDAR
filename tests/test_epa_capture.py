"""Exercise a real EPA overflow with a deliberately undersized test buffer.

Frozen box poses are from mujoco_warp 3.16 collision_gjk_test.test_box_box_horizon
(Apache-2.0). Its 13-edge case lets us test capture/replay without pretending it
reproduces the user's reported 24-edge GPU failure.
"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import mujoco
import mujoco_warp as mjw
import numpy as np
from mjlab.envs.mdp.dr.body import body_com_offset
from mjlab.sim import Simulation, SimulationCfg

from scripts.replay_epa_dump import compare_contacts, replay_epa_dump, temporary_epa_horizon
from src.utils.epa_capture import EpaCaptureError, install_epa_capture


class EpaCaptureTests(unittest.TestCase):
  def setUp(self):
    self.directory = TemporaryDirectory()
    self.addCleanup(self.directory.cleanup)
    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <geom name="fixed" type="box" size=".025 .025 .025"/>
      <body><freejoint/><geom name="moving" type="box" size=".025 .025 .025"/></body>
      </worldbody></mujoco>""")
    self.sim = Simulation(num_envs=2, cfg=SimulationCfg(nconmax=64, njmax=64), model=model, device="cpu")
    self.sim.expand_model_fields(body_com_offset.model_fields)
    positions = np.array([[.065118454396725, -.125125020742416, .124963559210300],
                          [.065104484558105, -.124979749321938, .174992129206657]], dtype=np.float32)
    rotations = np.array([
      [[.996357858181000, .085266821086407, -.000942531623878],
       [-.085266284644604, .996358215808868, .000591202871874],
       [.000989508931525, -.000508683384396, .999999582767487]],
      [[.996556758880615, -.082913912832737, -.000453041866422],
       [.082915119826794, .996536433696747, .006357696373016],
       [-.000075668765930, -.006373368669301, .999979794025421]],
    ], dtype=np.float32)
    separate = positions.copy()
    separate[1, 2] += 1
    self.sim.wp_data.geom_xpos.assign(np.stack([separate, positions]))
    self.sim.wp_data.geom_xmat.assign(np.broadcast_to(rotations, (2, 2, 3, 3)).copy())
    # Collision-only operation preserves the recorded float32 transforms.
    self.sim.forward = lambda: mjw.collision(self.sim.wp_model, self.sim.wp_data)

  def test_real_overflow_is_saved_and_replays_at_larger_capacities(self):
    guard = install_epa_capture(self.sim, self.directory.name)
    with temporary_epa_horizon(5):
      with self.assertRaises(EpaCaptureError):
        self.sim.forward()
    with np.load(guard.dump_path, allow_pickle=False) as archive:
      metadata = json.loads(archive["metadata_json"].item())
      self.assertEqual(metadata["dumped_env_ids"], [1])
      self.assertEqual(metadata["epa_horizon"], 5)
      self.assertEqual(metadata["capture_status"], "complete", metadata["capture_errors"])
      self.assertIn("tendon_length0", metadata["skipped_model_fields"])
      self.assertEqual(archive["geom_xpos"].shape, (1, 2, 3))
    self.assertTrue(Path(self.directory.name, "epa_dump_latest.npz").is_file())
    report = replay_epa_dump(guard.dump_path, capacities=(24, 64, 128))
    self.assertTrue(report["baseline_reproduced"], report)
    self.assertTrue(report["passed"], report)
    self.assertGreater(report["reference"]["contacts"], report["baseline"]["contacts"])
    self.assertTrue(all(case["passed"] for case in report["candidates"]))
    self.assertEqual(set(report["recovered_geom_pairs"][0]["geom_names"]), {"fixed", "moving"})
    self.assertTrue(self.sim.wp_model.opt.warn_overflow & int(mjw.OverflowType.EPA_HORIZON))

  def test_unrelated_diagnostics_do_not_trigger_or_get_cleared(self):
    flags = np.array([int(mjw.OverflowType.LS_ITERATIONS), 0], dtype=np.int32)
    self.sim.wp_data.overflow.assign(flags)
    guard = install_epa_capture(self.sim, self.directory.name)
    np.testing.assert_equal(self.sim.wp_data.overflow.numpy(), flags)
    guard.check("forward")
    self.assertIsNone(guard.dump_path)
    self.assertEqual(list(Path(self.directory.name).iterdir()), [])

  def test_model_save_failure_preserves_core_and_original_physics_error(self):
    def failed_operation():
      self.sim.wp_data.overflow.assign(np.array([0, int(mjw.OverflowType.EPA_HORIZON)], dtype=np.int32))
      raise ValueError("original physics failure")

    self.sim.forward = failed_operation
    guard = install_epa_capture(self.sim, self.directory.name)
    with patch("src.utils.epa_capture.mujoco.mj_saveModel", side_effect=OSError("save failed")):
      with self.assertRaisesRegex(ValueError, "original physics failure"):
        self.sim.forward()
    with np.load(guard.dump_path, allow_pickle=False) as archive:
      metadata = json.loads(archive["metadata_json"].item())
      self.assertEqual(metadata["capture_status"], "partial")
      self.assertIn("model_file", metadata["capture_errors"])
      self.assertIn("geom_xpos", archive)
    with self.assertRaisesRegex(ValueError, "incomplete"):
      replay_epa_dump(guard.dump_path)


class EpaReplayVerdictTests(unittest.TestCase):
  def test_reported_seven_contacts_can_pass_without_contact_count_increase(self):
    # Regression for the GPU report: 24 overflows; 30 and 256 both produce the
    # same seven sane contacts. Mock only collision execution, not the verdict.
    contacts = {
      "worldid": np.zeros(7, dtype=np.int32),
      "geom": np.tile([0, 1], (7, 1)),
      "pos": np.arange(21, dtype=np.float32).reshape(7, 3) * .001,
      "dist": np.full(7, -.001, dtype=np.float32),
      "frame": np.tile(np.eye(3, dtype=np.float32), (7, 1, 1)),
    }
    metadata = {"format": "epa_collision_snapshot_v1", "capture_status": "complete",
                "versions": {}, "epa_horizon": 24, "model_file": "unused.mjb",
                "device": "cuda:0", "dumped_env_ids": [3569], "geom_names": ["terrain", "wheel"]}

    def run(*args):
      capacity = args[-1]
      overflow = capacity == 24
      return ({"capacity": capacity, "overflow": [256 if overflow else 0],
               "epa_overflow": overflow, "other_overflow": False,
               "contacts": 7, "contact_sanity": True}, contacts)

    with TemporaryDirectory() as directory:
      path = Path(directory, "epa.npz")
      np.savez(path, metadata_json=json.dumps(metadata), geom_xpos=np.zeros((1, 2, 3)),
               geom_xmat=np.tile(np.eye(3), (1, 2, 1, 1)))
      with patch("scripts.replay_epa_dump._run_collision", side_effect=run):
        report = replay_epa_dump(path, capacities=(30, 256))
      self.assertTrue(report["baseline_reproduced"])
      self.assertTrue(report["passed"], report)
      self.assertTrue(report["candidates"][0]["passed"])
      self.assertEqual(report["recovered_geom_pairs"], [])
      self.assertEqual(report["candidates"][0]["max_position_error_m"], 0)

  def test_equal_counts_still_require_matching_contact_positions(self):
    contacts = {"worldid": np.array([0]), "geom": np.array([[0, 1]]),
                "pos": np.zeros((1, 3)), "dist": np.zeros(1), "frame": np.eye(3)[None]}
    shifted = {**contacts, "pos": np.array([[0, 0, .02]])}
    self.assertFalse(compare_contacts(shifted, contacts)["matches"])


if __name__ == "__main__":
  unittest.main()
