"""Verify diagnostic controls reach actual CCD kernels and restore on exit."""

import unittest

import mujoco
import mujoco_warp as mjw
from mujoco_warp._src import collision_convex
import numpy as np
import warp as wp

from scripts.inspect_nan_dump import contact_detail
from src.utils.collision_probe import collision_fma_mode, temporary_epa_horizon


class CollisionProbeTests(unittest.TestCase):
  def setUp(self):
    self.model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <geom name="terrain" type="box" size="1.8 1.8 .02" pos="0 0 -.02"/>
      <body pos=".2 .3 .08"><freejoint/>
        <geom name="wheel" type="cylinder" size=".086 .0225" quat="1 1 0 0"/>
      </body></worldbody></mujoco>""")
    with wp.ScopedDevice("cpu"):
      self.wm = mjw.put_model(self.model)
      self.data = mjw.make_data(self.model, nworld=1, nconmax=64, njmax=128)
      mjw.kinematics(self.wm, self.data)

  def test_fma_override_reaches_real_ccd_kernel_and_is_restored(self):
    builder = collision_convex.ccd_kernel_builder
    for mode in ("off", "on"):
      with self.subTest(mode=mode), collision_fma_mode(mode) as audit:
        mjw.collision(self.wm, self.data)
        wp.synchronize()
        self.assertGreater(int(self.data.nacon.numpy()[0]), 0)
        self.assertTrue(audit)
        for entry in audit.values():
          self.assertEqual(entry["fuse_fp"], mode == "on")
          self.assertEqual(wp.get_module_options(entry["module"])["fuse_fp"], mode == "on")
      self.assertIs(collision_convex.ccd_kernel_builder, builder)
      for entry in audit.values():
        self.assertTrue(wp.get_module_options(entry["module"])["fuse_fp"])

  def test_diagnostic_exception_restores_capacity_builders_and_options(self):
    capacity = collision_convex.MJ_MAX_EPAHORIZON
    builder = collision_convex.ccd_kernel_builder
    with self.assertRaisesRegex(RuntimeError, "original diagnostic error"):
      with temporary_epa_horizon(30), collision_fma_mode("off") as audit:
        mjw.collision(self.wm, self.data)
        wp.synchronize()
        self.assertEqual(collision_convex.MJ_MAX_EPAHORIZON, 30)
        raise RuntimeError("original diagnostic error")
    self.assertEqual(collision_convex.MJ_MAX_EPAHORIZON, capacity)
    self.assertIs(collision_convex.ccd_kernel_builder, builder)
    for entry in audit.values():
      self.assertTrue(wp.get_module_options(entry["module"])["fuse_fp"])

  def test_finite_but_far_away_contact_is_rejected_before_solver(self):
    centers = np.array([[0, 0, -.02], [.2, .3, .08]])
    pair = np.array([1, 0])
    normal = np.eye(3)
    far = contact_detail(self.model, pair, np.array([.2, 62, 0]), -.0000188, normal, centers)
    near = contact_detail(self.model, pair, np.array([.2, .3, 0]), -.0000188, normal, centers)
    self.assertFalse(far["contact_sanity"])
    self.assertTrue(near["contact_sanity"])
    invalid = contact_detail(self.model, pair, np.array([np.nan, .3, 0]), -.0000188, normal, centers)
    self.assertFalse(invalid["contact_sanity"])
    self.assertIsNone(invalid["position"][0])


if __name__ == "__main__":
  unittest.main()
