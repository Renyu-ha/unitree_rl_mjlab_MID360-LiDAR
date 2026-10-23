"""Continuous wheel servos and bounded-joint compatibility."""

import unittest

import mujoco
from mjlab.actuator import builtin_actuator
from mjlab.utils import spec as spec_utils

from src.utils.mjlab_compat import enable_continuous_velocity_actuators


class VelocityActuatorCompatibilityTests(unittest.TestCase):
  def make_spec(self, joint_attributes):
    enable_continuous_velocity_actuators()
    return mujoco.MjSpec.from_string(f"""
      <mujoco><compiler angle="radian"/><worldbody><body>
        <joint name="wheel" type="hinge" {joint_attributes}/>
        <geom type="sphere" size="0.1" mass="1"/>
      </body></worldbody></mujoco>
    """)

  def test_continuous_joint_ignores_empty_or_informational_position_range(self):
    for attributes in ('limited="false"', 'limited="false" range="-2 2"'):
      with self.subTest(attributes=attributes):
        spec = self.make_spec(attributes)
        builtin_actuator.create_velocity_actuator(
          spec, "wheel", damping=2.0, effort_limit=45.0, armature=0.0,
        )
        model = spec.compile()
        self.assertFalse(model.jnt_limited[0])
        self.assertFalse(model.actuator_ctrllimited[0])
        self.assertEqual(model.actuator_forcerange[0].tolist(), [-45.0, 45.0])
        # Native MuJoCo still computes the velocity error and clamps torque.
        data = mujoco.MjData(model)
        data.ctrl[0] = 5.0
        mujoco.mj_forward(model, data)
        self.assertAlmostEqual(data.actuator_force[0], 10.0)
        data.ctrl[0] = 100.0
        mujoco.mj_forward(model, data)
        self.assertAlmostEqual(data.actuator_force[0], 45.0)

  def test_bounded_joint_keeps_original_factory_settings(self):
    spec = self.make_spec('limited="true" range="-2 2"')
    reference = self.make_spec('limited="true" range="-2 2"')
    factory = builtin_actuator.create_velocity_actuator
    actual = factory(spec, "wheel", damping=2.0, effort_limit=45.0)
    expected = factory.__wrapped__(reference, "wheel", damping=2.0, effort_limit=45.0)
    for field in ("inheritrange", "ctrllimited", "forcelimited"):
      self.assertEqual(getattr(actual, field), getattr(expected, field), field)
    for field in ("ctrlrange", "forcerange", "gainprm", "biasprm"):
      self.assertEqual(list(getattr(actual, field)), list(getattr(expected, field)), field)

  def test_patch_is_idempotent_and_updates_both_factory_references(self):
    enable_continuous_velocity_actuators()
    factory = builtin_actuator.create_velocity_actuator
    enable_continuous_velocity_actuators()
    self.assertIs(builtin_actuator.create_velocity_actuator, factory)
    self.assertIs(spec_utils.create_velocity_actuator, factory)


if __name__ == "__main__":
  unittest.main()
