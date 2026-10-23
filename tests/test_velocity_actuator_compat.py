"""Native upstream velocity servos must keep continuous wheels unrestricted."""

import unittest

import mujoco
from mjlab.actuator import builtin_actuator


class VelocityActuatorCompatibilityTests(unittest.TestCase):
  def make_spec(self, joint_attributes):
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

  def test_bounded_joint_velocity_target_is_not_a_position_range(self):
    spec = self.make_spec('limited="true" range="-2 2"')
    builtin_actuator.create_velocity_actuator(spec, "wheel", damping=2.0, effort_limit=45.0)
    model=spec.compile()
    self.assertTrue(model.jnt_limited[0])
    self.assertFalse(model.actuator_ctrllimited[0])
    self.assertEqual(model.actuator_forcerange[0].tolist(), [-45.0,45.0])


if __name__ == "__main__":
  unittest.main()
