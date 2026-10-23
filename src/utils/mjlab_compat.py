"""Compatibility fixes for the pinned mjlab 1.2.0 dependency."""

from functools import wraps

from mjlab.actuator import builtin_actuator
from mjlab.actuator.actuator import TransmissionType
from mjlab.utils import spec as spec_utils


def enable_continuous_velocity_actuators() -> None:
  """Apply ZhongL1's continuous-joint range handling without editing mjlab.

  BuiltinVelocityActuator imports the factory into its own module, so both
  references must be updated. The wrapper preserves the built-in actuator
  type and its batching path; bounded joints keep their existing behavior.
  """
  original = builtin_actuator.create_velocity_actuator
  if getattr(original, "_continuous_velocity_compat", False):
    return

  @wraps(original)
  def create_velocity_actuator(spec, joint_name, **kwargs):
    continuous = (
      kwargs.get("transmission_type", TransmissionType.JOINT) == TransmissionType.JOINT
      and not spec_utils.is_joint_limited(spec.joint(joint_name))
    )
    if continuous:
      kwargs["inheritrange"] = 0.0
    actuator = original(spec, joint_name, **kwargs)
    if continuous:
      actuator.inheritrange = 0.0
      actuator.ctrllimited = False
    return actuator

  create_velocity_actuator._continuous_velocity_compat = True
  spec_utils.create_velocity_actuator = create_velocity_actuator
  builtin_actuator.create_velocity_actuator = create_velocity_actuator
