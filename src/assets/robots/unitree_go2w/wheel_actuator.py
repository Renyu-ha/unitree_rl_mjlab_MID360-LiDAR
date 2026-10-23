"""Velocity servos for continuous wheels in mjlab 1.2.0."""

from dataclasses import dataclass

import mujoco
import torch
from mjlab.actuator.actuator import Actuator, ActuatorCfg, ActuatorCmd
from mjlab.utils.spec import create_velocity_actuator


@dataclass(kw_only=True)
class WheelVelocityActuatorCfg(ActuatorCfg):
  damping: float = 2.0
  effort_limit: float = 45.0
  velocity_limit: float = 20.0

  def build(self, entity, target_ids, target_names):
    return WheelVelocityActuator(self, entity, target_ids, target_names)


class WheelVelocityActuator(Actuator[WheelVelocityActuatorCfg]):
  def edit_spec(self, spec: mujoco.MjSpec, target_names: list[str]) -> None:
    # mjlab 1.2.0 inherits velocity ctrlrange from joint position limits. A
    # continuous joint has no position range; set an explicit speed range.
    # Use the custom-actuator interface: the built-in batching table in 1.2.0
    # dispatches on exact concrete types and cannot handle built-in subclasses.
    for name in target_names:
      actuator = create_velocity_actuator(
        spec, name, damping=self.cfg.damping, effort_limit=self.cfg.effort_limit,
        armature=self.cfg.armature, frictionloss=self.cfg.frictionloss,
        inheritrange=0.0,
      )
      actuator.ctrllimited = True
      actuator.ctrlrange[:] = (-self.cfg.velocity_limit, self.cfg.velocity_limit)
      self._mjs_actuators.append(actuator)

  def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
    return cmd.velocity_target
