"""Joint actions that honor the requested target order on the pinned mjlab."""

from dataclasses import dataclass

from mjlab.envs.mdp.actions import (
  JointPositionAction as _JointPositionAction,
  JointPositionActionCfg as _JointPositionActionCfg,
  JointVelocityAction as _JointVelocityAction,
  JointVelocityActionCfg as _JointVelocityActionCfg,
)
from mjlab.utils.lab_api.string import resolve_matching_names

__all__ = ["OrderedJointPositionActionCfg", "OrderedJointVelocityActionCfg"]


class _OrderedJointTargets:
  def _find_targets(self, cfg):
    # First use native resolution to keep only actuated joints. Native joint
    # actions currently ignore preserve_order; reorder their names and IDs here.
    ids, names = super()._find_targets(cfg)
    if cfg.preserve_order:
      order, names = resolve_matching_names(cfg.actuator_names, names, preserve_order=True)
      ids = [ids[index] for index in order]
    return ids, names


class _OrderedJointPositionAction(_OrderedJointTargets, _JointPositionAction):
  pass


class _OrderedJointVelocityAction(_OrderedJointTargets, _JointVelocityAction):
  pass


@dataclass(kw_only=True)
class OrderedJointPositionActionCfg(_JointPositionActionCfg):
  preserve_order: bool = True

  def build(self, env):
    return _OrderedJointPositionAction(self, env)


@dataclass(kw_only=True)
class OrderedJointVelocityActionCfg(_JointVelocityActionCfg):
  preserve_order: bool = True

  def build(self, env):
    return _OrderedJointVelocityAction(self, env)
