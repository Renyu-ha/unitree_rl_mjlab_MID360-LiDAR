"""Contact sensors with explicit query ordering for cross-sensor wheel data."""

from dataclasses import dataclass

from mjlab.sensor import ContactSensor, ContactSensorCfg

__all__ = ["OrderedContactSensorCfg"]


class OrderedContactSensor(ContactSensor):
  cfg: "OrderedContactSensorCfg"

  def _resolve_primary_names(self, entities, match):
    names = super()._resolve_primary_names(entities, match)
    if not self.cfg.preserve_order or match.entity in (None, ""):
      return names
    entity = entities[match.entity]
    find = entity.find_geoms if match.mode == "geom" else entity.find_bodies
    _, ordered = find(match.pattern, preserve_order=True)
    # Keep native validation and exclusion semantics; only change ordering.
    allowed = set(names)
    return [name for name in ordered if name in allowed]


@dataclass(kw_only=True)
class OrderedContactSensorCfg(ContactSensorCfg):
  preserve_order: bool = True

  def build(self):
    return OrderedContactSensor(self)
