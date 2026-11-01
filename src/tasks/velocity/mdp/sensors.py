"""Contact sensors with explicit query ordering for cross-sensor wheel data."""

from dataclasses import dataclass

from mjlab.sensor import ContactSensor, ContactSensorCfg, GridPatternCfg

__all__ = ["ElevatedTerrainGridPatternCfg", "OrderedContactSensorCfg"]


@dataclass(kw_only=True)
class ElevatedTerrainGridPatternCfg(GridPatternCfg):
  """Privileged ground truth from above terrain, independent of body clearance.

  Yaw-aligned rays keep this Z offset vertical. Unlike actor scans, these rays
  can see the top of an obstacle taller than their attached robot body.
  """

  origin_height: float = 10.0

  def generate_rays(self, mj_model, device):
    if self.origin_height <= 0:
      raise ValueError("Terrain ground-truth ray height must be positive")
    offsets, directions = super().generate_rays(mj_model, device)
    offsets[:, 2] += self.origin_height
    return offsets, directions


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
