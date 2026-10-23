"""One wheel-ground sensor exposing total and normal-only forces per wheel."""

from dataclasses import dataclass

import torch
from mjlab.sensor import ContactSensor, ContactSensorCfg
from mjlab.sensor.contact_sensor import ContactData, _ContactSlot


@dataclass
class WheelGroundContactData(ContactData):
  net_normal_force_w: torch.Tensor | None = None


def aggregate_wheel_contacts(data: ContactData, num_slots: int) -> WheelGroundContactData:
  """Reduce raw contact-frame slots without mixing friction into normal forces.

  MuJoCo stores force components in normal, first-tangent, second-tangent
  order. Its normal and first tangent already include the primary-side sign;
  the third force component includes that sign itself.
  """
  assert data.found is not None and data.force is not None
  assert data.normal is not None and data.tangent is not None
  force = data.force.unflatten(1, (-1, num_slots))
  normal = data.normal.reshape_as(force)
  tangent = data.tangent.reshape_as(force)
  second_tangent = torch.cross(normal, tangent, dim=-1)
  normal_force = force[..., :1] * normal
  total_force = normal_force + force[..., 1:2] * tangent + force[..., 2:3] * second_tangent
  return WheelGroundContactData(
    found=data.found.unflatten(1, (-1, num_slots)).amax(dim=-1),
    force=total_force.sum(dim=2),
    net_normal_force_w=normal_force.sum(dim=2),
  )


class WheelGroundContactSensor(ContactSensor):
  """Keep per-wheel observation/air-time shapes while sharing raw contacts."""

  def __init__(self, cfg: ContactSensorCfg):
    if cfg.reduce != "none" or cfg.global_frame or cfg.history_length != 0:
      raise ValueError("Wheel ground contacts require raw local slots without history")
    if not {"found", "force", "normal", "tangent"}.issubset(cfg.fields):
      raise ValueError("Wheel ground contacts require found, force, normal and tangent")
    super().__init__(cfg)

  def edit_spec(self, scene_spec, entities) -> None:
    """Pack all four fields into one MuJoCo sensor for each wheel.

    Separate raw sensors can assign different slot orders on CUDA. A combined
    sensor keeps each force paired with its own normal and tangent.
    """
    self._slots.clear()
    primaries = self._resolve_primary_names(entities, self.cfg.primary)
    secondary = None
    if self.cfg.secondary is not None and self.cfg.secondary_policy != "any":
      secondary = self._resolve_single_secondary(entities, self.cfg.secondary, self.cfg.secondary_policy)
    for primary in primaries:
      name = f"{self.cfg.name}_{primary}_contacts"
      self._add_contact_sensor_to_spec(scene_spec, name, primary, secondary, "force")
      # MuJoCo contact field bits: found=0, force=1, normal=5, tangent=6.
      scene_spec.sensors[-1].intprm[0] = (1 << 0) | (1 << 1) | (1 << 5) | (1 << 6)
      self._slots.append(_ContactSlot(primary_name=primary, field_name="contacts", sensor_name=name))

  def _extract_sensor_data(self) -> WheelGroundContactData:
    packed = torch.stack([slot.data_view for slot in self._slots], dim=1)
    packed = packed.unflatten(2, (self.cfg.num_slots, 10))
    raw = ContactData(
      found=packed[..., 0].flatten(1, 2),
      force=packed[..., 1:4].flatten(1, 2),
      normal=packed[..., 4:7].flatten(1, 2),
      tangent=packed[..., 7:10].flatten(1, 2),
    )
    return aggregate_wheel_contacts(raw, self.cfg.num_slots)


@dataclass
class WheelGroundContactSensorCfg(ContactSensorCfg):
  def build(self) -> WheelGroundContactSensor:
    return WheelGroundContactSensor(self)
