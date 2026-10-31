"""Stationary command distributions, partitioned by motion category and speed.

Each bucket has two phases. Ordinary buckets repeat their bounds; transition
buckets alternate bounds. Values and durations are drawn afresh at runtime.
"""

from dataclasses import dataclass

import torch


@dataclass
class VelocityBuckets:
  names: tuple[str, ...]
  categories: tuple[str, ...]
  category_ids: torch.Tensor
  lower: torch.Tensor
  upper: torch.Tensor
  modes: torch.Tensor  # 0 body, 1 world, 2 heading

  @property
  def size(self):
    return len(self.names)

  def description(self):
    return {"categories": self.categories, "names": self.names,
      "lower": self.lower.cpu().tolist(), "upper": self.upper.cpu().tolist(),
      "modes": self.modes.cpu().tolist(), "category_ids": self.category_ids.cpu().tolist()}


def make_velocity_buckets(ranges, subdivisions=8, *, heading_command=True, device="cpu"):
  if subdivisions < 1:
    raise ValueError("Velocity subdivisions must be positive")
  bounds = (ranges.lin_vel_x, ranges.lin_vel_y, ranges.ang_vel_z)
  if any(not (-float("inf") < lo <= -0.1 and 0.1 <= hi < float("inf")) for lo, hi in bounds):
    raise ValueError("SGS velocity ranges must include both signs with magnitude at least 0.1")
  categories, names, groups, lowers, uppers, modes = [], [], [], [], [], []
  zero = [(0., 0.)] * 3

  def axis_bounds(axis, sign, lo_fraction=0., hi_fraction=1.):
    maximum = bounds[axis][1] if sign > 0 else -bounds[axis][0]
    lo = .1 + (maximum - .1) * lo_fraction
    hi = .1 + (maximum - .1) * hi_fraction
    return (lo, hi) if sign > 0 else (-hi, -lo)

  def add(name, axis=None, sign=1, fractions=(0., 1.), *, mode=0, mixed=False, transition=None):
    group = len(categories)
    categories.append(name)
    count = 1 if axis is None else subdivisions
    for j in range(count):
      interval = (fractions[0] + (fractions[1] - fractions[0]) * j / count,
        fractions[0] + (fractions[1] - fractions[0]) * (j + 1) / count)
      phase = list(bounds) if mixed else list(zero)
      if axis is not None:
        phase[axis] = axis_bounds(axis, sign, *interval)
      if mode == 2:
        phase[2] = (0., 0.)  # yaw comes from heading feedback, not a sampled rate
      other = list(phase)
      if transition == "start":
        phase = list(zero)
      elif transition == "stop":
        other = list(zero)
      elif transition == "reverse":
        other[axis] = axis_bounds(axis, -sign, *interval)
      names.append(f"{name}/{j}")
      groups.append(group)
      lowers.append([[ab[0] for ab in phase], [ab[0] for ab in other]])
      uppers.append([[ab[1] for ab in phase], [ab[1] for ab in other]])
      modes.append(mode)

  add("stand")
  for name, fractions in (("forward_low", (0., .3)), ("forward_mid", (.3, .6)), ("forward_high", (.6, 1.))):
    add(name, 0, fractions=fractions)
  for sign, direction in ((1, "left"), (-1, "right")):
    for speed, fractions in (("low", (0., .5)), ("high", (.5, 1.))):
      add(f"side_{direction}_{speed}", 1, sign, fractions)
      add(f"turn_{direction}_{speed}", 2, sign, fractions)
  for speed, fractions in (("low", (0., .5)), ("high", (.5, 1.))):
    add(f"backward_{speed}", 0, -1, fractions)
  for sign, direction in ((1, "forward"), (-1, "backward")):
    add(f"mixed_{direction}", 0, sign, mixed=True)
    add(f"world_{direction}", 0, sign, mode=1, mixed=True)
    if heading_command:
      add(f"heading_{direction}", 0, sign, mode=2, mixed=True)
  add("stand_to_forward", 0, transition="start")
  add("forward_to_stand", 0, transition="stop")
  add("forward_backward", 0, transition="reverse")
  add("turn_left_right", 2, transition="reverse")
  return VelocityBuckets(tuple(names), tuple(categories),
    torch.tensor(groups, device=device, dtype=torch.long),
    torch.tensor(lowers, device=device), torch.tensor(uppers, device=device),
    torch.tensor(modes, device=device, dtype=torch.long))
