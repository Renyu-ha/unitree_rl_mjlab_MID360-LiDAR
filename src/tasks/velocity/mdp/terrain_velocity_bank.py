"""Allowed terrain/velocity pairs with equal base mass per terrain tile."""

from dataclasses import dataclass

import torch

from .velocity_buckets import VelocityBuckets


STAIR_TERRAINS = frozenset((
  "pyramid_stairs", "pyramid_stairs_inv", "pyramid_stairs_2", "pyramid_stairs_inv_2",
))


@dataclass
class TerrainVelocityBank:
  rows: int
  terrain_names: tuple[str, ...]
  tile_ids: torch.Tensor
  bucket_ids: torch.Tensor
  group_ids: torch.Tensor
  base_probs: torch.Tensor
  longitudinal_tiles: torch.Tensor

  @property
  def size(self):
    return self.tile_ids.numel()

  def decode(self, config_ids):
    return self.tile_ids[config_ids], self.bucket_ids[config_ids]

  def description(self):
    return {"rows": self.rows, "terrain_names": self.terrain_names,
      "tile_ids": self.tile_ids.cpu().tolist(), "bucket_ids": self.bucket_ids.cpu().tolist(),
      "longitudinal_tiles": self.longitudinal_tiles.cpu().tolist()}


def make_terrain_velocity_bank(rows, terrain_names, buckets: VelocityBuckets, *, longitudinal_from_level=6):
  """Stairs always use body X; other non-flat tiles use body X from the given row.

  A compact ID identifies an allowed pair, never a disabled Cartesian-product
  entry. Base sampling chooses a tile, an allowed category, then its speed bin.
  """
  names = tuple(terrain_names)
  if rows < 1 or not names or len(set(names)) != len(names):
    raise ValueError("SGS bank needs positive rows and unique terrain column names")
  if longitudinal_from_level is not None and longitudinal_from_level < 0:
    raise ValueError("SGS longitudinal_from_level must be nonnegative or None")
  device = buckets.lower.device
  tile_count = rows * len(names)
  restricted = torch.tensor([
    name in STAIR_TERRAINS or (name != "flat" and longitudinal_from_level is not None
      and level >= longitudinal_from_level)
    for level in range(rows) for name in names
  ], device=device, dtype=torch.bool)
  longitudinal = ((buckets.modes == 0) & (buckets.lower[:, :, 1:] == 0).all(dim=(1, 2))
    & (buckets.upper[:, :, 1:] == 0).all(dim=(1, 2)))
  if not bool(longitudinal.any()):
    raise ValueError("SGS bank needs at least one body-longitudinal velocity bucket")
  allowed = ~restricted[:, None] | longitudinal[None, :]
  tile_ids, bucket_ids = allowed.nonzero(as_tuple=True)
  keys = tile_ids * len(buckets.categories) + buckets.category_ids[bucket_ids]
  group_keys, groups, bins_per_group = torch.unique(keys, sorted=True,
    return_inverse=True, return_counts=True)
  categories_per_tile = torch.bincount(group_keys // len(buckets.categories), minlength=tile_count)
  base = 1.0 / (tile_count * categories_per_tile[tile_ids] * bins_per_group[groups]).float()
  return TerrainVelocityBank(rows, names, tile_ids, bucket_ids, groups, base, restricted)
