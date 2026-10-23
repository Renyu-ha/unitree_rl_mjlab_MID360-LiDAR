"""Inspect the four outer boxes emitted by mjlab 1.2's terrain generator."""

import re

import mujoco
import numpy as np


def find_terrain_border(model: mujoco.MjModel) -> tuple[int, ...]:
  """Recognize the generated outer ring, without assuming terrain geom IDs."""
  numbered = []
  for geom_id in range(model.ngeom):
    name = model.geom(geom_id).name.rsplit("/", 1)[-1]
    match = re.fullmatch(r"terrain_(\d+)", name)
    if match:
      numbered.append((int(match.group(1)), geom_id))
  if len(numbered) < 4:
    return ()
  ids = tuple(g for _, g in sorted(numbered)[-4:])
  top, bottom, left, right = ids
  if not all(model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX for g in ids):
    return ()
  p, s = model.geom_pos, model.geom_size
  # Validate make_border's top/bottom/left/right topology. A border-free model
  # must not mistake the last four ordinary terrain boxes for an outer ring.
  checks = (
    np.allclose(p[top, :2], -p[bottom, :2]),
    np.allclose(p[left, :2], -p[right, :2]),
    np.allclose(s[top], s[bottom]), np.allclose(s[left], s[right]),
    np.allclose(model.geom_quat[list(ids)], [1.0, 0.0, 0.0, 0.0]),
    np.allclose(p[list(ids), 2] + s[list(ids), 2], 0.0),
    abs(p[top, 0]) < 1e-8, abs(p[left, 1]) < 1e-8,
    p[top, 1] > 0.0, p[left, 0] < 0.0,
    np.isclose(p[top, 1] - s[top, 1], s[left, 1]),
    np.isclose(p[right, 0] - s[right, 0], s[top, 0] - 2 * s[right, 0]),
  )
  return ids if all(checks) else ()


def add_border_inspection(viewer) -> None:
  """Overlay the border and report contacts reconstructed for each saved pose."""
  ids = find_terrain_border(viewer.model)
  if not ids:
    viewer.server.gui.add_markdown("No standard four-box terrain border was identified.")
    return
  handles = []
  for geom_id in ids:
    handles.append(viewer.server.scene.add_box(
      f"/fixed_bodies/border_inspection/{geom_id}", color=(255, 140, 0),
      dimensions=2.0 * viewer.model.geom_size[geom_id],
      position=viewer.model.geom_pos[geom_id], wireframe=True,
    ))
  with viewer.server.gui.add_folder("Border inspection"):
    visible = viewer.server.gui.add_checkbox("Highlight outer border", initial_value=True)
    report = viewer.server.gui.add_markdown("")

  @visible.on_update
  def change_visibility(_):
    for handle in handles:
      handle.visible = visible.value

  def update_report(_=None):
    frame = int(viewer.step_slider.value)
    env_index = int(viewer.env_slider.value) if hasattr(viewer, "env_slider") else 0
    key = viewer.state_keys[frame]
    # Separate CPU data avoids sharing the playback callback's mutable MjData.
    data = mujoco.MjData(viewer.model)
    mujoco.mj_setState(viewer.model, data, viewer.dump[key][env_index], mujoco.mjtState.mjSTATE_PHYSICS)
    mujoco.mj_fwdPosition(viewer.model, data)
    names = []
    for contact in data.contact:
      a, b = int(contact.geom1), int(contact.geom2)
      if a in ids or b in ids:
        names.append(
          f"- `{viewer.model.geom(a).name}` / `{viewer.model.geom(b).name}`: "
          f"distance **{contact.dist * 1000:.2f} mm**"
        )
    top, bottom, left, right = ids
    inner_x = viewer.model.geom_pos[right, 0] - viewer.model.geom_size[right, 0]
    inner_y = viewer.model.geom_pos[top, 1] - viewer.model.geom_size[top, 1]
    report.content = (
      f"Frame **{frame}**, environment **{viewer.dumped_env_ids[env_index]}**\n\n"
      f"Root XY: **({data.qpos[0]:.5f}, {data.qpos[1]:.5f}) m**\n\n"
      f"Grid bounds: X ±{inner_x:g} m, Y ±{inner_y:g} m\n\n"
      "**CPU reconstructed border contacts** (negative distance means penetration):\n\n"
      + ("\n".join(names) if names else "None in this frame.")
      + "\n\nThese are reconstructed contacts, not recorded GPU forces."
    )

  viewer.step_slider.on_update(update_report)
  if hasattr(viewer, "env_slider"):
    viewer.env_slider.on_update(update_report)
  update_report()
