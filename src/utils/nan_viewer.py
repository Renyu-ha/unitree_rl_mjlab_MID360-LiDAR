"""NaN replay with per-frame status and honest handling of invalid poses."""

from html import escape

import mujoco
import numpy as np
from mjlab.scripts.nan_viz import NanDumpViewer


class PhysicsNanDumpViewer(NanDumpViewer):
  def _get_info_html(self):
    key = self.state_keys[self.current_step]
    state = self.dump[key][self.current_env]
    failure = key == self.metadata.get("failure_state_key")
    phase = self.metadata.get("frame_phases", {}).get(key, "before_step (legacy dump)")
    counts = []
    if failure:
      for name in ("qpos", "qvel", "act", "ctrl", "qacc", "qacc_warmstart", "sensordata",
                   "qacc_smooth", "qfrc_actuator", "qfrc_constraint"):
        array_key = f"failure_{name}"
        if array_key in self.dump:
          value = self.dump[array_key][self.current_env]
          count = int((~np.isfinite(value)).sum())
          counts.append(f"{name}: {count} nonfinite")
    else:
      counts.append(f"stored physics state: {int((~np.isfinite(state)).sum())} nonfinite")
    status = "RECORDED FAILURE" if failure else "PRE-OPERATION STATE"
    has_predecessor = any(np.isfinite(self.dump[k][self.current_env]).all()
                          for k in self.state_keys[:self.current_step])
    render_note = (
      ("Invalid physics state cannot be rendered. The scene holds the previous renderable pose."
       if has_predecessor else
       "No renderable saved pose. The scene shows the model default pose.")
      if not np.isfinite(state).all() else
      "Pose is finite. A finite pose does not imply finite acceleration or sensors."
    )
    if failure and "failure_contact_worldid" in self.dump:
      env_id = self.dumped_env_ids[self.current_env]
      mask = self.dump["failure_contact_worldid"] == env_id
      counts.append(f"Recorded Warp contacts: {int(mask.sum())}")
      if "failure_contact_dist" in self.dump and mask.any():
        distances = self.dump["failure_contact_dist"][mask]
        counts.append(f"Recorded minimum contact distance: {float(distances.min()):.6f} m")
    legacy_note = "" if "failure_state_key" in self.metadata else (
      "<br/><strong>Legacy dump:</strong> failure result, acceleration and GPU contacts "
      "were not recorded. The last frame precedes detection."
    )
    return (
      '<div style="font-size: 0.85em; line-height: 1.4; padding: 0 1em 0.5em 1em;">'
      f"<strong>Frame:</strong> {escape(key)}<br/>"
      f"<strong>Environment:</strong> {self.dumped_env_ids[self.current_env]}<br/>"
      f"<strong>Status:</strong> {status}<br/>"
      f"<strong>Phase:</strong> {escape(phase)}<br/>"
      + "<br/>".join(counts) + "<br/>" + render_note + legacy_note + "</div>"
    )

  def _update_state(self):
    key = self.state_keys[self.current_step]
    state = self.dump[key][self.current_env]
    # Never send NaN transforms to Viser or silently replace invalid angles.
    # Find a renderable predecessor for the same environment, so switching
    # worlds on the failure frame cannot leave another world's pose on screen.
    render_state = state
    if not np.isfinite(state).all():
      render_state = next(
        (self.dump[k][self.current_env] for k in reversed(self.state_keys[:self.current_step])
         if np.isfinite(self.dump[k][self.current_env]).all()), None,
      )
    mujoco.mj_resetData(self.model, self.data)
    if render_state is None:
      # Default pose is only a display fallback; status still reports failure.
      mujoco.mj_fwdPosition(self.model, self.data)
    else:
      state_spec = self.metadata.get("state_spec", mujoco.mjtState.mjSTATE_PHYSICS.value)
      mujoco.mj_setState(self.model, self.data, render_state, state_spec)
      # Position only: CPU forces/acceleration must not masquerade as the
      # recorded GPU failure. Contact geometry is still CPU reconstructed.
      mujoco.mj_fwdPosition(self.model, self.data)
    self.scene.update_from_mjdata(self.data)
    self.info_html.content = self._get_info_html()

  def setup(self):
    super().setup()
    # Start at the relevant frame instead of the oldest history frame.
    self.current_step = self.num_steps - 1
    self.step_slider.value = self.current_step
    self._update_state()
