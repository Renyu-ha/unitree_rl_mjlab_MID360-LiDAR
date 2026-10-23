"""Read-only browser visualization of the environment used for training."""

from __future__ import annotations

import math
import time
from types import SimpleNamespace
from typing import Any

from mjlab.envs import ManagerBasedRlEnv


class LiveTrainingViewer(ManagerBasedRlEnv):
  """Observe rollout steps without running a separate policy or simulation loop.

  Only the selected environment is copied to the browser. Synchronization runs
  on the training thread after a step, at a bounded wall-clock frame rate.
  Browser callbacks only change display options, never the training state.
  """

  def __init__(
    self,
    env: ManagerBasedRlEnv,
    *,
    host: str = "127.0.0.1",
    port: int = 8080,
    fps: float = 15.0,
  ):
    import viser
    from mjlab.viewer.viser.scene import ViserMujocoScene

    if not math.isfinite(fps) or fps <= 0:
      raise ValueError("viewer_fps must be finite and positive")
    if not 0 <= env.cfg.viewer.env_idx < env.num_envs:
      raise ValueError("env.viewer.env_idx must select an existing environment")
    if not 1 <= port <= 65535:
      raise ValueError("viewer_port must be between 1 and 65535")
    self._wrapped_env = env
    self._frame_interval = 1.0 / fps
    self._last_update = float("-inf")
    self._selected_env = env.cfg.viewer.env_idx
    self._server = viser.ViserServer(host=host, port=port, label="Training")
    try:
      # A single display instance avoids copying/rendering thousands of worlds.
      self._scene = ViserMujocoScene.create(
        self._server, env.unwrapped.sim.mj_model, num_envs=1,
      )
      self._server.gui.add_markdown(
        "Live training rollout. Display controls do not pause or reset training."
      )
      self._env_selector = self._server.gui.add_slider(
        "Training environment", min=0, max=max(1, env.num_envs - 1), step=1,
        initial_value=self._selected_env, disabled=env.num_envs == 1,
      )
      self._debug_checkbox = self._server.gui.add_checkbox(
        "Sensors and commands", initial_value=False,
      )
      self._scene.create_visualization_gui(
        camera_distance=env.cfg.viewer.distance,
        camera_azimuth=env.cfg.viewer.azimuth,
        camera_elevation=env.cfg.viewer.elevation,
        show_debug_viz_control=False,
      )
    except BaseException:
      self._server.stop()
      raise
    display_host = "127.0.0.1" if host == "0.0.0.0" else host
    print(f"[INFO] Live training viewer: http://{display_host}:{self._server.get_port()}")

  def __getattr__(self, name: str) -> Any:
    return getattr(self._wrapped_env, name)

  @property
  def unwrapped(self) -> ManagerBasedRlEnv:
    return self._wrapped_env.unwrapped

  def reset(self, **kwargs: Any) -> Any:
    result = self._wrapped_env.reset(**kwargs)
    self.update(force=True)
    return result

  def step(self, action: Any) -> Any:
    result = self._wrapped_env.step(action)
    self.update()
    return result

  def update(self, *, force: bool = False) -> None:
    if not force and (
      not self._server.get_clients()
      or time.monotonic() - self._last_update < self._frame_interval
    ):
      return
    index = int(self._env_selector.value)
    if index != self._selected_env or (
      self._scene.debug_visualization_enabled and not self._debug_checkbox.value
    ):
      self._scene.clear_debug_all()
    self._selected_env = index
    self._scene.env_idx = index
    self._scene.debug_visualization_enabled = self._debug_checkbox.value
    if self._scene.debug_visualization_enabled:
      self._scene.clear()
      self.unwrapped.update_visualizers(self._scene)
    data = self.unwrapped.sim.data
    # Slice on the simulation device before the scene transfers tensors to CPU.
    selected = SimpleNamespace(
      xpos=data.xpos[index:index + 1], xmat=data.xmat[index:index + 1],
      qpos=data.qpos[index:index + 1], qvel=data.qvel[index:index + 1],
    )
    if self.unwrapped.sim.mj_model.nmocap > 0:
      selected.mocap_pos = data.mocap_pos[index:index + 1]
      selected.mocap_quat = data.mocap_quat[index:index + 1]
    self._scene.update(selected, env_idx=0)
    self._last_update = time.monotonic()

  def close(self) -> None:
    try:
      self._wrapped_env.close()
    finally:
      self._server.stop()
