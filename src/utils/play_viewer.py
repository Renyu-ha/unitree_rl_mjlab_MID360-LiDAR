"""Adapt MuJoCo camera settings and initial poses to the pinned Viser viewer."""

from dataclasses import replace

from mjlab.viewer import ViserPlayViewer as MjlabViserPlayViewer


class ViserPlayViewer(MjlabViserPlayViewer):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    # MuJoCo looks down at negative elevation; mjviser uses positive elevation.
    # Keep a viewer-local copy so native rendering retains its camera settings.
    self.cfg = replace(self.cfg, elevation=-self.cfg.elevation)

  def setup(self):
    super().setup()
    # Paused ticks skip scene updates until an action occurs. Publish the real
    # initial transforms before the first client sees zero-positioned meshes.
    self._scene.paused = self._is_paused
    self._scene.update(self.env.unwrapped.sim.data)
