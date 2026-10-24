"""Camera conventions and paused startup must preserve the simulation state."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from mjlab.viewer.viewer_config import ViewerConfig

from src.utils.play_viewer import MjlabViserPlayViewer, ViserPlayViewer


class PlayViewerTests(unittest.TestCase):
  def make_env(self, elevation=-25):
    env = SimpleNamespace(cfg=SimpleNamespace(viewer=ViewerConfig(elevation=elevation)),
                          sim=SimpleNamespace(data=SimpleNamespace(qpos=torch.tensor([[1., 2., 3.]]))),
                          step=Mock(), reset=Mock())
    env.unwrapped = env
    return env

  def test_camera_conversion_keeps_the_native_and_video_configuration(self):
    for elevation in (-25., 15.):
      env = self.make_env(elevation)
      viewer = ViserPlayViewer(env, Mock(), viser_server=Mock())
      self.assertEqual(viewer.cfg.elevation, -elevation)
      self.assertEqual(env.cfg.viewer.elevation, elevation)
      self.assertIsNot(viewer.cfg, env.cfg.viewer)
      self.assertEqual(viewer.cfg.distance, env.cfg.viewer.distance)

  def test_paused_setup_publishes_initial_poses_without_advancing_or_resetting(self):
    env = self.make_env()
    qpos = env.sim.data.qpos.clone()
    viewer = ViserPlayViewer(env, Mock(), viser_server=Mock())
    viewer.pause()
    scene = Mock()

    def setup(_viewer):
      _viewer._scene = scene

    with patch.object(MjlabViserPlayViewer, "setup", autospec=True, side_effect=setup):
      viewer.setup()
    scene.update.assert_called_once_with(env.sim.data)
    self.assertTrue(scene.paused)
    env.step.assert_not_called()
    env.reset.assert_not_called()
    torch.testing.assert_close(env.sim.data.qpos, qpos, rtol=0, atol=0)


if __name__ == "__main__":
  unittest.main()
