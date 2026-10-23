"""Ensure display updates cannot take ownership of the training rollout."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from src.utils.training_viewer import LiveTrainingViewer


class TrainingViewerTests(unittest.TestCase):
  def make_viewer(self):
    # Exercise the step/update path independently of networking and mesh setup.
    viewer = LiveTrainingViewer.__new__(LiveTrainingViewer)
    data = SimpleNamespace(
      xpos=torch.arange(36).reshape(4, 3, 3), xmat=torch.zeros(4, 3, 3, 3),
      qpos=torch.zeros(4, 7), qvel=torch.zeros(4, 6),
    )
    env = Mock()
    env.unwrapped = env
    env.sim.data = data
    env.sim.mj_model.nmocap = 0
    viewer._wrapped_env = env
    viewer._server = Mock()
    viewer._server.get_clients.return_value = {1: object()}
    viewer._scene = Mock(debug_visualization_enabled=False)
    viewer._env_selector = SimpleNamespace(value=2)
    viewer._debug_checkbox = SimpleNamespace(value=False)
    viewer._selected_env = 2
    viewer._frame_interval = 0.1
    viewer._last_update = float("-inf")
    return viewer, env

  def test_display_keeps_rollout_results_and_only_selects_one_world(self):
    viewer, env = self.make_viewer()
    actions = torch.zeros(4, 16)
    expected = (object(), object(), object(), object(), object())
    env.step.return_value = expected
    before = env.sim.data.xpos.clone()
    self.assertIs(viewer.step(actions), expected)
    env.step.assert_called_once_with(actions)
    env.reset.assert_not_called()
    env.update_visualizers.assert_not_called()
    selected = viewer._scene.update.call_args.args[0]
    torch.testing.assert_close(selected.xpos, before[2:3])
    torch.testing.assert_close(env.sim.data.xpos, before)
    self.assertEqual(selected.qpos.shape[0], 1)

  def test_disconnected_or_rate_limited_display_never_skips_training_steps(self):
    viewer, env = self.make_viewer()
    viewer._last_update = 5.0
    with patch("src.utils.training_viewer.time.monotonic", return_value=5.05):
      viewer.step(None)
    viewer._server.get_clients.return_value = {}
    with patch("src.utils.training_viewer.time.monotonic", return_value=10.0):
      viewer.step(None)
    self.assertEqual(env.step.call_count, 2)
    viewer._scene.update.assert_not_called()

  def test_selected_debug_environment_and_cleanup_after_error(self):
    viewer, env = self.make_viewer()
    viewer._debug_checkbox.value = True
    viewer.update(force=True)
    self.assertEqual(viewer._scene.env_idx, 2)
    env.update_visualizers.assert_called_once_with(viewer._scene)
    env.close.side_effect = RuntimeError("environment cleanup failed")
    with self.assertRaisesRegex(RuntimeError, "environment cleanup failed"):
      viewer.close()
    viewer._server.stop.assert_called_once_with()


if __name__ == "__main__":
  unittest.main()
