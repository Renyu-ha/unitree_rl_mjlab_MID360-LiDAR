"""Parse real training arguments without starting a simulator or requiring GPUs."""

import sys
import unittest
import warnings
from unittest.mock import patch

import scripts.train as train


class TrainCliTests(unittest.TestCase):
  def parse(self, task, *args):
    argv = ["scripts/train.py", task, *args]
    with patch.object(sys, "argv", argv), patch.object(train, "launch_training") as launch:
      with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        train.main()
      launch.assert_called_once()
      config = launch.call_args.kwargs["args"]
    for warning in caught:
      message = str(warning.message)
      self.assertFalse(
        "env.commands.twist.ranges" in message or "env.commands.twist.viz" in message,
        message,
      )
    return config

  def test_sgs_gpu_one_and_nested_command_overrides(self):
    cfg = self.parse(
      "Unitree-Go2W-Perceptive-Rough-v2-SGS",
      "--env.scene.num-envs=4096", "--gpu-ids", "[1]",
      "--env.commands.twist.ranges.lin-vel-x", "(-0.4, 0.9)",
      "--env.commands.twist.viz.scale", "0.75",
      "--env.commands.twist.sgs.velocity-subdivisions", "4",
      "--env.commands.twist.warm-start=True",
    )
    self.assertEqual(cfg.gpu_ids, [1])
    self.assertEqual(cfg.env.scene.num_envs, 4096)
    self.assertEqual(cfg.env.commands["twist"].ranges.lin_vel_x, (-0.4, 0.9))
    self.assertEqual(cfg.env.commands["twist"].viz.scale, 0.75)
    self.assertEqual(cfg.env.commands["twist"].sgs.velocity_subdivisions, 4)
    self.assertTrue(cfg.env.commands["twist"].warm_start)

  def test_non_sgs_multi_gpu_list(self):
    cfg = self.parse("Unitree-Go2W-Perceptive-Rough-v2", "--gpu-ids", "[0,1]")
    self.assertEqual(cfg.gpu_ids, [0, 1])

  def test_uniform_control_cpu(self):
    cfg = self.parse("Unitree-Go2W-Perceptive-Rough-v2-SGS-Uniform", "--gpu-ids=None")
    self.assertIsNone(cfg.gpu_ids)
    self.assertFalse(cfg.env.commands["twist"].sgs.adaptive)


if __name__ == "__main__":
  unittest.main()
