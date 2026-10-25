"""Exercise Go2W noise routing with the pinned mjlab observation manager."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from mjlab.managers.observation_manager import ObservationManager
from mjlab.tasks.registry import list_tasks, load_env_cfg

import src.tasks  # noqa: F401 -- register task variants


WIDTHS = dict(base_ang_vel=3, projected_gravity=3, command=3, joint_pos=12,
              joint_vel=12, actions=16, height_scan=187, wheel_vel=4,
              base_lin_vel=3, foot_height=4, foot_contact=4, foot_contact_forces=12)


def marked_observation(env, name):
  return env.values[name]


def make_manager(play=False):
  groups = load_env_cfg("Unitree-Go2W-Perceptive-Rough-v2", play=play).observations
  values = {name: torch.arange(2 * width, dtype=torch.float32).reshape(2, width) / 10
            for name, width in WIDTHS.items()}
  env = SimpleNamespace(num_envs=2, device="cpu", scene={}, values=values)
  for group in groups.values():
    group.concatenate_terms = False  # Inspect each term without changing its processing.
    for name, term in group.terms.items():
      # Replace physical readout only; keep production noise, scale and history.
      term.func, term.params = marked_observation, {"name": name}
  scan = groups["actor"].terms["height_scan"]
  # Fix lag at one step to make the production noise/delay/history order observable.
  scan.delay_min_lag = scan.delay_max_lag = 1
  scan.delay_hold_prob = 0.
  scan.delay_update_period = 0
  return env, ObservationManager(groups, env)


class Go2WObservationNoiseTests(unittest.TestCase):
  def test_every_train_and_play_variant_declares_noise_only_for_actor_sensors(self):
    for task in list_tasks():
      if not task.startswith("Unitree-Go2W-"):
        continue
      for play in (False, True):
        with self.subTest(task=task, play=play):
          cfg = load_env_cfg(task, play=play)
          actor, critic = cfg.observations["actor"], cfg.observations["critic"]
          self.assertEqual(actor.enable_corruption, not play)
          self.assertFalse(critic.enable_corruption)
          self.assertTrue(all(term.noise is None for term in critic.terms.values()))
          self.assertIsNone(actor.terms["command"].noise)
          self.assertIsNone(actor.terms["actions"].noise)
          amplitudes = ((.3, .08, .015, 2., .15) if "v2" in task else
                        (.25, .06, .012, 1.8, .12) if "v1" in task else
                        (.2, .05, .01, 1.5, .1))
          for name, amplitude in zip(
            ("base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "wheel_vel"),
            (*amplitudes[:4], amplitudes[3]), strict=True,
          ):
            noise = actor.terms[name].noise
            self.assertEqual((noise.operation, noise.n_min, noise.n_max),
                             ("add", -amplitude, amplitude))
          if "height_scan" in actor.terms:
            noise = actor.terms["height_scan"].noise
            self.assertEqual((noise.n_min, noise.n_max), (-amplitudes[4], amplitudes[4]))
          self.assertEqual(actor.terms["wheel_vel"].scale, .05)
          self.assertEqual(critic.terms["wheel_vel"].scale, .1)
          self.assertEqual(critic.terms["foot_height"].scale, 5.)

  def test_native_manager_keeps_commands_actions_and_critic_exact(self):
    for play in (False, True):
      with self.subTest(play=play):
        env, manager = make_manager(play=play)
        before = {name: value.clone() for name, value in env.values.items()}
        # Use the actual UniformNoiseCfg.apply with a controlled uniform draw.
        with patch("mjlab.utils.noise.noise_cfg.torch.rand_like",
                   side_effect=lambda value: torch.full_like(value, .75)) as draw:
          observations = manager.compute(update_history=True)
        self.assertEqual(draw.call_count, 0 if play else 6)
        for group, terms in observations.items():
          for name, value in terms.items():
            term = manager.get_term_cfg(group, name)
            raw = before[name]
            if term.noise is not None:
              raw = raw + .5 * term.noise.n_max
            expected = raw * (term.scale if term.scale is not None else 1.)
            expected = expected[:, None].expand(-1, 3 if group == "actor" else 1, -1)
            torch.testing.assert_close(value, expected.flatten(1), msg=f"{play}/{group}/{name}")
        for name in before:
          torch.testing.assert_close(env.values[name], before[name])

  def test_native_scan_history_contains_delayed_already_noised_and_scaled_frames(self):
    env, manager = make_manager()
    snapshots = []
    for frame, draw_value in enumerate((.75, .25, .5)):
      env.values["height_scan"].fill_(float(frame + 1))
      with patch("mjlab.utils.noise.noise_cfg.torch.rand_like",
                 side_effect=lambda value: torch.full_like(value, draw_value)):
        snapshots.append(manager.compute(update_history=True)["actor"]["height_scan"])
    # The first frame backfills history. Lag one then repeats it before frame two arrives.
    frame1 = (1. + .075) * .2
    frame2 = (2. - .075) * .2
    for actual, expected in zip(snapshots, ((frame1,) * 3, (frame1,) * 3,
                                           (frame1, frame1, frame2)), strict=True):
      target = torch.tensor(expected).view(1, 3, 1).expand(2, 3, 187).flatten(1)
      torch.testing.assert_close(actual, target)


if __name__ == "__main__":
  unittest.main()
