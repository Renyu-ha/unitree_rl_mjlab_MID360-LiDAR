"""Trace real RSL-RL sampling, critic augmentation, gradients and checkpoint replay."""

from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator
import torch
from rsl_rl.extensions import Symmetry
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg

import src.tasks  # noqa: F401
from src.tasks.velocity.rl.go2w_runner import Go2WOnPolicyRunner
from src.tasks.velocity.rl.go2w_symmetry import Go2WSymmetry
from tests.test_go2w_symmetry import make_symmetry_env


class Go2WSymmetryPipelineTests(unittest.TestCase):
  def test_native_extension_augments_actor_critic_and_paired_targets_for_every_task(self):
    for task in list_tasks():
      if not task.startswith("Unitree-Go2W-"):
        continue
      with self.subTest(task=task):
        env = make_symmetry_env(task)
        transform = Go2WSymmetry(env)
        n = 3
        obs = TensorDict({group: torch.arange(n * len(transform.transforms[group][0])).reshape(n, -1).float()
                          for group in ("actor", "critic")}, batch_size=[n])
        scalars = [torch.arange(n).reshape(n, 1).float() + marker for marker in (10, 20, 30, 40)]
        batch = RolloutStorage.Batch(observations=obs, actions=torch.arange(n * 16.).reshape(n, 16),
          values=scalars[0], advantages=scalars[1], returns=scalars[2], old_actions_log_prob=scalars[3],
          old_distribution_params=(torch.zeros(n, 16), torch.ones(n, 16)))
        original = deepcopy(batch)
        symmetry = Symmetry(env=env, **asdict(load_rl_cfg(task))["algorithm"]["symmetry_cfg"])
        symmetry.augment_batch(batch, n)
        for group in ("actor", "critic"):
          torch.testing.assert_close(batch.observations[group][:n], original.observations[group])
          torch.testing.assert_close(batch.observations[group][n:], transform.mirror(group, original.observations[group]))
        torch.testing.assert_close(batch.actions[n:], transform.mirror("actions", original.actions))
        for field in ("values", "advantages", "returns", "old_actions_log_prob"):
          torch.testing.assert_close(getattr(batch, field), getattr(original, field).repeat(2, 1))
        # KL uses only the original slice, so its old distribution stays N rows.
        self.assertEqual(batch.old_distribution_params[0].shape[0], n)

  def test_real_runner_updates_actor_and_critic_from_mirrored_rows_then_restores_and_exports(self):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    cfg = load_env_cfg("Unitree-Go2W-Perceptive-Rough-v2")
    cfg.seed, cfg.scene.num_envs = 23, 4
    env = None
    try:
      env = ManagerBasedRlEnv(cfg, device="cpu")
      wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
      plain_cfg = asdict(load_rl_cfg("Unitree-Go2W-Perceptive-Rough-v2"))
      # Keep the production 512/256/128 networks, 24 steps, 5 epochs, 4 batches.
      runner = Go2WOnPolicyRunner(wrapped, deepcopy(plain_cfg), log_dir=None, device="cpu")
      report, first_state, first_storage, first_updated = self._trace_learning(runner)
      self._compare_without_augmentation(wrapped, plain_cfg, first_state, first_storage, first_updated, report)
      self._check_checkpoint_and_export(runner, wrapped, plain_cfg, report)
      print("GO2W_SYMMETRY_PIPELINE_OK", json.dumps(report), flush=True)
    finally:
      if env is not None:
        env.close()
      torch.set_num_threads(previous_threads)

  def _trace_learning(self, runner):
    alg, env = runner.alg, runner.env.unwrapped
    storage = alg.storage
    report = {"transitions": 0, "timeouts": 0, "updates": 0, "batches": 0,
              "actor_mirror_gradient_l1": 0., "critic_mirror_gradient_l1": 0.,
              "actor_parameter_gradient_l1": 0., "critic_parameter_gradient_l1": 0.,
              "optimizer_steps": 0, "kl_calls": 0, "detached_mirror_loss_calls": 0,
              "mirror_log_ratio_mean_abs": [], "original_ratio_max_error": []}
    state = {"phase": "rollout", "expected": None, "pending": None, "snapshots": []}
    original_act, original_step = alg.act, alg.process_env_step
    original_update, original_augment = alg.update, alg.symmetry.augment_batch
    original_log_prob, original_optimizer_step = alg.actor.get_output_log_prob, alg.optimizer.step
    original_kl, original_mirror_loss = alg.actor.get_kl_divergence, alg.symmetry.compute_loss
    original_env_step = runner.env.step
    normalizer_counts = []

    def act(obs):
      value = original_act(obs)
      mean, std = alg.transition.distribution_params
      expected_log_prob = (-.5 * ((value - mean) / std).square() - std.log() - .5 * math.log(2 * math.pi)).sum(-1)
      torch.testing.assert_close(alg.transition.actions_log_prob, expected_log_prob, atol=2e-5, rtol=2e-5)
      state["pending"] = {"observations": obs.clone(), "actions": value.clone(),
                           "values": alg.transition.values.clone(),
                           "log_prob": alg.transition.actions_log_prob.clone()}
      return value

    def env_step(actions):
      if report["transitions"] in (2, 27):
        env.episode_length_buf[0] = env.max_episode_length - 1
      return original_env_step(actions)

    def process_step(obs, rewards, dones, extras):
      index, pending = storage.step, state["pending"]
      timeouts = extras["time_outs"]
      expected_reward = rewards + alg.gamma * pending["values"].squeeze(-1) * timeouts
      original_step(obs, rewards, dones, extras)
      for group in ("actor", "critic"):
        torch.testing.assert_close(storage.observations[group][index], pending["observations"][group])
      torch.testing.assert_close(storage.actions[index], pending["actions"])
      torch.testing.assert_close(storage.values[index], pending["values"])
      torch.testing.assert_close(storage.actions_log_prob[index, :, 0], pending["log_prob"])
      torch.testing.assert_close(storage.rewards[index, :, 0], expected_reward)
      torch.testing.assert_close(storage.dones[index, :, 0].long(), dones)
      report["transitions"] += 1
      report["timeouts"] += int(timeouts.sum())

    def augment(batch, n):
      flat = storage.actions.flatten(0, 1)
      difference = (batch.actions[:, None] - flat[None]).abs().amax(-1)
      error, indices = difference.min(-1)
      self.assertTrue((error == 0).all())
      for group in ("actor", "critic"):
        torch.testing.assert_close(batch.observations[group], storage.observations[group].flatten(0, 1)[indices])
      for field, source in (("values", storage.values), ("returns", storage.returns),
                            ("advantages", storage.advantages), ("old_actions_log_prob", storage.actions_log_prob)):
        torch.testing.assert_close(getattr(batch, field), source.flatten(0, 1)[indices])
      original = deepcopy(batch)
      original_augment(batch, n)
      self.assertEqual(batch.observations["actor"].shape, (2 * n, 720))
      self.assertEqual(batch.observations["critic"].shape, (2 * n, 263))
      transform = env._go2w_symmetry
      for group in ("actor", "critic"):
        torch.testing.assert_close(batch.observations[group][:n], original.observations[group])
        torch.testing.assert_close(batch.observations[group][n:], transform.mirror(group, original.observations[group]))
      for field in ("values", "returns", "advantages", "old_actions_log_prob"):
        torch.testing.assert_close(getattr(batch, field), getattr(original, field).repeat(2, 1))
      torch.testing.assert_close(batch.actions[n:], transform.mirror("actions", original.actions))
      state["expected"] = batch
      state["n"] = n
      state["first_batch"] = report["batches"] % 20 == 0
      report["batches"] += 1

    def log_prob(actions):
      result = original_log_prob(actions)
      if state["phase"] == "update":
        n, batch = state["n"], state["expected"]
        if state["first_batch"]:
          ratios = (result.detach() - batch.old_actions_log_prob.squeeze(-1)).exp()
          error = float((ratios[:n] - 1.).abs().max())
          self.assertLess(error, 2e-4)
          report["original_ratio_max_error"].append(error)
          report["mirror_log_ratio_mean_abs"].append(float(
            (result.detach()[n:] - batch.old_actions_log_prob[n:, 0]).abs().mean()))
        result.register_hook(lambda gradient: report.__setitem__("actor_mirror_gradient_l1",
          report["actor_mirror_gradient_l1"] + float(gradient[n:].abs().sum())))
      return result

    def critic_output(module, inputs, output):
      if state["phase"] == "update" and output.requires_grad:
        n = state["n"]
        output.register_hook(lambda gradient: report.__setitem__("critic_mirror_gradient_l1",
          report["critic_mirror_gradient_l1"] + float(gradient[n:].abs().sum())))

    def kl(old, new):
      self.assertEqual(old[0].shape[0], state["n"])
      self.assertEqual(new[0].shape[0], state["n"])
      report["kl_calls"] += 1
      return original_kl(old, new)

    def mirror_loss(*args, **kwargs):
      result = original_mirror_loss(*args, **kwargs)
      self.assertFalse(result.requires_grad)
      report["detached_mirror_loss_calls"] += 1
      return result

    def normalization(group):
      def check(module, inputs, output):
        if state["phase"] == "update":
          torch.testing.assert_close(inputs[0], state["expected"].observations[group])
          torch.testing.assert_close(output, (inputs[0] - module._mean) / (module._std + module.eps))
      return check

    def optimizer_step(*args, **kwargs):
      for label, model in (("actor", alg.actor), ("critic", alg.critic)):
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
        # PPO's clipped losses can legitimately have zero gradient on a batch.
        report[label + "_parameter_gradient_l1"] += sum(float(g.abs().sum()) for g in gradients)
      report["optimizer_steps"] += 1
      return original_optimizer_step(*args, **kwargs)

    def update():
      if report["updates"] == 0:
        state["initial_state"] = deepcopy(alg.save())
        state["initial_storage"] = {field: getattr(storage, field).clone() for field in
          ("observations", "actions", "rewards", "dones", "values", "returns", "advantages", "actions_log_prob")}
        state["initial_storage"]["distribution_params"] = tuple(p.clone() for p in storage.distribution_params)
      advantages = storage.returns - storage.values
      torch.testing.assert_close(storage.advantages, (advantages - advantages.mean()) / (advantages.std() + 1e-8))
      state["phase"] = "update"
      torch.manual_seed(101 + report["updates"])
      losses = original_update()
      self.assertTrue(all(np.isfinite(value) for value in losses.values()))
      state["phase"] = "rollout"
      if report["updates"] == 0:
        state["first_updated"] = {name: p.detach().clone() for name, p in alg.actor.named_parameters()}
        state["first_updated_critic"] = {name: p.detach().clone() for name, p in alg.critic.named_parameters()}
      report["updates"] += 1
      normalizer_counts.append((int(alg.actor.obs_normalizer.count), int(alg.critic.obs_normalizer.count)))
      return losses

    with ExitStack() as stack:
      for target, name, replacement in ((alg, "act", act), (runner.env, "step", env_step),
        (alg, "process_env_step", process_step), (alg.symmetry, "augment_batch", augment),
        (alg.actor, "get_output_log_prob", log_prob), (alg.actor, "get_kl_divergence", kl),
        (alg.symmetry, "compute_loss", mirror_loss), (alg.optimizer, "step", optimizer_step), (alg, "update", update)):
        stack.enter_context(patch.object(target, name, replacement))
      for module, hook in ((alg.actor.obs_normalizer, normalization("actor")),
                           (alg.critic.obs_normalizer, normalization("critic")), (alg.critic, critic_output)):
        handle = module.register_forward_hook(hook)
        stack.callback(handle.remove)
      runner.learn(num_learning_iterations=2, init_at_random_ep_len=True)
    self.assertEqual(report["transitions"], 48)
    self.assertGreaterEqual(report["timeouts"], 2)
    self.assertEqual(report["batches"], 40)
    self.assertEqual(report["optimizer_steps"], 40)
    self.assertEqual(report["kl_calls"], 40)
    self.assertEqual(report["detached_mirror_loss_calls"], 40)
    self.assertGreater(report["actor_mirror_gradient_l1"], 0.)
    self.assertGreater(report["critic_mirror_gradient_l1"], 0.)
    self.assertGreater(report["actor_parameter_gradient_l1"], 0.)
    self.assertGreater(report["critic_parameter_gradient_l1"], 0.)
    self.assertEqual(normalizer_counts, [(96, 96), (192, 192)])
    self.assertEqual(alg.actor.distribution.std_param.numel(), 16)
    report["normalizer_counts"] = normalizer_counts
    report["batch_shapes"] = {"actor": [48, 720], "critic": [48, 263], "actions": [48, 16]}
    return report, state["initial_state"], state["initial_storage"], (
      state["first_updated"], state["first_updated_critic"])

  def _compare_without_augmentation(self, env, config, saved, rollout, augmented_weights, report):
    control = Go2WOnPolicyRunner(env, deepcopy(config), log_dir=None, device="cpu")
    control.alg.load(saved, load_cfg=None, strict=True)
    for field, values in rollout.items():
      if field == "distribution_params":
        control.alg.storage.distribution_params = tuple(p.clone() for p in values)
      elif field == "observations":
        control.alg.storage.observations.copy_(values)
      else:
        getattr(control.alg.storage, field).copy_(values)
    control.alg.storage.step = 24
    control.alg.symmetry.use_data_augmentation = False
    torch.manual_seed(101)
    control.alg.update()
    for label, model, weights in zip(("actor", "critic"), (control.alg.actor, control.alg.critic), augmented_weights, strict=True):
      delta = max(float((p.detach() - weights[name]).abs().max()) for name, p in model.named_parameters())
      self.assertGreater(delta, 1e-7)
      report[label + "_augmentation_parameter_difference"] = delta

  def _check_checkpoint_and_export(self, runner, env, config, report):
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "model.pt"
      runner.save(str(path))
      restored = Go2WOnPolicyRunner(env, deepcopy(config), log_dir=None, device="cpu")
      infos = restored.load(str(path), map_location="cpu")
      self.assertEqual(infos["env_state"]["common_step_counter"], env.unwrapped.common_step_counter)
      # Older checkpoints may carry extra version records. Native loading should
      # accept these as ordinary infos without checking the current installation.
      checkpoint = torch.load(path, map_location="cpu", weights_only=False)
      checkpoint["infos"]["rsl_rl_backend"] = {"version": "5.0.1", "source_sha256": {}}
      legacy_path = Path(directory) / "legacy_model.pt"
      torch.save(checkpoint, legacy_path)
      legacy_infos = restored.load(str(legacy_path), map_location="cpu")
      self.assertEqual(legacy_infos["rsl_rl_backend"], checkpoint["infos"]["rsl_rl_backend"])
      torch.testing.assert_close(restored.alg.actor.state_dict(), runner.alg.actor.state_dict())
      torch.testing.assert_close(restored.alg.critic.state_dict(), runner.alg.critic.state_dict())
      torch.testing.assert_close(restored.alg.optimizer.state_dict(), runner.alg.optimizer.state_dict())
      self.assertTrue(restored.alg.symmetry.use_data_augmentation)
      # Inference must use only the actor; no augmentation is needed in play.
      restored.alg.symmetry.data_augmentation_func = Mock(side_effect=AssertionError("Augmentation during inference"))
      sample = env.get_observations()[:1]
      with torch.inference_mode():
        expected = restored.get_inference_policy()(sample).numpy()
      model = onnx.load(Path(directory) / "policy.onnx")
      onnx.checker.check_model(model)
      actual = ReferenceEvaluator(model).run(None, {model.graph.input[0].name: sample["actor"].numpy()})[0]
      np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
      report["onnx_max_error"] = float(np.max(np.abs(actual - expected)))


if __name__ == "__main__":
  unittest.main()
