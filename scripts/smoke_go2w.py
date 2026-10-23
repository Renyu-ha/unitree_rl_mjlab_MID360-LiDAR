"""Headless Go2W reset/step/PPO/export smoke test (CUDA by default).

Example: python scripts/smoke_go2w.py --task Unitree-Go2W-Perceptive-Rough-v2
"""

import argparse
from dataclasses import asdict
from pathlib import Path

import torch


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--task", default="Unitree-Go2W-Perceptive-Rough-v2")
  parser.add_argument("--num-envs", type=int, default=4)
  parser.add_argument("--steps", type=int, default=32)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--logger", choices=("tensorboard", "wandb"), help="Override the task logger for this smoke test only.")
  parser.add_argument("--train-iterations", type=int, default=1)
  parser.add_argument("--checkpoint", type=Path, help="Optional Go2W checkpoint to verify stage compatibility.")
  parser.add_argument("--fully-sparse", action="store_true", help="Exercise the final MID360 curriculum stage immediately.")
  args = parser.parse_args()
  if args.device.startswith("cuda") and not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable. Run the CPU checks with: python -m unittest discover -s tests -v")

  import src.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

  cfg = load_env_cfg(args.task)
  if args.fully_sparse:
    scan = cfg.observations["actor"].terms.get("height_scan")
    if scan is None or "fixed_curriculum_step" not in scan.params:
      parser.error("--fully-sparse requires a Go2W perceptive v2 task")
    scan.params["fixed_curriculum_step"] = 6000 * 24
  cfg.scene.num_envs = args.num_envs
  if cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.num_rows = 1
    cfg.scene.terrain.terrain_generator.num_cols = 2
  env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
  try:
    agent = load_rl_cfg(args.task)
    if args.logger is not None:
      agent.logger = args.logger
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    obs, _ = wrapped.reset()
    assert wrapped.num_actions == 16
    if agent.clip_actions is None:
      probe = torch.zeros(args.num_envs, 16, device=args.device)
      probe[:, 12:] = 2.0
      wrapped.step(probe)
      torch.testing.assert_close(env.sim.data.ctrl[:, -4:], torch.full((args.num_envs, 4), 40.0, device=args.device))
      assert (env.sim.data.actuator_force[:, -4:].abs() <= 45.001).all()
      obs, _ = wrapped.reset()
      print("Unclipped action probe passed: action=2 produces a 40 rad/s wheel target.")
    for _ in range(args.steps):
      obs, rewards, _, _ = wrapped.step(torch.zeros(args.num_envs, 16, device=args.device))
      assert torch.isfinite(rewards).all()
      assert all(torch.isfinite(value).all() for value in obs.values())
    print("Finite reset/step observations:", {name: tuple(value.shape) for name, value in obs.items()})
    if args.train_iterations:
      agent.algorithm.num_learning_epochs = 1
      agent.algorithm.num_mini_batches = 1
      directory = Path("logs") / "go2w_smoke" / args.task
      runner = load_runner_cls(args.task)(wrapped, asdict(agent), str(directory), args.device)
      if args.checkpoint is not None:
        runner.load(str(args.checkpoint), map_location=args.device)
      runner.learn(num_learning_iterations=args.train_iterations, init_at_random_ep_len=True)
      assert all(torch.isfinite(p).all() for p in runner.alg.get_policy().parameters())
      runner.save(str(directory / "smoke.pt"))
      import onnx
      model = onnx.load(directory / "policy.onnx")
      onnx.checker.check_model(model)
      metadata = {entry.key: entry.value for entry in model.metadata_props}
      assert len(metadata["action_joint_names"].split(",")) == 16
      assert metadata["action_types"].split(",") == ["position"] * 12 + ["velocity"] * 4
      assert metadata["action_clip"] == str(agent.clip_actions)
      # ONNX play requires a single environment because exports use batch=1.
      if args.num_envs == 1:
        from src.utils.onnx_policy import OnnxPolicy
        policy = OnnxPolicy(directory / "policy.onnx", wrapped, clip_actions=agent.clip_actions)
        action = policy(obs)
        assert torch.isfinite(action).all()
      print("PPO update and ONNX export passed:", directory.resolve())
  finally:
    env.close()


if __name__ == "__main__":
  main()
