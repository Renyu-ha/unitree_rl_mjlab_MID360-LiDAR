"""Package a Go2W checkpoint/ONNX for the original G1 deployment framework."""

import argparse
from dataclasses import asdict
from pathlib import Path
import tempfile
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  source = parser.add_mutually_exclusive_group(required=True)
  source.add_argument("--checkpoint", type=Path)
  source.add_argument("--onnx", type=Path)
  parser.add_argument("--task", default="Unitree-Go2W-Perceptive-Rough-v2")
  parser.add_argument("--output", required=True, type=Path)
  parser.add_argument("--device", default="cpu")
  args = parser.parse_args()
  if args.output.exists() and any(args.output.iterdir()):
    parser.error("Output directory must be empty; keep hardware configuration in config/config.yaml")

  import onnx
  from src.utils.go2w_deploy import write_deploy_yaml

  with tempfile.TemporaryDirectory() as temporary:
    directory = Path(temporary)
    if args.checkpoint:
      import src.tasks  # noqa: F401
      from mjlab.envs import ManagerBasedRlEnv
      from mjlab.rl import RslRlVecEnvWrapper
      from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
      if not args.task.startswith("Unitree-Go2W-"):
        parser.error("Choose a registered Unitree-Go2W task")
      env_cfg = load_env_cfg(args.task, play=True)
      env_cfg.scene.num_envs = 1
      if env_cfg.scene.terrain.terrain_generator is not None:
        env_cfg.scene.terrain.terrain_generator.num_rows = 2
        env_cfg.scene.terrain.terrain_generator.num_cols = 2
      agent_cfg = load_rl_cfg(args.task)
      env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
      try:
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = load_runner_cls(args.task)(wrapped, asdict(agent_cfg), device=args.device)
        runner.load(str(args.checkpoint.resolve()), load_cfg={"actor": True}, strict=True, map_location=args.device)
        runner.export_deploy_policy(str(directory))
      finally:
        env.close()
      model = onnx.load(str(directory / "policy.onnx"))
    else:
      model = onnx.load(str(args.onnx.resolve()))
    # Validate before creating the requested output. Inline external tensor data
    # so the copied policy never depends on files outside its deployment bundle.
    onnx.external_data_helper.convert_model_from_external_data(model)
    onnx.save_model(model, directory / "policy.onnx")
    config = write_deploy_yaml(directory / "policy.onnx")
    exported = args.output / "exported"
    exported.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, exported / "policy.onnx")
    import shutil
    shutil.copytree(directory / "params", args.output / "params")
  print(f"Packaged {config['observation_dim']} -> 16 Go2W policy: {args.output.resolve()}")


if __name__ == "__main__":
  main()
