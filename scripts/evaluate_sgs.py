"""Evaluate Go2W actors on one fresh, uniform SGS bank with finite episodes."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def evaluate(checkpoint, *, num_envs=128, episodes=512, device="cuda:0", seed=20261031,
             output=Path("outputs/sgs_evaluation.json")):
  import torch
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
  import src.tasks  # noqa: F401
  from src.tasks.velocity.rl.sgs_runner import Go2WSGSRunner

  checkpoint = Path(checkpoint).resolve()
  if not checkpoint.is_file():
    raise FileNotFoundError(checkpoint)
  if num_envs < 1 or episodes < 1:
    raise ValueError("num_envs and episodes must be positive")
  task = "Unitree-Go2W-Perceptive-Rough-v2-SGS"
  cfg = load_env_cfg(task, play=True)
  cfg.scene.num_envs, cfg.seed = num_envs, seed
  cfg.commands["twist"].sgs.bank_seed = seed
  cfg.scene.terrain.terrain_generator.seed = seed
  names = list(cfg.scene.terrain.terrain_generator.sub_terrains)
  env = ManagerBasedRlEnv(cfg, device=device)
  try:
    wrapped = RslRlVecEnvWrapper(env)
    agent_cfg = load_rl_cfg(task)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    # Preserve user-selected MLP widths rather than requiring the factory defaults.
    for name in ("actor", "critic"):
      for key, value in (saved.get("infos") or {}).get("sgs_model_cfg", {}).get(name, {}).items():
        setattr(getattr(agent_cfg, name), key, value)
      state = saved.get(f"{name}_state_dict", {})
      keys = sorted((k for k in state if k.startswith("mlp.") and k.endswith(".weight")),
                    key=lambda k: int(k.split(".")[1]))
      if keys:
        getattr(agent_cfg, name).hidden_dims = tuple(state[k].shape[0] for k in keys[:-1])
    runner = Go2WSGSRunner(wrapped, asdict(agent_cfg), device=device)
    runner.load(str(checkpoint), load_cfg={"actor": True}, map_location=device)
    policy = runner.get_inference_policy(device)
    command = env.command_manager.get_term("twist")
    records = []
    max_steps = env.max_episode_length * ((episodes + num_envs - 1) // num_envs) * 10
    with torch.inference_mode():
      obs = wrapped.get_observations()
      for _ in range(max_steps):
        obs, _, _, _ = wrapped.step(policy(obs))
        result = command.last_episodes
        if result is not None:
          valid = result["valid"]
          ids = result["config_ids"][valid].cpu().tolist()
          success = result["success"][valid].cpu().tolist()
          failed = result["terminated"][valid].cpu().tolist()
          rms = result["rms"][valid].cpu().tolist()
          for config_id, passed, fell, error in zip(ids, success, failed, rms):
            tile = config_id // command.cfg.sgs.profiles_per_tile
            records.append({"terrain": names[tile % command.cols], "level": tile // command.cols,
              "success": passed, "fell": fell, "linear_rms": error[0], "yaw_rms": error[1]})
            if len(records) == episodes:
              break
        if len(records) == episodes:
          break
    if len(records) < episodes:
      raise RuntimeError(f"Only {len(records)}/{episodes} valid episodes; check skipped resets/map boundaries")

    def summary(rows):
      if not rows:
        return {"episodes": 0, "success_rate": None, "fall_rate": None,
          "linear_rms_mean": None, "yaw_rms_mean": None}
      return {"episodes": len(rows), "success_rate": sum(x["success"] for x in rows) / len(rows),
        "fall_rate": sum(x["fell"] for x in rows) / len(rows),
        "linear_rms_mean": sum(x["linear_rms"] for x in rows) / len(rows),
        "yaw_rms_mean": sum(x["yaw_rms"] for x in rows) / len(rows)}

    report = {"checkpoint": str(checkpoint), "seed": seed, "sampler": "uniform",
      "success_settings": asdict(command.cfg.sgs), "overall": summary(records),
      "by_terrain": {name: summary([r for r in records if r["terrain"] == name]) for name in names},
      "records": records}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report["overall"], indent=2))
    print(f"Evaluation written to {output.resolve()}")
    return report
  finally:
    env.close()


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", required=True, type=Path)
  parser.add_argument("--num-envs", type=int, default=128)
  parser.add_argument("--episodes", type=int, default=512)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--seed", type=int, default=20261031)
  parser.add_argument("--output", type=Path, default=Path("outputs/sgs_evaluation.json"))
  evaluate(**vars(parser.parse_args()))


if __name__ == "__main__":
  main()
