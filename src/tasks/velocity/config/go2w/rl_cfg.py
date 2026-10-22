"""PPO configs for mixed Go2W actions; rough stages share one checkpoint root."""

from src.tasks.velocity.config.go2.rl_cfg import unitree_go2_ppo_runner_cfg


def unitree_go2w_ppo_runner_cfg(stage: str = "v2"):
  cfg = unitree_go2_ppo_runner_cfg()
  cfg.experiment_name = "go2w_flat" if stage == "flat" else "go2w_perceptive"
  cfg.run_name = stage
  cfg.logger = "tensorboard"
  cfg.upload_model = False
  cfg.clip_actions = 1.0
  cfg.actor.distribution_cfg["init_std"] = 0.5
  cfg.max_iterations = 30001 if stage == "v2" else 15001
  if stage == "stepup":
    cfg.max_iterations = 4000
    cfg.algorithm.learning_rate = 3e-4
    cfg.algorithm.entropy_coef = 0.004
  elif stage == "v2":
    cfg.algorithm.learning_rate = 7.5e-4
    cfg.algorithm.entropy_coef = 0.008
  return cfg
