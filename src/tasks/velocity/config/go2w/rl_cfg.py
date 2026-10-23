"""Go2W PPO configs built directly from mjlab's configuration primitives."""

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


def unitree_go2w_ppo_runner_cfg(stage: str = "v2") -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.004 if stage == "stepup" else (0.008 if stage == "v2" else 0.01),
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=3e-4 if stage == "stepup" else (7.5e-4 if stage == "v2" else 1e-3),
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="go2w_flat" if stage == "flat" else "go2w_perceptive",
    run_name=stage,
    logger="tensorboard",
    upload_model=False,
    clip_actions=1.0,
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=4000 if stage == "stepup" else (30001 if stage == "v2" else 15001),
  )
