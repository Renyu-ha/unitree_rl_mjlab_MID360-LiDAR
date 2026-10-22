"""Register Go2W wheel-legged velocity and perceptive tasks."""

from mjlab.tasks.registry import register_mjlab_task

from src.tasks.velocity.config.go2w.env_cfgs import (
  unitree_go2w_flat_env_cfg,
  unitree_go2w_perceptive_rough_v1_env_cfg,
  unitree_go2w_perceptive_rough_v2_env_cfg,
  unitree_go2w_perceptive_step14_env_cfg,
  unitree_go2w_perceptive_stepup_env_cfg,
  unitree_go2w_rough_env_cfg,
)
from src.tasks.velocity.config.go2w.rl_cfg import unitree_go2w_ppo_runner_cfg
from src.tasks.velocity.rl.go2w_runner import Go2WOnPolicyRunner

for task_id, factory, stage in (
  ("Unitree-Go2W-Flat", unitree_go2w_flat_env_cfg, "flat"),
  ("Unitree-Go2W-Rough", unitree_go2w_rough_env_cfg, "rough"),
  ("Unitree-Go2W-Perceptive-Rough-v1", unitree_go2w_perceptive_rough_v1_env_cfg, "v1"),
  ("Unitree-Go2W-Perceptive-Rough-v2", unitree_go2w_perceptive_rough_v2_env_cfg, "v2"),
  ("Unitree-Go2W-Perceptive-Rough-v2-StepUp-Finetune", unitree_go2w_perceptive_stepup_env_cfg, "stepup"),
  ("Unitree-Go2W-Perceptive-Rough-v2-Step14-Up", unitree_go2w_perceptive_step14_env_cfg, "v2"),
):
  register_mjlab_task(
    task_id=task_id, env_cfg=factory(), play_env_cfg=factory(play=True),
    rl_cfg=unitree_go2w_ppo_runner_cfg(stage), runner_cls=Go2WOnPolicyRunner,
  )
