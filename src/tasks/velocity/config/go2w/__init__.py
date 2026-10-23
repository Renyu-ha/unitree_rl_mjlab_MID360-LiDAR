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

register_mjlab_task(
  task_id="Unitree-Go2W-Flat",
  env_cfg=unitree_go2w_flat_env_cfg(), play_env_cfg=unitree_go2w_flat_env_cfg(play=True),
  rl_cfg=unitree_go2w_ppo_runner_cfg("flat"), runner_cls=Go2WOnPolicyRunner,
)
register_mjlab_task(
  task_id="Unitree-Go2W-Rough",
  env_cfg=unitree_go2w_rough_env_cfg(), play_env_cfg=unitree_go2w_rough_env_cfg(play=True),
  rl_cfg=unitree_go2w_ppo_runner_cfg("rough"), runner_cls=Go2WOnPolicyRunner,
)
register_mjlab_task(
  task_id="Unitree-Go2W-Perceptive-Rough-v1",
  env_cfg=unitree_go2w_perceptive_rough_v1_env_cfg(),
  play_env_cfg=unitree_go2w_perceptive_rough_v1_env_cfg(play=True),
  rl_cfg=unitree_go2w_ppo_runner_cfg("v1"), runner_cls=Go2WOnPolicyRunner,
)
register_mjlab_task(
  task_id="Unitree-Go2W-Perceptive-Rough-v2",
  env_cfg=unitree_go2w_perceptive_rough_v2_env_cfg(),
  play_env_cfg=unitree_go2w_perceptive_rough_v2_env_cfg(play=True),
  rl_cfg=unitree_go2w_ppo_runner_cfg("v2"), runner_cls=Go2WOnPolicyRunner,
)
register_mjlab_task(
  task_id="Unitree-Go2W-Perceptive-Rough-v2-StepUp-Finetune",
  env_cfg=unitree_go2w_perceptive_stepup_env_cfg(),
  play_env_cfg=unitree_go2w_perceptive_stepup_env_cfg(play=True),
  rl_cfg=unitree_go2w_ppo_runner_cfg("stepup"), runner_cls=Go2WOnPolicyRunner,
)
register_mjlab_task(
  task_id="Unitree-Go2W-Perceptive-Rough-v2-Step14-Up",
  env_cfg=unitree_go2w_perceptive_step14_env_cfg(),
  play_env_cfg=unitree_go2w_perceptive_step14_env_cfg(play=True),
  rl_cfg=unitree_go2w_ppo_runner_cfg("v2"), runner_cls=Go2WOnPolicyRunner,
)
