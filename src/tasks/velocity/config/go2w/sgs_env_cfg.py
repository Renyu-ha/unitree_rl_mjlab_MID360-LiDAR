"""Independent SGS and uniform-bank controls with unchanged Go2W V2 actor I/O."""

from dataclasses import replace

from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg

from src.tasks.velocity.config.go2w.env_cfgs import _make_go2w_env_cfg, _terrain_cfg
from src.tasks.velocity.config.go2w.rl_cfg import unitree_go2w_ppo_runner_cfg
from src.tasks.velocity.mdp.sgs import SGSConfig
from src.tasks.velocity.mdp.sgs_velocity_command import (
  SGSVelocityCommandCfg, sgs_reset_curriculum, sgs_tracking_step,
)


def unitree_go2w_sgs_env_cfg(play=False, *, adaptive=True):
  cfg = _make_go2w_env_cfg(play, stage="v2")
  old = cfg.commands["twist"]
  settings = SGSConfig(adaptive=adaptive and not play, bank_seed=20261031 if play else 20261030)
  # Velocity intervals cover the final V2 command range from the start.
  # Their definitions stay fixed while concrete commands are sampled online.
  cfg.commands["twist"] = SGSVelocityCommandCfg(
    entity_name=old.entity_name, resampling_time_range=old.resampling_time_range,
    debug_vis=old.debug_vis, heading_command=old.heading_command,
    heading_control_stiffness=old.heading_control_stiffness,
    rel_standing_envs=old.rel_standing_envs, rel_heading_envs=old.rel_heading_envs,
    rel_world_envs=old.rel_world_envs, rel_forward_envs=old.rel_forward_envs,
    init_velocity_prob=0.0,
    ranges=replace(old.ranges, lin_vel_x=(-0.6, 1.1), lin_vel_y=(-0.6, 0.6), ang_vel_z=(-0.8, 0.8)),
    viz=old.viz, sgs=settings, evaluation=play,
  )
  cfg.scene.terrain.terrain_generator = _terrain_cfg(False, "v2")
  cfg.scene.terrain.terrain_generator.seed = settings.bank_seed
  cfg.scene.terrain.max_init_terrain_level = None
  cfg.episode_length_s = 20.0
  cfg.curriculum.pop("terrain_levels", None)
  cfg.curriculum.pop("command_vel", None)
  cfg.curriculum["sgs_sampling"] = CurriculumTermCfg(func=sgs_reset_curriculum)
  cfg.events.pop("randomize_terrain", None)
  # Keep V2's reset_base: fresh XY/yaw randomization, independent of the SGS ID.
  cfg.events["sgs_tracking"] = EventTermCfg(func=sgs_tracking_step, mode="step")
  return cfg


def unitree_go2w_sgs_ppo_runner_cfg(*, adaptive=True):
  cfg = unitree_go2w_ppo_runner_cfg("v2")
  cfg.run_name = "v2_sgs" if adaptive else "v2_uniform_bank"
  return cfg
