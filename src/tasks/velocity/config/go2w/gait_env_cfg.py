"""Isolated V2 comparison: restore G1 rewards, excluding ordinary foot slip.

The original Go2W builder, task factories and PPO settings remain the baseline.
Only this variant removes the custom air-time/contact-bonus rewards. Restored
reward weights, height targets and curricula start from this repository's G1 V2.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg

from src.assets.robots.unitree_go2w.go2w_constants import WHEEL_SITES
from src.tasks.velocity import mdp
from src.tasks.velocity.config.go2w.env_cfgs import _make_go2w_env_cfg
from src.tasks.velocity.config.go2w.rl_cfg import unitree_go2w_ppo_runner_cfg


def unitree_go2w_perceptive_gait_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Add a phase-conditioned diagonal trot to a fresh Go2W V2 environment."""
  cfg = _make_go2w_env_cfg(play, stage="v2")

  # ------------------------------ Observations ------------------------------
  # Append a noiseless sin/cos clock to both networks; Actor keeps three frames.
  cfg.observations["actor"].terms["phase"] = ObservationTermCfg(
    func=mdp.phase, params={"period": 0.6, "command_name": "twist"},
  )
  cfg.observations["critic"].terms["phase"] = ObservationTermCfg(
    func=mdp.phase, params={"period": 0.6, "command_name": "twist"},
  )

  # ------------------------------ Rewards ------------------------------
  cfg.rewards.pop("wheel_air_time_timeout", None)
  cfg.rewards.pop("wheel_air_time_memory")
  cfg.rewards.pop("wheel_contact_bonus")
  cfg.rewards["angular_momentum"] = RewardTermCfg(
    func=mdp.angular_momentum_penalty, weight=-0.035,
    params={"sensor_name": "robot/root_angmom"},
  )
  cfg.rewards["foot_gait"] = RewardTermCfg(
    func=mdp.feet_gait, weight=0.5,
    params={
      "period": 0.6, "offset": [0.0, 0.5, 0.5, 0.0],  # FL, FR, RL, RR.
      "threshold": 0.56, "command_threshold": 0.1,
      "command_name": "twist", "sensor_name": "feet_ground_contact",
    },
  )
  cfg.rewards["foot_clearance"] = RewardTermCfg(
    func=mdp.feet_clearance, weight=-1.8,
    params={
      "target_height": 0.22, "relative_to_lowest_foot": True,
      "command_name": "twist", "command_threshold": 0.1,
      "asset_cfg": SceneEntityCfg("robot", site_names=WHEEL_SITES),
    },
  )
  cfg.rewards["foot_swing_height"] = RewardTermCfg(
    func=mdp.feet_swing_height, weight=-1.0,
    params={
      "sensor_name": "feet_ground_contact", "target_height": 0.24,
      "relative_to_lowest_foot": True,
      "command_name": "twist", "command_threshold": 0.1,
      "asset_cfg": SceneEntityCfg("robot", site_names=WHEEL_SITES, preserve_order=True),
    },
  )

  # ------------------------------ Curriculum ------------------------------
  if not play:
    cfg.curriculum["foot_clearance_weight"] = CurriculumTermCfg(
      func=mdp.reward_weight,
      params={
        "reward_name": "foot_clearance",
        "weight_stages": [
          {"step": 0, "weight": -1.8},
          {"step": 8000 * 24, "weight": -1.4},
          {"step": 14000 * 24, "weight": -1.1},
        ],
      },
    )
    cfg.curriculum["foot_swing_height_weight"] = CurriculumTermCfg(
      func=mdp.reward_weight,
      params={
        "reward_name": "foot_swing_height",
        "weight_stages": [
          {"step": 0, "weight": -1.0},
          {"step": 8000 * 24, "weight": -0.75},
          {"step": 14000 * 24, "weight": -0.5},
        ],
      },
    )
  return cfg


def unitree_go2w_perceptive_gait_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Keep V2 PPO and symmetry; separate comparison runs by their run name."""
  cfg = unitree_go2w_ppo_runner_cfg("v2")
  cfg.run_name = "v2_gait"
  return cfg
