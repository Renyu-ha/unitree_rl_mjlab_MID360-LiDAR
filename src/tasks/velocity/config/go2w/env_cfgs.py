"""Go2W velocity tasks built from the common velocity environment root.

All registered variants start at the same private Go2W builder instead of
calling another task factory. Rough/perceptive variants share observation and
action definitions. Observation history follows the corresponding G1 stage.
"""

from copy import deepcopy
from dataclasses import replace
from typing import Literal

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg, JointVelocityActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, GridPatternCfg, ObjRef, RayCastSensorCfg
from mjlab.terrains.config import ROUGH_TERRAINS_CFG
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from src.assets.robots.unitree_go2w.go2w_constants import (
  LEG_ACTION_SCALE, LEG_JOINTS, WHEEL_ACTION_SCALE, WHEEL_BODIES,
  WHEEL_GEOMS, WHEEL_JOINTS, WHEEL_RADIUS, WHEEL_SITES, get_go2w_robot_cfg,
)
from src.tasks.velocity import mdp
from src.tasks.velocity.mdp.wheel_rewards import wheel_spin_contact, wheel_stand_still
from src.tasks.velocity.mdp.spawn_clearance import ensure_spawn_clearance
from src.tasks.velocity.velocity_env_cfg import OffsetGridPatternCfg, make_velocity_env_cfg

SCAN = {
  "size_x_m": 1.6, "size_y_m": 1.0, "resolution_m": 0.1,
  "x_offset_m": 0.35, "y_offset_m": 0.0,
}


def _terrain_cfg(play: bool, stage: str):
  terrain = deepcopy(ROUGH_TERRAINS_CFG)
  terrain.curriculum = not play
  if stage in ("flat", "rough"):
    if play:
      terrain.num_rows = terrain.num_cols = 5
      terrain.border_width = 10.0
    return terrain
  v1 = stage == "v1"
  stepup = stage == "stepup"
  terrain.num_rows = 5 if play else 10
  terrain.num_cols = 5 if play else 6
  terrain.border_width = 10.0 if play else 20.0
  terrain.sub_terrains["flat"] = replace(
    terrain.sub_terrains["flat"], proportion=0.05 if stepup else (0.15 if v1 else 0.08),
  )
  terrain.sub_terrains["pyramid_stairs"] = replace(
    terrain.sub_terrains["pyramid_stairs"], proportion=0.05 if stepup else (0.30 if v1 else 0.14),
    step_height_range=(0.14, 0.14) if play else ((0.04, 0.16) if v1 else (0.04, 0.14)),
    step_width=0.45 if v1 else 0.40,
  )
  terrain.sub_terrains["pyramid_stairs_inv"] = replace(
    terrain.sub_terrains["pyramid_stairs_inv"],
    proportion=0.60 if stepup else (0.20 if v1 else 0.30),
    step_height_range=(0.14, 0.14) if play else ((0.03, 0.14) if v1 else (0.04, 0.18)),
    step_width=0.45 if v1 else 0.40,
  )
  terrain.sub_terrains["hf_pyramid_slope"] = replace(
    terrain.sub_terrains["hf_pyramid_slope"],
    proportion=0.04 if stepup else (0.10 if v1 else 0.08),
    slope_range=(0.0, 0.35) if v1 else (0.0, 0.40),
    platform_width=2.5 if v1 else 2.0, border_width=0.5,
  )
  terrain.sub_terrains["hf_pyramid_slope_inv"] = replace(
    terrain.sub_terrains["hf_pyramid_slope_inv"],
    proportion=0.06 if stepup else (0.05 if v1 else 0.18),
    slope_range=(0.0, 0.30) if v1 else (0.0, 0.35),
    platform_width=2.5 if v1 else 2.0, border_width=0.5,
  )
  terrain.sub_terrains["random_rough"] = replace(
    terrain.sub_terrains["random_rough"], proportion=0.10 if stepup else (0.10 if v1 else 0.12),
    noise_range=(0.005, 0.05) if v1 else (0.005, 0.06), noise_step=0.01, border_width=0.5,
  )
  terrain.sub_terrains["wave_terrain"] = replace(
    terrain.sub_terrains["wave_terrain"], proportion=0.10,
    amplitude_range=(0.0, 0.08) if v1 else (0.0, 0.10), num_waves=3, border_width=0.5,
  )
  return terrain


def _make_go2w_env_cfg(
  play: bool,
  *,
  stage: Literal["flat", "rough", "v1", "v2", "stepup", "step14"],
) -> ManagerBasedRlEnvCfg:
  """Build each variant from the shared root and explicit Go2W settings."""
  cfg = make_velocity_env_cfg()
  perceptive = stage not in ("flat", "rough")
  v2 = perceptive and stage != "v1"
  stepup = stage == "stepup"
  cfg.scene.entities = {"robot": get_go2w_robot_cfg()}
  cfg.sim.nconmax = 256 if perceptive else 128
  cfg.sim.njmax = 4096 if perceptive else cfg.sim.njmax
  cfg.sim.mujoco.ccd_iterations = 50 if perceptive else 500
  cfg.sim.contact_sensor_maxmatch = 128 if perceptive else 500
  cfg.scene.terrain.terrain_generator = _terrain_cfg(play, stage)
  cfg.scene.terrain.max_init_terrain_level = (1 if play and v2 else 2) if perceptive else 5

  terrain_scan = RayCastSensorCfg(
    name="terrain_scan",
    frame=ObjRef(type="body", name="base_link", entity="robot"),
    ray_alignment="yaw", include_geom_groups=(0, 1),
    pattern=GridPatternCfg(size=(1.6, 1.0), resolution=0.1),
    max_distance=5.0, exclude_parent_body=True, debug_vis=True,
    viz=RayCastSensorCfg.VizCfg(show_normals=True),
  )
  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="geom", pattern=WHEEL_GEOMS, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"), reduce="netforce", num_slots=1, track_air_time=True,
  )
  nonfoot_ground_cfg = ContactSensorCfg(
    name="nonfoot_ground_touch",
    primary=ContactMatch(
      mode="geom", entity="robot", pattern=r".*_collision\d*$", exclude=WHEEL_GEOMS,
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"), reduce="none", num_slots=1, history_length=4,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
    fields=("found", "force"), reduce="none", num_slots=1, history_length=4,
  )
  cfg.scene.sensors = (terrain_scan, feet_ground_cfg, nonfoot_ground_cfg, self_collision_cfg)
  if v2:
    mid360 = RayCastSensorCfg(
      name="terrain_scan_mid360",
      frame=ObjRef(type="body", name="base_link", entity="robot"),
      ray_alignment="yaw", include_geom_groups=(0, 1),
      pattern=OffsetGridPatternCfg(
        size=(SCAN["size_x_m"], SCAN["size_y_m"]), resolution=SCAN["resolution_m"],
        x_offset=SCAN["x_offset_m"], y_offset=SCAN["y_offset_m"],
      ),
      max_distance=5.0, exclude_parent_body=True, debug_vis=True,
      viz=RayCastSensorCfg.VizCfg(show_normals=True),
    )
    cfg.scene.sensors = (terrain_scan, feet_ground_cfg, nonfoot_ground_cfg, self_collision_cfg, mid360)

  ang_noise = 0.3 if v2 else (0.25 if perceptive else 0.2)
  gravity_noise = 0.08 if v2 else (0.06 if perceptive else 0.05)
  pos_noise = 0.015 if v2 else (0.012 if perceptive else 0.01)
  vel_noise = 2.0 if v2 else (1.8 if perceptive else 1.5)

  cfg.actions = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot", actuator_names=LEG_JOINTS,
      scale=LEG_ACTION_SCALE, use_default_offset=True,
    ),
    "wheel_vel": JointVelocityActionCfg(
      entity_name="robot", actuator_names=WHEEL_JOINTS,
      scale=WHEEL_ACTION_SCALE, use_default_offset=False,
    ),
  }
  cfg.observations = {
    "actor": ObservationGroupCfg(
      history_length=3 if v2 else 1, concatenate_terms=True, enable_corruption=True,
      terms={
        "base_ang_vel": ObservationTermCfg(
          func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_ang_vel"},
          noise=Unoise(n_min=-ang_noise, n_max=ang_noise),
        ),
        "projected_gravity": ObservationTermCfg(
          func=mdp.projected_gravity, noise=Unoise(n_min=-gravity_noise, n_max=gravity_noise),
        ),
        "command": ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "twist"}),
        "phase": ObservationTermCfg(func=mdp.phase, params={"period": 0.6, "command_name": "twist"}),
        "joint_pos": ObservationTermCfg(
          func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS)},
          noise=Unoise(n_min=-pos_noise, n_max=pos_noise),
        ),
        "joint_vel": ObservationTermCfg(
          func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS)},
          noise=Unoise(n_min=-vel_noise, n_max=vel_noise),
        ),
        "actions": ObservationTermCfg(func=mdp.last_action, noise=Unoise(n_min=-0.03, n_max=0.03) if v2 else None),
        "height_scan": ObservationTermCfg(
          func=envs_mdp.height_scan, params={"sensor_name": "terrain_scan"},
          noise=Unoise(n_min=-0.1, n_max=0.1), scale=0.2,
        ),
        "wheel_vel": ObservationTermCfg(
          func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINTS)},
          noise=Unoise(n_min=-vel_noise, n_max=vel_noise), scale=0.05,
        ),
      },
    ),
    "critic": ObservationGroupCfg(
      history_length=1, concatenate_terms=True, enable_corruption=False,
      terms={
        "base_ang_vel": ObservationTermCfg(
          func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_ang_vel"},
          noise=Unoise(n_min=-ang_noise, n_max=ang_noise),
        ),
        "projected_gravity": ObservationTermCfg(
          func=mdp.projected_gravity, noise=Unoise(n_min=-gravity_noise, n_max=gravity_noise),
        ),
        "command": ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "twist"}),
        "phase": ObservationTermCfg(func=mdp.phase, params={"period": 0.6, "command_name": "twist"}),
        "joint_pos": ObservationTermCfg(
          func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS)},
          noise=Unoise(n_min=-pos_noise, n_max=pos_noise),
        ),
        "joint_vel": ObservationTermCfg(
          func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS)},
          noise=Unoise(n_min=-vel_noise, n_max=vel_noise),
        ),
        "actions": ObservationTermCfg(func=mdp.last_action, noise=Unoise(n_min=-0.03, n_max=0.03) if v2 else None),
        "height_scan": ObservationTermCfg(
          func=envs_mdp.height_scan, params={"sensor_name": "terrain_scan"}, scale=0.2,
        ),
        "base_lin_vel": ObservationTermCfg(
          func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_lin_vel"},
          noise=Unoise(n_min=-0.5, n_max=0.5),
        ),
        "foot_height": ObservationTermCfg(
          func=mdp.foot_height, params={"asset_cfg": SceneEntityCfg("robot", site_names=WHEEL_SITES)},
        ),
        "foot_air_time": ObservationTermCfg(func=mdp.foot_air_time, params={"sensor_name": "feet_ground_contact"}),
        "foot_contact": ObservationTermCfg(func=mdp.foot_contact, params={"sensor_name": "feet_ground_contact"}),
        "foot_contact_forces": ObservationTermCfg(func=mdp.foot_contact_forces, params={"sensor_name": "feet_ground_contact"}),
        "wheel_vel": ObservationTermCfg(
          func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINTS)},
          noise=Unoise(n_min=-vel_noise, n_max=vel_noise), scale=0.05,
        ),
      },
    ),
  }
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = WHEEL_GEOMS
  cfg.events["encoder_bias"].params["asset_cfg"].joint_names = LEG_JOINTS
  cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)
  cfg.events["base_com"].params["ranges"] = {
    0: (-0.075, 0.075), 1: (-0.05, 0.05), 2: (-0.04, 0.04),
  } if v2 else {0: (-0.05, 0.05), 1: (-0.05, 0.05), 2: (-0.05, 0.05)}
  if v2:
    cfg.events["foot_friction"].params["ranges"] = (0.25, 1.8)
    cfg.events["encoder_bias"].params["bias_range"] = (-0.02, 0.02)

  cfg.rewards["pose"].params["asset_cfg"] = SceneEntityCfg("robot", joint_names=LEG_JOINTS)
  cfg.rewards["stand_still"].params["asset_cfg"] = SceneEntityCfg("robot", joint_names=LEG_JOINTS)
  cfg.rewards["joint_pos_limits"].params["asset_cfg"] = SceneEntityCfg("robot", joint_names=LEG_JOINTS)
  cfg.rewards["joint_acc_l2"].params["asset_cfg"] = SceneEntityCfg("robot", joint_names=LEG_JOINTS)
  cfg.rewards["pose"].params["std_standing"] = {
    r".*(FR|FL|RR|RL)_hip_joint.*": 0.05,
    r".*(FR|FL|RR|RL)_thigh_joint.*": 0.1,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.15,
  }
  cfg.rewards["pose"].params["std_walking"] = {
    r".*(FR|FL|RR|RL)_hip_joint.*": 0.15,
    r".*(FR|FL|RR|RL)_thigh_joint.*": 0.35,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.5,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*(FR|FL|RR|RL)_hip_joint.*": 0.15,
    r".*(FR|FL|RR|RL)_thigh_joint.*": 0.35,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.5,
  }
  cfg.rewards["body_orientation_l2"].params["asset_cfg"].body_names = ("base_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("base_link",)
  # Gait phase and mandatory foot clearance conflict with continuous rolling.
  cfg.rewards.pop("foot_gait")
  cfg.rewards.pop("foot_clearance")
  cfg.rewards.pop("foot_slip")
  cfg.rewards.pop("angular_momentum")
  cfg.rewards["wheel_spin_contact"] = RewardTermCfg(
    func=wheel_spin_contact, weight=-0.15,
    params={
      "sensor_name": "feet_ground_contact", "wheel_radius": WHEEL_RADIUS,
      "contact_threshold": 1.0,
      "joint_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINTS),
      "body_cfg": SceneEntityCfg("robot", body_names=WHEEL_BODIES),
    },
  )
  cfg.rewards["wheel_stand_still"] = RewardTermCfg(
    func=wheel_stand_still, weight=-0.02,
    params={"command_name": "twist", "asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINTS)},
  )
  cfg.rewards["wheel_speed"] = RewardTermCfg(
    func=mdp.joint_vel_l2, weight=-1e-4,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINTS)},
  )
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost, weight=-2.5 if v2 else (-2.0 if perceptive else -1.0),
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )
  cfg.rewards["track_linear_velocity"].weight = 1.5 if perceptive else 1.0
  cfg.rewards["track_angular_velocity"].weight = 0.7 if v2 else (0.75 if perceptive else 1.0)
  # User-selected reward experiments override the corresponding G1 values.
  cfg.rewards["track_angular_velocity"].params["std"] = 0.5
  cfg.rewards["pose"].weight = 0.5
  cfg.rewards["body_orientation_l2"].weight = -0.7
  cfg.rewards["body_ang_vel"].weight = -0.10 if v2 else (-0.08 if perceptive else -0.05)
  cfg.rewards["action_rate_l2"].weight = -0.06 if v2 else (-0.04 if perceptive else -0.05)
  cfg.rewards["soft_landing"].weight = -0.004 if v2 else (-0.003 if perceptive else -0.001)
  cfg.rewards["stand_still"].weight = -0.5 if perceptive else -1.0
  cfg.terminations["terrain_out_of_bounds"] = TerminationTermCfg(
    func=mdp.terrain_out_of_bounds, time_out=True,
    params={"asset_cfg": SceneEntityCfg("robot"), "distance_buffer": 3.0},
  )

  twist = cfg.commands["twist"]
  twist.rel_standing_envs = 0.03 if v2 else 0.05
  cfg.viewer.body_name = "base_link"
  cfg.viewer.distance = 2.0
  cfg.viewer.elevation = -10.0

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain, mode="reset", params={},
    )
    if perceptive:
      twist.ranges.lin_vel_x = (0.20, 0.55) if v2 else (0.0, 0.6)
      twist.ranges.lin_vel_y = (-0.10, 0.10) if v2 else (-0.2, 0.2)
      twist.ranges.ang_vel_z = (-0.10, 0.10) if v2 else (-0.35, 0.35)

  if perceptive:
    _configure_perception(cfg, play=play, version=1 if stage == "v1" else 2)
  if stage == "v1":
    _configure_perceptive_v1_curriculum(cfg)
  elif v2:
    _configure_perceptive_v2_curriculum(cfg, play=play)
  if stepup:
    _configure_stepup(cfg, play=play)
  if stage == "step14":
    _configure_step14(cfg)
  if stage == "flat":
    cfg.sim.nconmax = None
    cfg.sim.njmax = 300
    cfg.sim.mujoco.ccd_iterations = 50
    cfg.sim.contact_sensor_maxmatch = 64
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    cfg.scene.sensors = (feet_ground_cfg, nonfoot_ground_cfg, self_collision_cfg)
    cfg.observations["actor"].terms.pop("height_scan")
    cfg.observations["critic"].terms.pop("height_scan")
    cfg.curriculum.pop("terrain_levels", None)
    cfg.events.pop("randomize_terrain", None)
    if play:
      twist.ranges.lin_vel_x = (-0.5, 1.0)
      twist.ranges.lin_vel_y = (-0.5, 0.5)
      twist.ranges.ang_vel_z = (-0.5, 0.5)
  # Run after every other reset event, including play-mode terrain relocation.
  if "randomize_terrain" in cfg.events:
    relocation = cfg.events.pop("randomize_terrain")
    cfg.events = {"randomize_terrain": relocation, **cfg.events}
  cfg.events["ensure_spawn_clearance"] = EventTermCfg(
    func=ensure_spawn_clearance, mode="reset", params={"clearance": 0.005},
  )
  return cfg


def _configure_perception(cfg: ManagerBasedRlEnvCfg, *, play: bool, version: int) -> None:
  """Apply a sensing stage to a fresh Go2W config, without calling a task."""
  scan = cfg.observations["actor"].terms["height_scan"]
  scan.noise = Unoise(n_min=-0.12, n_max=0.12)
  scan.delay_max_lag = 1
  scan.delay_hold_prob = 0.35
  scan.delay_update_period = 2
  if version == 1:
    return
  scan.func = mdp.bridged_mid360_height_scan_memory()
  scan.params = {
    "sensor_name": "terrain_scan", "mid360_sensor_name": "terrain_scan_mid360", **SCAN,
    "visible_x_min_m": 0.25, "visible_x_max_m": 1.15, "visible_y_abs_m": 0.50,
    "keep_prob_min": 1.0 if play else 0.65, "keep_prob_max": 1.0,
    "frame_dropout_prob": 0.0 if play else 0.02,
    "switch_start_step": 500 * 24, "switch_end_step": 2500 * 24,
    "sparse_start_step": 2500 * 24, "sparse_end_step": 6000 * 24,
    "fixed_curriculum_step": 6000 * 24 if play else None,
    "memory_max_age_s": 0.8, "enable_memory_shift": True, "miss_value": 5.0,
  }
  scan.noise = Unoise(n_min=-0.15, n_max=0.15)
  scan.delay_max_lag = 0 if play else 2
  scan.delay_hold_prob = 0.0 if play else 0.35
  scan.delay_update_period = 0 if play else 3


def _configure_perceptive_v1_curriculum(cfg: ManagerBasedRlEnvCfg) -> None:
  """Keep the V1 command schedule separate from the V2 configuration."""
  if "command_vel" in cfg.curriculum:
    cfg.curriculum["command_vel"].params["velocity_stages"] = [
      {
        "step": 0,
        "lin_vel_x": (0.0, 0.5),
        "lin_vel_y": (-0.15, 0.15),
        "ang_vel_z": (-0.25, 0.25),
      },
      {
        "step": 5000 * 24,
        "lin_vel_x": (0.0, 0.8),
        "lin_vel_y": (-0.2, 0.2),
        "ang_vel_z": (-0.35, 0.35),
      },
      {
        "step": 10000 * 24,
        "lin_vel_x": (-0.1, 1.0),
        "lin_vel_y": (-0.3, 0.3),
        "ang_vel_z": (-0.5, 0.5),
      },
      {
        "step": 20000 * 24,
        "lin_vel_x": (-0.2, 1.2),
        "lin_vel_y": (-0.4, 0.4),
        "ang_vel_z": (-0.7, 0.7),
      },
    ]


def _configure_perceptive_v2_curriculum(cfg: ManagerBasedRlEnvCfg, *, play: bool) -> None:
  """Match this repository's unitree_g1_perceptive_rough_v2_env_cfg.

  terrain_levels is inherited from make_velocity_env_cfg, as it is for G1.
  Foot-clearance/swing curricula are excluded for rolling; pose_weight is
  excluded because the user's pose reward stays fixed at 0.5.
  """
  if "command_vel" in cfg.curriculum:
    cfg.curriculum["command_vel"].params["velocity_stages"] = [
      {
        "step": 0,
        "lin_vel_x": (0.15, 0.45),
        "lin_vel_y": (-0.05, 0.05),
        "ang_vel_z": (-0.10, 0.10),
      },
      {
        "step": 4000 * 24,
        "lin_vel_x": (0.15, 0.65),
        "lin_vel_y": (-0.08, 0.08),
        "ang_vel_z": (-0.18, 0.18),
      },
      {
        "step": 9000 * 24,
        "lin_vel_x": (0.10, 0.90),
        "lin_vel_y": (-0.12, 0.12),
        "ang_vel_z": (-0.30, 0.30),
      },
      {
        "step": 16000 * 24,
        "lin_vel_x": (0.05, 1.10),
        "lin_vel_y": (-0.18, 0.18),
        "ang_vel_z": (-0.45, 0.45),
      },
    ]

  if not play:
    cfg.curriculum["track_linear_velocity_weight"] = CurriculumTermCfg(
      func=mdp.reward_weight,
      params={
        "reward_name": "track_linear_velocity",
        "weight_stages": [
          {"step": 0, "weight": 1.5},
          {"step": 7000 * 24, "weight": 1.35},
          {"step": 14000 * 24, "weight": 1.2},
        ],
      },
    )


def _configure_stepup(cfg: ManagerBasedRlEnvCfg, *, play: bool) -> None:
  cfg.rewards["track_linear_velocity"].weight = 2.0
  cfg.rewards["track_angular_velocity"].weight = 1.15
  cfg.rewards["soft_landing"].weight = -0.006
  cfg.rewards["forward_velocity_floor"] = RewardTermCfg(
    func=mdp.forward_velocity_floor, weight=-1.2,
    params={"min_velocity": 0.22, "command_name": "twist", "command_threshold": 0.1},
  )
  cfg.rewards["straight_yaw_rate_l2"] = RewardTermCfg(
    func=mdp.yaw_rate_l2_when_straight, weight=-0.60,
    params={"command_name": "twist", "command_x_threshold": 0.1,
      "command_y_threshold": 0.08, "command_yaw_threshold": 0.04},
  )
  cfg.rewards["straight_heading_drift_l2"] = RewardTermCfg(
    func=mdp.heading_drift_l2_when_straight, weight=-0.85,
    params={"command_name": "twist", "command_x_threshold": 0.1,
      "command_y_threshold": 0.08, "command_yaw_threshold": 0.04},
  )
  if "command_vel" in cfg.curriculum:
    cfg.curriculum["command_vel"].params["velocity_stages"] = [
      {
        "step": 0,
        "lin_vel_x": (0.25, 0.50),
        "lin_vel_y": (-0.04, 0.04),
        "ang_vel_z": (0.0, 0.0),
      },
      {
        "step": 4000 * 24,
        "lin_vel_x": (0.30, 0.60),
        "lin_vel_y": (-0.06, 0.06),
        "ang_vel_z": (0.0, 0.0),
      },
    ]
  if play:
    cfg.commands["twist"].ranges.lin_vel_x = (0.35, 0.45)
    cfg.commands["twist"].ranges.lin_vel_y = (-0.10, 0.10)
    cfg.commands["twist"].ranges.ang_vel_z = (-0.10, 0.10)


def _configure_step14(cfg: ManagerBasedRlEnvCfg) -> None:
  generator = cfg.scene.terrain.terrain_generator
  generator.sub_terrains["flat"] = replace(generator.sub_terrains["flat"], proportion=0.0)
  generator.sub_terrains["pyramid_stairs"] = replace(
    generator.sub_terrains["pyramid_stairs"], proportion=0.0, step_height_range=(0.14, 0.14),
  )
  generator.sub_terrains["pyramid_stairs_inv"] = replace(
    generator.sub_terrains["pyramid_stairs_inv"], proportion=1.0, step_height_range=(0.14, 0.14),
  )
  generator.sub_terrains["hf_pyramid_slope"] = replace(generator.sub_terrains["hf_pyramid_slope"], proportion=0.0)
  generator.sub_terrains["hf_pyramid_slope_inv"] = replace(generator.sub_terrains["hf_pyramid_slope_inv"], proportion=0.0)
  generator.sub_terrains["random_rough"] = replace(generator.sub_terrains["random_rough"], proportion=0.0)
  generator.sub_terrains["wave_terrain"] = replace(generator.sub_terrains["wave_terrain"], proportion=0.0)
  generator.num_rows = generator.num_cols = 1
  generator.border_width = 10.0
  generator.curriculum = False
  cfg.scene.terrain.max_init_terrain_level = 0


def unitree_go2w_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return _make_go2w_env_cfg(play, stage="rough")


def unitree_go2w_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return _make_go2w_env_cfg(play, stage="flat")


def unitree_go2w_perceptive_rough_v1_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return _make_go2w_env_cfg(play, stage="v1")


def unitree_go2w_perceptive_rough_v2_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return _make_go2w_env_cfg(play, stage="v2")


def unitree_go2w_perceptive_stepup_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return _make_go2w_env_cfg(play, stage="stepup")


def unitree_go2w_perceptive_step14_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return _make_go2w_env_cfg(play, stage="step14")
