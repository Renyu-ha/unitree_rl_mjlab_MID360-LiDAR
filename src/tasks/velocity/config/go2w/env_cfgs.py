"""Go2W velocity tasks built from the common velocity environment root.

All registered variants start at the same private Go2W builder instead of
calling another task factory. Rough/perceptive variants share observation and
action dimensions, including three actor history frames.
"""

from copy import deepcopy
from dataclasses import replace
from typing import Literal

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg, JointVelocityActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, ObjRef, RayCastSensorCfg
from mjlab.tasks.velocity.mdp.terminations import illegal_contact
from mjlab.terrains.config import ROUGH_TERRAINS_CFG
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from src.assets.robots.unitree_go2w.go2w_constants import (
  LEG_ACTION_SCALE, LEG_JOINTS, WHEEL_ACTION_SCALE, WHEEL_BODIES,
  WHEEL_GEOMS, WHEEL_JOINTS, WHEEL_RADIUS, WHEEL_SITES, get_go2w_robot_cfg,
)
from src.tasks.velocity import mdp
from src.tasks.velocity.mdp.wheel_rewards import wheel_spin_contact, wheel_stand_still
from src.tasks.velocity.velocity_env_cfg import OffsetGridPatternCfg, make_velocity_env_cfg

SCAN = {
  "size_x_m": 1.6, "size_y_m": 1.0, "resolution_m": 0.1,
  "x_offset_m": 0.35, "y_offset_m": 0.0,
}


def _terrain_cfg(play: bool, perceptive: bool = False, stepup: bool = False):
  terrain = deepcopy(ROUGH_TERRAINS_CFG)
  terrain.curriculum = not play
  terrain.num_rows = 5 if play else 10
  terrain.num_cols = 5 if play else 6
  terrain.border_width = 10.0
  # Environments spawn in the centre: inverted pyramids train outward ascent.
  terrain.sub_terrains["flat"] = replace(
    terrain.sub_terrains["flat"], proportion=0.10 if perceptive else 0.20,
  )
  terrain.sub_terrains["pyramid_stairs"] = replace(
    terrain.sub_terrains["pyramid_stairs"], proportion=0.10 if stepup else 0.20,
    step_height_range=(0.08, 0.08) if play else (0.02, 0.14 if perceptive else 0.10),
    step_width=0.40, platform_width=2.0,
  )
  terrain.sub_terrains["pyramid_stairs_inv"] = replace(
    terrain.sub_terrains["pyramid_stairs_inv"],
    proportion=0.60 if stepup else (0.35 if perceptive else 0.20),
    step_height_range=(0.08, 0.08) if play else (0.02, 0.14 if perceptive else 0.10),
    step_width=0.40, platform_width=2.0,
  )
  terrain.sub_terrains["hf_pyramid_slope"] = replace(
    terrain.sub_terrains["hf_pyramid_slope"],
    proportion=0.05 if stepup else 0.10, slope_range=(0.0, 0.30),
  )
  terrain.sub_terrains["hf_pyramid_slope_inv"] = replace(
    terrain.sub_terrains["hf_pyramid_slope_inv"],
    proportion=0.05 if stepup else 0.10, slope_range=(0.0, 0.30),
  )
  terrain.sub_terrains["random_rough"] = replace(
    terrain.sub_terrains["random_rough"], proportion=0.05 if stepup else 0.10,
    noise_range=(0.005, 0.035), noise_step=0.005,
  )
  terrain.sub_terrains["wave_terrain"] = replace(
    terrain.sub_terrains["wave_terrain"], proportion=0.05, amplitude_range=(0.0, 0.05),
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
  stepup = stage in ("stepup", "step14")
  cfg.scene.entities = {"robot": get_go2w_robot_cfg()}
  cfg.sim.nconmax = 128
  cfg.sim.njmax = 2048
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 128
  cfg.scene.terrain.terrain_generator = _terrain_cfg(play, perceptive=perceptive, stepup=stepup)
  cfg.scene.terrain.max_init_terrain_level = 0

  terrain_scan = RayCastSensorCfg(
    name="terrain_scan",
    frame=ObjRef(type="body", name="base_link", entity="robot"),
    ray_alignment="yaw", include_geom_groups=(0, 1),
    pattern=OffsetGridPatternCfg(
      size=(SCAN["size_x_m"], SCAN["size_y_m"]), resolution=SCAN["resolution_m"],
      x_offset=SCAN["x_offset_m"], y_offset=SCAN["y_offset_m"],
    ),
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

  cfg.scene.sensors = (terrain_scan, feet_ground_cfg, nonfoot_ground_cfg)

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
      history_length=3, concatenate_terms=True, enable_corruption=True,
      terms={
        "base_ang_vel": ObservationTermCfg(
          func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_ang_vel"},
          noise=Unoise(n_min=-0.2, n_max=0.2),
        ),
        "projected_gravity": ObservationTermCfg(
          func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05),
        ),
        "command": ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "twist"}),
        "phase": ObservationTermCfg(func=mdp.phase, params={"period": 0.6, "command_name": "twist"}),
        "joint_pos": ObservationTermCfg(
          func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS)},
          noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "joint_vel": ObservationTermCfg(
          func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS)},
          noise=Unoise(n_min=-1.5, n_max=1.5),
        ),
        "actions": ObservationTermCfg(func=mdp.last_action),
        "height_scan": ObservationTermCfg(
          func=envs_mdp.height_scan, params={"sensor_name": "terrain_scan"},
          noise=Unoise(n_min=-0.1, n_max=0.1), scale=0.2,
        ),
        "wheel_vel": ObservationTermCfg(
          func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINTS)},
          noise=Unoise(n_min=-1.5, n_max=1.5), scale=0.05,
        ),
      },
    ),
    "critic": ObservationGroupCfg(
      history_length=1, concatenate_terms=True, enable_corruption=False,
      terms={
        "base_ang_vel": ObservationTermCfg(
          func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_ang_vel"},
          noise=Unoise(n_min=-0.2, n_max=0.2),
        ),
        "projected_gravity": ObservationTermCfg(
          func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05),
        ),
        "command": ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "twist"}),
        "phase": ObservationTermCfg(func=mdp.phase, params={"period": 0.6, "command_name": "twist"}),
        "joint_pos": ObservationTermCfg(
          func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS)},
          noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "joint_vel": ObservationTermCfg(
          func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS)},
          noise=Unoise(n_min=-1.5, n_max=1.5),
        ),
        "actions": ObservationTermCfg(func=mdp.last_action),
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
          noise=Unoise(n_min=-1.5, n_max=1.5), scale=0.05,
        ),
      },
    ),
  }
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = WHEEL_GEOMS
  cfg.events["encoder_bias"].params["asset_cfg"].joint_names = LEG_JOINTS
  cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)
  cfg.events["base_com"].params["ranges"] = {
    0: (-0.03, 0.03), 1: (-0.02, 0.02), 2: (-0.02, 0.02),
  }

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
  cfg.rewards["track_linear_velocity"].weight = 1.5
  cfg.rewards["track_angular_velocity"].weight = 0.75
  cfg.rewards["pose"].weight = 0.5
  cfg.rewards["body_orientation_l2"].weight = -1.5
  cfg.rewards["action_rate_l2"].weight = -0.03
  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=illegal_contact,
    params={"sensor_name": nonfoot_ground_cfg.name, "force_threshold": 10.0},
  )

  twist = cfg.commands["twist"]
  twist.rel_standing_envs = 0.05
  twist.heading_command = False
  twist.ranges.heading = None
  twist.ranges.lin_vel_x = (0.0, 0.6)
  twist.ranges.lin_vel_y = (0.0, 0.0)
  twist.ranges.ang_vel_z = (-0.2, 0.2)
  if not play:
    cfg.curriculum["command_vel"].params["velocity_stages"] = [
      {"step": 0, "lin_vel_x": (0.0, 0.6), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-0.2, 0.2)},
      {"step": 4000 * 24, "lin_vel_x": (-0.2, 1.0), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-0.4, 0.4)},
      {"step": 9000 * 24, "lin_vel_x": (-0.3, 1.2), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-0.6, 0.6)},
    ]
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
    _configure_perception(cfg, play=play, version=1 if stage == "v1" else 2)
  if stepup:
    _configure_stepup(cfg, play=play)
  if stage == "step14":
    _configure_step14(cfg)
  if stage == "flat":
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    cfg.scene.sensors = (feet_ground_cfg, nonfoot_ground_cfg)
    cfg.observations["actor"].terms.pop("height_scan")
    cfg.observations["critic"].terms.pop("height_scan")
    cfg.curriculum.pop("terrain_levels", None)
    cfg.events.pop("randomize_terrain", None)
  return cfg


def _configure_perception(cfg: ManagerBasedRlEnvCfg, *, play: bool, version: int) -> None:
  """Apply a sensing stage to a fresh Go2W config, without calling a task."""
  scan = cfg.observations["actor"].terms["height_scan"]
  scan.noise = Unoise(n_min=-0.08, n_max=0.08)
  scan.delay_max_lag = 0 if play else 1
  if version == 1:
    return
  scan.func = mdp.bridged_mid360_height_scan_memory()
  scan.params = {
    # Both views share rays; the memory term applies visibility and sparsity.
    "sensor_name": "terrain_scan", "mid360_sensor_name": "terrain_scan", **SCAN,
    "visible_x_min_m": 0.25, "visible_x_max_m": 1.15, "visible_y_abs_m": 0.50,
    "keep_prob_min": 1.0 if play else 0.65, "keep_prob_max": 1.0,
    "frame_dropout_prob": 0.0 if play else 0.02,
    "switch_start_step": 500 * 24, "switch_end_step": 2500 * 24,
    "sparse_start_step": 2500 * 24, "sparse_end_step": 6000 * 24,
    "fixed_curriculum_step": 6000 * 24 if play else None,
    "memory_max_age_s": 0.8, "enable_memory_shift": True, "miss_value": 5.0,
  }
  scan.noise = Unoise(n_min=-0.10, n_max=0.10)
  scan.delay_max_lag = 0 if play else 2
  scan.delay_hold_prob = 0.0 if play else 0.35
  scan.delay_update_period = 0 if play else 3


def _configure_stepup(cfg: ManagerBasedRlEnvCfg, *, play: bool) -> None:
  cfg.commands["twist"].rel_standing_envs = 0.02
  cfg.commands["twist"].ranges.lin_vel_x = (0.15, 0.45)
  cfg.commands["twist"].ranges.ang_vel_z = (0.0, 0.0)
  if not play:
    cfg.curriculum["command_vel"].params["velocity_stages"] = [
      {"step": 0, "lin_vel_x": (0.15, 0.45), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (0.0, 0.0)},
    ]


def _configure_step14(cfg: ManagerBasedRlEnvCfg) -> None:
  generator = cfg.scene.terrain.terrain_generator
  generator.sub_terrains["flat"] = replace(generator.sub_terrains["flat"], proportion=0.0)
  generator.sub_terrains["pyramid_stairs"] = replace(generator.sub_terrains["pyramid_stairs"], proportion=0.0)
  generator.sub_terrains["pyramid_stairs_inv"] = replace(
    generator.sub_terrains["pyramid_stairs_inv"], proportion=1.0, step_height_range=(0.14, 0.14),
  )
  generator.sub_terrains["hf_pyramid_slope"] = replace(generator.sub_terrains["hf_pyramid_slope"], proportion=0.0)
  generator.sub_terrains["hf_pyramid_slope_inv"] = replace(generator.sub_terrains["hf_pyramid_slope_inv"], proportion=0.0)
  generator.sub_terrains["random_rough"] = replace(generator.sub_terrains["random_rough"], proportion=0.0)
  generator.sub_terrains["wave_terrain"] = replace(generator.sub_terrains["wave_terrain"], proportion=0.0)
  generator.num_rows = generator.num_cols = 1
  generator.curriculum = False
  cfg.curriculum.pop("terrain_levels", None)


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
