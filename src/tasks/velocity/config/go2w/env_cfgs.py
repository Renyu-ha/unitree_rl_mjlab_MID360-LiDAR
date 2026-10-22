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
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
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
  proportions = {
    "flat": 0.10 if perceptive else 0.20,
    "pyramid_stairs": 0.10 if stepup else 0.20,
    "pyramid_stairs_inv": 0.60 if stepup else (0.35 if perceptive else 0.20),
    "hf_pyramid_slope": 0.05 if stepup else 0.10,
    "hf_pyramid_slope_inv": 0.05 if stepup else 0.10,
    "random_rough": 0.05 if stepup else 0.10,
    "wave_terrain": 0.05,
  }
  for name, sub in terrain.sub_terrains.items():
    changes = {"proportion": proportions.get(name, 0.0)}
    if name in ("pyramid_stairs", "pyramid_stairs_inv"):
      changes.update(
        step_height_range=(0.08, 0.08) if play else (0.02, 0.14 if perceptive else 0.10),
        step_width=0.40, platform_width=2.0,
      )
    elif name.startswith("hf_pyramid_slope"):
      changes["slope_range"] = (0.0, 0.30)
    elif name == "random_rough":
      changes.update(noise_range=(0.005, 0.035), noise_step=0.005)
    elif name == "wave_terrain":
      changes["amplitude_range"] = (0.0, 0.05)
    terrain.sub_terrains[name] = replace(sub, **changes)
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

  for sensor in cfg.scene.sensors:
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.frame.name = "base_link"
      sensor.include_geom_groups = (0, 1)
      # Dense and sparse observations refer to exactly the same grid cells.
      sensor.pattern = OffsetGridPatternCfg(
        size=(SCAN["size_x_m"], SCAN["size_y_m"]), resolution=SCAN["resolution_m"],
        x_offset=SCAN["x_offset_m"], y_offset=SCAN["y_offset_m"],
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

  # Keep the existing netforce sensor for observations/landing rewards. Raw
  # contact slots let wheel_spin_contact sum normal reactions without friction.
  wheel_spin_contact_cfg = ContactSensorCfg(
    name="wheel_spin_contact",
    primary=ContactMatch(mode="geom", pattern=WHEEL_GEOMS, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("force", "normal"), reduce="none", num_slots=8,
  )
  cfg.scene.sensors += (feet_ground_cfg, nonfoot_ground_cfg, wheel_spin_contact_cfg)

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
  for name, group in cfg.observations.items():
    group.terms = deepcopy(group.terms)
    group.history_length = 3 if name == "actor" else 1
    # Absolute wheel angles grow without bound and carry no useful pose signal.
    for term in ("joint_pos", "joint_vel"):
      group.terms[term].params["asset_cfg"] = SceneEntityCfg("robot", joint_names=LEG_JOINTS)
    group.terms["wheel_vel"] = replace(
      deepcopy(group.terms["joint_vel"]),
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINTS)},
      scale=0.05,
    )
  cfg.observations["critic"].terms["foot_height"].params["asset_cfg"].site_names = WHEEL_SITES
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = WHEEL_GEOMS
  cfg.events["encoder_bias"].params["asset_cfg"].joint_names = LEG_JOINTS
  cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)
  cfg.events["base_com"].params["ranges"] = {
    0: (-0.03, 0.03), 1: (-0.02, 0.02), 2: (-0.02, 0.02),
  }

  for term in ("pose", "stand_still", "joint_pos_limits", "joint_acc_l2"):
    cfg.rewards[term].params["asset_cfg"] = SceneEntityCfg("robot", joint_names=LEG_JOINTS)
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
  for term in ("foot_gait", "foot_clearance", "foot_slip"):
    cfg.rewards.pop(term)
  cfg.rewards["wheel_spin_contact"] = RewardTermCfg(
    func=wheel_spin_contact, weight=-0.15,
    params={
      "sensor_name": wheel_spin_contact_cfg.name, "wheel_radius": WHEEL_RADIUS,
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
    cfg.scene.sensors = tuple(s for s in cfg.scene.sensors if s.name != "terrain_scan")
    for group in cfg.observations.values():
      group.terms.pop("height_scan")
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
  dense = next(s for s in cfg.scene.sensors if s.name == "terrain_scan")
  scan.func = mdp.bridged_mid360_height_scan_memory()
  scan.params = {
    # Both views share rays; the memory term applies visibility and sparsity.
    "sensor_name": dense.name, "mid360_sensor_name": dense.name, **SCAN,
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
  for name, sub in generator.sub_terrains.items():
    if name == "pyramid_stairs_inv":
      generator.sub_terrains[name] = replace(sub, proportion=1.0, step_height_range=(0.14, 0.14))
    else:
      generator.sub_terrains[name] = replace(sub, proportion=0.0)
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
