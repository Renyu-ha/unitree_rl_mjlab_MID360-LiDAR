"""Go2W asset settings aligned with ZhongL1/unitree_rl_mjlab_go2w.

See SOURCE.md for the pinned reference and mjlab 1.2.0 compatibility adapters.
The reference configuration is distributed under Apache-2.0.
"""

from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

from .wheel_actuator import WheelVelocityActuatorCfg

GO2W_XML = Path(__file__).parent / "xmls" / "go2w.xml"
LEG_JOINTS = (r".*_(hip|thigh|calf)_joint",)
WHEEL_JOINTS = (r".*_wheel_joint",)
# Natural model order, also used by contact sensors, hub sites and rewards.
WHEEL_SITES = ("FL", "FR", "RL", "RR")
WHEEL_BODIES = ("FL_wheel_link", "FR_wheel_link", "RL_wheel_link", "RR_wheel_link")
WHEEL_GEOMS = (
  "FL_wheel_link_collision", "FR_wheel_link_collision",
  "RL_wheel_link_collision", "RR_wheel_link_collision",
)
WHEEL_RADIUS = 0.086  # metres, measured from the upstream wheel STL
LEG_ACTION_SCALE = 0.5  # radians
WHEEL_ACTION_SCALE = 20.0  # radians/second


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(GO2W_XML))
  # Embed both the shared Go2 visuals and Go2W meshes before scene attachment.
  # The XML paths also work when loaded directly by MuJoCo.
  assets = {}
  update_assets(assets, GO2W_XML.parent / "assets", "assets", glob="*.stl")
  update_assets(
    assets, GO2W_XML.parent / "../../unitree_go2/xmls/assets",
    "../../unitree_go2/xmls/assets", glob="*.obj",
  )
  spec.assets = assets
  return spec


def get_go2w_robot_cfg() -> EntityCfg:
  """Return fresh reference asset settings with continuous velocity-controlled wheels."""
  return EntityCfg(
    spec_fn=get_spec,
    init_state=EntityCfg.InitialStateCfg(
      pos=(0.0, 0.0, 0.4),
      joint_pos={
        "FR_hip_joint": 0.1,
        "FR_thigh_joint": 0.9,
        "FR_calf_joint": -1.8,
        "FL_hip_joint": -0.1,
        "FL_thigh_joint": 0.9,
        "FL_calf_joint": -1.8,
        "RR_hip_joint": 0.1,
        "RR_thigh_joint": 0.9,
        "RR_calf_joint": -1.8,
        "RL_hip_joint": -0.1,
        "RL_thigh_joint": 0.9,
        "RL_calf_joint": -1.8,
        ".*wheel_joint": 0.0,
      },
      joint_vel={".*": 0.0},
    ),
    articulation=EntityArticulationInfoCfg(
      actuators=(
        BuiltinPositionActuatorCfg(
          target_names_expr=(".*_hip_joint",),
          stiffness=20.0, damping=1.0, effort_limit=23.5, armature=0.01,
        ),
        BuiltinPositionActuatorCfg(
          target_names_expr=(".*_thigh_joint",),
          stiffness=20.0, damping=1.0, effort_limit=23.5, armature=0.01,
        ),
        BuiltinPositionActuatorCfg(
          target_names_expr=(".*_calf_joint",),
          stiffness=40.0, damping=2.0, effort_limit=45.0, armature=0.02,
        ),
        WheelVelocityActuatorCfg(
          target_names_expr=WHEEL_JOINTS,
          damping=2.0, effort_limit=45.0, armature=0.02,
          velocity_limit=WHEEL_ACTION_SCALE,
        ),
      ),
      soft_joint_pos_limit_factor=0.9,
    ),
    collisions=(CollisionCfg(
      geom_names_expr=(r".*_collision\d*$",),
      # Enable robot-to-robot contacts as well as robot-to-terrain contacts.
      # MuJoCo retains its default same-body / parent-child contact filtering.
      contype=1, conaffinity=1,
      # The reference FULL_COLLISION uses CollisionCfg's condim=3, priority=0
      # defaults. Keep the MJCF friction coefficients without extra overrides.
      condim=3, priority=0,
    ),),
  )
