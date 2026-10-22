"""Go2W model and mixed position/velocity actuators for mjlab 1.2.0."""

from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

from .wheel_actuator import WheelVelocityActuatorCfg

GO2W_XML = Path(__file__).parent / "xmls" / "go2w.xml"
LEG_JOINTS = (r".*_(hip|thigh|calf)_joint",)
WHEEL_JOINTS = (r".*_wheel_joint",)
# Natural model order, also used by contact sensors, hub sites and rewards.
WHEEL_SITES = ("FL", "FR", "RL", "RR")
WHEEL_BODIES = tuple(f"{leg}_wheel_link" for leg in WHEEL_SITES)
WHEEL_GEOMS = tuple(f"{body}_collision" for body in WHEEL_BODIES)
WHEEL_RADIUS = 0.086  # metres, measured from the upstream wheel STL
LEG_ACTION_SCALE = 0.5  # radians
WHEEL_ACTION_SCALE = 20.0  # radians/second


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(GO2W_XML))
  # Embed both the shared Go2 visuals and Go2W meshes before scene attachment.
  # The XML paths also work when loaded directly by MuJoCo.
  spec.assets = {
    mesh.file: (GO2W_XML.parent / mesh.file).read_bytes() for mesh in spec.meshes
  }
  return spec


def get_go2w_robot_cfg() -> EntityCfg:
  """Return an independent config; wheels are continuous velocity-controlled joints."""
  return EntityCfg(
    spec_fn=get_spec,
    init_state=EntityCfg.InitialStateCfg(
      pos=(0.0, 0.0, 0.42),
      joint_pos={
        ".*L_hip_joint": 0.1,
        ".*R_hip_joint": -0.1,
        ".*thigh_joint": 0.8,
        ".*calf_joint": -1.5,
        ".*wheel_joint": 0.0,
      },
      joint_vel={".*": 0.0},
    ),
    articulation=EntityArticulationInfoCfg(
      actuators=(
        BuiltinPositionActuatorCfg(
          target_names_expr=(".*_(hip|thigh)_joint",),
          stiffness=30.0, damping=1.0, effort_limit=23.7, armature=0.01,
        ),
        BuiltinPositionActuatorCfg(
          target_names_expr=(".*_calf_joint",),
          stiffness=40.0, damping=1.5, effort_limit=45.43, armature=0.02,
        ),
        WheelVelocityActuatorCfg(
          target_names_expr=WHEEL_JOINTS,
          damping=0.5, effort_limit=15.0, armature=0.01, frictionloss=0.01,
          velocity_limit=WHEEL_ACTION_SCALE,
        ),
      ),
      soft_joint_pos_limit_factor=0.9,
    ),
    collisions=(CollisionCfg(
      geom_names_expr=(r".*_collision\d*$",),
      contype=1, conaffinity=0,
      condim={r".*_wheel_link_collision": 6, r"(?!.*_wheel_link).*_collision\d*": 1},
      priority={r".*_wheel_link_collision": 1},
      friction={r".*_wheel_link_collision": (0.8, 0.02, 0.001)},
    ),),
  )
