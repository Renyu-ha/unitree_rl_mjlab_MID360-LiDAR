"""Bind the unmodified upstream Go2W model to this task's mjlab interfaces.

The XML and meshes remain verbatim upstream files. This adapter adds names,
wheel-hub sites and task sensors; it does not define link geometry or inertias.
"""

import mujoco


def _bind_wheel(spec: mujoco.MjSpec, leg: str, lateral: float) -> None:
  body = spec.body(f"{leg}_wheel_link")
  body.geoms[1].name = f"{leg}_wheel_link_collision"
  # The source contains a decorative duplicate of the physical wheel mesh.
  # Keep it visual-only, matching the existing task's collision selection.
  body.geoms[0].contype = 0
  body.geoms[0].conaffinity = 0
  body.geoms[0].group = 2
  body.add_site(name=leg, pos=(0.0, lateral, 0.0), size=(0.01, 0.005, 0.005), group=5)


def bind_task_interfaces(spec: mujoco.MjSpec) -> None:
  """Add explicit task bindings in memory, leaving the source file untouched."""
  # Upstream collision geoms are unnamed. The current contact sensors and
  # CollisionCfg select these names; retain their existing order and shapes.
  base = spec.body("base_link")
  base.geoms[5].name = "base_link_collision0"
  base.geoms[6].name = "base_link_collision1"
  base.geoms[7].name = "base_link_collision2"

  spec.body("FL_hip").geoms[2].name = "FL_hip_collision"
  spec.body("FL_thigh").geoms[2].name = "FL_thigh_collision"
  spec.body("FL_calf").geoms[1].name = "FL_calf_collision0"
  spec.body("FL_calf").geoms[2].name = "FL_calf_collision1"
  _bind_wheel(spec, "FL", 0.05)

  spec.body("FR_hip").geoms[2].name = "FR_hip_collision"
  spec.body("FR_thigh").geoms[2].name = "FR_thigh_collision"
  spec.body("FR_calf").geoms[1].name = "FR_calf_collision0"
  spec.body("FR_calf").geoms[2].name = "FR_calf_collision1"
  _bind_wheel(spec, "FR", -0.05)

  spec.body("RL_hip").geoms[2].name = "RL_hip_collision"
  spec.body("RL_thigh").geoms[2].name = "RL_thigh_collision"
  spec.body("RL_calf").geoms[1].name = "RL_calf_collision0"
  spec.body("RL_calf").geoms[2].name = "RL_calf_collision1"
  _bind_wheel(spec, "RL", 0.05)

  spec.body("RR_hip").geoms[2].name = "RR_hip_collision"
  spec.body("RR_thigh").geoms[2].name = "RR_thigh_collision"
  spec.body("RR_calf").geoms[1].name = "RR_calf_collision0"
  spec.body("RR_calf").geoms[2].name = "RR_calf_collision1"
  _bind_wheel(spec, "RR", -0.05)

  # Preserve every original telemetry sensor. Add the existing observation
  # names instead of rewriting/removing the upstream sensor group.
  spec.add_sensor(
    name="imu_ang_vel", type=mujoco.mjtSensor.mjSENS_GYRO,
    objtype=mujoco.mjtObj.mjOBJ_SITE, objname="imu",
  )
  spec.add_sensor(
    name="imu_lin_vel", type=mujoco.mjtSensor.mjSENS_VELOCIMETER,
    objtype=mujoco.mjtObj.mjOBJ_SITE, objname="imu",
  )
  spec.add_sensor(
    name="imu_lin_acc", type=mujoco.mjtSensor.mjSENS_ACCELEROMETER,
    objtype=mujoco.mjtObj.mjOBJ_SITE, objname="imu",
  )
  spec.add_sensor(
    name="root_angmom", type=mujoco.mjtSensor.mjSENS_SUBTREEANGMOM,
    objtype=mujoco.mjtObj.mjOBJ_BODY, objname="base_link",
  )
