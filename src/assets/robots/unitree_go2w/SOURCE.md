# Go2W model provenance

Source: [Unitree Robotics / unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco/tree/1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d/unitree_robots/go2w), commit `1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d`.

`go2w.xml`, `calf.stl`, `calf_mirror.stl` and `wheel.stl` are adapted/copied under the adjacent BSD-3-Clause LICENSE. The existing Go2 OBJ visuals are referenced by relative paths to avoid duplicating them.

The XML keeps the upstream link transforms, masses, inertias, joint axes and ranges. Adaptations: named collision geoms, continuous wheel joints, non-colliding wheel visual geoms, four wheel-hub sites and mjlab IMU/momentum sensors. Upstream torque actuators are removed; mjlab adds 12 leg PD position servos and 4 wheel velocity servos. Servo gains and the initial pose are training defaults, not calibrated hardware parameters. Wheel radius is approximately 0.086 m from the supplied mesh. The MID360 is represented by ray sensors; its payload mass/extrinsics must be calibrated for deployment.
