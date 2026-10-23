# Go2W model provenance

Source: [Unitree Robotics / unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco/tree/1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d/unitree_robots/go2w), commit `1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d`.

`go2w.xml`, `calf.stl`, `calf_mirror.stl` and `wheel.stl` are adapted/copied under the adjacent BSD-3-Clause LICENSE. The existing Go2 OBJ visuals are referenced by relative paths to avoid duplicating them.

The XML keeps the upstream link transforms, masses, inertias, joint axes and ranges. Adaptations: named collision geoms, continuous wheel joints, non-colliding wheel visual geoms, four wheel-hub sites and mjlab IMU/momentum sensors. Upstream torque actuators are removed; mjlab adds 12 leg PD position servos and 4 wheel velocity servos. Wheel radius is approximately 0.086 m from the supplied mesh. The MID360 is represented by ray sensors; its payload mass/extrinsics must be calibrated for deployment.

## Robot configuration reference

The initial state, servo parameters and collision settings follow [ZhongL1/unitree_rl_mjlab_go2w](https://github.com/ZhongL1/unitree_rl_mjlab_go2w/blob/fe0deeaef2e6e579db63b782f1082b5d2b4ed071/mjlab/asset_zoo/robots/unitree_go2w/go2w_constants.py), commit `fe0deeaef2e6e579db63b782f1082b5d2b4ed071`, under that repository's Apache-2.0 license. These are reference training settings, not calibrated hardware parameters.

| Setting | Reference value used here |
| --- | --- |
| Initial base position | `(0, 0, 0.4)` m |
| Right / left hip position | `+0.1` / `-0.1` rad |
| Thigh / calf / wheel position | `0.9` / `-1.8` / `0` rad |
| Initial joint velocity | `0` rad/s |
| Hip and thigh KP / KD / effort / armature | `20` / `1` / `23.5` Nm / `0.01` |
| Calf KP / KD / effort / armature | `40` / `2` / `45` Nm / `0.02` |
| Wheel velocity gain / effort / armature | `2` / `45` Nm / `0.02` |
| Actuator friction loss | `0` (reference actuator default) |
| Collision contact dimension / priority | `3` / `0` (reference `CollisionCfg` defaults) |
| Collision filtering | `contype=1`, `conaffinity=0`; visual collisions disabled |
| Friction | Inherited from MJCF, including wheel `(0.8, 0.02, 0.01)` |

The reference uses the same Go2W link geometry, masses and inertias; the three Go2W STL files have identical Git blob hashes. Local names for collision geoms, wheel-hub sites and sensors are retained for the perceptive task. The mjlab 1.2.0 continuous-wheel adapter is also retained to give velocity servos a usable control range.

`contact_sensor.py` is a local shared-contact adapter. The single `feet_ground_contact` sensor exposes both total world-space forces and normal-only world-space forces, plus per-wheel contact timing. Each wheel uses one packed MuJoCo contact sensor so raw force/normal/tangent slots stay aligned; no separate spin-contact sensor is created. The 128 raw slots match the configured contact matching capacity before per-wheel aggregation.

This update aligns the robot asset and initial state. The MID360 observation curriculum, robot_lab contact-spin reward, existing action order and task action scales remain local task settings. In particular, wheel targets remain `20 * action` rad/s (with an explicit actuator range of +/-20 rad/s), while the reference repository's hybrid task uses `35 * action` rad/s. Its SDK action ordering and task rewards are not imported. Existing checkpoints keep the same tensor shapes, but their position offsets and actuator dynamics have changed; start a fresh run for the aligned configuration.
