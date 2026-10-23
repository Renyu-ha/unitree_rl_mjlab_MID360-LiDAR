# Go2W model provenance

Model source: [ZhongL1/unitree_rl_mjlab_go2w](https://github.com/ZhongL1/unitree_rl_mjlab_go2w/blob/fe0deeaef2e6e579db63b782f1082b5d2b4ed071/mjlab/asset_zoo/robots/unitree_go2w/xmls/go2w.xml), commit `fe0deeaef2e6e579db63b782f1082b5d2b4ed071`.

The model actually loaded by `get_spec()`, `xmls/go2w.xml`, is a byte-for-byte copy of that repository's XML (Git blob `ae8ab3e32e3043e06752009fdd33be20d74ccc85`). There is no separately rewritten local XML. All 17 meshes declared by it are copied byte-for-byte into `xmls/assets/`; the active model no longer references the Go2 asset directory. `xmls/model_source.json` records the source commit, XML Git blob hash, every mesh's path/size/hash, and the task-interface adapter. Local `.gitattributes` preserves these bytes across Windows checkouts. Unused meshes from the source directory are not included.

The model originated in [Unitree Robotics / unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco/tree/1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d/unitree_robots/go2w); the adjacent BSD-3-Clause LICENSE is retained. The ZhongL1 repository's configuration code is distributed under Apache-2.0.

`model_adapter.py` binds task interfaces in memory after loading the unmodified XML: names for the existing collision geoms, four wheel-hub sites, and four additional IMU/momentum sensor names used by this task. All 53 upstream telemetry sensors remain present (57 total with the task sensors). The decorative duplicate of each wheel mesh is kept visual-only, matching the current task's collision selection. No link transforms, masses, inertias, joint axes/ranges or physical shapes are defined by the adapter. The original wheel joints are already unlimited; no wheel-joint override is needed. mjlab names the unnamed free joint itself and still adds the same 12 leg PD position servos and 4 wheel velocity servos. The existing `EntityCfg` collision settings still apply, including self collision. The bindings are explicit and contain no loops or comprehensions. Wheel radius is approximately 0.086 m from the supplied mesh. The MID360 is represented by ray sensors; its payload mass/extrinsics must be calibrated for deployment.

This model-file replacement leaves all training configurations unchanged, including initial state, gains, action scales/order, rewards, observations, terrain and PPO. The previous model's physical geometry and mesh contents already matched the pinned source. Compiled physics parameters and the train/play/PPO configurations are checked against the pre-replacement baseline; the replacement itself is not a NaN fix.

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
| Collision filtering | `contype=1`, `conaffinity=1`; self collision enabled; visual collisions disabled |
| Friction | Inherited from MJCF, including wheel `(0.8, 0.02, 0.01)` |

All 17 mesh files and the original XML match the pinned source's Git blob hashes. Local names for collision geoms, wheel-hub sites and sensors are retained for the perceptive task. The mjlab 1.2.0 continuous-wheel adapter is also retained to give velocity servos a usable control range.

At the user's request, robot collision affinity is now `1` rather than the reference's `0`. Non-adjacent robot links can collide; MuJoCo's default same-body and parent-child filtering remains enabled. This applies to all Go2W training/play tasks. The default standing pose has no self contacts; a folded-leg fixture produces wheel-to-wheel and wheel-to-calf contacts. Policies trained with affinity `0` have compatible interfaces but different contact dynamics and should be retrained or fine-tuned with the new settings. Previously saved MJB dumps retain their original masks.

The single `feet_ground_contact` uses ordinary mjlab `ContactSensorCfg` with `fields=("found", "force")`, `reduce="netforce"`, `num_slots=1` and per-wheel contact timing. Critic observations, landing rewards and the spin gate share its total world-space contact force. The gate uses the current force magnitude > 1 N, including friction; this intentionally differs from robot_lab's normal-only gate while retaining its spin formula. No custom contact adapter or separate spin-contact sensor is needed.

The initial state and servo settings were aligned in an earlier change; this model-only replacement preserves them. The MID360 observation curriculum, robot_lab contact-spin reward, existing action order and task action scales remain local task settings. In particular, wheel targets remain `20 * action` rad/s (with an explicit actuator range of +/-20 rad/s), while the reference repository's hybrid task uses `35 * action` rad/s. Its SDK action ordering and task rewards are not imported. This replacement preserves checkpoint interfaces and compiled actuator dynamics; the earlier self-collision change still affects policies trained with collision affinity `0` as described above.
