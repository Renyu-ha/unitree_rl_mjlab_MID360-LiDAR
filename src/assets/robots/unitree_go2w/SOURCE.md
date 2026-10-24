# Go2W model source

Based on [ZhongL1/unitree_rl_mjlab_go2w](https://github.com/ZhongL1/unitree_rl_mjlab_go2w/blob/fe0deeaef2e6e579db63b782f1082b5d2b4ed071/mjlab/asset_zoo/robots/unitree_go2w/xmls/go2w.xml), commit `fe0deeaef2e6e579db63b782f1082b5d2b4ed071`. The underlying Unitree model retains the adjacent BSD-3-Clause LICENSE; the reference configuration code is Apache-2.0.

`xmls/go2w.xml` is the single final model. Collision names, four wheel-hub sites, visual-only wheel duplicates and the three sensors (`imu_ang_vel`, `imu_lin_vel`, `root_angmom`) are defined directly in it. The angular momentum reward is disabled for testing; its `root_angmom` sensor is retained so the reward can be restored independently. There is no runtime XML adapter, source copy or model manifest.

The 14 meshes actually used by geometry are stored in `xmls/assets/`, unchanged from the source. Unused `calf_mirror_0.obj`, `calf_mirror_1.obj` and `foot.obj` are removed. Link transforms, masses, inertias and joint axes/ranges retain the reference values.

For the wheel-contact experiment, the four named wheel collision meshes are replaced directly in the XML by cylinders. Each cylinder has radius 0.086 m and half-width 0.0225 m (45 mm full width). Its centre is at body-local Y = +0.050764256 m on the left and -0.050764256 m on the right; its local Z axis is rotated onto the wheel joint's Y axis. The radius and centre retain the original mesh's radial extent and bounding-box centre. The wheel visual mesh remains unchanged and does not collide. Explicit wheel mass (0.98 kg) and inertia are retained rather than recomputed from cylinder density. Control gains, torque limits, armature, friction, solver settings, sensors and self collision are unchanged by this experiment. Commit `6ff9526` is the pre-cylinder revision; its robot model matches `63ec8cb`, used for the compiled-model comparison. See the [contact survey](../../../../doc/wheeled_contact_survey_zh.md) for source comparisons and experimental limits.

`get_spec()` only loads the final XML using MuJoCo's native asset path resolution. Initial state, KP/KD, action scales/order, rewards, observations and PPO remain in the existing training configurations. Self collision stays enabled. The MID360 is represented by ray sensors; hardware payload mass/extrinsics still require calibration.

The reference XML's global `<option cone="elliptic" impratio="100"/>` is removed.
MuJoCo options belong to the entire scene, not to an attached robot. Training
already used the task's `MujocoCfg` values (`pyramidal`, `impratio=1`), overriding
the XML values; newer MuJoCo/mjlab now warn about that conflict. The task remains
the single owner of these options. This also makes standalone XML loading use
MuJoCo's default pyramidal cone and impedance ratio 1 instead of elliptic/100.
Shared velocity tasks use solver/linesearch budgets 50/100 after fixed-state
convergence regression checks; contact definitions and solver tolerance are
unchanged by this numerical-budget adjustment.

On 2026-10-24 the training collision profile was aligned with this repository's G1: non-wheel collision geoms use `condim=1, priority=0`, while the four wheels use `condim=3, priority=1`. This overrides the XML contact dimensions and priority in `get_go2w_robot_cfg()`, and allows randomized wheel friction below the terrain's friction to govern wheel-ground contacts. Non-wheel base sliding friction is set to `1` directly in the XML; wheel friction remains `0.8 0.02 0.01` before task randomization. Collision masks are retained.

The mjlab-main / MuJoCo-Warp 3.16 branch initially set `margin="0"` on the five box collision geoms (the base and four thighs), because version 3.16 rejects box-box contacts with nonzero margin when native CCD or MultiCCD is enabled. On 2026-10-24 zero margin was extended to all non-wheel collision geoms, matching G1. The wheel defaults explicitly retain `margin="0.001"`. Native CCD and MultiCCD stay enabled at their engine defaults. Geometry dimensions and wheel tread shape are unchanged by the margin adjustment.

The initial cylinder used the original mesh's 51.814914 mm full axial bounding box. The current width is narrowed to 45 mm, matching the cylinder declarations in the surveyed HIMLoco, DreamWaQ and public robot_lab Go2W URDFs. Each edge moves inward by 3.407457 mm while the wheel centre stays fixed. Commit `4699c76` is the wider-cylinder baseline with the same training configuration for this width-only comparison. A straight cylinder still does not reproduce the original rounded tread shoulders; the survey records this geometric limitation and cross-section measurements.

Wheels use the pinned latest mjlab's native `BuiltinVelocityActuatorCfg`, with damping 2, effort limit 45 Nm and armature 0.0. The old mjlab 1.2 continuous-joint monkey patch is removed: upstream now sets `inheritrange=0` and disables control clipping for velocity actuators. All actuators explicitly set `frictionloss=0.0` to preserve the old configuration's effective value instead of inheriting the XML's joint friction. Policy action clipping remains disabled (`clip_actions=None`); wheel targets remain `20 * action` rad/s and native torque limiting remains active. See [installation and migration](../../../../doc/setup_zh.md) and the [G1 alignment record](../../../../doc/go2w_g1_alignment_zh.md).
