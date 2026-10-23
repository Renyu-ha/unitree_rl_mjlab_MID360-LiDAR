# Go2W model source

Based on [ZhongL1/unitree_rl_mjlab_go2w](https://github.com/ZhongL1/unitree_rl_mjlab_go2w/blob/fe0deeaef2e6e579db63b782f1082b5d2b4ed071/mjlab/asset_zoo/robots/unitree_go2w/xmls/go2w.xml), commit `fe0deeaef2e6e579db63b782f1082b5d2b4ed071`. The underlying Unitree model retains the adjacent BSD-3-Clause LICENSE; the reference configuration code is Apache-2.0.

`xmls/go2w.xml` is the single final model. Collision names, four wheel-hub sites, visual-only wheel duplicates and the three sensors (`imu_ang_vel`, `imu_lin_vel`, `root_angmom`) are defined directly in it. The angular momentum reward is disabled for testing; its `root_angmom` sensor is retained so the reward can be restored independently. There is no runtime XML adapter, source copy or model manifest.

The 14 meshes actually used by geometry are stored in `xmls/assets/`, unchanged from the source. Unused `calf_mirror_0.obj`, `calf_mirror_1.obj` and `foot.obj` are removed. Link transforms, masses, inertias, joint axes/ranges and physical shapes retain the reference values.

`get_spec()` only loads the final XML and embeds its mesh files. Initial state, KP/KD, action scales/order, rewards, observations and PPO remain in the existing training configurations. Self collision stays enabled. The MID360 is represented by ray sensors; hardware payload mass/extrinsics still require calibration.

Wheels now use mjlab's `BuiltinVelocityActuatorCfg`, with damping 2, effort limit 45 Nm and armature 0.0 as requested. `src/utils/mjlab_compat.py` applies the continuous-joint range handling used by the reference repository's [velocity factory](https://github.com/ZhongL1/unitree_rl_mjlab_go2w/blob/fe0deeaef2e6e579db63b782f1082b5d2b4ed071/mjlab/utils/spec.py#L204-L213): no inherited control range or control clipping on an unlimited joint. The Go2W asset package enables it at import in each process, including workers importing a pickled `spec_fn`, without modifying installed dependency files. The previous custom wheel actuator is removed. Target speeds remain bounded by policy action clipping and the 20 rad/s action scale; torque limiting remains in the native actuator.
