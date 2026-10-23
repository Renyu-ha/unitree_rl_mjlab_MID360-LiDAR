# Go2W model source

Based on [ZhongL1/unitree_rl_mjlab_go2w](https://github.com/ZhongL1/unitree_rl_mjlab_go2w/blob/fe0deeaef2e6e579db63b782f1082b5d2b4ed071/mjlab/asset_zoo/robots/unitree_go2w/xmls/go2w.xml), commit `fe0deeaef2e6e579db63b782f1082b5d2b4ed071`. The underlying Unitree model retains the adjacent BSD-3-Clause LICENSE; the reference configuration code is Apache-2.0.

`xmls/go2w.xml` is the single final model. Collision names, four wheel-hub sites, visual-only wheel duplicates and the three task sensors (`imu_ang_vel`, `imu_lin_vel`, `root_angmom`) are defined directly in it. Unused telemetry has been deleted. There is no runtime XML adapter, source copy or model manifest.

The 14 meshes actually used by geometry are stored in `xmls/assets/`, unchanged from the source. Unused `calf_mirror_0.obj`, `calf_mirror_1.obj` and `foot.obj` are removed. Link transforms, masses, inertias, joint axes/ranges and physical shapes retain the reference values.

`get_spec()` only loads the final XML and embeds its mesh files. Initial state, KP/KD, action scales/order, rewards, observations and PPO remain in the existing training configurations. Self collision stays enabled. The MID360 is represented by ray sensors; hardware payload mass/extrinsics still require calibration.
