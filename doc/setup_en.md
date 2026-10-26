# Installation and training

## Environment

Activate a Python environment and run this from the repository root:

```bash
pip install -e .
```

This installs the project, pinned mjlab and a fixed MuJoCo-Warp fork through standard dependency metadata. Git and access to GitHub/PyPI are required. Python 3.11 is recommended; 3.11–3.13 are supported. No extra package index, constraints file, uv command, log-directory engine copy or startup source patch is needed.

| Component | Pinned version |
| --- | --- |
| mjlab | `8bb2bc9e187797689ec82c78c35019afd2fd0376` (package 1.6.0) |
| MuJoCo | 3.16.0 |
| MuJoCo-Warp | 3.16.0.post1, fork commit `6dea56edf221906310a90896e3ad913ad12459c3` |
| Warp | 1.18.0 |
| RSL-RL | 5.5.1 |
| PyTorch | 2.14.1 |

Core versions are pinned in `pyproject.toml`; pip resolves transitive dependencies. PyTorch uses the default PyPI package for the platform. The NVIDIA driver must support the installed PyTorch/Warp packages.

For a new Conda environment:

```bash
git clone --recurse-submodules -b feat/mjlab-main-conda-3-16 https://github.com/Renyu-ha/unitree_rl_mjlab_MID360-LiDAR.git
cd unitree_rl_mjlab_MID360-LiDAR
conda env create -f environment.yml
conda activate unitree_mjlab_main
pip install -e .
```

For an existing environment, update this branch and rerun `pip install -e .`. Remove collision-experiment `PYTHONPATH` prefixes and the former `--epa-horizon` / `--capture-epa` options. Train, play and recorded-input NaN replay verify the installed backend version and capacity. Training saves its source path and checksum in `params/collision.json`.

RSL-RL is pinned to 5.5.1 in `pyproject.toml` and resolved by `pip install -e .` during installation. Train, play, evaluation and checkpoint loading do not run extra RSL-RL version, source or hash checks or print this information. Check dependency consistency after installation with:

```bash
python -m pip check
```

## Fixed backend

The fork starts at official MuJoCo-Warp v3.16.0, backports upstream [#1771](https://github.com/google-deepmind/mujoco_warp/pull/1771) (EPA contact extrapolation) and [#1796](https://github.com/google-deepmind/mujoco_warp/pull/1796) (degenerate closest faces), and fixes `MJ_MAX_EPAHORIZON=64`. Exact source commits and the original upstream regression tests are documented in [BACKPORTS.md](https://github.com/Renyu-ha/mujoco_warp/blob/6dea56edf221906310a90896e3ad913ad12459c3/BACKPORTS.md). Version `3.16.0.post1` satisfies mjlab's `~=3.16.0` requirement.

64 is an EPA edge-buffer capacity, not an accuracy setting or iteration count. It increases buffer memory without forcing 64 operations per collision. FMA, stopping tolerances, default CCD/MultiCCD and overflow warnings remain enabled. Capacity handles EPA overflow; the two algorithm fixes address invalid contacts.

## Train, resume and play

```bash
python scripts/train.py Unitree-Go2W-Perceptive-Rough-v2 \
  --env.scene.num-envs=4096 \
  --enable-nan-guard=True \
  --env.sim.nan-guard.output-dir=logs/go2w_nan_dump \
  --env.observations.actor.nan-policy=error \
  --env.observations.critic.nan-policy=error
```

Default: GPU 0, W&B project `go2w_mid360`, local model files under `logs/rsl_rl/go2w_perceptive/`. Use `--agent.logger=tensorboard` for local-only logging and `--gpu-ids=None` for CPU checks. A new backend may compile Warp kernels on its first run; subsequent runs use the cache. Optional live viewer: `--viewer=viser --viewer-host=0.0.0.0 --viewer-port=8096`.

The Viser display adapter converts MuJoCo's elevation sign convention while preserving native/video camera settings. Paused play publishes real initial transforms before the first frame, without requiring a Step action. Normal play and border inspection share the fix; the training viewer uses the same elevation conversion. Lighting and materials retain upstream defaults.

Select the source run explicitly when resuming:

```bash
python scripts/train.py Unitree-Go2W-Perceptive-Rough-v2 \
  --env.scene.num-envs=4096 --agent.resume=True \
  --agent.load-run='<source-run-directory>' \
  --agent.load-checkpoint='model_1000.pt' --enable-nan-guard=True

python scripts/play.py Unitree-Go2W-Perceptive-Rough-v2 \
  --checkpoint-file=logs/rsl_rl/go2w_perceptive/<run>/model_1000.pt

python scripts/play.py Unitree-Go2W-Perceptive-Rough-v2 \
  --onnx-file=<export>/policy.onnx --device=cpu --viewer=viser
```

v2/finetune/Step14 have compatible observation shapes; Rough/v1 have a different history length. Backend fixes can change collision results, so resumed trajectories need not match the old backend. ONNX play uses ONNX's ReferenceEvaluator without an extra ONNX Runtime installation. Existing robot tasks and deployment functionality are retained.

Go2W has two static wheel collision entries: default `xmls/go2w.xml` uses cylinders of radius 86 mm and full width 45 mm; `xmls/go2w_mesh.xml` uses the original `wheel.stl` convex hull, retaining its approximately 51.815 mm axial width. Both share the body/assets/sensors in `go2w_common.xml` and use zero wheel margin. In `src/assets/robots/unitree_go2w/go2w_constants.py`, change `GO2W_XML = GO2W_CYLINDER_XML` to `GO2W_XML = GO2W_MESH_XML` and restart train/play. There is no CLI switch; control and reward settings are shared.

## Training guards

All Go2W train/play resets end with terrain clearance: current joint pose and orientation, conservative geom bounds and the actual terrain envelope determine the required upward correction, with 5 mm plus collision margins. Nominal root height remains 0.4 m. Current planes, stairs, slopes, rough and wave terrains are covered; overlapping heightfield cells use all vertices. Unsupported tilted plane/heightfield configurations fail explicitly. This protects reset placement, not all subsequent dynamic motion.

Shared velocity tasks retain solver/linesearch budgets 50/100. Captured states that exhausted 10/20 converge with the current budget and match the 100/200 reference at the same tolerance. The solver still stops early. XML `cone/impratio` conflicts were resolved by using the scene's existing `pyramidal/1` values as the single source. All Go2W collision geoms have zero margin; non-wheel geoms use `condim=1, priority=0`, and wheels use `condim=3, priority=1`.

NaN capture atomically writes essential data before optional diagnostics. Empty tendon fields are skipped; optional reads, writes or latest-copy failures preserve the core archive. Keep NPZ/JSON/MJB together:

```bash
python scripts/viz_nan.py <directory>/nan_dump_latest.npz
python scripts/inspect_nan_dump.py <directory>/nan_dump_latest.npz --device cuda:0 --output logs/nan_replay.json
python scripts/inspect_nan_dump.py <directory>/nan_dump_latest.npz --summary-only
```

Replay restores recorded controls, warmstart, model randomization and solver settings, selecting a pre-failure input by default. Incomplete/legacy captures support summary/viewing rather than full recorded-input replay. Original GPU contacts and recomputed contacts are reported separately. MJB files compiled by MuJoCo 3.5/3.17 need their matching old environments.

## Validation

```bash
python -m pip check
python scripts/list_envs.py
python -m unittest discover -s tests -v
python -m mujoco_warp._src.collision_gjk_test GJKTest.test_cylinder_box_side_witness
python -m mujoco_warp._src.collision_driver_test CollisionTest.test_cylinder_box_shallow_parity0 CollisionTest.test_cylinder_box_shallow_parity1
```

Project regressions retain full-task NaN injection and save-failure handling, frozen collisions, solver limits, all six Go2W terrain/reset recipes, real backend buffer allocation, model/observation/reward checks and ONNX metadata. Temporary migration comparisons, capacity/FMA scans, four-backend matrices, EPA capture and duplicate smoke CLIs were removed. Original upstream algorithm tests live in the backend fork.

On the original RTX 4090 NaN input, the combined fixes preserve contacts and finite forward/step results; all three upstream GPU regressions pass. The contact position still differs from native MuJoCo by 1.364 mm, exceeding the old diagnostic tool's arbitrary 1 mm threshold, so its overall `passed:false` must not be described as a full pass. A separate EPA snapshot matched the 256 reference at capacities 30/64/128, but capacity 30 later overflowed in training. The user has not reported warnings in the current capacity-64 trial.

Current CPU validation passes 71 project regressions, including zero margins, both wheel models and terrain clearance. Each model completed 32 environments, 64 rollout control steps, two PPO updates, checkpoint/ONNX export, and 16 playback control steps each for checkpoint and ONNX, with finite actions, observations and rewards. Packaging includes both entry XMLs and the shared body. Earlier checks passed three upstream algorithm regressions, dependency consistency, normal-entry resume and original NaN input replay. A baseline backend environment installed the pinned fork Git commit with one `pip install -e .`.

Frozen-state results and short CPU runs do not prove long-run GPU stability; continue observing training on the target 4090. See the [Go2W guide](go2w_perceptive_zh.md) for task and perception details.
