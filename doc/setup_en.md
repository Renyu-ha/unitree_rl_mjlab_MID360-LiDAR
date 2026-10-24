# Installation: mjlab Main with Conda

Branch: `feat/mjlab-main-conda-3-16`. Upstream checked on 2026-10-23:
mjlab commit `8bb2bc9e187797689ec82c78c35019afd2fd0376`, MuJoCo and
mujoco-warp 3.16.0, Warp 1.18.0, RSL-RL 5.5.1 and PyTorch 2.14.1.
These versions satisfy upstream dependency requirements. No dependency override
or uv command is needed. The mjlab source reports package version 1.6.0;
the Git commit identifies the exact main-branch source.

Use Ubuntu 22.04 and an NVIDIA GPU for training. The driver must support both
Warp 1.18 and the chosen PyTorch CUDA package. CPU validation also works on
Windows. Keep the previous training environment for old model dumps.

```bash
git clone --recurse-submodules -b feat/mjlab-main-conda-3-16 https://github.com/Renyu-ha/unitree_rl_mjlab_MID360-LiDAR.git
cd unitree_rl_mjlab_MID360-LiDAR
conda env create -f environment.yml
conda activate unitree_mjlab_main
pip install -e .
```

This single command installs PyTorch and all Python dependencies. The standard
metadata in `pyproject.toml` pins the core versions and the exact mjlab Git
commit. No constraints file, extra package index or separate PyTorch install is
needed. PyTorch and Warp use the default PyPI wheels for the current platform;
an already installed CPU/CUDA build of the same PyTorch version also satisfies
the requirement. Transitive dependencies are resolved by pip. Upstream's
`uv_build` backend is installed automatically in an isolated build environment.
Git and access to GitHub and PyPI are required. After installation, optionally
run `python -m pip check`.

Run the following on the training GPU before starting a long run:

```bash
python scripts/check_collision_regression.py --device cuda:0
python scripts/check_go2w_upgrade.py --device cuda:0 --num-envs 2 --steps 64 --learn-iterations 2
python -m unittest discover -s tests -v
```

Replace `cuda:0` with `cpu` for local CPU checks. These checks do not establish
long-run training stability or multi-GPU correctness. See the
[Chinese guide](setup_zh.md) for training, visualization, NaN capture and model
compatibility details. The 3.17 comparison branch remains available as
`feat/mjlab-latest-mujoco-warp-3-17`.

On 2026-10-24 a fresh Windows Python 3.11 environment installed successfully
with only `pip install -e .`, without preinstalling PyTorch or supplying extra
indexes or constraints. It passed `pip check`, all 56 tests, a 64-step Go2W
perceptive rollout, two PPO iterations, checkpoint save/load and ONNX export.
The tests include a full Go2W NaN injection and preservation of the core dump
when tendon fields are empty or optional diagnostics fail. The existing Conda
CPU environment also passed all 56 tests. Earlier checks covered interactive
playback and the normal training entry with NaN guard, Viser and offline W&B
logging. All Go2W non-wheel collision geoms now use zero
margin, condim 1 and base sliding friction 1, matching G1; wheels retain 1 mm
margin and use condim 3, priority 1. NaN capture saves the core archive before
optional diagnostics, skips empty tendon fields and preserves the original
physics error if optional reads or writes fail. GPU and long-run stability
still require validation on the training machine.

The pinned MuJoCo-Warp 3.5 source did not emit these two iteration-limit warnings;
3.16 explicitly reports a budget reached before its stopping criteria are met.
An old log without warnings therefore does not establish convergence, and a
new limit warning does not by itself mean NaN.

Shared velocity tasks now use solver/linesearch iteration budgets of 50/100
instead of 10/20. Convergence tolerances, contact settings and overflow warnings
remain unchanged; convergence still stops iterations early. On the full 10-by-6
Go2W terrain, CPU checks with 32 environments and 256 Gaussian-action steps
recorded 40,992 per-world physics calls per run, including reset forwards:
These budget-comparison records predate the reset-clearance guard; the fixed
state regression preserves the original inputs.

| Action standard deviation | Old 10/20: linesearch / solver limits | New 50/100 |
| --- | --- | --- |
| 1 | 10 / 0 | 0 / 0 |
| 2 | 5 / 1 | 0 / 0 |

Replaying 15 captured states with 50/100 produced the same accelerations as a
100/200 reference at the same tolerance. The captured solver-limit case needed
11 iterations. Two small fixed-state fixtures now reproduce the original
failures in the test suite; all 58 tests pass on CPU. The updated configuration
also passes a 64-step rollout, two PPO iterations, checkpoint save/load and ONNX
export in the plain-pip environment. These finite CPU checks
do not establish convergence for every GPU state or an entire training run.
Check the actual training device with:

```bash
python scripts/check_solver_convergence.py --device cuda:0 --num-envs 4096 --steps 256 --output logs/go2w_solver_check.json
```

This diagnostic checks real overflow bits after every reset/rollout physics
call and exits nonzero on any limit or capacity overflow. Inspect environment
construction / CUDA graph-capture messages separately. Per-call device reads
synchronize execution, so diagnostic time is not training throughput. Add
`--iterations 10 --ls-iterations 20` for an old-budget comparison or
`--action-std 2` for larger perturbations. Changed budgets may require a first
compilation of some solver kernels before the Warp cache is reused.

The robot XML formerly declared global `cone="elliptic", impratio="100"`,
conflicting with the scene's `pyramidal, 1`. These options apply to the entire
scene, and attachment retains the parent values. The duplicate XML declaration
is removed, leaving the task's `MujocoCfg` as the sole owner. Effective training
values remain `pyramidal, 1`; standalone XML loading now also uses those defaults.
See [MuJoCo option documentation](https://mujoco.readthedocs.io/en/stable/XMLreference.html#option).

All Go2W variants now run `ensure_spawn_clearance` as their last reset event.
Nominal height remains 0.4 m; only reset worlds needing additional clearance
are lifted. The guard checks conservative robot geom bounds against terrain
tops and every overlapping heightfield cell's vertices, retaining 5 mm plus
collision margins. It supports the current plane, box/stair and upright
heightfield terrains; tilted planes/heightfields fail explicitly. It protects
reset placement, not all subsequent dynamic motion or self collision. Play-mode
terrain relocation now precedes root reset so placement uses the new origin.

```bash
python scripts/check_spawn_clearance.py --device cuda:0 --output logs/spawn_clearance_check.json
```

This checks every configured terrain type and difficulty row, comparing frozen
0.4/0.5/0.6 m resets with Warp and native MuJoCo collisions. CPU V2 checks confirmed
native initial penetration in 16/40 rough and 4/40 wave samples at 0.4 m; the guard
cleared both backends' penetration checks. All six Go2W task recipes passed,
covering 3,456 reset poses. All 66 CPU tests, a 64-step rollout, two PPO iterations,
checkpoint save/load, ONNX export and the normal training entry with both NaN/EPA
capture passed.
This did not reproduce the reported EPA=24 training state, so its cause remains
unconfirmed. Diagnostic timing is not training throughput.

Append `--capture-epa=True` to a normal training command to save the first EPA overflow
and stop that diagnostic run, without waiting for NaN. Capture checks synchronize
the device. Files go under the run's `epa_capture/rank_0/`; keep the NPZ and its
referenced MJB together. Constructor graph captures precede installation and
must be inspected separately. Replay the frozen collision on the original device:

```bash
python scripts/replay_epa_dump.py <run>/epa_capture/rank_0/epa_dump_latest.npz --device cuda:0
```

Replay must reproduce the original overflow before testing 64/128 against 256.
A passing candidate produces nonempty contacts with finite, plausible positions/normals,
zero overflow and agreement with the larger-capacity reference. This is evidence
for one frozen state, not independent ground truth or long-run stability. The
contact count need not increase, and `recovered_geom_pairs` may be empty. The
reported GPU state passed these checks at 30/64/128: all seven contacts exactly
matched the 256 reference. The earlier requirement for an increased contact count
incorrectly reported `passed:false` and has been removed. The
temporary private allocation override affects only replay, not installed files
or training, and leaves warnings enabled. The adapter requires mujoco-warp 3.16.0
and captured package versions. A controlled 5-edge overflow tests the capture
chain; it does not reproduce the user's reported 24-edge GPU case.

The separate `nan_dump_20261024_120617_484587` is world 1618, not the EPA
snapshot's world 3569. Its GPU contact between the front-right wheel and box
`terrain_196` is 61.87 m from the wheel center (bounding radius 0.0889 m), despite
a recorded distance of only -0.0000188 m. Restoring the last finite input, saved
controls, warmstart and randomized fields produces a valid wheel-side contact
and finite results in both native MuJoCo and CPU Warp. Capacities 24/30/64 agree
on CPU. The local machine has no CUDA driver, so neither the internal cause nor
a fix through horizon capacity is established. Probe it on
the training GPU with the MJB beside the NPZ:

```bash
python scripts/inspect_nan_dump.py <directory>/nan_dump_20261024_120617_484587.npz \
  --recorded-inputs --device cuda:0 --output logs/nan_replay.json
```

This selects the last pre-operation input and preserves the recorded solver,
controls, warmstart and randomized fields, rejecting incomplete captures. A
single-world replay with recomputed transforms is not an exact GPU checkpoint.
Subsequent NaN captures also save the original device `failure_geom_xpos`,
`failure_geom_xmat` and `failure_overflow` as optional extras; extraction failures
still preserve the core archive.
The verdict fix and recorded-input probe passed all 68 CPU tests.

The user subsequently reproduced the exact bad contact before the solver on the
original RTX 4090 with driver 580.173.02 and CUDA Toolkit 13.4/Driver 13.0. Both
forward acceleration and step velocity become nonfinite. Reproduction needs no
PPO, multiple worlds or training CUDA graph. This localizes the error to the
GPU cylinder/box collision path but does not establish a compiler/driver defect.

Two independent diagnostic controls were run with the environment fixed:

```bash
python scripts/inspect_nan_dump.py logs/go2w_nan_dump/nan_dump_20261024_120617_484587.npz \
  --recorded-inputs --device cuda:0 --epa-horizon 64 --collision-fma on \
  --output logs/nan_replay/horizon64.json

python scripts/inspect_nan_dump.py logs/go2w_nan_dump/nan_dump_20261024_120617_484587.npz \
  --recorded-inputs --device cuda:0 --epa-horizon 24 --collision-fma off \
  --output logs/nan_replay/fma_off.json
```

The original defaults are horizon 24 with FMA enabled. FMA contraction changes
rounding; sensitivity to it is not proof that FMA or CUDA is defective. The probe
sets and records `fuse_fp` on the actual dynamically built CCD kernel modules,
not merely the source function's module. It rejects an experiment if no CCD
kernel options were applied. All in-memory overrides are restored on exit,
including exceptions; installed files, training defaults and warnings are unchanged.
Judge contact sanity/position as well as finite forward/step outputs. Finite
solver output with an implausible contact is not a fix. The user's subsequent
`horizon64.json` retained the same 61.87 m bad contact and NaN. `fma_off.json`
was finite but lost all contacts. Neither fixes this NaN; rounding sensitivity
does not establish a CUDA defect. Upstream collision fixes are the next check.
All 71 CPU tests passed, including real CCD option application and restoration
after exceptions. Both CLI controls also completed against the uploaded NaN inputs on CPU.

### Pinned upstream collision candidates

[Newton #4573](https://github.com/newton-physics/newton/issues/4573) reports the
same class of cylinder/box far-witness contact followed by solver NaN. Its
maintainer points to [MuJoCo-Warp #1771](https://github.com/google-deepmind/mujoco_warp/pull/1771),
which selects an EPA face containing the witness projection rather than
extrapolating outside it. [#1796](https://github.com/google-deepmind/mujoco_warp/pull/1796)
excludes sliver faces with near-repeated vertices from nearest-face selection.
Both PRs were open and unmerged when checked on 2026-10-24. Exact pinned upstream
diffs and regression tests are bundled; no local collision formula is added.

After updating the repository, run one command in the original 4090 environment,
keeping the original MJB beside the NPZ:

```bash
python scripts/check_upstream_collision_fix.py \
  --dump logs/go2w_nan_dump/nan_dump_20261024_120617_484587.npz --device cuda:0
```

The tool creates four private engine copies: baseline, 1771, 1796 and both.
Fresh processes verify their actual source paths and checksums. FMA stays on and
horizon stays 24; installed packages and normal training are unchanged, with no
reinstallation required. Results go to
`logs/upstream_collision_check/<timestamp>/summary.json`, with per-variant replay
and test logs. First-time compilation can take several minutes.

A candidate requires reproduction on baseline, finite forward/step physics,
zero diagnostic flags, nonempty sane contacts matching native MuJoCo in pairs,
counts, position (1 mm), distance (50 micrometers) and normal angle (5 degrees),
and all three upstream regressions passing. The selected candidate must then
pass the project regression suite and a 32-world, 64-step Go2W rollout with two
PPO iterations. Only then is overall `passed:true`. If baseline does not
reproduce, status is `baseline_not_reproduced_or_unverified` with a nonzero exit;
this does not establish a GPU fix.

On local CPU, baseline passed 0/3 upstream cases, 1771 and 1796 each passed 2/3,
and both passed 3/3. All four versions had valid finite contacts for the user's
original input on CPU, so that comparison remains inconclusive. The combined
engine separately passed all 76 CPU project tests and the short Go2W/PPO check.
Original-GPU validation and longer task checks remain necessary. These bounded
checks do not prove all-terrain or long-run stability. Only after verification
should a reviewed source revision become the pinned production dependency,
preserving `pip install -e .`. The tool does not promote candidates automatically
or claim to resolve the separate EPA-capacity overflow snapshot.

### Training with the tested combined copy and an explicit EPA capacity

The original 4090 report `20261024_135019_915918` reproduced NaN and the far
contact on baseline and 1796. With 1771 or both patches, contacts were retained,
physics stayed finite and all three upstream GPU cases passed. The contact
position still differed from native MuJoCo by 1.364 mm, exceeding the diagnostic
tool's chosen 1 mm threshold. Overall remained `passed:false`, so GPU project
regressions and the smoke run were not automatically executed. The user chose
to proceed with a training trial while retaining this precision discrepancy.

The patches do not increase EPA capacity. After updating the repository, reuse
the tested `both` copy and select capacity 30 explicitly:

```bash
GO2W_PATCH_DIR="$PWD/logs/upstream_collision_check/20261024_135019_915918/both"
GO2W_PATCH_CACHE="$PWD/logs/upstream_collision_check/20261024_135019_915918/warp_cache"

PYTHONPATH="$GO2W_PATCH_DIR:$PWD" WARP_CACHE_PATH="$GO2W_PATCH_CACHE" \
python scripts/train.py Unitree-Go2W-Perceptive-Rough-v2 \
  --env.scene.num-envs=4096 --epa-horizon=30 --enable-nan-guard=True \
  --env.sim.nan-guard.output-dir=logs/go2w_nan_dump_upstream_both_h30 \
  --env.observations.actor.nan-policy=error \
  --env.observations.critic.nan-policy=error \
  --agent.run-name=v2_upstream_1771_1796_h30
```

Each worker selects capacity before environment creation, buffer allocation and
CUDA graph capture, retaining it through reset, rollout and close. Restart the
training process for the option to take effect. Startup `Collision settings`
prints the actual engine path, collision-source checksum and `epa_horizon: 30`;
these are also saved in the run's `params/collision.json` after checkpoint
selection, so creating a new log directory cannot disturb default resume lookup.
Omitting the option retains the engine's default 24. Installed files, FMA and
overflow warnings are unchanged; installation remains `pip install -e .`.

All 77 local CPU project tests passed, including actual EPA buffer allocation
and restoration after a training error. The combined engine with capacity 30
also passed the normal training entry with 32 worlds, two PPO iterations,
checkpoint saving and resumed training. GPU capacity coverage remains to be observed.

The previously captured EPA scene passed at 30/64/128 against the seven-contact
256 reference on the original engine. That does not prove 30 covers subsequent
combined-engine training states. Add `--capture-epa=True` to save and stop on
the first overflow, then replay the new scene under the same combined copy.
To resume, retain `--agent.resume=True` and specify `--agent.load-run` and
`--agent.load-checkpoint` explicitly.

