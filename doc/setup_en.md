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
A passing candidate restores contacts with finite, plausible positions/normals,
zero overflow and agreement with the larger-capacity reference. This is evidence
for one frozen state, not independent ground truth or long-run stability. The
temporary private allocation override affects only replay, not installed files
or training, and leaves warnings enabled. The adapter requires mujoco-warp 3.16.0
and captured package versions. A controlled 5-edge overflow tests the capture
chain; it does not reproduce the user's uncaptured 24-edge GPU case.

