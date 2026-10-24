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

