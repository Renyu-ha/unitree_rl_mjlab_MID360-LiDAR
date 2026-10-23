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
python -m pip install "torch==2.14.1" --index-url https://download.pytorch.org/whl/cu126
python -m pip install -c requirements/constraints-mjlab-main.txt --extra-index-url https://pypi.nvidia.com -e .
python -m pip check
```

For CPU validation, replace the PyTorch index with
`https://download.pytorch.org/whl/cpu`. The constraints file pins the core
validation versions, rather than every transitive package. The standard Git
dependency in `pyproject.toml` pins mjlab for pip as well. Upstream's `uv_build`
build backend is installed automatically by pip in an isolated build environment.

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

The independent Conda CPU environment passed `pip check`, 50 tests, a 64-step
Go2W perceptive rollout, two PPO iterations, checkpoint save/load, ONNX export
and interactive playback. The normal training entry also passed with NaN guard,
Viser and offline W&B logging. Go2W's five box collision geoms use zero margin
for MuJoCo-Warp 3.16 native CCD support; wheel margins remain 1 mm. GPU and
long-run stability still require validation on the training machine.

