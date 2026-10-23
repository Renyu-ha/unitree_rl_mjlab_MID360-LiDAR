# mjlab 最新主线：Conda 安装与验证

本分支为 `feat/mjlab-main-conda-3-16`。版本基线核对日期：2026-10-23。

| 组件 | 固定版本 |
| --- | --- |
| mjlab | 主线提交 `8bb2bc9e187797689ec82c78c35019afd2fd0376`，源码版本号仍为 1.6.0 |
| MuJoCo / mujoco-warp | 3.16.0 / 3.16.0，与主线 ~=3.16.0 一致 |
| Warp | 1.18.0，与该主线 uv.lock 一致，满足 >=1.15.0 |
| RSL-RL | 5.5.1，由固定 mjlab 提交声明 |
| PyTorch | 2.14.1，下面默认选择 CUDA 12.6 包，满足主线 >=2.14.0 |
| Python | 推荐 3.11；项目支持 >=3.11,<3.14 |

mjlab 源码通过标准 Python Git 依赖固定在 `pyproject.toml` 中，pip 可以直接识别。本分支使用 Conda 管理 Python 环境、pip 安装依赖，移除了之前的 uv 专用配置和 3.17 依赖覆盖。`requirements/constraints-mjlab-main.txt` 固定核心验证版本，均满足上游依赖声明；它不是所有间接依赖的完整锁文件。

上游 mjlab 自身使用 `uv_build` 作为 Python 构建后端，pip 会在隔离构建环境中自动安装它；用户无需安装或运行 uv 命令。

原 `feat/mjlab-latest-mujoco-warp-3-17` 分支保留，便于碰撞问题对照。正式训练使用本分支的 3.16 组合。

## 1. 创建 Conda 环境

以下流程创建独立 Conda 环境 `unitree_mjlab_main`：

```bash
git clone --recurse-submodules -b feat/mjlab-main-conda-3-16 https://github.com/Renyu-ha/unitree_rl_mjlab_MID360-LiDAR.git
cd unitree_rl_mjlab_MID360-LiDAR
conda env create -f environment.yml
conda activate unitree_mjlab_main
python -m pip install "torch==2.14.1" --index-url https://download.pytorch.org/whl/cu126
python -m pip install -c requirements/constraints-mjlab-main.txt --extra-index-url https://pypi.nvidia.com -e .
python -m pip check
```

已有仓库先 `git fetch origin`，再 `git switch feat/mjlab-main-conda-3-16`。也可用以下命令代替 `conda env create`，自行选择环境名：

```bash
conda create -n unitree_mjlab_main python=3.11 pip -y
```

安装需要 Git，以及访问 GitHub、PyPI、NVIDIA、Conda 和 PyTorch 软件包源。`--extra-index-url` 用于获得 NVIDIA 发布的 Warp 包。NVIDIA 驱动必须同时满足 Warp 1.18 与所选 PyTorch CUDA 包的要求；以前“驱动 >=550”的说明不能作为这套版本的支持保证。PyTorch cu126 包不改变 Warp 1.18 内部的 CUDA 运行要求。

Windows 或 Linux 仅做 CPU 检查时，将 PyTorch 安装命令改为：

```bash
python -m pip install "torch==2.14.1" --index-url https://download.pytorch.org/whl/cpu
```

其余安装命令相同。后续命令都在 `conda activate unitree_mjlab_main` 后执行。不要在现有 3.5/3.17 训练环境中直接覆盖安装。

仓库当前 ONNX 回放使用 ONNX 自带的参考执行器，无需额外安装 ONNX Runtime。普通 `python -m pip install -e .` 同样能识别固定的 mjlab Git 源和 3.16 版本范围；使用上面的 constraints 可以保持核心版本与本次验证一致。

## 2. 先检查碰撞与完整任务

在实际训练 GPU 上运行原始故障姿态回归：

```bash
python scripts/check_collision_regression.py --device cuda:0
python scripts/check_go2w_upgrade.py --device cuda:0 --num-envs 2 --steps 64 --learn-iterations 2
```

第一个检查只构建箱体与圆柱，复用 20261023_212106 转储的有限姿态和 float32 几何变换，检查接触是否有限、是否位于轮子和地形附近，并检查同一几何共同平移后的结果。它保留当时的 51.814914 mm 宽圆柱作为历史回归样本，不修改当前 45 mm 轮子模型。

第二个检查构建完整 Go2W 感知任务，执行 MID360/接触观测、动力学、部分环境重置、短程 PPO、checkpoint 保存/加载和 ONNX 导出。为了快速检查，它只缩小地形规模和 PPO 采样量；正常训练任务的奖励、KP/KD、action scale 与模型尺寸没有因此改变。

CPU 上将 `cuda:0` 改为 `cpu`。运行单元检查：

```bash
python -m unittest discover -s tests -v
```

本分支保留模型文件直接加载、显式碰撞属性、执行器摩擦默认值、命令部分重置接口及新版浏览器可视化适配。连续轮关节使用新 mjlab 的原生速度执行器。NaN 当步原始数据捕获功能也已保留。

3.16 额外适配：MuJoCo-Warp 3.16 不支持 native CCD / MultiCCD 下非零 margin 的箱体-箱体碰撞。Go2W 最终 XML 中的机身和四个大腿箱体显式设置 `margin="0"`；轮子等其他形状仍保留 1 mm margin。保留引擎默认的 native CCD 和 MultiCCD，自碰撞、几何尺寸、摩擦及控制参数不变。这个调整会使这五个箱体的接触激活距离减少 1 mm。

## 3. Go2W 训练和可视化

GPU 回归检查通过后，先用少量环境试跑，再恢复 4096 环境：

```bash
python scripts/train.py Unitree-Go2W-Perceptive-Rough-v2 \
  --env.scene.num-envs=4096 \
  --enable-nan-guard=True \
  --env.sim.nan-guard.output-dir=logs/go2w_nan_dump_316 \
  --env.observations.actor.nan-policy=error \
  --env.observations.actor.nan-check-per-term=True \
  --env.observations.critic.nan-policy=error \
  --env.observations.critic.nan-check-per-term=True \
  --agent.logger=wandb \
  --agent.wandb-project=go2w_mid360 \
  --agent.run-name=mjlab316
```

需要浏览器实时查看时，在训练命令后增加：

```bash
--viewer=viser --viewer-host=0.0.0.0 --viewer-port=8096
```

使用两块 GPU 时增加 `--gpu-ids 0 1`；`num-envs` 是每个进程的环境数量。没有 GPU 时优先使用上一节的 CPU 检查脚本。

checkpoint 回放：

```bash
python scripts/play.py Unitree-Go2W-Perceptive-Rough-v2 \
  --checkpoint-file=logs/rsl_rl/go2w_perceptive/实际运行目录/model_100.pt \
  --num-envs=1 --viewer=viser --viewer-port=8096
```

## 4. 已知验证边界

本次已实际新建独立 Conda 环境，按上述 pip 流程安装，`pip check` 通过。CPU 上通过 50 项测试（含原始错误接触姿态和带地面箱体的 Warp 模型加载）、64 步完整 Go2W 感知任务、2 轮 PPO、checkpoint 保存/加载、ONNX 导出与交互回放；实际训练入口也通过 NaN guard、Viser 和 W&B 离线日志检查。W&B 离线检查未向云端上传数据。

- 本地 CPU 验证不能代替训练机上的 CUDA 内核、多 GPU 和长期训练验证。
- 固定碰撞姿态通过，并不意味着所有可能的碰撞姿态都已无误。GPU 上应重新运行同一检查。
- MuJoCo 3.5/3.17 编译的旧 `.mjb` 应使用对应旧环境回放；3.16 环境从 XML 重新编译。
- 同名、同维度的旧权重仍需单独验证。保存格式和观测顺序必须相符；本分支不承诺跨版本继续加载旧优化器状态等同于原训练。
- 仓库原有 R1 XML 的 contact exclude 引用了不存在的 wrist body，单独编译 R1 仍会失败；旧 3.5 环境也能复现，本次升级没有修改该模型。Go2W、Go2、G1/23DOF、A2、As2、H1_2、H2 均已通过单独模型编译检查。

