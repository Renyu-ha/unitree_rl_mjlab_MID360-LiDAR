# 最新 mjlab / MuJoCo Warp 分支安装与验证

本分支为 `feat/mjlab-latest-mujoco-warp-3-17`。版本基线核对日期：2026-10-23。

| 组件 | 固定版本 |
| --- | --- |
| mjlab | 主线提交 `8bb2bc9e187797689ec82c78c35019afd2fd0376`，源码版本号仍为 1.6.0 |
| MuJoCo / mujoco-warp | 3.17.0 / 3.17.0 |
| Warp | 1.18.0 |
| RSL-RL | 5.5.1，由固定 mjlab 提交声明 |
| PyTorch | 依照 `uv.lock`，Python 3.11 环境为 2.14.1 |
| Python | 推荐 3.11；项目支持 >=3.11,<3.14 |

mjlab 该主线提交仍声明 MuJoCo/Warp ~=3.16.0。本分支在 `pyproject.toml` 中通过 **uv override-dependencies** 明确覆盖为 3.17.0，并提交 `uv.lock`。这属于本仓库的适配组合，不代表上游已经声明支持 3.17。请使用下面的 uv 安装流程；普通 `pip install -e .` 不会应用这些覆盖，也不会应用固定的 Git 源。

## 1. 建立独立环境

保留原来的 Conda/虚拟环境。以下命令在仓库内新建 `.venv`，不覆盖旧训练环境：

```bash
git clone --recurse-submodules -b feat/mjlab-latest-mujoco-warp-3-17 https://github.com/Renyu-ha/unitree_rl_mjlab_MID360-LiDAR.git
cd unitree_rl_mjlab_MID360-LiDAR
python -m pip install 'uv>=0.13.5'
uv sync --locked --python 3.11 --extra cu126
source .venv/bin/activate
```

已有仓库可先 `git fetch origin`，再切换到该远端分支。安装需要访问 GitHub、PyPI、NVIDIA 和所选 PyTorch 软件包索引。NVIDIA 驱动必须同时满足 Warp 1.18 与所选 PyTorch CUDA 包的要求；以前“驱动 >=550”的说明不能作为这套新版本的支持保证。

- `--extra cu126`：PyTorch CUDA 12.6 包。
- `--extra cu130`：PyTorch CUDA 13.0 包。
- `--extra cpu`：CPU 检查。三个选项互斥。

仓库当前 ONNX 回放使用 ONNX 自带的参考执行器，不需要额外安装 ONNX Runtime。

上面的 cu126/cu130 只选择 PyTorch 包，不改变 Warp 1.18 软件包内部的 CUDA 运行要求。

Windows CPU 检查可使用：

```powershell
uv sync --locked --python 3.11 --extra cpu
.venv\Scripts\Activate.ps1
```

后续命令都在激活的环境中执行。不要在 uv 环境里单独升级这四个仿真依赖，否则不再是锁定的验证组合。

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

本分支适配了模型文件直接加载、显式碰撞属性、执行器摩擦默认值、命令部分重置接口及新版浏览器可视化。旧连续轮关节 monkey patch 已移除，使用新 mjlab 的原生速度执行器。

## 3. Go2W 训练和可视化

GPU 回归检查通过后，先用少量环境试跑，再恢复 4096 环境：

```bash
python scripts/train.py Unitree-Go2W-Perceptive-Rough-v2 \
  --env.scene.num-envs=4096 \
  --enable-nan-guard=True \
  --env.sim.nan-guard.output-dir=logs/go2w_nan_dump_317 \
  --env.observations.actor.nan-policy=error \
  --env.observations.actor.nan-check-per-term=True \
  --env.observations.critic.nan-policy=error \
  --env.observations.critic.nan-check-per-term=True \
  --agent.logger=wandb \
  --agent.wandb-project=go2w_mid360 \
  --agent.run-name=mjlab317
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

- 本地 CPU 验证不能代替训练机上的 CUDA 内核、多 GPU 和长期训练验证。
- 固定碰撞姿态通过，并不意味着所有可能的碰撞姿态都已无误；原有扩展样本中新版仍出现少量浅接触漏检。
- MuJoCo 3.5 编译的旧 `.mjb` 不能直接由 3.17 加载，需保留旧环境回放；新版环境从 XML 重新编译。
- 同名、同维度的旧权重仍需单独验证。保存格式和观测顺序必须相符；本分支不承诺跨版本继续加载旧优化器状态等同于原训练。
- 仓库原有 R1 XML 的 contact exclude 引用了不存在的 wrist body，单独编译 R1 仍会失败；旧 3.5 环境也能复现，本次升级没有修改该模型。Go2W、Go2、G1/23DOF、A2、As2、H1_2、H2 均已通过单独模型编译检查。

