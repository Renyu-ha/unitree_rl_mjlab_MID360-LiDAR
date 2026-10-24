# mjlab 最新主线：Conda 安装与验证

本分支为 `feat/mjlab-main-conda-3-16`。版本基线核对日期：2026-10-23。

| 组件 | 固定版本 |
| --- | --- |
| mjlab | 主线提交 `8bb2bc9e187797689ec82c78c35019afd2fd0376`，源码版本号仍为 1.6.0 |
| MuJoCo / mujoco-warp | 3.16.0 / 3.16.0，与主线 ~=3.16.0 一致 |
| Warp | 1.18.0，与该主线 uv.lock 一致，满足 >=1.15.0 |
| RSL-RL | 5.5.1，由固定 mjlab 提交声明 |
| PyTorch | 2.14.1，从 PyPI 安装适合当前平台的默认包，满足主线 >=2.14.0 |
| Python | 推荐 3.11；项目支持 >=3.11,<3.14 |

mjlab 源码及 MuJoCo、mujoco-warp、Warp、RSL-RL、PyTorch 的核心版本直接固定在 `pyproject.toml` 的标准依赖中，均满足上游声明。激活 Python 环境后，在仓库根目录执行 `pip install -e .` 即可安装项目及全部 Python 依赖，无需预装 PyTorch、额外包源、constraints 文件或 uv 命令。间接依赖仍由 pip 解析，核心版本固定不等于所有间接依赖的完整锁文件。

上游 mjlab 自身使用 `uv_build` 作为 Python 构建后端，pip 会在隔离构建环境中自动安装它；用户无需安装或运行 uv 命令。

原 `feat/mjlab-latest-mujoco-warp-3-17` 分支保留，便于碰撞问题对照。正式训练使用本分支的 3.16 组合。

## 1. 创建 Conda 环境

以下流程创建独立 Conda 环境 `unitree_mjlab_main`：

```bash
git clone --recurse-submodules -b feat/mjlab-main-conda-3-16 https://github.com/Renyu-ha/unitree_rl_mjlab_MID360-LiDAR.git
cd unitree_rl_mjlab_MID360-LiDAR
conda env create -f environment.yml
conda activate unitree_mjlab_main
pip install -e .
```

已有仓库先 `git fetch origin`，再 `git switch feat/mjlab-main-conda-3-16`。也可用以下命令代替 `conda env create`，自行选择环境名：

```bash
conda create -n unitree_mjlab_main python=3.11 pip -y
```

安装需要 Git，以及访问 GitHub 和 PyPI；创建 Conda 环境另需访问 Conda 包源。Warp 1.18.0 和 PyTorch 2.14.1 均直接从 PyPI 安装，无需 NVIDIA 额外包源。默认使用当前平台的 PyPI 包，不再强制 PyTorch cu126；环境内已有的同版本 CPU/CUDA 包可满足版本约束。NVIDIA 驱动仍须同时满足实际安装的 Warp 和 PyTorch 包的要求。

安装完成后可选检查依赖一致性：

```bash
python -m pip check
```

CPU 检查和 GPU 训练使用相同的项目安装命令。后续命令都在 `conda activate unitree_mjlab_main` 后执行；旧版 3.5/3.17 的转储应保留对应旧环境回放。

仓库当前 ONNX 回放使用 ONNX 自带的参考执行器，无需额外安装 ONNX Runtime。

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

本分支保留模型文件直接加载、显式碰撞属性、执行器摩擦默认值、命令部分重置接口及新版浏览器可视化适配。连续轮关节使用新 mjlab 的原生速度执行器。NaN 当步原始数据先以原子写入保存核心转储，再补充接触、约束和随机化模型字段；空 tendon 字段会跳过，附加读取或保存失败不会覆盖核心转储和原始 NaN 异常。

3.16 额外适配：MuJoCo-Warp 3.16 不支持 native CCD / MultiCCD 下非零 margin 的箱体-箱体碰撞。当前 Go2W 已进一步对齐 G1：所有非轮子碰撞体使用 `margin="0"`、`condim=1` 和基础滑动摩擦 `1`，轮子保留 1 mm margin 并使用 `condim=3, priority=1`。保留引擎默认的 native CCD 和 MultiCCD；具体接触调整见 [Go2W 模型来源](../src/assets/robots/unitree_go2w/SOURCE.md)。

### 求解器收敛与全局参数冲突

对比本机固定版本源码，旧 MuJoCo-Warp 3.5 没有这两类迭代上限告警，
3.16 会在达到预算而未满足停止条件时设置诊断位并输出提示。因此，
旧日志没有提示不能证明旧版从未耗尽预算；新版告警也不能直接等同于 NaN。

共享速度任务的主求解器/线搜索预算从 `10/20` 调整为 `50/100`，对应
`iterations` 和 `ls_iterations`。保持原来的收敛容差、接触模型、摩擦、
奖励、观测和所有溢出警告。求解收敛后仍提前结束，预算不是每一步固定
执行次数。Go2W 与对应 G1 任务继续使用相同的共享仿真设置。

2026-10-24 在完整 10×6 地形、32 环境、256 步的 CPU 随机动作检查中，
每种设置统计了 40,992 次逐环境物理调用（包括重置 forward）：

| 动作标准差 | 原 10/20：线搜索/主求解器上限次数 | 新 50/100 |
| --- | --- | --- |
| 1 | 10 / 0 | 0 / 0 |
| 2（扩大扰动） | 5 / 1 | 0 / 0 |

共保存并回放 15 个触发状态；新预算的加速度与同容差的 `100/200`
参考结果一致。触发主求解器上限的状态实际需要 11 次迭代，原 10 次
预算截断了求解。这些结果支持当前修复，但 CPU 固定状态和有限随机
测试不能保证任意 GPU 状态或整个训练过程都不再触及预算。
修复后 58 项 CPU 测试、64 步完整任务检查、两次 PPO 迭代、checkpoint
保存/加载及 ONNX 导出均通过；测试包含两种原始上限状态的回放。

在实际训练 GPU 上检查，不需要 W&B 或启动 PPO：

```bash
python scripts/check_solver_convergence.py --device cuda:0 --num-envs 4096 --steps 256 --output logs/go2w_solver_check.json
```

脚本保留原始警告，逐次读取真实 `overflow` 位；出现任一求解上限或
容量溢出时返回非零退出码。报告范围是环境构建后的 reset 和 rollout，
构建时 CUDA graph capture 的提示仍应在终端检查。该检查会同步设备
读取诊断数据，其耗时不能作为训练吞吐。`--iterations 10 --ls-iterations 20`
可用于同设备对照；`--action-std 2` 可扩大动作扰动。本次预算改变可能
让首次运行重新编译部分求解内核，后续可复用 Warp 缓存。

`cone/impratio Attach conflict` 的来源是机器人 XML 原声明
`cone="elliptic", impratio="100"`，而场景使用 `pyramidal, 1`。
`cone` 选择摩擦锥模型，`impratio` 控制切向与法向接触阻抗之比。
这些是全场景参数，MuJoCo 挂接机器人时保留父场景值。已移除 XML 中
这处重复全局声明，由任务的 `MujocoCfg` 统一管理；训练实际使用的
`pyramidal, 1` 不变。单独加载 XML 时也将使用默认的 `pyramidal, 1`。

参数含义参见 [MuJoCo option 文档](https://mujoco.readthedocs.io/en/stable/XMLreference.html#option)。

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

2026-10-24 在全新 Windows Python 3.11 虚拟环境中，仅用 `pip install -e .` 完成安装，未预装 PyTorch、未使用额外包源或 constraints。该环境通过 `pip check`、56 项测试（包含完整 Go2W 任务注入 NaN、空 tendon 字段、附加读取/写入失败时保留故障转储）、64 步完整 Go2W 感知任务、2 轮 PPO、checkpoint 保存/加载和 ONNX 导出。原独立 Conda CPU 环境也通过相同的 56 项测试。

此前独立 Conda 环境已通过交互回放，以及实际训练入口的 NaN guard、Viser 和 W&B 离线日志检查；W&B 离线检查未向云端上传数据。

- 本地 CPU 验证不能代替训练机上的 CUDA 内核、多 GPU 和长期训练验证。
- 固定碰撞姿态通过，并不意味着所有可能的碰撞姿态都已无误。GPU 上应重新运行同一检查。
- MuJoCo 3.5/3.17 编译的旧 `.mjb` 应使用对应旧环境回放；3.16 环境从 XML 重新编译。
- 同名、同维度的旧权重仍需单独验证。保存格式和观测顺序必须相符；本分支不承诺跨版本继续加载旧优化器状态等同于原训练。
- 仓库原有 R1 XML 的 contact exclude 引用了不存在的 wrist body，单独编译 R1 仍会失败；旧 3.5 环境也能复现，本次升级没有修改该模型。Go2W、Go2、G1/23DOF、A2、As2、H1_2、H2 均已通过单独模型编译检查。

