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
以下是加入出生净空保护之前的预算对照记录；固定状态回归仍保留原输入。

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

### 出生/重置净空与 EPA 故障现场

Go2W 名义初始高度仍为 `0.4 m`。所有 Go2W train/play 任务在每次 reset
末尾执行 `ensure_spawn_clearance`：先用当前关节姿态和朝向计算运动学，
再对每个有效机器人碰撞体的保守包围盒检查实际地形，仅抬高需要调整的
环境。箱体等有限地形取重叠区域的包围盒顶面；高度场检查与机器人包围盒
重叠的全部网格单元顶点，覆盖脚旁的尖峰，不依赖稀疏射线。净空保留
5 mm，另计碰撞 margin；实际几何间隙可能因保守包围盒而更大。
水平地面、台阶、正反斜坡、随机粗糙和波浪地形均适用；倾斜的 plane/
heightfield 会明确报错。这个保护针对出生/重置，不能保证后续任意高速
运动都不产生动态穿透，也不能替代 EPA 算法的故障诊断。

play 模式已将 `randomize_terrain` 调整到根状态重置之前，保证机器人使用
新地形原点；净空检查最后执行。默认关节角、随机化范围、奖励和控制参数
不变；原本贴地或穿入地形的出生位置可能升高。

检查所有地形类型和每个难度行，在相同冻结姿态下比较 `0.4/0.5/0.6 m`：

```bash
python scripts/check_spawn_clearance.py --device cuda:0 --output logs/spawn_clearance_check.json
```

每类地形/难度行默认采样两次、重复两轮 reset。脚本同时检查 MuJoCo-Warp
与原生 MuJoCo 的接触距离、EPA 位及剩余净空需求。2026-10-24 CPU 的 V2
检查中，原生 MuJoCo 确认原 `0.4 m` 在随机粗糙地形 16/40 个样本、波浪
地形 4/40 个样本存在初始负接触距离；保护后两个后端均为 0。六个 Go2W
任务配方的覆盖检查均通过，还检查了离开中心平台的阶梯/斜坡位置和部分
环境 reset 的隔离，共覆盖 3,456 个出生姿态。修复后 66 项 CPU 测试、
64 步任务检查、两次 PPO、保存/加载与 ONNX 导出通过；正常训练入口也
通过了同时启用 NaN/EPA 捕获的短程检查。没有在这些出生样本中复现用户训练时的 EPA=24，
因此不能据此认定那次告警的根因。检查耗时包含同步和 CPU 交叉验证，
不能用作训练性能比较。

如果正常训练仍出现 EPA 告警，在原训练命令末尾增加 `--capture-epa=True`。
它逐次检查 step/forward 的诊断位，在首次 EPA 告警后保存并停止本次诊断
运行，文件位于该次训练日志的 `epa_capture/rank_0/`。不要求同时发生 NaN。
记录实际 float32 几何位置/旋转、非空随机化字段、原始接触及 MJB；核心
NPZ 先保存，附加保存失败会保留 core_saved/partial 状态。环境构建期间的
graph capture 早于安装，此阶段告警应另行检查。该开关会同步 GPU，
诊断完成后去掉即可。

```bash
python scripts/replay_epa_dump.py <训练日志>/epa_capture/rank_0/epa_dump_latest.npz --device cuda:0
```

保持 NPZ 与其 metadata 指定的 MJB 在同一目录。回放不运行 PPO，也不积分，
直接恢复同一碰撞现场；先验证原容量 24 是否重现溢出，再比较 64、128 与
256 参考容量。只有原错误重现、更大容量产生非空接触、诊断位清零、接触点/
深度/法线合理且与更高容量一致，才报告该冻结现场通过。报告也会列出
新增接触的碰撞体名称，便于定位具体部位。接触点数量不必增加，
`recovered_geom_pairs` 为空也不代表失败；高度场局部查询溢出后，最终汇总
的接触点数量仍可能相同。用户在 GPU 回放中确认容量 30/64/128 均消除
24 的溢出，且七个接触点与 256 的参考完全一致。旧回放脚本要求接触点
数量增加，错误地报告了 `passed:false`，此判定已修正。单个现场通过
不是整个训练稳定性的证明。CPU 不复现 GPU 故障时报告未通过，需在原 GPU
回放。容量修改只在诊断进程内临时生效，不改已安装包或正常训练容量，
也不屏蔽警告。回放适配器限定 mujoco-warp 3.16.0，并要求包版本与现场一致。
测试中使用 5 条边的受控缓冲溢出来验证保存/回放链路，不冒充用户的 24
条边故障复现。

#### 2026-10-24 的另一份 NaN 现场

`nan_dump_20261024_120617_484587` 记录的是环境 1618，与上述 12:11:59、
环境 3569 的 EPA 现场不同。其原始 GPU 接触为右前轮
`robot/FR_wheel_link_collision` 对箱体 `terrain_196`：轮心约
`(-35.7632, -7.7279, -0.0764)`，接触点却在
`(-37.5669, -69.5721, -0.1807)`，距离轮心约 61.87 m；轮子包围半径
仅 0.0889 m。记录的接触距离约 -0.0000188 m。接触位置明显异常，
不能仅依据这个很小的负接触距离判定碰撞结果正常。

恢复最后一次有限输入（006309）、原始控制量、warmstart 和随机化字段后，
本机 CPU 的原生 MuJoCo 与 Warp 均生成轮边的接触点
（约 `(-35.7330, -7.7287, -0.1600)`），计算保持有限；Warp 容量
24/30/64 的结果相同。尚未在本机复现原 GPU 的错误接触位置，因此
不能断言扩大 EPA 容量解决了这个 NaN，也不能确认 GPU 内部的具体根因。

在训练 GPU 上使用原始输入进行单环境核对：

```bash
python scripts/inspect_nan_dump.py <目录>/nan_dump_20261024_120617_484587.npz \
  --recorded-inputs --device cuda:0 --output logs/nan_replay.json
```

MJB 与 NPZ 保持同目录。该模式默认选择最后一次操作前的输入，保留原求解器，
不使用默认探测模式的人工保持控制量；不完整转储会被拒绝。原始 GPU 接触与
重新计算的接触会分开报告。单环境、重新计算的几何旋转仍可能与原训练不同，
未复现不代表原错误不存在。后续 NaN 转储增加 `failure_geom_xpos`、
`failure_geom_xmat` 和 `failure_overflow`，保存碰撞使用的原始变换和溢出位；
这些附加字段提取失败仍不会丢失已经保存的核心转储。
本次回放判定修复与原始输入检查通过全部 68 项 CPU 测试。

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

