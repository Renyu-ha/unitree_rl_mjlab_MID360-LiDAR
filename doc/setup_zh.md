# 安装与训练

## 固定环境

激活 Python 环境后，在仓库根目录执行一条命令：

```bash
pip install -e .
```

项目通过标准 Python 依赖安装 mjlab 和固定的 MuJoCo-Warp 后端，不再使用日志目录中的引擎副本、运行时源码补丁或容量开关。需要 Git、GitHub/PyPI 网络访问；推荐 Python 3.11，支持范围为 3.11–3.13。

| 组件 | 固定版本 |
| --- | --- |
| mjlab | `8bb2bc9e187797689ec82c78c35019afd2fd0376`，包版本 1.6.0 |
| MuJoCo | 3.16.0 |
| MuJoCo-Warp | 3.16.0.post1，派生仓库提交 `6dea56edf221906310a90896e3ad913ad12459c3` |
| Warp | 1.18.0 |
| RSL-RL | 5.5.1 |
| PyTorch | 2.14.1 |

`pyproject.toml` 固定核心版本；间接依赖由 pip 解析。PyTorch 使用当前平台的默认 PyPI 包，不需要额外包源、constraints 文件或 uv 命令。NVIDIA 驱动需要支持实际安装的 PyTorch/Warp 包。

新建 Conda 环境可执行：

```bash
git clone --recurse-submodules -b feat/mjlab-main-conda-3-16 https://github.com/Renyu-ha/unitree_rl_mjlab_MID360-LiDAR.git
cd unitree_rl_mjlab_MID360-LiDAR
conda env create -f environment.yml
conda activate unitree_mjlab_main
pip install -e .
```

已有环境在更新当前分支后重新执行 `pip install -e .`，将替换原版后端。移除旧训练命令中的碰撞实验 `PYTHONPATH` 前缀和 `--epa-horizon`、`--capture-epa` 参数；这些临时接口已删除。训练、play 和 NaN 原始输入回放会检查实际后端版本和容量，并打印加载路径；训练同时写入 `params/collision.json`。

## 固定后端的内容

后端基于官方 MuJoCo-Warp v3.16.0，只回移两项上游修复，并固定 `MJ_MAX_EPAHORIZON=64`：

- [#1771](https://github.com/google-deepmind/mujoco_warp/pull/1771)：限制 EPA 接触点外插，避免产生远离碰撞体的接触点。
- [#1796](https://github.com/google-deepmind/mujoco_warp/pull/1796)：避免近重复顶点组成的退化三角面参与最近面选择。

源码差异、固定来源提交及上游回归用例保存在 [后端 BACKPORTS.md](https://github.com/Renyu-ha/mujoco_warp/blob/6dea56edf221906310a90896e3ad913ad12459c3/BACKPORTS.md)。`3.16.0.post1` 满足 mjlab 的 `~=3.16.0` 约束。安装时正常构建后端，不修改其他已安装包。

64 是 EPA 边界缓冲容量，不是精度或迭代次数。容量足够时不会要求每次执行 64 次计算；增大容量会增加缓冲区内存。保留默认 FMA、收敛容差、CCD/MultiCCD 和所有溢出警告。它处理 EPA 容量不足；两项算法修复处理异常接触，两者用途不同。

## 正常训练、续训与回放

```bash
python scripts/train.py Unitree-Go2W-Perceptive-Rough-v2 \
  --env.scene.num-envs=4096 \
  --enable-nan-guard=True \
  --env.sim.nan-guard.output-dir=logs/go2w_nan_dump \
  --env.observations.actor.nan-policy=error \
  --env.observations.critic.nan-policy=error
```

默认使用 GPU 0、本地 TensorBoard，日志位于 `logs/rsl_rl/go2w_perceptive/`。选择显卡可用 `--gpu-ids 0`；CPU 检查用 `--gpu-ids=None`。首次安装或更换后端后的首次运行可能编译 Warp 内核，后续复用缓存。

需要 W&B 时追加 `--agent.logger=wandb --agent.wandb-project=go2w_mid360`，提前按自己的账号登录。实时查看追加 `--viewer=viser --viewer-host=0.0.0.0 --viewer-port=8096`，浏览器访问训练机的 8096 端口。可视化只同步选定环境，有额外同步开销。

续训显式选择原运行目录和 checkpoint：

```bash
python scripts/train.py Unitree-Go2W-Perceptive-Rough-v2 \
  --env.scene.num-envs=4096 \
  --agent.resume=True \
  --agent.load-run='<来源运行目录名>' \
  --agent.load-checkpoint='model_1000.pt' \
  --enable-nan-guard=True
```

来源是 `logs/rsl_rl/go2w_perceptive/` 下的目录。v2/精调/Step14 观测形状兼容；Rough/v1 与 v2 的历史长度不同，不能直接恢复。修复后的物理后端会改变部分碰撞结果，续训不保证与原后端逐步一致。

```bash
python scripts/play.py Unitree-Go2W-Perceptive-Rough-v2 \
  --checkpoint-file=logs/rsl_rl/go2w_perceptive/<运行目录>/model_1000.pt

python scripts/play.py Unitree-Go2W-Perceptive-Rough-v2 \
  --onnx-file=<导出目录>/policy.onnx --device=cpu --viewer=viser
```

ONNX 回放使用 ONNX 自带的 ReferenceEvaluator，无需额外安装 ONNX Runtime。任务、模型、奖励、感知、边界检查及部署功能保留；详细参数见 [Go2W 说明](go2w_perceptive_zh.md)。

## 保留的训练保护

- 所有 Go2W train/play 任务在出生及每次 reset 最后执行地形净空保护。名义根高度仍为 0.4 m；根据当前关节姿态、朝向、机器人碰撞体包围盒和地形包络，仅抬高需要调整的环境，保留 5 mm 加碰撞 margin 的间隙。高度场检查重叠网格单元的全部顶点。覆盖当前平地、台阶、正反斜坡、粗糙和波浪地形；不支持的倾斜 plane/heightfield 会报错。这不限制后续动态运动的穿透量。
- 共享速度任务主求解器/线搜索预算保留 `50/100`。原 `10/20` 的冻结触发状态在现预算下收敛，结果与 `100/200` 参考一致。达到停止条件会提前结束，预算不是每步固定执行次数，容差与告警保持不变。
- `cone/impratio Attach conflict` 原因是 XML 的 `elliptic/100` 与场景的 `pyramidal/1` 重复冲突。已删除机器人 XML 的全局声明，继续由场景统一使用 `pyramidal/1`。
- Go2W 非轮碰撞体使用零 margin、`condim=1, priority=0`；轮子保留 1 mm margin、`condim=3, priority=1`，摩擦按任务随机化。模型来源见 [SOURCE.md](../src/assets/robots/unitree_go2w/SOURCE.md)。

NaN guard 先原子保存核心故障数据，再补充接触和随机化字段。无 tendon 的空字段会跳过；附加读取、保存或 latest 副本失败不会丢失已保存的核心转储。保留 NPZ、JSON 和配套 MJB，可逐帧查看或回放原始输入：

```bash
python scripts/viz_nan.py <目录>/nan_dump_latest.npz
python scripts/inspect_nan_dump.py <目录>/nan_dump_latest.npz --device cuda:0 --output logs/nan_replay.json
python scripts/inspect_nan_dump.py <目录>/nan_dump_latest.npz --summary-only
```

原始输入回放恢复记录的控制量、warmstart、随机化和求解器，默认选择故障前输入；不完整/旧版转储只能摘要或查看，不能伪装成完整输入回放。原始 GPU 接触与重算结果分开报告。MuJoCo 3.5/3.17 编译的 MJB 仍需对应旧环境。

## 验证与已知边界

```bash
python -m pip check
python scripts/list_envs.py
python -m unittest discover -s tests -v
```

有意义的回归集中在 `tests/`：完整任务 NaN 注入、空 tendon 与附加保存失败、冻结碰撞、求解器上限、六个 Go2W 配方的全地形出生/部分重置、后端真实缓冲分配、模型/观测/奖励和 ONNX 元数据。迁移对比、容量/FMA 扫描、四后端矩阵、EPA 专项捕获及重复的 smoke CLI 已删除；三项上游算法回归保留在后端仓库，可运行：

```bash
python -m mujoco_warp._src.collision_gjk_test GJKTest.test_cylinder_box_side_witness
python -m mujoco_warp._src.collision_driver_test CollisionTest.test_cylinder_box_shallow_parity0 CollisionTest.test_cylinder_box_shallow_parity1
```

原 RTX 4090 的固定 NaN 场景中，原版和仅 #1796 产生约 61.87 m 的异常接触；组合修复保留接触且 forward/step 有限，三项上游 GPU 回归通过。组合结果与原生 MuJoCo 的接触位置仍相差 1.364 mm，超过原诊断脚本自设的 1 mm 门槛，因此旧报告整体 `passed:false`，不能改称全部通过。另一份 EPA 现场在 30/64/128 时与 256 参考的七个接触一致，但 30 后续训练仍溢出；用户当前 64 试训尚未报告告警。

本次清理后，本机 CPU 通过 65 项项目回归、三项上游算法回归和依赖一致性检查；普通训练入口完成 32 环境、两轮 PPO、checkpoint/ONNX 导出及一轮断点续训。checkpoint 与 ONNX 各完成 16 个回放控制步，动作、观测和奖励有限；原始 NaN 输入回放也完成。原版后端环境通过一条 `pip install -e .` 安装了上述 fork 的固定 Git 提交。

这些证据支持当前训练配置，不证明所有 GPU 状态或长期训练必然稳定。本次固定安装的本机验证使用 CPU；目标 4090 的长期训练仍需持续观察。
