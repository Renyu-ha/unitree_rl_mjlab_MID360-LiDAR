# Go2W 速度跟随 SGS 训练

新增两个独立任务，原有 v2、Gait、StepUp 等任务不变：

| 任务 | 采样方式 |
| --- | --- |
| `Unitree-Go2W-Perceptive-Rough-v2-SGS` | 成功率引导采样，默认混合 10% 均匀探索 |
| `Unitree-Go2W-Perceptive-Rough-v2-SGS-Uniform` | 同一固定场景库均匀采样，用作对照 |

SGS 根据 [A Balanced Data Diet，arXiv:2610.12465](https://arxiv.org/abs/2610.12465) 的公式独立实现。速度跟随成功标签参考了 [capstan 的公开实现](https://github.com/soujanya957/capstan-drive-leg/blob/1d263fa52b096f58c7b93ca72f855d57334db202/CapstanPedal/fullbuild_v4/rl/mjx_walk.py)，再加入低速、分段和重置处理；没有复制这些项目的源代码，也没有引入它们的依赖。

## 训练与对照

在已安装本项目依赖的环境里，从仓库根目录运行：

```bash
python scripts/train.py Unitree-Go2W-Perceptive-Rough-v2-SGS --env.scene.num-envs=4096 --gpu-ids "[0]" --agent.seed=0
```

使用相同种子、环境数量、PPO 参数和训练预算运行均匀对照：

```bash
python scripts/train.py Unitree-Go2W-Perceptive-Rough-v2-SGS-Uniform --env.scene.num-envs=4096 --gpu-ids "[0]" --agent.seed=0
```

任务默认继承 v2 的 PPO、镜像增强、奖励、动作、720 维 Actor 历史输入和 MID360 观测。运行名分别为 `v2_sgs`、`v2_uniform_bank`，日志仍在 `logs/rsl_rl/go2w_perceptive/` 下。

当前 SGS 专用 Runner 支持**单训练进程/单 GPU**。多进程会明确报错，避免只保存 rank 0 的统计却宣称完整恢复。原有任务的多 GPU 路径不变。

GPU 编号使用带引号的 Python 列表语法：第二张卡写 `--gpu-ids "[1]"`。入口启用了 mjlab 的集合字面量解析，不能写成 `--gpu-ids 1`。未提前限制 `CUDA_VISIBLE_DEVICES` 时，`[1]` 选择物理 GPU 1；启动器将其映射为进程内的 `cuda:0`，因此日志显示 `cuda:0` 属于正常情况。

## 固定场景库

每个配置包含：地形块（类型和等级）、出生 XY 偏移及偏航、整回合速度指令序列、每段持续时间和命令模式。训练 v2 默认实际生成 10 行 × 9 类地形，每块 16 个配置，共 **1440 个配置**；运行时读取真实地形网格，不使用 `num_cols=6` 推算列数。

每块保证包含站立、前进、侧移、原地转向和倒退，其余配置按原命令模式生成，保留 heading/world/forward 模式。指令每段 3–8 秒，仍使用逐轴 0.1 死区。出生位置位于块中心 ±0.5 m，净空检查继续在最后执行。

SGS 与均匀对照使用相同的固定地形种子、指令库与出生配置。质量、摩擦、推扰等现有随机化继续作为场景内的扰动，因此这里是**固定场景条件上的成功率估计**，不是将全部随机化状态固定下来。

SGS 接管地形分配；原距离升降课程和按步扩展速度范围的课程关闭。场景库从一开始覆盖 v2 最终范围：X `[-0.6, 1.1]` m/s、Y `[-0.6, 0.6]` m/s、Yaw `[-0.8, 0.8]` rad/s，避免同一配置随训练阶段改变含义。奖励权重课程和 MID360 感知课程继续保留。

SGS 的均匀对照也采用上述场景库和指令范围；它与原 v2 不是同一个训练分布。比较 SGS 的贡献应优先比较这两个新增任务。

## 成功标签

在物理步完成、环境自动重置及命令更新之前，读取该步实际执行的机体坐标系 `twist` 和真实机体速度，独立累计误差。每次**重新采样命令**后的前 0.6 秒不计误差，heading 每步反馈更新不会重启宽限期。

对每个至少有 1 秒有效计分时间的指令段，计算：

```text
linear_rms = sqrt(mean((vx - cmd_x)^2 + (vy - cmd_y)^2))
yaw_rms    = sqrt(mean((wz - cmd_yaw)^2))
cmd_linear_rms = sqrt(mean(cmd_x^2 + cmd_y^2))
cmd_yaw_rms    = sqrt(mean(cmd_yaw^2))
```

| 指令段 | 通过条件 |
| --- | --- |
| 非零平移 | `linear_rms < min(0.2, 0.5 * cmd_linear_rms)` |
| 零平移 | `linear_rms < 0.1` m/s |
| 非零转向 | `yaw_rms < min(0.35, 0.5 * cmd_yaw_rms)` |
| 零转向 | `yaw_rms < 0.15` rad/s |

相对误差门槛用于拒绝低速指令下停着不动。两项条件都要通过；回合中较早失败的段不能被后面成功的段平均掉。结束时不足 1 秒的尾段不参与判定。

**一次成功**要求完整跑完 20 秒、没有终止失败、至少有一个有效指令段，且所有有效段都通过。翻倒即使发生在宽限期内，也记录一次失败。人工重置、手柄接管、RSL-RL 初次随机缩短的回合、地图越界和非有限状态不进入成功历史。不会通过新增奖励或成功终止改变 PPO 目标。

这些阈值是可配置的初始设置，不是已在 Go2W 真机上标定的标准。先观察真实策略的误差分布及场景覆盖率，再决定是否调整；比较不同采样器时保持判据一致。

## SGS 参数与日志

默认每个配置保存最近 H=100 次真实 Boolean 结果。未访问配置的估计初始化为目标成功率 0.66；首次观察后只计算已有结果，不把未观察回合算失败。

```text
w_i = (p_i + eps)^(k*t) * (1-p_i + eps)^(k*(1-t))
logit_i = log(max(w_i, eps) + eps)
q_sgs = softmax(logit / max(T, 1))
q = 0.9 * q_sgs + 0.1 / N
t=0.66, k=5, T=2, eps=1e-8
```

10% 均匀混合和未访问配置初始化属于本项目的工程选择。设置以下选项可以移除均匀混合：

```bash
python scripts/train.py Unitree-Go2W-Perceptive-Rough-v2-SGS --env.scene.num-envs=4096 --env.commands.twist.sgs.uniform-mix=0.0
```

其他命令行字段见 `scripts/train.py <任务名> --help`：

```text
--env.commands.twist.sgs.history-length
--env.commands.twist.sgs.target-success
--env.commands.twist.sgs.concentration
--env.commands.twist.sgs.temperature
--env.commands.twist.sgs.profiles-per-tile
--env.commands.twist.sgs.bank-seed
--env.commands.twist.sgs.settle-s
--env.commands.twist.sgs.linear-rms
--env.commands.twist.sgs.yaw-rms
--env.commands.twist.sgs.relative-error
```

`bank-seed` 控制出生/指令库；地形生成种子是独立的 `env.scene.terrain.terrain_generator.seed` 配置，默认训练为 20261030。

日志前缀为 `Curriculum/sgs_sampling/`：`success_rate` 是本次有效重置的成功比例；`valid_episodes`、`recorded_total`、`skipped_total` 是有效/排除计数；`linear_rms`、`yaw_rms` 是物理误差；`bank_coverage` 是访问覆盖率；`bank_success` 是已访问配置的近期均值。它们用于检查采样运行状态，**不能代替独立评估**。

## 保存、续训与回放

SGS Runner 在普通模型 checkpoint 的 `infos.sgs_state` 中保存场景库、历史环形缓冲区、计数和采样随机数状态；照常导出 `policy.onnx` 及部署 YAML。每个同时完成的配置都会写入自己的历史槽位，超过 H 个同时结果时保留最新 H 个。

```bash
python scripts/train.py Unitree-Go2W-Perceptive-Rough-v2-SGS --env.scene.num-envs=4096 --agent.resume=True --agent.load-run=原SGS运行目录名 --agent.load-checkpoint=model_1000.pt
```

续训必须保持场景库、地形和成功判据设置一致；不一致会报错。旧 v2 checkpoint 没有 SGS 历史时会明确提示从未访问库开始。恢复的是模型与采样状态，mjlab 不恢复物理轨迹；恢复后的在途回合被排除，下一次环境重置后再开始记分。

```bash
python scripts/play.py Unitree-Go2W-Perceptive-Rough-v2-SGS --checkpoint-file=完整模型路径
```

play 配置使用另一随机种子生成的新场景库、均匀采样和有限 20 秒回合，不恢复训练采样历史，也禁止保存训练 checkpoint。修改 play 任务的 CLI 参数属于不同评估条件，应记录下来。

## 独立评估

对两个模型使用相同种子、回合数和环境数量：

```bash
python scripts/evaluate_sgs.py --checkpoint 路径/model_1000.pt --num-envs 128 --episodes 512 --seed 20261031 --device cuda:0 --output outputs/sgs_eval.json
```

输出整体及各地形成功率、翻倒率、RMS 误差和逐回合结果。评估使用新地形/出生/指令库，均匀抽样，采用最终 MID360 感知配置，关闭 play 本来就关闭的随机推扰；两个采样器采用同样条件。默认使用上表的成功判据，报告会保存它们。训练中若改过成功阈值，评估仍按这套固定默认标准比较。

至少比较 3 个训练种子；不要仅比较训练日志中的采样成功率。当前代码完成了 CPU 流程验证，尚未完成 GPU 性能测试、收敛实验或真机验证。

## 代码入口

- `src/tasks/velocity/mdp/sgs.py`：采样器和指令段成功统计。
- `src/tasks/velocity/mdp/sgs_velocity_command.py`：固定场景库、重置归因和持久化。
- `src/tasks/velocity/config/go2w/sgs_env_cfg.py`：独立实验配置。
- `src/tasks/velocity/rl/sgs_runner.py`：checkpoint 保存与恢复。
- `scripts/evaluate_sgs.py`：独立均匀评估。
- `tests/test_sgs.py`：重复索引、低速停滞、宽限、分段、排除规则和状态恢复回归。
