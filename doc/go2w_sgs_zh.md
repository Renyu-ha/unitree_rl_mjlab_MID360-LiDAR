# Go2W 速度跟随 SGS：连续速度区间版

当前 SGS 调度 **地形块 × 速度任务/子区间**，出生位置和朝向保持原 v2 的每回合随机化。具体速度和持续时间在运行时重新抽取，不再复用 16 组固定序列。

| 任务 | 采样方式 |
| --- | --- |
| `Unitree-Go2W-Perceptive-Rough-v2-SGS` | 成功率引导，默认混合 10% 类别均衡探索 |
| `Unitree-Go2W-Perceptive-Rough-v2-SGS-Uniform` | 相同区间和随机化，按地形/运动大类均衡抽样 |

核心权重参考 [SGS 论文](https://arxiv.org/abs/2610.12465)。速度标签参考 [capstan 代码](https://github.com/soujanya957/capstan-drive-leg/blob/1d263fa52b096f58c7b93ca72f855d57334db202/CapstanPedal/fullbuild_v4/rl/mjx_walk.py)。区间内在线采样、分层概率收缩和类别均衡探索是本项目的工程扩展，并非原论文固定具体配置的严格复现。

## 训练与对照

```bash
python scripts/train.py Unitree-Go2W-Perceptive-Rough-v2-SGS --env.scene.num-envs=4096 --gpu-ids "[1]" --agent.seed=0
python scripts/train.py Unitree-Go2W-Perceptive-Rough-v2-SGS-Uniform --env.scene.num-envs=4096 --gpu-ids "[1]" --agent.seed=0
```

使用相同种子、PPO 参数、区间和训练预算比较。GPU 参数必须采用列表语法 `[1]`，不能写成整数 `1`。未限制 CUDA_VISIBLE_DEVICES 时选择物理 GPU 1，进程内日志显示 cuda:0 属于正常映射。SGS 目前仅支持单训练进程/单 GPU。

原 v2、Gait、StepUp 等任务保留原机制。SGS 继承 v2 的 PPO、镜像增强、奖励、混合动作、720 维 Actor 历史输入及 MID360 观测。日志位于 `logs/rsl_rl/go2w_perceptive/`。

## 任务与速度覆盖

默认 24 类任务：

| 类别 | 数量 |
| --- | ---: |
| 站立 | 1 |
| 前进低/中/高速 | 3 |
| 后退低/高速 | 2 |
| 左右侧移 × 低/高速 | 4 |
| 左右原地转向 × 低/高速 | 4 |
| 前进/后退混合运动 | 2 |
| 前进/后退世界坐标运动 | 2 |
| 前进/后退朝向控制 | 2 |
| 起步、停止、前后切换、左右转向切换 | 4 |

参考速度范围为 X `[-0.6,1.1]` m/s、Y `[-0.6,0.6]` m/s、Yaw `[-0.8,0.8]` rad/s。前进三级按正向可用范围的 0–30%、30–60%、60–100% 分割，默认边界为 0.1、0.4、0.7、1.1 m/s；后退、侧移和转向分别按各方向可用范围等分两级。

每个运动大类沿主速度轴再细分 8 个子区间，站立保持一个桶：`1 + 23*8 = 185` 个速度桶。默认实际地图为 10 行 × 9 列，共 **16650 个 SGS 统计项**。关闭 heading_command 会去掉两个朝向控制大类，实际数量随地形网格和类别数计算，不硬编码列数。

每次重新采样指令，均在该桶范围内重新生成速度及 3–8 秒持续时间；其余非主轴按类别规定范围采样。普通任务在整个回合保持桶编号；切换任务在两组固定范围之间交替，幅值仍重新随机抽取。世界坐标速度会转换到机体坐标；朝向控制 yaw 由实时反馈产生，所以区间描述的是参考指令。成功判定始终使用最终执行的机体坐标指令。逐轴 0.1 死区保留。

类别分层替代旧 rel_standing_envs/rel_forward_envs 等指令模式占比；低速前进不会被旧 forward 模式强制抬到 0.3 m/s。

出生状态直接保留 v2 的 reset_base：每次在所选块中心附近重新抽取 X/Y ±0.5 m 和 yaw [-3.14,3.14]，净空检查仍在最后执行。出生位置、朝向不进入 SGS 编号。其他动力学随机化、推扰和噪声按原机制执行。

SGS 接管地形分配，关闭原来的距离升降地形课程和逐步扩大速度范围课程；奖励权重课程及 MID360 感知课程继续运行。

可通过 `--env.commands.twist.sgs.velocity-subdivisions` 调整统计分辨率：1 对应默认地图 2160 项，4 对应 8370 项，8 对应 16650 项。所有设置都会在线生成连续速度；增加子区间不扩大速度总范围。更改细分数量需要新实验或权重热启动。

## 成功定义

物理步完成后、自动重置及命令更新之前，比较真实基座机体速度与该步实际执行指令。每次重新采样指令后的前 0.6 秒留作响应时间；heading 每步反馈不重启宽限期。

每段至少有 1 秒有效计分时间，计算平移误差 `sqrt(mean((vx-cx)^2+(vy-cy)^2))` 和转向误差 `sqrt(mean((wz-cw)^2))`。条件为：

| 情况 | 误差上限（严格小于） |
| --- | --- |
| 非零平移 | `min(0.2 m/s, 0.5*指令平移RMS)` |
| 零平移 | `0.1 m/s` |
| 非零转向 | `min(0.35 rad/s, 0.5*指令转向RMS)` |
| 零转向 | `0.15 rad/s` |

一次成功要求完整跑满 20 秒、没有失败终止、至少一个有效段且所有有效段通过。前面失败的段不能被后面平均掉；不足 1 秒计分的尾段不参与判定。跌倒即使发生在宽限期内也记失败。人工重置、手柄接管、初次随机缩短回合、地图越界及非有限状态排除。

这些门槛是尚未实机标定的初始值。成功标签只用于 SGS 统计，不增加奖励或成功终止。

## 成功统计与抽样

细分配置及父组（同一地形块/运动大类）各保存最近 H=32 个 Boolean 结果。默认 a=group_prior_strength=4：

```text
p_group = (group_successes + a*0.66)/(group_samples + a)
p_i = (successes_i + a*p_group)/(samples_i + a)
```

样本数和成功数只取各自近期窗口。完全未访问组的概率为 0.66；同组成绩能辅助未访问子区间。设置 a=0 可关闭收缩，恢复各项独立的经验成功率。滚动结果仍只存真实成功/失败，不把未访问当失败。

对照基础分布先均匀选地形块和运动大类，再均匀选该类子区间，站立不会因只有一个桶而被稀释。它不是三维速度空间的均匀密度，也不是所有细分编号等概率。

```text
base_i = 1/(地形块数 * 大类数 * 该类子区间数)
w_i = (p_i+eps)^(k*t) * (1-p_i+eps)^(k*(1-t))
logit_i = log(max(w_i,eps)+eps)
q_sgs = softmax(logit_i/max(T,1) + log(base_i))
q_i = 0.9*q_sgs + 0.1*base_i
默认 t=0.66, k=5, T=2, eps=1e-8
```

常用参数前缀 `--env.commands.twist.sgs.`：`velocity-subdivisions`、`history-length`、`group-prior-strength`、`target-success`、`concentration`、`temperature`、`uniform-mix`、`bank-seed`、`settle-s`、`min-segment-s`、`linear-rms`、`yaw-rms`、`relative-error`、`standing-linear-rms`、`standing-yaw-rms`。旧 profiles-per-tile 参数已移除。

bank-seed + agent.seed 初始化指令随机流；任务选择使用单独随机流，出生随机化由环境随机流负责。地形生成种子为 env.scene.terrain.terrain_generator.seed，默认训练为 20261030。

日志前缀 `Curriculum/sgs_sampling/`：success_rate 是当前有效重置成功率，bank_success 是已访问细分配置的原始近期成功率均值，bank_coverage/group_coverage 是两层访问覆盖率，min_visits 是最低访问次数，configurations 是统计项总数；另有有效/排除计数及速度 RMS。采样分布一直变化，训练成功率不能替代独立评估。

## 保存、续训与旧模型

checkpoint 的 infos.sgs_state 保存区间定义、两层历史及计数、任务选择和指令生成两条随机流，照常导出 ONNX 与部署 YAML。同一编号或父组同时完成的结果逐一写入，超过窗口大小时保留最新结果。

正常续训保持相同任务定义与成功判据：

```bash
python scripts/train.py Unitree-Go2W-Perceptive-Rough-v2-SGS --gpu-ids "[1]" --agent.resume=True --agent.load-run=运行目录名 --agent.load-checkpoint=model_1000.pt
```

旧固定序列/固定出生 checkpoint 是版本 1，当前是版本 2，成功历史不兼容。可在模型结构匹配时仅加载 Actor/Critic 权重（含观测归一化），清空优化器、迭代数、课程计数和 SGS 历史：

```bash
python scripts/train.py Unitree-Go2W-Perceptive-Rough-v2-SGS --gpu-ids "[1]" --agent.resume=True --agent.load-run=旧运行目录名 --agent.load-checkpoint=model_1000.pt --env.commands.twist.warm-start=True
```

热启动后的常规续训应移除 warm-start=True。mjlab 不保存物理轨迹或完整环境随机状态，恢复后在途回合会被排除，下次实际重置再记分，不能宣称精确重放整场仿真。

## 回放与独立评估

```bash
python scripts/play.py Unitree-Go2W-Perceptive-Rough-v2-SGS --checkpoint-file=完整模型路径
python scripts/evaluate_sgs.py --checkpoint 路径/model_1000.pt --num-envs 128 --episodes 512 --seed 20261031 --device cuda:1 --output outputs/sgs_eval.json
```

评估使用新地形种子、新指令随机流和随机出生，按类别均衡采样；采用最终 MID360 配置，关闭 play 原本关闭的推扰。只加载模型，不恢复训练采样历史，play 禁止覆盖训练 checkpoint。评估脚本的 --device 是进程内 PyTorch 设备，不经过训练入口的 GPU 重映射。

报告包含整体、地形、大类、子区间的成功率、失败终止率、RMS 和逐回合记录，保存区间定义与抽样测度。按环境分配固定有效回合配额，等待全部完成，避免只收集最快结束的失败回合。未覆盖项显示 episodes=0 和空指标。512 回合不足以覆盖全部细分任务；完整覆盖需增加预算。默认成功判据固定，训练中自定义判据不自动进入评估。

至少比较 3 个训练种子。当前仅能验证 CPU 流程，GPU 性能、长期收敛及真机效果需要后续实验。

## 代码入口

- src/tasks/velocity/mdp/sgs.py：采样器、分层概率估计及成功判定。
- src/tasks/velocity/mdp/velocity_buckets.py：类别与速度子区间。
- src/tasks/velocity/mdp/sgs_velocity_command.py：在线指令、重置归因及持久化。
- src/tasks/velocity/config/go2w/sgs_env_cfg.py：独立任务配置。
- src/tasks/velocity/rl/sgs_runner.py：续训和权重热启动。
- scripts/evaluate_sgs.py：独立评估。
- tests/test_sgs.py：统计、区间覆盖、切换、排除规则和恢复检查。
