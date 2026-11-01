# Go2W 速度跟随 SGS：连续速度区间版

当前 SGS 调度 **允许的地形块与速度任务/子区间组合**：楼梯与高等级非平地仅给机体纵向速度，简单地形保留全向。出生位置和朝向保持原 v2 的每回合随机化。具体速度和持续时间在运行时重新抽取。

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

原 v2、Gait、StepUp 等任务保留各自的课程与采样机制。SGS 继承 v2 的 PPO、镜像增强、奖励、混合动作、720 维 Actor 历史输入及 MID360 观测。日志位于 `logs/rsl_rl/go2w_perceptive/`。

各训练配置原有的 `forward_velocity_floor` 已统一替换为 `dont_wait`，权重 `-0.1`：仅当平移指令模长大于 `0.2 m/s` 时启用，实际 XY 速度沿指令方向的投影低于 `0.15 m/s`、`0`、`-0.15 m/s` 时分别累加一档代价。低速指令和纯转向不启用此项，准确跟随低速不会再因未达到固定 `0.22 m/s` 被惩罚。速度跟踪奖励继续负责跟踪精度。

## 任务与速度覆盖

完整速度集合有 24 类任务，按下述地形规则选取允许的类别：

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

每个运动大类沿主速度轴再细分 2 个子区间，站立保持一个桶：完整集合为 `1 + 23*2 = 47` 个速度桶，纵向集合为 `1 + 8*2 = 17` 个。默认实际地图为 10 行 × 9 列，过滤后共 **2550 个 SGS 统计项、1320 个父组**。同一地形块/运动大类的两个子区间分别记录结果，并共享父组先验。关闭 heading_command 会去掉完整集合中的两个朝向控制大类，实际数量由地形与允许类别计算。

| 地形 | 允许的速度任务 |
| --- | --- |
| pyramid_stairs、pyramid_stairs_inv、pyramid_stairs_2、pyramid_stairs_inv_2，所有等级 | 仅机体纵向 |
| flat，所有等级 | 完整全向集合 |
| 坡地、粗糙地、波浪地，0–5 级 | 完整全向集合 |
| 坡地、粗糙地、波浪地，6 级及以上 | 仅机体纵向 |

纵向集合包含站立、三级前进、两级后退、起步、停止与前后切换，共 9 类；两阶段都满足 `vy=0, wz=0`，使用机体系指令，不启用 heading/world 反馈。X 方向速度范围不变。出生朝向仍在 [-3.14,3.14] 随机，因此纵向不等于沿台阶法线方向；此设置没有加入对齐楼梯方向的重置。

高等级分界通过 `--env.commands.twist.sgs.longitudinal-from-level=6` 调整；等级从 0 开始，6 是当前初值。设为 `None` 时，其他地形所有等级保留全向，四种楼梯仍仅纵向，此时默认为 3030 项。SGS 和 Uniform 对照、训练和 play 使用相同过滤规则。受限组合直接从表中移除，不先抽取全向任务再清零。

配置编号是紧凑表的行号，须通过 `configuration_bank.tile_ids/configuration_bank.bucket_ids` 查到地形和速度桶，不能再用 `编号 // 47` 或 `编号 % 47` 解码。地形生成器需使用每列一种类型的 curriculum 模式；地形的等级分配仍由 SGS 控制。

默认单轴任务的两个子区间如下；后退、侧移和转向使用速度幅值表示，方向由任务对应的正负号决定：

| 任务 | 子区间 1 | 子区间 2 | 单位 |
| --- | --- | --- | --- |
| 低速前进 | 0.10–0.25 | 0.25–0.40 | m/s |
| 中速前进 | 0.40–0.55 | 0.55–0.70 | m/s |
| 高速前进 | 0.70–0.90 | 0.90–1.10 | m/s |
| 低速后退、侧移 | 0.10–0.225 | 0.225–0.35 | m/s |
| 高速后退、侧移 | 0.35–0.475 | 0.475–0.60 | m/s |
| 低速转向 | 0.10–0.275 | 0.275–0.45 | rad/s |
| 高速转向 | 0.45–0.625 | 0.625–0.80 | rad/s |

混合运动、世界坐标和朝向控制任务沿主平移速度轴细分；切换任务沿原来的主轴细分，不将三个轴分别二分后扩成八种组合。

每次重新采样指令，均在该桶范围内重新生成速度及 3–8 秒持续时间；其余非主轴按类别规定范围采样。普通任务在整个回合保持桶编号；切换任务在两组固定范围之间交替，幅值仍重新随机抽取。世界坐标速度会转换到机体坐标；朝向控制 yaw 由实时反馈产生，所以区间描述的是参考指令。成功判定始终使用最终执行的机体坐标指令。逐轴 0.1 死区保留。

类别分层替代旧 rel_standing_envs/rel_forward_envs 等指令模式占比；低速前进不会被旧 forward 模式强制抬到 0.3 m/s。

出生状态直接保留 v2 的 reset_base：每次在所选块中心附近重新抽取 X/Y ±0.5 m 和 yaw [-3.14,3.14]，净空检查仍在最后执行。出生位置、朝向不进入 SGS 编号。其他动力学随机化、推扰和噪声按原机制执行。

SGS 接管地形分配，关闭原来的距离升降地形课程和逐步扩大速度范围课程；奖励权重课程及 MID360 感知课程继续运行。

可通过 `--env.commands.twist.sgs.velocity-subdivisions` 调整统计分辨率：在默认地形规则下，1 对应 1320 项，2（默认）对应 2550 项，4 对应 5010 项，8 对应 9930 项。所有设置都会在线生成连续速度；增加子区间不扩大速度总范围，但会缩小同一回合内重新采样速度的范围。更改细分数量或高等级分界需要清空 SGS 历史或新实验；它们不决定成功标准。

## 成功定义

物理步完成后、自动重置及命令更新之前，比较真实基座机体速度与该步实际执行指令。每次重新采样指令后的前 0.6 秒留作响应时间；heading 每步反馈不重启宽限期。

每段至少有 1 秒有效计分时间，满足时长的段才纳入统计。将这些段在宽限期之后的样本合并，计算整回合平移误差 `sqrt(mean((vx-cx)^2+(vy-cy)^2))` 和转向误差 `sqrt(mean((wz-cw)^2))`，按有效样本数加权，不平均各段的 RMS，也不要求每段都通过。条件为：

| 情况 | 误差上限（严格小于） |
| --- | --- |
| 非零平移 | `min(0.2 m/s, 0.6*整回合指令平移RMS)` |
| 零平移 | `0.1 m/s` |
| 非零转向 | `min(0.35 rad/s, 0.6*整回合指令转向RMS)` |
| 零转向 | `0.15 rad/s` |

一次成功要求完整跑满 20 秒、没有失败终止、至少一个有效段且整回合两项 RMS 通过。不足 1 秒计分的尾段既不参与成功判定，也不进入打印的整回合 RMS。跌倒即使发生在宽限期内也记失败。人工重置、手柄接管、初次随机缩短回合、地图越界及非有限状态排除。

相较旧版，取消了逐段一票否决，相对误差比例从 50% 放宽到 60%；绝对上限、静止门槛、宽限期和完整回合要求保持不变。例如 `0.15 m/s` 指令的线速度误差上限由 `0.075` 放宽到 `0.09 m/s`，但原地不动仍不合格。先对误差和指令分别求平方均值，再开方，前进/后退或左右转向的正负切换不会互相抵消。

这些门槛是尚未实机标定的初始值。成功标签只用于 SGS 统计，不增加奖励或成功终止。

## 成功统计与抽样

细分配置及父组（同一地形块/运动大类）各保存最近 H=32 个 Boolean 结果。默认 a=group_prior_strength=4：

```text
p_group = (group_successes + a*0.66)/(group_samples + a)
p_i = (successes_i + a*p_group)/(samples_i + a)
```

样本数和成功数只取各自近期窗口。完全未访问组的估计成功率为 0.66；同组成绩能辅助未访问子区间。设置 a=0 可关闭收缩，恢复各项独立的经验成功率。滚动结果仍只存真实成功/失败，不把未访问当失败。

对照基础分布先均匀选地形块，再均匀选该块允许的运动大类，最后均匀选该类子区间。默认每块基础概率仍为 1/90，楼梯不会因允许类别较少而失去基础探索占比；站立不会因只有一个桶而被稀释。SGS 的自适应部分仍可改变最终地形占比，10% 基础探索分布保留同样的地形保障。

```text
base_i = 1/(地形块数 * 该块允许的大类数 * 该类子区间数)
w_i = (p_i+eps)^(k*t) * (1-p_i+eps)^(k*(1-t))
logit_i = log(max(w_i,eps)+eps)
q_sgs = softmax(logit_i/max(T,1) + log(base_i))
q_i = 0.9*q_sgs + 0.1*base_i
默认 t=0.66, k=5, T=2, eps=1e-8
```

常用参数前缀 `--env.commands.twist.sgs.`：`velocity-subdivisions`、`longitudinal-from-level`、`history-length`、`group-prior-strength`、`target-success`、`concentration`、`temperature`、`uniform-mix`、`bank-seed`、`settle-s`、`min-segment-s`、`linear-rms`、`yaw-rms`、`relative-error`、`standing-linear-rms`、`standing-yaw-rms`。旧 profiles-per-tile 参数已移除。

bank-seed + agent.seed 初始化指令随机流；任务选择使用单独随机流，出生随机化由环境随机流负责。地形生成种子为 env.scene.terrain.terrain_generator.seed，默认训练为 20261030。

日志前缀 `Curriculum/sgs_sampling/`：success_rate 是当前有效重置成功率，bank_success 是已访问细分配置的原始近期成功率均值，bank_coverage/group_coverage 是两层访问覆盖率，min_visits 是最低访问次数，configurations 是统计项总数；另有有效/排除计数及速度 RMS。采样分布一直变化，训练成功率不能替代独立评估。

成功率长期为零时，可结合以下诊断项定位；逐段诊断不再否决整回合标签：

| 字段 | 含义 |
| --- | --- |
| segment_success_rate | 当前有效结束回合内，通过平移和转向两项判据的有效指令段比例 |
| linear_failed_rate | 整回合平移 RMS 超标的回合比例 |
| yaw_failed_rate | 整回合转向 RMS 超标的回合比例 |
| terminated_rate | 发生失败终止的回合比例 |
| full_timeout_rate | 无失败终止且完整跑满时长的回合比例 |
| no_valid_segment_rate | 没有任何满足计分时长要求的指令段的回合比例 |
| recent_successes | 全部细分配置近期历史窗口内的真实成功记录条数，不含平滑先验，不是累计成功数 |

除 recent_successes 外，各项只统计当前批次有效回合；segment_success_rate 按段计数，其余比例按回合计数。linear_failed_rate/yaw_failed_rate 对应当前整回合判据，即使 segment_success_rate 小于 1，整回合仍可能成功。失败原因可以重叠，不能相加当作总失败率。RSL-RL 仍对本轮各次上报取平均，因此比例不是整轮按回合或指令段数量加权的汇总。recent_successes 显示的是窗口成功条数快照均值，有助于分辨 bank_success 保留四位小数后显示零与窗口里完全没有成功。

## 保存、续训与旧模型

checkpoint 的 infos.sgs_state 保存允许组合表、基础采样概率、区间定义、两层历史及计数、任务选择和指令生成两条随机流，照常导出 ONNX 与部署 YAML。同一编号或父组同时完成的结果逐一写入，超过窗口大小时保留最新结果。

当前 checkpoint 为版本 4，记录 `configuration_bank` 和 `success_aggregation=episode_rms`。同版本、相同任务定义与成功判据下，正常续训会恢复 SGS 历史：

旧版全地形全向组合表中的 4230/16650 项统计，不能恢复到当前过滤后的表。版本 3 虽然已采用整回合 RMS，仍使用旧组合表；版本 2 还使用逐段判据。这些历史都需要清空后再续训。

```bash
python scripts/train.py Unitree-Go2W-Perceptive-Rough-v2-SGS --gpu-ids "[1]" --agent.resume=True --agent.load-run=运行目录名 --agent.load-checkpoint=model_1000.pt
```

从旧任务表或旧判据模型继续训练时，推荐只清空 SGS 历史，保留 Actor/Critic、观测归一化、优化器、学习率、迭代数和课程进度：

```bash
python scripts/train.py Unitree-Go2W-Perceptive-Rough-v2-SGS --env.scene.num-envs=4096 --gpu-ids "[1]" --agent.resume=True --agent.load-run=旧运行目录名 --agent.load-checkpoint=model_8000.pt --env.commands.twist.reset-sgs-history=True
```

此选项从当前配置的全新 SGS 表与随机流开始，不读取旧的成功标签。新判据下保存 checkpoint 后，再次续训应移除 `reset-sgs-history=True`，避免反复清空统计。若保留 8 份细分，另加 `--env.commands.twist.sgs.velocity-subdivisions=8`。旧版成功率与新版成功率不应直接比较。

旧固定序列/固定出生 checkpoint 是版本 1。若希望同时重启优化器及课程，也可在模型结构匹配时使用权重热启动，只加载 Actor/Critic（含观测归一化）：

```bash
python scripts/train.py Unitree-Go2W-Perceptive-Rough-v2-SGS --gpu-ids "[1]" --agent.resume=True --agent.load-run=旧运行目录名 --agent.load-checkpoint=model_1000.pt --env.commands.twist.warm-start=True
```

热启动后的常规续训应移除 warm-start=True。mjlab 不保存物理轨迹或完整环境随机状态，恢复后在途回合会被排除，下次实际重置再记分，不能宣称精确重放整场仿真。

## 回放与独立评估

```bash
python scripts/play.py Unitree-Go2W-Perceptive-Rough-v2-SGS --checkpoint-file=完整模型路径
python scripts/evaluate_sgs.py --checkpoint 路径/model_1000.pt --num-envs 128 --episodes 512 --seed 20261031 --device cuda:1 --output outputs/sgs_eval.json
```

评估使用新地形种子、新指令随机流和随机出生，按地形块及其允许类别均衡采样；采用最终 MID360 配置，关闭 play 原本关闭的推扰。evaluate_sgs.py 会读取版本 4 checkpoint 的 velocity_subdivisions 和 longitudinal_from_level，保持训练时的地形限制；旧模型使用当前默认限制。play.py 仍采用当前配置，训练中自定义了这两项时，回放需传入相同命令行参数。只加载模型，不恢复训练采样历史，play 禁止覆盖训练 checkpoint。评估脚本的 --device 是进程内 PyTorch 设备，不经过训练入口的 GPU 重映射。

报告包含整体、地形、大类、子区间的成功率、失败终止率、RMS 和逐回合记录，保存区间定义、configuration_bank 与抽样测度，每条记录标注 longitudinal_only。按环境分配固定有效回合配额，等待全部完成，避免只收集最快结束的失败回合。未覆盖项显示 episodes=0 和空指标。512 回合不足以覆盖全部细分任务；完整覆盖需增加预算。默认成功判据固定，训练中自定义成功阈值不自动进入评估。

至少比较 3 个训练种子。当前仅能验证 CPU 流程，GPU 性能、长期收敛及真机效果需要后续实验。

## 代码入口

- src/tasks/velocity/mdp/sgs.py：采样器、分层概率估计及成功判定。
- src/tasks/velocity/mdp/velocity_buckets.py：类别与速度子区间。
- src/tasks/velocity/mdp/terrain_velocity_bank.py：地形限制、允许组合表及基础地形占比。
- src/tasks/velocity/mdp/sgs_velocity_command.py：在线指令、重置归因及持久化。
- src/tasks/velocity/config/go2w/sgs_env_cfg.py：独立任务配置。
- src/tasks/velocity/rl/sgs_runner.py：续训和权重热启动。
- scripts/evaluate_sgs.py：独立评估。
- tests/test_sgs.py：统计、区间覆盖、切换、排除规则和恢复检查。
