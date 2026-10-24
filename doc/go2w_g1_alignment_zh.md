# Go2W 与本仓库 G1 的通用设置对齐

2026-10-23，用户要求以当前仓库自己的 G1 为比较基准，并明确保留之前手动指定的三项奖励参数。基准为 `src/tasks/velocity/config/g1/env_cfgs.py` 和 `rl_cfg.py`，G1 文件没有修改。Go2W 继续从公共根层直接构建，没有导入或调用 G1 task，没有新增配置循环或推导式。

## 已对齐的内容

| 项目 | 当前 Go2W 行为 |
| --- | --- |
| 策略动作裁剪 | `clip_actions=None`，训练、PT play 和 ONNX play 共用该值；轮目标为 `20 * action`，没有额外的 ±20 rad/s 目标限制 |
| PPO | 对应阶段的 actor/critic、归一化、Gaussian std=1、学习率、熵、更新轮数、采样步数、保存间隔、总迭代数与 G1 相同；PPO `clip_param=0.2` 仍保留 |
| 日志默认值 | 与 G1 一样使用 mjlab 默认 W&B、上传模型；实验目录和运行名保留 Go2W 名称，可用命令行覆盖 logger/project |
| 仿真 | 对应任务的时间步、decimation、接触容量、约束容量、CCD 和接触传感器容量逐项一致 |
| 接触维数与优先级 | 2026-10-24 按 G1 分配：非轮碰撞体 `condim=1、priority=0`，四轮 `condim=3、priority=1`，使轮摩擦随机化控制轮地接触 |
| 非轮部接触参数 | 非轮碰撞体 `margin=0`，基础滑动摩擦系数为 `1`；直接定义在最终 XML 中 |
| 地形 | 对应阶段的网格、border、台阶高度/宽度/比例、坡道、粗糙度、初始地形等级一致；v2 上行台阶 4–18 cm，训练 border=20 m，play border=10 m |
| 指令与课程 | 对应 G1 阶段的 heading 模式、站立概率、初始指令范围、速度课程及 v2 线速度跟踪权重课程一致；不再额外固定侧向指令为 0 |
| 观测 | 公共项的缩放、噪声、延迟、损坏开关一致；Rough/v1 单帧，v2 三帧，Critic 单帧 |
| 高度扫描 | 与 G1 一样保留居中密集扫描 `terrain_scan`，v2 另配前移 0.35 m 的 `terrain_scan_mid360`；稀疏化、丢帧和 0.8 s 记忆一致 |
| 随机化 | 对应阶段的轮地摩擦、编码器偏差、机体 COM 扰动和推力参数一致，机器人名称/轮几何/腿关节映射使用 Go2W |
| 公共奖励 | 身体角速度、动作变化、落地、停车、自碰撞等权重按阶段一致；精调保留 G1 的前向速度下限、直行偏航及朝向漂移项 |
| 终止 | 移除此前额外的 `illegal_contact` 终止，公共项与 G1 一样为 time_out/fell_over |

## 明确保留的差异

- 用户指定 `pose.weight=0.5`、`body_orientation_l2.weight=-0.7`、角速度跟踪 `std=0.5`。pose 固定为 0.5，因此不启用 G1 会覆盖它的 pose 权重课程。角速度跟踪的权重仍随 G1 阶段变化。
- 轮足动作、关节/碰撞/机体映射、轮速度观测、默认姿态和执行器沿用 Go2W。腿动作 scale=0.5，轮 scale=20；轮速度增益 2、力矩上限 45 Nm、armature=0；继续使用圆柱轮和自碰撞。
- 轮碰撞体的基础摩擦仍继承 XML 的 `0.8 0.02 0.01`，训练时滑动摩擦按对应阶段随机化；轮 `margin=0.001 m`，作为非轮部零 margin 的显式例外。
- 禁用普通足滑移、固定双足步态、强制抬脚/摆脚/前伸落脚等不适合持续滚动的足运动奖励及相应课程；角动量奖励按之前的实验要求保持停用。原 XML 遥测传感器继续保留。
- 保留已迁移的 robot_lab 轮接触空转、轮停车、轮速奖励。公共自碰撞奖励现在增加独立 self_collision 传感器，轮接地/接触时间/空转仍共用 feet_ground_contact。旧 nonfoot_ground_touch 保留作诊断，不再触发提前终止。
- 保留用户要求的 `terrain_out_of_bounds` 超时，不计入 `is_terminated` 失败惩罚。新 v2 训练的外沿阈值随 border=20 自动成为 X ±57 m / Y ±41 m。
- 调试相机和指令箭头高度按 Go2W 尺寸配置，ONNX 导出继续携带腿位置/轮速度的混合动作元数据。

## 指定基准：这个仓库的 G1 V2 curriculum

比较对象明确为 `Unitree-G1-Perceptive-Rough-v2`，入口是 `src/tasks/velocity/config/g1/env_cfgs.py::unitree_g1_perceptive_rough_v2_env_cfg()`。G1 文件没有改动。此前只验证了最终生效的配置值，没有说明 Go2W 使用集中 stage 分支、先写 V1 再覆盖为 V2 的构建方式差异；不能因此称源码组织方式完全相同。

本次把 V1/V2 课程分别放入明确命名的独立配置段，V2 直接配置自己的阶段列表，使用 G1 原有的 `cfg.curriculum["command_vel"].params["velocity_stages"]` 和 `CurriculumTermCfg(func=mdp.reward_weight, ...)` 写法。Go2W 仍遵守先前约定，从公共根构建，不调用 G1 或其他已注册任务的工厂函数。

| G1 V2 课程项 | Go2W V2 | 对应规则或保留差异 |
| --- | --- | --- |
| `terrain_levels` | 保留 | 同一公共根配置、同一 `mdp.terrain_levels_vel`；行进距离和指令距离判定升降级 |
| `command_vel` | 保留 | 同一 `mdp.commands_vel`，阶段阈值 `0 / 4000*24 / 9000*24 / 16000*24`，各轴范围逐项相同 |
| `track_linear_velocity_weight` | 保留 | 同一 `mdp.reward_weight`，`0 / 7000*24 / 14000*24` 对应 `1.5 / 1.35 / 1.2` |
| `foot_clearance_weight` | 停用 | 对应强制抬脚奖励不用于持续滚动 |
| `foot_swing_height_weight` | 停用 | 对应摆脚高度奖励不用于持续滚动 |
| `pose_weight` | 停用 | 保留用户明确指定的固定 `pose.weight=0.5`，避免课程重新改回 0.75/0.6 |

`step` 是 `common_step_counter` 的控制步数，乘 24 来自每轮采样 24 个控制步。运行函数原本采用 `common_step_counter > stage["step"]`；Go2W 沿用同一函数，没有改成 `>=`。训练启用上述课程，play 与 G1 一样清空课程。高度扫描稀疏化与记忆的阶段参数仍位于 observation 的 height_scan term，不是额外的 manager curriculum 条目，位置和机制也按 G1 V2 对照。

新增检查比较整个 curriculum 键集合及每个保留项的函数和参数，明确排除上述三项；同时在每个阶段阈值以及阈值加一个控制步时，实际执行 G1/Go2W 两套配置，比较指令范围和奖励权重。课程格式重排前后，六个任务的 train/play/PPO 共 18 组配置快照一致。

完整集合检查另发现此前遗漏的差异：Go2W Step14-Up 训练删除了 `terrain_levels`，而这个仓库的 G1 V2 Step14 保留该 manager 课程项，仅设置地形生成器 `curriculum=False`。现已按 G1 补回；这是本次唯一改变课程生效配置的修正，标准 V2 的阶段参数没有再次改变。先前只比较部分课程条目的检查不足以证明整个课程配置一致，现已替换为完整集合检查。

课程专项修正后，42 项检查全部通过，包含完整课程集合比较及阈值前后执行检查；Step14-Up 的 CPU 重置与 4 个控制步也通过，Actor/Critic 维度为 726/269 且观测有限。

## 旧训练与回放

新启动的训练默认不裁剪动作。命令行显式传入 `--agent.clip-actions=1.0` 仍会覆盖默认值；要与 G1 比较，应省略该覆盖。

新 ONNX 的 `action_clip` 记录为字符串 `None`。回放校验区分 `None` 与旧模型中的 `1.0`，不静默改变动作语义，也不修改旧 ONNX。旧 v2 模型的网络形状仍为 Actor 726 / Critic 269，但输入扫描布局和训练分布发生变化，应从头训练建立当前基准。Rough/v1 已改成 Actor 242，Flat 为 Actor 55；它们不能直接作为 v2 的 checkpoint。

若进行圆柱与 mesh 的纯形状比较，应在上述同一训练配置下仅切换四个轮碰撞体。不要把地形、随机化与裁剪一起变化后的差异归因于单一模型形状。

## 验证

`tests/test_go2w.py` 对六个任务的 PPO 和 train/play 共 12 组环境逐项对比 G1，检查仿真、地形、指令、公共观测、事件、公共奖励和适用课程；机器人映射和上述保留项分别校验。配置独立性及无循环检查继续执行。`tests/test_onnx_play.py` 检查无裁剪元数据、旧数值裁剪元数据及不匹配时的拒绝行为。

本地 40 项回归检查通过。CPU 已完成无裁剪探针（动作 2 得到 40 rad/s 目标，执行器仍限制 45 Nm）、16 个控制步、1 次 PPO 更新，以及 `action_clip=None` 的 ONNX 导出和实际推理。该检查显式使用 TensorBoard；W&B 离线后台在本机启动停住后已中止，没有据此宣称云端上传验证通过。GPU 大规模训练稳定性仍需在训练机评估。
