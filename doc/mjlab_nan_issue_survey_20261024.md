# mjlab NaN/Inf 问题调研

核查日期：2026-10-24；文中日期按北京时间（UTC+8）。仓库：[mujocolab/mjlab](https://github.com/mujocolab/mjlab)。

## 范围与方法

- 分页扫描 GitHub 当时可访问的全部 **329 个 issue、835 个 PR** 的标题和正文，共 1164 条记录；最后一页已检查为空。
- 搜索正文及评论中的 NaN、NaNs、NaNing、Inf、nan_guard、nan_policy、non-finite、std、normal expects、instability、divergence、overflow、heightfield、solver 等词。
- 对筛选出的 **48 个 issue 阅读 API 返回的 225 条可访问评论**；其中 22 条直接涉及 NaN、其传播/诊断或相同 std 报错，另外 26 条作为相关问题或排除项记录。另核对这 22 条的关闭事件，避免把关闭状态等同于根因已修复。#1153 的元数据评论计数为 2，但常规接口和原始 GitHub API 均只返回 1 条；未将不可访问内容计入已读。
- 所有 23 个正文/标题含 NaN 相关关键词的 PR 均核对合并状态；另追踪 7 个重要关联 PR，以及 MuJoCo Warp/RSL-RL 的关联讨论。
- 本报告整理文字证据、代码和修复链；没有运行每个 issue 的私有工程、全部模型附件和视频。未检索 GitHub Discussions，无法覆盖未公开、已删除或尚未索引的评论。

## 结论与我们最相关的证据

1. **NaN 不是一种单一故障。** 需要区分物理状态/接触先坏、传感器先坏、有限观测导致网络发散，以及日志统计中的除零。最终报错都可能落在策略 std 上，单看报错字符串无法区分。[#520](https://github.com/mujocolab/mjlab/issues/520)、[#870](https://github.com/mujocolab/mjlab/issues/870)、[#565](https://github.com/mujocolab/mjlab/issues/565)。
2. **最相似的几何问题是 #931。** 圆柱/轮形接触体在大地形上下边框处抖动，增大 border_width 更明显，左右边框不明显。维护者转交 MuJoCo 团队并因接触改进而关闭，但没有给出能唯一对应我们旧失败姿态的提交。它强烈支持优先核查轮—border 的后端接触，不证明两者必然同根因。[边框定位评论](https://github.com/mujocolab/mjlab/issues/931#issuecomment-4442191596)、[边框尺寸评论](https://github.com/mujocolab/mjlab/issues/931#issuecomment-4448196563)、[后续接触改进](https://github.com/mujocolab/mjlab/issues/931#issuecomment-4929513560)。
3. **过硬接触有真实案例。** #224 最终定位为特定 solref/solimp，在 GPU 上恢复默认值后改善。我们刚核查的 solref=(0.02,1)、solimp=(0.9,0.95,0.001,0.5,2) 已是默认值，因此没有证据要求继续硬化。[报告者最终结论](https://github.com/mujocolab/mjlab/issues/224#issuecomment-3482299538)。
4. **高度场后端确实有过 NaN 回归。** #467 经 mjwarp #995、mjlab #470 修复；#414 由 #423 关闭；#738 在更新 mjwarp 的 #810 后用户报告超过一万轮稳定。不能把这些物理缺陷全部解释成奖励权重问题。[#467](https://github.com/mujocolab/mjlab/issues/467)、[#470](https://github.com/mujocolab/mjlab/pull/470)、[#414](https://github.com/mujocolab/mjlab/issues/414)、[#738](https://github.com/mujocolab/mjlab/issues/738)。
5. **一个已明确修复的网络路径：缺少观测归一化。** PR #1104 于 2026-07-18 合并，关闭 #870/#1044/#1053，并为 #912 的地形法线拟合加入非有限协方差保护。我们当前 mjlab 提交包含它，Go2W actor/critic 也已 obs_normalization=True；无需重复补这项。[#1104](https://github.com/mujocolab/mjlab/pull/1104)。
6. **检测完善不等于物理修复。** #521 检查 sensordata，#522 增加观测检查，#400 阻止奖励传播并清理坏 warmstart；这些有助于定位和恢复，但不能证明原始接触算法不再产生 NaN。[#521](https://github.com/mujocolab/mjlab/pull/521)、[#522](https://github.com/mujocolab/mjlab/pull/522)、[#400](https://github.com/mujocolab/mjlab/pull/400)。
7. **轮 priority=1 有上游直接依据。** #267 的问题正是摩擦随机化被地面的最大值规则盖掉，维护者指出 G1/Go1 用足 priority=1。我们已完成同样的轮优先级设置；这改善摩擦语义，但不是已证明的 NaN 修复。[#267](https://github.com/mujocolab/mjlab/issues/267#issuecomment-3469964286)。

## 22 条直接关联 issue

表内“关闭”只是 GitHub 状态；具体属于根因修复、缓解、诊断或未复现关闭，见最后一列。

| Issue | 创建 / 关闭日期 | 核对后的结论 |
| --- | --- | --- |
| [#46 Add NaN guard with rolling state buffer and conditional dump on NaN/Inf](https://github.com/mujocolab/mjlab/issues/46) | 2025-09-18 / 2025-10-15 | 提出保存滚动状态历史和模型以定位 NaN/Inf；由 PR #183 引入 NaN guard。属于诊断能力，不是物理根因修复。 |
| [#174 Encountered issues with NaN values during training](https://github.com/mujocolab/mjlab/issues/174) | 2025-10-13 / 2025-10-15 | 自定义类 G1 模型在粗糙地形随机崩溃，换平地可完成训练；维护者要求 guard 转储。报告者称调整执行器数据有改善，另有操作任务用户补充模型姿态 NaN；没有确认一个适用于所有模型的根因。 |
| [#186 Objects start tunneling after some time in simulation](https://github.com/mujocolab/mjlab/issues/186) | 2025-10-15 / 2025-10-29 | 报告者最后定位到运行时随机化 geom_size；关闭该随机化后穿透消失。维护者解释尺寸改变后 geom_aabb、geom_rbound 等派生数据未同步。当时不支持这种直接尺寸随机化，不能把所有穿透都归因于求解器迭代不足。 |
| [#224 Training with multiple environments frequently crashes due to invalid `std`](https://github.com/mujocolab/mjlab/issues/224) | 2025-10-24 / 2025-11-02 | 4096 环境下可在首次 PPO 更新前出现物理 NaN；碰撞体减少后发生概率下降。最终报告者确认将 solimp='0.95 1.0 0.001 0.5 2'、solref='0.005 1.0' 恢复默认值后不再崩溃。同时促成 #227 修复故障环境捕获、#238 增加 NaN 终止。 |
| [#291 RuntimeError: normal expects all elements of std >= 0.0](https://github.com/mujocolab/mjlab/issues/291) | 2025-11-07 / 2025-11-13 | 网络的 log_std、actor、critic 参数都已成为 NaN；维护者要求按 FAQ 定位，后来因缺少后续信息关闭。网络损坏是观测到的结果，帖中把 log_std 称为根因没有充分证据。 |
| [#311 RuntimeError: normal expects all elements of std >= 0.0](https://github.com/mujocolab/mjlab/issues/311) | 2025-11-13 / 2025-12-12 | 恢复训练约 1.4 万或 2 万轮后 std 报错，改 log 参数化仍失败。讨论包含两个不同问题：一个用户遗漏 DC motor 的 effort_limit；原报告者怀疑课程计数未恢复。维护者推荐 #400 的奖励/warmstart 修复后关闭，不能将三种解释混成单一已证实根因。 |
| [#357 I added nan_guard however no NaN dump files are saved!](https://github.com/mujocolab/mjlab/issues/357) | 2025-11-24 / 2025-12-19 | 添加 nan_detection 终止项后没有生成 dump。终止项与启用滚动转储 guard 是不同机制；维护者询问是否真正发生 NaN，最后因无后续信息关闭，未确认新物理缺陷。 |
| [#399 Bug: nan value guard](https://github.com/mujocolab/mjlab/issues/399) | 2025-12-11 / 2025-12-11 | NaN 终止虽重置观测，奖励仍可能包含 NaN，且坏环境下一步继续异常。#400 修复奖励非有限值传播并在 clear_state 时清理 qacc_warmstart。 |
| [#412 Bug: nan value in contact data](https://github.com/mujocolab/mjlab/issues/412) | 2025-12-12 / 2025-12-27 | 自定义四足在平地、多 GPU 情况下，接触数据 NaN 进入 critic。维护者无法在给定条件下复现，最后关闭并继续征求最小复现；属于未确认算法根因的报告。 |
| [#414 Bug: Training Crash on Rough Terrain (G1) - `nefc overflow` and `RuntimeError`](https://github.com/mujocolab/mjlab/issues/414) | 2025-12-13 / 2025-12-18 | 复杂 G1 粗糙地形出现 nefc、contact match、CCD 警告及 NaN。增大容量先消除部分警告，但后来仍有 NaN。维护者最终指向 #423（92c36c2，含粗糙地形修正和 mjwarp 升级）确认解决。评论中的 LLM 四元数/高速旋转推断不是独立验证结论。 |
| [#467 Broken G1 Rough Terrain](https://github.com/mujocolab/mjlab/issues/467) | 2026-01-08 / 2026-01-09 | 维护者明确确认 mjwarp 高度场回归会导致 NaN，使 G1 rough 训练损坏。修复链为 mjwarp #995 → mjlab #470；这是已确认的后端碰撞缺陷。 |
| [#520 Nan in sensors is not caught in by nan guard and crashes training](https://github.com/mujocolab/mjlab/issues/520) | 2026-01-23 / 2026-02-15 | 接触传感器力先出现 NaN，旧 guard 未检查 sensordata，最终污染策略。#521 添加 sensordata 检查，#522 添加观测 NaN/Inf 处理。维护者未能复现接触 NaN 的物理起因；检测修复不能等同于接触算法已修复。 |
| [#534 Viewer crashes with certain asset orientations](https://github.com/mujocolab/mjlab/issues/534) | 2026-01-28 / 2026-02-15 | 最初是固定物体姿态下的 viewer/碰撞问题，报告者称显式配置碰撞后解决；随后又报告自定义操作奖励下的 std 错误。维护者未能复现后关闭；两种问题没有统一已确认根因。 |
| [#565 self.bin_count=1 could cause nan value in H_norm](https://github.com/mujocolab/mjlab/issues/565) | 2026-02-03 / 2026-02-16 | 跟踪任务仅一个采样 bin 时 H/log(bin_count) 除以零，产生 H_norm NaN。#641 已合并修复。与 Go2W 速度任务的物理接触没有直接对应。 |
| [#738 Training of environment Mjlab-Velocity-Rough-Unitree-Go1 crashes because of abnormal reward values](https://github.com/mujocolab/mjlab/issues/738) | 2026-03-06 / 2026-03-18 | 内置 Go1 rough 出现异常奖励和 std 错误。维护者让用户测试 c208410（#810，升级 mjwarp），用户随后报告超过 10000 轮无异常奖励，帖子关闭。计划中的奖励优化与实际验证过的依赖升级应分开看。 |
| [#765 RuntimeError: normal expects all elements of std >= 0.0 due to unknown cause](https://github.com/mujocolab/mjlab/issues/765) | 2026-03-10 / 2026-03-18 | B2W 类轮足平地训练 std 报错，物理 guard 没 dump；裁剪动作/观测后有改善但任务表现变差。维护者建议 log/std 下界；作者也承认把 ActionTermCfg.clip 当成缩放前裁剪是误用。没有证明 action_clip=None 就是所有案例的根因。 |
| [#846 [Bug] CUDA Error 700 (Illegal Memory Access) during step() with Muscle Actuators and Tendon Wrapping Geoms](https://github.com/mujocolab/mjlab/issues/846) | 2026-03-26 / 2026-07-18 | 肌肉/空间肌腱模型的 NaN 或 CUDA 700。作者后续指出纯 mjwarp 的 80/4/1 肌肉模型短测能通过，怀疑 mjlab 集成层；维护者要求最小复现，最后因无后续信息关闭。对我们的普通轮足执行器关联很低。 |
| [#870 Mjlab-Velocity-Rough-Unitree-Go1  normal expects all elements of std >= 0.0](https://github.com/mujocolab/mjlab/issues/870) | 2026-04-01 / 2026-07-18 | 维护者早期明确看到物理无 NaN、但 value loss/策略均值发散。后续不是一直悬而未决：#1104 确认 Go1 actor/critic 关闭观测归一化，极端但有限的粗糙地形观测可使 critic 发散，已启用归一化并关闭本 issue。 |
| [#912 [ObservationManager] NaN/Inf value](https://github.com/mujocolab/mjlab/issues/912) | 2026-04-16 / 2026-07-18 | 轮足复杂地形多个状态观测 NaN，即便 warn 清理观测，地形法线拟合 torch.linalg.eigh 仍崩溃。关闭事件关联 #1104：在特征分解前过滤非有限协方差。该修复防止一个坏环境导致整批 eigh 异常，不负责修复最初的物理发散。 |
| [#918 Roadmap for v1.5.0](https://github.com/mujocolab/mjlab/issues/918) | 2026-04-17 / 2026-07-18 | v1.5.0 规划将 NaN hunting 列为重点并关联 #870、#912，说明维护者将观测、奖励、物理步的定位分开处理。路线图关闭本身不能当作所有 NaN 已消失的证据。 |
| [#1044 "Mjlab-Velocity-Rough-Unitree-Go1" failing with "normal expects all elements of std >= 0.0"](https://github.com/mujocolab/mjlab/issues/1044) | 2026-06-04 / 2026-07-18 | 默认 Go1 rough 再现 std 报错；正文引用其他仓库关于未命中射线的猜测。实际关闭关联 #1104 的 Go1 观测归一化修复，因此不应把该猜测直接写成此 issue 的已确认根因。 |
| [#1053 why RuntimeError: normal expects all elements of std >= 0.0 happens](https://github.com/mujocolab/mjlab/issues/1053) | 2026-06-08 / 2026-07-18 | 自定义机器人在 4070 正常、4090 std 报错，用户称旧版 MuJoCo 3.4/早期 warp 可运行。最终也由 #1104 关闭；不能仅据此归因为某型号 GPU 故障。 |

## 26 条相关或排除项

| Issue | 当前状态 | 关联性与限制 |
| --- | --- | --- |
| [#101 Heightfield terrains are buggy](https://github.com/mujocolab/mjlab/issues/101) | 关闭 | 早期高度场编译慢、机器人下沉；关联 mjwarp #736，维护者指向高度场碰撞修复提交 7e54bd9 和 mjlab #406。 |
| [#193 About contact_sensor_maxmatch](https://github.com/mujocolab/mjlab/issues/193) | 关闭 | contact_sensor_maxmatch 溢出与 nconmax/njmax 是不同容量；单纯扩大后两者不消除传感器匹配溢出。 |
| [#267 Terrain friction randomization](https://github.com/mujocolab/mjlab/issues/267) | 关闭 | 足摩擦随机化为低值却仍抓地，报告者遗漏 foot priority=1。维护者确认高优先级足参数会覆盖默认地面摩擦。与本次提高轮 priority 的修改直接一致。 |
| [#403 During training, the episode_length is always 1, and during play the robot is floating in the air](https://github.com/mujocolab/mjlab/issues/403) | 关闭 | episode_length=1、机器人悬空，讨论集中于自定义任务/终止配置和正确回放入口；未定位 NaN。 |
| [#444 Regarding the Stability and Entropy Loss Issues in Velocity Task Training](https://github.com/mujocolab/mjlab/issues/444) | 关闭 | 站立足部滑移及熵损失是否正常，维护者只给出一般任务指标判断；没有 NaN 根因或修复证据。 |
| [#448 velocity  training](https://github.com/mujocolab/mjlab/issues/448) | 关闭 | 自定义机器人姿态奖励容差的解释及穿透观感，#450 增加注释。不是直接 NaN 报告。 |
| [#480 about foot collision](https://github.com/mujocolab/mjlab/issues/480) | 关闭 | 只报告 G1 足部触地振荡，没有评论或确认根因；不能据此判定是 Warp 回归。 |
| [#518 Excplicit actuators are unstable and unusable.](https://github.com/mujocolab/mjlab/issues/518) | 关闭 | 显式 DC/PD 执行器抖动。维护者指出 Python 默认值覆盖 XML 的 armature/frictionloss；报告者补齐这两个参数后确认解决。说明必须核对运行时值。 |
| [#547 Mesh collisions](https://github.com/mujocolab/mjlab/issues/547) | 关闭 | mesh 碰撞穿透，维护者通过 #549 升级 mjwarp。与形状/后端版本相关，但没有证明所有 mesh 或所有圆柱都不稳定。 |
| [#558 mujoco.FatalError: mj_stackAlloc: out of memory, stack overflow at mj_collision](https://github.com/mujocolab/mjlab/issues/558) | 关闭 | 向单一模型添加很多重叠 geom 导致原生 MuJoCo 碰撞栈溢出；维护者明确不是 mjlab 数值 NaN 问题，勿与 GPU 多环境容量混淆。 |
| [#588 [BUG] Resetting any env changes data for other non-reset envs.](https://github.com/mujocolab/mjlab/issues/588) | 关闭 | reset 后派生位置变化源于 mj_step 后 qpos/qvel 与派生量不同时刻；维护者确认应在控制步结束无条件 forward。状态一致性问题，不是已经证实的 NaN 根因。 |
| [#739 Problem with primitive geom collisions with floor](https://github.com/mujocolab/mjlab/issues/739) | 关闭 | mjwarp 3.5 下部分基本几何掉穿地面。维护者指向上游 #1192、清缓存/尝试 Warp 1.12，报告者确认有效；是特定版本/编译缓存情境。 |
| [#766 Setting contact physics params in MJLab](https://github.com/mujocolab/mjlab/issues/766) | 关闭 | 2026-03 讨论中维护者说明，当时 mjwarp 不支持全局 OVERRIDE 和 NoSlip，建议用每个 geom 的 solref/solimp/margin 等参数。历史能力说明不能无条件套到所有后续版本。 |
| [#783 Warning: opt.ccd_iterations, currently set to 500, needs to be increased.](https://github.com/mujocolab/mjlab/issues/783) | 关闭 | 3.5→3.6 后 CCD 迭代警告增多；部分源于警告中迭代变量显示错误，上游 #1217 修复显示，mjlab #814 升级。后续仍有警告，维护者称多为无害；并没有建议无限加大迭代数，也不能把所有真实碰撞异常都忽略。 |
| [#784 Robot penetrates terrain due to contact buffer overflow in large-batch environments (2048/4096)](https://github.com/mujocolab/mjlab/issues/784) | 关闭 | 2048/4096 环境、完整轮足碰撞时穿透，少量环境或 wheels-only 不复现。报告者称改 BuiltinPositionActuator 后消失，但维护者未确认执行器与容量之间的根因。标题的 buffer overflow 归因仍需实际溢出证据。 |
| [#838 TerrainHeightSensor reports height from bottom face of box terrain during penetration](https://github.com/mujocolab/mjlab/issues/838) | 关闭 | 足部穿入箱体表面时向下射线可能命中箱底，返回箱厚而非零，导致有限但错误的高度。与 #835 的射线未命中问题不同；不能统称 NaN。 |
| [#885 XML contact disable flag seems ineffective in runtime](https://github.com/mujocolab/mjlab/issues/885) | 关闭 | 维护者确认 MjSpec.attach 不合并子模型全局 option，必须使用 MujocoCfg。直接解释我们 XML 中 elliptic/impratio=100 没用于训练的现象。 |
| [#931 contact_sensor often produces many brief collisions that resolve mid-substep.](https://github.com/mujocolab/mjlab/issues/931) | 关闭 | 与我们最接近：轮足/圆柱在 terrain 上下边框振荡、穿透，左右边框较稳定，border_width 越大越明显。维护者转给 MuJoCo 团队；kbayes 后称已改进 mjwarp 接触，最终关闭。线程未给出一个可唯一对应当前故障姿态的补丁编号。 |
| [#1086 qfrc_constraint seems incorrect with multiple environments](https://github.com/mujocolab/mjlab/issues/1086) | 关闭 | 批量环境 qfrc_constraint 异常，上游 mjwarp #1500 确认稀疏求解器清零已收敛世界的约束力。该上游 issue 明确说不影响已求出的 qacc，不能直接当作 qacc NaN 的根因。 |
| [#1101 Air-time tracking inherits float32 sim-clock quantization error; compute_first_contact misses touchdowns as sim time grows](https://github.com/mujocolab/mjlab/issues/1101) | 关闭 | 接触时间累计使用 float32 仿真时钟导致长时间接触事件漏检；可能影响步态奖励，但不是 NaN 证据。 |
| [#1108 CUDA error 700](https://github.com/mujocolab/mjlab/issues/1108) | 关闭 | RTX4090、较新 mjlab/mjwarp、启动质量随机化且 n>=128 时 CUDA700；维护者最终确认上游修复并要求升级 mjlab 1.5.3。与晚期运行的接触 NaN 是不同阶段。 |
| [#1153 adaptive_uniform_ratio is not a uniform floor: effective strength scales with num_envs and failure rate](https://github.com/mujocolab/mjlab/issues/1153) | 开放 | 开放：跟踪任务自适应采样的 uniform 项不是固定概率下界，随失败数和环境数改变。候选 #1237 提到零总质量可导致 NaN，但该 PR 关闭未合并，不能算主线修复；与速度任务无直接关系。 |
| [#1170 BuiltinPdActuator applies swapped pos/vel targets when EntityCfg.sort_actuators=True](https://github.com/mujocolab/mjlab/issues/1170) | 开放 | 开放：sort_actuators=True 与双通道 BuiltinPdActuator 的控制排列不一致，可能把位置/速度目标写反。Go2W 当前使用分开的 BuiltinPosition/BuiltinVelocity，不直接对应。 |
| [#1197 `_eigh_3x3_jacobi` skips the rotation when two diagonal entries are equal, so `dr.pseudo_inertia` can write wrong principal moments](https://github.com/mujocolab/mjlab/issues/1197) | 关闭 | pseudo_inertia 的小型特征分解在相等对角元时漏旋转，写出错误主惯量；这条讨论不是 NaN 实例。当前 Go2W 也没有启用 pseudo_inertia。 |
| [#1216 Partial reset advances actuator delay buffers in environments that were not reset](https://github.com/mujocolab/mjlab/issues/1216) | 开放 | 开放：部分 reset 推进未重置环境的执行器延迟缓存，评论有可复现实验和候选 #1218。若启用延迟需额外核查；当前 Go2W 没配置该延迟。 |
| [#1233 Add depth_clipping_behavior to CameraSensorCfg (similar to Isaac Lab)](https://github.com/mujocolab/mjlab/issues/1233) | 开放 | CameraSensor 未命中时可返回 Inf，提议提供深度裁剪策略，仍开放。这可能是传感器未命中约定，不等于物理已经发散；我们的 MID360 用 RayCastSensor，并设置有限 miss_value。 |

## NaN 相关 PR 与合并状态

下面前 23 项是全量标题/正文关键词命中项，含文档和诊断工具改动；后 7 项为关键修复链的补充。关闭未合并的 PR 不计入已发布修复。

| PR | 状态 / 合并日期 | 主要作用 |
| --- | --- | --- |
| [#183 NaN Guard for Debugging Numerical Instability](https://github.com/mujocolab/mjlab/pull/183) | 已合并 2025-10-15 | 首次引入滚动转储与回放。 |
| [#227 Fixes to NaN Guard](https://github.com/mujocolab/mjlab/pull/227) | 已合并 2025-10-24 | 捕获所有环境、仅导出坏环境，修复故障环境编号大于前五个时漏存的问题。 |
| [#228 Use symlinks for latest NaN dumps instead of copies.](https://github.com/mujocolab/mjlab/pull/228) | 已合并 2025-10-24 | latest 文件改用符号链接；涉及 Windows 文件系统权限兼容。 |
| [#232 NaN vis / play viewer refactor](https://github.com/mujocolab/mjlab/pull/232) | 已合并 2025-11-07 | NaN 回放/Play viewer 重构。 |
| [#238 Adds NaN detection as a termination condition to prevent training crashes caused by NaN propagation.](https://github.com/mujocolab/mjlab/pull/238) | 已合并 2025-10-26 | 显式 opt-in 的 NaN 终止，属于防传播措施。 |
| [#283 Add information regarding Windows compatibility](https://github.com/mujocolab/mjlab/pull/283) | 已合并 2025-11-05 | 记录 Windows NaN guard 测试文件权限问题；不是 GPU 物理 NaN 修复。 |
| [#400 Fix NaN propagation in rewards and physics state.](https://github.com/mujocolab/mjlab/pull/400) | 已合并 2025-12-11 | 奖励非有限值清理和 qacc_warmstart 清零。 |
| [#443 fix: prevent NaNs in `apply_delta_pose` for zero rotation](https://github.com/mujocolab/mjlab/pull/443) | 已合并 2025-12-29 | apply_delta_pose 零旋转避免除零。 |
| [#446 chore: improve apply_delta_pose test and add modification note](https://github.com/mujocolab/mjlab/pull/446) | 已合并 2025-12-29 | 补充 #443 的测试和说明。 |
| [#470 Fixes heightfield collision issues with mujoco_warp update and adjusts sim params](https://github.com/mujocolab/mjlab/pull/470) | 已合并 2026-01-09 | 纳入 mjwarp #995 高度场碰撞修复。 |
| [#521 Add sensordata to NaN guard checks.](https://github.com/mujocolab/mjlab/pull/521) | 已合并 2026-01-23 | guard 增加 sensordata 检查。 |
| [#522 Add NaN/Inf checking to observation manager](https://github.com/mujocolab/mjlab/pull/522) | 已合并 2026-01-23 | 观测组提供 disabled/warn/sanitize/error 与逐项检查。 |
| [#641 Fix nan in H_norm when bin_count=1](https://github.com/mujocolab/mjlab/pull/641) | 已合并 2026-02-16 | 单 bin 时避免 H_norm 除零。 |
| [#707 Fix outdated examples and references in FAQ](https://github.com/mujocolab/mjlab/pull/707) | 已合并 2026-02-28 | 更新 FAQ 中 NaN 终止的配置示例。 |
| [#755 Capture mocap body poses in NaN guard](https://github.com/mujocolab/mjlab/pull/755) | 已合并 2026-03-09 | NaN dump 包含 mocap 位置/姿态和 state_spec。 |
| [#906 Add top-level --help for train/play, rename list_envs to list-envs](https://github.com/mujocolab/mjlab/pull/906) | 已合并 2026-04-15 | CLI 帮助及 list-envs 命名；仅提及 viz-nan，无 NaN 算法修复。 |
| [#1037 Fix terrain generation bugs and rework heightfield coloring](https://github.com/mujocolab/mjlab/pull/1037) | 已合并 2026-06-02 | 地形难度 0 等边界情况、渲染 NaN 和生成问题；不等同运行时物理 NaN。 |
| [#1078 Fix NaN in bad_orientation termination from unclamped acos](https://github.com/mujocolab/mjlab/pull/1078) | 已合并 2026-06-30 | bad_orientation 的 acos 输入钳位，避免终止条件被静默跳过。 |
| [#1104 Fix Go1 velocity training crashes and velocity-task curriculum/observation bugs](https://github.com/mujocolab/mjlab/pull/1104) | 已合并 2026-07-18 | Go1 归一化、地形法线非有限输入、编码器偏差观测和初始课程重置。 |
| [#1199 Keep massless bodies unchanged in dr.pseudo_inertia and reject log_uniform](https://github.com/mujocolab/mjlab/pull/1199) | 已合并 2026-09-29 | massless pseudo_inertia 除零保护，拒绝不适用的 log_uniform。 |
| [#1230 Add opt-in substep observation delay](https://github.com/mujocolab/mjlab/pull/1230) | 开放，未合并 | 物理子步观测延迟方案，含逐项检查调用；开放候选。 |
| [#1237 Add opt-in mixture mode for adaptive motion sampling](https://github.com/mujocolab/mjlab/pull/1237) | 关闭，未合并 | 自适应采样零质量保护及 mixture 模式提案；未合并。 |
| [#1261 Support MuJoCo 3.17 and fix batched backend state handling](https://github.com/mujocolab/mjlab/pull/1261) | 开放，未合并 | 3.17 支持与批量状态、reset、课程/历史、dump 文件兼容等；开放候选，不是现有主线能力。 |
| [#339 Warn when DcMotorActuator has unrealistic effort_limit.](https://github.com/mujocolab/mjlab/pull/339) | 已合并 2025-11-17 | DC motor effort_limit 合理性告警。 |
| [#406 Upgrade warp-lang and mjwarp.](https://github.com/mujocolab/mjlab/pull/406) | 已合并 2025-12-12 | 早期升级 Warp 和 mjwarp，包含高度场改进。 |
| [#423 Various rough terrain fixes.](https://github.com/mujocolab/mjlab/pull/423) | 已合并 2025-12-18 | 粗糙地形系列修正、初始姿态/高度场等和 mjwarp 升级，关闭 #414。 |
| [#549 Update mjwarp dependency to 1dc288cf1fa819fc3346ec5c9546e2cc2b7be667](https://github.com/mujocolab/mjlab/pull/549) | 已合并 2026-02-01 | mesh/capsule 问题关联的 mjwarp 升级。 |
| [#810 Update mujoco-warp to ada89af7](https://github.com/mujocolab/mjlab/pull/810) | 已合并 2026-03-17 | 升级到 mjwarp ada89af7，#738 用户验证所用提交。 |
| [#814 Update mujoco-warp to bb928a1d](https://github.com/mujocolab/mjlab/pull/814) | 已合并 2026-03-18 | 升级到 mjwarp bb928a1d，关联 CCD 警告问题。 |
| [#835 Fix TerrainHeightSensor reporting max_distance during ground contact](https://github.com/mujocolab/mjlab/pull/835) | 已合并 2026-03-23 | TerrainHeightSensor 未命中时的高度回退；关联奖励正确性。 |

## 上游问题链与容易误用的建议

- [mjwarp #736](https://github.com/google-deepmind/mujoco_warp/issues/736)：早期高度场索引与接触异常；对应 mjlab #101。
- [mjwarp #995](https://github.com/google-deepmind/mujoco_warp/pull/995)：在 heightfield 局部坐标系计算碰撞，已合并；对应 #467/#470。
- [mjwarp #921](https://github.com/google-deepmind/mujoco_warp/pull/921)、[#922](https://github.com/google-deepmind/mujoco_warp/pull/922)：#414 评论曾测试这些候选，查询时均未合并；不能写成正式修复已包含这两个 PR。
- [mjwarp #1217](https://github.com/google-deepmind/mujoco_warp/pull/1217)：修正 EPA 警告使用的迭代计数变量，已合并。它主要修正警告，不足以证明接触点/法线正确。
- [mjwarp #1500](https://github.com/google-deepmind/mujoco_warp/issues/1500)：稀疏求解器清零已收敛环境的 qfrc_constraint。报告明确说 qacc 不受该问题影响；不能据此解释所有物理 NaN。
- [RSL-RL #190](https://github.com/leggedrobotics/rsl_rl/pull/190)：旧版 std 下界方案，查询时未合并。PR 作者关于具体采样内核检查的说法没有在本调研中复现，不能当作 PyTorch 所有版本的事实。我们安装的 RSL-RL 5.5.1 已有 std_range=(1e-6,1e6) 钳位；scalar 和 log 路径都应用。NaN 参数仍可能穿过 clamp，因此不能把切换 log 当成通用根因修复。
- #414 部分评论直接包含 LLM 自动分析，诸如以某个普通角速度认定四元数归一化故障，证据不足。本报告以维护者确认的修复、可复现代码和作者的控制变量结果区分可信度。
- #783 清缓存/调 tolerance 的建议在后续评论中并未对所有用户奏效；不要把它写成已验证通用解决方案。nconmax、njmax、contact_sensor_maxmatch 分别负责不同容量，需要结合实际 overflow 信息。

## 对我们当前配置的核对

审计基线为本机验证环境：mjlab Git 提交 `8bb2bc9e187797689ec82c78c35019afd2fd0376`（源码版本 1.6.0）、MuJoCo/mujoco-warp 3.16.0、Warp 1.18.0、RSL-RL 5.5.1。这里的“已包含”是提交祖先和本机代码检查，不等同训练机已经安装成功。

| 项目 | 本机核对结果 | 对此次排查的含义 |
| --- | --- | --- |
| #400 奖励/warmstart | 合并提交是当前 mjlab 的祖先；代码有奖励 nan_to_num、warmstart 清理 | 不能再按未修复的旧 guard 行为推断 |
| #521/#522 | 合并提交已包含；guard 检查 sensordata，观测支持逐项 error | 捕获接触/传感器先 NaN 的路径已有基础能力 |
| #1104 | 已包含；Go2W actor/critic 都启用 obs_normalization | Go1 当时缺归一化的根因不能原样套用 |
| #1078 acos | 已包含，bad_orientation 先 clamp 到 [-1,1] | 不需再补同样钳位 |
| #641 单 bin | 已包含，且当前运行速度任务 | 与当前轮地接触故障无直接对应 |
| #1199 massless | 已包含；当前事件使用 body_com_offset，未启用 pseudo_inertia | 与此项质量/惯量除零问题不直接匹配 |
| std 范围 | RSL-RL 实际代码已有 scalar/log 两条路径的范围钳位 | 仍应区分物理先坏与 PPO 更新先坏，不盲改 std_type |
| 接触软硬 | solref=(0.02,1)，solimp=(0.9,0.95,0.001,0.5,2) | 已避开 #224 报告的特别硬配置 |
| 轮地摩擦 | 轮 condim=3、priority=1；非轮 condim=1、priority=0 | 与 #267 维护者说明一致 |
| margin / 基础摩擦 | 非轮 margin=0、滑动摩擦=1；轮 margin=1 mm、基础 friction=(0.8,0.02,0.01) | 这是接触实验条件变化；没有 issue 证据证明它已彻底消除 NaN |
| V2 容量 | nconmax=256、njmax=4096、contact_sensor_maxmatch=128 | 数值高于一些旧帖初始值不代表永不会溢出，需要实际计数 |
| RayCast 未命中 | height_scan 用有限 miss_value；MID360 扫描指定 miss_value=5 | 未命中约定与物理状态先 NaN 分开定位 |
| 观测检查默认值 | ObservationGroupCfg.nan_policy 默认 disabled，nan_check_per_term 默认 True | 只有显式启用 error/warn 等策略，逐项检查才生效 |

当前仓库自己的 PhysicsNanCapture 已覆盖 step 和 forward，检查 ctrl、qacc_smooth、qfrc_actuator、qfrc_constraint、sensordata 等，并保存原始控制/故障帧/接触和约束。它比很多早期 issue 的转储保留更多信息。

收到的旧 `nan_dump_20261024_000140_473518.json` 记录的是 **mjlab 1.2.0 + MuJoCo/mjwarp 3.5.0 + Warp 1.12.0、4096 环境**，在 after_step 阶段同一环境的 qpos/qvel/qacc/qacc_warmstart/sensordata/qfrc_constraint 非有限。它支持物理路径需要优先排查；不能当作新 3.16 环境已经失败，也不能只根据最后一个 std 报错归因于策略噪声。

## 建议的后续定位顺序

1. **先在训练机确认实际运行版本。** 区分旧 3.5 转储、新 3.16 环境、尚未合并的 3.17 候选；同时记录 GPU、驱动和 Warp 缓存所属版本。
2. **优先做轮—border 的最小物理复现。** 保留失败时的圆柱姿态、原始 qpos/qvel、ctrl、warmstart、逐环境随机化值，比较旧 3.5 与新 3.16 的原生 MuJoCo、MJWarp CPU、目标 GPU。单独固定种子、边框尺寸和 margin，避免同时变化掩盖原因。
3. **捕获第一个非有限字段。** 物理先坏时比较 contact.pos/dist/frame、efc 和 qfrc_constraint；物理/传感器全有限但 PPO 更新后坏时，再追踪 value、advantage、log_prob、ratio、gradient 和 std。不要用普通观测清零代替抓取现场。
4. **容量按实际 overflow 调整。** 分清约束、接触、传感器匹配三类；CCD 迭代数也不能代替容量设置。不要仅因为旧帖出现 500/2000 就照搬。
5. **保留接触默认软硬作为基准。** 目前没有证据建议增大 solimp 或减小 solref；若比较 margin，应单独改一项。
6. **若将来启用随机惯量、执行器延迟或双通道 BuiltinPd，重新检查对应开放 issue。** 当前任务不涉及的模块无需为排 NaN 一并改动。

本次为调研记录，没有依据帖子猜测继续修改训练逻辑、奖励、裁剪或求解器设置。

## MID360 原仓库的依赖版本与时间线

这里的“原仓库”指 [1YI-DING/unitree_rl_mjlab_MID360-LiDAR](https://github.com/1YI-DING/unitree_rl_mjlab_MID360-LiDAR)，不是我们后续修改的 Renyu-ha 分支。2026-10-24 再次核查其公开 main：只有 2026-05-22 的两次提交，初始提交 55d3e96 与 README 提交 01cbe73；两者的依赖声明相同，第二次只修改 README。

### 原仓库实际声明了什么

| 组件 | 原仓库安装时的约束 | 证据及能确定的范围 |
| --- | --- | --- |
| mjlab | **==1.2.0** | 原仓库 setup.py 直接固定 |
| mujoco-warp | **==3.5.0** | 原仓库 setup.py 直接固定，不能自动获得后续 3.5.0.1/3.5.0.2 或 3.6+ 的修复 |
| RSL-RL | **==5.0.1** | mjlab 1.2.0 的传递依赖固定 |
| MuJoCo | **>=3.5.0** | mjlab 1.2.0 的传递依赖最低要求，没有固定精确版本 |
| Warp | **>=1.12.0** | mjlab 1.2.0 的传递依赖最低要求，没有固定精确版本 |
| PyTorch | **>=2.7.0** | mjlab 1.2.0 的传递依赖最低要求，没有固定精确版本 |
| Python | 文档推荐 **3.11** | 原仓库 setup_zh.md 的安装步骤 |

来源：[原仓库 setup.py](https://github.com/1YI-DING/unitree_rl_mjlab_MID360-LiDAR/blob/01cbe731b86a33e1c36afeed5fdd6f44d0afccf6/setup.py)、[原仓库安装说明](https://github.com/1YI-DING/unitree_rl_mjlab_MID360-LiDAR/blob/01cbe731b86a33e1c36afeed5fdd6f44d0afccf6/doc/setup_zh.md)、[mjlab v1.2.0 依赖声明](https://github.com/mujocolab/mjlab/blob/v1.2.0/pyproject.toml)。原仓库没有完整训练依赖锁文件，因此这些声明不足以还原原作者训练机上所有组件的精确安装版本。

另需区分 mjlab 自己的源码开发环境：其 v1.2.0 的 uv.lock 锁了 MuJoCo 3.5.1.dev870253531，以及 Git 提交 5a86ec28 的 mujoco-warp（元数据版本 3.5.0.2）。这不是 MID360 文档要求的普通 pip 安装结果；pip 安装 MID360 不会使用依赖包的 uv.lock，也不会读取依赖包的 tool.uv.sources。[mjlab v1.2.0 uv.lock](https://github.com/mujocolab/mjlab/blob/v1.2.0/uv.lock)。

### 时间线（北京时间）

| 日期 | 上游版本 / 事件 | 对 MID360 和 NaN 排查的意义 |
| --- | --- | --- |
| 2026-02-13 | [mujoco-warp 3.5.0 发布](https://github.com/google-deepmind/mujoco_warp/releases/tag/v3.5.0) | 后来 MID360 固定使用的后端版本 |
| 2026-03-07 | [mjlab 1.2.0 发布](https://github.com/mujocolab/mjlab/releases/tag/v1.2.0) | MID360 后来固定使用的框架版本；RSL-RL 5.0.1 |
| 2026-03-17 | [#810 更新 mjwarp 到 ada89af7](https://github.com/mujocolab/mjlab/pull/810)，mjlab 提交 c208410 | #738 用户在更新后报告 Go1 rough 训练超过一万轮稳定；晚于 1.2.0，原固定组合未获得这次依赖更新 |
| 2026-04-15 | [mjlab 1.3.0 发布](https://github.com/mujocolab/mjlab/releases/tag/v1.3.0) | 官方转向 MuJoCo 3.7.0 / mjwarp 3.7.0.1，并包含执行器、轮关节等修正 |
| 2026-05-22 19:38 | [MID360 首个可见提交 55d3e96](https://github.com/1YI-DING/unitree_rl_mjlab_MID360-LiDAR/commit/55d3e963f19bea9428b6faa09bf261770d6d513b) | 已有 1.3.0 时仍明确固定 mjlab 1.2.0 + mjwarp 3.5.0；首次提交日期不等于作者开始开发的日期 |
| 2026-05-22 19:49 | [MID360 更新 README，01cbe73](https://github.com/1YI-DING/unitree_rl_mjlab_MID360-LiDAR/commit/01cbe731b86a33e1c36afeed5fdd6f44d0afccf6) | 未更新依赖；截至本次核查仍为原仓库 main 最新提交 |
| 2026-05-27 | [mjlab 1.4.0 发布](https://github.com/mujocolab/mjlab/releases/tag/v1.4.0) | 官方转向 MuJoCo 3.8 / mjwarp 3.8.0.3+（3.8 系列） |
| 2026-06-29 | [mjlab 1.5.0 发布](https://github.com/mujocolab/mjlab/releases/tag/v1.5.0) | 官方转向 MuJoCo / mjwarp 3.10，取消开发用 nightly/Git 后端来源 |
| 2026-07-18 04:19 | [#1104 合并](https://github.com/mujocolab/mjlab/pull/1104)，提交 3cf17ee | 修复 Go1 缺少观测归一化造成的训练发散，并为非有限地形法线拟合加保护；关闭 #870/#1044/#1053/#912 |
| 2026-07-18 04:21 | [#931 关闭](https://github.com/mujocolab/mjlab/issues/931) | 轮形碰撞体与地形上下边框的相关问题因接触改进而关闭；讨论没有给出可唯一对应的修复提交，不能指定一个确定的首次修复版本 |
| 2026-07-22 | [mjlab 1.5.3 发布](https://github.com/mujocolab/mjlab/releases/tag/v1.5.3) | mjwarp 3.10.0.3 修复特定 GPU 上质量随机化导致的 CUDA 700；这与长期运行后的物理 NaN 不是同一种已证实故障 |
| 2026-08-09 | [mjlab 1.6.0 正式发布](https://github.com/mujocolab/mjlab/releases/tag/v1.6.0) | 正式标签依赖 MuJoCo / mjwarp 3.11；包含多项部分环境 reset 状态泄漏修复 |
| 2026-10-14 | [mjwarp 3.16.0 发布](https://github.com/google-deepmind/mujoco_warp/releases/tag/v3.16.0) | 当前采用的后端版本 |
| 2026-10-19 03:04 | [mjlab main 提交 8bb2bc9](https://github.com/mujocolab/mjlab/commit/8bb2bc9e187797689ec82c78c35019afd2fd0376) | 我们当前锁定的源码，版本字段仍是 1.6.0，但依赖已为 MuJoCo / mjwarp 3.16、RSL-RL 5.5.1；不能只凭“1.6.0”与正式标签等同 |
| 2026-10-23 07:38 | [mjwarp 3.17.0 发布](https://github.com/google-deepmind/mujoco_warp/releases/tag/v3.17.0) | 发布了新的后端，不代表当时锁定的 mjlab main 已正式支持它 |
| 2026-10-23 23:08 → 23:57 | 本项目提交 33378ee → 82b5a99 | 先建立 3.17 适配分支，随后当前分支改回与锁定 mjlab main 一致的 3.16 路线；是本项目迁移历史，不是原作者历史 |
| 2026-10-24 00:01 | 收到的旧环境 NaN 转储 | 实际记录 mjlab 1.2.0 / MuJoCo 3.5.0 / mjwarp 3.5.0 / Warp 1.12.0 / Torch 2.14.1；证明这次失败来自旧运行环境 |
| 2026-10-24 10:34 | 本项目当前提交 242dbd7 | 核心安装约束写入标准元数据：8bb2bc9、MuJoCo / mjwarp 3.16.0、Warp 1.18.0、RSL-RL 5.5.1、Torch 2.14.1 |

已通过 Git 提交祖先关系核对：1.2.0 本来就包含早期 #400 奖励/warmstart 和 #470 高度场修复；它不包含后来的 #810、#1078、#1104。当前 8bb2bc9 包含上述全部 mjlab 修改。#810 的核心是后端依赖更新，不能仅复制 mjlab Python 代码就认为后端修复也已安装。

因此，原 MID360 是明确固定在较早框架/后端的一套工程。升级可以取得之后的已知修复，但不能据此宣布当前轮—边框 NaN 已消除；仍需在目标 GPU 上用同一失败状态比较旧 3.5 与新 3.16。
