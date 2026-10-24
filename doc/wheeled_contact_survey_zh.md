# 轮足机器人碰撞与控制参数调研

核对日期：2026-10-23。以下数据来自固定提交的模型文件、任务配置、公共基类和执行器调用路径。没有运行这些外部仓库的训练，因此“声明圆柱”与“导入后实测形状”分开表述，也不据此宣称某组参数已解决本仓库的 GPU NaN。

## 直接相关的发现

- HIMLoco-Go2W、DreamWaQ-Go2W 的训练 URDF 都用圆柱描述轮胎，半径 86 mm、全宽 45 mm，且明确关闭 IsaacGym 的圆柱转胶囊选项。
- 公共 robot_lab 的 Go2W URDF 也声明相同尺寸的圆柱；B2W 则声明轮胎 mesh。轮足项目并没有统一使用一种碰撞形状。
- 查到的三套外部 Go2W 配置，名义轮速度增益均为 0.5，力矩上限约 23.5–23.7 Nm。我们当前为 2 和 45 Nm；这值得后续单独对照，但不能单凭数字判断是 NaN 根因。
- 双轮 Wheel-Legged-Gym 的 URDF 虽写圆柱，导入选项却会将其替换为胶囊，说明只看模型文件不够。

## 1. 碰撞体和模型

R 为半径，L 为沿轮轴的全宽。下表中的原点均在各自轮连杆坐标系内，不能脱离连杆变换直接互换。

| 项目 | 轮胎碰撞声明与导入设置 | 名义轮连杆质量 | 自碰撞配置 |
| --- | --- | --- | --- |
| HIMLoco-Go2W | URDF cylinder，R=0.086 m、L=0.045 m；圆柱不转胶囊；轮内原点 `(0,0,0)` | 0.520 kg，另有 0.646 kg 固定电机连杆 | IsaacGym `self_collisions=1`，关闭 |
| DreamWaQ-Go2W | URDF cylinder，R=0.086 m、L=0.045 m；圆柱不转胶囊；左轮原点 Y=0.05 m | 0.520 kg，另有固定电机连杆 | `self_collisions=0`，开启 |
| 公共 robot_lab / Go2W | URDF cylinder，R=0.086 m、L=0.045 m；左轮原点 Y=0.048 m；未实测转换后碰撞表示 | 0.520 kg，固定电机连杆单列 | PhysX 与 Newton 配置均关闭 |
| 公共 robot_lab / B2W | URDF mesh（如 `FL_foot.dae`）；未实测导入后的凸体表示和半径 | 1.083 kg | PhysX 与 Newton 配置均关闭 |
| Wheel-Legged-Gym，双轮六关节 | URDF cylinder，R=0.0675 m、L=0.025 m；`replace_cylinder_with_capsule=True` 实际传入 AssetOptions | 1.22 kg | IsaacGym `self_collisions=1`，关闭 |
| 我们当前圆柱实验 | 原生 MJCF cylinder，R=0.086 m、L=0.045 m；四轮轴向与关节 Y 轴对齐，外观保留 mesh | 原有 0.98 kg 与显式惯量不变 | 开启；保留同刚体和直接父子过滤 |

模型来源：[HIM URDF][him-model]、[DreamWaQ URDF][dream-model]、[robot_lab Go2W URDF][rl-go2-model]、[robot_lab B2W URDF][rl-b2-model]、[Wheel-Legged-Gym URDF][wlg-model]。导入与自碰撞设置见各项目的[HIM 任务][him-cfg]、[DreamWaQ 任务][dream-cfg]、[robot_lab 资产][rl-asset]、[双轮基类][wlg-base]及[导入实现][wlg-runtime]。

HIM 仓库另有 MuJoCo XML，但其训练任务配置指向 URDF；不能用旁边的 MJCF 接触参数解释 IsaacGym 训练。URDF 与导入选项也不能证明 IsaacGym 内部和 MuJoCo 原生圆柱使用同一种接触算法。本表没有把 Go2W 的 `foot_motor` 电机壳圆柱误当成 `foot` 轮胎。各模型对固定连杆、轮胎与电机的质量拆分不同，不能只复制轮连杆的 0.520 kg。

### 当前圆柱是否偏宽

缩窄前（提交 `4699c76`）编译后的四个轮碰撞体尺寸均为 `size="0.086 0.025907457"`。MuJoCo cylinder 的第二个 size 值是半宽，因此原全宽为 51.814914 mm，比上面三套 Go2W 的 45 mm 圆柱宽 6.814914 mm，约 15.14%。当前四轮已改为 `size="0.086 0.0225"`，全宽 45 mm，轮心不变，每侧边缘收进约 3.407457 mm。

之前用 51.815 mm 是为了保留原 wheel STL 的轴向完整包围盒：左轮局部 Y 范围约 24.856799–76.671713 mm，中心 50.764256 mm。这不是其他仓库的轮胎宽度参数，也不能等同于胎面的有效承载宽度。原网格有圆弧肩部，最大半径附近明显变窄；包围盒圆柱则把整个 51.815 mm 宽度都扩展到 86 mm 半径，肩部的几何会比原网格更饱满。

以当前左轮局部零轮角、轮轴水平的几何剖面为例，原网格凸包最低点上方 1 mm 以内的轴向范围约 15.871 mm；上方 2 mm 以内约 23.601 mm。圆柱在同样高度范围内均覆盖全宽。这是固定姿态下的几何测量，未计算接触求解器实际的接触面积或轮胎形变。

按用户要求，本次仅将四个圆柱的第二个 size 值改为 `0.0225`，保留半径 86 mm、中心位置、质量、惯量与控制参数。45 mm 圆柱仍不能完全复现圆弧胎面；需要将宽度对照与原 mesh/圆柱形状对照分开。宽度 A/B 可用提交 `4699c76` 的 51.815 mm 圆柱作为基线，两者训练配置相同。

本次重新获取并用 XML 解析器核对了三套 Go2W 的四个有效轮胎 collision，均声明 86 mm 半径和 45 mm 全宽。公共 robot_lab URDF 的注释中还保留旧 mesh 定义，比较时应忽略这些注释；此处的数据是有效 XML 元素的声明，不是其实机尺寸测量。

## 2. 控制和动作

轮子的 KP=0 时，理想未饱和速度反馈为 `tau = KD * (target_velocity - velocity)`。实际还受求解器、积分方式、随机增益、关节摩擦及电机力矩限幅影响；显式算力矩与原生速度执行器不能仅凭同一个 KD 认为完全等价。

| 项目 | 轮子控制、KP/KD | 轮力矩限制 | 轮动作 scale | 腿 KP/KD；位置 scale | 物理步长 / decimation |
| --- | --- | --- | --- | --- | --- |
| HIMLoco-Go2W | 显式算力矩，0 / 0.5 | URDF effort=23.7 Nm，代码裁剪 | 10 rad/s / action | 40 / 1；0.25 rad / action | 0.005 s / 4，50 Hz 策略 |
| DreamWaQ-Go2W | 显式算力矩，0 / 0.5 | 23.7 Nm；`urdf_limit_scale=1` | 10 | 40 / 1；0.25 | 0.005 / 4 |
| 公共 robot_lab / Go2W | ImplicitActuator，0 / 0.5 | 配置 joint effort=23.5 Nm | 5 | 25 / 0.5；髋 0.125、其他 0.25 | 0.005 / 4 |
| 公共 robot_lab / B2W | 轮 ImplicitActuator，0 / 1 | 20 Nm | 5 | 腿 DC motor，160 / 5；髋 0.125、其他 0.25 | 0.005 / 4 |
| Wheel-Legged-Gym | 显式算力矩，0 / 0.5 | URDF effort=5 Nm，代码裁剪 | 10 | 40 / 1；0.5 | 0.005 / 2，100 Hz 策略 |
| 我们 Go2W | MuJoCo BuiltinVelocity，速度增益 2 | 45 Nm | 20；现与本仓库 G1 一样不裁剪动作 | 髋/大腿 20 / 1，小腿 40 / 2；0.5 | 0.005 / 4 |

配置来源：[HIM][him-cfg]及[力矩实现][him-runtime]、[DreamWaQ][dream-cfg]及[力矩实现][dream-runtime]、[robot_lab 资产][rl-asset]与 [Go2W][rl-go2-cfg] / [B2W][rl-b2-cfg] 任务、[双轮任务][wlg-cfg]与[公共基类][wlg-base]。

此前采用的 [ZhongL1/mjlab 资产配置](https://github.com/ZhongL1/unitree_rl_mjlab_go2w/blob/fe0deeaef2e6e579db63b782f1082b5d2b4ed071/mjlab/asset_zoo/robots/unitree_go2w/go2w_constants.py) 本身就使用 BuiltinVelocity、轮速度增益 2、力矩上限 45 Nm、armature=0.02；我们的 2 / 45 来源于这份配置，轮 armature 后按用户实验要求设为 0。它说明 Go2W 参数并非所有仓库都取 0.5 / 23.5，比较时还要考虑同一仿真后端和模型版本。

动作 scale 是换算系数，不能直接解释为最大轮速。HIM、DreamWaQ 与双轮项目的 `clip_actions=100`；公共 robot_lab 任务另设 action clip，语义也应按所用 IsaacLab 版本核对。我们在初次迁移中曾裁剪原始动作到 ±1，后按用户要求与本仓库 G1 对齐而关闭；当前轮目标为 `20 * action`，不是限制在 ±20 rad/s。HIM/DreamWaQ 的 URDF wheel velocity 声明为 30.1 rad/s，公共 robot_lab Go2W 执行器声明 30 rad/s，B2W 为 50 rad/s；声明速度限制不代表所有显式力矩路径都主动裁剪轮速。

HIM 与双轮 AssetOptions armature=0；DreamWaQ 同样继承 0，且 `wheel_armature_add=0`。我们当前轮 armature=0，腿为 0.01/0.02。圆柱替换不调整这些参数。公开 robot_lab 的资产配置没有在这里显式设置轮 armature，不假定其依赖库默认值与 MuJoCo 相同。

## 3. 接触、随机化与轮相关奖励

| 项目 | 接触/求解器配置 | 摩擦随机化 |
| --- | --- | --- |
| HIM 与 DreamWaQ | PhysX TGS；位置迭代 4、速度迭代 0；contact_offset=0.01 m、rest_offset=0；max_depenetration_velocity=1 m/s；substeps=1 | 启用，0.25–1.25；地形静/动摩擦各 0.8 |
| Wheel-Legged-Gym | 同样 TGS 4/0、offset 0.01/0、最大去穿透速度 1 | 启用，0.1–2.0 |
| 公共 robot_lab | PhysX 资产 Go2W 4/1、B2W 4/0，最大去穿透速度 1；另提供 Newton 配置，不将 PhysX 参数当成另一后端的实际设置 | 公共事件静摩擦 0.3–1.0、动摩擦 0.3–0.8、恢复系数 0–0.5 |
| 我们 Go2W v2 | MuJoCo/MJWarp；iterations=10、ls_iterations=20、ccd_iterations=50；非轮碰撞体 condim=1、priority=0，轮 condim=3、priority=1 | 现对齐本仓库 G1 v2，轮 geom friction 启动随机化 0.25–1.8；四轮共享抽样，轮优先级高于默认地面，避免低摩擦抽样被地面摩擦覆盖 |

来源：[HIM 基类][him-base]、[DreamWaQ 基类][dream-base]、[双轮基类][wlg-base]、[robot_lab 资产][rl-asset]及[公共环境][rl-env]。PhysX `contact_offset`、MuJoCo `margin`、软约束 `solref/solimp` 不是一一对应的参数，求解器迭代次数也不能直接横向比较。MuJoCo 的接触参数解释见[官方模型文档](https://mujoco.readthedocs.io/en/stable/modeling.html#solver-parameters)。

这些项目也通过奖励约束运动。HIM 对腿/机身非法接触给 `collision=-1`，`feet_stumble=-0.1`；后者比较轮连杆水平接触力与竖直接触力（水平范数 > 3 倍竖直力绝对值）。DreamWaQ 同名项也是 -1 / -0.1，并有轮加速度项 -1e-7。公共 robot_lab Go2W 将普通 `feet_slide` 设为 0，轮加速度惩罚为 -2.5e-9；不能把其普通足滑移项当成已启用的轮胎纵向滑移约束。以上是各自原始权重，时间步缩放、求和方式、速度单位不同，不能直接搬到我们奖励管理器。[HIM 配置][him-cfg]与[实现][him-runtime]、[DreamWaQ 配置][dream-cfg]、[robot_lab Go2W 任务][rl-go2-cfg]。

## 4. 与你们本地 robot_lab 的区别

本地只读核对：`D:/DeskTop/Github/robot_lab`，提交 `1a8193fa8635088600c33c7957473ce289f514f5`。它已使用 `loco_lab` 包，不能与上表公共 `fan-ziqi/robot_lab` 混为同一个配置。

- `source/loco_lab/loco_lab/tasks/physx/assets/unitree.py` 的 Go2W 使用外部绝对路径 `go2w_v4.usd`，该 USD 不在本次可检查的仓库文件中，因此尚不能确认其实际轮碰撞体。
- 轮执行器为 ScaledDCMotor：KP=0、KD=0.5；actuator effort 与 saturation effort 均为 23.5 Nm，速度参数为 30 rad/s；joint effort=1e9 是另一层求解器限制，不能误读成电机可以输出 1e9 Nm。还有 0.02 的关节摩擦/动态摩擦。电机模型可能随速度与故障状态进一步降低有效力矩。
- 自碰撞开启，PhysX 配置位置/速度迭代 4/0，最大去穿透速度 1 m/s。
- 当前 `.../config/wheeled/go2w/tolerance/rough_env_cfg.py` 的 wheel_scale=1，腿 scale 为髋 0.25、大腿/小腿 0.5；动作实现还乘 motor_strength，并包含故障目标处理。因此此值也不是公共 robot_lab 的 scale=5。

## 5. 本仓库的对照方式

最初圆柱实验仅替换四个轮碰撞体，提交为 `ea0e9a8`，圆柱前版本为 `6ff9526`。随后用户要求通用设置与本仓库 G1 对齐，已关闭动作裁剪并更新感知、随机化、地形与课程，详见 [G1 对齐记录](go2w_g1_alignment_zh.md)。若做纯碰撞形状的 A/B 比较，应在同一套训练设置下仅改变轮碰撞几何，不能直接把前后多项改动的结果归因于圆柱。

此次在 `4699c76` 基础上将四轮碰撞体全宽从 51.814914 mm 缩到 45 mm，训练配置、轮心、半径和其他物理参数保持一致，用于单独比较轮宽的影响。

最初圆柱版通过了编译模型差异检查、38 项回归检查，以及 2 个 MJWarp CPU 环境的 16 个控制步、1 次 PPO 更新和 ONNX 导出。此次 45 mm 版通过 42 项回归检查；与缩窄前编译模型的 454 个数组字段对比，仅四个轮的宽度及其派生包围体变化。单个 MJWarp CPU 环境在最终稀疏阶段完成 16 个控制步、1 次 PPO 更新、ONNX 导出和推理，观测与策略输出有限。仍需在原 GPU 训练规模下，使用相同种子、课程阶段和 NaN 检查做对照。旧 MJB 的严格几何校验会正确拒绝新圆柱模型；交互检查与训练命令见[Go2W 说明](go2w_perceptive_zh.md#使用-onnx-交互检查边界)。

如果圆柱对照后仍出现接触异常，再单独测试轮速度增益、力矩限幅、时间步或接触求解设置。例如可先仅将轮速度增益从 2 降到 0.5，而保持其他参数不变；这只是依据源码差异提出的实验，不是已验证的修复。本次没有自动应用其他项目的增益、力矩、奖励或关闭自碰撞。

[him-model]: https://github.com/TrackinBIT/HIMLoco-for-Go2W/blob/011693738c61603c3f22f2bce755098dd36fa7eb/resources/robots/go2w/urdf/go2w.urdf
[him-cfg]: https://github.com/TrackinBIT/HIMLoco-for-Go2W/blob/011693738c61603c3f22f2bce755098dd36fa7eb/legged_gym/envs/go2w/go2w_config.py
[him-base]: https://github.com/TrackinBIT/HIMLoco-for-Go2W/blob/011693738c61603c3f22f2bce755098dd36fa7eb/legged_gym/envs/base/legged_robot_config.py
[him-runtime]: https://github.com/TrackinBIT/HIMLoco-for-Go2W/blob/011693738c61603c3f22f2bce755098dd36fa7eb/legged_gym/envs/base/legged_robot.py
[dream-model]: https://github.com/ShengqianChen/DreamWaQ_Go2W/blob/b64f7b90856fc71595870cb283ce29dd0f4d6f0f/legged_gym/resources/robots/go2w_30_unitree/urdf/go2w.urdf
[dream-cfg]: https://github.com/ShengqianChen/DreamWaQ_Go2W/blob/b64f7b90856fc71595870cb283ce29dd0f4d6f0f/legged_gym/legged_gym/envs/go2w/go2w_config.py
[dream-base]: https://github.com/ShengqianChen/DreamWaQ_Go2W/blob/b64f7b90856fc71595870cb283ce29dd0f4d6f0f/legged_gym/legged_gym/envs/base/legged_robot_config.py
[dream-runtime]: https://github.com/ShengqianChen/DreamWaQ_Go2W/blob/b64f7b90856fc71595870cb283ce29dd0f4d6f0f/legged_gym/legged_gym/envs/go2w/go2w_robot.py
[rl-asset]: https://github.com/fan-ziqi/robot_lab/blob/26acb225aad129e222da7a34ba11f39e29b73e39/source/robot_lab/robot_lab/assets/unitree.py
[rl-go2-model]: https://github.com/fan-ziqi/robot_lab/blob/26acb225aad129e222da7a34ba11f39e29b73e39/source/robot_lab/robot_lab/assets/data/Robots/unitree/go2w_description/urdf/go2w_description.urdf
[rl-b2-model]: https://github.com/fan-ziqi/robot_lab/blob/26acb225aad129e222da7a34ba11f39e29b73e39/source/robot_lab/robot_lab/assets/data/Robots/unitree/b2w_description/urdf/b2w_description.urdf
[rl-go2-cfg]: https://github.com/fan-ziqi/robot_lab/blob/26acb225aad129e222da7a34ba11f39e29b73e39/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/unitree_go2w/rough_env_cfg.py
[rl-b2-cfg]: https://github.com/fan-ziqi/robot_lab/blob/26acb225aad129e222da7a34ba11f39e29b73e39/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/unitree_b2w/rough_env_cfg.py
[rl-env]: https://github.com/fan-ziqi/robot_lab/blob/26acb225aad129e222da7a34ba11f39e29b73e39/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/velocity_env_cfg.py
[wlg-model]: https://github.com/clearlab-sustech/Wheel-Legged-Gym/blob/c354431e5633eb98d9663fa3fbe444201d2de383/resources/robots/wl/urdf/wl.urdf
[wlg-cfg]: https://github.com/clearlab-sustech/Wheel-Legged-Gym/blob/c354431e5633eb98d9663fa3fbe444201d2de383/wheel_legged_gym/envs/wheel_legged/wheel_legged_config.py
[wlg-base]: https://github.com/clearlab-sustech/Wheel-Legged-Gym/blob/c354431e5633eb98d9663fa3fbe444201d2de383/wheel_legged_gym/envs/base/legged_robot_config.py
[wlg-runtime]: https://github.com/clearlab-sustech/Wheel-Legged-Gym/blob/c354431e5633eb98d9663fa3fbe444201d2de383/wheel_legged_gym/envs/base/legged_robot.py
