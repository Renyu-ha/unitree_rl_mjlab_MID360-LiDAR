# Go2W＋MID360：按原 G1 框架部署

面向 Ubuntu 22.04 x86_64。快速操作见仓库根目录 `README_UBUNTU22_GO2W.md`。

## 目录和入口

```text
deploy/robots/g1/                      # 原作者参考实现
deploy/robots/go2w/
  main.cpp                            # 同样的 CtrlFSM 入口
  include/Types.h                      # SDK 类型别名
  include/Go2WLidar.h                   # 从 G1 适配的点云/里程计处理
  include/go2w_core.h                   # 16 电机映射、高度网格、记忆
  src/State_RLBase.cpp                  # 同名 RLBase，12 腿位置＋4 轮速度
  config/config.yaml                   # 硬件/FSM 配置
deploy/include/                        # 复用 G1 的 FSM、观测/动作管理器、ONNX
unitree_sdk2/include/unitree/dds_wrapper/robots/go2w/
patches/unitree_sdk2/include/unitree/dds_wrapper/robots/go2w/
simulate/src/main.cc                   # 原 unitree_mujoco 入口
simulate/src/go2w_bridge.h             # 与 G1Bridge 并列的 Go2WBridge
simulate/config_go2w.yaml
```

这次扩展的 Go2W SDK 包装器使用 `unitree_go` 的 LowCmd/LowState，保留 G1 包装器的调用接口。
20 个协议槽位只启用前 16 个；不使用 G1 的 `unitree_hg` 电机协议或 `mode_machine`。
`patches/` 是父仓库可版本管理的源文件，构建脚本将其同步进 SDK 子模块。
因此复制目录或从源码包解压均可复现，无需提交或修改 SDK 子模块历史。

## Ubuntu 构建与自检

```bash
python3 scripts/verify_go2w_release.py  # 从发布包解压时执行，验证未修改的源文件
bash scripts/build_go2w_ubuntu22.sh --install-deps
python3 scripts/check_go2w_ubuntu22.py
```

无桌面机器增加 `--headless`；已安装系统依赖时可以省略 `--install-deps`。
脚本使用系统 GCC/CMake、仓库 SDK/Cyclone DDS、ONNX Runtime 1.22.0 和
原仿真器自带的 MuJoCo 3.3.6，修复 Windows checkout 的 `.so` 文本链接，
并用标准库 Python 生成相对路径的场景。构建与 DDS 自检不需要安装训练环境。
训练环境使用 MuJoCo 3.16.0，不能直接拿它的头文件与仿真器 3.3.6 动态库混用。

输出仍是 `deploy/robots/go2w/build/go2w_ctrl` 和 `simulate/build/unitree_mujoco`。
同一个仿真程序支持原图形窗口和 `--headless`。G1 的源文件和图形入口保留。

自检使用不同的非零 DDS domain，网卡固定 `lo`，启动实际 C++ 控制器和仿真进程。
检查正常站立、只读零发布、点云/里程计/LowState 断流、错误 CRC、轮速映射与站立中停止，
日志及 JSON 输出在 `outputs/go2w_selfcheck/`。
正常站立使用真实物理；只读、轮速映射和通信故障场景固定机身姿态，避免故障后的自由跌落干扰接口验证。
`--relaxed-deadlines` 仅供慢速 CPU 仿真器验证，放宽数据期限；实机使用默认配置。

手动仿真，两个终端分别运行：

```bash
simulate/build/unitree_mujoco --config simulate/config_go2w.yaml --auto
deploy/robots/go2w/build/go2w_ctrl --sim --network=lo --domain=1 \
  --policy tests/fixtures/go2w_deploy --actuate
```

无界面运行第一条命令增加 `--headless --duration=30`。
`--auto` 在控制器连接后自动按下站立/策略组合键，也可连接原仿真器支持的手柄。
仿真点云从场景向下射线生成，同时发布机体系里程计；用于通信和部署接口验证，
没有模拟 MID360 的完整扫描、噪声和时间畸变。
自动按键在收到控制器 `--sim` 模式的就绪通知后开始；只读模式同样发送该独立状态通知，但不会创建 LowCmd 发布者。

`tests/fixtures/go2w_deploy` 是零动作策略；`go2w_wheel_probe` 输出不同轮速以核对映射。
两者均带 `smoke_*` 标记，启用实机输出时会被拒绝。它们不能用于评价运动策略能力。

## 从实际训练模型导出

在按照仓库训练说明安装的 Python 3.11 环境中，使用实际训练任务：

```bash
python scripts/prepare_go2w_deploy.py \
  --task Unitree-Go2W-Perceptive-Rough-v2 \
  --checkpoint /path/to/model.pt \
  --output deploy/robots/go2w/config/policy/velocity/current
```

输出目录必须为空。导出只加载 Actor，不会继续训练。新训练保存模型时也会生成
包含部署契约的 ONNX，可直接打包：

```bash
python scripts/prepare_go2w_deploy.py --onnx /path/to/policy.onnx --output /path/to/go2w_bundle
```

整个部署包包括 `exported/policy.onnx` 与 `params/deploy.yaml`，外部张量合并到 ONNX 内。
观测顺序、历史、缩放、电机顺序、增益及动作参数从真实训练环境导出，并嵌入 ONNX。
运行时核对 YAML 与嵌入契约；不要手工改策略 YAML 来迁就旧模型。
本机找到的旧 726 维模型不能冒充当前 720 维模型，需使用训练时对应的代码和配置重新导出。
导出成功不代表模型收敛，本次没有把短程验证权重标为可实机策略。

| 任务系列 | Actor 输入 | 动作 |
| --- | ---: | --- |
| Flat | 53 | 12 位置＋4 轮速 |
| Perceptive Rough/v1 | 240 | 12 位置＋4 轮速 |
| v2/StepUp/Step14 | 720 | 12 位置＋4 轮速 |

历史在每个观测项内部从旧到新排列，第一帧真实观测填满历史；轮累计转角不输入 Actor。
轮速观测缩放为 0.05，ONNX 内部包含 Actor 的观测归一化。
腿位置与轮速度的缩放/偏置只在共享 ActionManager 中计算一次。

## MID360 与独立里程计

修改硬件文件 `deploy/robots/go2w/config/config.yaml`：

| 项目 | 约定 |
| --- | --- |
| 点云 | 默认 `/utlidar/cloud_livox_mid360`，sensor_msgs/PointCloud2，米，float32 x/y/z |
| 点云帧 | 与 `frame_id` 一致，默认 `livox_frame` |
| 外参 | `translation_m` 和 `rpy_deg` 表示雷达到 `base_link` 的变换 |
| 独立里程计 | `/go2w/odometry`，nav_msgs/Odometry，child_frame_id=`base_link` |
| 里程计 pose/twist | pose 为连续局部世界系；twist.linear 为机体系 |
| 自身过滤 | 默认机身包围盒，按腿部、雷达和支架的实际反射补充 |

ROS2 名称转换为 DDS `rt/...`，使用相同 domain、兼容的消息类型及 Cyclone DDS 网络配置。
传感器订阅请求 BestEffort/Volatile/KeepLast(1)，兼容 ROS2 常见的 SensorDataQoS 发布者。
程序不内置 MID360 驱动或 LIO 状态估计器，现场必须提供上述两个数据源。
零外参是占位值，不能当作实测结果；不要沿用 G1 的雷达翻转角度。
关闭机器人内置运动控制后，独立里程计不能依赖 SportModeState，程序不会静默用轮速积分代替它。

v2 默认 `training_forward` 记忆沿用训练约定：17×11 格、前移 0.35 m、最长 0.8 秒，
只补偿正向 X 位移。可见掩码随训练导出，旧点云不会反复当作新数据更新。
可选 `memory_mode: pose` 增加位移/偏航/高度补偿，会改变训练记忆语义，需要另行对照验证。
本次未实现逐点去畸变或精确点云/IMU 时间同步；雷达和支架的实际质量、重心也需纳入训练。

## 实机控制与保护

先完成仿真、吊装电机编号与方向检查、雷达标定、独立里程计核对及负载模型检查。
完成后设置硬件文件的 `safety.hardware_calibrated: true`，并使用训练好的部署包。
按对应固件的低层控制流程停止其他运动控制程序；本程序不自动关闭官方运动服务。
启动时短时间检测到 `rt/lowcmd` 已有指令会拒绝输出，该检查不代替现场控制权管理。

```bash
# 接收与推理，默认无 LowCmd 发布者
deploy/robots/go2w/build/go2w_ctrl --network=eth0 --domain=0 --policy /path/to/go2w_bundle

# 完成标定、确认控制权后显式输出
deploy/robots/go2w/build/go2w_ctrl --network=eth0 --domain=0 --policy /path/to/go2w_bundle --actuate
```

网卡按实际修改。手柄先按住 L2 再按上进入站立，站立完成后按住 R2 再按 A 进入策略，
L2＋B 回阻尼。无输出模式也可运行该流程以检查观测和推理。

策略顺序为 FL/FR/RL/RR 的 12 个腿关节，再接同顺序四个轮关节；
SDK 映射为 `[3,4,5,0,1,2,9,10,11,6,7,8,13,12,15,14]`。
腿写 `q`，轮写 `dq`，轮 `kp=0`；与[宇树官方 Go2W 仿真电机顺序](https://github.com/unitreerobotics/unitree_mujoco/blob/main/unitree_robots/go2w/go2w.xml)一致。

低层状态默认 100 ms、点云 300 ms、里程计 200 ms、策略结果 60 ms 超时触发保护；
错误 LowState CRC 不刷新接收时间，断流后触发同样的保护。
LowCmd/LowState 校验按[宇树 Python SDK 的 Go 协议格式](https://github.com/unitreerobotics/unitree_sdk2_python/blob/master/unitree_sdk2py/utils/crc.py)显式打包零填充字节，避免 DDS 未传输的 C++ 内存填充影响结果；构建时会执行 C++ 与 Python 的逐值对比。
覆盖不足、姿态异常、非有限观测/动作、越界目标或推理异常也会返回阻尼。
独立 500 Hz 发布线程在 FSM 心跳超过 50 ms 时输出阻尼，避免持续保持旧策略目标。
保护阈值触发故障，不会把训练时无裁剪的动作静默改为裁剪。
站立阶段同样响应停止和看门狗故障，重新发出站立指令才会清除故障锁存。
Go2W 订阅在对象释放前显式关闭读取器，并保留监听对象至回调结束；构建中包含持续收发时反复关闭订阅的回归测试。
恢复数据后需重新操作状态切换；现场仍需验证固件急停与手柄断连行为。

## 验证范围

本次在 Windows 主机上交叉编译并完整链接 Ubuntu 22.04 x86_64 依赖，
通过隔离 QEMU 的 Ubuntu 22.04.5 用户态运行实际 C++/DDS 程序。
该验证使用 Alpine 引导内核和放宽的仿真期限，不代表已经在完整 Ubuntu 桌面或机器人上实测。
具体结果见 `doc/go2w_ubuntu22_validation.json`。

训练环境回归：

```bash
python -m unittest tests.test_go2w_deploy tests.test_go2w_actions_runtime tests.test_go2w_ordering
```

可选安装 `onnxruntime==1.22.0` 比较 ONNX 与 PyTorch Actor 输出；设置 `GO2W_CORE_TEST`
为 C++ 核心测试路径可与实际训练记忆函数逐数值比较。图形版本已验证构建，窗口交互仍需目标桌面验证。
