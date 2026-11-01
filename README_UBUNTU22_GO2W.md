# Go2W＋MID360：训练导出与 Ubuntu 22.04 部署

训练继续使用本仓库原有 mjlab／RSL-RL 任务。部署采用 `deploy/rl_sar` 工作区中的原 C++／ROS 1 RL SDK 和 FSM。

完整操作、雷达启动、观测动作、关节顺序、只读检查及真机步骤见 [Go2W 专用 README](../deploy/rl_sar/src/rl_sar/models/go2w/README.md)。复制到其他目录后，请在对应 RLSAR 工作区查找 `src/rl_sar/models/go2w/README.md`。

## 导出实际训练任务

```bash
python scripts/prepare_go2w_deploy.py \
  --checkpoint /path/to/model_XXXX.pt \
  --task Unitree-Go2W-Perceptive-Rough-v2 \
  --output /path/to/go2w_policy --device cpu
```

`--task` 必须填写实际训练任务；不根据输入维度猜测检查点类型。复制整个输出目录，保留 `exported/policy.onnx` 和 `params/deploy.yaml`。

v1 为240维，v2／SGS为720维，Gait为726维。所有任务使用12腿位置＋4轮速度动作，策略关节顺序为FL、FR、RL、RR腿，然后四个轮；SDK顺序另有映射。部署不得使用Nezha3的轮索引或复制G1关节配置。

在 RLSAR 根目录显式生成原生配置：

```bash
python3 tools/import_go2w_policy.py \
  --bundle /path/to/go2w_policy \
  --output-config src/rl_sar/models/go2w/robot_lab/config.yaml
```

## 与原 G1 部署的差异必须明确

原 G1 部署只使用当前点云，没有里程计订阅。v1 当前帧观测可沿用这个输入集合。

现有 v2 训练中的近场记忆使用实际前向速度；这一训练观测不能靠把里程计要求写进文档就算完成迁移，也不能静默关闭。RLSAR只保留训练需要的前向补偿，移除了另加的完整位姿重投影。没有实际速度输入时，v2的这项硬件闭环仍未验收。

本次未改变既有PPO、SGS或Gait训练算法。修正了导出器写死的命令范围、实际网格参数和增益顺序声明，以及v1奖励引用不存在传感器的问题。旧独立DDS部署说明不再作为本次RLSAR交付的操作入口。

本次不安装环境。Ubuntu22.04目标需要已有兼容的ROS1环境；完整ROS、Gazebo和实机运行需在目标机器上验收。
