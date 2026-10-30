# Go2W＋MID360：Ubuntu 22.04 x86_64

本包按原作者 G1 的目录与入口适配 Go2W：`main.cpp → CtrlFSM → State_RLBase → ONNX`，仿真入口仍为 `simulate/build/unitree_mujoco`。SDK 的 `robots/go2w` 包装器已包含在内，无需另外找一个 Go2W SDK 仓库。

## 解压、编译、验证

```bash
tar -xzf go2w_ubuntu22_x86_64.tar.gz
cd go2w_ubuntu22_x86_64
python3 scripts/verify_go2w_release.py
bash scripts/build_go2w_ubuntu22.sh --install-deps
python3 scripts/check_go2w_ubuntu22.py
```

没有桌面环境时，编译命令增加 `--headless`。构建使用系统 C++ 编译器、仓库 SDK、ONNX Runtime 1.22.0 和原仿真器配套的 MuJoCo 3.3.6；无需安装训练环境、CUDA、ROS 或 Python ONNX 包。自检启动两个实际 C++ 进程，使用 loopback DDS，结果写入 `outputs/go2w_selfcheck/`。

自检策略是标记为 `smoke_*` 的常量策略，只验证通信、推理入口和保护逻辑。实机输出模式会拒绝它们。训练环境使用 MuJoCo 3.16.0，属于独立依赖，详见仓库训练说明。

## 打开仿真窗口

在两个终端分别运行：

```bash
simulate/build/unitree_mujoco --config simulate/config_go2w.yaml --auto
```

```bash
deploy/robots/go2w/build/go2w_ctrl --sim --network=lo --domain=1 \
  --policy tests/fixtures/go2w_deploy --actuate
```

`--auto` 会自动执行“阻尼 → 站立 → 策略”；换成已训练部署包可观察策略表现。仿真生成的点云和里程计用于验证部署接口，不模拟 MID360 的完整光学扫描。

## 换入训练好的策略

在已安装仓库训练环境的机器上，以实际训练任务重新导出：

```bash
python scripts/prepare_go2w_deploy.py \
  --task Unitree-Go2W-Perceptive-Rough-v2 \
  --checkpoint /path/to/model.pt --output /path/to/go2w_policy
```

将整个 `go2w_policy` 目录拷贝到 Ubuntu，再给 `--policy` 指定该路径。目录包含 `exported/policy.onnx` 和 `params/deploy.yaml`，两者必须配套。旧版 726 维模型不能当作当前 720 维模型直接使用；必须用训练时对应的代码和配置导出。

## 实机接入

先按 [完整部署说明](doc/go2w_deploy_zh.md) 填写雷达到机体的实测外参、点云坐标系与独立里程计来源，完成吊装检查后设置 `safety.hardware_calibrated: true`，并停止其他运动控制程序。`eth0` 应替换为连接机器人的实际网卡。

```bash
# 接收状态、运行推理，不发送电机指令
deploy/robots/go2w/build/go2w_ctrl --network=eth0 --domain=0 --policy /path/to/go2w_policy

# 完成吊装检查和校准后，显式启用电机输出
deploy/robots/go2w/build/go2w_ctrl --network=eth0 --domain=0 --policy /path/to/go2w_policy --actuate
```

手柄：`L2＋上` 站立，`R2＋A` 进入策略，`L2＋B` 回阻尼。代码与仿真联调不能替代实际电机方向、负载参数、雷达安装和训练策略的实机验证。
