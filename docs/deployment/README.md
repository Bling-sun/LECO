# 3. 模型部署

[流程总览](../../README.md) → [数据生成](../data/README.md) → [训练](../training/README.md) → **模型部署** → [仿真验证](../simulation/README.md)

输入：`best.pth`。推荐导出为 `policy.onnx`，交给 C++ `tracker_neural` 运行。Python/PyTorch 部署用于调试 checkpoint。

## 导出 ONNX

```bash
cd "$TRACKER_ROOT/neural_tracker"
MPLCONFIGDIR=/tmp .venv/bin/python scripts/export_onnx.py \
  --model models/demo_train/best.pth \
  --output models/demo_train/policy.onnx
```

导出会恢复 checkpoint 配置并检查 PyTorch / ONNX 输出的一致性，默认容差 `1e-4`。验收：输出 `OK: ONNX export is numerically faithful.`，且 `policy.onnx` 已生成。验证小规模模型时，将 `demo_train` 替换为训练步骤的 `demo_smoke`。

## 部署并启动仿真

```bash
cd "$TRACKER_ROOT"
./start_tracking.sh --backend neural \
  --onnx neural_tracker/models/demo_train/policy.onnx \
  --map-json src/simulator/map/maze2.json --no-rviz
```

脚本会等待 simulator 和模型加载完成，随后打印 `Ready`。查看运行：`tmux attach -t tracking_neural`。目标运动与闭环检查见[仿真验证](../simulation/README.md)。

不传 `--onnx` 时使用已发布的 `v41_directional_recovery/policy.onnx`，可跳过训练、导出直接运行。模型校验：

```bash
sha256sum -c neural_tracker/models/SHA256SUMS
```

## Python/PyTorch 调试

先停止 ONNX 仿真，再执行：

```bash
./start_tracking.sh --backend neural --stop
./start_tracking.sh --backend python \
  --model neural_tracker/models/demo_train/best.pth --device cpu --no-rviz
# 结束后：
./start_tracking.sh --backend python --stop
```

Python 模式需要 `neural_tracker/.venv`，默认加载 20×20 m 的 `expert_collection.yaml`，地图尺寸 / 分辨率必须与 checkpoint 一致。更大的演示地图使用 C++/ONNX。节点以持久 QoS 接收静态地图，控制发布到 `/tracker/cmd_vel`。

默认权重为 `v41_directional_recovery_retrain_clean_20260717/best.pth`。Python 的丢失恢复采用 `--lost-policy predict`，与 C++ 的真值恢复策略不同，实验结果应分别记录。

## 单独接入 ROS

已有传感器和底盘接口时，可以只启动 C++ 跟踪器：

```bash
source /opt/ros/jazzy/setup.bash
source "$TRACKER_ROOT/install/setup.bash"
ros2 launch tracker tracking_neural.launch.py \
  onnx_path:="$TRACKER_ROOT/neural_tracker/models/demo_train/policy.onnx" \
  onnx_threads:=4
```

需要适配 `/tracker/odom`、`/tracker/scan`、`/target/detection` 和 `/tracker/cmd_vel`，并核对坐标系、速度上限和目标观测定义。默认 FOV 恢复还使用 `/target/ground_truth`；真实传感器部署应显式配置恢复策略。当前一键脚本提供的是仿真部署入口。
