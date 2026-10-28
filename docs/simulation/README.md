# 4. 仿真验证

[流程总览](../../README.md) → [数据生成](../data/README.md) → [训练](../training/README.md) → [部署](../deployment/README.md) → **仿真验证**

以下命令从项目根目录执行。同一 ROS 域一次运行一套仿真；所有终端使用相同的 `ROS_DOMAIN_ID`。

## 交互验证

```bash
cd "$TRACKER_ROOT"
./start_tracking.sh --map-json src/simulator/map/maze2.json
```

默认使用 C++/ONNX v41。在 RViz 发布 `2D Goal Pose` 让目标运动；只看到循环日志不足以证明模型已推理。检查机器人跟随、目标检测、轨迹和 `神经分支` 的非零样本数 / ONNX 耗时。

复现主页 GIF 的地图和目标路线（默认 ONNX 后端）：

```bash
./start_tracking.sh --stop
./start_tracking.sh --map-json src/simulator/map/maze2.json \
  --goals docs/assets/tracking-demo-goals.json
```

无界面时，加 `--no-rviz`，然后在加载相同 ROS 环境的另一终端发送 maze2 的目标点：

```bash
source /opt/ros/jazzy/setup.bash
source "$TRACKER_ROOT/install/setup.bash"
ros2 topic pub --once /goal_pose geometry_msgs/msg/PoseStamped \
  '{header: {frame_id: world}, pose: {position: {x: 4.725, y: 27.575}, orientation: {w: 1.0}}}'
```

## 会话管理

| 操作 | 默认 ONNX 会话命令 |
| --- | --- |
| 启动 | `./start_tracking.sh` |
| 查看状态 | `./start_tracking.sh --status` |
| 查看日志 | `tmux attach -t tracking_neural` |
| 关闭整套仿真 | `./start_tracking.sh --stop` |

仿真在 tmux 后台运行；关闭终端或按 `Ctrl-b` 再按 `d` 离开 tmux 后仍会运行。结束实验使用 `--stop`，它会停止该会话的 simulator、控制器及路线进程。

提示 `session 'tracking_neural' exists` / `is already running` 时，先关闭上一次仿真再启动：

```bash
./start_tracking.sh --stop
./start_tracking.sh
```

停止其他后端或自定义会话：

```bash
./start_tracking.sh --backend traditional --stop
./start_tracking.sh --backend python --stop
./start_tracking.sh --session demo --stop
```

自定义 `--session demo` 后，查看状态也使用 `./start_tracking.sh --session demo --status`，查看日志使用 `tmux attach -t demo`。切换后端前须停止原后端。

## 自动路线

```bash
./start_tracking.sh --stop
./generate_random_goals.sh maze2 experiments/routes/maze2_demo.json \
  --count 5 --seed 42
./start_tracking.sh --map-json src/simulator/map/maze2.json \
  --goals experiments/routes/maze2_demo.json --no-rviz
```

路线生成器检查目标点净空和路径连通性；已存在的输出会拒绝覆盖。`--goals` 在节点就绪后回放，目标进入 1 m 范围再发下一个点。地图变更后须重新生成路线。

手工录制与外部参考轨迹：

```bash
# 先停止上一次仿真，再任选一种。
./start_tracking.sh --record-waypoints experiments/routes/manual.json
./start_tracking.sh --map-json path/to/map.json --reference path/to/reference.json
```

录制时在 RViz 连续选择目标点，结束时先在录制窗格按 `Ctrl-c` 保存。`--reference` 接受 `elastic-target-reference-v1`，自动开启外部目标控制；起始位姿须与地图匹配。三种路线模式互斥。

## 后端与场景参数

| 用途 | 命令 / 参数 |
| --- | --- |
| 新模型 | `./start_tracking.sh --onnx neural_tracker/models/demo_train/policy.onnx` |
| 传统 MPC | `./start_tracking.sh --backend traditional` |
| Python 推理（20 m 地图） | `./start_tracking.sh --backend python --model neural_tracker/models/demo_train/best.pth` |
| 临时障碍物 | `./start_tracking.sh --map-json src/simulator/map/maze1.json --temporary-obstacles --obstacle-seed 42 --direct-target` |
| 自定义仿真配置 | `./start_tracking.sh --sim-config src/simulator/config/environments/forest.yaml --sim-arg map.seed:=101` |
| 传统速度 / 加速度 | `--backend traditional --tracker-arg max_linear_speed:=2.5 --tracker-arg max_linear_accel:=3.0` |
| 真值输入 | `--direct-target`，单独记录实验模式 |
| C++ 真值恢复延迟 | `--truth-fallback-after 3.0`，默认 FOV 模式 |
| 推理计时 | `--no-rviz --onnx-threads 4 --timing-interval 1.0` |

切换后端前停止上一套仿真；traditional / python 的默认会话分别为 `tracking_traditional` / `tracking_python`。全部参数见 `./start_tracking.sh --help`。

## 正式指标采集

普通启动只做演示。正式实验先停止其他仿真，再用实验工具启动并采集：

```bash
./start_tracking.sh --stop
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 run experiment neural_tracker_experiment \
  experiments/routes/maze2_demo.json \
  --run-label demo_train_maze2_01 --algorithm-label demo_train \
  --map-file src/simulator/map/maze2.json \
  --policy-file neural_tracker/models/demo_train/policy.onnx
```

每轮使用新的 `--run-label`。输出在 `experiments/<日期>/<run-label>/`，包括 `metrics.json` 和日志；退出码 `0` 表示有效且全部试次成功，`1` 表示有效但有失败，`2` 表示启动/采集无效。记录模型哈希、路线、地图、后端和发现环境，见[实验工具](../../src/experiment/README.md)。

## 常见问题

| 现象 | 检查 |
| --- | --- |
| 找不到模型 / 二进制 | 确认模型路径并按总 README 构建 |
| 会话已存在 | 默认会话执行 `./start_tracking.sh --stop`；自定义会话加 `--session NAME`，然后重新启动 |
| 启动失败 | 脚本打印对应窗格日志并清理本次会话 |
| 目标静止 / 神经分支为零 | 发送可达目标，检查目标检测及静止保护 |
| 跨终端收不到话题 | 核对 `ROS_DOMAIN_ID`、发现范围和工作空间 |
