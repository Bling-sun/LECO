# 实验工具

闭环验证步骤见[仿真 README](../../docs/simulation/README.md)。以下命令从项目根目录、已加载 ROS / 工作空间的终端执行。

## 路线

```bash
# 不需要运行仿真。
./generate_random_goals.sh maze2 experiments/routes/demo.json --count 5 --seed 42

# 启动仿真并录制 RViz 点击目标；Ctrl-c 停止录制并保存。
./start_tracking.sh --record-waypoints experiments/routes/manual.json

# 停止前一套仿真后，重新启动并回放。
./start_tracking.sh --stop
./start_tracking.sh --goals experiments/routes/manual.json
```

随机路线检查目标点净空、路径连通性和相邻点距离。可用 `--min-clearance`、`--path-clearance`、`--min-step`、`--max-step` 调整；默认输出为 `experiments/<地图>_goal<N>.json`。

## 自动实验

```bash
./start_tracking.sh --stop
ros2 run experiment neural_tracker_experiment \
  experiments/routes/demo.json \
  --run-label v41_maze2_01 --algorithm-label v41 \
  --map-file src/simulator/map/maze2.json \
  --policy-file neural_tracker/models/v41_directional_recovery/policy.onnx
```

工具负责启动、就绪检查、metrics 采集、路线回放和退出清理。已有其他 ROS 节点时会拒绝启动；每轮使用唯一 `--run-label`。输出在 `experiments/<日期>/<run-label>/`。

| 退出码 | 含义 |
| --- | --- |
| `0` | 数据有效，全部试次成功 |
| `1` | 数据有效，至少一次试次失败 |
| `2` | 启动、采集或验收无效 |

`metrics.json` 记录成功率、可见性、距离、控制平滑性、分支使用和规划耗时。正式对比须保持地图、路线、起始位姿、速度上限及 FOV / 真值模式一致；不能只凭进程退出判断实验有效。

## 单独使用工具

| 命令 | 用途 |
| --- | --- |
| `ros2 run experiment goal_pose_sequence record FILE` | 录制 Goal 序列 |
| `ros2 run experiment goal_pose_sequence replay FILE --timing reached` | 按到达顺序回放 |
| `ros2 run experiment neural_tracker_metrics --help` | 神经跟踪器指标采集 |
| `ros2 run experiment traditional_tracker_metrics --help` | 传统跟踪器指标采集 |
| `ros2 run experiment neural_tracker_results --help` | 神经结果汇总 |
| `ros2 run experiment traditional_tracker_results --help` | 传统结果汇总 |
| `ros2 run experiment neural_tracker_fov_heatmap --help` | 神经 FOV 热力图 |
| `ros2 run experiment traditional_tracker_fov_heatmap --help` | 传统 FOV 热力图 |

手动采集时先启动选定后端，再启动 metrics，最后回放路线；使用一致的 run-label。自定义外部目标轨迹可通过统一入口的 `--reference FILE` 回放。
