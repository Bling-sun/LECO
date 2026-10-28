# 1. 数据生成

[流程总览](../../README.md) → **数据生成** → [模型训练](../training/README.md) → [部署](../deployment/README.md) → [仿真验证](../simulation/README.md)

输入：ROS 仿真中的传统 MPC 专家。输出：`train.pkl`、`val.pkl` 和数据划分清单。先按总 README 安装 ROS 依赖并构建工作空间；数据处理所需的 Python 环境见[训练环境](../training/README.md#环境)。

## 采集

在项目根目录执行。先停止同一 ROS 域中的其他仿真；脚本自行启动 simulator、MPC、自动目标点发布器和录制器。

```bash
cd "$TRACKER_ROOT"
bash neural_tracker/scripts/collect_expert_data_ros.sh \
  --seeds 2 --start-seed 100 --map-styles blocks \
  --samples-per-seed 4 --episode-seconds 12 --sample-hz 10 \
  --save-dir data/expert_ros_demo --no-rviz
```

默认使用 `src/simulator/config/expert_collection.yaml`：20×20 m、0.05 m 分辨率，与训练坐标一致。种子和地图风格通过 ROS 参数覆盖，不修改源配置。采集结束只清理本次创建的进程组。

| 参数 | 用途 |
| --- | --- |
| `--seeds` / `--start-seed` | 种子数量 / 起始种子 |
| `--map-styles blocks,pillars,corridors` | 每个种子遍历的地图风格 |
| `--samples-per-seed` | 每种地图、每个种子的 episode 数 |
| `--episode-seconds` / `--sample-hz` | 每段时长 / 采样频率 |
| `--save-dir` | 输出目录，相对 `neural_tracker/`；也可用绝对路径 |
| `--sim-config` | 自定义采集 YAML，须保持训练需要的地图尺寸 |
| `--no-rviz` | 无界面采集 |
| `--help` | 全部录制和目标点参数 |

原始数据：`neural_tracker/data/expert_ros_demo/blocks/sample_*.pkl`，附带预览图。日志：`log/data_collection/`。每帧保存地图、机器人状态、目标观测及专家控制/规划序列；未来序列的有效部分通过 mask 标记。

## 检查与合并

```bash
cd "$TRACKER_ROOT/neural_tracker"
.venv/bin/python scripts/filter_expert_visibility.py \
  --data_dir data/expert_ros_demo --recursive

.venv/bin/python scripts/merge_expert_data.py \
  --input_dir data/expert_ros_demo --recursive \
  --output_dir data/scenarios_demo \
  --val_ratio 0.25 --seed 42 --split-unit episode
```

质量检查默认只报告，不删除数据。需要删除不合格 episode 时显式加 `--delete-bad`。合并至少需要两个非空 episode，并拒绝覆盖已有结果；重新实验时换输出目录。

输出：

```text
neural_tracker/data/scenarios_demo/
├── train.pkl
├── val.pkl
└── split_manifest.json
```

按 episode 划分可防止同一段轨迹跨训练集和验证集；多地图读取须带 `--recursive`。正式泛化评估还应分离地图/种子。保留原始数据和 `split_manifest.json` 中的输入哈希。

验收：采集命令成功退出、质量报告可用、两个数据集非空。随后在训练命令中使用 `--data-dir data/scenarios_demo`。
