# LECO

**Learning Direct Control from Optimization for Dynamic Target Tracking**

[论文](https://arxiv.org/abs/2610.34929) · [代码](https://github.com/Bling-sun/LECO)

从 MPC 专家数据训练跟踪策略，导出 ONNX，并在 ROS 2 仿真中验证闭环跟踪。

## 完整流程

| 步骤 | 文档 | 输入 → 输出 |
| --- | --- | --- |
| 1. 数据生成 | [数据 README](docs/data/README.md) | MPC 仿真 → 原始 episode → `train.pkl` / `val.pkl` |
| 2. 模型训练 | [训练 README](docs/training/README.md) | 数据集 → `best.pth` / `latest.pth` |
| 3. 模型部署 | [部署 README](docs/deployment/README.md) | `best.pth` → `policy.onnx` → ROS 跟踪节点 |
| 4. 仿真验证 | [仿真 README](docs/simulation/README.md) | 模型 + 地图 + 路线 → 闭环轨迹 / `metrics.json` |

仓库含已发布的 v41 权重，可直接进入仿真。完整训练需要自行采集数据；历史原始训练集未发布，新训练不保证复现相同权重和论文数值。[下载与复现验收](docs/reproduction/README.md)。

## 环境与构建

支持 Ubuntu 24.04、x86_64、Python 3.12、ROS 2 Jazzy。先安装 [ROS 2 Jazzy](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html)，再安装依赖：

```bash
sudo apt update
sudo apt install -y build-essential cmake git tmux python3-venv \
  python3-colcon-common-extensions python3-pytest python3-yaml \
  python3-numpy python3-scipy python3-matplotlib python3-opencv libopencv-dev \
  ros-jazzy-desktop ros-jazzy-pcl-conversions ros-jazzy-tf2-ros \
  ros-jazzy-xacro ros-jazzy-robot-state-publisher \
  ros-jazzy-ament-cmake-gtest ros-jazzy-ament-cmake-pytest

git clone https://github.com/Bling-sun/LECO.git
cd LECO
export TRACKER_ROOT="$(pwd -P)"
source /opt/ros/jazzy/setup.bash
CMAKE_BUILD_PARALLEL_LEVEL=2 colcon build \
  --base-paths src --packages-select simulator tracker experiment \
  --executor sequential --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
```

构建预期为 `3 packages finished`。C++ ONNX Runtime 随仓库提供。构建和 ROS 采集使用系统 Python；训练、导出及 Python 部署的虚拟环境安装见[训练 README](docs/training/README.md)。

已有本地项目直接进入根目录并设置 `TRACKER_ROOT`。后续文档均以此变量表示项目根目录；每个新终端重新设置，并加载 ROS / `install/setup.bash`。

## 直接运行已有模型

```bash
sha256sum -c neural_tracker/models/SHA256SUMS
./start_tracking.sh
```

在 RViz 使用 `2D Goal Pose` 让目标移动。无界面时加 `--no-rviz`，发送目标或自动路线的方法见[仿真 README](docs/simulation/README.md)。

### 查看、停止与重新启动

启动后仿真在 tmux 后台运行；关闭终端或按 `Ctrl-b` 再按 `d` 离开 tmux，仿真仍会继续运行。结束实验时执行停止命令。

| 操作 | 命令 |
| --- | --- |
| 查看状态 | `./start_tracking.sh --status` |
| 查看日志 | `tmux attach -t tracking_neural` |
| 关闭整套仿真 | `./start_tracking.sh --stop` |

遇到 `session 'tracking_neural' exists` / `is already running`，说明上一次仿真仍在运行。先停止，再重新启动：

```bash
./start_tracking.sh --stop
./start_tracking.sh
```

其他后端分别使用 `./start_tracking.sh --backend traditional --stop` 和 `./start_tracking.sh --backend python --stop`。自定义会话用 `./start_tracking.sh --session demo --stop` 关闭，详见[会话管理](docs/simulation/README.md#会话管理)。

## Shell 入口

仅保留三个用户入口；完整参数均可用 `--help` 查看。

| 脚本 | 用途 | 常用命令 |
| --- | --- | --- |
| `start_tracking.sh` | 统一启动 / 停止仿真、控制器及可选路线 | `./start_tracking.sh --backend neural --no-rviz` |
| `neural_tracker/scripts/collect_expert_data_ros.sh` | 批量采集 MPC 专家数据 | `bash neural_tracker/scripts/collect_expert_data_ros.sh --seeds 2 --map-styles blocks --no-rviz` |
| `generate_random_goals.sh` | 按地图生成可达目标点路线 | `./generate_random_goals.sh maze2 experiments/routes/demo.json --count 5 --seed 42` |

`start_tracking.sh` 的后端为 `neural`（默认 C++/ONNX）、`traditional`（MPC）、`python`（PyTorch）。地图专用的 neural/traditional 包装脚本已合并；使用 `--map-json`、`--goals`、`--record-waypoints`、`--reference` 和 `--temporary-obstacles` 组合场景。三个后端统一通过 tmux 管理。

`src/experiment/env-hooks/experiment_prefix_priority.sh` 是 ROS 自动加载的内部 hook，无需手工执行。训练、评估和导出直接运行文档中的 Python 命令。

## 目录与检查

| 目录 | 内容 |
| --- | --- |
| `neural_tracker/` | 训练代码、数据工具、发布模型 |
| `src/simulator/` | 仿真、地图、传感器配置 |
| `src/tracker/` | C++ 跟踪器、预测与 ONNX 推理 |
| `src/experiment/` | 路线录制 / 回放、指标、结果汇总 |
| `docs/` | 四个阶段的操作说明 |

从 GitHub 下载、安装依赖后，可用一条命令验收构建、采集、训练、导出、三个后端及闭环实验：

```bash
/usr/bin/python3 tools/verify_reproduction.py --ros-domain-id 187
```

预期打印 `passed`；日志位于 `.runtime/reproduction_<时间>/`，详见[复现验收](docs/reproduction/README.md)。

```bash
colcon test --base-paths src --packages-select simulator tracker experiment
colcon test-result --verbose
```

训练数据和实验输出默认不纳入 Git。源码 / launch 变更后重新构建；移动或改名项目后重新生成 `build/` 和 `install/`。
