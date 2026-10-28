# LECO 模型快速上手

流程：**准备环境 → 小样本训练 → 导出 ONNX → 仿真 → 自采数据正式训练**。

仅训练不需要 ROS；采集数据和仿真前，先完成[项目 README 的 ROS 安装与构建](../README.md#环境与构建)。以下路径按本机目录填写，项目放在别处时修改第一行。

## 1. 准备 Python 环境

每个新终端先执行前三行；创建环境和安装依赖只需做一次。

```bash
cd ~/Documents/LECO
export TRACKER_ROOT="$(pwd -P)"
cd "$TRACKER_ROOT/neural_tracker"
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pip check
.venv/bin/python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

后续直接使用 `.venv/bin/python`，无需手动激活环境。默认锁定 CPU 环境，CUDA 输出 `False` 是正常的；GPU 安装方法见[训练指南](../docs/training/README.md#环境)。

## 2. 先用自带数据跑通训练和导出

无需采集数据，也无需 GPU。下面只训练一轮、两个批次，用于验证流程。

```bash
cd "$TRACKER_ROOT/neural_tracker"
sha256sum -c examples/repro_data/SHA256SUMS
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python train.py \
  --data-dir examples/repro_data --run-name quickstart \
  --epochs 1 --batch-size 4 --num-workers 0 --device cpu --seed 42 \
  --max-train-batches 2 --max-val-batches 2 --disable-tqdm

.venv/bin/python scripts/export_onnx.py \
  --model models/quickstart/best.pth --output models/quickstart/policy.onnx
```

成功标志：生成 `models/quickstart/` 下的 `best.pth`、`latest.pth`、`policy.onnx`；导出打印 `OK: ONNX export is numerically faithful.`。这个小模型只验证流程，不代表正式训练效果。

## 3. 部署到仿真，运行后记得停止

先完成项目的 ROS 构建，再执行：

```bash
cd "$TRACKER_ROOT"
./start_tracking.sh --onnx neural_tracker/models/quickstart/policy.onnx
```

看到 `Ready` 后，在 RViz 用 `2D Goal Pose` 指定目标点。观察机器人跟随、轨迹和日志中的非零 ONNX 推理样本。需要直接演示发布模型时，使用 `./start_tracking.sh`，不传 `--onnx`。

| 操作（项目根目录） | 命令 |
| --- | --- |
| 查看状态 | `./start_tracking.sh --status` |
| 查看日志 | `tmux attach -t tracking_neural` |
| 结束仿真 | `./start_tracking.sh --stop` |

关闭终端或按 `Ctrl-b` 再按 `d` 离开 tmux 后，仿真仍会运行。重新启动或更换模型前先执行 `--stop`。无界面运行、自动路线和正式指标见[仿真指南](../docs/simulation/README.md)。

## 4. 用自己的数据正式训练

先停止同一 ROS 域内的其他仿真，再采集专家数据：

```bash
cd "$TRACKER_ROOT"
./start_tracking.sh --stop
bash neural_tracker/scripts/collect_expert_data_ros.sh \
  --seeds 2 --start-seed 100 --map-styles blocks \
  --samples-per-seed 4 --episode-seconds 12 --sample-hz 10 \
  --save-dir data/expert_ros_demo --no-rviz
```

原始数据位于 `neural_tracker/data/expert_ros_demo/blocks/`。检查质量、合并，再训练：

```bash
cd "$TRACKER_ROOT/neural_tracker"
.venv/bin/python scripts/filter_expert_visibility.py \
  --data_dir data/expert_ros_demo --recursive
.venv/bin/python scripts/merge_expert_data.py \
  --input_dir data/expert_ros_demo --recursive \
  --output_dir data/scenarios_demo --val_ratio 0.25 --seed 42 --split-unit episode

.venv/bin/python train.py \
  --data-dir data/scenarios_demo --run-name demo_train \
  --epochs 35 --batch-size 32 --learning-rate 0.0001 \
  --num-workers 8 --device cpu --seed 42
```

质量检查默认只报告，不删除数据；有不合格 episode 时按[数据指南](../docs/data/README.md)筛选或补采。合并成功应得到 `data/scenarios_demo/{train.pkl,val.pkl,split_manifest.json}`；再次合并须换输出目录。

训练成功生成 `models/demo_train/best.pth` 和 `latest.pth`。配置好 GPU 后可改为 `--device cuda`；内存不足减小 `--batch-size`。新实验更换 `--run-name`；示例采集量用于上手，正式效果需要更充分的数据。

接下来重复第 2、3 步的导出与仿真命令，将模型路径中的 `quickstart` 换成 `demo_train`。

## 常用操作与排错

下面的训练命令均在 `neural_tracker/` 内执行。

| 需求 / 问题 | 操作 |
| --- | --- |
| 查看训练曲线 | `.venv/bin/tensorboard --logdir logs/demo_train` |
| 续训到总计 50 轮 | `.venv/bin/python train.py --resume models/demo_train/latest.pth --epochs 50 --device cpu`；不要重复传原来的 `--run-name` |
| 找不到 `train.py` / `.venv/bin/python` | `cd "$TRACKER_ROOT/neural_tracker"`；缺少环境则执行第 1 步 |
| 找不到 `train.pkl` / `val.pkl` | 先完成采集与合并，核对 `--data-dir` |
| 提示会话已存在 | `"$TRACKER_ROOT/start_tracking.sh" --stop`，再重新启动 |

详细说明：[数据生成](../docs/data/README.md) · [训练与评估](../docs/training/README.md) · [模型部署](../docs/deployment/README.md) · [仿真验证](../docs/simulation/README.md)。
