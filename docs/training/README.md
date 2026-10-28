# 2. 模型训练

[流程总览](../../README.md) → [数据生成](../data/README.md) → **模型训练** → [部署](../deployment/README.md) → [仿真验证](../simulation/README.md)

输入：`train.pkl` 和 `val.pkl`。输出：`models/<run-name>/best.pth`、`latest.pth`。

## 环境

```bash
cd "$TRACKER_ROOT/neural_tracker"
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pip check
.venv/bin/python -c "import torch; print(torch.__version__, 'CUDA:', torch.cuda.is_available())"
```

已有环境可跳过安装。锁定环境使用 CPU 版 PyTorch，无需 GPU。训练与导出使用 `.venv/bin/python`；构建和 ROS 采集使用系统 Python。

需要 GPU 时，在虚拟环境中执行 `.venv/bin/python -m pip install --force-reinstall -r requirements-repro.txt` 安装 CUDA 版依赖；确认 `torch.cuda.is_available()` 为 `True` 后，将训练命令的 `--device cpu` 改为 `--device cuda`。

## 先做小规模验证

仓库自带两段录制数据，用于验证训练链路，无需先采集或启动 ROS：

```bash
cd "$TRACKER_ROOT/neural_tracker"
sha256sum -c examples/repro_data/SHA256SUMS
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python train.py \
  --data-dir examples/repro_data --run-name demo_smoke \
  --epochs 1 --batch-size 4 --num-workers 0 --device cpu --seed 42 \
  --max-train-batches 2 --max-val-batches 2 --disable-tqdm
```

验收：完成反向传播并生成 `models/demo_smoke/best.pth` 和 `latest.pth`。小样本和两批训练仅验证流程；评估模型效果须使用正式数据和完整训练。

## 正式训练

```bash
cd "$TRACKER_ROOT/neural_tracker"
.venv/bin/python train.py \
  --data-dir data/scenarios_demo --run-name demo_train \
  --epochs 35 --batch-size 32 --learning-rate 0.0001 \
  --num-workers 8 --device cpu --seed 42
```

CPU 可以完成正式训练，但速度较慢；内存不足时减小 batch。每次实验使用新的 `--run-name`。

| 参数 | 用途 |
| --- | --- |
| `--data-dir` | 上一步生成的训练/验证目录 |
| `--run-name` | 模型和日志子目录 |
| `--strategy current/v41/v43` | 选择策略；不传时沿用当前配置 |
| `--epochs` / `--batch-size` / `--learning-rate` | 总轮数 / 批量 / 学习率 |
| `--num-workers` | 数据加载进程数；调试设为 0 |
| `--seed` | 固定本次调用的随机种子 |
| `--device cpu/cuda` | 训练设备 |
| `--help` | 全部参数 |

`best.pth` 按代码的 `checkpoint_rank` 规则选择；`latest.pth` 用于续训，`epoch_*.pth` 为周期保存。TensorBoard 日志在 `logs/demo_train/`：

```bash
.venv/bin/tensorboard --logdir logs/demo_train
```

## 续训、微调、评估

```bash
# 继续已有实验；epochs 表示总目标轮数，不重复传原 run-name。
.venv/bin/python train.py \
  --resume models/demo_train/latest.pth --epochs 50 --device cpu

# 用已有权重开始新实验。
.venv/bin/python train.py \
  --init-from models/demo_train/best.pth \
  --data-dir data/scenarios_demo --run-name demo_finetune \
  --learning-rate 0.00005 --epochs 35 --device cpu

.venv/bin/python scripts/eval_models.py \
  --models models/demo_train/best.pth --datasets data/scenarios_demo/val.pkl \
  --batch-size 32 --num-workers 0 --device cpu
```

`--resume` 恢复配置与训练状态；`--init-from` 只初始化新实验权重，两者互斥。历史 v41 原始训练集未发布，因此新训练不保证得到相同权重或论文数值；复现范围见[验收记录](../reproduction/README.md)。

下一步：[导出 ONNX 并部署](../deployment/README.md)。
