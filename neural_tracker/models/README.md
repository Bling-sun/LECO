# 发布模型

模型随 Git 仓库下载，无需 Git LFS 或额外下载。

| 模型目录 | 默认用途 | 部署文件 |
| --- | --- | --- |
| `v41_directional_recovery/` | `start_tracking.sh` 的 C++/ONNX 模式 | `policy.onnx` |
| `v41_directional_recovery_retrain_clean_20260717/` | `--backend python` 的 PyTorch 模式 | `best.pth` |

每组同时保留 `best.pth`、`latest.pth`、周期 checkpoint 和 ONNX，共 19 个模型文件。`latest.pth` 用于续训；新实验使用不同目录，避免覆盖发布模型。

在项目根目录校验：

```bash
sha256sum -c neural_tracker/models/SHA256SUMS
```

全部应显示 `OK`。训练与部署方法见[模型快速上手](../README.md)，复现范围见[复现验收](../../docs/reproduction/README.md)。
