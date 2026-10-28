# 当前使用的模型

模型随 Git 仓库提供，无需 Git LFS 或额外下载。只保留启动脚本默认使用的两个文件：

| 文件 | 用途 |
| --- | --- |
| `v41_directional_recovery/policy.onnx` | 默认 C++ / ONNX 后端 |
| `v41_directional_recovery_retrain_clean_20260717/best.pth` | `--backend python` 的 PyTorch 后端 |

在项目根目录校验，两个文件均应显示 `OK`：

```bash
sha256sum -c neural_tracker/models/SHA256SUMS
```

重新训练时使用新目录；训练生成的 checkpoint 不纳入发布。用法见[模型快速上手](../README.md)和[完整复现流程](../../docs/reproduction/README.md)。
