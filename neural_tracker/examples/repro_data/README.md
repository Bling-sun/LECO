# CPU 流程验证数据

由本项目 ROS 录制器实采的两个独立 episode，各 121 帧，20×20 m blocks 地图，种子 100，目标初始 (-2, 0)，跟踪器初始 (-4.5, 0)，10 Hz，每段 12 秒。

按 episode、seed=42、val_ratio=0.25 合并，划分及原始文件哈希见 split_manifest.json。仅供小规模训练/导出流程验证。质量检查发现 sample_0001 目标丢失比例约 21.5%，高于默认 20% 阈值；保留它用于验证丢失样本及 mask 的处理。此数据不是历史 v41 训练集，也不是正式性能评测集。

从 neural_tracker/ 执行 `sha256sum -c examples/repro_data/SHA256SUMS`。运行步骤见 ../../../docs/training/README.md。
