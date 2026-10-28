# 下载与复现验收

支持 Ubuntu 24.04、x86_64、Python 3.12、ROS 2 Jazzy。先按[总 README](../../README.md)安装 ROS / 系统依赖，再按[训练指南](../training/README.md)安装 `requirements-lock.txt` 中的 CPU 环境，无需 GPU。

## 下载后检查

```bash
git clone --branch repro-20261028 https://github.com/Bling-sun/LECO.git
cd LECO
export TRACKER_ROOT="$(pwd -P)"
sha256sum -c neural_tracker/models/SHA256SUMS
```

`repro-20261028` 固定本次验收的源码和模型版本。两套发布模型直接随仓库提供；无需 Git LFS。本地 `build/`、`install/` 和 `.venv/` 由构建 / 安装命令生成。

## 一条命令验证完整链路

在已安装依赖的项目根目录执行：

```bash
/usr/bin/python3 tools/verify_reproduction.py --ros-domain-id 187
```

脚本依次执行：干净源码构建、ROS 测试、模型回归、真实 MPC 专家采集、数据质量检查、episode 划分、CPU 小规模训练、评估、ONNX 导出、随机路线生成、三个后端启动 / 停止，以及发布模型和新模型的闭环指标实验。使用独立 ROS 域，退出时清理本次会话。

预期最终打印 `passed`，详细日志及 `summary.json` 在 `.runtime/reproduction_<时间>/`。`--skip-build` 可复用当前 clone 已构建的产物。新训练权重保存在 `neural_tracker/models/reproduce_<时间>/`，不要覆盖发布模型。

## 复现范围

- 可以直接运行仓库发布的模型，或重新采集数据完成训练、导出和仿真全流程。
- 小规模训练只验收功能，不代表完整训练质量或论文基准。
- 历史 v41 原始训练集和完整随机状态未包含在当前项目中，不能保证重新训练得到逐位相同的权重或论文数值。
- ROS 采集受异步时序影响；正式实验须保存原始 episode、划分清单、地图 / 路线、模型哈希及配置。

## 本次发布验收：2026-10-28

在独立源码目录和全新 Python 环境中验收：3 个 ROS 包构建成功，44 项 ROS / 实验测试、14 项模型回归测试全部通过；重新采集、合并、训练、评估、导出和三个后端启动 / 停止均通过。发布模型与新训练模型都完成了有效且成功的闭环试次，并确认实际使用神经分支。

细节：[验收 JSON](verification.json) · [Python 环境](python-environment.json) · [锁定依赖](../../neural_tracker/requirements-lock.txt)。使用了已有 ROS 系统依赖，未在全新操作系统中重装 apt 环境；GPU 训练和完整论文基准未作为本次验收内容。
