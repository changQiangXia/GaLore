# Sophia&GaLore

一个面向小显存环境的大模型微调项目，核心目标是在受限硬件下，系统验证 GaLore（Gradient Low-Rank Projection）在全参数训练中的可用性、稳定性和工程落地路径。

## 项目定位

这个项目强调两件事：

1. 算法层面：低秩梯度投影在真实训练流程中的可执行实现
2. 工程层面：在复杂依赖组合下，保证训练可跑、可恢复、可复现

项目不是只“跑通脚本”，而是对以下问题给出可复现实验答案：

- 如何在有限显存下组织全参数微调流程
- 如何让 GaLore 与量化、梯度检查点、断点恢复协同工作
- 如何在不兼容配置下自动降级，避免训练中途崩溃

## 算法创新点

### 1. 形状安全的 GaLore Hook 回退实现

对应文件：`models/galore_hook.py`

- 在 backward hook 中完成低秩投影
- 保证 hook 返回梯度与原梯度形状一致
- 使用“投影到低秩子空间 + 投影回原空间”的策略兼容 autograd 约束

这使得自定义 GaLore 路径在 PyTorch 训练循环中稳定可用。

### 2. 兼容性感知的优化器路由策略

对应文件：`train.py`

- 自动识别 `8-bit + Linear8bitLt` 的高风险组合
- 在不安全组合下跳过易崩溃路径
- 在 GaLore 参数不可用时自动回退到稳定 AdamW 路径

核心价值是把“容易踩坑的配置空间”变成“可预期的训练策略”。

### 3. 量化配置的前置有效性检查

对应文件：`models/model_loader.py`

- 在训练前检查 int8 模块内部状态是否完整
- 对不兼容配置直接早失败并给出可执行修复建议

避免“启动很久后在深层调用栈崩溃”的低效调试模式。

### 4. 梯度累积与断点续训的语义修正

对应文件：`train.py`

- 修正 micro-batch 与 optimizer-step 对齐逻辑
- 修正中断恢复后 epoch 内跳步逻辑
- 处理尾部不足累积窗口的梯度刷新

保证中断前后训练行为一致，结果可比。

### 5. 短任务场景下的显存报告兜底

对应文件：`utils/memory_monitor.py`

- 即使未命中日志步，也能输出非空 summary
- 统一生成结构化 JSON 显存报告，便于实验对比

## 训练模式说明

项目支持两类实用模式：

### A. 小显存稳定模式（默认更稳）

- `quantization.enabled: true`
- `load_in_8bit: true`
- 多数情况下会走 AdamW 稳定路径

适合快速验证流程和显存行为。

### B. 真正 GaLore 模式（算法验证）

- `quantization.enabled: false`
- `optimizer.type: galore`

适合验证纯 GaLore 优化行为和收敛特性。

## 最新实验结果（本仓库实测）

| 模式 | 输出目录 | 第 3 个 epoch 的 Eval Loss | 峰值 Allocated | 峰值 Reserved |
|---|---|---:|---:|---:|
| 8-bit 稳定路径（AdamW） | `checkpoints/qwen2-0.5b-galore-fullrun-20260227_093005` | 1.3806 | 1.192 GB | 3.029 GB |
| 真 GaLore 路径 | `checkpoints/qwen2-0.5b-galore-fullrun-truegalore` | 1.2074 | 1.842 GB | 4.434 GB |

结论（当前实现与环境）：

- 真 GaLore 路径在该实验中取得更低的最终 eval loss
- 真 GaLore 路径显存占用更高

## 项目结构

```text
Sophia&GaLore/
- configs/
  - galore_config.yaml
  - galore_config.fullrun.yaml
  - galore_compare_galore.yaml
  - galore_compare_adamw.yaml
- data/
  - __init__.py
  - dataset.py
  - sample_data.json
- models/
  - __init__.py
  - model_loader.py
  - galore_hook.py
- utils/
  - __init__.py
  - checkpoint.py
  - memory_monitor.py
- train.py
- download_model.py
- download_model_modelscope.py
- requirements.txt
- LICENSE
```

## 快速开始

### 1) 环境准备

```bash
conda create -n galore python=3.10 -y
conda activate galore
pip install -r requirements.txt
```

### 2) 下载模型

```bash
python download_model.py
# 或
python download_model_modelscope.py
```

### 3) 生成示例数据

```bash
python train.py --prepare-data
```

### 4) 启动训练

```bash
python train.py --config configs/galore_config.fullrun.yaml
```

## 常用命令

对照实验：

```bash
python train.py --config configs/galore_compare_galore.yaml
python train.py --config configs/galore_compare_adamw.yaml
```

显式指定优化器：

```bash
python train.py --config configs/galore_config.yaml --optimizer galore
python train.py --config configs/galore_config.yaml --optimizer adamw
```

断点续训：

```bash
python train.py --config configs/galore_config.yaml --resume
```

## 可复现性说明

- 随机种子由配置文件统一控制（`training.seed`）
- checkpoint 保存模型、优化器、GaLore 状态和 RNG 状态
- 显存报告以 JSON 输出，便于复现实验和对比分析

## License

MIT License
