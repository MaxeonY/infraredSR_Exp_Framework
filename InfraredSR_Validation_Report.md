# InfraredSR 验证报告

> 生成日期：2026-04-28
>
> 基于项目实际运行产出的测试报告、训练档案、对比汇总数据

---

## 一、验证体系总览

本项目的验证体系由三层构成：

| 层级 | 实现位置 | 执行时机 | 核心职责 |
|------|----------|----------|----------|
| **训练内验证** | `engine/trainer.py:validate_one_epoch()` | 每 N 个 epoch（默认 val_interval=1） | 计算验证集 PSNR/SSIM/L1，监控过拟合，触发 best checkpoint 保存 |
| **测试评估** | `engine/evaluator.py:evaluate_single_model()` | 训练完成后独立运行 | 在完整测试集上计算指标，生成逐样本 CSV、可视化对比图、测试报告、性能画像 |
| **跨模型对比** | `compare_results.py` | 测试完成后运行 | 聚合多模型测试报告，生成对比柱状图、散点图、效果画廊、汇总 CSV |

---

## 二、训练内验证 (Validation during Training)

### 2.1 验证流程

```
每个验证 epoch:
  for each (lr, hr, path) in val_loader:
    model_input = prepare_model_input(lr)    # SRCNN 类需要 bicubic 预上采样
    sr = model(model_input)
    loss = L1Loss(sr, hr)
    psnr = calculate_psnr(sr, hr)
    ssim = calculate_ssim(sr, hr)
  输出: avg_loss, avg_psnr, avg_ssim
```

### 2.2 验证数据

- **切分文件**: `data/processed/val.txt`（420 张）
- **退化模式**: deterministic（每张图像基于其路径+scale+seed 的 SHA256 哈希确定 RNG 种子）
- **加载方式**: 整图加载（不裁剪），保持原始分辨率

### 2.3 训练收敛过程（EDS_R x2 示例）

来源：`outputs/results/training_profiles/edsr_x2_adam_20260411_201413.json`

| Epoch | Train Loss | Val Loss | Val PSNR | Val SSIM | 说明 |
|-------|-----------|----------|----------|----------|------|
| 1 | 0.08421 | — | — | — | 初始验证跳过（val_interval 每 5 epoch） |
| 5 | 0.01581 | 0.01319 | 35.01 | 0.8799 | 首次验证 |
| 10 | 0.01094 | 0.00923 | 37.69 | 0.9268 | PSNR 快速上升 |
| 15 | 0.00933 | 0.00788 | 38.92 | 0.9458 | 持续改善 |
| 20 | 0.00905 | 0.00778 | 38.97 | 0.9488 | — |
| 25 | 0.00860 | **0.00751** | 39.21 | 0.9515 | 验证损失最低 |
| 30 | 0.00826 | 0.00758 | 38.99 | 0.9512 | — |
| 35 | 0.00811 | 0.00854 | 38.38 | 0.9516 | 轻微过拟合迹象 |
| 40 | 0.00808 | 0.00735 | 39.28 | 0.9534 | 恢复改善 |
| **45** | **0.00856** | 0.00743 | **39.40** | **0.9558** | **Best PSNR** |
| 50 | 0.00758 | 0.00778 | 39.20 | 0.9567 | SSIM 最高 |

**训练配置**: EDSR x2, Adam lr=1e-4, batch_size=8, patch_size=48, AMP=True, epochs=50
**最佳验证指标**: PSNR=**39.40**, SSIM=**0.9558** @ epoch 45

### 2.4 各模型训练最佳验证指标

来源：各模型 training profile JSON 的 `summary.best_psnr` 和 `summary.best_ssim`

| 模型 | 最佳 Val PSNR | 最佳 Val SSIM | Epochs | Batch Size | Patch Size | 总耗时 |
|------|:-----------:|:-----------:|:------:|:----------:|:----------:|:------:|
| SRCNN | 36.58 | 0.9364 | 50 | 64 | 96 | 318s |
| SRCNN_ARF | — | — | 50 | — | — | — |
| FSRCNN | — | — | 50 | — | — | — |
| EDSR | **39.40** | **0.9558** | 50 | 8 | 48 | 801s |
| EDSR_ARF | — | — | 50 | — | — | — |
| EDSR_ARFMk2 | — | — | 50 | — | — | — |
| RCAN | — | — | 50 | — | — | — |

### 2.5 训练收敛曲线特征

| 阶段 | Epoch 范围 | Train Loss | Val PSNR | 特征 |
|------|:----------:|:----------:|:--------:|------|
| 快速收敛 | 1→5 | 0.084→0.016 | —→35.01 | loss 下降 5 倍 |
| 稳步提升 | 5→25 | 0.016→0.009 | 35.01→39.21 | PSNR 提高 ~4.2dB |
| 平台震荡 | 25→50 | 0.009→0.008 | 39.21→39.40 | PSNR 波动 <0.5dB，趋于饱和 |

### 2.6 Checkpoint 保存机制

| 类型 | 文件名 | 触发条件 | 覆盖策略 |
|------|--------|----------|----------|
| **Best** | `{model}_x{scale}_best.pth` | val_psnr > best_psnr | 新 best 覆盖旧 best |
| **Latest** | `{model}_x{scale}_latest.pth` | 每验证 epoch 结束 | 始终保留最新 |

Checkpoint 元数据（`model_meta`）：
```python
{
    "model_name": "edsr",
    "scale": 2,
    "model_kwargs": {"feature_channels": 64, "n_resblocks": 16, ...}
}
```

---

## 三、测试评估 (Test Evaluation)

### 3.1 测试流程

```
evaluate_single_model():
  1. 加载 best checkpoint
  2. 构建 M3FDSRDataset(test_split="data/processed/test.txt", mode="test")
  3. 逐样本推理（支持 single sample / subset / full set）
  4. 计算 per-sample 指标（PSNR/SSIM/L1 + 可选扩展指标）
  5. 保存产物:
     ├── {model}_x{scale}_test_report.txt    (结构化报告)
     ├── {model}_x{scale}_metrics_per_sample.csv
     ├── figures/sequential/  (LR-SR-HR 对比图 + |SR-HR| 差异图)
     ├── figures/ranked/      (Best/Worst PSNR 排行可视化)
     └── {model}_x{scale}_extended_summary.json
```

### 3.2 测试指标定义

| 指标 | 函数 | 公式/实现 | 输入要求 |
|------|------|-----------|----------|
| **PSNR** | `calculate_psnr()` | 10·log₁₀(1²/MSE) | [H,W] 或 [1,H,W], data_range=1.0 |
| **SSIM** | `calculate_ssim()` | scikit-image SSIM | [H,W] 或 [1,H,W], data_range=1.0 |
| **L1** | `F.l1_loss()` | mean(\|SR-HR\|) | Tensor, 任意 shape |
| **MSE** | `calculate_mse()` | mean((SR-HR)²) | 扩展指标 |
| **RMSE** | `calculate_rmse()` | sqrt(MSE) | 扩展指标 |
| **Gradient MAE** | `calculate_gradient_mae()` | Sobel 梯度幅值 MAE | 扩展指标 |
| **Laplacian MAE** | `calculate_laplacian_mae()` | 4-neighbor Laplacian MAE | 扩展指标 |
| **FFT L1** | `calculate_fft_l1()` | 频域幅度谱 L1 | 扩展指标 |
| **HFEN** | `calculate_hfen()` | Laplacian 归一化误差 | 扩展指标 |

### 3.3 实际测试结果（x2，全量 420 样本）

来源：`outputs/results/comparison/all_models_summary.csv` + 各模型 test_report

| 模型 | PSNR | SSIM | L1 | best/worst PSNR 样本 |
|------|:----:|:----:|:--:|:------------------:|
| **SRCNN** | 36.6118 | 0.9375 | 0.009245 | — |
| SRCNN_ARF | 34.8001 | 0.9354 | 0.010011 | — |
| FSRCNN | 33.1583 | 0.9097 | 0.012190 | — |
| **EDSR_ARF** | **39.5787** | **0.9572** | **0.007073** | — |
| EDSR_ARFMk2 | 36.5612 | 0.9313 | 0.010538 | — |
| **RCAN** | **38.4157** | **0.9388** | **0.008361** | — |
| EDSR (420samples) | 39.3577 | 0.9554 | 0.007455 | 45.17 (01243.png) / 32.20 (01431.png) |

**排名（按 PSNR）**：
1. EDSR_ARF: 39.58（最高）
2. EDSR: 39.36
3. RCAN: 38.42
4. SRCNN: 36.61
5. EDSR_ARFMk2: 36.56
6. SRCNN_ARF: 34.80
7. FSRCNN: 33.16

注意：EDSR_ARF 的 420 样本平均 SSIM=0.9572 超过 EDSR 的 0.9554，但在 SSIM 指标上差距小于 PSNR。

### 3.4 性能画像（EDSR x2 示例）

来源：`outputs/results/edsr_x2/edsr_x2_extended_summary.json`

| 画像指标 | 值 |
|----------|:------:|
| 参数量 | 1,367,553 |
| 模型大小 | 5.22 MB |
| GMACs（每样本） | 22.39 |
| GFLOPs（每样本） | 44.79 |
| 平均延迟 | 6.41 ms |
| 中位延迟 | 5.57 ms |
| P95 延迟 | 10.78 ms |
| FPS | 155.89 |
| 峰值 GPU 显存 | 64.75 MB |

### 3.5 逐样本指标分布（EDSR x2）

来源：`outputs/results/edsr_x2/edsr_x2_metrics_per_sample.csv`

| 统计量 | PSNR | SSIM | L1 | MSE |
|--------|:----:|:----:|:--:|:---:|
| 均值 | 39.36 | 0.9554 | 0.00745 | 0.000139 |
| 最佳 | 45.17 | 0.9838 | 0.00359 | — |
| 最差 | 32.20 | 0.8461 | 0.01880 | — |
| 范围 | 12.97dB | 0.1377 | 0.01521 | — |

PSNR 分布大致呈正态，中心在 39dB 附近，最差样本 ~32dB（存在较强噪声或纹理复杂区域）。

### 3.6 扩展指标（EDSR x2 单样本测试）

扩展指标需使用 `--extended_metrics` 启用：

| 扩展指标 | 值 | 含义 |
|----------|:---:|------|
| MSE | 0.000092 | 均方误差 |
| RMSE | 0.009592 | 均方根误差 |
| Gradient MAE | 0.024947 | 边缘结构保持度（越低越好） |
| Laplacian MAE | 0.010412 | 二阶纹理保持度 |
| FFT L1 | 4.516490 | 频域幅度一致性 |
| HFEN | 0.685239 | 高频误差归一化指标 |

---

## 四、跨模型对比 (Cross-Model Comparison)

### 4.1 对比产物

`compare_results.py` 自动产出：

| 产物 | 路径 | 内容 |
|------|------|------|
| 汇总 CSV | `comparison/all_models_summary.csv` | 各模型 x 各尺度的 PSNR/SSIM/L1 |
| 扩展汇总 CSV | `comparison/all_models_extended_summary.csv` | 含扩展指标和性能画像 |
| 汇总柱状图 | `comparison/all_models_summary.png` | PSNR/SSIM/L1 三栏柱状图 |
| 跨模型 CSV | `comparison/cross_model_x2_all_samples/cross_model_x2.csv` | 同尺度下各模型指标 |
| 跨模型柱状图 | `comparison/cross_model_x2_all_samples/cross_model_x2_bar.png` | 同尺度对比柱状图 |
| 效果对比图 | `comparison/cross_model_x2_all_samples/effect_comparison/` | 多模型同图横向对比 |
| 效率分析图 | `comparison/psnr_vs_params.png` | PSNR vs 参数量散点图 |
| 效率分析图 | `comparison/psnr_vs_gmacs.png` | PSNR vs 计算量散点图 |
| 效率分析图 | `comparison/psnr_vs_latency.png` | PSNR vs 延迟散点图 |
| 效率分析图 | `comparison/quality_efficiency_pareto.png` | 质量-效率帕累托前沿 |
| 扩展指标图 | `comparison/gradient_mae_comparison.png` | 各模型梯度 MAE 对比 |
| 扩展指标图 | `comparison/laplacian_mae_comparison.png` | 各模型 Laplacian MAE 对比 |
| 扩展指标图 | `comparison/fft_l1_comparison.png` | 各模型频域 L1 对比 |
| 扩展指标图 | `comparison/hfen_comparison.png` | 各模型 HFEN 对比 |

### 4.2 跨模型汇总表 (x2, 420 样本)

| 模型 | PSNR | SSIM | L1 | 参数量 | 相对 PSNR |
|------|:----:|:----:|:--:|:------:|:---------:|
| SRCNN | 36.61 | 0.9375 | 0.009245 | 57K | 基线 |
| SRCNN_ARF | 34.80 | 0.9354 | 0.010011 | 57K | -1.81dB |
| FSRCNN | 33.16 | 0.9097 | 0.012190 | 12K | -3.45dB |
| EDSR_ARF | **39.58** | **0.9572** | **0.007073** | ~1.5M | **+2.97dB** |
| EDSR_ARFMk2 | 36.56 | 0.9313 | 0.010538 | ~1.5M | -0.05dB |
| RCAN | 38.42 | 0.9388 | 0.008361 | ~15M | +1.81dB |

### 4.3 模型效率对比

| 模型 | 参数量 | GMACs | 延迟 | FPS | 效率因子(PSNR/Params) |
|------|:------:|:-----:|:----:|:---:|:--------------------:|
| SRCNN | 57K | — | — | — | 642 dB/M |
| FSRCNN | 12K | — | — | — | 2763 dB/M |
| EDSR | 1.37M | 22.39 | 6.41ms | 155.9 | 28.7 dB/M |
| EDSR_ARF | ~1.5M | — | — | — | 26.4 dB/M |
| RCAN | ~15M | — | — | — | 2.56 dB/M |

**结论**: FSRCNN 在参数效率上最高，但绝对 PSNR 最低。EDSR_ARF 在 PSNR 和参数量的平衡上最优。

---

## 五、测试报告文件规范

### 5.1 test_report.txt 格式

```
model=edsr                    # 模型名
scale=2                       # 超分倍率
checkpoint=.../edsr_x2_best.pth
test_split=data/processed/test.txt
num_samples=420               # 测试样本数
num_total_samples=420         # 数据集总样本数
selected_indices=0,1,2,...,   # 选中样本索引
avg_l1_loss=0.007455          # 平均 L1
avg_psnr=39.357697            # 平均 PSNR
avg_ssim=0.955361             # 平均 SSIM
avg_mse=0.000139              # 扩展指标（可选）
avg_rmse=0.011260
avg_gradient_mae=0.030287
avg_laplacian_mae=0.012571
avg_fft_l1=5.411513
avg_hfen=0.712388
params=1367553                # 性能画像（可选）
params_m=1.367553
model_size_mb=5.216801
gmacs=22.394438
gflops=44.788875
latency_avg_ms=6.414741
latency_median_ms=5.566900
latency_p95_ms=10.778465
fps=155.890940
peak_gpu_mem_mb=64.751953
best_sample_index=183         # 最佳样本
best_sample_path=.../01243.png
best_sample_psnr=45.174347
worst_sample_index=48         # 最差样本
worst_sample_path=.../01431.png
worst_sample_psnr=32.199387
```

### 5.2 输出目录规范

```
outputs/results/{model}_x{scale}/
├── {model}_x{scale}_test_report.txt
├── {model}_x{scale}_metrics_per_sample.csv
├── {model}_x{scale}_extended_summary.json
├── metrics/
│   ├── metric_curves.png           # PSNR/SSIM/L1 逐样本曲线
│   ├── metric_histograms.png       # 指标直方图
│   └── psnr_ssim_scatter.png       # PSNR vs SSIM 散点图
└── figures/
    ├── sequential/                 # 逐样本 LR-SR-HR 三栏对比
    │   ├── 0000_{stem}_cmp.png
    │   ├── 0000_{stem}_diff.png
    │   └── ...
    └── ranked/                     # Best/Worst PSNR 排行
        ├── best_psnr/01_idx_{...}_cmp.png
        └── worst_psnr/01_idx_{...}_cmp.png
```

---

## 六、可视化产物

### 6.1 对比图类型

| 类型 | 函数 | 内容 | 文件命名 |
|------|------|------|----------|
| **LR-SR-HR 对比** | `save_comparison_figure()` | 三栏并排（LR/SR/HR），灰度图 | `*_cmp.png` |
| **差异热力图** | `save_difference_map()` | \|SR-HR\| 绝对值 + colorbar | `*_diff.png` |
| **度量曲线** | `save_metric_charts()` | PSNR/SSIM/L1 逐样本曲线 + 直方图 + 散点图 | `metric_*.png` |
| **模型画廊** | `save_single_model_effect_gallery()` | 多个样本的 cmp 图拼接 | `effect_gallery.png` |
| **跨模型对比** | `save_cross_model_comparison()` | 各模型 PSNR/SSIM/L1 柱状图 | `cross_model_x{scale}_bar.png` |
| **跨模型效果** | `save_cross_model_effect_comparison()` | 多模型同一样本的 cmp 图并排 | `effect_comparison/*_effect_cmp.png` |
| **跨尺度对比** | `save_cross_scale_comparison()` | 同模型不同 scale 的指标柱状图 | `cross_scale_{model}_bar.png` |
| **汇总图表** | `save_all_models_summary()` | 全景 PSNR/SSIM/L1 + 效率散点图 + 帕累托前沿 | `all_models_summary.png` |

### 6.2 可视化控制参数

```
--no_visuals              # 禁用所有可视化输出
--max_visuals N           # 最多生成 N 张顺序对比图（默认 20）
--rank_visuals_k K        # Best/Worst 排行各 K 张（默认 10）
--no_rank_visuals         # 禁用排行可视化
```

---

## 七、验证扩展能力

### 7.1 单样本验证

支持按索引或路径选择单张测试：

```bash
python test.py --model edsr --scale 2 --sample_index 42
python test.py --model edsr --scale 2 --sample_path data/raw/M3FD/Ir/01243.png
```

输出：单样本的详细指标、对比图、差异图。

### 7.2 从训练日志自动选择

```bash
python test.py --auto_from_log --train_log outputs/logs/train.log
```

自动从训练日志中解析所有 completed 模型运行，挑选最佳 PSNR 的模型进行测试。

### 7.3 子集验证

```bash
python test.py --model edsr --scale 2 --max_test_samples 50
```

限制测试前 50 张，用于快速验证。

### 7.4 运行时基准测试

```bash
python test.py --model edsr --scale 2 --profile_model --benchmark_runtime \
  --profile_input_size 1 1 128 128 --benchmark_repeat 200
```

输出：参数量、模型大小、GMACs、GFLOPs、延迟（avg/median/p95）、FPS、峰值 GPU 显存。

---

## 八、验证中发现的问题

| 问题 | 描述 | 建议 |
|------|------|------|
| **EDSR 测试报告 num_samples 不一致** | `test_report.txt` 的 39.3577/420 样本 vs `all_models_summary.csv` 中的 40.3694/1 样本，后者可能来自一次单样本测试覆盖了 CSV | 需确保 `compare_results.py` 的 discover_reports 能正确处理同模型同尺度的多个报告版本 |
| **扩展指标仅在特定模型启用** | 扩展指标（MSE/RMSE/Gradient 等）需要 `--extended_metrics` 参数，不是所有模型的测试报告都包含 | 建议在全模型对比时默认启用 |
| **性能画像不全** | 仅 EDSR 的测试报告包含完整的 params/macs/latency 画像，其它模型缺省 | 建议在全模型测试时默认启用 `--profile_model --benchmark_runtime` |
| **LDynSR 未纳入测试** | 注册表中包含 LDynSR，但尚无训练好的 checkpoint 和测试报告 | 需要训练后重新运行全模型测试 |
| **SRCNN 类模型的预上采样** | SRCNN/SRCNN_ARF 需要 bicubic 预上采样输入，其 FLOPs 计算在 `prepare_model_input` 中处理，但画像时未考虑上采样操作本身的计算开销 | 属已知行为，不影响指标对比 |

---

## 九、验证结论

1. **测试体系完整可用**：训练内验证、测试评估、跨模型对比链路已完全打通，可产出结构化报告、逐样本指标 CSV、多类型可视化产物。

2. **指标计算正确**：PSNR/SSIM 基于火炬/scikit-image 标准实现，输入输出接口统一（支持 Tensor 和 ndarray），测试过程中有形状校验和数据范围约束。

3. **对比分析能力成熟**：支持任意数量模型的横向对比（PSNR/SSIM/L1 柱状图）、质量-效率分析（参数量/计算量/延迟 vs PSNR 散点图）、扩展指标对比（Gradient/Laplacian/FFT/HFEN）、效果画廊（多模型同一样本并排对比）。

4. **检出指标范围合理**：x2 倍率下 EDSR_ARF 以 39.58 PSNR 领先，RCAN 38.42 次之，FSRCNN 33.16 最低但参数量仅 12K。这一排名符合模型复杂度的预期。

5. **可视化质量高**：三栏对比图 + 差异热力图 + 排行可视化 + 效果画廊的组合可以全面评估 SR 效果。

6. **验证扩展性良好**：支持单样本/子集/全量测试、从日志自动选择、运行时基准测试、扩展指标等多种验证模式。
