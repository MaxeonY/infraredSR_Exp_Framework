# InfraredSR

InfraredSR 是一个面向红外图像单帧超分辨率（SISR）的 PyTorch 项目，目标是基于 M3FD 红外图像构建可复现的训练、评估、对比和推理流程。

## 项目目的

1. 统一红外 SR 的数据处理、训练、测试与可视化流程。
2. 在同一数据切分上比较 `SRCNN / SRCNN_ARF / FSRCNN / EDSR / EDSR_ARF / RCAN / LDynSR` 的效果差异。
3. 产出可直接使用的推理脚本，对单图或目录批量生成 SR 结果。

## 已完成部分（截至 2026-03-23）

1. 数据链路已完成  
   已完成 M3FD 红外图像扫描、切分文件生成（`train/val/test`）与 Dataset 封装，支持训练随机裁剪、测试整图加载和路径鲁棒处理。
2. 模型实现已完成  
   `SRCNN / SRCNN_ARF / FSRCNN / EDSR / EDSR_ARF / RCAN / LDynSR` 可通过统一注册表与工厂接口构建。
3. 训练链路已完成  
   `train.py` 已支持完整训练流程、日志记录、`best/latest` checkpoint 保存。
4. 测试与对比链路已完成  
   `test.py` 负责单模型/多模型评估；`compare_results.py` 负责指标图表与跨模型/跨尺度对比导出。
5. 推理脚本已完成  
   新增 `infer.py`，支持单图/目录推理、自动从 checkpoint 推断模型与倍率、可选 GT 指标计算与可视化输出。

## 当前实验结果（M3FD test, x4, 420 张）

| Model  | PSNR   | SSIM    | L1       |
|--------|--------|---------|----------|
| SRCNN  | 34.9807 | 0.940599 | 0.008615 |
| FSRCNN | 32.5005 | 0.901788 | 0.011732 |
| EDSR   | 35.4182 | 0.945398 | 0.008131 |
| RCAN   | 35.3447 | 0.945224 | 0.008215 |
| LDynSR | TBD     | TBD      | TBD      |

当前 x4 设定下，`EDSR` 的平均 PSNR 最优（来自 `outputs/results/comparison/all_models_summary.csv`）。

## 项目结构（当前仓库）

```text
infraredSR/
├─ data/
│  ├─ raw/M3FD/
│  └─ processed/
├─ datasets/
│  ├─ preprocess.py
│  ├─ degrade.py
│  ├─ m3fd_dataset.py
│  └─ ...
├─ models/
│  ├─ srcnn.py
│  ├─ srcnn_arf.py
│  ├─ fsrcnn.py
│  ├─ edsr.py
│  ├─ edsr_arf.py
│  ├─ rcan.py
│  ├─ registry.py
│  ├─ builder.py
│  ├─ ldynsr/
│  │  ├─ dam.py / pa.py / bta.py / dyna.py / frm.py / ldynsr.py
│  │  └─ __init__.py
│  └─ __init__.py
├─ engine/
│  ├─ trainer.py
│  ├─ evaluator.py
│  └─ inferencer.py
├─ configs/
│  ├─ dataset/m3fd.yaml
│  ├─ train/default.yaml
│  ├─ train/ldynsr.yaml
│  └─ model/*.yaml
├─ utils/
│  ├─ checkpoint.py
│  ├─ logger.py
│  ├─ metrics.py
│  ├─ seed.py
│  └─ visualize.py
├─ outputs/
│  ├─ checkpoints/
│  ├─ logs/
│  └─ results/
├─ main.py
├─ train.py
├─ test.py
├─ compare_results.py
├─ infer.py
├─ requirements.txt
└─ README.md
```

## 环境安装

```bash
pip install -r requirements.txt
```

## 数据准备

1. 将 M3FD 数据放在 `data/raw/M3FD/`，保证存在 `Ir` 目录。
2. 生成切分文件：

```bash
python datasets/preprocess.py --overwrite
```

## 训练

所有模型统一使用 Adam 优化器（`lr=1e-4`, `weight_decay=0`，可通过 CLI 或 `configs/train/*.yaml` 覆盖）。

示例（x4）：

```bash
python train.py --model srcnn     --scale 4 --batch_size 4 --epochs 20
python train.py --model srcnn_arf --scale 4 --batch_size 4 --epochs 20
python train.py --model fsrcnn    --scale 4 --batch_size 4 --epochs 20
python train.py --model edsr      --scale 4 --batch_size 4 --epochs 20
python train.py --model edsr_arf  --scale 4 --batch_size 4 --epochs 20
python train.py --model rcan      --scale 4 --batch_size 4 --epochs 20
python train.py --model ldynsr    --scale 4 --batch_size 8 --epochs 30
```

训练完成后会在 `outputs/results/training_profiles/` 自动生成：

1. `*.json`：包含参数量、FLOPs、总训练时间、每 epoch 时间、最佳 PSNR/SSIM 等信息
2. `*.csv`：每 epoch 的训练/验证时间、估算 FLOPs、指标明细

## 测试与对比

单模型评估：

```bash
python test.py --model edsr --scale 4
```

全部模型评估：

```bash
python test.py --all_models --scales 4
```

基于测试报告生成对比图表与汇总：

```bash
python compare_results.py --save_results_dir outputs/results
```

默认会按测试范围自动分组导出（例如 `all_samples`、`single sample`），同时生成：

1. 参数图表（PSNR/SSIM/L1）
2. 效果对比图（`effect_comparison` 与单模型 `effect_gallery`）

注意：效果对比图依赖 `test.py` 生成的 `figures/sequential/*_cmp.png`，因此测试阶段不要使用 `--no_visuals`。

单模型横向对比（示例）：

```bash
python compare_results.py \
  --target_report outputs/results/edsr_x4/edsr_x4_test_report.txt \
  --quick_compare
```

## 统一入口（main.py）

`main.py` 用于统一调度 `preprocess/train/test/compare_results/infer` 脚本，参数会原样透传到对应脚本。

示例：

```bash
python main.py preprocess -- --overwrite
python main.py train -- --model edsr --scale 4 --batch_size 4 --epochs 20
python main.py test -- --all_models --scales 4
python main.py compare_results -- --save_results_dir outputs/results
python main.py infer -- --input path/to/lr_dir --recursive --checkpoint outputs/checkpoints/edsr_x4_best.pth
```

## 推理（infer.py）

单张图推理：

```bash
python infer.py --input path/to/lr.png --model edsr --scale 4
```

用 checkpoint 自动推断模型与倍率：

```bash
python infer.py --input path/to/lr.png --checkpoint outputs/checkpoints/edsr_x4_best.pth
```

目录批量推理（递归）并对齐 GT 计算指标：

```bash
python infer.py \
  --input path/to/lr_dir \
  --recursive \
  --checkpoint outputs/checkpoints/edsr_x4_best.pth \
  --gt path/to/hr_dir \
  --save_visuals
```

默认输出到 `outputs/infer/`，并生成：

1. SR 图像：`*_sr_{model}_x{scale}.png`
2. 指标表：`infer_{model}_x{scale}_metrics.csv`（提供 `--gt` 时含 PSNR/SSIM/L1）
3. 可选可视化：`visuals/` 下的对比图和误差图（`--save_visuals`）

## LDynSR 复现说明

`LDynSR` 结构已按论文主干接入：浅层卷积 -> DFEB(`DynA`堆叠) -> FRM -> 重建，并与 bicubic 残差相加。  
其中 `BTA` 在论文中张量级融合细节不完全展开，当前实现采用“亮度分支 + 纹理分支 + 融合注意图”的合理补全，代码内已标注该复现补全假设。
