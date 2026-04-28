# InfraredSR 实验框架项目报告

> 项目日期：2026-04-28
>
> 基于 M3FD 红外数据集的统一单帧超分辨率（SISR）实验平台，覆盖 SRCNN / FSRCNN / EDSR / EDSR_ARF / EDSR_ARFMk2 / RCAN / LDynSR 共 8 种模型。

---

## 一、项目定位

InfraredSR 是一个 **PyTorch 红外图像超分辨率实验框架**，核心设计理念是：

1. **统一平台**：所有模型共享同一套数据切分、退化流程、训练管线、测试链路和对比导出体系。
2. **可复现**：通过固定随机种子、确定性退化流程、结构化配置管理确保实验可复现。
3. **可扩展**：新增模型只需在 `models/` 下实现并注册，无需改造入口脚本。
4. **可对比**：内置跨模型/跨尺度结果汇总、指标图表生成、效果画廊等对比工具。

---

## 二、目录结构

```
infraredSR/
│
├── main.py                     # 统一调度入口（preprocess / train / test / compare / infer）
├── train.py                    # 训练入口
├── test.py                     # 测试/评估入口
├── infer.py                    # 推理入口
├── compare_results.py          # 跨模型对比与可视化
├── requirements.txt            # Python 依赖
├── .gitignore
├── README.md                   # 项目说明（中文）
├── RECONSTRUCTION.md           # LDynSR 接入重构任务书
│
├── configs/                    # ★ 配置体系
│   ├── dataset/
│   │   └── m3fd.yaml           #   数据退化管线配置
│   ├── model/
│   │   ├── srcnn.yaml
│   │   ├── srcnn_arf.yaml
│   │   ├── fsrcnn.yaml
│   │   ├── edsr.yaml
│   │   ├── edsr_arf.yaml
│   │   ├── edsr_arfmk2.yaml
│   │   ├── rcan.yaml
│   │   └── ldynsr.yaml
│   └── train/
│       ├── default.yaml        #   默认训练超参
│       └── ldynsr.yaml         #   LDynSR 专用训练超参
│
├── data/                       # ★ 数据层
│   ├── raw/
│   │   ├── KAIST/              #   （可选）KAIST 数据集下载位置
│   │   └── M3FD/Ir/            #   M3FD 红外原始图像（PNG 序列）
│   └── processed/
│       ├── train.txt           #   3360 张训练图像路径
│       ├── val.txt             #   420 张验证图像路径
│       ├── test.txt            #   420 张测试图像路径
│       ├── train_groups.txt
│       ├── val_groups.txt
│       └── test_groups.txt
│
├── datasets/                   # ★ 数据集实现
│   ├── __init__.py
│   ├── builder.py              #   Dataset 工厂
│   ├── degrade.py              #   退化函数（bicubic / noise / JPEG）
│   ├── download.py             #   KAIST Google Drive 下载
│   ├── m3fd_dataset.py         #   M3FDSRDataset 主类
│   ├── kaist_dataset.py        #   兼容别名（KAISTSRDataset = M3FDSRDataset）
│   ├── preprocess.py           #   扫描 + 切分文件生成
│   └── transforms.py           #   compose 工具
│
├── models/                     # ★ 模型实现
│   ├── __init__.py             #   统一导出
│   ├── registry.py             #   模型注册表
│   ├── builder.py              #   模型工厂 + 默认参数字典
│   ├── srcnn.py                #   SRCNN
│   ├── srcnn_arf.py            #   SRCNN + Adaptive Receptive Field
│   ├── fsrcnn.py               #   FSRCNN
│   ├── edsr.py                 #   EDSR
│   ├── edsr_arf.py             #   EDSR + ARF
│   ├── edsr_arfmk2.py          #   EDSR + ARF Mk2（多分支软路由）
│   ├── rcan.py                 #   RCAN
│   └── ldynsr/                 #   LDynSR 子模块
│       ├── __init__.py
│       ├── common.py           #     公共卷积辅助
│       ├── dam.py              #     Dynamic Attention Mixer
│       ├── pa.py               #     Pixel Attention
│       ├── bta.py              #     Brightness-Texture Attention
│       ├── dyna.py             #     Dynamic Attention block
│       ├── frm.py              #     Feature Reconstruction Module
│       └── ldynsr.py           #     顶层 LDynSR 模型
│
├── engine/                     # ★ 执行引擎
│   ├── __init__.py
│   ├── trainer.py              #   训练循环、验证、FLOPs 分析
│   ├── evaluator.py            #   测试评估、指标计算、可视化
│   └── inferencer.py           #   推理（单图/目录）
│
├── utils/                      # ★ 工具函数
│   ├── __init__.py
│   ├── checkpoint.py           #   断点保存/加载
│   ├── logger.py               #   日志配置
│   ├── metrics.py              #   PSNR / SSIM 计算
│   ├── misc.py                 #   YAML 加载、字典合并
│   ├── seed.py                 #   随机种子管理
│   └── visualize.py            #   对比图、差异图生成
│
└── outputs/                    # ★ 输出产物
    ├── checkpoints/            #   模型断点（best / latest）
    ├── logs/                   #   训练/测试/推理日志
    ├── figures/                #   （预留）图表
    └── results/                #   测试结果、对比输出
        ├── comparison/         #     跨模型汇总图表 + CSV
        ├── training_profiles/  #     训练性能档案（JSON / CSV）
        └── {model}_x{scale}/   #     各模型测试结果
```

---

## 三、数据管道

### 3.1 原始数据

| 项目 | 说明 |
|------|------|
| 数据集 | M3FD 红外数据集 |
| 图像位置 | `data/raw/M3FD/Ir/` |
| 图像格式 | 灰度 PNG（如 `00001.png`） |
| 切分比例 | 训练 80% / 验证 10% / 测试 10%（共约 4200 张） |
| 切分生成 | `python datasets/preprocess.py --overwrite` |

### 3.2 退化管线

定义在 `configs/dataset/m3fd.yaml`：

```yaml
degradation:
  downsample: bicubic          # 下采样方式
  noise:
    type: gaussian
    prob: 0.5                  # 50% 概率施加噪声
    sigma: [1, 10]             # 噪声标准差范围
  jpeg:
    prob: 0.5                  # 50% 概率施加 JPEG 压缩
    quality: [30, 95]          # 压缩质量范围
  deterministic_eval: true     # 评估时使用确定性退化（per-image 种子）
  eval_seed: 3407
```

退化流程（`datasets/degrade.py`）：

```
HR → mod_crop (边界对齐) → bicubic 下采样 (scale) → LR
LR → (可选: Gaussian noise, 50%) → (可选: JPEG, 50%) → 最终 LR
```

### 3.3 M3FDSRDataset 核心逻辑

| 模式 | 行为 |
|------|------|
| **训练** | 随机裁剪 64×64 patch、随机翻转/旋转、实时 bicubic 退化 + 随机噪声/JPEG |
| **验证** | 整图加载、确定性退化（按图像路径+scale+seed 的 SHA256 哈希确定 RNG 种子） |
| **测试** | 同验证模式，返回 `(lr, hr, path)` 用于结果追溯 |

### 3.4 关键特性

- **路径鲁棒性**：自动修正绝对/相对路径，支持 `data/processed/*.txt` 中存储的路径跨平台使用。
- **LR 缓存**：训练时可选缓存退化后的 LR 图像以加速迭代。
- **确定性退化**：验证/测试时每个图像使用固定种子，保证跨模型退化一致。

---

## 四、模型体系

### 4.1 已注册模型总览

| 模型名称 | 文件 | 参数量级 | 特点 |
|----------|------|----------|------|
| `srcnn` | `models/srcnn.py` | ~57K | 经典 3 层 Conv (9,5,5)，需预上采样输入 |
| `srcnn_arf` | `models/srcnn_arf.py` | ~57K | 自适应核尺寸（深度/通道计算 kernel size） |
| `fsrcnn` | `models/fsrcnn.py` | ~12K | 收缩-映射-展开 + 转置卷积上采样，PReLU |
| `edsr` | `models/edsr.py` | ~1.5M | 残差块堆叠 + PixelShuffle，res_scale=0.1 |
| `edsr_arf` | `models/edsr_arf.py` | ~1.5M | EDSR 全残差体用自适应核 |
| `edsr_arfmk2` | `models/edsr_arfmk2.py` | ~1.5M | 3 路深度可分离分支 + MLP 软路由融合 |
| `rcan` | `models/rcan.py` | ~15M+ | 残差组 + RCAB + 通道注意力（Sigmoid 门控） |
| `ldynsr` | `models/ldynsr/ldynsr.py` | ~500K | 轻量动态注意力 + 亮度-纹理双分支 |

### 4.2 模型注册机制

**注册表** (`models/registry.py`)：

```python
MODEL_REGISTRY = {}  # name -> class

def register_model(name):       # 装饰器注册
def get_model_class(name):      # 按名查询
def list_models():              # 列出所有注册模型
def build_registered(name, **kwargs):  # 统一构建
```

**构建工厂** (`models/builder.py`)：

```python
def build_model(name, scale=None, model_kwargs=None):
    # 1. 合并默认参数 + YAML 配置 + 显式覆盖
    # 2. 调用 build_registered() 构建实例
    # 3. SRCNN/SRCNN_ARF 不传 scale；其余模型传 scale 作为架构参数
```

### 4.3 模型架构细节

#### SRCNN
```
输入 (bicubic 上采样后) → Conv9 + ReLU → Conv5 + ReLU → Conv5 → 输出
```
亮点：权重使用 Gauss(0, 0.001) 初始化。

#### SRCNN_ARF
与 SRCNN 结构相同，但卷积核尺寸通过 ARF 公式计算：
```
d_i = min(s_i, t_i)
c_i 由 desigmoid 归一化
kernel = _nearest_odd(d_i)
```
支持手动 `kernel_override`。

#### FSRCNN
```
Conv5 (特征提取) → Conv1 (收缩) → m×Conv3 (映射) → Conv1 (展开) → ConvTranspose9 (上采样)
```
亮点：中心映射层数 `m` 可配置，PReLU 激活。

#### EDSR
```
Conv3 (head) → N×ResBlock(res_scale=0.1) + skip → UpsampleBlock(PixelShuffle) → Conv3 (tail)
```
亮点：移除 BN 层、残差缩放、Kaiming 初始化。

#### EDSR_ARF
EDSR 的基础上，所有卷积层（head、resblock 内两个卷积、body tail、upsample、tail）均替换为自适应核尺寸。

#### EDSR_ARFMk2
EDSR_ARF 的升级版本。每层使用 3 个并行深度可分离分支：
- DW-3×3-d1（标准 3×3）
- DW-3×3-d2（空洞 3×3, rate=2）
- DW-5×5-d1（5×5 感受野）

通过 MLP 路由器（输入 `[d_i, c_i, d_i*c_i]` → Softmax）生成 3 个分支的融合权重。
融合后接 1×1 Conv 做通道混合。

**注意**：MK2 使用深度可分离卷积，参数量远低于标准 EDSR_ARF。

#### RCAN
```
Conv3 (head) → G×RCAB blocks (每组 N 个 RCAB + Conv3 tail) + skip → UpsampleBlock → Conv3 (output)
```
每个 RCAB：`Conv+ReLU+Conv → Channel Attention (GAP→FC→ReLU→FC→Sigmoid) → scale → res_add(res_scale=0.1)`

#### LDynSR

```
输入 LR → Conv3 (浅层特征)
       → DFEB: [DynA × N] → Conv3
       → FRM (上采样)
       → Conv3 (重建)
       + Bicubic 上采样 (全局残差)
       → 输出 SR
```

**DynA**（Dynamic Attention Block）：
```
输入 → Conv1 (pre) → 三分支并行:
  ├─ PA (Pixel Attention)
  ├─ BTA (Brightness-Texture Attention)
  └─ NOA (普通 3×3 Conv)
  → DAM (动态权重融合)
  → Conv1 (post)
  → + 输入 (残差)
```

**DAM**（Dynamic Attention Mixer）：
```
GAP → FC → ReLU → FC → Softmax → 3 分支权重 (和为 1)
```

**PA**（Pixel Attention）：
```
1×1 Conv + Sigmoid (权重) × 3×3 Conv (特征) → 残差 → 3×3 Conv (精炼)
```

**BTA**（Brightness-Texture Attention）—— 论文不完整处做了合理补全：
```
亮度分支: GAP → MLP → Sigmoid
纹理分支: 3×3 Conv → 1×1 Conv → Sigmoid
两分支注意图 → Concat → 调制输入特征
```

**FRM**（Feature Reconstruction Module）：
- x2: 1 级 `ConvTranspose → PA → Conv3`
- x4: 2 级 `ConvTranspose → PA → Conv3`

---

## 五、引擎层

### 5.1 Trainer (`engine/trainer.py`)

训练循环核心功能：

| 功能 | 说明 |
|------|------|
| 优化器 | Adam (`lr=1e-4`, `weight_decay=0`) |
| 损失函数 | L1 Loss |
| 混合精度 | CUDA 可用时自动启用 AMP (`torch.cuda.amp`) |
| 验证指标 | PSNR / SSIM |
| Checkpoint 保存 | `best`（按 PSNR）+ `latest`（每 epoch） |
| FLOPs 分析 | 注册 forward hook，估算 MACs 和参数量 |
| 训练档案 | 输出 `training_profiles/*.json` 和 `*.csv`，含每 epoch 时间、指标、FLOPs |

### 5.2 Evaluator (`engine/evaluator.py`)

测试评估核心功能：

| 功能 | 说明 |
|------|------|
| 单模型评估 | 加载 best checkpoint，遍历测试集 |
| 全模型评估 | 遍历注册表中所有模型 + 所有指定 scale |
| 指标输出 | 每样本 PSNR/SSIM/L1 + 统计平均 |
| 可视化 | 保存 LR/SR/HR 三栏对比图、|SR-HR| 差异图(带 colorbar) |
| 排行可视化 | 按 PSNR 排序，输出 best/worst 样本效果 |
| 测试报告 | 生成 `{model}_x{scale}_test_report.txt`（含配置回溯） |

### 5.3 Inferencer (`engine/inferencer.py`)

推理引擎核心功能：

| 功能 | 说明 |
|------|------|
| 单图推理 | 输入单张 LR，输出 SR |
| 目录批量推理 | 支持递归扫描，按文件名匹配 |
| 自动模型解析 | 从 checkpoint 元数据读取 `model_name` 和 `scale` |
| GT 指标计算 | 提供 `--gt` 时自动计算 PSNR/SSIM/L1 并导出 CSV |
| 可视化 | `--save_visuals` 输出对比图和差异图 |

---

## 六、配置体系

采用 **三级配置覆盖** 机制：

```
模型默认参数 (models/builder.py 中 MODEL_DEFAULT_KWARGS)
  ← 被 YAML 配置覆盖 (configs/model/*.yaml)
    ← 被 CLI 参数覆盖 (train.py 的 --batch_size 等)
```

### 6.1 训练配置

默认训练配置 (`configs/train/default.yaml`)：

```yaml
train:
  patch_size: 64
  batch_size: 16
  epochs: 20
  lr: 1e-4
  weight_decay: 0
  seed: 42
  val_interval: 1         # 每 N 个 epoch 验证一次
```

LDynSR 专用覆盖 (`configs/train/ldynsr.yaml`)：

```yaml
train:
  batch_size: 8
  epochs: 30
  lr: 2e-4
```

### 6.2 模型配置示例

EDSR 配置 (`configs/model/edsr.yaml`)：

```yaml
model:
  in_channels: 1
  out_channels: 1
  feature_channels: 64
  n_resblocks: 16
  res_scale: 0.1
```

LDynSR 配置 (`configs/model/ldynsr.yaml`)：

```yaml
model:
  in_channels: 1
  out_channels: 1
  feat_channels: 48
  num_dyna: 6
  dam_reduction: 16
```

---

## 七、指标与可视化

### 7.1 评价指标

| 指标 | 实现位置 | 说明 |
|------|----------|------|
| PSNR | `utils/metrics.py` | 支持单通道图像，Tensor 或 ndarray 输入 |
| SSIM | `utils/metrics.py` | 基于 scikit-image，单通道，data_range 自适应 |
| L1 | `trainer.py` 训练 / `evaluator.py` 测试 | 平均绝对误差 |

### 7.2 可视化工具

| 功能 | 实现位置 | 生成内容 |
|------|----------|----------|
| 三栏对比图 | `utils/visualize.py:save_comparison_figure()` | LR / SR / HR 并列 |
| 差异热力图 | `utils/visualize.py:save_difference_map()` | \|SR-HR\| + colorbar |
| 效果画廊 | `compare_results.py` | 多模型同图横向效果对比 |
| 指标曲线 | `compare_results.py` | PSNR/SSIM 柱状图、散点图、直方图 |
| 跨尺度对比 | `compare_results.py` | 同模型不同 scale 指标对比 |

---

## 八、使用方法

### 8.1 环境安装

```bash
pip install -r requirements.txt
```

### 8.2 数据准备

```bash
# 将 M3FD 红外图像放入 data/raw/M3FD/Ir/
python datasets/preprocess.py --overwrite
```

### 8.3 训练

```bash
python train.py --model edsr --scale 4 --batch_size 4 --epochs 20
python train.py --model ldynsr --scale 4 --batch_size 8 --epochs 30
```

### 8.4 测试

```bash
# 单模型
python test.py --model edsr --scale 4

# 全模型
python test.py --all_models --scales 4

# 单样本
python test.py --model edsr --scale 4 --sample_index 42
```

### 8.5 跨模型对比

```bash
python compare_results.py --save_results_dir outputs/results
```

### 8.6 推理

```bash
# 从 checkpoint 自动推理
python infer.py --input path/to/lr.png --checkpoint outputs/checkpoints/edsr_x4_best.pth

# 批量推理 + GT 指标
python infer.py --input path/to/lr_dir --recursive --checkpoint ckpt.pth --gt path/to/hr_dir
```

### 8.7 统一入口

```bash
python main.py train -- --model edsr --scale 4
python main.py test -- --all_models --scales 4
python main.py infer -- --input lr.png --checkpoint ckpt.pth
```

---

## 九、当前实验结果

基于 M3FD 测试集（420 张，x4 倍率）的最新指标：

| Model  | PSNR     | SSIM     | L1        | 参数量 |
|--------|----------|----------|-----------|--------|
| SRCNN  | 34.9807  | 0.940599 | 0.008615  | ~57K   |
| FSRCNN | 32.5005  | 0.901788 | 0.011732  | ~12K   |
| EDSR   | 35.4182  | 0.945398 | 0.008131  | ~1.5M  |
| RCAN   | 35.3447  | 0.945224 | 0.008215  | ~15M+  |
| LDynSR | TBD      | TBD      | TBD       | ~500K  |

**结论**：x4 设定下 **EDSR** 以较低参数量取得了最高 PSNR，EDSR 和 RCAN 效果相近。SRCNN 以极小参数量表现优异。FSRCNN 作为轻量级模型参数量最低但指标相对落后。

---

## 十、设计要点与最佳实践

### 10.1 可复现性设计

- **确定性退化**：验证/测试时每个图像使用基于路径+scale+seed 的 SHA256 哈希作为 RNG 种子。
- **全局种子管理**：`set_seed(seed)` 统一设置 Python random、NumPy、PyTorch CPU/CUDA、cuDNN deterministic。
- **DataLoader 种子**：`seed_worker()` 确保多进程 DataLoader 的可重复性。

### 10.2 模型接入规范

接入新模型到平台的标准步骤：

1. 在 `models/` 下实现模型类
2. 在 `models/builder.py` 中用 `@register_model(name)` 注册
3. 在 `MODEL_DEFAULT_KWARGS` 中填写默认架构参数
4. 在 `configs/model/` 下添加 YAML 配置（可选）
5. 在 `configs/train/` 下添加训练配置覆盖（可选）
6. 在 `build_model()` 中添加 scale 参数处理逻辑（如需要）

### 10.3 Checkpoint 元数据规范

每个 checkpoint 保存以下元数据以实现自动恢复：

```python
{
    'model_state_dict': ...,
    'optimizer_state_dict': ...,
    'epoch': ...,
    'best_metric': ...,
    'model_meta': {
        'model_name': 'ldynsr',     # 注册表名称
        'scale': 4,                 # 超分倍率
        'model_kwargs': {...}       # 完整构造参数
    }
}
```

### 10.4 输出目录规范

```
outputs/
├── checkpoints/
│   ├── {model}_x{scale}_best.pth
│   └── {model}_x{scale}_latest.pth
├── logs/
│   ├── train_{timestamp}.log
│   ├── test_{model}_x{scale}_{timestamp}.log
│   └── infer_{model}_x{scale}_{timestamp}.log
└── results/
    ├── {model}_x{scale}/
    │   ├── {model}_x{scale}_test_report.txt     # 测试报告
    │   ├── {model}_x{scale}_metrics_per_sample.csv
    │   ├── figures/
    │   │   ├── comparison/                       # LR-SR-HR 对比图
    │   │   ├── difference/                       # |SR-HR| 差异图
    │   │   └── sequential/                       # 序列样本
    │   └── ranked/                               # best/worst 排行
    ├── comparison/                               # 跨模型汇总
    │   ├── all_models_summary.csv
    │   ├── psnr_comparison.png
    │   ├── ssim_comparison.png
    │   └── effect_comparison/
    └── training_profiles/
        ├── {model}_x{scale}_profile.json
        └── {model}_x{scale}_profile.csv
```

---

## 十一、代码量化统计

| 模块 | 文件数 | 核心代码行数（约） |
|------|--------|-------------------|
| `datasets/` | 7 | 450 |
| `models/` | 16 | 1300 |
| `engine/` | 4 | 850 |
| `utils/` | 6 | 500 |
| `configs/` | 11 | 120 |
| 入口脚本 | 5 | 540 |
| **合计** | **49** | **~3760** |

---

## 十二、扩展建议

1. **更多退化协议**：当前使用 M3FD + bicubic/noise/JPEG 统一协议，可扩展支持 FLIR、Thermal101、VAIS 等数据集，以及 MATLAB 风格的退化协议（如 MATLAB imresize）。
2. **更多模型**：可继续接入 SwinIR、HAT、DAT、DASR 等基于 Transformer 的 SR 模型。
3. **感知损失**：支持额外的感知损失（LPIPS）或对抗损失（GAN）训练。
4. **YAML 全配置化**：进一步将 CLI 参数全部并入 YAML，实现 `train.py --config experiment.yaml` 的全配置驱动训练。
5. **实验管理**：集成实验管理工具（如 MLflow、WandB）自动记录指标、配置和产物。
6. **ONNX 导出**：增加 ONNX/TensorRT 导出能力，支持边缘端部署。
