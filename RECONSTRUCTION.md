---

# 项目重构任务摘要（供 Agent 执行）

## 1. 总目标

在现有 `InfraredSR` 项目中**直接接入 LDynSR**，而不是新建独立项目。
重构后的项目应保持原有定位：**在同一数据集、同一预处理、同一训练/测试/推理链路、同一量化指标下，统一比较不同 SR 模型效果**。当前项目已具备这一平台基础，包括统一的数据处理、模型工厂、训练测试流程和对比导出能力。

LDynSR 接入后，应与 `SRCNN / FSRCNN / EDSR / RCAN` 处于同等地位，作为平台中的一个标准模型参与训练、测试、推理和横向比较。原论文中的 LDynSR 核心模块包括：浅层卷积、DFEB（由多个 DynA 组成）、FRM、全局 bicubic 残差；DynA 内含 DAM、PA、BTA 和 NOA 分支。

---

## 2. 重构原则

### 2.1 保持平台定位不变

不要为 LDynSR 单独建立专属数据流程或专属评测流程。
所有模型继续共享：

* 同一数据集
* 同一预处理/退化流程
* 同一训练与验证流程
* 同一测试与指标统计流程
* 同一结果对比导出流程。

### 2.2 向后兼容

重构完成后，旧模型 `SRCNN / FSRCNN / EDSR / RCAN` 的训练、测试、推理行为应保持不变。

### 2.3 模型接入标准化

新增模型时，应只需要：

1. 在 `models/` 下增加模型实现
2. 在模型注册表中登记

不应在多个脚本里写分散的 `if model == ...` 特判。

### 2.4 训练脚本与模型内部解耦

`train.py / test.py / infer.py` 应只负责调度，不应包含 LDynSR 特有的结构逻辑。
模型内部细节全部收敛到 `models/ldynsr/` 中。

---

## 3. 当前项目中可复用部分

根据 README，现有项目已经完成以下内容，可直接作为宿主平台保留：

* 数据链路：扫描、切分文件生成、Dataset 封装、训练随机裁剪、测试整图加载、路径鲁棒处理
* 模型工厂：`SRCNN / FSRCNN / EDSR / RCAN` 统一构建
* 训练链路：`train.py`、日志、best/latest checkpoint
* 测试与对比链路：`test.py`、`compare_results.py`
* 推理链路：`infer.py` 自动推断模型与倍率
* 指标体系：PSNR / SSIM / L1 统一输出。

因此，本次任务是**增量重构 + 新模型接入**，不是重写整个项目。

---

## 4. 重构后的目标目录结构

Agent 应尽量将项目整理为以下结构：

```text
infraredSR/
├─ configs/
│  ├─ dataset/
│  │  └─ m3fd.yaml
│  ├─ train/
│  │  ├─ default.yaml
│  │  └─ ldynsr.yaml
│  └─ model/
│     ├─ srcnn.yaml
│     ├─ fsrcnn.yaml
│     ├─ edsr.yaml
│     ├─ rcan.yaml
│     └─ ldynsr.yaml
│
├─ data/
│  ├─ raw/M3FD/
│  └─ processed/
│
├─ datasets/
│  ├─ preprocess.py
│  ├─ degrade.py
│  ├─ transforms.py
│  ├─ m3fd_dataset.py
│  └─ builder.py
│
├─ models/
│  ├─ srcnn.py
│  ├─ fsrcnn.py
│  ├─ edsr.py
│  ├─ rcan.py
│  ├─ registry.py
│  ├─ builder.py
│  └─ ldynsr/
│     ├─ __init__.py
│     ├─ common.py
│     ├─ dam.py
│     ├─ pa.py
│     ├─ bta.py
│     ├─ dyna.py
│     ├─ frm.py
│     └─ ldynsr.py
│
├─ engine/
│  ├─ trainer.py
│  ├─ evaluator.py
│  └─ inferencer.py
│
├─ utils/
│  ├─ checkpoint.py
│  ├─ logger.py
│  ├─ metrics.py
│  ├─ seed.py
│  ├─ visualize.py
│  ├─ profile.py
│  └─ misc.py
│
├─ outputs/
├─ train.py
├─ test.py
├─ infer.py
├─ compare_results.py
├─ main.py
└─ README.md
```

说明：
`registry.py / builder.py / engine/` 是本次重构重点，用来把原先可能散落在脚本里的逻辑统一收拢。

---

## 5. 必须完成的改造任务

## 5.1 模型注册机制重构

### 任务

新增统一模型注册表与构建接口。

### 要求

* 建立 `models/registry.py`
* 建立 `models/builder.py`
* 所有模型统一通过 `build_model(name, **kwargs)` 构建
* 注册表至少包含：

  * `srcnn`
  * `fsrcnn`
  * `edsr`
  * `rcan`
  * `ldynsr`

### 目的

避免 `train.py / test.py / infer.py` 中散落大量模型判断逻辑。

---

## 5.2 抽离训练/评估/推理执行层

### 任务

将训练、测试、推理中的通用执行逻辑迁移到 `engine/` 目录。

### 目标拆分

* `engine/trainer.py`：训练与验证循环
* `engine/evaluator.py`：测试评估、指标汇总、可视化调用
* `engine/inferencer.py`：单图/目录推理与结果保存

### 要求

`train.py / test.py / infer.py` 重构后只保留：

* 参数解析
* 配置加载
* dataloader 构建
* model 构建
* engine 调用

不要在这三个入口文件中写 LDynSR 结构细节。

---

## 5.3 接入 LDynSR 模型实现

### 任务

新增 `models/ldynsr/` 目录，实现 LDynSR。

### 必需文件

* `dam.py`
* `pa.py`
* `bta.py`
* `dyna.py`
* `frm.py`
* `ldynsr.py`

### 结构要求

LDynSR 顶层应符合论文主干：

1. 输入 LR 图像
2. 经过一个 `3×3 Conv` 进行浅层特征提取
3. 进入 DFEB：若干 `DynA` 堆叠 + 一个 `3×3 Conv`
4. 进入 FRM 进行特征重建和上采样
5. 输入图像同时经过 bicubic 上采样
6. 两路相加得到最终 SR 输出。

### DynA 要求

每个 DynA 至少包含：

* 输入侧 `1×1 Conv`
* 三条分支：

  * PA 分支
  * BTA 分支
  * NOA 分支（非注意力分支，可视为 `3×3 Conv`）
* DAM：给三条分支生成动态权重
* 分支加权融合后再过 `1×1 Conv`
* 与输入做残差相加。

### DAM 要求

DAM 应遵循论文逻辑：

* GAP
* FC
* ReLU
* FC
* Softmax
* 输出三条分支权重，三者之和为 1。

### PA 要求

PA 应体现像素注意力与局部特征增强，包含：

* `1×1 Conv + Sigmoid` 生成像素权重
* `3×3 Conv` 提取特征
* 点乘
* 残差连接
* 再经 `3×3 Conv` 精炼。

### BTA 要求

BTA 应体现亮度-纹理双分支思想：

* 亮度分支：基于全局平均池化提取亮度信息
* 纹理分支：基于卷积提取纹理信息
* 两分支融合生成注意图
* 对输入特征进行调制。

注意：论文对 BTA 的张量细节没有写得完全严密，工程实现允许做**合理补全**，但需要在代码注释或 README 中说明“此处为复现补全实现”。

### FRM 要求

FRM 应支持：

* `x2`：单级 `TransposedConv -> PA -> Conv`
* `x4`：两级逐步上采样。

---

## 5.4 数据与退化流程保持平台统一

### 任务

不要为 LDynSR 新建独立 dataset。
继续使用现有平台的 Dataset、切分和预处理体系。README 已说明现有项目以统一切分和统一流程比较多个模型。

### 要求

* `datasets/m3fd_dataset.py` 继续作为主数据集入口
* 如有必要，可增加 `datasets/builder.py`
* `degrade.py` 不重写，但应参数化，便于未来扩展

### 建议参数化项

* 下采样方式
* 噪声类型
* 噪声强度
* 其他可能的退化参数

### 说明

当前任务目标是“统一平台比较”，不是“严格复刻 LDynSR 原论文的数据协议”。论文原始实验采用 FLIR / Thermal101，并使用 bicubic + AWGN 的退化过程。
本项目中可先保持 M3FD 统一平台协议不变。

---

## 5.5 checkpoint 元数据增强

### 任务

确保保存的 checkpoint 能完整恢复 LDynSR 模型。

### 必须保存的元数据

至少包括：

* `model_name`
* `scale`
* 与模型结构相关的必要参数，例如：

  * `feat_channels`
  * `num_dyna`
  * `dam_reduction`

### 原因

README 说明当前 `infer.py` 支持从 checkpoint 自动推断模型与倍率。LDynSR 接入后也必须兼容这一机制。

---

## 5.6 全模型测试与对比流程兼容 LDynSR

### 任务

让 LDynSR 自动进入现有全模型对比体系。

### 要求

* `test.py --all_models` 时自动包含 `ldynsr`
* 输出目录与命名规则保持统一
* 测试报告格式保持统一
* `compare_results.py` 可以自动识别并纳入 LDynSR 结果

### 原因

当前项目已经支持“全部模型评估”和“汇总对比导出”，这是平台的核心价值之一。

---

## 6. 配置化要求

Agent 应尽量把模型相关超参收进配置文件，而不是硬编码在脚本中。

### 至少为 LDynSR 配置的参数

* `in_channels`
* `out_channels`
* `feat_channels`
* `num_dyna`
* `dam_reduction`
* `scale`

### 建议位置

* `configs/model/ldynsr.yaml`
* `configs/train/ldynsr.yaml`

### 原因

论文给出了 LDynSR 的模块设计，但并未公开所有具体超参，因此这些值在项目中应视为**实现配置项**，便于后续调参与消融。

---

## 7. 不要做的事情

Agent 不应执行以下操作：

### 7.1 不要新建独立 LDynSR 项目

本任务目标是**在现有 InfraredSR 平台中接入 LDynSR**。

### 7.2 不要破坏旧模型现有行为

重构后旧模型应仍能正常训练、测试、推理。

### 7.3 不要为 LDynSR 私自引入专属数据链路

除非后续明确要求做论文协议复现，否则当前版本中 LDynSR 应继续使用平台统一数据流程。

### 7.4 不要在 `train.py / test.py / infer.py` 中写大量 LDynSR 专用逻辑

LDynSR 的结构细节必须封装在 `models/ldynsr/` 中。

---

## 8. 建议执行顺序

## Phase 1：轻量工程重构

1. 新建 `models/registry.py`
2. 新建 `models/builder.py`
3. 抽离 `engine/trainer.py`
4. 抽离 `engine/evaluator.py`
5. 抽离 `engine/inferencer.py`
6. 验证旧模型仍能正常运行

## Phase 2：接入 LDynSR

1. 新建 `models/ldynsr/`
2. 实现 `DAM / PA / BTA / DynA / FRM / LDynSR`
3. 在注册表中加入 `ldynsr`
4. 为 LDynSR 增加配置文件
5. 扩展 checkpoint 元数据

## Phase 3：平台集成

1. 让 `train.py` 支持 `--model ldynsr`
2. 让 `test.py --all_models` 自动包含 `ldynsr`
3. 让 `infer.py` 支持从 LDynSR checkpoint 自动恢复模型
4. 让 `compare_results.py` 能汇总 LDynSR 结果

## Phase 4：验证

1. 旧模型回归测试
2. LDynSR 单模型训练/测试验证
3. LDynSR 与 `SRCNN / FSRCNN / EDSR / RCAN` 的统一平台横向比较

---

## 9. 任务完成判定标准

当满足以下条件时，视为本次重构完成：

1. 项目仍然保持“统一平台比较多个 SR 模型”的定位不变。
2. 旧模型 `SRCNN / FSRCNN / EDSR / RCAN` 可正常训练、测试、推理。
3. 新模型 `LDynSR` 已成功接入，并能在统一流程下训练、测试、推理。
4. `test.py --all_models` 能自动包含 `ldynsr`
5. `compare_results.py` 能输出包含 LDynSR 的统一对比结果
6. checkpoint 能完整恢复 LDynSR
7. 代码结构比当前更清晰，新增模型时不需要到多个脚本里打补丁

---