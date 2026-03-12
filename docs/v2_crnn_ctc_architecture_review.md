# OCR 架构演进实录：从传统切分到 CTC 端到端建模

## 📌 背景与设计哲学 (The Whys)

在计算机视觉的历史长河中，文字识别（OCR）经历了一个从“先切再认”到“端到端全读”的范式转移。
我们的项目依循**第一性原理**从零构建了这个演进。
在 V1 版本中，我们实现了**“经典传统 CV 切片 + 孤立 CNN 识别”**模型。虽然逻辑直观，但我们在可视化工具（投影柱状图）前残酷地确认了一个事实：一旦字符粘连、断裂或受到光照干扰，投影谷底寻找法将完全失灵，导致后续极强甚至完美的 CNN CNN单点分类也于事无补。因为“垃圾进，垃圾出 (GIGO)”。

因此，V2 架构引入了 **CRNN (Convolutional Recurrent Neural Network) 伴随 CTC (Connectionist Temporal Classification)** 损失算法。

## 🧩 V2 CRNN 架构拆解 (The Whats)

在 V2 的重构中，有三个核心文件发生了解耦协作：

### 1. 数据对撞变迁 (`utils/data_generator.py`)

端到端网络不接受单字，它需要阅读整行上下文。因此我们创建了 `SyntheticTextDataset`。

- **改动核心**：为了批量送入 PyTorch CNN 并维持后续时序步伐对齐的整齐度，我们放弃了任性大小，强制网络输入分辨率必须被 resize/padded 到 **256宽 x 32高**。
- **Padding 策略**：字符优先靠左对齐，剩余的右侧空间用全白背景填充。这能确保 LSTM 从左向右扫描时首先建立文字特征，空白留给最后消化。

### 2. 时序维度强留 (`models/crnn_ctc.py`)

在普通的 ResNet/VGG 中，卷积池化会将高度(H)和宽度(W)无差别地向 $1\times1$ 压缩。但为了将**宽度视作时间步 (Time Step)**，我们在特征提取部分采用了**不对称池化**。

- **改动核心**：我们在 `CRNN` 定义里，将后面的 `MaxPool2d` kernel size 设置成 `(2, 1)`（即仅对高度除以 2，而放过跨度特征）。最终配合 `AdaptiveAvgPool2d((1, 32))` 将特征压缩拉平至 `H=1, W(T)=32`。
- 接驳到了一层 **Bidirectional LSTM**。网络不再是看“一块像素”，而是能够回忆其左右的关联特征。

### 3. CTC 与概率热图展示 (`pages/v2_crnn_ctc.py` & `train_crnn.py`)

双向 LSTM 吐出的是针对 `T=32` 每一步上 11 个分类的特征概率预测：包含数字 0-9 以及最核心的 `Blank` (空白符)。

- **训练层**：在 `train_crnn.py` 中，使用 `torch.nn.CTCLoss` 承接网络输出 $32 \times \text{Batch} \times 11$。它使用前向后向动态规划，穷举所有去重并移除 `Blank` 后能对标上目标的可能路径汇总概率，极大宽慰了样本人工标注时序位置的巨大开销（俗称 Alignment Free）。
- **展现层重构**：为了在 Streamlit UI 深刻感受 CTC 机制的“防连败”特性，我们放弃了传统的原图直接画框。我们抓取全图序列推理矩阵。
  - 创建了两套图层 Mask 以叠加 Seaborn Heatmap。
  - 第一行，孤立取出 `Blank (类别0)`，用**极强的暖红热源色**渲染。让你直观看到当网络在经过密集粘连段后，往往会极其确信且剧烈地喷发一次红色 Blank 信号来打断重复叠加。
  - 剩余行，采用冷蓝色调渲染实体类别。

## 💡 技术复盘与未来延展

在这个跨越三个基建节点的架构升级中，系统演化为：
输入 $256 \times 32$ 纯数字块 $\xrightarrow{CNN+Pool(2,1)}$ 提取时间特征切面 $\xrightarrow{Bi-LSTM}$ 赋予特征前后文联系 $\xrightarrow{CTC Loss}$ 通过波形去重概率预测最终序列。

目前，整个可视化已经重构为**基于 SelectBox 动态路由的多页面架构 (`app.py` 整合 `pages/v...`)**，不仅实现了新旧版本在逻辑与控制面板的完全隔离，也为未来的 V3 甚至引入 Attention/Transformer、真实世界开源场景图支持留下了坚实规整的沙箱基础。
