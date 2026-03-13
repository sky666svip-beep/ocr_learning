# 🧭 OCR 架构演进教学沙箱 (OCR Learning Sandbox)

这是一个端到端的深度学习交互式沙箱项目，旨在通过**第一性原理**从零构建并剖析光学字符识别（OCR）技术的历史演进。本项目基于 PyTorch 和 Streamlit，将极度抽象的网络张量变化、序列对齐、注意力机制（Attention）转化为了浏览器中可滑动、可推演的直观热力图。

通过五个极具代表性的架构迭代（V1 ➡️ V5），本项目展示了 OCR 技术是如何从“传统切分”一步步走向“视觉与语言大一统”的。

## ✨ 核心特性

- **🚀 架构全景演化**：包含 5 代 OCR 核心架构（CNN、CRNN、Seq2Seq、Transformer Joint、ViT）。
- **📊 极客级可视化**：实时解剖传统垂直投影切割柱状图、CTC 对齐热力矩阵、注意力机制游走光晕（Attention Map）、以及 Transformer 的自相关（Self-Attention）全景矩阵。
- **⚙️ 浏览器内实时训练**：内置 `SyntheticTextDataset` 动态数据生成器，支持在侧边栏直接调节超参数（Epoch, LR, Batch Size）并实时重训练模型观察收敛过程。
- **🏭 SOTA 工业级挂载**：V5 版本支持一键挂载 HuggingFace 预训练大模型（如微软 TrOCR），体验工业级中英文手写/印刷体识别与跨模态对齐魔法。
- **🛡️ 统一抗毁滤网**：内置处理 Domain Shift 的图像预处理流水线（灰度同化、中值降噪、CLAHE 直方图均衡），自动抵抗外部真实图的噪点干扰。

## 🗺️ 五代架构全景 (Evolutionary Stages)

沙箱左侧导航栏提供了穿越五个时代的入口：

### 1. V1: 经典切分架构 (传统 CV + CNN)
- **原理**：图像预处理 $\rightarrow$ 投影柱状图寻谷底切分 $\rightarrow$ CNN 孤立分类。
- **教学目的**：展现传统图像处理如何被“字体粘连”、“倾斜”与“噪声”轻易击败，暴露出“先切再认”的致命木桶效应。

### 2. V2: 端到端序列建模 (CRNN + CTC)
- **原理**：不对称池化 CNN 提取时序切片 $\rightarrow$ Bi-LSTM 赋予前后文关联 $\rightarrow$ CTC 动态规划求解序列。
- **教学目的**：展示 CTC 损失函数如何利用 `Blank`（空白符）极其优雅地解决字符粘连问题，彻底消灭物理切分步骤。

### 3. V3: 隐式语言模型 (Seq2Seq + Attention)
- **原理**：CNN 空间展开 $\rightarrow$ Bahdanau Attention 权重映射 $\rightarrow$ RNN Decoder 自回归。
- **教学目的**：打破 CTC“字符独立假设”的弱点，让模型学会“结合语境猜字”。你可以拖动滑块，直观看到模型在输出字符时，目光是如何在原图上游走的。

### 4. V4: 混合强架构 (CNN + Transformer Encoder + CTC/Attn)
- **原理**：引入 Transformer Encoder 替代 RNN 解决串行瓶颈，并采用 `0.5*CTC + 0.5*CrossEntropy` 的双头联合训练。
- **教学目的**：展现工程界的极致妥协与优化——用 CTC 分支作“保底安全网”强制硬对齐，防止纯 Attention 分支在长图上的“视线漂移（Alignment Drift）”。

### 5. V5: 双模式极境 (纯 Vision Transformer)
- **原理**：抛弃 CNN！利用 Patch Embedding 将图像碎为 256 块，纯粹依赖 ViT Encoder (Self-Attention) 和 Transformer Decoder (Cross-Attention) 翻译。
- **教学目的**：展示多模态大一统时代的 SOTA 思路。支持手搓微型 ViT 洞析底层原理，也支持挂载 `microsoft/trocr-small-printed` 窥探亿级参数模型的绝对实力。

## 🛠️ 项目结构

```text
ocr_learning/
├── app.py                      # Streamlit 主入口，统一路由控制
├── pages/                      # 各代架构对应的交互 UI 页面
│   ├── v1_traditional.py
│   ├── v2_crnn_ctc.py
│   ├── v3_seq2seq_attn.py
│   ├── v4_transformer_joint.py
│   └── v5_vit_ocr.py
├── models/                     # 纯净的 PyTorch 网络架构定义
│   ├── cnn_classifier.py
│   ├── crnn_ctc.py
│   ├── seq2seq_attn.py
│   ├── v4_transformer_joint.py
│   └── v5_vit_ocr.py
├── utils/                      # 基础组件
│   ├── data_generator.py       # 合成数据集生成器 (定长/变长/格式化语境)
│   └── image_processing.py     # 统一图像清洗防线与 CV 操作
├── docs/                       # 各版本的深度架构复盘与原理探究文档
├── train_*.py                  # 各版本的独立训练脚本
└── ...

```

## 🚀 安装与运行

**1. 克隆项目与安装依赖**

```bash
git clone [https://github.com/your-username/ocr_learning.git](https://github.com/your-username/ocr_learning.git)
cd ocr_learning
pip install torch torchvision numpy opencv-python pillow matplotlib seaborn streamlit transformers

```

**2. 启动沙箱**

```bash
streamlit run app.py

```

> 系统将自动在浏览器中打开 Web 交互界面。建议拥有 CUDA 环境的机器运行以获得最佳的实时训练体验（CPU 亦可流畅进行推理演示）。

## 📖 深入阅读

我们在项目中内置了详尽的复盘记录，深入解释了每一代代码背后的设计哲学与妥协，强烈建议阅读 `docs/` 目录下的复盘文档：

* [V1 架构演进复盘记录](https://www.google.com/search?q=docs/v1_traditional_architecture_review.md)
* [V2 CRNN_CTC 架构解析](https://www.google.com/search?q=docs/v2_crnn_ctc_architecture_review.md)
* [V3 Seq2Seq_Attention 的凝视](https://www.google.com/search?q=docs/v3_seq2seq_attn_architecture_review.md)
* [V4 联合架构双面真理](https://www.google.com/search?q=docs/v4_transformer_joint_architecture_review.md)
* [V5 视觉语言大一统 ViT](https://www.google.com/search?q=docs/v5_vit_ocr_architecture_review.md)

---

*“这是所有端到端 OCR 方案的对照基石，理解序列建模核心价值的必经之路。”*

```

```
