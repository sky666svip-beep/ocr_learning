# 架构演进复盘记录：V5 (双模式 Vision Transformer - ViT/TrOCR)

## 1. 演化终局：视觉与语言的彻底大一统
从 V1 的传统 CV 物理刀刃切割，到 V2 让模型依靠 CTC 列概率序列式蒙猜，再到 V3/V4 用 Attention 做空间像素级矩阵映射与双重损失挂载。CNN 提取加上序列回归，曾是不可撼动的准则。

**然而 V5 宣告了这一切的终结：CNN 被彻底拔除。**
ViT (Vision Transformer) 将一切碾碎重组为同质化的 “Token”，无论是图片（Patch Token），还是文字（Text Token），在这个纪元被统一到了同一个自回归和注意力矩阵的维度上来处理。

本页面特别采用了**【手搓微型 ViT 教学模式】**与**【HuggingFace 工业级 TrOCR 直接挂载】**双轨分离结构，只为一次性揭开顶级工业 SOTA 是如何运作的。

## 2. 核心架构解剖

我们所搭建的极简 ViT 模型，与微软的 `TrOCR (DeiT + RoBERTa)` 结构如出一辙：

```mermaid
graph TD
    subgraph 视觉理解 (ViT Encoder)
        A[Image] -->|4x8 切成 256 块| B[Patch Embedding]
        B -->|线性映射 + Learnable Pose| C(Patch Token 序列)
        C -->|N 层 Self-Attention| D[Encoder Output Memory]
    end

    subgraph 语言输出 (Transformer Decoder)
        E[当前生成的文本 Token] -->|Embedding + Pose| F(Text Token 序列)
        F -->|Masked 防止看未来| G[Self-Attention]
        G --> H{Cross-Attention}
        D -.->|Memory 注入| H
        H --> I[生成下一个预测字符]
    end
```

### 部件 1：抛弃 CNN 的 Patch Embedding
以前，我们千方百计用 `Conv2d` 和 `MaxPool` 保留下边缘和纹理特征。
现在，我们直接用一个等同于 Patch Size (`4x8`) 的单层 Conv 无步长卷积，直接把图硬切成 256 块长方形碎片。每一个碎片就是一行 64 维度的向量（Token）。

### 部件 2：ViT Encoder 的自相关 (Self-Attention)
这些毫无情感的 256 个图像碎片，进入 Transformer 后开始通过 `Self-Attention` 互相建立羁绊。这也就是为什么在**左侧的交互面板**里，你能看到一个 256x256 的热力图：每一块碎片都在“注意”其他的碎片，试图拼凑成一张有意义的图片网。

### 部件 3：译码器的跨界凝视 (Cross-Attention)
当轮到文本生成时，Decoder 不仅关注自己说了什么（文本到文本的 Masked Self-Attention），**更重要的是，每一个文本 Token 都拥有一次向那 256 个图像 Patch 发起 Query（提问）的权力**。
这就形成了图表下半区最壮观的一幕：随着文本吐出 `A`，你能清晰地看到 `Cross-Attention` 热力图的红色焦点死死锁在了原图左上角代表 A 形状的几个 Patch 块上！

## 3. 工业 SOTA 挂载 (HuggingFace TrOCR)
为了展示真实力学，应用内提供了一键挂载 `microsoft/trocr...` 等仓库的功能。
工业级的 SOTA 所采用的机制完全没有跳脱上述三个部件的框体。但它们的参数量是数千万甚至十亿级（如 62M/330M 参数层）。借由预训练时见过的上千万张带有真实噪点的文档，它能完美处理极度扭曲、阴影、手写潦草等人类也难以分辨的字体。

你可以通过拖拽其内置的 Cross-Attention Layer 取样点，真切地看到大模型在输出中文、英文时，视线在庞大原图上扫描游走的骇人准确度。

## 4. 结语
OCR 算法从人工特征计算，演变为了纯粹的 Token 间概率转移。这是一个不可逆转的深度学习大一统浪潮。希望通过这个跨越 5 代沙盒演练台，你能深度理解每一个矩阵后方的逻辑与妥协。
