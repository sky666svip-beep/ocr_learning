# 任务规划：OCR V1（字符切分 + 单字符分类）及 Streamlit 展示

## 目标

从0开始构建基于传统图像处理（字符切分）和基础 CNN（单字符分类）的数字 OCR 模型，并开发 Streamlit Web 界面以展示处理过程、提取文字和可视化检测框。

## 当前阶段

阶段 2

## 阶段划分

### 阶段 1：需求澄清与基础架构研讨 (Phase 1: Requirements & Discovery)

- [x] 理解用户意图与核心痛点
- [x] 触发 Socratic Gate 提出战略性问题
- [x] 确定数据集与深度学习框架偏好
- [x] 创建并在 findings.md 中记录初始发现
- **Status:** complete

### 阶段 2：技术方案与框架搭建 (Phase 2: Planning & Structure)

- [x] 定义完整的技术栈
- [x] 划分训练代码、图像处理与 UI 代码的目录结构
- [x] 文档化技术选型理由
- **Status:** complete

### 阶段 3：模型训练与核心算法实现 (Phase 3: Implementation)

- [x] 获取预处理数字数据集
- [x] 构建并训练 CNN 模型（例如 LeNet 或 小型 ResNet）
- [x] 实现传统图像处理方案（垂直投影、连通域分析）
- **Status:** complete

### 阶段 4：Streamlit 集成与可视化 (Phase 4: Web UI & Integration)

- [x] 构建 Streamlit 用户界面（支持图片上传与结果展示）
- [x] 接入切分与分类推理流水线
- [x] 实时渲染 bbox 框和文本，可选展示处理中间态
- **Status:** complete

### 阶段 5：测试与交付 (Phase 5: Testing & Delivery)

- [x] 测试切割极差、文本倾斜等边界情况
- [x] 优化并交付最终代码
- **Status:** complete

### 阶段 6: V2 CRNN + CTC 序列建模架构升级 (Phase 6: V2 CRNN Implementation)

- [x] 构建用于 V2 训练的 `SyntheticTextDataset`
- [x] 构建 `CRNN` 模型结构与 CTC 损失处理
- [x] 编写并执行 `train_crnn.py`
- [x] 改造 `app.py`：V1/V2 切换与 CTC 热力图可视化
- **Status:** complete

### 阶段 7: V3 Seq2Seq + Attention 隐式语言模型升级 (Phase 7: V3 Seq2Seq + Attn)

- [x] 构建包含 `<SOS>`, `<EOS>`, `<PAD>` 字典特性的训练集与字典
- [x] 构建 `Seq2Seq` 注意力架构（CNN Encoder + Attention Decoder）
- [x] 编写并执行 `train_v3.py`，支持 Teacher Forcing 和 CrossEntropyLoss
- [x] 开发 `pages/v3_seq2seq_attn.py`，接入全系统路由，实现 Attention Map 可视化
- **Status:** complete

### 阶段 8: V4 CNN+Transformer 混合架构与双分支联合训练 (Phase 8: V4 Joint Training)

- [x] 构建带位置编码 (Positional Encoding) 的 Transformer Encoder
- [x] 构建包含 CTC 分支与 Transformer Decoder (或纯 Attn 分支) 的双头网络
- [x] 编写并执行 `train_v4_joint.py` 进行 CTC+CE Loss 的动态联合优化
- [x] 开发 `pages/v4_transformer_joint.py` 实现自注意力与双端对齐可视化
- **Status:** complete

## 关键问题

1. [等待 Socratic Gate 用户反馈]
2. [等待 Socratic Gate 用户反馈]

## 已作出的决策

| 决策                          | 理由                                      |
| ----------------------------- | ----------------------------------------- |
| 遵循 planning-with-files 规范 | 用户规则指定，确保项目长期的执行记忆      |
| 传统切分结合基础 CNN          | 揭示 OCR 原理，作为端到端深度学习的参照物 |

## 错误记录

| 错误 | 尝试次数 | 解决方案 |
| ---- | -------- | -------- |
|      | 1        |          |

## 笔记

- 遵循 Socratic Gate 原则，进入后续阶段前必须解答核心痛点。
- 所有注释和变量均使用中文。
- 遵循第一性原理(First Principles Thinking)与KISS原则构建极简稳定的代码库。
