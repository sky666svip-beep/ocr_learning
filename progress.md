# 进度日志

## 会话: 2026-03-08

### 阶段 1：需求澄清与基础架构研讨

- **Status:** complete
- **Started:** 2026-03-08 16:33
- 采取的行动:
  - 阅读并理解了 planning-with-files 技能的工作流模板。
  - 在 d:\Projects\ocr_learning 初始化 task_plan.md, findings.md, progress.md。
  - 基于 Socratic Gate (TIER 0 规则) 拦截直接开发意图，向用户抛出了核心架构问题。
  - 接收到反馈，记录数据集、边界条件、PyTorch 偏好及 UI 排布逻辑。
- 创建/修改的文件:
  - task_plan.md (created)
  - findings.md (created)
  - progress.md (created)

### 晚间追溯阶段：附加训练控制台 (Training Dashboard)

- **Status:** complete
- **Started:** 2026-03-08 16:55
- 采取的行动:
  - 为了回应用户“前端直接控制/监控训练过程”的诉求：
  - 重构 `train.py` 将死循环的打点修改为支持 `progress` 和 `metric` 两种回调。
  - 在 `app.py` 增加 sidebar 支持调节 `Epoch`, `Learning Rate`。
  - 在前端直接集成 `matplotlib` 即时回执 Loss & Val_Accuracy 图并替换全局单例缓存。
- 创建/修改的文件:
  - train.py (updated)
  - app.py (updated)
  - progress.md (updated)
  - brain/task.md (created)
  - brain/implementation_plan.md (created)

### 阶段 2：技术方案与框架搭建

- **Status:** complete
- **Started:** 2026-03-08 16:39
- 采取的行动:
  - 建立 implementation_plan.md 和 agent task.md Artifact
  - 用户审批 Synthetic dataset 实施落地，关闭技术选项阶段。
- 创建/修改的文件:
  - task_plan.md (updated)
  - findings.md (updated)
  - progress.md (updated)
  - brain/task.md (created)
  - brain/implementation_plan.md (created)

### 阶段 3 & 4：核心模型流水线及 WebUI 实施

- **Status:** complete
- **Started:** 2026-03-08 16:40
- 采取的行动:
  - 编写了 utils/data_generator.py (合成 0-9 扭曲图像作为特征输入)
  - 编写了 models/cnn_classifier.py (定义了标准的轻度卷积网络负责单字符辨识)
  - 编写了 utils/image_processing.py (包含了灰度化、基于 Otsu 的二值化、Morphological 操作及基于波流的垂直投影切词与 Bounding box 生成逻辑)
  - 编写运行了 train.py 生成了 best_model.pth (在纯控制集上跑分达到 100%)
  - 编写了 app.py Streamlit 脚手架用以承接左右两侧的多态化视图输出以体现“投影切割困局”的核心痛点教育意义。
- 产生的文件:
  - utils/data_generator.py
  - models/cnn_classifier.py
  - utils/image_processing.py
  - train.py
  - models/best_model.pth
  - app.py
  - test_multi.png 等验证材料

### 阶段 5：验证与交付

- **Status:** complete
- **Started:** 2026-03-08 16:44
- 采取的行动:
  - 检查所有的模型训练指标和 UI 组件依赖。本地启动后台实例以备浏览器测试。
  - 用户拦截自主 Web UI 确认流程，表明对前端信赖放行。
  - 生成 `walkthrough.md`。
- 新增文件:
  - walkthrough.md

### 最终阶段：知识沉淀与架构复盘 (Documentation)

- **Status:** complete
- **Started:** 2026-03-08 17:28
- 采取的行动:
  - 响应“跨越多文件重构后产出复盘记录”的需求。
  - 创建了 `docs/` 目录存放技术文档。
  - 编写了 `v2_crnn_ctc_architecture_review.md`，深度拆分总结了 V2 在数据 Padding 对齐、CRNN 不对称降维机制、及 CTC Blank 热力学掩模层渲染三大核心步骤的原理。
- 产生的文件:
  - docs/v2_crnn_ctc_architecture_review.md

### 新生阶段：V3 隐式语言模型与自回归 Attention (Phase 7)

- **Status:** complete
- **Started:** 2026-03-09 10:10
- 采取的行动:
  - 用户选定使用轻量自定义 CNN 作为 Encoder 骨干并启用 Streamlit Slider 进行步进游走渲染。
  - **数据集**: 开发 `SemanticTextDataset`，构建包含 `0-9` 配合 `: - A B C` 等特殊字典的“准格式化”序列（日期、车牌段）增强语言模型特征。
  - **模型核心**: 开发 `Seq2Seq+Attention` (`models/seq2seq_attn.py`)，使用 Bahdanau 机制让 LSTM 能够查询并按概率权重获取 CNN 提供的像素局部矩阵信息。
  - **训练**: 编写包含 Teacher Forcing 退火策略与 CrossEntropy 的 `train_v3.py`，后台跑测截除收敛正常。
  - **可视化构建**: 创建 `v3_seq2seq_attn.py`，用 OpenCV 的 `applyColorMap` 以及 Alpha Blending 将 `4x32` 的网络视野覆写到 `256x32` 原图切片上。

- 产生/修改的文件:
  - utils/data_generator.py (added SemanticTextDataset)
  - models/seq2seq_attn.py (created)
  - train_v3.py (created)
  - pages/v3_seq2seq_attn.py (created)
  - app.py (updated routes)

### 全局优化阶段：统一图像预处理滤网 (Domain Shift 缓解)

- **Status:** complete
- **Started:** 2026-03-09 10:31
- 采取的行动:
  - 为了应对外界图片噪点、模糊引起的 CNN 特征崩盘，开发统一清洗管线。
  - 在 `utils/image_processing.py` 建立 `unified_enhance_image` 函数。
  - 编排了 `cvtColor` $\rightarrow$ `medianBlur` (中值除噪点) $\rightarrow$ `CLAHE` (限制对比度自适应直方图均衡，拔高原图对比度)。
  - 对于 `pages/v1_...`, `v2_...`, `v3_...` 所有的图片上传入口 (Streamlit 推理侧) 执行了底层滤网劫持映射。用户上传的“脏图”会在此变清澈，随后进入后续步骤。

### 并联深水区：V4 CNN+Transformer 混合与联合训练 (Phase 8)

- **Status:** complete
- **Started:** 2026-03-09 10:48
- 采取的行动:
  - 为了根除 Attention 分支长文漂移以及 LSTM 的串行低效，采纳并行架构。
  - **模型核心 (`models/v4_transformer_joint.py`)**: 编写 `V4JointModel`，装配 Positional Encoding + TransformerEncoder 作为全局时间特征提取器。用此替换了 V2 和 V3 陈旧的双向 LSTM。在此之上架设两颗头：直接输出 CTC Loss 的 `ctc_head` 与沿用 V3 的自回归 `AttentionDecoder`。
  - **训练 (`train_v4_joint.py`)**: 构建了双损失结合逻辑：$\mathcal{L}_{total} = 0.5 \mathcal{L}_{CTC} + 0.5 \mathcal{L}_{CE\_Attn}$，并在训练时修复了 Validation 下目标集退化丢步导致的 Shape mismatch BUG。
  - **全景对流端 (`pages/v4_transformer_joint.py`)**: 完成了三位一体可视化。左列监控 CTC 贪心容差防漂移墙、右侧观察 Attention 视线聚集点、并在 Expand 控制面里渲染出 Transformer 的三十二维自相映射热力图。
  - **理论落成 (`docs/v4_transformer_joint_architecture_review.md`)**: 将这一过渡演进期与“大一统”架构思想固化为了独立复盘文件。

### 终极演进：V5 双模式 Vision Transformer (Phase 9)

- **Status:** complete
- **Started:** 2026-03-11 09:10
- 采取的行动:
  - 为了彻底证实“大一统”架构和剔除 CNN 带来的优势，构建完全无卷积的端到端框架。
  - **自研教学模型 (`models/v5_vit_ocr.py`)**: 编写了基于原生 `nn.MultiheadAttention` 及 `PatchEmbedding` 的微型 ViT。保留了探出 `Self-Attention` 和 `Cross-Attention` 的钩子 (Hook)。
  - **训练脚手架 (`train_v5_vit.py`)**: 针对微型 ViT 设计的自回归生成训练网。纯粹基于 CrossEntropy 且通过 Teacher Forcing 对源序列进行对齐退火，不再依赖 CTC Loss。
  - **SOTA 接入与全景 UI (`pages/v5_vit_ocr.py`)**: 构建了双模式控制台。左侧可使用自研沙盒调参训练，查探 Patch 间的原生羁绊；也可开启「预训练 TrOCR」模式，动态挂载 HuggingFace 模型 (采用 `microsoft/trocr-small-printed` 英文SOTA模型与 `ZihCiLin/trocr-traditional-chinese-baseline` 中文SOTA模型)，进行惊人的零样本外生源图推断与注意力的抽取剖析。
  - **架构复盘 (`docs/v5_vit_ocr_architecture_review.md`)**: 对全工程 V1->V5 系列做了最终结题性总结论断。

### 全局优化阶段：V1-V5 统一训练参数超调 (Phase 9 Integration)

- **Status:** complete
- **Started:** 2026-03-12 15:10
- 采取的行动:
  - 响应用户提议，在前端页面为 V1 至 V5 模型暴露统一基准的超参控制器。
  - 开放了极度精细的 `学习率 (LR)` 步进器（支持小数位精度）、`训练轮次 (Epochs)` 滑动条（最大延展至 100 轮），以及 `Batch Size` 的即时调控。
  - 移除了散落各处的由于历史迭代遗留的硬编码默认值（重点包括 V5 的 epochs 25 写死），使整个沙箱形成彻底的模型演化闭环并完全自由化。

## 测试结果

| 测试               | 输入                   | 预期                    | 实际          | 状态 |
| ------------------ | ---------------------- | ----------------------- | ------------- | ---- |
| 数据生成器测试     | 执行 data_generator.py | 生产 PIL Image 和 Label | 通过          | ✅   |
| 构建微训练         | epoch=3                | Validation accuracy>90% | 100% 极速收敛 | ✅   |
| Streamlit 切割渲染 | process_image 组装     | 成功抽出投影直方图数据  | 通过          | ✅   |
| V2 架构演进重写    | 拔除核心注入           | 达成多页面无缝切换      | 通过          | ✅   |
| V3 模型验证与整合  | train_v3.py 后台训练   | Loss呈现下降梯次退火    | 通过          | ✅   |
| V4 宏架构双重收敛  | train_v4_joint.py 执行 | CE 与 CTC 能够携同收缩  | 并行计算通过  | ✅   |
| V5 ViT双模式架构   | train_v5_vit.py 执行   | 抛弃 CNN 下的纯自稳收敛 | 纯注意力系通过| ✅   |

## 错误日志

| 时间戳 | 错误                  | 尝试次数 | 解决方案                                                                                 |
| ------ | --------------------- | -------- | ---------------------------------------------------------------------------------------- |
| 16:42  | ModuleNotFoundError   | 1        | 缺失 torchvision/cv2。引入 pip 安装所有框架依赖并重置代码。                              |
| 17:39  | CNN 与 V2 空距切边    | 1        | 追加 Padding `margin` 以及 `cv2.findNonZero` 防溢出剪裁。                                |
| 11:20  | V4 CTC Batch Mismatch | 1        | 验证循环 `trg=None` 退化长测 20 步，引出交叉矩阵碰撞，通过恢复 Target 时序长度修补齐长。 |
| 10:56  | V4 UI ImportError     | 1        | 因梳理分流逻辑丢失 CTC 解码词库映射接口 `id2char_ctc`，已补充定义代码修复了 UI 载入。    |

## 5-Question Reboot Check

| 问题                 | 答案                                                                                                              |
| -------------------- | ----------------------------------------------------------------------------------------------------------------- |
| Where am I?          | V4 Phase Completed (混合架构与双目标网络)                                                                         |
| Where am I going?    | 在完成传统到深度学习、局部到全局的横跨之后，将目标瞄准极致归一的终极版：V5 Vision Transformer。                   |
| What's the goal?     | 展现打破 LSTM 后并行算力的强大，以及用 CTC 的强对齐辅路挽救自回归主路发生幻视的工程巧思。                         |
| What have I learned? | 联合损失 `0.5*CTC + 0.5*CrossEntropy` 不仅能利用 CTC 的强硬度杜绝特征失忆，又能利用 CE 获取脱离图元的排版潜规则。 |
| What have I done?    | V1(定宽单层), V2(列序列 CTC), V3(视线映射), V4(结合前朝全部大成的双轨架构) 的核心组件已全部构筑并串联合流！       |
