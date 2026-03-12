import os
import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

# Custom Modules
from models.crnn_ctc import CRNN
from train_crnn import train_crnn_model

def render_v2_ui():
    st.title("OCR V2: CRNN + CTC 端到端序列建模")
    st.markdown("""
    > “消灭切分，连接时序”
    > V2 引入了 Bi-LSTM 结合上下文特征，并使用最核心的 CTC (Connectionist Temporal Classification) 损失函数。
    > 它允许网络自由输出包含 `Blank` (空白/转移符) 的序列流，从而**完美解决了由于字体粘连导致的切割失败问题**。
    """)
    
    # --- 侧边栏: 训练控制台 ---
    st.sidebar.header("⚙️ V2 训练控制台 (CRNN)")
    st.sidebar.markdown("实时重训练定长序列识别网络。")
    
    # 【GPU 硬件展示】
    device_txt = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
    st.sidebar.success(f"当前加速引擎: **{device_txt}**")

    # CRNN 需要比单纯 CNN 更多的步数才能让特征与序列对齐收敛，因此默认升至 15 Epoch
    train_epochs = st.sidebar.slider("🏃 训练轮次 (Epochs)", min_value=1, max_value=100, value=15, key="v2_ep")
    batch_size = st.sidebar.selectbox("📦 Batch Size", [16, 32, 64, 128], index=1, key="v2_bs")
    learning_rate = st.sidebar.number_input("📉 学习率 (LR)", min_value=0.00001, max_value=0.1, value=0.001, step=0.0001, format="%.5f", key="v2_lr")
    
    if st.sidebar.button("🚀 开始训练新模型 (V2)"):
        st.sidebar.info("构建序列数据集并开辟 LSTM 寻路...")
        
        progress_bar = st.sidebar.progress(0)
        metrics_placeholder = st.sidebar.empty()
        chart_placeholder = st.sidebar.empty()
        
        history = {"train_loss": [], "val_loss": []}
        
        def update_progress(current, total):
            progress_bar.progress(current / total)
            
        def update_metrics(epoch, train_loss, val_loss):
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            metrics_placeholder.markdown(f"**Epoch {epoch}** - Train Loss: `{train_loss:.4f}` | Val Loss: `{val_loss:.4f}`")
            
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(range(1, len(history["train_loss"])+1), history["train_loss"], 'r-o', label="Train CTC Loss")
            ax.plot(range(1, len(history["val_loss"])+1), history["val_loss"], 'b-o', label="Val CTC Loss")
            ax.set_title("V2 CTC Loss Curve")
            ax.legend()
            
            chart_placeholder.pyplot(fig)
            plt.close(fig)

        best_loss = train_crnn_model(
            epochs=train_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            save_path="models/best_crnn.pth",
            progress_callback=update_progress,
            metric_callback=update_metrics
        )
        
        st.sidebar.success(f"V2训练完成！最佳 Loss: {best_loss:.4f}")
        st.cache_resource.clear()


    # --- 核心推理与热力图 UI ---
    MODEL_PATH = "models/best_crnn.pth"
    
    @st.cache_resource
    def load_crnn_model():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = CRNN(num_classes=11)
        if os.path.exists(MODEL_PATH):
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
            model.to(device)
            model.eval()
            return model, device
        return None, device
        
    @st.cache_resource
    def get_v2_transform():
        # V2 模型输入预处理 (灰度归一化)
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: 1.0 - x),
            transforms.Normalize((0.5,), (0.5,))
        ])

    model, device = load_crnn_model()
    if model is None:
        st.warning(f"缺失 V2 权重 `{MODEL_PATH}`，请先在侧边栏进行极速验证训练。")
        st.stop()
        
    uploaded_file = st.file_uploader("请上传一整行数字文本图 (系统将补帧或压缩为 256x32)", type=["png", "jpg", "jpeg"], key="v2_file")

    if uploaded_file is not None:
        image_pil = Image.open(uploaded_file).convert('L') # 强制灰度
        
        # --- 注入 V1-V5 统一图片清洗增强防线 ---
        from utils.image_processing import unified_enhance_image
        image_pil = unified_enhance_image(image_pil)
        
        img_np = np.array(image_pil)
        # 【BUG FIX】先消除四周多余留白区域，否则包含巨大白边的图片压缩到 H:32 后连带文字变得极其微小
        coords = cv2.findNonZero(255 - img_np)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            # 预留少许 5 像素的边缘 padding 防止特征贴边丢失
            x_s, y_s = max(0, x-5), max(0, y-5)
            x_e, y_e = min(img_np.shape[1], x+w+5), min(img_np.shape[0], y+h+5)
            image_pil = image_pil.crop((x_s, y_s, x_e, y_e))
            
        # 兼容固定长宽输入 256x32 的限定规则：
        orig_w, orig_h = image_pil.size
        # 等比例放大宽度
        new_w = int(orig_w * (32 / max(orig_h, 1)))
        
        # 【BUG FIX】极其重要的保护：如果拉长后宽度超越特征序列 256 上限，则强制挤压至 256
        # 防止 Image.paste() 粗暴切断文本
        if new_w > 256:
            new_w = 256
            
        scaled_img = image_pil.resize((new_w, 32), Image.Resampling.LANCZOS)
        
        # 组装最终喂给网络的基座画布（白底）
        canv = Image.new('L', (256, 32), color=255)
        # 将文本贴入左侧略带边距
        paste_x = min(5, 256 - new_w)
        canv.paste(scaled_img, (paste_x, 0))
        
        st.subheader("1. 序列化输入特征图")
        st.image(canv, caption="自适应组装为 [256 x 32] 维度的连续信号列", use_container_width=False)
        
        # 推理
        transform = get_v2_transform()
        tensor_input = transform(canv).unsqueeze(0).to(device) # shape: [1, 1, 32, 256]
        
        with torch.no_grad():
            outputs = model(tensor_input) # shape: [T(32), Batch(1), NumClass(11)]
            
            # 转 Softmax 获取概率
            probs = torch.nn.functional.softmax(outputs, dim=2)
            # 挤压 batch (现在是单张推理), 转置为绘图形状：[NumClasses(11), TimeSteps(32)]
            probs = probs.squeeze(1).T.cpu().numpy()
            
        st.subheader("2. CTC 对齐概率矩阵 (热力图)")
        st.markdown("*X轴：时间步 Sequence T  |  Y轴：0代表 Blank空白分隔符，1-10 映射真实数字*")
        
        fig, ax = plt.subplots(figsize=(15, 6))
        
        # 约定用户自定义色盘需求：
        # Blank (row 0) 使用极其显眼的暖色（亮红色凸出），其他使用深蓝到浅蓝（Blues）
        # 我们巧妙创建一个叠加的 mask 热图实现分层渲染。
        
        mask_blank = np.ones_like(probs, dtype=bool)
        mask_blank[0, :] = False  # 释放第一行(Blank)用暖色系
        
        mask_chars = np.ones_like(probs, dtype=bool)
        mask_chars[1:, :] = False # 释放后十行(字符)用冷色系
        
        # 画 Blank (行 0) 的暖色调
        sns.heatmap(probs, mask=mask_blank, cmap='Reds', cbar=False, ax=ax,
                    yticklabels=["Blank (∅)"] + [str(i) for i in range(10)])
                    
        # 画 字符 (行 1-10) 的冷色调
        sns.heatmap(probs, mask=mask_chars, cmap='Blues', cbar=True, ax=ax)
        
        plt.xlabel("Time Step (T)")
        plt.ylabel("Character Classes")
        plt.tight_layout()
        st.pyplot(fig)
        
        st.subheader("3. 动态规划路径解锁 (Greedy Decoding)")
        
        # 进行最简便的 CTC 贪心解码：取每一步概率最大者，去重，跳过 0(Blank)
        best_path_indices = np.argmax(probs, axis=0) # [T]
        
        raw_seq = [str(i-1) if i > 0 else "∅" for i in best_path_indices]
        st.code("  ".join(raw_seq), language="bash")
        
        decoded_string = []
        prev_idx = -1
        for idx in best_path_indices:
            # 去除重复及去 Blank(0)
            if idx != prev_idx and idx != 0:
                decoded_string.append(str(idx - 1))
            prev_idx = idx
            
        final_text = "".join(decoded_string)
        st.success("CRNN 序列收敛推测成功！", icon="✅")
        st.metric(label="最终文本", value=final_text)
