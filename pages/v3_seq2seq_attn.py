import os
import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Custom Modules
from models.seq2seq_attn import Seq2Seq
from train_v3 import train_seq2seq_model, VOCAB_SIZE, id2char, EOS_TOKEN
from train_v3 import text_to_tensor

def render_v3_ui():
    st.title("OCR V3: Seq2Seq + Attention 隐式语言模型")
    st.markdown("""
    > “赋予机器语境的凝视”
    > V3 彻底摒弃了 CTC 假设“字与字之间互相独立”的弱局。通过 Encoder-Decoder 架构，模型在每一步推断时，
    > 不仅**看着从 CNN 传来的图** (Attention Map)，还在**读着自己上一步写下的字** (Language Model)。
    > 它能通过学习，掌握类似类似 `YYYY-MM` 日期格式或车牌号的潜在组合规则。
    """)
    
    # --- 侧边栏: 训练控制台 ---
    st.sidebar.header("⚙️ V3 训练控制台 (Seq2Seq)")
    st.sidebar.markdown("由于语言模型参数极大，其训练过程会略长于前两代。")
    
    # 【GPU 硬件展示】
    device_txt = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
    st.sidebar.success(f"当前加速引擎: **{device_txt}**")

    # 包含 Teacher forcing 和大量参数，默认给足迭代期
    train_epochs = st.sidebar.slider("🏃 训练轮次 (Epochs)", min_value=1, max_value=100, value=25, key="v3_ep")
    batch_size = st.sidebar.selectbox("📦 Batch Size", [16, 32, 64, 128], index=1, key="v3_bs")
    learning_rate = st.sidebar.number_input("📉 学习率 (LR)", min_value=0.00001, max_value=0.1, value=0.001, step=0.0001, format="%.5f", key="v3_lr")
    
    if st.sidebar.button("🚀 开始训练新模型 (V3)"):
        st.sidebar.info("生成结构化语言图集并在 LSTM+Attention 加持下开始训练...")
        
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
            ax.plot(range(1, len(history["train_loss"])+1), history["train_loss"], 'r-o', label="Train CE Loss")
            ax.plot(range(1, len(history["val_loss"])+1), history["val_loss"], 'b-o', label="Val CE Loss")
            ax.set_title("V3 Seq2Seq Loss Curve")
            ax.legend()
            
            chart_placeholder.pyplot(fig)
            plt.close(fig)

        best_loss = train_seq2seq_model(
            epochs=train_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            save_path="models/best_v3_seq2seq.pth",
            progress_callback=update_progress,
            metric_callback=update_metrics
        )
        
        st.sidebar.success(f"V3训练完成！最佳 Loss: {best_loss:.4f}")
        st.cache_resource.clear()


    # --- 核心推理与 Attention UI ---
    MODEL_PATH = "models/best_v3_seq2seq.pth"
    
    @st.cache_resource
    def load_v3_model():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Seq2Seq(vocab_size=VOCAB_SIZE)
        if os.path.exists(MODEL_PATH):
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
            model.to(device)
            model.eval()
            return model, device
        return None, device
        
    @st.cache_resource
    def get_v3_transform():
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: 1.0 - x),
            transforms.Normalize((0.5,), (0.5,))
        ])

    model, device = load_v3_model()
    if model is None:
        st.warning(f"缺失 V3 权重 `{MODEL_PATH}`，请先在侧边栏点击进行验证训练。")
        st.stop()
        
    st.markdown("测试提示：您可以上传一段类似 `2026-03`，`A678-C` 这样的具有潜在排版含义的随机文字图。")
    uploaded_file = st.file_uploader("请上传数字与符号复合字符串图 (系统将自适应铺填至 256x32)", type=["png", "jpg", "jpeg"], key="v3_file")

    if uploaded_file is not None:
        image_pil = Image.open(uploaded_file).convert('L')
        
        # --- 注入 V1-V5 统一图片清洗增强防线 ---
        from utils.image_processing import unified_enhance_image
        image_pil = unified_enhance_image(image_pil)
        
        # 裁剪并贴图防撕裂机制 (与 V2 相同)
        img_np = np.array(image_pil)
        coords = cv2.findNonZero(255 - img_np)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            x_s, y_s = max(0, x-5), max(0, y-5)
            x_e, y_e = min(img_np.shape[1], x+w+5), min(img_np.shape[0], y+h+5)
            image_pil = image_pil.crop((x_s, y_s, x_e, y_e))
            
        orig_w, orig_h = image_pil.size
        new_w = int(orig_w * (32 / max(orig_h, 1)))
        if new_w > 256: new_w = 256
            
        scaled_img = image_pil.resize((new_w, 32), Image.Resampling.LANCZOS)
        
        # 为了美观，V3 的 attention 热图适合放在带有微底色的全黑或全白舞台
        canv = Image.new('L', (256, 32), color=255)
        paste_x = min(5, 256 - new_w)
        canv.paste(scaled_img, (paste_x, 0))
        
        # 获取推断结构
        transform = get_v3_transform()
        tensor_input = transform(canv).unsqueeze(0).to(device) # [1, 1, 32, 256]
        
        with torch.no_grad():
            # max_len 设置推断上限
            outputs, attention_maps, _ = model(tensor_input, targets=None, teacher_forcing_ratio=0.0, max_len=12)
            # outputs: [1, max_len, Vocab] -> 取 argmax 得到字
            predictions = outputs.argmax(dim=2).squeeze(0).cpu().numpy() # [max_len]
            # attention_maps: [1, max_len, 128 (4*32)]
            attn_weights = attention_maps.squeeze(0).cpu().numpy()
            
        # 截断到 EOS Token 或者有效长度
        decoded_chars = []
        valid_steps = 0
        for p in predictions:
            if p == EOS_TOKEN:
                break
            decoded_chars.append(id2char.get(p, "?"))
            valid_steps += 1
            
        if valid_steps == 0:
            st.error("未能识别出序列，请提供清晰图像或再多训练几个 Epoch。")
            st.stop()
            
        st.success(f"🔥 模型最终译码结果: **{''.join(decoded_chars)}**")

        st.markdown("---")
        st.subheader("👁️ 注意力游走矩阵展现 (Attention Visualizer)")
        st.markdown("滑动下方控制器，**观察模型在吐出每一个字符时，究竟死死盯住了图片上的哪一处像素区。**")
        
        # 定义滑动控制轴
        step_idx = st.slider("时间步推断指针 (Time Step):", min_value=0, max_value=valid_steps-1, value=0, 
                             format="步数 %d")
        
        current_char = decoded_chars[step_idx]
        st.info(f"第 **{step_idx+1}** 步，模型决定输出:  `{current_char}`")
        
        # 提取当前步的 attention map 并且 reshape 为与特征层级一致的空间结构 (H=4, W=32)
        attn = attn_weights[step_idx] # [128]
        # 注意：在 Encoder 中我们是 .view(B, C, -1).permute(0, 2, 1) 展平的，
        # 原始形状是 [B, C, H(4), W(32)], 因此 H*W = 128
        # 所以它的重构应该是：(4, 32)
        attn_matrix = attn.reshape((4, 32))
        
        # 使用 OpenCV 将这层迷你的 4x32 注意力墙 Resize(插值放大) 回原图比例 32x256
        attn_up = cv2.resize(attn_matrix, (256, 32), interpolation=cv2.INTER_CUBIC)
        
        # 制造热力调色 Alpha Blending
        # 1. 取出原图底子转换为 RGB 黑底更具热成像感（把底片从白底黑字 Invert 过来看）
        bg_img = np.array(canv)
        bg_rgb = cv2.cvtColor(255 - bg_img, cv2.COLOR_GRAY2RGB)
        
        # 2. 把归一化 Attention 上热力色
        attn_norm = np.uint8(255 * attn_up / np.max(attn_up))
        heatmap = cv2.applyColorMap(attn_norm, cv2.COLORMAP_JET)
        
        # 3. 叠加并使用 Streamlit 展示
        overlay = cv2.addWeighted(bg_rgb, 0.4, heatmap, 0.6, 0)
        
        st.image(overlay, caption="Attention Heatmap (Jet Overlay)", use_container_width=False)
