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
from models.v4_transformer_joint import V4JointModel
from train_v4_joint import train_v4_joint_model
from train_v4_joint import VOCAB_SIZE_ATTN, VOCAB_SIZE_CTC, id2char_attn, id2char_ctc, EOS_TOKEN

def render_v4_ui():
    st.title("OCR V4: CNN+Transformer 与 CTC-Attention 双轨联合训练")
    st.markdown("""
    > “效率与精度的双面真理”
    > 本系统抛弃了慢速串行的 LSTM，用强大的 **Transformer Encoder** (平行算力与全局感受野) 取代之。
    > 同时引入了并联的双路解法：一条辅路依靠 `CTC` 进行严厉的硬对齐保底，另一条主路依靠 `Attention` 预测语言语义。
    > 它解决了单独使用 Attention 容易发生的**对齐漂移现象 (Alignment Drift)**。
    """)
    
    # --- 侧边栏: 训练控制台 ---
    st.sidebar.header("⚙️ V4 训练控制台 (Joint Training)")
    st.sidebar.markdown("双 Loss 共同回传，模型将收敛出兼顾时序与上下文的稳固特征。")
    
    device_txt = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
    st.sidebar.success(f"当前加速引擎: **{device_txt}**")

    train_epochs = st.sidebar.slider("🏃 训练轮次 (Epochs)", min_value=1, max_value=100, value=20, key="v4_ep")
    batch_size = st.sidebar.selectbox("📦 Batch Size", [16, 32, 64, 128], index=1, key="v4_bs")
    learning_rate = st.sidebar.number_input("📉 学习率 (LR)", min_value=0.00001, max_value=0.1, value=0.001, step=0.0001, format="%.5f", key="v4_lr")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("🧠 **混合架构剖析 (沙盒动态调参) ⚠️**")
    use_stn = st.sidebar.checkbox("📐 启用前置 STN (Spatial Transformer Network)", value=False, help="让网络自己学习如何对扭曲、透视变形的文本区域进行几何拉伸和矫正。开启后需重新训练独立模型权重。")
    lambda_ctc = st.sidebar.slider("⚖️ CTC Loss 权重比例 (0~1)", min_value=0.0, max_value=1.0, value=0.2, step=0.1, key="v4_lam", help="1.0 = 纯 CTC (绝对硬对齐但无语境)；  0.0 = 纯 Attention (容易视线漂移出错)。  默认为 0.2，让主路学习语言逻辑，让 CTC 在一旁当纠错保底。")
    num_layers = st.sidebar.slider("🧱 Transformer 层数", min_value=1, max_value=3, value=1, key="v4_nl", help="对于 32x256 的简单特征序列，1 层足以交流。更深的层数反而容易造成全局注意力稀释化并退化为噪声。")
    nhead = st.sidebar.selectbox("🎯 Multi-head Attention 头数", [1, 2, 4], index=1, key="v4_nh", help="决定了网络有多几个不同维度的观察视角。")
    
    if st.sidebar.button("🚀 开始训练新模型 (V4)"):
        st.sidebar.info("编译 Transformer 参数，开启多目标损失联合优化...")
        
        progress_bar = st.sidebar.progress(0)
        metrics_placeholder = st.sidebar.empty()
        chart_placeholder = st.sidebar.empty()
        
        history = {"train_loss": [], "val_loss": []}
        
        def update_progress(current, total):
            progress_bar.progress(current / total)
            
        def update_metrics(epoch, train_loss, val_loss):
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            metrics_placeholder.markdown(f"**Epoch {epoch}** - Train Joint Loss: `{train_loss:.4f}` | Val Joint Loss: `{val_loss:.4f}`")
            
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(range(1, len(history["train_loss"])+1), history["train_loss"], 'r-o', label="Train (CTC+CE) Loss")
            ax.plot(range(1, len(history["val_loss"])+1), history["val_loss"], 'b-o', label="Val (CTC+CE) Loss")
            ax.set_title("V4 Joint Loss Curve")
            ax.legend()
            
            chart_placeholder.pyplot(fig)
            plt.close(fig)

        stn_suffix = "_stn" if use_stn else ""
        save_path = f"models/best_v4{stn_suffix}_joint.pth"
        
        best_loss = train_v4_joint_model(
            epochs=train_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            save_path=save_path,
            progress_callback=update_progress,
            metric_callback=update_metrics,
            lambda_ctc=lambda_ctc,
            nhead=nhead,
            num_layers=num_layers,
            use_stn=use_stn
        )
        
        st.sidebar.success(f"V4 联合训练完成！最佳 Loss: {best_loss:.4f}")
        st.cache_resource.clear()

    # --- 核心推理与双面真理对比 UI ---
    stn_suffix = "_stn" if use_stn else ""
    MODEL_PATH = f"models/best_v4{stn_suffix}_joint.pth"
    
    @st.cache_resource
    def load_v4_model(nl, nh, stn_enabled):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = V4JointModel(output_dim_attn=VOCAB_SIZE_ATTN, output_dim_ctc=VOCAB_SIZE_CTC, nhead=nh, num_layers=nl, use_stn=stn_enabled)
        if os.path.exists(MODEL_PATH):
            try:
                model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
                model.to(device)
                model.eval()
                return model, device
            except RuntimeError:
                st.sidebar.error("⚠️ 本地权重模型的 Transformer 层数或头数与当前左侧选择不匹配！请点击上方「开始训练」重新编译对齐尺寸。")
                return None, device
        return None, device
        
    @st.cache_resource
    def get_v4_transform():
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: 1.0 - x),
            transforms.Normalize((0.5,), (0.5,))
        ])

    model, device = load_v4_model(num_layers, nhead, use_stn)
    if model is None:
        st.warning(f"缺失 V4 权重 `{MODEL_PATH}`，请先在侧边栏点击进行验证训练。{' (因开启了 STN，需独立训练)' if use_stn else ''}")
        st.stop()
        
    uploaded_file = st.file_uploader("请上传数字与字母混合图 (如被拉伸/加长/大量背景空白/多行文本，将绝佳地展现 CTC 如何压制漂移)", type=["png", "jpg", "jpeg"], key="v4_file")

    if uploaded_file is not None:
        image_pil = Image.open(uploaded_file).convert('L')
        
        from utils.image_processing import unified_enhance_image, extract_text_lines
        enhanced_res = unified_enhance_image(image_pil)
        enhanced_img = enhanced_res['image']
        
        # 多行文本水平切割
        text_lines = extract_text_lines(enhanced_img)
        
        if len(text_lines) > 1:
            st.info(f"检测到多行文本，已切分为 {len(text_lines)} 个单行片段 (Line Segmentation)。")
            
        # UI: 让用户选择哪一行来深入剖析注意力矩阵
        selected_idx = 0
        if len(text_lines) > 1:
            cols = st.columns(len(text_lines))
            for i, line_img in enumerate(text_lines):
                cols[i].image(line_img, caption=f"行 {i+1}", use_container_width=True)
            selected_idx = st.radio("选择要分析的行：", range(len(text_lines)), format_func=lambda x: f"行 {x+1}", horizontal=True)
            
        target_line_pil = text_lines[selected_idx]
            
        orig_w, orig_h = target_line_pil.size
        # V4 / V2 固定高度 32, 宽度最长 256
        new_w = int(orig_w * (32 / max(orig_h, 1)))
        if new_w > 256: new_w = 256
            
        scaled_img = target_line_pil.resize((new_w, 32), Image.Resampling.LANCZOS)
        
        canv = Image.new('L', (256, 32), color=255)
        paste_x = min(5, 256 - new_w)
        canv.paste(scaled_img, (paste_x, 0))
        
        transform = get_v4_transform()
        tensor_input = transform(canv).unsqueeze(0).to(device) # [1, 1, 32, 256]
        
        with torch.no_grad():
            # 推理阶段 tf_ratio=0.0，不用 target，由其自主发散最长 20 步
            ctc_outputs, outputs_attn, attention_maps, self_attn_weights = model(tensor_input, trg=None, teacher_forcing_ratio=0.0)
            
            # 1. Attention 解析 (取最自信的结果)
            predictions = outputs_attn.argmax(dim=2).squeeze(0).cpu().numpy() # [max_len]
            attn_weights = attention_maps.squeeze(0).cpu().numpy() # [max_len, 128]
            
            # 2. CTC 解析 (获取贪婪矩阵概率)
            ctc_probs = torch.softmax(ctc_outputs.squeeze(0), dim=1).cpu().numpy() # [T=32, VOCAB_CTC=16]
            
            # 3. Transformer 自注意力解析
            # 默认 average_attn_weights=True，返回 [B, T, T] -> [1, 32, 32]
            if len(self_attn_weights) > 0 and self_attn_weights[-1] is not None:
                last_self_attn = self_attn_weights[-1].squeeze(0).cpu().numpy() # [32, 32]
            else:
                last_self_attn = None
            
        # 截断 Attention 结果
        decoded_chars = []
        valid_steps = 0
        for p in predictions:
            if p == EOS_TOKEN:
                break
            decoded_chars.append(id2char_attn.get(p, "?"))
            valid_steps += 1
            
        st.success(f"🟢 **Attention 分支语境译码结果**: `{''.join(decoded_chars)}`")

        st.markdown("---")
        st.subheader("⚖️ 双分支联合对齐论证中心 (CTC vs Attention)")
        st.markdown("在这里，你将能清楚洞悉，当网络在进行特征寻找时，两条分支各自秉承的世界观与对齐方案。通过这二者的结合，诞生了最稳固的全局视野模型。")

        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("##### 🔍 1. 辅路约束：CTC 严格硬对齐热图")
            st.markdown("使用 Blank (深红色) 填塞一切无字之境，严格杜绝虚幻的复读与漂移。")
            fig_ctc, ax_ctc = plt.subplots(figsize=(8, 4))
            
            # 矩阵转置：[类别, 时间跨度 T=32]。Y轴为所有的字母种类，X轴为特征窗格。
            ctc_T = ctc_probs.T
            
            # 将 Blank(0) 单独拿出来渲染暖色调吸引眼球，其余冷色调
            cmap_custom = sns.diverging_palette(240, 10, as_cmap=True) # 蓝到红
            
            sns.heatmap(ctc_T, cmap="coolwarm", cbar=True, ax=ax_ctc)
            ylabels = ["Blank(0)"] + [id2char_ctc.get(i, "?") for i in range(1, VOCAB_SIZE_CTC)]
            ax_ctc.set_yticks(np.arange(VOCAB_SIZE_CTC) + 0.5)
            ax_ctc.set_yticklabels(ylabels, rotation=0, fontsize=8)
            ax_ctc.set_xlabel("Time Steps (T=32)")
            st.pyplot(fig_ctc)
            plt.close(fig_ctc)

        with col2:
            st.markdown("##### 👁️ 2. 主路决策：Attention 多态视线叠加")
            if valid_steps == 0:
                st.info("Attention 未能在图片上解码出合法语义。")
            else:
                st.markdown("通过 LSTM/Transformer 混合，自回归地回读上一个字符进而去追踪下一个图像焦点。")
                step_idx = st.slider("👉 滑动游走每一帧决策点：", min_value=0, max_value=max(1, valid_steps-1), value=0, key="v4_slide")
                
                if step_idx < valid_steps:
                    current_char = decoded_chars[step_idx]
                    st.caption(f"当前指针聚焦在字符: **{current_char}**")
                    
                    attn = attn_weights[step_idx]
                    attn_matrix = attn.reshape((1, 32)) 
                    attn_up = cv2.resize(attn_matrix, (256, 32), interpolation=cv2.INTER_CUBIC)
                    
                    bg_img = np.array(canv)
                    bg_rgb = cv2.cvtColor(255 - bg_img, cv2.COLOR_GRAY2RGB)
                    
                    # 规避除零错误
                    max_attn = np.max(attn_up)
                    if max_attn == 0: max_attn = 1
                        
                    attn_norm = np.uint8(255 * attn_up / max_attn)
                    heatmap = cv2.applyColorMap(attn_norm, cv2.COLORMAP_JET)
                    overlay = cv2.addWeighted(bg_rgb, 0.4, heatmap, 0.6, 0)
                    
                    st.image(overlay, caption=f"Jet Heatmap Overlay (Step: {step_idx})", use_container_width=True)

        if last_self_attn is not None:
            with st.expander("🛠️ 【高阶向】Transformer 全局自相注意力矩阵 (Self-Attention)", expanded=False):
                st.markdown("""
                这是为何 Transformer 能够击溃古老 RNN 的终极所在：一次性并发计算全域！
                这幅图描绘了 **32 个时间切片彼此之间的联想强度**。对角线代表自身特征，非对角线处的亮斑意味着 `左边部分` 成功察觉到了 `右边部分` 的遥远羁绊。
                """)
                fig_sa, ax_sa = plt.subplots(figsize=(6, 6))
                sns.heatmap(last_self_attn, cmap="viridis", square=True, cbar=True, ax=ax_sa)
                ax_sa.set_xlabel("Query Time Steps")
                ax_sa.set_ylabel("Key Time Steps")
                st.pyplot(fig_sa)
                plt.close(fig_sa)
