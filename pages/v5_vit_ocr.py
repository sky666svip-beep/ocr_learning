import os
import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

# V5 专属库
from models.v5_vit_ocr import V5ViTOCR
from train_v5_vit import train_v5_vit_model
from train_v5_vit import VOCAB_SIZE, id2char, EOS_TOKEN
from utils.image_processing import unified_enhance_image

# ==========================================
# 页面缓存机制
# ==========================================
@st.cache_resource
def load_v5_local_model(nl, nh):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = V5ViTOCR(vocab_size=VOCAB_SIZE, d_model=64, nhead=nh, num_enc_layers=nl, num_dec_layers=nl, dim_ff=256)
    model_path = "models/best_v5_vit.pth"
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            model.to(device)
            model.eval()
            return model, device
        except Exception as e:
            st.sidebar.error("⚠️ 本地微型 ViT 权重与所选层数不匹，请重新训练！")
            return None, device
    return None, device

@st.cache_resource
def get_v5_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: 1.0 - x),
        transforms.Normalize((0.5,), (0.5,))
    ])

@st.cache_resource
def load_huggingface_trocr(repo_id):
    try:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        processor = TrOCRProcessor.from_pretrained(repo_id)
        model = VisionEncoderDecoderModel.from_pretrained(repo_id)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        return processor, model, device
    except Exception as e:
        return None, None, None

def render_v5_ui():
    st.title("OCR V5: 双模式 Vision Transformer (ViT-OCR/TrOCR)")
    st.markdown("""
    > “视觉与语言的大一统时代 (A Unified Transformer View)”
    > 彻底抛除 CNN 提取器！这是本沙箱的最强演化形态。
    > 本版本采用双层验证体系：**左侧可选用教学手搓版**（洞析 256 个切块如何自相注意），也能**一键加载 HuggingFace 预训练大模型**体验工业级中文/英文排版 SOTA 读取并窥探 Cross-Attention 对齐魔法。
    """)
    
    # --- 侧边栏 ---
    st.sidebar.header("🕹️ V5 引擎启动器")
    
    engine_mode = st.sidebar.radio("选择运行核心：", ["🔬 手搓微型 ViT (教学模式)", "🏭 预训练 TrOCR (工业 SOTA)"])
    st.sidebar.divider()
    
    if engine_mode == "🔬 手搓微型 ViT (教学模式)":
        st.sidebar.markdown("**本地沙盒网络调参**")
        num_layers = st.sidebar.slider("🧱 Encoder/Decoder 层数", 1, 3, 2, key="v5_nl")
        nhead = st.sidebar.selectbox("🎯 Attention 头数", [1, 2, 4], index=1, key="v5_nh")
        train_epochs = st.sidebar.slider("🏃 训练轮次 (Epochs)", min_value=1, max_value=100, value=25, key="v5_ep")
        batch_size = st.sidebar.selectbox("📦 Batch Size", [16, 32, 64, 128], index=1, key="v5_bs")
        learning_rate = st.sidebar.number_input("📉 学习率 (LR)", min_value=0.00001, max_value=0.1, value=0.001, step=0.0001, format="%.5f", key="v5_lr")
        
        if st.sidebar.button("🚀 开始训练本地 ViT (V5)"):
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
                ax.plot(range(1, len(history["train_loss"])+1), history["train_loss"], 'r-o', label="Train Loss")
                ax.plot(range(1, len(history["val_loss"])+1), history["val_loss"], 'b-o', label="Val Loss")
                ax.set_title("V5 ViT-OCR Loss Curve")
                ax.legend()
                
                chart_placeholder.pyplot(fig)
                plt.close(fig)
            
            best_loss = train_v5_vit_model(
                epochs=train_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                save_path="models/best_v5_vit.pth",
                progress_callback=update_progress,
                metric_callback=update_metrics,
                nhead=nhead,
                num_enc_layers=num_layers,
                num_dec_layers=num_layers
            )
            st.sidebar.success(f"本地 ViT 训练完成！最佳 Loss: {best_loss:.4f}")
            st.cache_resource.clear()
            
    else:
        st.sidebar.markdown("**HuggingFace 在线模型库**")
        hf_models = [
            "microsoft/trocr-small-printed",
            "microsoft/trocr-base-stage1",
            "ZihCiLin/trocr-traditional-chinese-baseline", # 用于中文增强
            "自定义（手动输入）"
        ]
        chosen_hf = st.sidebar.selectbox("选择或输入仓库 ID:", hf_models)
        if chosen_hf == "自定义（手动输入）":
            chosen_hf = st.sidebar.text_input("HuggingFace Repo ID:")
            
        if not chosen_hf:
            st.stop()
            
        with st.sidebar.status(f"正在挂载 {chosen_hf}..."):
            hf_proc, hf_model, hf_dev = load_huggingface_trocr(chosen_hf)
            if hf_model is None:
                st.sidebar.error("预训练模型加载失败，请检查网络或库是否存在。")
                st.stop()
            st.sidebar.success("挂载极速完成！")

    # --- 主检验场 ---
    uploaded_file = st.file_uploader("请上传待推理的图片 (推荐上传带有较长文本的图像考验纯 Transformer 的解析力)", type=["png", "jpg", "jpeg"], key="v5_file")

    if uploaded_file is not None:
        image_pil = Image.open(uploaded_file).convert('RGB')
        
        # 显示图片区域
        st.image(image_pil, caption="用户原图", use_container_width=True)
        
        if engine_mode == "🔬 手搓微型 ViT (教学模式)":
            model, device = load_v5_local_model(num_layers, nhead)
            if model is None:
                st.warning("缺失 V5 教学版权重，请点击左侧开展实地演戏训练！")
                st.stop()
                
            # 执行预处理 (转换为我们沙箱规范定义的 256x32 L)
            image_gray = image_pil.convert('L')
            image_gray = unified_enhance_image(image_gray)
            
            # TODO: 进行裁剪排版对其至 256x32
            img_np = np.array(image_gray)
            coords = cv2.findNonZero(255 - img_np)
            if coords is not None:
                x, y, w, h = cv2.boundingRect(coords)
                image_gray = image_gray.crop((x, y, x+w, y+h))
            
            orig_w, orig_h = image_gray.size
            new_w = min(256, int(orig_w * (32 / max(orig_h, 1))))
            scaled_img = image_gray.resize((new_w, 32), Image.Resampling.LANCZOS)
            canv = Image.new('L', (256, 32), color=255)
            canv.paste(scaled_img, (min(5, 256-new_w), 0))
            
            transform = get_v5_transform()
            tensor_input = transform(canv).unsqueeze(0).to(device) # (1, 1, 32, 256)
            
            # 推断
            SOS_ID = 1 # 根据前面定义
            EOS_ID = 2
            PAD_ID = 0
            preds, self_attns, cross_attns = model.inference(tensor_input, SOS_ID, EOS_ID)
            
            clean_preds = [p for p in preds if p not in (SOS_ID, EOS_ID, PAD_ID)]
            decoded_text = "".join([id2char.get(p, "?") for p in clean_preds])
            
            st.success(f"🟢 **【微型 ViT 解码器推断结果】**: `{decoded_text}`")
            
            render_local_vit_visualizations(canv, decoded_text, self_attns, cross_attns)
            
        else:
            # 工业 SOTA 路线
            st.info("TrOCR 将通过内部自带的 Processor 解构整幅彩色图像...")
            pixel_values = hf_proc(images=image_pil, return_tensors="pt").pixel_values.to(hf_dev)
            
            with torch.no_grad():
                # 要求强制输出 Attention 以供剖析
                out = hf_model.generate(pixel_values, output_attentions=True, output_scores=True, return_dict_in_generate=True)
                
            decoded_text = hf_proc.batch_decode(out.sequences, skip_special_tokens=True)[0]
            st.success(f"🔵 **【TrOCR SOTA 工业级译码结果】**: `{decoded_text}`")
            
            # HuggingFace 的 Attention Tuple 极为复杂，抽取其末端层以便展示
            # 注意: 生成器在每一解码步都会输出一层 attention
            if hasattr(out, 'encoder_attentions') and out.encoder_attentions is not None:
                enc_attn = out.encoder_attentions[-1] # tuple of layers, we take the last layer
            else:
                enc_attn = None
                
            if hasattr(out, 'cross_attentions') and out.cross_attentions is not None:
                # out.cross_attentions 是包含解码器各个步的元组
                # 一步包含了一层的 cross attention tuple 
                dec_cross = out.cross_attentions 
            else:
                dec_cross = None
                
            render_hf_trocr_visualizations(image_pil, decoded_text, enc_attn, dec_cross, out.sequences[0])

# ==========================================
# 独立的可视化排版模块
# ==========================================
def render_local_vit_visualizations(canv, decoded_text, self_attns, cross_attns):
    st.markdown("---")
    st.subheader("👁️ 纯 Attention 矩阵多维解刨")
    
    # 1. Self Attention Map (Encoder)
    with st.expander("🧩 1. ViT Encoder 的内部共振图 (Self-Attention Maps)", expanded=True):
        st.markdown("此时图片已化为了 256 个切成 `4x8` 大小的图像切块 (Patch)。这些切片在进入注意力头时相互沟通。")
        if len(self_attns) > 0:
            last_self = self_attns[-1].squeeze().cpu().numpy() # [256, 256] 或是多头 [2, 256, 256]
            if len(last_self.shape) == 3:
                last_self = last_self.mean(axis=0) # [256, 256]
            
            fig, ax = plt.subplots(figsize=(7, 7))
            sns.heatmap(last_self, cmap="mako", cbar=False, ax=ax, xticklabels=False, yticklabels=False)
            ax.set_title("Patch 到 Patch 的交谈密度")
            st.pyplot(fig)
            plt.close(fig)

    # 2. Cross Attention (Decoder gazing at Encoder)
    if len(cross_attns) > 0 and len(decoded_text) > 0:
        with st.expander("🔦 2. Transformer Decoder 投射与凝视 (Cross-Attention Maps)", expanded=True):
            st.markdown("当文本解码器试图读出一个字时，它会向所有 256 个图像 Patch 广播求索。这就是多模态交互时“视线追踪”的核心机密。")
            
            step = st.slider("拖拽以查看逐步跨模态对齐", 0, len(decoded_text)-1, 0, key="v5_s1")
            
            # cross_attns 是 decoder 每一层的输出列表。我们取最后层的 cross-attention
            last_cross_attn = cross_attns[-1] # 形如 (1, tgt_len, num_patches)
            
            # tgt_len 等于生成的序列长度（包含逐步加入的前缀）。
            # step 取值从 0 到 L-1。
            attn_val = last_cross_attn[0, step, :].detach().cpu().numpy() # [256]
                
            # 我们将 256 还原回 8(高) x 32(宽) 的网格结构！
            attn_grid = attn_val.reshape((8, 32))
            attn_resize = cv2.resize(attn_grid, (256, 32), interpolation=cv2.INTER_CUBIC)
            
            max_val = np.max(attn_resize)
            if max_val == 0: max_val = 1
            heatmap = np.uint8(255 * attn_resize / max_val)
            heat_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            
            bg_rgb = cv2.cvtColor(255 - np.array(canv), cv2.COLOR_GRAY2RGB)
            over = cv2.addWeighted(bg_rgb, 0.4, heat_color, 0.6, 0)
            
            st.image(over, caption=f"解构字符 '{decoded_text[step]}' 时产生的绝对空间视界焦点", use_container_width=True)

def render_hf_trocr_visualizations(image_pil, decoded_text, enc_attn, dec_cross, sequences):
    st.markdown("---")
    st.subheader("🌋 工业 SOTA 注意力投影溯源")
    if enc_attn is None or dec_cross is None:
        st.info("该模型的输出结构无法完美兼容我们的注意力挖掘钩子，暂只提供译码展示。")
        return
        
    # HuggingFace 的图像通常是较大的如 384x384。Patch 长度对应着诸如 576 等 (取决于 16x16 切块)
    import math
    
    with st.expander("🎯 Cross-Attention 跨模态投影", expanded=True):
        st.markdown("探测 HuggingFace 巨兽是如何盯住并抽出语义的。")
        
        # dec_cross 是个长列表, 每个 item 对应 decode step
        valid_steps = min(len(decoded_text), len(dec_cross))
        if valid_steps > 0:
            step = st.slider("洞察 TrOCR 解码步骤", 0, valid_steps-1, 0, key="v5_hf_s1")
            
            # 取该 step 的最后一个 decoder layer 的 cross attention
            # dec_cross[step] -> shape tuple for each layer. 
            step_layers = dec_cross[step]
            last_layer_attn = step_layers[-1] # [B, num_heads, tgt_len=1, num_patches]
            
            attn_vec = last_layer_attn[0, :, -1, :].detach().cpu().numpy().mean(axis=0) # [num_patches]
            
            # 去除一些特殊 token 如 cls token 
            # 视模型而定。假设 TrOCR-small-printed 用 DeiT，有 1 个 cls token
            if len(attn_vec) == 577: 
                attn_vec = attn_vec[1:] # 移除 Cls token (变成了 576)
                
            num_patch = len(attn_vec)
            side = int(math.sqrt(num_patch))
            
            if side * side == num_patch:
                attn_grid = attn_vec.reshape((side, side))
                
                # 重塑到原图尺寸并叠加
                orig_w, orig_h = image_pil.size
                attn_resize = cv2.resize(attn_grid, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
                
                max_v = np.max(attn_resize)
                if max_v == 0: max_v=1
                heatmap = np.uint8(255 * attn_resize / max_v)
                heat_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                
                img_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
                over = cv2.addWeighted(img_cv, 0.5, heat_color, 0.5, 0)
                
                st.image(cv2.cvtColor(over, cv2.COLOR_BGR2RGB), caption=f"预测 '{decoded_text[step]}' 步的视点", use_container_width=True)
            else:
                st.warning("Patch 数量非严格正方形，重塑失败。")
