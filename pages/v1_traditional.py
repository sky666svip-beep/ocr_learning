import os
import streamlit as st

def render_v1_ui():
    import torch
    from PIL import Image, ImageDraw
    import matplotlib.pyplot as plt
    from torchvision import transforms

    # Custom Modules
    from models.cnn_classifier import SimpleDigitCNN
    from utils.image_processing import process_image, extract_character_patches
    from train import train_model

    st.title("OCR V1: 传统字符切分与 CNN 分类")
    st.markdown("""
    > “这是所有端到端 OCR 方案的对照基石，理解序列建模核心价值的必经之路”
    > 本演示模拟了最原始的 OCR 实现思路：**传统图像先切分 -> CNN 挨个认**。
    > 您可以观察到它是如何轻易被**粘连**和**排版噪声**所击败的，这是孕育现代 CRNN 等方案的根本动因。
    """)
    
    # --- 侧边栏: 训练控制台 ---
    st.sidebar.header("⚙️ V1 训练控制台")
    st.sidebar.markdown("支持在此直接重训练 CNN 模型观察表现。")
    
    # 【GPU 硬件展示】
    device_txt = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
    st.sidebar.success(f"当前加速引擎: **{device_txt}**")

    train_epochs = st.sidebar.slider("🏃 训练轮次 (Epochs)", min_value=1, max_value=100, value=10, key="v1_ep")
    batch_size = st.sidebar.selectbox("📦 Batch Size", [16, 32, 64, 128], index=2, key="v1_bs")
    learning_rate = st.sidebar.number_input("📉 学习率 (LR)", min_value=0.00001, max_value=0.1, value=0.001, step=0.0001, format="%.5f", key="v1_lr")
    if st.sidebar.button("🚀 开始训练新模型 (V1)"):
        st.sidebar.info("开始生成数据与训练...")
        
        progress_bar = st.sidebar.progress(0)
        metrics_placeholder = st.sidebar.empty()
        chart_placeholder = st.sidebar.empty()
        
        history = {"train_loss": [], "val_acc": []}
        
        def update_progress(current, total):
            progress_bar.progress(current / total)
            
        def update_metrics(epoch, train_loss, val_acc):
            history["train_loss"].append(train_loss)
            history["val_acc"].append(val_acc)
            metrics_placeholder.markdown(f"**Epoch {epoch}** - Loss: `{train_loss:.4f}` | Val Acc: `{val_acc:.2f}%`")
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
            ax1.plot(range(1, len(history["train_loss"])+1), history["train_loss"], 'r-o')
            ax1.set_title("Training Loss")
            
            ax2.plot(range(1, len(history["val_acc"])+1), history["val_acc"], 'b-o')
            ax2.set_title("Validation Accuracy (%)")
            
            chart_placeholder.pyplot(fig)
            plt.close(fig)

        best_acc = train_model(
            epochs=train_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            save_path="models/best_model.pth",
            progress_callback=update_progress,
            metric_callback=update_metrics
        )
        
        st.sidebar.success(f"V1训练完毕！最佳准确率: {best_acc:.2f}%. 新权重已生效。")
        st.cache_resource.clear()

    # --- 核心 UI ---
    MODEL_PATH = "models/best_model.pth"
    
    @st.cache_resource
    def load_model():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SimpleDigitCNN()
        if os.path.exists(MODEL_PATH):
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
            model.to(device)
            model.eval()
            return model, device
        return None, device

    @st.cache_resource
    def get_transform():
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: 1.0 - x),
            transforms.Normalize((0.5,), (0.5,))
        ])

    model, device = load_model()
    if model is None:
        st.warning(f"缺失 V1 权重 `{MODEL_PATH}`，请先在侧边栏点击训练。")
        st.stop()
        
    uploaded_file = st.file_uploader("请上传 0-9 纯数字测试图片 (白底黑字)", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image_pil = Image.open(uploaded_file)
        
        # --- 注入 V1-V5 统一图片清洗增强防线 ---
        from utils.image_processing import unified_enhance_image
        image_pil = unified_enhance_image(image_pil)
        
        col_left, col_right = st.columns([1, 1])
        
        with col_left:
            st.subheader("🛠️ 过程解剖 (切分依据)")
            gray_img, binary_img, projection, bboxes = process_image(image_pil)
            
            with st.expander("1. 灰度化图像", expanded=False):
                st.image(gray_img, caption="丢弃色彩", use_container_width=True)
                
            with st.expander("2. Otsu 二值化 + 形态学处理", expanded=False):
                st.image(binary_img, caption="信号变为白色亮度", use_container_width=True)
                
            with st.expander("3. 垂直投影柱状图 (切割核心)", expanded=True):
                fig, ax = plt.subplots(figsize=(8, 3))
                ax.plot(projection, color='black')
                ax.fill_between(range(len(projection)), projection, alpha=0.3)
                ax.set_title("Vertical Projection Histogram")
                ax.set_xlim(0, len(projection))
                for b in bboxes:
                    ax.axvline(x=b[0], color='green', linestyle='--', alpha=0.5)
                    ax.axvline(x=b[2], color='red', linestyle='--', alpha=0.5)
                st.pyplot(fig)

            with st.expander("4. 提取的单点序列切片", expanded=True):
                patches = extract_character_patches(gray_img, bboxes)
                if patches:
                    cols = st.columns(len(patches))
                    for i, patch in enumerate(patches):
                        cols[i].image(patch, caption=f"P{i}", width=40)
                else:
                    st.error("没有检测到有效切割区域！")

        with col_right:
            st.subheader("🎯 推理结果")
            if patches:
                img_with_boxes = image_pil.copy()
                draw = ImageDraw.Draw(img_with_boxes)
                transform = get_transform()
                predictions = []
                
                for i, bbox in enumerate(bboxes):
                    x1, y1, x2, y2 = bbox
                    draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
                    
                    tensor_input = transform(patches[i]).unsqueeze(0).to(device)
                    with torch.no_grad():
                        outputs = model(tensor_input)
                        _, predicted_class = torch.max(outputs.data, 1)
                        digit_str = str(predicted_class.item())
                        predictions.append(digit_str)
                        draw.text((x1, max(0, y1-20)), digit_str, fill="blue")

                st.image(img_with_boxes, caption="切分框定叠加", use_container_width=True)
                final_text = "".join(predictions)
                st.success("识别成功！", icon="✅")
                st.metric(label="最终提取得出", value=final_text)
            else:
                st.info("尚未识别到内容。")
