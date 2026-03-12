import streamlit as st

st.set_page_config(page_title="OCR Learning Platform", layout="wide")

# 侧边栏导航隔离层
st.sidebar.title("🧭 导航矩阵")
st.sidebar.markdown("探索端到端序列建模的进化历程")

pages = {
    "V1 经典切分架构 (传统CV + CNN)": "v1_traditional",
    "V2 端到端序列建模 (CRNN + CTC)": "v2_crnn_ctc",
    "V3 隐式语言模型 (Seq2Seq + Attention)": "v3_seq2seq_attn",
    "V4 混合强架构 (CNN+Transformer+CTC/Attn)": "v4_transformer_joint",
    "V5 双模式极境 (纯 ViT-OCR/TrOCR)": "v5_vit_ocr"
}

selection = st.sidebar.radio("选择进入的演化时代", list(pages.keys()))

st.sidebar.divider()

if selection == "V1 经典切分架构 (传统CV + CNN)":
    from pages.v1_traditional import render_v1_ui
    render_v1_ui()

elif selection == "V2 端到端序列建模 (CRNN + CTC)":
    # 动态导入已被剥离隔离的子组件页面渲染包
    from pages.v2_crnn_ctc import render_v2_ui
    render_v2_ui()

elif selection == "V3 隐式语言模型 (Seq2Seq + Attention)":
    from pages.v3_seq2seq_attn import render_v3_ui
    render_v3_ui()

elif selection == "V4 混合强架构 (CNN+Transformer+CTC/Attn)":
    from pages.v4_transformer_joint import render_v4_ui
    render_v4_ui()

elif selection == "V5 双模式极境 (纯 ViT-OCR/TrOCR)":
    from pages.v5_vit_ocr import render_v5_ui
    render_v5_ui()

else:
    st.subheader(f"{selection}")
    st.info("架构演进中，敬请期待...")
