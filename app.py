import os
import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn.functional as F
from PIL import Image
import json
import streamlit as st
import io, base64

st.markdown('<meta name="viewport" content="width=device-width, initial-scale=1.0">', unsafe_allow_html=True)
if "is_desktop" not in st.session_state:
    st.session_state.is_desktop = True  # default
    

# ======= Konfigurasi =======
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = 'mobilenetv3large_flowers102.pth'
topk = 1

# ======= Load Model & Label =======
with open('label_map.json', 'r') as f:
    class_names = json.load(f)

model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = torch.nn.Linear(model.last_channel, 102)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

transform_eval = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict(image, model, topk=5):
    image_tensor = transform_eval(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = F.softmax(output[0], dim=0)
        top_probs, top_classes = probabilities.topk(topk)
    return top_probs.cpu().numpy(), top_classes.cpu().numpy()

def img_to_base64(img):
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()

# ======= Page Config =======
st.set_page_config(
    page_title="🌸 Klasifikasi Bunga 102",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ======= Dark Mode Toggle di Sidebar =======
with st.sidebar:
    st.markdown("### 🎨 Tema")
    dark_mode = st.toggle("🌙 Dark Mode", value=False, key="dark_toggle")

# ======= Custom CSS Mewah + Animasi + Dark Mode =======
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Montserrat:wght@400;500;600;700&display=swap');

    html, body, .stApp {{
        font-family: 'Montserrat', sans-serif;
        transition: all 0.4s ease;
    }}

    .stApp {{
        background: { "linear-gradient(135deg, #1a0f2e 0%, #2d1b4e 100%)" if dark_mode else "linear-gradient(135deg, #f8a5c2 0%, #ffe0ef 50%, #fff 100%)" } !important;
    }}

    .main-header {{
        background: rgba(255,255,255,0.15);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 30px;
        padding: 2.5rem 2rem;
        text-align: center;
        box-shadow: 0 15px 35px rgba(0,0,0,0.2);
        animation: fadeIn 1.2s ease;
    }}

    .main-header h1 {{
        font-family: 'Playfair Display', serif;
        color: { "#e0b3ff" if dark_mode else "#f5576c" };
        font-size: clamp(2.4rem, 6vw, 3.8rem);
        font-weight: 700;
        text-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }}

    .upload-section {{
        background: rgba(255,255,255,0.12);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.18);
        padding: 2rem;
        border-radius: 28px;
        animation: fadeInUp 0.8s ease;
    }}

    .image-container, .result-card {{
        background: rgba(255,255,255,0.15);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 24px;
        padding: 1.8rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }}

    .image-container:hover, .result-card:hover {{
        transform: translateY(-8px);
        box-shadow: 0 20px 40px rgba(0,0,0,0.25);
    }}

    .framed-image {{
        border-radius: 22px;
        border: 7px solid transparent;
        background: linear-gradient(white, white) padding-box,
                    linear-gradient(135deg, #f8a5c2, #c445a8) border-box;
        transition: all 0.4s ease;
    }}

    .framed-image:hover {{
        transform: scale(1.03);
    }}

    .prediction-item {{
        background: linear-gradient(90deg, #c445a8, #f5576c);
        color: white;
        padding: 1.3rem;
        border-radius: 18px;
        margin: 1rem 0;
        display: flex;
        align-items: center;
        gap: 15px;
        transition: all 0.4s ease;
        animation: slideIn 0.6s ease forwards;
    }}

    .prediction-item:hover {{
        transform: scale(1.03);
        box-shadow: 0 10px 25px rgba(244, 87, 108, 0.4);
    }}

    .confidence-bar {{
        height: 22px;
        background: rgba(255,255,255,0.25);
        border-radius: 12px;
        overflow: hidden;
    }}

    .confidence-fill {{
        height: 100%;
        background: linear-gradient(90deg, #ffd700, #ff8c00);
        transition: width 1.2s cubic-bezier(0.34, 1.56, 0.64, 1);
    }}

    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(20px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}

    @keyframes fadeInUp {{
        from {{ opacity: 0; transform: translateY(40px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}

    @keyframes slideIn {{
        from {{ opacity: 0; transform: translateX(-30px); }}
        to {{ opacity: 1; transform: translateX(0); }}
    }}

    .stTabs [data-baseweb="tab"] {{
        border-radius: 16px;
        transition: all 0.3s ease;
    }}

    .stTabs [aria-selected="true"] {{
        background: { "#6b3a8e" if dark_mode else "#ffe0ef" } !important;
        color: { "#e0b3ff" if dark_mode else "#f5576c" } !important;
    }}
</style>
""", unsafe_allow_html=True)

# ======= Header =======
st.markdown("""
<div class="main-header">
    <h1>🌸 Klasifikasi Bunga 102 🌸</h1>
    <p style="color:#ddd; font-size:1.25rem; margin-top:0.8rem;">
        Identifikasi 102 spesies bunga dengan kecerdasan buatan
    </p>
</div>
""", unsafe_allow_html=True)

# ======= Upload Section =======
st.markdown('<div class="upload-section">', unsafe_allow_html=True)
st.markdown(f"<h3 style='color:{"#e0b3ff" if dark_mode else "#f5576c"}; text-align:center;'>📸 Unggah atau Ambil Foto Bunga</h3>", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📁 Unggah File", "📷 Kamera"])

with tab1:
    uploaded_file = st.file_uploader("Pilih gambar", type=["jpg","jpeg","png"], label_visibility="collapsed")

with tab2:
    camera_photo = st.camera_input("Ambil foto", label_visibility="collapsed")

image_source = uploaded_file if uploaded_file is not None else camera_photo

# ======= Prediksi =======
if image_source is not None:
    with st.spinner("🌺 Sedang menganalisis keindahan bunga..."):
        img = Image.open(image_source).convert('RGB')
        probs, classes = predict(img, model, topk=topk)
        labels = [class_names.get(str(cls + 1), f'Unknown') for cls in classes]

    st.success("🎉 Analisis selesai!", icon="✨")

    col1, col2 = st.columns([1, 1.3])

    with col1:
        st.markdown('<div class="image-container"><h4 style="color:#ddd;">📷 Gambar Anda</h4></div>', unsafe_allow_html=True)
        st.markdown(
            f'<img src="data:image/png;base64,{img_to_base64(img)}" class="framed-image">',
            unsafe_allow_html=True
        )

    with col2:
        st.markdown('<div class="result-card"><h3 style="color:#ddd;">🌟 Hasil Prediksi</h3></div>', unsafe_allow_html=True)
        
        for i in range(topk):
            confidence = float(probs[i] * 100)
            st.markdown(f"""
            <div class="prediction-item">
                <div style="flex:1;">
                    <strong>#{i+1} {labels[i]}</strong>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width:{confidence}%"></div>
                    </div>
                </div>
                <div style="font-size:1.6rem; font-weight:700; min-width:85px;">
                    {confidence:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.caption("Model: MobileNetV2 • Akurasi tinggi pada dataset Oxford Flowers 102")

else:
    st.markdown("""
    <div style="text-align:center; padding:5rem 1rem; border-radius:30px; background:rgba(255,255,255,0.08); margin:3rem 0;">
        <div style="font-size:6.5rem; margin-bottom:1rem; opacity:0.9;">🌸</div>
        <h2 style="color:#ddd;">Siap menemukan keindahan bunga?</h2>
        <p style="color:#bbb; font-size:1.1rem;">Unggah gambar atau ambil foto di atas</p>
    </div>
    """, unsafe_allow_html=True)
