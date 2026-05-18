import os
import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn.functional as F
from PIL import Image
import json
import streamlit as st
import io, base64

# ======= Konfigurasi =======
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = 'mobilenetv3large_flowers102.pth'
topk = 1

# ======= Load label map =======
with open('label_map.json', 'r') as f:
    class_names = json.load(f)

# ======= Load Model - MobileNetV3 Large =======
model = models.mobilenet_v3_large(pretrained=False)

# Sesuaikan classifier (MobileNetV3 menggunakan 'classifier' dengan Linear di index 3)
in_features = model.classifier[3].in_features
model.classifier[3] = torch.nn.Linear(in_features, 102)

model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# ======= Transformasi gambar =======
transform_eval = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ======= Fungsi Prediksi =======
def predict(image, model, topk=5):
    image_tensor = transform_eval(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = F.softmax(output[0], dim=0)
        top_probs, top_classes = probabilities.topk(topk)
    return top_probs.cpu().numpy(), top_classes.cpu().numpy()

# ======= Convert gambar ke base64 =======
def img_to_base64(img):
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()

# ======= Konfigurasi Halaman =======
st.set_page_config(
    page_title="🌸 Klasifikasi Bunga",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ======= Custom CSS Responsif =======
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700&display=swap');

    .stApp {
        background: linear-gradient(135deg, #f8a5c2 0%, #ffe0ef 60%, #fff 100%) !important;
        font-family: 'Montserrat', sans-serif !important;
    }

    .main-header {
        background: rgba(255,255,255,0.35);
        backdrop-filter: blur(10px);
        border-radius: 30px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.15);
        padding: 2rem 1.5rem;
    }
    .main-header h1 {
        color: #f5576c;
        font-size: clamp(2rem, 5vw, 3rem);
        font-weight: 800;
    }
    .main-header p {
        color: #2c3e50;
        font-size: clamp(1rem, 3vw, 1.2rem);
    }

    .upload-section, .image-container, .result-card {
        background: rgba(255,255,255,0.7);
        backdrop-filter: blur(8px);
        padding: 1.8rem;
        border-radius: 22px;
        box-shadow: 0 10px 30px rgba(248,165,194,0.12);
    }

    .framed-image {
        border: 6px solid;
        border-image: linear-gradient(135deg, #f8a5c2 0%, #ffe0ef 100%) 1;
        border-radius: 20px;
        padding: 5px;
        background: white;
        width: 100%;
        max-height: 420px;
        object-fit: contain;
    }

    .prediction-item {
        background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.2rem;
        border-radius: 16px;
        margin: 0.8rem 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 12px;
    }

    .confidence-bar {
        background: rgba(255,255,255,0.3);
        height: 20px;
        border-radius: 10px;
        overflow: hidden;
        flex: 1;
        min-width: 140px;
    }

    .confidence-fill {
        background: linear-gradient(90deg, #f8a5c2 0%, #4ECDC4 100%);
        height: 100%;
    }

    @media (max-width: 768px) {
        .stColumns [data-testid="column"] { width: 100% !important; }
        .framed-image { max-height: 320px; }
        .upload-section, .image-container, .result-card { padding: 1.4rem 1rem; }
    }

    @media (max-width: 480px) {
        .prediction-item { flex-direction: column; align-items: flex-start; }
        .confidence-bar { width: 100%; }
    }
</style>
""", unsafe_allow_html=True)

# ======= Header =======
st.markdown("""
<div class="main-header">
    <h1>🌸 Klasifikasi Bunga 102 🌸</h1>
    <p>Unggah gambar bunga dan temukan spesiesnya menggunakan <strong>MobileNetV3 Large</strong></p>
</div>
""", unsafe_allow_html=True)

# ======= Upload Section =======
st.markdown('<div class="upload-section"><h3 style="color:#f5576c;">📁 Unggah Gambar Bunga</h3></div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📁 Unggah File", "📷 Ambil Foto"])

with tab1:
    uploaded_file = st.file_uploader("Pilih file gambar", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

with tab2:
    camera_photo = st.camera_input("Ambil foto bunga", label_visibility="collapsed")

image_source = uploaded_file if uploaded_file is not None else camera_photo

# ======= Prediksi =======
if image_source is not None:
    with st.spinner("🔍 Sedang menganalisis gambar bunga..."):
        img = Image.open(image_source).convert('RGB')
        probs, classes = predict(img, model, topk=topk)
        
        labels = [class_names.get(str(cls + 1), f'Class {cls + 1}') for cls in classes]

    st.success("✅ Analisis selesai!")

    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.markdown('<div class="image-container"><h4 style="color:#2c3e50;">📷 Gambar Anda</h4></div>', unsafe_allow_html=True)
        st.markdown(f'<img src="data:image/png;base64,{img_to_base64(img)}" class="framed-image">', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="result-card"><h3 style="color:#2c3e50;">🎯 Hasil Prediksi</h3></div>', unsafe_allow_html=True)
        
        for i in range(topk):
            confidence = float(probs[i] * 100)
            st.markdown(f"""
            <div class="prediction-item">
                <div>
                    <strong>🌸 #{i+1} {labels[i]}</strong>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width:{confidence}%"></div>
                    </div>
                </div>
                <div style="font-size:1.4rem; font-weight:bold;">{confidence:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        st.info("💡 Model MobileNetV3 Large - Trained on 102 Flower Dataset")

else:
    st.markdown("""
    <div style="text-align:center; padding:4rem 1rem; background:#f8f9fa; border-radius:20px; margin:2rem 0;">
        <div style="font-size:5.5rem; margin-bottom:1rem;">🌸</div>
        <h3>Siap mengidentifikasi bunga?</h3>
        <p style="color:#6c757d;">Unggah gambar atau ambil foto di atas untuk memulai</p>
    </div>
    """, unsafe_allow_html=True)
