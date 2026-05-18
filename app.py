import os
import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn.functional as F
from PIL import Image
import numpy as np
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

# ======= Load Model =======
model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = torch.nn.Linear(model.last_channel, 102)
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
    byte_im = buf.getvalue()
    return base64.b64encode(byte_im).decode()

# ======= Konfigurasi halaman =======
st.set_page_config(
    page_title="🌸 Klasifikasi Bunga",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ======= Custom CSS =======
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap');

    html, body, .stApp {
        font-family: 'Montserrat', sans-serif !important;
        background: linear-gradient(135deg, #f8a5c2 0%, #ffe0ef 60%, #fff 100%) !important;
    }

    .main-header {
        background: rgba(255,255,255,0.35);
        backdrop-filter: blur(8px);
        border-radius: 30px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.15);
        padding: 2.5rem;
    }

    .main-header h1 {
        color: #f5576c;
        font-size: 3rem;
        font-weight: 800;
    }

    .main-header p {
        color: #2c3e50;
        font-size: 1.2rem;
    }

    .upload-section {
        background: rgba(255,255,255,0.55);
        backdrop-filter: blur(6px);
        padding: 2rem;
        border-radius: 22px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(248,165,194,0.10);
    }

    .upload-section h3 {
        color: #f5576c;
        font-size: 1.5rem;
    }

    .image-container, .result-card {
        background: rgba(255,255,255,0.7);
        backdrop-filter: blur(4px);
        padding: 1.5rem;
        border-radius: 20px;
        box-shadow: 0 5px 18px rgba(248,165,194,0.13);
        margin-bottom: 1rem;
    }

    .framed-image {
        border: 5px solid;
        border-image: linear-gradient(135deg, #f8a5c2 0%, #ffe0ef 100%) 1;
        border-radius: 20px;
        padding: 4px;
        background: white;
        width: 100%;
    }

    .file-badge {
        display: inline-block;
        background: linear-gradient(90deg, #f8a5c2 0%, #ffe0ef 100%);
        color: #f5576c;
        font-weight: bold;
        border-radius: 12px;
        padding: 0.3rem 1rem;
        margin-top: 0.7rem;
    }

    .prediction-item {
        background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem;
        border-radius: 14px;
        margin: 0.7rem 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .confidence-bar {
        background: rgba(255,255,255,0.25);
        height: 18px;
        border-radius: 9px;
        overflow: hidden;
        margin-top: 0.5rem;
        width: 140px;
    }

    .confidence-fill {
        background: linear-gradient(90deg, #f8a5c2 0%, #4ECDC4 100%);
        height: 100%;
    }

    .stTabs [data-baseweb="tab"] {
        color: #2c3e50 !important;
        font-weight: 600;
        border-radius: 14px 14px 0 0;
    }

    .stTabs [aria-selected="true"] {
        background: #ffe0ef !important;
        color: #f5576c !important;
    }
</style>
""", unsafe_allow_html=True)

# ======= Header =======
st.markdown("""
<div class="main-header">
    <h1>🌸 Klasifikasi Bunga 102 🌸</h1>
    <p>Unggah gambar bunga dan temukan spesiesnya menggunakan model MobileNetV3</p>
</div>
""", unsafe_allow_html=True)

# ======= Upload Section =======
st.markdown("""
<div class="upload-section">
    <h3>📁 Unggah Gambar Bunga</h3>
    <p style="color: #2c3e50;">
        Pilih gambar dari file atau ambil foto menggunakan kamera
    </p>
</div>
""", unsafe_allow_html=True)

# ======= Tabs Upload =======
tab1, tab2 = st.tabs(["📁 Unggah File", "📷 Ambil Foto"])

with tab1:
    uploaded_file = st.file_uploader(
        "Pilih file gambar",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

with tab2:
    camera_photo = st.camera_input(
        "Ambil foto bunga",
        label_visibility="collapsed"
    )

# ======= Pilih sumber gambar =======
if uploaded_file is not None:
    image_source = uploaded_file
elif camera_photo is not None:
    image_source = camera_photo
else:
    image_source = None

# ======= Prediksi =======
if image_source is not None:

    with st.spinner("🔍 Sedang menganalisis gambar bunga..."):

        img = Image.open(image_source).convert('RGB')

        probs, classes = predict(img, model, topk=topk)

        labels = [
            class_names.get(str(cls + 1), f'class_{cls + 1}')
            for cls in classes
        ]

    st.success("✅ Analisis selesai! Berikut hasil prediksinya:")

    col1, col2 = st.columns([1, 1.5])

    # ======= Kolom gambar =======
    with col1:

        st.markdown("""
        <div class="image-container">
            <h4 style="color: #2c3e50;">📷 Gambar Anda</h4>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(
            f'<img src="data:image/png;base64,{img_to_base64(img)}" class="framed-image">',
            unsafe_allow_html=True
        )

        if hasattr(image_source, 'name'):
            st.markdown(
                f'<div class="file-badge">{image_source.name}</div>',
                unsafe_allow_html=True
            )

    # ======= Kolom hasil =======
    with col2:

        st.markdown("""
        <div class="result-card">
            <h3 style="color: #2c3e50;">🎯 Hasil Prediksi</h3>
        </div>
        """, unsafe_allow_html=True)

        for i in range(topk):

            confidence = probs[i] * 100

            st.markdown(f"""
            <div class="prediction-item">
                <div>
                    <strong>🌸 #{i+1} {labels[i]}</strong>

                    <div class="confidence-bar">
                        <div class="confidence-fill"
                             style="width:{confidence}%">
                        </div>
                    </div>
                </div>

                <div style="font-size:1.2rem;font-weight:bold;">
                    {confidence:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div style="
            background:#f8f9fa;
            padding:1rem;
            border-radius:10px;
            margin-top:1rem;
        ">
            <p style="
                margin:0;
                color:#6c757d;
                font-size:0.9rem;
            ">
                💡 <strong>Tips:</strong>
                Model menampilkan spesies bunga yang paling mungkin
                berdasarkan gambar yang diunggah.
            </p>
        </div>
        """, unsafe_allow_html=True)

# ======= Placeholder =======
else:

    st.markdown("""
<div style="text-align: center; padding: 3rem; background: #f8f9fa; border-radius: 15px; margin: 2rem 0;">
    <div style="font-size: 4rem; margin-bottom: 1rem;">🌸</div>
    <h3 style="color: #2c3e50;">Siap mengidentifikasi bunga?</h3>
    <p style="color: #6c757d;">Unggah gambar bunga di atas untuk memulai!</p>
</div>
""", unsafe_allow_html=True)
