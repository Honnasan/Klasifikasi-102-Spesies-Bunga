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

# ======= Load label map =======
with open('label_map.json', 'r') as f:
    class_names = json.load(f)

# ======= Load Model =======
@st.cache_resource
def load_model():
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = torch.nn.Linear(model.last_channel, 102)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

model = load_model()

# ======= Transform =======
transform_eval = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict(image, model, topk=1):
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
    page_title="🌸 Luxe Bloom | Klasifikasi Bunga",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ======= Custom CSS Mewah =======
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Montserrat:wght@400;500;600&display=swap');

    .stApp {
        background: linear-gradient(135deg, #fff0f5 0%, #f8e1f0 40%, #e0f2fe 100%);
    }

    .main-header {
        background: linear-gradient(135deg, rgba(255,255,255,0.95), rgba(248,240,255,0.85));
        backdrop-filter: blur(12px);
        border-radius: 30px;
        padding: 3rem 2rem;
        text-align: center;
        box-shadow: 0 20px 60px rgba(245, 87, 108, 0.15);
        border: 1px solid rgba(255,255,255,0.6);
        margin-bottom: 2.5rem;
    }

    .main-header h1 {
        font-family: 'Playfair Display', serif;
        font-size: clamp(2.8rem, 6vw, 4.2rem);
        background: linear-gradient(90deg, #f5576c, #c44569);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        letter-spacing: -2px;
    }

    .main-header p {
        font-size: 1.25rem;
        color: #5f4a6f;
        margin-top: 1rem;
    }

    .glass-card {
        background: rgba(255, 255, 255, 0.75);
        backdrop-filter: blur(16px);
        border-radius: 28px;
        border: 1px solid rgba(255,255,255,0.6);
        box-shadow: 0 15px 35px rgba(245, 87, 108, 0.12);
        padding: 2rem;
    }

    .framed-image {
        border-radius: 24px;
        overflow: hidden;
        border: 8px solid white;
        box-shadow: 0 15px 40px rgba(0,0,0,0.15);
        transition: transform 0.4s ease;
    }
    .framed-image:hover {
        transform: scale(1.03);
    }

    .prediction-item {
        background: linear-gradient(135deg, #ff6b9d, #c44569);
        color: white;
        padding: 1.4rem 1.8rem;
        border-radius: 20px;
        margin: 1rem 0;
        box-shadow: 0 10px 25px rgba(196, 69, 105, 0.25);
        transition: all 0.3s ease;
    }
    .prediction-item:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 30px rgba(196, 69, 105, 0.35);
    }

    .confidence-bar {
        height: 22px;
        background: rgba(255,255,255,0.25);
        border-radius: 50px;
        overflow: hidden;
    }
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #ffd93d, #ff6b9d);
        box-shadow: 0 0 15px rgba(255, 217, 61, 0.6);
        transition: width 1s cubic-bezier(0.34, 1.56, 0.64, 1);
    }

    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
        border-radius: 16px;
        padding: 12px 24px;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #f5576c, #c44569) !important;
        color: white !important;
    }

    /* Animasi */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .animate {
        animation: fadeInUp 0.8s ease forwards;
    }
</style>
""", unsafe_allow_html=True)

# ======= Header Elegan =======
st.markdown("""
<div class="main-header">
    <h1>🌸 Luxe Bloom</h1>
    <p>Identifikasi bunga dengan kecanggihan AI • Desain premium</p>
</div>
""", unsafe_allow_html=True)

# ======= Upload Section =======
st.markdown('<div class="glass-card" style="text-align:center; margin-bottom:2rem;">', unsafe_allow_html=True)
st.markdown("<h3 style='color:#c44569; margin-bottom:0.5rem;'>📸 Unggah Gambar Bunga Anda</h3>", unsafe_allow_html=True)
st.markdown("<p style='color:#6b5b7e;'>Pilih foto atau ambil langsung dari kamera</p>", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📁 Unggah dari Galeri", "📷 Ambil Foto Kamera"])

with tab1:
    uploaded_file = st.file_uploader("Pilih gambar", type=["jpg","jpeg","png"], label_visibility="collapsed")

with tab2:
    camera_photo = st.camera_input("Ambil foto", label_visibility="collapsed")

image_source = uploaded_file if uploaded_file is not None else camera_photo

# ======= Prediksi =======
if image_source is not None:
    with st.spinner("🌺 Sedang menganalisis dengan kecermatan tinggi..."):
        img = Image.open(image_source).convert('RGB')
        probs, classes = predict(img, model, topk=topk)
        labels = [class_names.get(str(cls + 1), f'Unknown') for cls in classes]

    st.success("🎉 Analisis selesai dengan hasil yang indah!", icon="✨")

    col1, col2 = st.columns([1, 1.1])

    with col1:
        st.markdown('<div class="glass-card animate">', unsafe_allow_html=True)
        st.markdown("<h4 style='color:#5f4a6f; text-align:center;'>📷 Gambar Anda</h4>", unsafe_allow_html=True)
        st.markdown(
            f'<img src="data:image/png;base64,{img_to_base64(img)}" class="framed-image" width="100%">',
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-card animate">', unsafe_allow_html=True)
        st.markdown("<h3 style='color:#c44569; text-align:center;'>🌸 Hasil Prediksi Premium</h3>", unsafe_allow_html=True)

        for i in range(topk):
            confidence = float(probs[i] * 100)
            st.markdown(f"""
            <div class="prediction-item">
                <div>
                    <strong style="font-size:1.3rem;">#{i+1} {labels[i]}</strong>
                    <div class="confidence-bar" style="margin-top:12px;">
                        <div class="confidence-fill" style="width:{confidence}%"></div>
                    </div>
                </div>
                <div style="font-size:2rem; font-weight:700; margin-left:15px;">
                    {confidence:.1f}<span style="font-size:1rem;">%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div style="background:rgba(255,255,255,0.6); padding:1.2rem; border-radius:18px; margin-top:1.5rem; font-size:0.95rem;">
            💎 Model ini dilatih khusus untuk mengenali 102 spesies bunga dengan akurasi tinggi.
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="text-align:center; padding:5rem 1rem; background:rgba(255,255,255,0.7); border-radius:30px; margin:3rem 0;">
        <div style="font-size:6.5rem; margin-bottom:1rem; opacity:0.9;">🌸</div>
        <h2 style="color:#c44569;">Siap menjelajahi keindahan bunga?</h2>
        <p style="color:#6b5b7e; font-size:1.1rem;">Unggah atau ambil foto bunga favorit Anda</p>
    </div>
    """, unsafe_allow_html=True)
