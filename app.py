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

# ======= Load Model =======
model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = torch.nn.Linear(model.last_channel, 102)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# ======= Transformasi =======
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

# ======= Convert ke base64 =======
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

# ======= Custom CSS - Green Theme + Responsif =======
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700&display=swap');

    html, body, .stApp {
        font-family: 'Montserrat', sans-serif !important;
        background: linear-gradient(135deg, #a8e6cf 0%, #d0f4e6 50%, #f0f9f4 100%) !important;
    }

    .main-header {
        background: rgba(255,255,255,0.75);
        backdrop-filter: blur(12px);
        border-radius: 30px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        padding: 2rem 1.5rem;
    }

    .main-header h1 {
        color: #2e8b57;
        font-size: clamp(2rem, 5vw, 3rem);
        font-weight: 800;
        margin: 0;
    }

    .main-header p {
        color: #1e5f4a;
        font-size: clamp(1rem, 3vw, 1.2rem);
    }

    .image-container, .result-card {
        background: rgba(255,255,255,0.85);
        backdrop-filter: blur(8px);
        padding: 1.5rem;
        border-radius: 20px;
        box-shadow: 0 5px 20px rgba(46,139,87,0.1);
    }

    .framed-image {
        border: 6px solid;
        border-image: linear-gradient(135deg, #2e8b57 0%, #66cdaa 100%) 1;
        border-radius: 20px;
        padding: 4px;
        background: white;
        width: 100%;
        max-height: 420px;
        object-fit: contain;
    }

    .prediction-item {
        background: linear-gradient(90deg, #2e8b57 0%, #66cdaa 100%);
        color: white;
        padding: 1.1rem;
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
        min-width: 120px;
    }

    .confidence-fill {
        background: linear-gradient(90deg, #a8e6cf 0%, #ffffff 100%);
        height: 100%;
    }

    /* Responsive Design */
    @media (max-width: 768px) {
        .stColumns [data-testid="column"] {
            width: 100% !important;
        }
        .framed-image {
            max-height: 320px;
        }
        .main-header, .image-container, .result-card {
            padding: 1.5rem 1rem;
        }
    }

    @media (max-width: 480px) {
        .prediction-item {
            flex-direction: column;
            align-items: flex-start;
        }
        .confidence-bar {
            width: 100%;
        }
    }
</style>
""", unsafe_allow_html=True)

# ======= Header =======
st.markdown("""
<div class="main-header">
    <h1>🌸 Klasifikasi Bunga 102 🌸</h1>
    <p>Unggah gambar bunga dan temukan spesiesnya</p>
</div>
""", unsafe_allow_html=True)

# ======= Tabs Upload =======
tab1, tab2 = st.tabs(["📁 Unggah File", "📷 Ambil Foto"])

with tab1:
    uploaded_file = st.file_uploader(
        "Pilih file gambar (jpg, jpeg, png)",
        type=["jpg", "jpeg", "png"],
        label_visibility="visible"
    )

with tab2:
    camera_photo = st.camera_input(
        "Ambil foto bunga",
        label_visibility="visible"
    )

# ======= Pilih sumber gambar =======
image_source = uploaded_file if uploaded_file is not None else camera_photo

# ======= Prediksi =======
if image_source is not None:
    with st.spinner("🔍 Sedang menganalisis gambar bunga..."):
        img = Image.open(image_source).convert('RGB')
        probs, classes = predict(img, model, topk=topk)
        labels = [class_names.get(str(cls + 1), f'class_{cls + 1}') for cls in classes]

    st.success("✅ Analisis selesai!")

    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.markdown('<div class="image-container"><h4 style="color:#2c3e50;">📷 Gambar Anda</h4></div>', unsafe_allow_html=True)
        st.markdown(
            f'<img src="data:image/png;base64,{img_to_base64(img)}" class="framed-image">',
            unsafe_allow_html=True
        )

    with col2:
        st.markdown('<div class="result-card"><h3 style="color:#2c3e50;">🎯 Hasil Prediksi</h3></div>', unsafe_allow_html=True)
        
        for i in range(topk):
            confidence = float(probs[i] * 100)
            st.markdown(f"""
            <div class="prediction-item">
                <div style="flex:1;">
                    <strong>🌸 #{i+1} {labels[i]}</strong>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width:{confidence}%"></div>
                    </div>
                </div>
                <div style="font-size:1.35rem; font-weight:bold;">
                    {confidence:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="text-align:center; padding:5rem 1rem; background:#f8f9fa; border-radius:20px; margin:2rem 0;">
        <div style="font-size:6rem; margin-bottom:1rem; opacity:0.7;">🌸</div>
        <h3 style="color:#2e8b57;">Siap mengidentifikasi bunga?</h3>
        <p style="color:#1e5f4a;">Silakan unggah gambar atau ambil foto di atas</p>
    </div>
    """, unsafe_allow_html=True)
