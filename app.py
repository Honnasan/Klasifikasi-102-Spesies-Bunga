import os
import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn.functional as F
from PIL import Image
import json
import streamlit as st
import io
import base64

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

# ======= Konfigurasi Halaman =======
st.set_page_config(
    page_title="🌸 Klasifikasi Bunga",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ======= CSS =======
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700&display=swap');

html, body, .stApp {
    font-family: 'Montserrat', sans-serif !important;
}

/* Background lebih soft */
.stApp {
    background: linear-gradient(
        135deg,
        #d8c3d5 0%,
        #b8a9c9 50%,
        #8d99ae 100%
    ) !important;
}

/* Header */
.main-header {
    background: rgba(255,255,255,0.15);
    backdrop-filter: blur(12px);
    border-radius: 28px;
    margin-bottom: 2rem;
    text-align: center;
    padding: 2rem 1.5rem;
    box-shadow: 0 8px 24px rgba(0,0,0,0.15);
}

.main-header h1 {
    color: #2b2d42;
    font-size: clamp(2rem, 5vw, 3rem);
    font-weight: 800;
    margin: 0;
}

.main-header p {
    color: #444;
    font-size: clamp(1rem, 3vw, 1.15rem);
    margin-top: 0.5rem;
}

/* Upload Box */
.upload-section {
    margin-bottom: 1rem;
}

/* Card */
.image-container,
.result-card {
    background: rgba(255,255,255,0.2);
    backdrop-filter: blur(10px);
    border-radius: 20px;
    padding: 1.4rem;
    box-shadow: 0 6px 20px rgba(0,0,0,0.12);
}

/* Gambar */
.framed-image {
    border: 5px solid rgba(255,255,255,0.35);
    border-radius: 18px;
    padding: 4px;
    background: rgba(255,255,255,0.3);
    width: 100%;
    max-height: 420px;
    object-fit: contain;
}

/* Prediction Card */
.prediction-item {
    background: linear-gradient(
        135deg,
        #5c5470 0%,
        #6d597a 100%
    );

    color: white;

    padding: 1rem;

    border-radius: 16px;

    margin: 0.8rem 0;

    display: flex;

    justify-content: space-between;

    align-items: center;

    flex-wrap: wrap;

    gap: 12px;

    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}

/* Progress Bar */
.confidence-bar {
    background: rgba(255,255,255,0.2);

    height: 18px;

    border-radius: 10px;

    overflow: hidden;

    margin-top: 0.5rem;

    width: 100%;
}

.confidence-fill {
    background: linear-gradient(
        90deg,
        #84a59d 0%,
        #52796f 100%
    );

    height: 100%;

    transition: width 0.6s ease;
}

/* Tabs */
.stTabs [data-baseweb="tab"] {
    color: #2b2d42 !important;

    font-weight: 600;

    background: rgba(255,255,255,0.15);

    border-radius: 12px 12px 0 0;
}

.stTabs [aria-selected="true"] {
    background: rgba(255,255,255,0.3) !important;

    color: #22223b !important;
}

/* File uploader */
.stFileUploader,
.stCameraInput {
    width: 100%;
}

/* Floating animation */
@keyframes float {
    0% {
        transform: translateY(0px);
    }

    50% {
        transform: translateY(-10px);
    }

    100% {
        transform: translateY(0px);
    }
}

/* Mobile */
@media (max-width: 768px) {

    .stColumns [data-testid="column"] {
        width: 100% !important;
        flex: 1 1 100% !important;
    }

    .framed-image {
        max-height: 320px;
    }
}

@media (max-width: 480px) {

    .prediction-item {
        flex-direction: column;
        align-items: flex-start;
    }
}

</style>
""", unsafe_allow_html=True)

# ======= Header =======
st.markdown("""
<div class="main-header">
    <h1>🌸 Klasifikasi Bunga 102 🌸</h1>

    <p>
        Unggah gambar bunga dan temukan spesiesnya
        menggunakan MobileNetV3
    </p>
</div>
""", unsafe_allow_html=True)

# ======= Upload Section =======
st.markdown("""
<div class="upload-section"></div>
""", unsafe_allow_html=True)

# ======= Tabs =======
tab1, tab2 = st.tabs([
    "📁 Unggah File",
    "📷 Ambil Foto"
])

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
image_source = (
    uploaded_file
    if uploaded_file is not None
    else camera_photo
)

# ======= Prediksi =======
if image_source is not None:

    with st.spinner("🔍 Sedang menganalisis gambar bunga..."):

        img = Image.open(image_source).convert('RGB')

        probs, classes = predict(img, model, topk=topk)

        labels = [
            class_names.get(
                str(cls + 1),
                f'class_{cls + 1}'
            )
            for cls in classes
        ]

    st.success("✅ Analisis selesai!")

    col1, col2 = st.columns([1, 1.2])

    # ======= Kolom Gambar =======
    with col1:

        st.markdown("""
        <div class="image-container">
            <h4 style="color:#2b2d42;">
                📷 Gambar Anda
            </h4>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(
            f'''
            <img
                src="data:image/png;base64,{img_to_base64(img)}"
                class="framed-image"
            >
            ''',
            unsafe_allow_html=True
        )

        if hasattr(image_source, 'name'):

            st.caption(f"📄 {image_source.name}")

    # ======= Kolom Hasil =======
    with col2:

        st.markdown("""
        <div class="result-card">
            <h3 style="color:#2b2d42;">
                🎯 Hasil Prediksi
            </h3>
        </div>
        """, unsafe_allow_html=True)

        for i in range(topk):

            confidence = float(probs[i] * 100)

            st.markdown(f"""
            <div class="prediction-item">

                <div style="flex:1;">

                    <strong>
                        🌸 #{i+1} {labels[i]}
                    </strong>

                    <div class="confidence-bar">

                        <div
                            class="confidence-fill"
                            style="width:{confidence}%">
                        </div>

                    </div>

                </div>

                <div style="
                    font-size:1.2rem;
                    font-weight:bold;
                    min-width:70px;
                    text-align:right;
                ">
                    {confidence:.1f}%
                </div>

            </div>
            """, unsafe_allow_html=True)

        st.info(
            "💡 Model menampilkan spesies bunga "
            "dengan probabilitas tertinggi."
        )

# ======= Placeholder =======
else:

    st.markdown("""
    <div style="
        text-align:center;
        padding:5rem 2rem;
        background:rgba(255,255,255,0.12);
        border-radius:28px;
        margin:2rem 0;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.12);
    ">

        <div style="
            font-size:6rem;
            margin-bottom:1rem;
            animation: float 3s ease-in-out infinite;
        ">
            🌸
        </div>

        <h2 style="
            color:#2b2d42;
            margin-bottom:0.7rem;
            font-weight:700;
        ">
            Selamat Datang 🌷
        </h2>

        <p style="
            color:#555;
            font-size:1.1rem;
            max-width:500px;
            margin:auto;
            line-height:1.7;
        ">
            Unggah gambar bunga atau ambil foto secara langsung
            untuk mulai mengidentifikasi spesies bunga menggunakan
            model AI MobileNetV3.
        </p>

    </div>
    """, unsafe_allow_html=True)
