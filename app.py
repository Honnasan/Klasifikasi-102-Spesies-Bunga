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
model_path = 'mobilenetv2_flowers102.pth'
topk = 1

# ======= Load label map (misalnya dari label_map.json) =======
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

# Helper untuk badge stylish
from PIL import ImageDraw, ImageFont

def img_to_base64(img):
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    byte_im = buf.getvalue()
    return base64.b64encode(byte_im).decode()

# ======= Custom CSS Styling =======
st.set_page_config(
    page_title="🌸 Flower Classifier",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS untuk styling yang lebih menarik
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap');
    html, body, .stApp {
        font-family: 'Montserrat', 'Segoe UI', Arial, sans-serif !important;
        background: linear-gradient(135deg, #f8a5c2 0%, #ffe0ef 60%, #fff 100%) !important;
    }
    /* Glassmorphism Header */
    .main-header {
        background: rgba(255,255,255,0.35);
        backdrop-filter: blur(8px);
        border-radius: 30px;
        margin-bottom: 2.5rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.15);
        position: relative;
        padding: 2.5rem 2rem 2rem 2rem;
        animation: fadeInDown 1.2s;
    }
    .main-header h1 {
        color: #f5576c;
        font-size: 3.2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: 2px;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 1rem;
    }
    .main-header h1 .logo {
        font-size: 2.5rem;
        margin-right: 0.5rem;
        filter: drop-shadow(0 2px 8px #f8a5c2);
    }
    .main-header p {
        color: #2c3e50;
        font-size: 1.25rem;
        margin: 1rem 0 0 0;
        opacity: 0.92;
        font-weight: 500;
        animation: fadeIn 2s;
    }
    @keyframes fadeInDown {
        0% { opacity: 0; transform: translateY(-40px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeIn {
        0% { opacity: 0; }
        100% { opacity: 1; }
    }
    /* Glass Card Upload */
    .upload-section {
        background: rgba(255,255,255,0.55);
        backdrop-filter: blur(6px);
        padding: 2.2rem 2rem 1.5rem 2rem;
        border-radius: 22px;
        text-align: center;
        margin-bottom: 2.2rem;
        box-shadow: 0 10px 30px rgba(248,165,194,0.10);
        border: 1.5px solid #ffe0ef;
        transition: box-shadow 0.3s;
        position: relative;
    }
    .upload-section:hover {
        box-shadow: 0 16px 40px rgba(245,87,108,0.13);
    }
    .upload-section h3 {
        color: #f5576c;
        font-size: 1.6rem;
        margin-bottom: 1rem;
        font-weight: 700;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
    }
    .upload-section h3 .icon {
        font-size: 1.5rem;
        filter: drop-shadow(0 2px 8px #f8a5c2);
    }
    /* Card effect */
    .image-container, .result-card {
        background: rgba(255,255,255,0.7);
        backdrop-filter: blur(4px);
        padding: 2rem 1.5rem;
        border-radius: 20px;
        box-shadow: 0 5px 18px rgba(248,165,194,0.13);
        margin-bottom: 1.2rem;
        border-left: 5px solid #f8a5c2;
        transition: box-shadow 0.3s;
    }
    .result-card {
        border-left: 5px solid #4ECDC4;
    }
    /* Frame gambar output dengan gradient border */
    .framed-image {
        border: 5px solid;
        border-image: linear-gradient(135deg, #f8a5c2 0%, #ffe0ef 100%) 1;
        border-radius: 22px;
        box-shadow: 0 4px 18px rgba(248,165,194,0.18);
        padding: 4px;
        background: #fff;
        margin-bottom: 0.7rem;
        max-width: 100%;
        height: auto;
        display: block;
        transition: transform 0.3s, box-shadow 0.3s;
    }
    .framed-image:hover {
        transform: scale(1.04);
        box-shadow: 0 8px 32px rgba(245,87,108,0.18);
    }
    /* Badge nama file */
    .file-badge {
        display: inline-block;
        background: linear-gradient(90deg, #f8a5c2 0%, #ffe0ef 100%);
        color: #f5576c;
        font-weight: 700;
        border-radius: 12px;
        padding: 0.3rem 1rem;
        font-size: 1rem;
        margin-bottom: 0.7rem;
        box-shadow: 0 2px 8px rgba(248,165,194,0.10);
    }
    /* Prediction item */
    .prediction-item {
        background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.1rem 1.2rem;
        border-radius: 14px;
        margin: 0.7rem 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        transition: box-shadow 0.3s, transform 0.3s;
        font-size: 1.15rem;
    }
    .prediction-item .flower-icon {
        font-size: 1.3rem;
        margin-right: 0.5rem;
        filter: drop-shadow(0 2px 8px #f8a5c2);
    }
    .prediction-item:hover {
        transform: translateY(-2px) scale(1.02);
        box-shadow: 0 8px 24px rgba(245,87,108,0.18);
    }
    .confidence-bar {
        background: rgba(255,255,255,0.25);
        height: 18px;
        border-radius: 9px;
        overflow: hidden;
        margin-top: 0.5rem;
        width: 140px;
        box-shadow: 0 2px 8px rgba(248,165,194,0.10);
    }
    .confidence-fill {
        background: linear-gradient(90deg, #f8a5c2 0%, #4ECDC4 100%);
        height: 100%;
        border-radius: 9px;
        transition: width 0.7s cubic-bezier(.4,2,.6,1);
    }
    /* Tombol Clear */
    .stButton > button#clear_btn {
        background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.9rem 2.4rem;
        font-weight: bold;
        font-size: 1.15rem;
        box-shadow: 0 4px 15px rgba(245,87,108,0.13);
        margin-top: 1.5rem;
        transition: all 0.3s;
        display: flex;
        align-items: center;
        gap: 0.7rem;
    }
    .stButton > button#clear_btn:hover {
        background: linear-gradient(90deg, #f5576c 0%, #f093fb 100%);
        transform: scale(1.06);
        box-shadow: 0 8px 24px rgba(245,87,108,0.18);
    }
    /* Tab upload */
    .stTabs [data-baseweb="tab"] {
        color: #2c3e50 !important;
        font-weight: 600;
        background: #f4f6fa !important;
        border-radius: 14px 14px 0 0;
        margin-right: 4px;
        padding: 0.8rem 2.2rem;
        font-size: 1.13rem;
        box-shadow: 0 2px 8px rgba(248,165,194,0.08);
        border-bottom: 2px solid #ffe0ef;
    }
    .stTabs [aria-selected="true"] {
        background: #ffe0ef !important;
        color: #f5576c !important;
        border-bottom: 3px solid #f8a5c2 !important;
        box-shadow: 0 4px 16px rgba(248,165,194,0.13);
    }
    /* Nama file dan ukuran file */
    .stFileUploader label,
    .stFileUploader .uploadedFileName,
    .stFileUploader .uploadedFileSize,
    .stFileUploader div[data-testid="stFileUploaderDropzone"] span,
    .stFileUploader div[data-testid="stFileUploaderDetails"] span,
    .stFileUploader div {
        color: #2c3e50 !important;
        font-weight: 600 !important;
    }
    /* Pesan sukses */
    .stAlert-success, .stAlert-success div {
        background-color: #eafaf1 !important;
        color: #18603a !important;
        border-left: 5px solid #4ECDC4 !important;
        font-weight: 700 !important;
    }
    .stAlert-success .stMarkdown, .stAlert-success div[data-testid="stAlertMessage"],
    .stAlert-success [data-testid="stMarkdownContainer"],
    .stAlert-success p,
    .stAlert-success span {
        color: #1a237e !important;
        font-weight: 700 !important;
    }
    /* Footer */
    .footer {
        background: rgba(44,62,80,0.65);
        color: #fff;
        padding: 1.2rem 1rem 0.7rem 1rem;
        border-radius: 14px;
        text-align: center;
        margin-top: 2.5rem;
        font-size: 0.97rem;
        box-shadow: 0 2px 8px rgba(44,62,80,0.10);
        letter-spacing: 0.5px;
    }
    .footer p {
        margin: 0;
        opacity: 0.92;
    }
    /* Spasi antar elemen */
    .stApp > div > div { margin-bottom: 1.2rem; }
    @media (max-width: 768px) {
        .main-header h1 { font-size: 2.1rem; }
        .upload-section, .image-container, .result-card { padding: 1.2rem 0.7rem; }
        .prediction-item { font-size: 1rem; }
        .confidence-bar { width: 90px; height: 12px; }
    }
</style>
""", unsafe_allow_html=True)

# ======= Header Section =======
st.markdown("""
<div class="main-header">
    <h1><span class="logo">🌸</span> Flower Classifier 102 🌸</h1>
    <p>Upload a flower image and discover its species with our advanced MobileNetV3 model</p>
</div>
""", unsafe_allow_html=True)

# ======= Upload Section =======
st.markdown("""
<div class="upload-section">
    <h3><span class="icon">📁</span> Upload Your Flower Image</h3>
    <p style="color: #2c3e50; opacity: 0.8;">Choose from file or take a photo with your camera</p>
</div>
""", unsafe_allow_html=True)

# Create tabs for different upload methods
tab1, tab2 = st.tabs(["📁 Upload File", "📷 Take Photo"])

with tab1:
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

with tab2:
    camera_photo = st.camera_input("Take a photo of your flower", label_visibility="collapsed")

# Use either uploaded file or camera photo
if uploaded_file is not None:
    image_source = uploaded_file
elif camera_photo is not None:
    image_source = camera_photo
else:
    image_source = None

if image_source is not None:
    # Progress indicator
    with st.spinner("🔍 Analyzing your flower image..."):
        # Load image
        img = Image.open(image_source).convert('RGB')
        
        # Prediksi
        probs, classes = predict(img, model, topk=topk)
        labels = [class_names.get(str(cls + 1), f'class_{cls + 1}') for cls in classes]

    # Success message
    st.success("✅ Analysis complete! Here are the results:")
    
    # Layout dengan 2 kolom
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.markdown("""
        <div class="image-container">
            <h4 style="color: #2c3e50; margin-bottom: 1rem;">📷 Your Image</h4>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f'<img src="data:image/png;base64,{img_to_base64(img)}" class="framed-image" alt="Uploaded Image"/>', unsafe_allow_html=True)
        # Badge nama file
        if hasattr(image_source, 'name'):
            st.markdown(f'<div class="file-badge">{image_source.name}</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="result-card">
            <h3 style="color: #2c3e50; margin-bottom: 1.5rem;">🎯 Prediction Results</h3>
        </div>
        """, unsafe_allow_html=True)
        # Tampilkan hasil prediksi dengan styling yang menarik
        for i in range(topk):
            confidence = probs[i] * 100
            st.markdown(f"""
            <div class="prediction-item">
                <span class="flower-icon">🌸</span>
                <div>
                    <strong>#{i + 1} {labels[i]}</strong>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: {confidence}%"></div>
                    </div>
                </div>
                <div style="font-size: 1.2rem; font-weight: bold;">
                    {confidence:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
        # Additional info
        st.markdown("""
        <div style="background: #f8f9fa; padding: 1rem; border-radius: 10px; margin-top: 1rem;">
            <p style="margin: 0; color: #6c757d; font-size: 0.9rem;">
                💡 <strong>Tip:</strong> The model shows the most likely flower species based on your image.
            </p>
        </div>
        """, unsafe_allow_html=True)


else:
    # Placeholder content when no image is uploaded
    st.markdown("""
    <div style="text-align: center; padding: 3rem; background: #f8f9fa; border-radius: 15px; margin: 2rem 0;">
        <div style="font-size: 4rem; margin-bottom: 1rem;">🌸</div>
        <h3 style="color: #2c3e50;">Ready to identify flowers?</h3>
        <p style="color: #6c757d;">Upload an image above to get started!</p>
    </div>
    """, unsafe_allow_html=True)
