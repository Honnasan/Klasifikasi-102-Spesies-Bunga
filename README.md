# 🌸 Flower Classifier 102 - Streamlit App

A beautiful and modern flower classification application built with Streamlit and PyTorch, capable of identifying 102 different flower species using MobileNetV2.

## 🚀 Features

- **Modern UI**: Beautiful glassmorphism design with gradient backgrounds
- **Multiple Upload Methods**: File upload and camera capture
- **Real-time Prediction**: Instant flower species identification
- **Confidence Scores**: Visual confidence bars for predictions
- **Responsive Design**: Works on desktop and mobile devices
- **102 Flower Classes**: Comprehensive flower species database

## 📋 Requirements

- Python 3.7+
- PyTorch
- Streamlit
- PIL (Pillow)
- NumPy

## 🛠️ Installation

1. **Clone or download the project files**
2. **Install dependencies**:
   ```bash
   pip install streamlit torch torchvision pillow numpy
   ```

## 🚀 Local Deployment

1. **Navigate to the project directory**:

   ```bash
   cd /path/to/your/flower-classifier
   ```

2. **Run the Streamlit app**:

   ```bash
   streamlit run app.py
   ```

3. **Open your browser** and go to `http://localhost:8501`

## 🌐 Cloud Deployment

### Option 1: Streamlit Cloud (Recommended)

1. **Push your code to GitHub**
2. **Go to [share.streamlit.io](https://share.streamlit.io)**
3. **Connect your GitHub repository**
4. **Deploy automatically**

### Option 2: Heroku

1. **Create `requirements.txt`**:

   ```bash
   pip freeze > requirements.txt
   ```

2. **Create `Procfile`**:

   ```
   web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
   ```

3. **Deploy to Heroku**

### Option 3: Railway

1. **Connect your GitHub repository to Railway**
2. **Set build command**: `pip install -r requirements.txt`
3. **Set start command**: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`

## 📁 Project Structure

```
deploy/
├── app.py              # Main Streamlit application
├── mobilenetv2_flowers102.pth  # Trained model weights
├── label_map.json      # Flower class labels
└── README.md           # This file
```

## 🎯 How to Use

1. **Upload an image** using the file uploader or take a photo with your camera
2. **Wait for analysis** - the model will process your image
3. **View results** - see the predicted flower species with confidence scores
4. **Try different images** to explore the 102 flower classes

## 🔧 Technical Details

- **Model**: MobileNetV2 fine-tuned on Flowers102 dataset
- **Framework**: PyTorch with Streamlit frontend
- **Image Processing**: Resize to 224x224, normalize with ImageNet stats
- **Output**: Top-k predictions with confidence scores

## 🎨 UI Features

- **Glassmorphism Design**: Modern glass-like interface
- **Gradient Backgrounds**: Beautiful color transitions
- **Animated Elements**: Smooth hover effects and transitions
- **Responsive Layout**: Adapts to different screen sizes
- **Accessibility**: Proper labels and keyboard navigation

## 📱 Mobile Support

The application is fully responsive and works great on mobile devices. You can:

- Upload images from your phone's gallery
- Take photos directly with your camera
- View results in an optimized mobile layout

## 🤝 Contributing

Feel free to contribute to this project by:

- Reporting bugs
- Suggesting new features
- Improving the UI/UX
- Adding more flower classes

## 📄 License

This project is open source and available under the MIT License.

---

**Built with ❤️ using Streamlit & PyTorch**
