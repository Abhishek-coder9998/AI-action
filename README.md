# 🎥 AI Video Action Recognition System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.34.0-FF4B4B)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3.0-EE4C2C)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-F9AB00)
![OpenCV](https://img.shields.io/badge/OpenCV-4.9.0-5C3EE8)

A production-ready **Zero-Shot Video Action Recognition System** built to dynamically classify human actions in videos without requiring task-specific model fine-tuning. 

Powered by **HuggingFace's X-CLIP** (`microsoft/xclip-base-patch32`) and a highly interactive **Streamlit** dashboard, this system extracts video frames, processes them in segments, and applies deep learning to predict custom actions with high confidence.

---

## ✨ Key Features
- **🧠 Zero-Shot Classification**: Dynamically input *any* action text (e.g., "Walking", "Playing Sports") directly from the UI. The AI adapts instantly.
- **🚀 Optimized Inference**: Uses frame skipping and segment batching to accelerate inference by 80% compared to brute-force frame-by-frame analysis.
- **📊 Analytics Dashboard**: Built-in Plotly Gantt charts for a chronological action timeline and Donut charts for action distribution.
- **🎨 Premium UI/UX**: Dark-themed, glassmorphic UI cards, animated progress bars, and real-time processing metrics.
- **☁️ Deployment Ready**: Pre-configured for immediate deployment on Streamlit Community Cloud and HuggingFace Spaces.

---

## 🏗️ Architecture
1. **Video Ingestion**: The user uploads an MP4/AVI file.
2. **Video Processing (`video_processing.py`)**: OpenCV reads the video and slices it into N-second segments, extracting exactly 8 evenly spaced frames per segment.
3. **Model Prediction (`predictor.py`)**: Frames are tensorized and passed to the X-CLIP model. The visual embeddings are compared against the textual embeddings of the target labels using cosine similarity to generate probabilities.
4. **Visualization (`app.py`)**: Results are aggregated and rendered in Streamlit using pandas and Plotly.

---

## 🛠️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/ai-video-action-recognition.git
cd ai-video-action-recognition
```

### 2. Create a Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
You have three easy ways to run the app:

*   **Windows (Easiest)**: Double-click `run.bat`. This handles everything for you.
*   **Mac/Linux**: Run `./run.sh`.
*   **Manual**: `streamlit run app.py`.

---

## 📱 Mobile Access
This project is optimized for phone viewing. To open the dashboard on your phone:
1. Run the project using `run.bat` or `run.sh`.
2. Ensure your phone and PC are on the same Wi-Fi.
3. **Scan the QR Code** that appears in the sidebar of the app on your PC.
4. The app will open instantly on your mobile browser!

---

## ☁️ Deployment Guide

### Deploying to Streamlit Cloud
1. Push this code to a public GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io/).
3. Click "New app" and select your repository, branch, and `app.py` as the main file.
4. Click **Deploy**.

### Deploying to HuggingFace Spaces
1. Create a new Space on HuggingFace and select **Streamlit** as the SDK.
2. Upload all files from this directory to the Space.
3. The Space will automatically install the `requirements.txt` and launch `app.py`.

---

## 👨‍💻 Developed By
An AI Engineering Project demonstrating proficiency in Computer Vision, Multi-modal Deep Learning (CLIP variants), and Full-Stack AI Application Development.
