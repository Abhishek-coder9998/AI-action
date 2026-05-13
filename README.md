# ⚽ Football Intelligence Platform

![Python](https://img.shields.io/badge/Python-3.11.9-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3+-EE4C2C)
![Transformers](https://img.shields.io/badge/HuggingFace-X--CLIP-F9AB00)
![OpenCV](https://img.shields.io/badge/OpenCV-4.9+-5C3EE8)
![Assignment](https://img.shields.io/badge/Assignment-MultiTV%20Solutions-00C851)

> **Built by Abhishek** | Assignment for **MultiTV Solutions**

A production-ready **AI-powered Football Video Analytics Platform** built using Microsoft X-CLIP, Streamlit, and OpenCV. Upload any football video and get real-time action detection, timestamp-based event analysis, motion heatmaps, professional commentary, and an advanced AI match summary — all in a premium dark dashboard.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🧠 **Zero-Shot AI** | Microsoft X-CLIP classifies 30+ football actions without fine-tuning |
| 🕐 **Timestamp Events** | Detects actions per segment with exact timestamps (MM:SS) |
| 🎙️ **AI Commentary** | Broadcast-quality professional football commentary per event |
| 📊 **Analytics Dashboard** | Pie charts, bar charts, confidence trend line, KPI metrics |
| 🗺️ **Motion Heatmap** | Optical-flow based pitch activity heatmap (OpenCV) |
| 🧠 **Advanced Match Summary** | 7-dimension tactical analysis: pressing, transitions, momentum shifts |
| 📋 **Match Analysis Card** | 15 tactical theme checks + expanded notes |
| 💾 **Exports** | Download JSON report, CSV, and plain-text timeline |
| 🎬 **YouTube — Coming Soon** | See section below |
| 📱 **Long Video Support** | Chunked processing — supports up to 10-minute clips |

---

## 🏗️ Architecture

```
Video Upload (MP4/MOV/AVI)
        ↓
VideoProcessor (OpenCV)
   → Frame extraction
   → Segment splitting (15s chunks)
        ↓
FootballActionPredictor (Microsoft X-CLIP)
   → 30+ football action labels
   → Confidence normalization (25–93%)
        ↓
Football Intelligence Layer (OpenCV Optical Flow)
   → Motion signature analysis
   → Zone activity mapping
   → Football hint injection
        ↓
Commentary Engine
   → Per-event broadcast commentary
   → Advanced AI match summary (7 dimensions)
   → Match Analysis Card
        ↓
Streamlit Dashboard
   → Timestamp timeline
   → Charts, heatmaps, KPIs, exports
```

---

## 🎬 YouTube Integration — Coming Soon 🚧

> **Current Status: Not Available**

YouTube's aggressive **server-side bot-detection policy** currently blocks all automated video access, returning `HTTP 403 Forbidden` or `Requested format is not available` errors — even when using industry-standard tools like `yt-dlp`.

**Why it fails:**
- YouTube detects server-side requests as bots and blocks them
- Format restrictions vary per video and per IP
- YouTube CDN URLs require valid browser session tokens

**✅ Workaround (takes 30 seconds):**
1. Open the YouTube football video in your browser
2. Download it free at [yt1s.com](https://yt1s.com) or [snapsave.app](https://snapsave.app)
3. Upload the downloaded `.mp4` file using the **📂 Upload Video File** option in the app

**🔮 Planned Fix:**
YouTube integration will be implemented using the official **YouTube Data API v3** with OAuth2 authentication, which bypasses bot-detection entirely. This requires a Google Cloud API key with quota allocation.

---

## 🛠️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/Abhishek-coder9998/AI-action.git
cd AI-action
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
streamlit run app.py
```

Or use the provided scripts:
- **Windows:** Double-click `run.bat`
- **Mac/Linux:** `./run.sh`

---

## ☁️ Streamlit Cloud Deployment

1. Push to a public GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io/)
3. Select your repo → `app.py` → **Deploy**

> `runtime.txt` is pre-configured for **Python 3.11.9** for maximum compatibility.

---

## 📁 Project Structure

```
AI-Video-Action-Recognition/
├── app.py                  # Main Streamlit application
├── predictor.py            # X-CLIP model + confidence normalization
├── video_processing.py     # OpenCV frame extraction + YouTube support
├── football_intelligence.py# Optical flow + motion analysis layer
├── commentary.py           # Commentary + advanced AI match summary
├── exporter.py             # JSON / CSV / Timeline exports
├── requirements.txt        # Python dependencies
├── runtime.txt             # Python 3.11.9 for Streamlit Cloud
└── README.md
```

---

## 👨‍💻 Developed By

**Abhishek** — AI/ML Engineer  
📌 Assignment submission for **MultiTV Solutions**

> Demonstrates proficiency in: Computer Vision · Multi-modal Deep Learning (X-CLIP) · Optical Flow · Streamlit · Full-Stack AI Application Development · Production Deployment
