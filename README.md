# Football Intelligence Dashboard
AI-powered football video analysis using CNN + X-CLIP ensemble

## Live Demo
[Click here to try the app](YOUR_STREAMLIT_URL)

## Features
- **ResNet50 CNN** spatial analysis
- **Microsoft X-CLIP** temporal action recognition  
- **Farneback Optical Flow** motion analysis
- **Detects 7+ unique football actions**:
  Goal Kick | Goalkeeper Save | Free Kick | 
  Dribbling | Corner Kick | Defensive Header | Standing Tackle
- **Synchronized video player** with seek-to-action
- **Tactical Pitch Heatmap**
- **Event Timeline** with timestamps
- **Demo Mode** for testing without GPU

## Tech Stack
- **Python + Streamlit**
- **PyTorch + ResNet50**
- **Microsoft X-CLIP** (xclip-base-patch32)
- **OpenCV** (Optical Flow)
- **Plotly** (Heatmap)

## How to Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Assignment
**Built by Abhishek | MultiTV Solutions**
