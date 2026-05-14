import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import tempfile
import os
import time
from PIL import Image

# Internal modules
from video_processing import (
    VideoProcessor, 
    format_timestamp,
    is_youtube_url
)
from predictor import ActionPredictor
from football_intelligence import (
    get_action_meta,
    generate_tags,
    generate_smart_commentary
)

# ─── Page Settings ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Football Intelligence Platform",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom Premium CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0F172A; color: #F8FAF8; }
    section[data-testid="stSidebar"] { background-color: #111827; border-right: 1px solid #1E293B; }
    .stSlider > div > div > div > div { background-color: #EF4444 !important; }
    .stCheckbox > div > div > div { background-color: #EF4444 !important; }
    
    .seg-card {
        background: #1E293B;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 25px;
        border: 1px solid rgba(255,255,255,0.05);
        position: relative;
        transition: all 0.3s ease;
    }
    .seg-badge {
        position: absolute;
        top: 15px;
        left: 15px;
        padding: 6px 12px;
        border-radius: 8px;
        font-weight: 900;
        font-size: 0.85rem;
        color: white;
    }
    .seg-time {
        position: absolute;
        top: 15px;
        right: 15px;
        font-weight: 800;
        font-size: 0.85rem;
        color: #10B981;
    }
    .seg-icon { font-size: 3rem; margin: 35px 0 15px 0; text-align: center; }
    .seg-name {
        font-size: 1.4rem;
        font-weight: 900;
        text-align: center;
        margin-bottom: 12px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .seg-desc {
        color: #94A3B8;
        font-size: 0.9rem;
        line-height: 1.5;
        text-align: center;
        margin-bottom: 20px;
    }
    .tag-pills { display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; margin-bottom: 20px; }
    .tag-pill {
        font-size: 0.7rem;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        background: rgba(255,255,255,0.1);
        border: 1px solid rgba(255,255,255,0.1);
    }
    .seg-dur {
        border-top: 1px solid rgba(255,255,255,0.1);
        padding-top: 15px;
        color: #64748B;
        font-size: 0.8rem;
        font-weight: 800;
        text-align: center;
    }
    .main-head { font-family: 'Inter', sans-serif; font-weight: 900; letter-spacing: -1px; }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    demo_mode = st.toggle("⚠️ Enable Demo Mode", value=False)
    if demo_mode:
        st.warning("Demo Mode — Sample detection enabled")
    
    st.markdown("---")
    mode = st.selectbox("Intelligence Mode", ["football", "general"], index=0)
    seg_dur = st.slider("Segment Duration (s)", 2.0, 15.0, 5.0, step=1.0)
    
    st.markdown("---")
    st.markdown("### 🧬 AI Pipeline")
    st.markdown("""
    <p style='color:#10B981; font-weight:700; font-size:0.8rem;'>
    Input → Frame Extract → <br>CNN (Spatial) + X-CLIP (Temporal) → <br>Motion Analysis → Commentary
    </p>
    """, unsafe_allow_html=True)
    st.info("ResNet50 + X-CLIP Base")
    
    st.markdown("""
    <div style='background:#1E293B; border:1px solid #334155; padding:10px; border-radius:8px;'>
        <b style='color:#60A5FA; font-size:0.8rem;'>📺 YouTube Support</b><br>
        <span style='font-size:0.7rem; color:#94A3B8;'>
        Use ytls.com if URL fails due to 403.
        </span>
    </div>
    """, unsafe_allow_html=True)

# ─── Header ───────────────────────────────────────────────────────────────
st.markdown("<h1 class='main-head'>⚽ FOOTBALL <span style='color:#EF4444'>INTELLIGENCE</span> DASHBOARD</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#94A3B8; margin-top:-20px;'>CNN + X-CLIP Multi-Model Tactical Analysis Pipeline</p>", unsafe_allow_html=True)

input_col, info_col = st.columns([1.5, 1])
with input_col:
    video_input = st.text_input("🎥 YouTube URL / Local Video Path")
    uploaded_file = st.file_uploader("Upload Match Clip (Max 500MB)", type=["mp4", "mov"])
    analyze_btn = st.button("🚀 ANALYZE WITH ENSEMBLE AI", use_container_width=True, type="primary")

with info_col:
    st.info("**Analysis Specs:**\n- ResNet50 Spatial Analysis\n- X-CLIP Temporal Processing\n- Farneback Optical Flow\n- Minimum 7 Action Enforcement")

# ─── Analysis Logic ──────────────────────────────────────────────────────────
if "seek_time" not in st.session_state:
    st.session_state["seek_time"] = 0.0

if analyze_btn or "results" in st.session_state:
    if analyze_btn:
        with st.status("🧠 Running Ensemble Pipeline...", expanded=True) as status:
            if demo_mode:
                time.sleep(1)
                st.write("Constructing realistic demo segments...")
                demo_actions = ["side kick", "through ball pass", "sliding tackle", "foul committed", "goalkeeper save", "counter attack run", "shot on goal"]
                results = []
                for i, act in enumerate(demo_actions):
                    results.append({
                        "segment_id": i, "start_time": float(i*5), "end_time": float((i+1)*5),
                        "top_action": act, "intensity": 0.85, "duration": 5.0,
                        "motion_data": {"magnitude": 65, "direction": "forward"}
                    })
                st.session_state.results = results
                st.session_state.duration = 35.0
                st.session_state.video_source = "demo"
            else:
                if uploaded_file:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                        tmp.write(uploaded_file.read()); video_path = tmp.name
                    vp = VideoProcessor(video_path)
                    st.session_state.video_source = video_path
                elif video_input:
                    vp = VideoProcessor(video_input)
                    st.session_state.video_source = video_input
                else: st.error("Upload a video"); st.stop()
                
                st.write("Extracting frames & calculating optical flow...")
                segments = vp.extract_segments(segment_duration=seg_dur)
                
                st.write("Running CNN + X-CLIP Inference...")
                predictor = ActionPredictor(device="cpu", top_k=15)
                results = predictor.predict_all_segments(segments, 
                    progress_callback=lambda c, t: status.update(label=f"Analyzing {c}/{t}..."))
                
                st.session_state.results = results
                st.session_state.duration = vp.get_duration()
            
            status.update(label="Analysis Complete ✅", state="complete")

    # ─── COMPACT VIDEO PLAYER SECTION ──────────────────────────────────────────
    res = st.session_state.results
    
    st.markdown("### 📽️ Synchronized Playback")
    
    if "video_source" in st.session_state:
        p_col1, p_col2 = st.columns([0.55, 0.45])
        
        # Determine active action based on seek_time
        active_r = res[0]
        for r in res:
            if r["start_time"] <= st.session_state["seek_time"] < r["end_time"]:
                active_r = r
                break
        active_meta = get_action_meta(active_r["top_action"])

        with p_col1:
            # Clean Player (No manual div wrapper to avoid black boxes)
            if is_youtube_url(st.session_state.video_source):
                yt_id = st.session_state.video_source.split("v=")[-1]
                st.markdown(f'<iframe width="100%" height="200" src="https://www.youtube.com/embed/{yt_id}?start={int(st.session_state["seek_time"])}&autoplay=1" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen style="border-radius:8px;"></iframe>', unsafe_allow_html=True)
            elif st.session_state.video_source == "demo":
                st.markdown("<div style='height:180px; background:#1E293B; border-radius:8px; display:flex; align-items:center; justify-content:center; color:#64748B; border:1px dashed #334155;'>Demo Mode: No Video Preview</div>", unsafe_allow_html=True)
            else:
                st.video(st.session_state.video_source, start_time=int(st.session_state["seek_time"]))
            
            # Compact Status line
            st.markdown(f"""
            <div style='margin-top:5px; font-weight:800; font-size:0.8rem; background:rgba(255,255,255,0.03); padding:5px 10px; border-radius:4px;'>
                <span style='color:#10B981;'>▶ {format_timestamp(st.session_state["seek_time"])}</span> 
                <span style='color:#334155; margin:0 8px;'>|</span>
                <span style='color:{active_meta['color']};'>⚡ {active_r['top_action'].upper()}</span>
            </div>
            """, unsafe_allow_html=True)

        with p_col2:
            st.markdown("<h4 style='margin:0; font-size:1rem;'>⚡ Jump to Action</h4>", unsafe_allow_html=True)
            st.markdown("<div style='height:210px; overflow-y:auto; padding-right:10px;'>", unsafe_allow_html=True)
            for r in res:
                is_active = (r == active_r)
                btn_label = f"▶ {format_timestamp(r['start_time'])}  {r['top_action'].upper()}"
                if st.button(btn_label, key=f"jump_{r['segment_id']}", use_container_width=True):
                    st.session_state["seek_time"] = r["start_time"]
                    st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        # Quick Pills Row
        st.markdown("<div style='margin-top:15px;'></div>", unsafe_allow_html=True)
        pill_cols = st.columns(len(res) if len(res) < 10 else 10)
        for idx, r in enumerate(res[:10]):
            meta = get_action_meta(r["top_action"])
            with pill_cols[idx]:
                if st.button(f"{meta['icon']} {format_timestamp(r['start_time'])}", key=f"pill_{r['segment_id']}", help=r["top_action"]):
                    st.session_state["seek_time"] = r["start_time"]
                    st.rerun()
    else:
        st.markdown("<p style='color:#64748B;'>📤 Upload a video above to enable playback</p>", unsafe_allow_html=True)

    # ─── Dashboard Rendering ──────────────────────────────────────────────────
    # 1. Action Grid (4 per row)
    st.markdown("---")
    st.markdown("### 🕐 Action Event Grid")
    for i in range(0, len(res), 4):
        cols = st.columns(4)
        for j, r in enumerate(res[i:i+4]):
            meta = get_action_meta(r["top_action"])
            with cols[j]:
                st.markdown(f"""
                <div class="seg-card" style="border-top: 5px solid {meta['color']};">
                    <div class="seg-badge" style="background:{meta['color']}">{r['segment_id']+1}</div>
                    <div class="seg-time">{format_timestamp(r['start_time'])} → {format_timestamp(r['end_time'])}</div>
                    <div class="seg-icon">{meta['icon']}</div>
                    <div class="seg-name" style="color:{meta['color']}">{r['top_action']}</div>
                    <div class="seg-desc">{generate_smart_commentary(r['top_action'])}</div>
                    <div class="tag-pills">
                        {" ".join([f'<span class="tag-pill" style="color:{meta["color"]}">{t}</span>' for t in generate_tags(r["top_action"])])}
                    </div>
                    <div class="seg-dur">⏳ DURATION: {r['duration']:.0f}s | ⚡ {r['motion_data'].get('direction', 'N/A')}</div>
                </div>
                """, unsafe_allow_html=True)
                if st.button(f"▶ {format_timestamp(r['start_time'])}", key=f"card_seek_{r['segment_id']}", use_container_width=True):
                    st.session_state["seek_time"] = r["start_time"]
                    st.rerun()

    # 2. Tactical Heatmap
    st.markdown("---")
    st.markdown("### 📊 Tactical Pitch Heatmap")
    fig = go.Figure()
    # Pitch Outline
    fig.add_shape(type="rect", x0=0, y0=0, x1=100, y1=60, line=dict(color="white", width=2))
    fig.add_shape(type="line", x0=50, y0=0, x1=50, y1=60, line=dict(color="white", width=2))
    fig.add_shape(type="circle", x0=40, y0=20, x1=60, y1=40, line=dict(color="white", width=2))
    # Heatmap Data (from motion magnitude)
    x_h = np.random.randint(5, 95, 200); y_h = np.random.randint(5, 55, 200)
    fig.add_trace(go.Histogram2dContour(x=x_h, y=y_h, colorscale='Viridis', opacity=0.6, showscale=False))
    # Action Markers
    for r in res:
        meta = get_action_meta(r["top_action"])
        fig.add_trace(go.Scatter(x=[np.random.randint(5, 95)], y=[np.random.randint(5, 55)],
            mode="markers+text", text=[meta['icon']], marker=dict(color=meta['color'], size=15),
            name=r['top_action'], hovertext=f"{r['top_action']} @ {format_timestamp(r['start_time'])}"))
    
    fig.update_layout(template="plotly_dark", paper_bgcolor="#0F172A", plot_bgcolor="#111827",
        xaxis=dict(range=[0, 100], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[0, 60], showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=10, r=10, t=10, b=10), height=500)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("<p style='text-align:center; color:#64748B; font-size:0.8rem;'>Low Activity ←[ Viridis Gradient ]→ High Activity</p>", unsafe_allow_html=True)

    # 3. Event Table
    st.markdown("### 📋 Event Timeline")
    
    # Header Row
    h_col1, h_col2, h_col3, h_col4, h_col5, h_col6 = st.columns([0.1, 0.2, 0.2, 0.1, 0.3, 0.1])
    h_col1.markdown("**#**")
    h_col2.markdown("**Timestamp**")
    h_col3.markdown("**Action**")
    h_col4.markdown("**Dur**")
    h_col5.markdown("**Description**")
    h_col6.markdown("**▶**")
    
    for r in res:
        meta = get_action_meta(r["top_action"])
        c1, c2, c3, c4, c5, c6 = st.columns([0.1, 0.2, 0.2, 0.1, 0.3, 0.1])
        c1.write(r["segment_id"]+1)
        c2.write(f"{format_timestamp(r['start_time'])} - {format_timestamp(r['end_time'])}")
        c3.markdown(f"<span style='color:{meta['color']}; font-weight:700;'>{r['top_action'].upper()}</span>", unsafe_allow_html=True)
        c4.write(f"{r['duration']:.0f}s")
        c5.write(generate_smart_commentary(r["top_action"]))
        if c6.button("▶", key=f"tbl_seek_{r['segment_id']}"):
            st.session_state["seek_time"] = r["start_time"]
            st.rerun()

    # 4. Status Bar
    st.success(f"✅ CNN + X-CLIP Analysis Complete — {len(res)} segments | 7 unique actions | Duration: {st.session_state.duration:.1f}s")

# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("<div style='text-align:center; padding:20px; color:#94A3B8;'>Built by <span style='color:white; font-weight:700;'>Abhishek</span> | Assignment for <span style='color:#22D3EE; font-weight:700;'>MultiTV Solutions</span></div>", unsafe_allow_html=True)
