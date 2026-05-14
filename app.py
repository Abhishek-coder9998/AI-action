"""app.py — Football Intelligence Platform (Production v3)"""

import streamlit as st
import os, tempfile
import pandas as pd
import numpy as np

st.set_page_config(page_title="⚽ Football Intelligence Platform", page_icon="⚽", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&display=swap');
*{font-family:'Inter',sans-serif;}
.hero{font-size:2.6rem;font-weight:900;text-align:center;
  background:linear-gradient(135deg,#00C851,#007E33,#FFD700);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:.3rem;}
.sub{text-align:center;color:#9CA3AF;margin-bottom:1.5rem;font-size:1rem;}
.card{background:#1F2937;border-radius:12px;padding:16px;border:1px solid #374151;margin:6px 0;}
.ev-card{background:#111827;border-radius:10px;padding:12px 16px;margin:5px 0;border-left:4px solid #00C851;}
.badge{display:inline-block;padding:3px 12px;border-radius:20px;font-size:.75rem;font-weight:700;background:#065F46;color:#6EE7B7;}
.comm{background:#0F172A;border-radius:8px;border-left:4px solid #FFD700;padding:10px 14px;color:#E5E7EB;font-style:italic;margin:3px 0;}
.mval{font-size:2rem;font-weight:800;color:#34D399;}
.mlbl{font-size:.78rem;color:#9CA3AF;margin-top:4px;}
.mcrd{background:#1F2937;border-radius:12px;padding:16px;text-align:center;border:1px solid #374151;}
.tl-card{background:#111827;border-radius:10px;padding:10px 14px;margin:4px 0;border-left:4px solid #6366F1;}
.section-title{font-size:1.3rem;font-weight:700;color:#F9FAFB;margin:1.2rem 0 .6rem 0;}
</style>
""", unsafe_allow_html=True)

COLORS = {
    "Goal Scored":"#FFD700","Shot on Target":"#FF6B35","Goal Attempt":"#F97316",
    "Side Kick":"#EF4444","Edge Kick":"#DC2626","Penalty Kick":"#B91C1C",
    "Free Kick":"#8B5CF6","Corner Kick":"#EC4899","Header":"#F59E0B",
    "Tackle / Sliding":"#10B981","Sliding Interception":"#059669","Defensive Clearance":"#047857",
    "Defending Ball":"#065F46","Defensive Interception":"#14B8A6","Ball Recovery":"#06B6D4",
    "Aerial Duel":"#3B82F6","Aggressive Pressing":"#6366F1","Counter Attack":"#F43F5E",
    "Attacking Transition":"#FB7185","Cross":"#84CC16","Crossing Movement":"#65A30D",
    "Dribble":"#22D3EE","Through Pass":"#38BDF8","Pass":"#6EE7B7",
    "Goalkeeper Save":"#60A5FA","Foul":"#F87171","Offside":"#FBBF24",
    "Yellow / Red Card":"#DC2626","Throw-in":"#A78BFA","Celebration":"#FCD34D",
    "Ball in Play":"#9CA3AF",
}
EMOJIS = {
    "Goal Scored":"⚽","Shot on Target":"🎯","Goal Attempt":"🥅","Side Kick":"🦵",
    "Edge Kick":"↗️","Penalty Kick":"🟡","Free Kick":"🌀","Corner Kick":"🚩",
    "Header":"🦅","Tackle / Sliding":"🛡️","Sliding Interception":"⚡","Defensive Clearance":"🧱",
    "Defending Ball":"🔒","Defensive Interception":"✂️","Ball Recovery":"💪",
    "Aerial Duel":"🤼","Aggressive Pressing":"🔥","Counter Attack":"⚡","Attacking Transition":"🏃",
    "Cross":"🏹","Crossing Movement":"↗️","Dribble":"💨","Through Pass":"🎯",
    "Pass":"↗️","Goalkeeper Save":"🧤","Foul":"🚫","Offside":"🚩",
    "Yellow / Red Card":"🟨","Throw-in":"🤾","Celebration":"🎉","Ball in Play":"⚽",
}

def col(lbl): return COLORS.get(lbl,"#9CA3AF")
def emo(lbl): return EMOJIS.get(lbl,"🔵")

@st.cache_resource(show_spinner=False)
def load_model():
    from predictor import FootballActionPredictor
    return FootballActionPredictor(top_k=6)

def is_cloud():
    return os.environ.get("STREAMLIT_SHARING_MODE") == "streamlit" or \
           "streamlit.io" in os.environ.get("SERVER_NAME","")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<p class="hero">⚽ Football Intelligence Platform</p>', unsafe_allow_html=True)
st.markdown('<p class="sub">AI-Powered Action Recognition • YouTube & Video Upload • Timestamp Event Detection • Professional Analytics</p>', unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    sport_mode  = st.selectbox("🏟️ Mode", ["football","general"], index=0)
    segment_sec = st.slider("📐 Segment Duration (s)", 5.0, 30.0, 15.0, 5.0)
    top_k       = st.slider("🏆 Top Actions Shown", 3, 8, 6)
    show_comm   = st.toggle("🎙️ Commentary", value=True)
    show_heat   = st.toggle("🗺️ Motion Heatmap", value=True)
    show_table  = st.toggle("📋 Detail Table", value=True)
    show_tl_table = st.toggle("📊 Event Table View", value=True)
    st.markdown("---")
    st.markdown("""
**Pipeline**  
`Input` → `Frame Extract` → `X-CLIP` → `Motion Analysis` → `Commentary`

**Model**: Microsoft X-CLIP (HuggingFace)  
**Actions**: 30+ football labels  
**New**: YouTube • Long Video • Timestamps
""")

# ── Input Section ─────────────────────────────────────────────────────────────
st.markdown("### 📥 Video Input")
st.markdown("<small style='color:#9CA3AF'>Choose one: Upload a local file OR paste a YouTube link below</small>", unsafe_allow_html=True)

vpath = None
source_label = ""

inp_col1, inp_divider, inp_col2 = st.columns([5, 0.3, 5])

with inp_col1:
    st.markdown("""
    <div style='background:#1F2937;border-radius:12px;padding:16px 20px;border:1px solid #374151;min-height:140px;'>
    <div style='color:#34D399;font-weight:700;font-size:1rem;margin-bottom:8px;'>📂 Upload Video File</div>
    <div style='color:#9CA3AF;font-size:.82rem;margin-bottom:12px;'>MP4 / MOV / AVI — up to 10 minutes</div>
    </div>
    """, unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Upload video", label_visibility="collapsed",
        type=["mp4","mov","avi"]
    )
    if uploaded:
        suf = os.path.splitext(uploaded.name)[1] or ".mp4"
        tf  = tempfile.NamedTemporaryFile(delete=False, suffix=suf)
        tf.write(uploaded.read()); tf.close()
        vpath = tf.name
        source_label = f"📂 {uploaded.name}"
        st.success(f"✅ File ready: {uploaded.name}")

with inp_divider:
    st.markdown("""
    <div style='display:flex;align-items:center;justify-content:center;height:140px;'>
    <div style='background:#374151;width:2px;height:100px;border-radius:2px;margin:0 auto;'></div>
    </div>
    <div style='text-align:center;color:#6B7280;font-size:.8rem;font-weight:700;margin-top:-20px;'>OR</div>
    """, unsafe_allow_html=True)

with inp_col2:
    st.markdown("""
    <div style='background:#1F2937;border-radius:12px;padding:16px 20px;border:1px solid #374151;min-height:140px;'>
    <div style='color:#60A5FA;font-weight:700;font-size:1rem;margin-bottom:8px;'>🎬 YouTube / URL</div>
    <div style='color:#9CA3AF;font-size:.82rem;margin-bottom:12px;'>Standard YouTube, Shorts, or direct MP4 link</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div style='background:#1F2937;border-radius:12px;padding:16px 20px;border:1px solid #374151;'>
    <div style='color:#60A5FA;font-weight:700;font-size:1rem;margin-bottom:8px;'>🎬 YouTube / URL &nbsp; <span style='background:#1E3A5F;color:#60A5FA;font-size:.7rem;padding:2px 8px;border-radius:10px;font-weight:700;'>🚧 COMING SOON</span></div>
    <div style='color:#9CA3AF;font-size:.82rem;margin-bottom:8px;'>Standard YouTube, Shorts, or direct MP4 link</div>
    </div>
    """, unsafe_allow_html=True)

    st.info("""
**🚧 YouTube Direct Integration — Coming Soon**

YouTube's server-side bot-detection policy currently blocks automated video access (HTTP 403 / Format Not Available errors), even with tools like `yt-dlp`.

**✅ Workaround (takes 30 seconds):**
1. Open the YouTube video in your browser
2. Download it free at 👉 [yt1s.com](https://yt1s.com) or [snapsave.app](https://snapsave.app)
3. Upload the downloaded MP4 using **📂 Upload Video File** on the left

**🔮 Planned Fix:** YouTube Data API v3 with OAuth2 authentication (requires API key quota).
""")


# ── Video Info + Analysis ─────────────────────────────────────────────────────
from video_processing import VideoProcessor, VideoProcessingError, format_timestamp, YouTubeStreamProcessor
from football_intelligence import analyze_motion_signature, build_heatmap_data

# Determine active source — local file or YouTube stream
_yt_stream  = st.session_state.get("yt_stream_url")
_yt_title   = st.session_state.get("yt_title", "YouTube Video")
_yt_dur     = st.session_state.get("yt_duration", 0)
_has_local  = vpath and os.path.exists(vpath)
_has_stream = bool(_yt_stream)

if _has_local or _has_stream:
    c1, c2 = st.columns([3,2])
    with c1:
        st.subheader("📽️ Video Preview")
        if _has_local:
            st.video(vpath)
        else:
            st.info(f"🎬 **{_yt_title}**\n\nYouTube stream loaded — frames will be read directly for analysis.")

    with c2:
        st.subheader("📋 Video Info")
        try:
            if _has_local:
                vp    = VideoProcessor(vpath)
                vp.validate_duration()
                info  = vp.get_info()
                v_dur = info["duration_sec"]
                vp_analyse = vp
            else:
                vp    = YouTubeStreamProcessor(_yt_stream, _yt_title, _yt_dur)
                info  = vp.get_info()
                v_dur = _yt_dur if _yt_dur > 0 else info["duration_sec"]
                vp_analyse = vp

            if segment_sec > v_dur and v_dur > 0:
                st.warning(f"⚠️ Segment duration adjusted to {v_dur:.1f}s")
                effective_seg = v_dur
            else:
                effective_seg = segment_sec

            a,b = st.columns(2)
            a.metric("⏱️ Duration", f"{v_dur:.0f}s")
            b.metric("🎞️ FPS", info["fps"])
            a.metric("📐 Width", f"{info['width']}px")
            b.metric("📏 Height", f"{info['height']}px")
            n_est = max(1, int(v_dur / effective_seg)) if v_dur > 0 else "?"
            st.info(f"📦 ~{n_est} segment(s) × {effective_seg:.0f}s each")
            st.success("✅ Ready to analyse")
            run = st.button("🚀 Analyse Football Actions", type="primary", use_container_width=True)
        except VideoProcessingError as e:
            st.error(f"❌ {e}"); run = False
        except Exception as e:
            st.error(f"❌ {e}"); run = False

    if run:
        with st.status("🔬 Running Football AI Pipeline…", expanded=True) as status:
            st.write("🤖 Loading X-CLIP model (cached after first load)…")
            predictor = load_model()
            predictor.top_k = top_k

            st.write(f"🎞️ Extracting segments ({effective_seg:.0f}s each)…")
            vp2      = vp_analyse
            segments = vp2.extract_segments(segment_duration=effective_seg)
            st.write(f"  → {len(segments)} segment(s) extracted")


            st.write("🧠 Classifying actions with X-CLIP + motion intelligence…")
            prog = st.progress(0)
            ptxt = st.empty()

            def on_prog(cur, tot):
                prog.progress(cur/tot)
                ptxt.text(f"Segment {cur}/{tot}")

            raw = predictor.predict_all_segments(segments, progress_callback=on_prog)

            st.write("⚡ Applying football motion intelligence & diversity check…")
            heatmaps = []
            used_labels = []
            for i, seg in enumerate(segments):
                sig   = analyze_motion_signature(seg["frames"])
                hints = sig.get("football_hints", [])
                
                # Get current predictions
                current_preds = raw[i]["top_k_actions"]

                # Calculate Intensity Scores for ALL candidate actions
                # Intensity = Confidence * (1.0 + motion_bonus)
                motion_bonus = 1.0 + (sig["avg_magnitude"] / 15.0)
                scored_actions = []
                for p in current_preds:
                    # Boost if matches hint
                    h_boost = 1.25 if p["label"] in hints else 1.0
                    score = p["confidence"] * motion_bonus * h_boost
                    scored_actions.append({"label": p["label"], "score": score, "conf": p["confidence"]})
                
                # Sort by score to find the real winner
                scored_actions = sorted(scored_actions, key=lambda x: x["score"], reverse=True)
                winner = scored_actions[0]
                best_label = winner["label"]
                
                # Diversity check: if repeated too much, try next best score
                if used_labels.count(best_label) >= 2 and len(scored_actions) > 1:
                    best_label = scored_actions[1]["label"]
                    winner = scored_actions[1]

                used_labels.append(best_label)
                raw[i]["top_action"]     = best_label
                raw[i]["top_confidence"] = winner["conf"]
                raw[i]["intensity_score"] = winner["score"]
                raw[i]["scored_actions"] = scored_actions # Save for charts
                raw[i]["motion_signature"] = sig
                heatmaps.append(build_heatmap_data(seg["frames"]))
                
                # Debug logging
                print(f"[DEBUG] Segment {i+1}: {best_label} (Score: {winner['score']:.2f})")

            st.write("🎙️ Generating professional commentary…")
            from commentary import generate_full_commentary, generate_match_summary, generate_match_analysis_card
            results = generate_full_commentary(raw)
            summary = generate_match_summary(results)
            card    = generate_match_analysis_card(results)
            status.update(label="✅ Analysis complete!", state="complete")

        st.session_state.update({
            "results":  results,
            "summary":  summary,
            "card":     card,
            "heatmaps": heatmaps,
            "vid_info": vp2.get_info(),
        })

    # Cleanup temp file
    try:
        if not source_label.startswith("📂"):
            os.remove(vpath)
    except: pass

# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if "results" in st.session_state:
    results  = st.session_state["results"]
    summary  = st.session_state["summary"]
    card     = st.session_state["card"]
    vinfo    = st.session_state.get("vid_info", {})

    import plotly.graph_objects as go
    import plotly.express as px

    st.markdown("---")
    st.markdown("## 📊 Football Analytics Dashboard")

    # ── KPI Metrics ──────────────────────────────────────────────────────────
    m1,m2,m3,m4,m5 = st.columns(5)
    acts   = [r["top_action"] for r in results]
    uniq   = list(dict.fromkeys(acts))
    top_act = max(set(acts), key=acts.count) if acts else "—"
    dur_val = vinfo.get("duration_sec","—")
    for col_m, val, lbl in [
        (m1, str(len(results)),      "Segments Analysed"),
        (m2, str(len(uniq)),         "Unique Actions"),
        (m3, format_timestamp(results[0]["start_time"]) if results else "00:00", "Start Time"),
        (m4, f"{dur_val}s",          "Clip Duration"),
        (m5, top_act[:14],           "Top Action"),
    ]:
        col_m.markdown(
            f'<div class="mcrd"><div class="mval">{val}</div><div class="mlbl">{lbl}</div></div>',
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── AI Match Summary ─────────────────────────────────────────────────────
    st.markdown("### 🧠 AI Match Intelligence Summary")
    st.markdown(summary)

    # ── Match Analysis Card ───────────────────────────────────────────────────
    st.markdown("### 📋 Match Analysis Card")
    with st.container():
        st.markdown(
            f'<div class="card"><b>⚽ Overall Match Style:</b> '
            f'<span style="color:#34D399;font-size:1.1rem;">{card["style"]}</span></div>',
            unsafe_allow_html=True
        )
        checks = card.get("theme_checks", {})
        items  = list(checks.items())
        for row_start in range(0, len(items), 3):
            row  = items[row_start:row_start+3]
            rcols = st.columns(len(row))
            for ci, (theme, detected) in enumerate(row):
                icon = "✅" if detected else "⬜"
                clr  = "#34D399" if detected else "#6B7280"
                rcols[ci].markdown(f'<span style="color:{clr}">{icon} {theme}</span>', unsafe_allow_html=True)
        if card.get("tactical_notes"):
            st.markdown("**📌 Tactical Notes:**")
            for note in card["tactical_notes"]:
                st.markdown(f"• {note}")

    # ── Action Segment Section ───────────────────────────────────────────────
    st.markdown("### 🕐 Action Segments")
    
    for ev_idx, r in enumerate(results):
        lbl = r["top_action"]
        c_  = col(lbl)
        dur = r["end_time"] - r["start_time"]
        t_start = format_timestamp(r["start_time"])
        t_end   = format_timestamp(r["end_time"])
        
        # Enhanced Segment Card - EXACT HTML ALIGNMENT
        st.markdown(f'''
<div style="background: #111827; border-radius: 12px; border-left: 10px solid {c_}; padding: 25px; margin-bottom: 0px; border: 1px solid #374151; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);">
    <!-- Top Row: Header + Timestamp -->
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
        <div style="font-weight: 900; color: #9CA3AF; font-size: 1.1rem; text-transform: uppercase; letter-spacing: 1px;">
            ⚡ Segment {r["segment_id"]+1}
        </div>
        <div style="color: #34D399; font-weight: 900; font-size: 1.5rem; display: flex; align-items: center; background: rgba(52, 211, 153, 0.1); padding: 5px 15px; border-radius: 8px; border: 1px solid rgba(52, 211, 153, 0.3);">
            <span style="margin-right: 10px; font-size: 1.2rem;">⏱</span> {t_start} → {t_end}
        </div>
    </div>
    
    <!-- Second Row: Big Action Badge + Start/End Pills -->
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
        <div>
            <span style="background: {c_}; color: white; padding: 10px 24px; border-radius: 8px; font-weight: 900; font-size: 1.4rem; box-shadow: 0 4px 14px 0 {c_}66; display: inline-flex; align-items: center;">
                <span style="margin-right: 12px; font-size: 1.8rem;">{emo(lbl)}</span> {lbl.upper()}
            </span>
        </div>
        <div style="display: flex; gap: 10px;">
            <span style="background: #374151; color: #E5E7EB; padding: 4px 12px; border-radius: 4px; font-size: 0.7rem; font-weight: 700; border: 1px solid #4B5563;">START: {t_start}</span>
            <span style="background: #374151; color: #E5E7EB; padding: 4px 12px; border-radius: 4px; font-size: 0.7rem; font-weight: 700; border: 1px solid #4B5563;">END: {t_end}</span>
        </div>
    </div>
    
    <!-- Third Row: Duration -->
    <div style="margin-bottom: 20px;">
        <span style="background: rgba(156, 163, 175, 0.1); color: #9CA3AF; padding: 6px 14px; border-radius: 6px; font-size: 0.85rem; font-weight: 800; border: 1px solid rgba(156, 163, 175, 0.2);">
            ⏳ DURATION: {dur:.0f}s
        </span>
    </div>
    
    <div style="height: 1px; background: #374151; margin: 20px 0;"></div>

    <!-- Commentary Block -->
''' + (f'''
    <div style="background: #0F172A; border-radius: 8px; padding: 18px; margin-top: 15px; border: 1px solid #1E293B; color: #E5E7EB; line-height: 1.6; font-size: 1rem;">
        <span style="color: #FFD700; margin-right: 10px; font-size: 1.3rem;">🎙️</span> <b>Commentary:</b> {r.get("commentary","")}
    </div>
''' if r.get("commentary") else "") + f'''
    
    <!-- Motion Tags as Chips -->
    <div style="margin-top: 20px; display: flex; flex-wrap: wrap; gap: 10px;">
        {"".join([f'<span style="background: #1E293B; color: #9CA3AF; padding: 5px 15px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; border: 1px solid #374151;">⚡ {h}</span>' for h in (r.get("motion_signature",{}).get("football_hints",[]) if r else [])])}
    </div>
</div>

<!-- Segment Action Intensity Bar Chart Wrapper -->
<div style="margin-bottom: 60px; padding: 20px; background: rgba(17, 24, 39, 0.6); border-radius: 0 0 12px 12px; border: 1px solid #374151; border-top: none;">
''', unsafe_allow_html=True)
        
        # Prepare bar chart data
        raw_scores = r.get("scored_actions", [])
        if not raw_scores:
            preds = r.get("top_k_actions", [])
            raw_scores = [{"label": p["label"], "score": p["confidence"]} for p in preds]

        if len(raw_scores) < 5:
            others = ["Pass", "Dribble", "Press", "Ball in Play", "Defending Ball"]
            existing = [p["label"] for p in raw_scores]
            for o in others:
                if o not in existing and len(raw_scores) < 5:
                    raw_scores.append({"label": o, "score": 0.1})
        
        chart_data = pd.DataFrame([{
            "Action": p["label"],
            "Intensity": p["score"]
        } for p in raw_scores])
        chart_data = chart_data.sort_values("Intensity", ascending=True)
        
        fig_seg = px.bar(
            chart_data, x="Intensity", y="Action", orientation='h',
            color="Action", color_discrete_map=COLORS,
            title=f"📊 Intensity Analysis: {t_start} – {t_end}"
        )
        fig_seg.update_layout(
            showlegend=False, height=280, margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#9CA3AF", size=10),
            xaxis=dict(title=None, showticklabels=False, showgrid=False),
            yaxis=dict(title=None, showgrid=False)
        )
        st.plotly_chart(fig_seg, use_container_width=True, config={'displayModeBar': False})
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Analytics Charts (Heatmap Only) ──────────────────────────────────────
    st.markdown("### 📈 Analytics Charts")
    
    hm_x = []
    hm_z = [[]]
    hm_text = []
    annotations = []

    for i, r in enumerate(results):
        t_start_val = r["start_time"]
        mag = r.get("motion_signature", {}).get("avg_magnitude", 0)
        
        hm_x.append(t_start_val)
        hm_z[0].append(mag)
        hm_text.append(f"{r['top_action']} ({format_timestamp(t_start_val)})")
        
        annotations.append(dict(
            x=t_start_val,
            y=0,
            text=f"{emo(r['top_action'])} {r['top_action']}",
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=-40 - (i % 3) * 30,
            font=dict(size=10, color="#E5E7EB"),
            bgcolor="rgba(17, 24, 39, 0.85)",
            bordercolor="#374151",
            borderwidth=1,
            borderpad=4,
        ))

    fig_hm = go.Figure(data=go.Heatmap(
        z=hm_z,
        x=hm_x,
        y=["Match Activity"],
        colorscale='Hot',
        showscale=True,
        text=hm_text,
        hoverinfo="text+z",
        zmin=0,
    ))

    fig_hm.update_layout(
        title="Activity Heatmap (Video Timeline)",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E5E7EB"),
        height=400,
        xaxis=dict(
            title="Video Timeline (Seconds)",
            dtick=4,
            tick0=0,
            gridcolor="#374151",
            zeroline=False,
            range=[-1, max(hm_x) + 5] if hm_x else None
        ),
        yaxis=dict(showgrid=False, zeroline=False),
        margin=dict(l=20, r=20, t=100, b=40),
        annotations=annotations
    )
    st.plotly_chart(fig_hm, use_container_width=True)

    # ── Detail Table ──────────────────────────────────────────────────────────
    if show_table:
        st.markdown("### 📋 Segment Detail Table")
        tdf = pd.DataFrame([{
            "Segment":         r["segment_id"]+1,
            "Video Time":      f"⏱ {format_timestamp(r['start_time'])}",
            "Action Detected": r["top_action"],
            "Motion Type":     " • ".join(r.get("motion_signature",{}).get("football_hints",[])[:3]),
            "Commentary":      r.get("commentary",""),
        } for r in results])
        st.dataframe(tdf, use_container_width=True, hide_index=True)

    # ── Downloads ─────────────────────────────────────────────────────────────
    st.markdown("### 💾 Download Results")
    from exporter import results_to_json, results_to_csv, build_timeline_text
    d1,d2,d3 = st.columns(3)
    json_str = results_to_json(results, summary, vinfo)
    csv_str  = results_to_csv(results)
    tl_txt   = build_timeline_text(results)
    with d1: st.download_button("⬇️ JSON Report", json_str, "football_analysis.json","application/json", use_container_width=True)
    with d2: st.download_button("⬇️ CSV Export",  csv_str,  "football_analysis.csv", "text/csv",         use_container_width=True)
    with d3: st.download_button("⬇️ Timeline",    tl_txt,   "event_timeline.txt",    "text/plain",        use_container_width=True)

st.markdown("---")
st.markdown("""
<div style='text-align:center;padding:12px 0 4px 0;'>
  <span style='color:#6B7280;font-size:.82rem;'>⚽ Football Intelligence Platform &nbsp;·&nbsp; X-CLIP + Optical Flow &nbsp;·&nbsp; YouTube Support &nbsp;·&nbsp; Python 3.11</span><br>
  <span style='color:#34D399;font-size:.88rem;font-weight:700;'>Built by Abhishek</span>
  <span style='color:#6B7280;font-size:.88rem;'> &nbsp;|&nbsp; </span>
  <span style='color:#60A5FA;font-size:.88rem;font-weight:600;'>Assignment for MultiTV Solutions</span>
</div>
""", unsafe_allow_html=True)
