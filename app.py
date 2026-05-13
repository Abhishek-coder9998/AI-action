"""app.py — Football Action Recognition AI (Advanced Dashboard)"""

import streamlit as st
import os, tempfile, socket
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from video_processing import VideoProcessor, VideoProcessingError, format_timestamp
from predictor import FootballActionPredictor
from commentary import generate_full_commentary, generate_match_summary, generate_match_analysis_card
from football_intelligence import analyze_motion_signature, build_heatmap_data, boost_confidence_with_hints
from exporter import results_to_json, results_to_csv, build_timeline_text

st.set_page_config(page_title="⚽ Football AI Analytics", page_icon="⚽", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&display=swap');
*{font-family:'Inter',sans-serif;}
.hero{font-size:2.6rem;font-weight:900;text-align:center;
  background:linear-gradient(135deg,#00C851,#007E33,#FFD700);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.sub{text-align:center;color:#9CA3AF;margin-bottom:1.5rem;}
.card{background:#1F2937;border-radius:12px;padding:16px;border:1px solid #374151;margin:6px 0;}
.ev-card{background:#111827;border-radius:10px;padding:12px 16px;margin:5px 0;border-left:4px solid #00C851;}
.badge{display:inline-block;padding:3px 12px;border-radius:20px;font-size:.75rem;font-weight:700;background:#065F46;color:#6EE7B7;}
.comm{background:#0F172A;border-radius:8px;border-left:4px solid #FFD700;padding:10px 14px;color:#E5E7EB;font-style:italic;margin:3px 0;}
.chk{color:#34D399;font-weight:700;}
.mval{font-size:2rem;font-weight:800;color:#34D399;}
.mlbl{font-size:.78rem;color:#9CA3AF;margin-top:4px;}
.mcrd{background:#1F2937;border-radius:12px;padding:16px;text-align:center;border:1px solid #374151;}
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
    return FootballActionPredictor(top_k=6)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<p class="hero">⚽ Football AI Analytics Platform</p>', unsafe_allow_html=True)
st.markdown('<p class="sub">Upload a football clip (max 40 s) • AI detects 6+ actions per 15-second segment • Motion intelligence • Professional commentary</p>', unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    sport_mode  = st.selectbox("🏟️ Mode", ["football","general"], index=0)
    segment_sec = st.slider("📐 Segment Duration (s)", 5.0, 15.0, 15.0, 5.0)
    top_k       = st.slider("🏆 Top Actions Shown", 3, 8, 6)
    show_comm   = st.toggle("🎙️ Commentary", value=True)
    show_heat   = st.toggle("🗺️ Motion Heatmap", value=True)
    show_table  = st.toggle("📋 Detail Table", value=True)
    st.markdown("---")
    st.markdown("""
**Pipeline**  
`Video` → `Frame Extract` → `X-CLIP` → `Confidence Normalize` → `Motion Analysis` → `Commentary`  

**Model**: Microsoft X-CLIP (HuggingFace)  
**Actions**: 30+ football labels  
**Confidence**: Normalized 25–93%  
""")

    st.markdown("---")
    try:
        # Detect local IP for phone access
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        st.markdown(f"### 📱 Mobile Access")
        st.markdown(f"Scan to open on phone:")
        qr_url = f"https://chart.googleapis.com/chart?chs=200x200&cht=qr&chl=http://{local_ip}:8501"
        st.image(qr_url, caption=f"http://{local_ip}:8501")
        st.caption("Ensure phone is on the same Wi-Fi.")
    except:
        pass

# ── Upload ────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader("📂 Upload Football Video (MP4 / MOV / AVI — max 40 s)", type=["mp4","mov","avi"])

if uploaded:
    suf   = os.path.splitext(uploaded.name)[1] or ".mp4"
    tf    = tempfile.NamedTemporaryFile(delete=False, suffix=suf)
    tf.write(uploaded.read()); tf.close()
    vpath = tf.name

    c1, c2 = st.columns([3,2])
    with c1:
        st.subheader("📽️ Uploaded Clip")
        st.video(vpath)
    with c2:
        st.subheader("📋 Video Info")
        try:
            vp = VideoProcessor(vpath); vp.validate_duration()
            info = vp.get_info()
            v_dur = info['duration_sec']
            
            # Auto-adjust segment duration if it exceeds video length
            if segment_sec > v_dur:
                st.warning(f"⚠️ Segment duration adjusted to match clip length ({v_dur:.1f}s)")
                effective_segment_sec = v_dur
            else:
                effective_segment_sec = segment_sec

            a,b = st.columns(2)
            a.metric("⏱️ Duration", f"{v_dur}s")
            b.metric("🎞️ FPS", info['fps'])
            a.metric("📐 Width", f"{info['width']}px")
            b.metric("📏 Height", f"{info['height']}px")
            st.success("✅ Valid video — ready to analyse")
            run = st.button("🚀 Analyse Football Actions", type="primary", use_container_width=True)
        except VideoProcessingError as e:
            st.error(f"❌ {e}"); run = False
        except Exception as e:
            st.error(f"❌ {e}"); run = False

    if run:
        with st.status("🔬 Running Football AI Pipeline…", expanded=True) as status:
            st.write("🤖 Loading X-CLIP model…")
            predictor = load_model()
            predictor.top_k = top_k

            st.write(f"🎞️ Extracting {effective_segment_sec:.0f}s segments…")
            vp2 = VideoProcessor(vpath)
            segments = vp2.extract_segments(segment_duration=effective_segment_sec)
            st.write(f"  → {len(segments)} segment(s)")

            st.write("🧠 Classifying actions (X-CLIP + motion analysis)…")
            prog = st.progress(0)
            ptxt = st.empty()

            def on_prog(cur, tot):
                prog.progress(cur/tot); ptxt.text(f"Segment {cur}/{tot}")

            raw = predictor.predict_all_segments(segments, progress_callback=on_prog)

            # Motion intelligence overlay
            st.write("⚡ Applying football motion intelligence…")
            heatmaps = []
            for i, seg in enumerate(segments):
                sig = analyze_motion_signature(seg["frames"])
                hints = sig.get("football_hints", [])
                
                # Fallback: If X-CLIP didn't find these hints, inject them
                detected_labels = [a["label"] for a in raw[i]["top_k_actions"]]
                for hint in hints:
                    if hint not in detected_labels:
                        raw[i]["top_k_actions"].append({
                            "label": hint,
                            "confidence": 0.45, # high enough to be visible
                            "rank": len(raw[i]["top_k_actions"]) + 1
                        })
                
                # Re-sort and update top action
                raw[i]["top_k_actions"] = sorted(raw[i]["top_k_actions"], key=lambda x: x["confidence"], reverse=True)[:top_k]
                raw[i]["top_action"]     = raw[i]["top_k_actions"][0]["label"]
                raw[i]["top_confidence"] = raw[i]["top_k_actions"][0]["confidence"]
                
                raw[i]["motion_signature"] = sig
                heatmaps.append(build_heatmap_data(seg["frames"]))

            st.write("🎙️ Generating professional commentary…")
            results = generate_full_commentary(raw)
            summary = generate_match_summary(results)
            card    = generate_match_analysis_card(results)
            status.update(label="✅ Analysis complete!", state="complete")

        st.session_state.update({
            "results":   results,
            "summary":   summary,
            "card":      card,
            "heatmaps":  heatmaps,
            "vid_info":  vp2.get_info(),
        })

    try: os.remove(vpath)
    except: pass

# ═══════════════════════════════════════════════════════════════════════════════
# DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
if "results" in st.session_state:
    results  = st.session_state["results"]
    summary  = st.session_state["summary"]
    card     = st.session_state["card"]
    heatmaps = st.session_state.get("heatmaps", [])
    vinfo    = st.session_state.get("vid_info", {})

    st.markdown("---")
    st.markdown("## 📊 Football Analytics Dashboard")

    # ── Metrics ───────────────────────────────────────────────────────────────
    m1,m2,m3,m4 = st.columns(4)
    acts    = [r["top_action"] for r in results]
    uniq    = list(dict.fromkeys(acts))
    avg_c   = sum(r["top_confidence"] for r in results)/len(results) if results else 0
    top_act = max(set(acts), key=acts.count)
    for col_m, val, lbl in [
        (m1, str(len(results)),           "Segments Analysed"),
        (m2, str(len(uniq)),              "Unique Actions"),
        (m3, f"{avg_c*100:.0f}%",         "Avg Confidence"),
        (m4, vinfo.get("duration_sec","—"), "Clip Duration (s)"),
    ]:
        col_m.markdown(f'<div class="mcrd"><div class="mval">{val}</div><div class="mlbl">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Match Summary ─────────────────────────────────────────────────────────
    st.markdown("### 🧠 AI Match Intelligence Summary")
    st.info(summary)

    # ── Match Analysis Card ───────────────────────────────────────────────────
    st.markdown("### 📋 Match Analysis Card")
    with st.container():
        st.markdown(f'<div class="card"><b>⚽ Overall Match Style:</b> <span style="color:#34D399;font-size:1.1rem;">{card["style"]}</span></div>', unsafe_allow_html=True)
        checks = card["theme_checks"]
        cols_per_row = 3
        items = list(checks.items())
        for row_start in range(0, len(items), cols_per_row):
            row = items[row_start:row_start+cols_per_row]
            rcols = st.columns(len(row))
            for ci, (theme, detected) in enumerate(row):
                icon = "✅" if detected else "⬜"
                clr  = "#34D399" if detected else "#6B7280"
                rcols[ci].markdown(f'<span style="color:{clr}">{icon} {theme}</span>', unsafe_allow_html=True)
        if card["tactical_notes"]:
            st.markdown("**📌 Tactical Notes:**")
            for note in card["tactical_notes"]:
                st.markdown(f"• {note}")
        rating = card["overall_rating"]
        st.progress(rating/100, text=f"AI Confidence Rating: {rating}%")

    # ── Event Timeline ────────────────────────────────────────────────────────
    st.markdown("### 🕐 Event Timeline")
    for r in results:
        ts  = format_timestamp(r["start_time"])
        tse = format_timestamp(r["end_time"])
        lbl = r["top_action"]
        cnf = r["top_confidence"]
        c   = col(lbl)
        st.markdown(
            f'<div class="ev-card" style="border-left-color:{c}">'
            f'<b>{emo(lbl)} {ts} → {tse}</b> &nbsp;'
            f'<span class="badge">{lbl}</span> &nbsp;'
            f'<small style="color:{c}">{cnf*100:.0f}% confidence</small>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if show_comm and r.get("commentary"):
            seg_info = f"Segment {r['segment_id']+1} [{ts}-{tse}]"
            st.markdown(f'<div class="comm">🎙️ <b>{seg_info}:</b> {r["commentary"]}</div>', unsafe_allow_html=True)

        # Motion hints badge
        sig   = r.get("motion_signature", {})
        hints = sig.get("football_hints", [])
        if hints:
            hint_str = " • ".join(hints[:3])
            st.markdown(f'<small style="color:#6B7280;padding-left:16px;">⚡ Motion intelligence: {hint_str}</small>', unsafe_allow_html=True)

    # ── Top-6 per Segment ─────────────────────────────────────────────────────
    st.markdown("### 🏆 Top 6 Actions per Segment")
    for r in results:
        seg_lbl = f"Segment {r['segment_id']+1}  ({format_timestamp(r['start_time'])} – {format_timestamp(r['end_time'])})"
        with st.expander(f"🔍 {seg_lbl}  →  {r['top_action']}", expanded=False):
            top_k_data = r.get("top_k_actions", [])
            if top_k_data:
                labels = [a["label"] for a in top_k_data]
                confs  = [a["confidence"]*100 for a in top_k_data]
                clrs   = [col(l) for l in labels]
                fig = go.Figure(go.Bar(
                    x=confs, y=labels, orientation="h",
                    marker=dict(color=clrs, opacity=0.85),
                    text=[f"{c:.0f}%" for c in confs], textposition="outside",
                ))
                fig.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#E5E7EB"), height=260,
                    xaxis=dict(title="Confidence (%)", range=[0, max(confs)*1.25]),
                    yaxis=dict(autorange="reversed"),
                    margin=dict(l=0,r=30,t=5,b=25),
                )
                st.plotly_chart(fig, use_container_width=True)

            # Confidence progress bars
            st.markdown("**Confidence Breakdown:**")
            for a in top_k_data[:6]:
                bar_col, val_col = st.columns([4,1])
                with bar_col:
                    st.progress(min(a["confidence"], 1.0), text=a["label"])
                with val_col:
                    st.markdown(f'<b style="color:{col(a["label"])}">{a["confidence"]*100:.0f}%</b>', unsafe_allow_html=True)

    # ── Charts ────────────────────────────────────────────────────────────────
    st.markdown("### 📈 Analytics Charts")
    ch1, ch2 = st.columns(2)
    df = pd.DataFrame([{
        "Action": r["top_action"],
        "Confidence": round(r["top_confidence"]*100,1),
        "Time": format_timestamp(r["start_time"]),
    } for r in results])

    with ch1:
        vc = df["Action"].value_counts().reset_index()
        vc.columns = ["Action","Count"]
        fig_pie = px.pie(vc, values="Count", names="Action", hole=0.42,
                         color="Action", color_discrete_map=COLORS, title="Action Breakdown")
        fig_pie.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                              font=dict(color="#E5E7EB"))
        st.plotly_chart(fig_pie, use_container_width=True)

    with ch2:
        fig_bar = px.bar(df, x="Time", y="Confidence", color="Action",
                         color_discrete_map=COLORS, title="Confidence per Segment",
                         labels={"Confidence":"Confidence (%)"})
        fig_bar.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                              font=dict(color="#E5E7EB"), showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── Motion Heatmap ────────────────────────────────────────────────────────
    if show_heat and heatmaps:
        st.markdown("### 🗺️ Motion Activity Heatmap")
        valid_hm = [h for h in heatmaps if h is not None]
        if valid_hm:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            combined = np.mean(valid_hm, axis=0) if len(valid_hm) > 1 else valid_hm[0]
            
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.heatmap(combined, cmap="hot", cbar=True, ax=ax, xticklabels=False, yticklabels=False)
            ax.set_title("Pitch Motion Intensity (Aggregated)", color="white")
            fig.patch.set_facecolor('#0E1117')
            ax.set_facecolor('#0E1117')
            
            st.pyplot(fig)
        else:
            st.info("Motion heatmap requires at least 2 frames per segment.")

    # ── Detail Table ──────────────────────────────────────────────────────────
    if show_table:
        st.markdown("### 📋 Segment Detail Table")
        tdf = pd.DataFrame([{
            "Timestamp":       format_timestamp(r["start_time"]),
            "Start (s)":       r["start_time"],
            "End (s)":         r["end_time"],
            "Action Detected": r["top_action"],
            "Confidence (%)":  round(r["top_confidence"]*100,1),
            "Commentary":      r.get("commentary",""),
        } for r in results])
        st.dataframe(tdf, use_container_width=True, hide_index=True)

    # ── Downloads ─────────────────────────────────────────────────────────────
    st.markdown("### 💾 Download Results")
    d1,d2,d3 = st.columns(3)
    json_str = results_to_json(results, summary, vinfo)
    csv_str  = results_to_csv(results)
    tl_txt   = build_timeline_text(results)
    with d1: st.download_button("⬇️ JSON Report", json_str, "football_analysis.json","application/json", use_container_width=True)
    with d2: st.download_button("⬇️ CSV Export",  csv_str,  "football_analysis.csv", "text/csv",         use_container_width=True)
    with d3: st.download_button("⬇️ Timeline",    tl_txt,   "event_timeline.txt",    "text/plain",        use_container_width=True)

st.markdown("---")
st.markdown('<p style="text-align:center;color:#374151;font-size:.8rem;">⚽ Football AI Analytics · X-CLIP + Optical Flow Intelligence · Built with Streamlit</p>', unsafe_allow_html=True)
