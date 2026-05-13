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
`Input` → `Frame Extract` → `X-CLIP` → `Confidence Normalize` → `Motion Analysis` → `Commentary`

**Model**: Microsoft X-CLIP (HuggingFace)  
**Actions**: 30+ football labels  
**Confidence**: Normalized 25–93%  
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

            st.write("⚡ Applying football motion intelligence…")
            heatmaps = []
            for i, seg in enumerate(segments):
                sig   = analyze_motion_signature(seg["frames"])
                hints = sig.get("football_hints", [])
                detected = [a["label"] for a in raw[i]["top_k_actions"]]
                for hint in hints:
                    if hint not in detected:
                        raw[i]["top_k_actions"].append({
                            "label": hint, "confidence": 0.45, "rank": len(raw[i]["top_k_actions"])+1
                        })
                raw[i]["top_k_actions"] = sorted(
                    raw[i]["top_k_actions"], key=lambda x: x["confidence"], reverse=True
                )[:top_k]
                raw[i]["top_action"]     = raw[i]["top_k_actions"][0]["label"]
                raw[i]["top_confidence"] = raw[i]["top_k_actions"][0]["confidence"]
                raw[i]["motion_signature"] = sig
                heatmaps.append(build_heatmap_data(seg["frames"]))

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
    heatmaps = st.session_state.get("heatmaps", [])
    vinfo    = st.session_state.get("vid_info", {})

    import plotly.graph_objects as go
    import plotly.express as px

    st.markdown("---")
    st.markdown("## 📊 Football Analytics Dashboard")

    # ── KPI Metrics ──────────────────────────────────────────────────────────
    m1,m2,m3,m4,m5 = st.columns(5)
    acts   = [r["top_action"] for r in results]
    uniq   = list(dict.fromkeys(acts))
    avg_c  = sum(r["top_confidence"] for r in results)/len(results) if results else 0
    top_act = max(set(acts), key=acts.count)
    dur_val = vinfo.get("duration_sec","—")
    for col_m, val, lbl in [
        (m1, str(len(results)),      "Segments Analysed"),
        (m2, str(len(uniq)),         "Unique Actions"),
        (m3, f"{avg_c*100:.0f}%",    "Avg Confidence"),
        (m4, str(dur_val)+"s",       "Clip Duration"),
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
        checks = card["theme_checks"]
        items  = list(checks.items())
        for row_start in range(0, len(items), 3):
            row  = items[row_start:row_start+3]
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

    # ── Timestamp Event Detection ─────────────────────────────────────────────
    st.markdown("### 🕐 Timestamp Event Detection")

    # Build structured event list
    event_list = []
    for r in results:
        event_list.append({
            "action":     r["top_action"],
            "timestamp":  format_timestamp(r["start_time"]),
            "ts_end":     format_timestamp(r["end_time"]),
            "confidence": round(r["top_confidence"]*100, 1),
            "segment":    r["segment_id"],
            "start_s":    r["start_time"],
        })

    # Timeline view + Table view tabs
    tl_tab1, tl_tab2 = st.tabs(["🔵 Timeline View", "📋 Table View"])

    with tl_tab1:
        for ev in event_list:
            lbl = ev["action"]
            c_  = col(lbl)
            # Get tactical explanation from card timeline if available
            tl  = card.get("event_timeline", [])
            expl = next((t["explanation"] for t in tl if t["segment_id"]==ev["segment"]), "")
            st.markdown(
                f'<div class="ev-card" style="border-left-color:{c_}">'
                f'<b>{emo(lbl)} {ev["timestamp"]} → {ev["ts_end"]}</b> &nbsp;'
                f'<span class="badge">{lbl}</span> &nbsp;'
                f'<small style="color:{c_}">{ev["confidence"]}% confidence</small>'
                + (f'<br><small style="color:#9CA3AF;margin-top:4px">{expl}</small>' if expl else "")
                + f'</div>',
                unsafe_allow_html=True,
            )
            if show_comm:
                r_match = next((r for r in results if r["segment_id"]==ev["segment"]), None)
                if r_match and r_match.get("commentary"):
                    st.markdown(
                        f'<div class="comm">🎙️ <b>Segment {ev["segment"]+1} [{ev["timestamp"]}–{ev["ts_end"]}]:</b> {r_match["commentary"]}</div>',
                        unsafe_allow_html=True
                    )
            # Motion hints
            r_match2 = next((r for r in results if r["segment_id"]==ev["segment"]), None)
            if r_match2:
                hints = r_match2.get("motion_signature",{}).get("football_hints",[])
                if hints:
                    st.markdown(
                        f'<small style="color:#6B7280;padding-left:16px">⚡ Motion: {" • ".join(hints[:3])}</small>',
                        unsafe_allow_html=True
                    )

    with tl_tab2:
        if show_tl_table:
            ev_df = pd.DataFrame([{
                "Timestamp":     e["timestamp"],
                "Action":        e["action"],
                "Confidence (%)":e["confidence"],
                "Segment":       e["segment"]+1,
                "Start (s)":     e["start_s"],
            } for e in event_list])
            st.dataframe(ev_df, use_container_width=True, hide_index=True)

        # Expandable analytics cards
        st.markdown("#### 🔍 Expandable Event Cards")
        for ev in event_list:
            lbl = ev["action"]
            with st.expander(f'{emo(lbl)} {ev["timestamp"]} — {lbl} ({ev["confidence"]}%)', expanded=False):
                r_match = next((r for r in results if r["segment_id"]==ev["segment"]), None)
                if r_match:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Action:** {lbl}")
                        st.markdown(f"**Timestamp:** {ev['timestamp']} → {ev['ts_end']}")
                        st.markdown(f"**Confidence:** {ev['confidence']}%")
                        st.progress(min(r_match["top_confidence"],1.0))
                    with col2:
                        expl = next((t["explanation"] for t in card.get("event_timeline",[]) if t["segment_id"]==ev["segment"]),"")
                        st.markdown(f"**Tactical Note:** {expl}")
                        if r_match.get("commentary"):
                            st.markdown(f"**Commentary:** *{r_match['commentary']}*")
                    if r_match.get("top_k_actions"):
                        st.markdown("**Top Actions:**")
                        for a in r_match["top_k_actions"][:4]:
                            st.progress(min(a["confidence"],1.0), text=f'{a["label"]} — {a["confidence"]*100:.0f}%')

    # ── Top-K per Segment ─────────────────────────────────────────────────────
    st.markdown("### 🏆 Top Actions per Segment")
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
            st.markdown("**Confidence Breakdown:**")
            for a in top_k_data[:6]:
                bc, vc = st.columns([4,1])
                with bc: st.progress(min(a["confidence"],1.0), text=a["label"])
                with vc: st.markdown(f'<b style="color:{col(a["label"])}">{a["confidence"]*100:.0f}%</b>', unsafe_allow_html=True)

    # ── Analytics Charts ──────────────────────────────────────────────────────
    st.markdown("### 📈 Analytics Charts")
    ch1, ch2 = st.columns(2)
    df = pd.DataFrame([{
        "Action":     r["top_action"],
        "Confidence": round(r["top_confidence"]*100, 1),
        "Time":       format_timestamp(r["start_time"]),
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

    # Confidence timeline line chart
    if len(results) > 2:
        fig_line = px.line(df, x="Time", y="Confidence", markers=True,
                           title="Confidence Trend Over Time",
                           labels={"Confidence":"Confidence (%)"})
        fig_line.update_traces(line_color="#00C851", marker_color="#FFD700")
        fig_line.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                               font=dict(color="#E5E7EB"))
        st.plotly_chart(fig_line, use_container_width=True)

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
        from video_processing import format_timestamp as fmt_ts
        tdf = pd.DataFrame([{
            "Timestamp":       format_timestamp(r["start_time"]),
            "Start (s)":       r["start_time"],
            "End (s)":         r["end_time"],
            "Action Detected": r["top_action"],
            "Confidence (%)":  round(r["top_confidence"]*100, 1),
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
