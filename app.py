import streamlit as st
import cv2
import numpy as np
import pandas as pd
from detect_drowsiness import run_drowsiness_detection
import os
from datetime import datetime
import json
from dotenv import load_dotenv

load_dotenv()
os.makedirs("reports", exist_ok=True)

### ---------------- GEMINI: DRIVER INSIGHT ---------------- ###
def get_gemini_summary(report):
    from openai import OpenAI
    
    drowsy = report['drowsy_percentage']
    yawns = report['total_yawns']
    NVIDIA_API_KEY = os.getenv('NVIDIA_API_KEY')
    
    if not NVIDIA_API_KEY:
        return "Stay alert and take regular breaks."
    
    try:
        client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=NVIDIA_API_KEY
        )
        completion = client.chat.completions.create(
            model="meta/llama-3.1-8b-instruct",
            messages=[{"role": "user", "content": f"You are a driver safety coach. Analyze: drowsy {drowsy}%, yawns {yawns}. Give: 1 evaluation sentence, likely cause, 2 safety tips. Use HTML <b> tags for headers. Keep under 80 words."}],
            temperature=0.2,
            top_p=0.7,
            max_tokens=150,
            stream=False
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"NVIDIA AI error: {e}")
        return "Stay alert and take regular breaks while driving."

### ---------------------- REPORT BUILD ---------------------- ###
def generate_report(df):
    total_time = len(df) / 30
    drowsy_periods = df[df['Status'] == 'Drowsy']
    return {
        "trip_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_duration_minutes": float(round(total_time / 60, 2)),
        "drowsy_percentage": float(round(len(drowsy_periods) / len(df) * 100, 2)),
        "total_yawns": int(df['Yawn Count'].max()),
        "average_ear": float(round(df['EAR'].mean(), 3)),
        "average_mar": float(round(df['MAR'].mean(), 3)),
        "drowsy_episodes": int((df['Status'] == 'Drowsy').sum())
    }

# ------------------ PAGE CONFIG ------------------- 
st.set_page_config(page_title="Driver Drowsiness Detection", layout="wide")

# ------------------ CUSTOM CSS ------------------- 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Inter:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0a0e1a;
    color: #e0e6f0;
}

.main-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2.8rem;
    font-weight: 700;
    letter-spacing: 2px;
    color: #ffffff;
    text-align: center;
    margin-bottom: 0;
}

.subtitle {
    text-align: center;
    color: #7a8aaa;
    font-size: 1rem;
    margin-top: 4px;
    margin-bottom: 32px;
    letter-spacing: 1px;
}

.stat-card {
    background: linear-gradient(135deg, #111827, #1a2540);
    border: 1px solid #1f3060;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
}

.stat-number {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: #f97316;
}

.stat-label {
    font-size: 0.78rem;
    color: #7a8aaa;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-top: 4px;
}

.warning-card {
    background: linear-gradient(135deg, #1a0d0d, #2d1515);
    border: 1px solid #5c1c1c;
    border-left: 4px solid #ef4444;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 12px;
}

.tip-card {
    background: linear-gradient(135deg, #0d1a12, #0f2318);
    border: 1px solid #1a4d2e;
    border-left: 4px solid #22c55e;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 12px;
}

.info-card {
    background: linear-gradient(135deg, #0d1530, #111e3a);
    border: 1px solid #1e3a6e;
    border-left: 4px solid #3b82f6;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 12px;
}

.section-header {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.4rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    color: #cbd5e1;
    text-transform: uppercase;
    border-bottom: 1px solid #1f3060;
    padding-bottom: 8px;
    margin-bottom: 16px;
    margin-top: 32px;
}

.checklist-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 14px;
    background: #111827;
    border: 1px solid #1f3060;
    border-radius: 8px;
    margin-bottom: 8px;
    font-size: 0.95rem;
}

.sidebar-alert {
    background: linear-gradient(135deg, #1a0d0d, #2d1010);
    border: 1px solid #7f1d1d;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 16px;
}

.sidebar-tip {
    background: #111827;
    border: 1px solid #1f3060;
    border-radius: 8px;
    padding: 12px 14px;
    margin-bottom: 8px;
    font-size: 0.88rem;
    line-height: 1.6;
    color: #cbd5e1;
}

.divider {
    border: none;
    border-top: 1px solid #1f3060;
    margin: 24px 0;
}

.badge-red {
    display: inline-block;
    background: #7f1d1d;
    color: #fca5a5;
    font-size: 0.7rem;
    padding: 2px 8px;
    border-radius: 20px;
    letter-spacing: 1px;
    text-transform: uppercase;
    font-weight: 600;
}

.badge-green {
    display: inline-block;
    background: #14532d;
    color: #86efac;
    font-size: 0.7rem;
    padding: 2px 8px;
    border-radius: 20px;
    letter-spacing: 1px;
    text-transform: uppercase;
    font-weight: 600;
}

[data-testid="stSidebarCollapseButton"] {
display: none;
}
</style>
""", unsafe_allow_html=True)

# SESSION STATE
if 'trip_data' not in st.session_state:
    st.session_state.trip_data = []
if 'show_sidebar' not in st.session_state:
    st.session_state.show_sidebar = False
if 'trip_running' not in st.session_state:
    st.session_state.trip_running = False

# DROWSINESS SIDEBAR 
if st.session_state.show_sidebar:
    with st.sidebar:
        col_title, col_close = st.columns([4, 1])
        with col_title:
            st.markdown("### 🚨 Drowsiness Alert")
        with col_close:
            if st.button("✕", key="close_sidebar"):
                st.session_state.show_sidebar = False
                st.rerun()

        st.markdown("""
        <div class="sidebar-alert">
            <b style="color:#f87171; font-size:1.1rem;"> You appear drowsy!</b><br>
            <span style="color:#fca5a5; font-size:0.9rem;">Take action immediately to stay safe.</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Immediate Actions:**")
        tips = [
            " Pull over to a safe spot now",
            " Take a 20-minute power nap",
            " Drink coffee or caffeinated drink",
            " Splash cold water on your face",
            " Open windows for fresh air",
            " Play upbeat, loud music",
            " Call someone to stay awake",
            " Take a break every 2 hours",
        ]
        for tip in tips:
            st.markdown(f'<div class="sidebar-tip">{tip}</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("**Did you know?**")
        st.markdown("""
        <div class="sidebar-tip">
        Driving drowsy is <b style="color:#f87171">as dangerous as drunk driving</b>. 
        Reaction time doubles after 17–19 hours awake.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="sidebar-tip">
        Most drowsy driving accidents occur between <b style="color:#fbbf24">2–6 AM</b> and <b style="color:#fbbf24">2–4 PM</b>.
        </div>
        """, unsafe_allow_html=True)

#  MAIN CONTENT
st.markdown('<div class="main-title"> DRIVER DROWSINESS DETECTION</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-POWERED REAL-TIME FATIGUE MONITORING SYSTEM</div>', unsafe_allow_html=True)

# STATS BAR  
st.markdown("""
<div style="display:flex; gap:16px; margin-bottom:8px;">
    <div class="stat-card" style="flex:1">
        <div class="stat-number">1 in 25</div>
        <div class="stat-label">Drivers fall asleep while driving monthly</div>
    </div>
    <div class="stat-card" style="flex:1">
        <div class="stat-number">1,40,000+</div>
        <div class="stat-label">Road accident deaths yearly in India</div>
    </div>
    <div class="stat-card" style="flex:1">
        <div class="stat-number">3×</div>
        <div class="stat-label">Higher crash risk after 6+ hrs without sleep</div>
    </div>
    <div class="stat-card" style="flex:1">
        <div class="stat-number">20%</div>
        <div class="stat-label">Of fatal crashes involve drowsy driving</div>
    </div>
</div>
""", unsafe_allow_html=True)

#TRIP CONTROLS 
st.markdown('<hr class="divider">', unsafe_allow_html=True)

FRAME_WINDOW = st.empty()
status_placeholder = st.empty()
stats_placeholder = st.empty()

c1, c2, c3 = st.columns(3)
with c1: start = st.button("▶️ Start Trip", use_container_width=True)
with c2: stop  = st.button("⏹️ End Trip", use_container_width=True)
with c3: clear = st.button("🗑️ Clear Data", use_container_width=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# HOME PAGE AWARENESS CONTENT (shown only when not running) 
if not st.session_state.trip_running:

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown('<div class="section-header"> Warning Signs of Drowsiness</div>', unsafe_allow_html=True)
        signs = [
            "Frequent yawning or rubbing eyes",
            "Heavy or drooping eyelids",
            "Drifting from your lane",
            "Difficulty remembering last few miles",
            "Blank staring or missing exits",
            "Jerky head movements or nodding off",
        ]
        for text in signs:
            st.markdown(f"""
            <div class="warning-card">
                <span style="color:#ef4444; font-size:1.1rem">☐</span>&nbsp;&nbsp;
                <span style="color:#fca5a5">{text}</span>
            </div>
            """, unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="section-header"> Pre-Trip Safety Checklist</div>', unsafe_allow_html=True)
        checks = [
            "Slept at least 7–8 hours last night",
            "Not driving between 2–6 AM or 2–4 PM",
            "Had water/light meal before driving",
            "Avoid alcohol or sedating medications",
            "Planned rest stops every 2 hours",
            "Emergency contact is saved on phone",
        ]
        for item in checks:
            st.markdown(f"""
            <div class="checklist-item">
                <span style="color:#22c55e; font-size:1.1rem">☐</span>
                <span style="color:#cbd5e1">{item}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown('<div class="section-header"> Stay Alert Tips</div>', unsafe_allow_html=True)
        tips_list = [
            ("", "Caffeine", "Effective for 3–4 hrs. Don't rely on it alone."),
            ("", "Power Nap", "20 min nap is the most effective reset."),
            ("", "Music", "Upbeat, loud music helps short-term alertness."),
            ("", "Hydration", "Dehydration causes fatigue. Drink water."),
        ]
        for icon, title, desc in tips_list:
            st.markdown(f"""
            <div class="tip-card">
                <span style="font-size:1.1rem">{icon}</span>
                <b style="color:#86efac"> {title}:</b>
                <span style="color:#a7c4a0; font-size:0.9rem"> {desc}</span>
            </div>
            """, unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="section-header"> High Risk Hours</div>', unsafe_allow_html=True)
        hours = [
            ("", "2 AM – 6 AM", "Highest risk — body's lowest alertness"),
            ("", "2 PM – 4 PM", "Post-lunch dip in alertness"),
            ("", "10 PM – 12 AM", "Late-night fatigue builds"),
            ("", "7 AM – 12 PM", "Peak alertness window"),
        ]
        for dot, time, desc in hours:
            st.markdown(f"""
            <div class="info-card">
                <span>{dot} <b style="color:#93c5fd">{time}</b></span><br>
                <span style="color:#94a3b8; font-size:0.88rem">{desc}</span>
            </div>
            """, unsafe_allow_html=True)

    with col_c:
        st.markdown('<div class="section-header"> Quick Facts</div>', unsafe_allow_html=True)
        facts = [
            "Drowsy driving causes ~100,000 police-reported crashes annually.",
            "You cannot reliably detect your own drowsiness.",
            "Microsleeps (2–30 sec) can happen with eyes open.",
            "Rolling down windows or turning up AC does NOT prevent drowsiness.",
            "Teens and young adults are at highest risk.",
        ]
        for fact in facts:
            st.markdown(f"""
            <div class="info-card">
                <span style="color:#93c5fd; font-size:0.88rem"> {fact}</span>
            </div>
            """, unsafe_allow_html=True)

#TRIP LOGIC 
if start:
    st.session_state.trip_running = True
    st.session_state.trip_data = []
    st.session_state.show_sidebar = False
    st.success("Trip Started — Stay Alert!")

if stop and len(st.session_state.trip_data) > 0:
    st.session_state.trip_running = False
    st.session_state.show_sidebar = False  # Auto-close sidebar on end trip

    df = pd.DataFrame(
        st.session_state.trip_data,
        columns=["EAR", "MAR", "Yawn Count", "Left Eye", "Right Eye", "Status"]
    ).reset_index().rename(columns={"index": "Frame"})

    report_data = generate_report(df)

    st.subheader(" Trip Summary")
    a, b, c = st.columns(3)
    a.metric("Trip Duration (mins)", report_data["total_duration_minutes"])
    b.metric("Drowsy %", f"{report_data['drowsy_percentage']}%")
    c.metric("Total Yawns", report_data["total_yawns"])

    st.write("###  EAR Trend (Eye Aspect Ratio)")
    st.line_chart(df["EAR"])
    st.write("###  MAR Trend (Mouth Aspect Ratio)")
    st.line_chart(df["MAR"])
    st.write("###  Awake vs Drowsy Frames")
    st.bar_chart(df["Status"].value_counts())

    insight = get_gemini_summary(report_data)
    st.subheader(" AI Driving Insight")
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #1e3a5f, #0d1f3c); padding: 20px; border-radius: 12px; border-left: 4px solid #4fc3f7;">
        <p style="color: #e0f7fa; font-size: 16px; line-height: 1.8; margin: 0;">{insight}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.download_button("📥 Download CSV", df.to_csv(index=False),
            file_name=f"trip_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv")
    with col2:
        st.download_button("📥 Download JSON", json.dumps(report_data, indent=4),
            file_name=f"trip_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", mime="application/json")

if clear:
    st.session_state.trip_data = []
    st.info("Data cleared.")

if st.session_state.get("trip_running", False):
    for frame, EAR, MAR, yawn, alarm, left_pred, right_pred in run_drowsiness_detection():
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        status = " DROWSY!" if alarm else " Awake"
        color = "red" if alarm else "green"
        status_placeholder.markdown(
            f"### Status: <span style='color:{color}'>{status}</span>",
            unsafe_allow_html=True
        )

        stats_placeholder.write({
            "EAR": round(EAR, 3),
            "MAR": round(MAR, 3),
            "Yawns": yawn,
            "Left Eye": left_pred,
            "Right Eye": right_pred
        })

        st.session_state.trip_data.append([
            EAR, MAR, yawn, left_pred, right_pred,
            "Drowsy" if alarm else "Awake"
        ])

        # Auto-open sidebar when drowsy detected
        if alarm and not st.session_state.show_sidebar:
            st.session_state.show_sidebar = True
            st.rerun()