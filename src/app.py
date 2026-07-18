import streamlit as st
import requests
from datetime import datetime

st.set_page_config(
    page_title="Omega Intake Router Console",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# GLOBAL STYLES
# ----------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    #MainMenu, footer, header {visibility: hidden;}

    .stApp {
        background: radial-gradient(circle at 15% 0%, #101826 0%, #0a0e17 45%, #070a10 100%);
    }

    /* ---- Top banner ---- */
    .omega-header {
        background: linear-gradient(120deg, #0f2942 0%, #123a52 50%, #0d3b3d 100%);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 18px;
        padding: 28px 34px;
        margin-bottom: 28px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.05);
        position: relative;
        overflow: hidden;
    }
    .omega-header::after {
        content: "";
        position: absolute;
        top: -60%; right: -10%;
        width: 300px; height: 300px;
        background: radial-gradient(circle, rgba(56,189,248,0.18) 0%, transparent 70%);
        pointer-events: none;
    }
    .omega-title {
        font-size: 30px;
        font-weight: 800;
        color: #f1f5f9;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .omega-subtitle {
        color: #7dd3fc;
        font-size: 14px;
        font-weight: 500;
        margin-top: 6px;
        letter-spacing: 0.3px;
        text-transform: uppercase;
    }
    .omega-tagline {
        color: #94a3b8;
        font-size: 14px;
        margin-top: 10px;
        max-width: 640px;
        line-height: 1.5;
    }

    /* ---- Status pill ---- */
    .live-pill {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        background: rgba(34, 197, 94, 0.12);
        border: 1px solid rgba(34, 197, 94, 0.35);
        color: #4ade80;
        font-size: 12.5px;
        font-weight: 600;
        padding: 6px 14px;
        border-radius: 999px;
        float: right;
        margin-top: 4px;
    }
    .live-dot {
        width: 7px; height: 7px;
        background: #4ade80;
        border-radius: 50%;
        box-shadow: 0 0 8px #4ade80;
        animation: pulse 1.6s infinite;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.35; }
        100% { opacity: 1; }
    }

    /* ---- Panels ---- */
    .panel {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(148,163,184,0.14);
        border-radius: 16px;
        padding: 22px 24px;
        margin-bottom: 18px;
    }
    .panel-label {
        font-size: 12.5px;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        color: #64748b;
        margin-bottom: 4px;
    }
    .panel-heading {
        font-size: 18px;
        font-weight: 700;
        color: #e2e8f0;
        margin-bottom: 14px;
    }

    /* ---- Text area ---- */
    .stTextArea textarea {
        background: #0b1220 !important;
        border: 1px solid rgba(148,163,184,0.2) !important;
        border-radius: 12px !important;
        color: #e2e8f0 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 13.5px !important;
    }
    .stTextArea textarea:focus {
        border-color: #38bdf8 !important;
        box-shadow: 0 0 0 3px rgba(56,189,248,0.15) !important;
    }

    /* ---- Button ---- */
    .stButton button {
        background: linear-gradient(120deg, #0ea5e9, #0284c7) !important;
        color: white !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 10px 22px !important;
        letter-spacing: 0.3px;
        box-shadow: 0 4px 14px rgba(14,165,233,0.35);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .stButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(14,165,233,0.5);
    }

    /* ---- Result card ---- */
    .result-card {
        background: linear-gradient(145deg, rgba(14,165,233,0.08), rgba(255,255,255,0.02));
        border: 1px solid rgba(56,189,248,0.3);
        border-radius: 16px;
        padding: 22px 24px;
        margin-top: 6px;
    }
    .result-eyebrow {
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #38bdf8;
        margin-bottom: 6px;
    }
    .result-specialty {
        font-size: 24px;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 18px;
    }
    .confidence-track {
        width: 100%;
        height: 10px;
        border-radius: 999px;
        background: rgba(148,163,184,0.15);
        overflow: hidden;
        margin-top: 8px;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 999px;
    }
    .confidence-label {
        display: flex;
        justify-content: space-between;
        font-size: 13px;
        color: #94a3b8;
        margin-top: 6px;
    }
    .badge-high { color: #4ade80; }
    .badge-med { color: #facc15; }
    .badge-low { color: #f87171; }

    .exception-banner {
        margin-top: 16px;
        background: rgba(248, 113, 113, 0.1);
        border: 1px solid rgba(248, 113, 113, 0.35);
        border-radius: 10px;
        padding: 12px 16px;
        color: #fca5a5;
        font-size: 13.5px;
        font-weight: 500;
    }

    .placeholder-card {
        border: 1px dashed rgba(148,163,184,0.25);
        border-radius: 14px;
        padding: 40px 20px;
        text-align: center;
        color: #475569;
        font-size: 13.5px;
    }

    .timestamp-note {
        color: #475569;
        font-size: 11.5px;
        margin-top: 14px;
        text-align: right;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------------
st.markdown("""
<div class="omega-header">
    <div class="live-pill"><span class="live-dot"></span>SERVICE ONLINE</div>
    <p class="omega-title">🏥 Omega Digital Platform</p>
    <p class="omega-subtitle">Clinical Intake Router · Care Coordination Layer</p>
    <p class="omega-tagline">
        Automated text routing engine that classifies unstructured physician transcriptions
        and triage notes to the correct specialty queue, in real time.
    </p>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Pipeline Configuration")
    api_url = st.text_input("FastAPI Endpoint", value="http://127.0.0.1:8000/api/v1/route")
    confidence_threshold = st.slider("Human Review Threshold", 0.0, 1.0, 0.60, 0.05)
    st.markdown("---")
    st.markdown("### 🧠 Model")
    st.caption("BioClinicalBERT — fine-tuned specialty classifier")
    st.markdown("---")
    st.markdown("### 📊 Session Stats")
    if "history" not in st.session_state:
        st.session_state.history = []
    st.metric("Records Routed", len(st.session_state.history))
    if st.session_state.history:
        avg_conf = sum(h["confidence"] for h in st.session_state.history) / len(st.session_state.history)
        st.metric("Avg. Confidence", f"{avg_conf * 100:.1f}%")

# ----------------------------------------------------------------------------
# MAIN LAYOUT
# ----------------------------------------------------------------------------
col1, col2 = st.columns([2, 1], gap="large")

with col1:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-label">Input</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-heading">Unstructured Physician Transcription</div>', unsafe_allow_html=True)

    user_input = st.text_area(
        " ",
        height=260,
        placeholder="Type or paste raw chart summaries, triage notes, or physician dictation here...",
        label_visibility="collapsed",
    )
    submit_button = st.button("🚀 Route Clinical Record", type="primary")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-label">Output</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-heading">Automated Routing Execution</div>', unsafe_allow_html=True)

    if submit_button:
        if not user_input.strip() or len(user_input) < 15:
            st.error("Input note text is too short to extract clean clinical features.")
        else:
            with st.spinner("Analyzing semantics via BioClinicalBERT..."):
                try:
                    response = requests.post(api_url, json={"note": user_input}, timeout=10)

                    if response.status_code == 200:
                        data = response.json()["routing_metrics"]
                        specialty = data["assigned_specialty"]
                        confidence = data["confidence_score"]

                        st.session_state.history.append({"specialty": specialty, "confidence": confidence})

                        if confidence >= 0.80:
                            badge_class, bar_color = "badge-high", "#4ade80"
                        elif confidence >= confidence_threshold:
                            badge_class, bar_color = "badge-med", "#facc15"
                        else:
                            badge_class, bar_color = "badge-low", "#f87171"

                        st.markdown(f"""
                        <div class="result-card">
                            <div class="result-eyebrow">Target Destination</div>
                            <div class="result-specialty">{specialty}</div>
                            <div class="confidence-label">
                                <span>Routing Confidence</span>
                                <span class="{badge_class}">{confidence * 100:.1f}%</span>
                            </div>
                            <div class="confidence-track">
                                <div class="confidence-fill" style="width:{confidence * 100:.1f}%; background:{bar_color};"></div>
                            </div>
                            <div class="timestamp-note">Processed at {datetime.now().strftime('%H:%M:%S')}</div>
                        </div>
                        """, unsafe_allow_html=True)

                        if confidence < confidence_threshold:
                            st.markdown("""
                            <div class="exception-banner">
                                ⚠️ Low routing confidence — flagged and sent to Human Exception Review.
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.error(f"API Backend issue. Error Code: {response.status_code}")
                except Exception as err:
                    st.error(f"Could not connect to FastAPI server pipeline: {str(err)}")
    else:
        st.markdown("""
        <div class="placeholder-card">
            📋 Routing results will appear here once a record is submitted.
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)