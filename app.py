import os
import pickle
import streamlit as st
import torch
import plotly.graph_objects as go
from collections import Counter
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ----------------------------------------------------------------------------
# GLOBAL PRODUCTION MODEL CONFIGURATION
# ----------------------------------------------------------------------------
MODEL_DIR = "miru05/clinical-note-bioclinicalbert"

ENCODER_PATH = "./models/transformer_label_encoder.pkl"


@st.cache_resource
def load_deep_learning_pipeline():
    """Load tokenizer, model from HF Hub and label encoder from local disk."""
   
    if not os.path.exists(ENCODER_PATH):
        return None, None, None, "missing_files"

    try:
        # Hugging Face safely downloads these into memory from the cloud repo string!
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()

        with open(ENCODER_PATH, "rb") as f:
            label_encoder = pickle.load(f)

        return tokenizer, model, label_encoder, device
    except Exception as load_err:
        return None, None, None, f"load_error: {load_err}"


def run_local_inference(note_text, tokenizer, model, label_encoder, device):
    """Tokenize input, run the model, and return (specialty, confidence)."""
    inputs = tokenizer(
        note_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        confidence, predicted_idx = torch.max(probs, dim=-1)

    specialty = label_encoder.inverse_transform([predicted_idx.item()])[0]
    return specialty, confidence.item()


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
        background: radial-gradient(circle at 15% 0%, #f4f8fc 0%, #eef2f8 45%, #e8edf5 100%);
    }

    /* ---- Top banner ---- */
    .omega-header {
        background: linear-gradient(120deg, #e0f2fe 0%, #dbeafe 50%, #e0f7f5 100%);
        border: 1px solid rgba(14, 165, 233, 0.25);
        border-radius: 18px;
        padding: 28px 34px;
        margin-bottom: 28px;
        box-shadow: 0 8px 24px rgba(56, 130, 200, 0.08), inset 0 1px 0 rgba(255,255,255,0.6);
        position: relative;
        overflow: hidden;
    }
    .omega-header::after {
        content: "";
        position: absolute;
        top: -60%; right: -10%;
        width: 300px; height: 300px;
        background: radial-gradient(circle, rgba(14,165,233,0.15) 0%, transparent 70%);
        pointer-events: none;
    }
    .omega-title {
        font-size: 30px;
        font-weight: 800;
        color: #0f172a;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .omega-subtitle {
        color: #0284c7;
        font-size: 14px;
        font-weight: 600;
        margin-top: 6px;
        letter-spacing: 0.3px;
        text-transform: uppercase;
    }
    .omega-tagline {
        color: #475569;
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
        color: #15803d;
        font-size: 12.5px;
        font-weight: 700;
        padding: 6px 14px;
        border-radius: 999px;
        float: right;
        margin-top: 4px;
    }
    .live-dot {
        width: 7px; height: 7px;
        background: #22c55e;
        border-radius: 50%;
        box-shadow: 0 0 8px #22c55e;
        animation: pulse 1.6s infinite;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.35; }
        100% { opacity: 1; }
    }

    /* ---- Panels ---- */
    .panel {
        background: #ffffff;
        border: 1px solid rgba(15, 23, 42, 0.06);
        border-radius: 16px;
        padding: 22px 24px;
        margin-bottom: 18px;
        box-shadow: 0 4px 18px rgba(15, 23, 42, 0.04);
    }
    .panel-label {
        font-size: 12.5px;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        color: #94a3b8;
        margin-bottom: 4px;
    }
    .panel-heading {
        font-size: 18px;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 14px;
    }

    /* ---- Text area ---- */
    .stTextArea textarea {
        background: #f8fafc !important;
        border: 1px solid rgba(15, 23, 42, 0.12) !important;
        border-radius: 12px !important;
        color: #0f172a !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 13.5px !important;
    }
    .stTextArea textarea:focus {
        border-color: #0ea5e9 !important;
        box-shadow: 0 0 0 3px rgba(14,165,233,0.15) !important;
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
        box-shadow: 0 4px 14px rgba(14,165,233,0.3);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .stButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(14,165,233,0.45);
    }

    /* ---- Result card ---- */
    .result-card {
        background: linear-gradient(145deg, rgba(14,165,233,0.06), rgba(255,255,255,0.9));
        border: 1px solid rgba(14,165,233,0.25);
        border-radius: 16px;
        padding: 22px 24px;
        margin-top: 6px;
    }
    .result-eyebrow {
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #0284c7;
        margin-bottom: 6px;
    }
    .result-specialty {
        font-size: 24px;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 18px;
    }
    .confidence-track {
        width: 100%;
        height: 10px;
        border-radius: 999px;
        background: rgba(15,23,42,0.08);
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
        color: #64748b;
        margin-top: 6px;
    }
    .badge-high { color: #16a34a; }
    .badge-med { color: #ca8a04; }
    .badge-low { color: #dc2626; }

    .exception-banner {
        margin-top: 16px;
        background: rgba(220, 38, 38, 0.06);
        border: 1px solid rgba(220, 38, 38, 0.25);
        border-radius: 10px;
        padding: 12px 16px;
        color: #b91c1c;
        font-size: 13.5px;
        font-weight: 500;
    }

    .placeholder-card {
        border: 1px dashed rgba(15,23,42,0.15);
        border-radius: 14px;
        padding: 40px 20px;
        text-align: center;
        color: #94a3b8;
        font-size: 13.5px;
        background: #fafcff;
    }

    .timestamp-note {
        color: #94a3b8;
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
tokenizer, model, label_encoder, device_or_error = load_deep_learning_pipeline()
model_ready = tokenizer is not None and model is not None and label_encoder is not None

with st.sidebar:
    st.markdown("### ⚙️ Pipeline Configuration")
    confidence_threshold = st.slider("Human Review Threshold", 0.0, 1.0, 0.60, 0.05)
    st.markdown("---")
    st.markdown("### 🧠 Model")
    st.caption(f"BioClinicalBERT — fine-tuned specialty classifier · `{MODEL_DIR}`")
    if model_ready:
        st.success(f"Loaded locally on **{device_or_error}**")
    elif device_or_error == "missing_files":
        st.error("Model folder or label encoder not found on disk.")
        st.code(f"{MODEL_DIR}\n{ENCODER_PATH}", language="text")
    else:
        st.error("Model failed to load.")
        st.caption(str(device_or_error))
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
        if not model_ready:
            st.error("Model is not loaded. Check the sidebar for the missing path or load error.")
        elif not user_input.strip() or len(user_input) < 15:
            st.error("Input note text is too short to extract clean clinical features.")
        else:
            with st.spinner("Analyzing semantics via BioClinicalBERT..."):
                try:
                    specialty, confidence = run_local_inference(
                        user_input, tokenizer, model, label_encoder, device_or_error
                    )

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
                except Exception as err:
                    st.error(f"Inference failed: {str(err)}")
    else:
        st.markdown("""
        <div class="placeholder-card">
            📋 Routing results will appear here once a record is submitted.
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# SPECIALTY ANALYTICS
# ----------------------------------------------------------------------------
if st.session_state.history:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-label">Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-heading">Specialty Routing Distribution</div>', unsafe_allow_html=True)

    chart_col1, chart_col2 = st.columns([1.2, 1])
    counts = Counter(h["specialty"] for h in st.session_state.history)
    specialties = list(counts.keys())
    tallies = list(counts.values())

    palette = ["#0ea5e9", "#22c55e", "#f59e0b", "#a855f7", "#ef4444",
               "#14b8a6", "#3b82f6", "#ec4899", "#84cc16", "#f97316"]
    colors = [palette[i % len(palette)] for i in range(len(specialties))]

    with chart_col1:
        bar_fig = go.Figure(go.Bar(
            x=tallies,
            y=specialties,
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=tallies,
            textposition="outside",
        ))
        bar_fig.update_layout(
            height=max(260, 48 * len(specialties)),
            margin=dict(l=10, r=30, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=True, gridcolor="#eef2f7", zeroline=False),
            yaxis=dict(showgrid=False, autorange="reversed"),
            font=dict(family="Inter", color="#334155", size=13),
        )
        st.plotly_chart(bar_fig, use_container_width=True, config={"displayModeBar": False})

    with chart_col2:
        donut_fig = go.Figure(go.Pie(
            labels=specialties,
            values=tallies,
            hole=0.62,
            marker=dict(colors=colors, line=dict(color="#ffffff", width=2)),
            textinfo="percent",
            textfont=dict(family="Inter", size=12, color="#334155"),
        ))
        donut_fig.update_layout(
            height=280,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, font=dict(size=11, color="#334155")),
            font=dict(family="Inter", color="#334155"),
            annotations=[dict(
                text=f"{len(st.session_state.history)}<br><span style='font-size:11px;color:#94a3b8'>Total</span>",
                x=0.5, y=0.5, showarrow=False, font=dict(size=22, color="#0f172a", family="Inter"),
            )],
        )
        st.plotly_chart(donut_fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="placeholder-card">
        📈 Specialty distribution charts will appear here after your first few records are routed.
    </div>
    """, unsafe_allow_html=True)