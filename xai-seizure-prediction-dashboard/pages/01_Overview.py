import streamlit as st
import pandas as pd
from pathlib import Path

from components.sidebar import render_sidebar
from components.header import render_header
from components.metric_cards import render_kpi_card, render_kpi_grid
from src.database import init_db, get_history, delete_analysis
from src.evaluation.metrics import load_model_comparison_metrics
from utils.config import ASSETS_DIR, RESULTS_DIR, RESEARCH_DISCLAIMER
from utils.session import init_session_state, get_active_eeg

# Ensure DB and session state are ready
init_db()
init_session_state()

# Inject CSS styling
css_path = ASSETS_DIR / "styles.css"
if css_path.exists():
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Render Sidebar
render_sidebar()

# Fetch Genuine Project Artifact Metrics
history = get_history()
metrics_df = load_model_comparison_metrics()

total_recordings = len(history)
total_windows = sum([item['total_segments'] for item in history if item.get('total_segments')]) if history else 0
last_timestamp = history[0]['timestamp'] if history else "No runs yet"

best_acc = "N/A"
best_sens = "N/A"
best_f1 = "N/A"
if not metrics_df.empty:
    best_row = metrics_df.sort_values(by="Accuracy", ascending=False).iloc[0]
    best_acc = f"{best_row['Accuracy']*100:.2f}%"
    best_sens = f"{best_row['Sensitivity']*100:.2f}%"
    best_f1 = f"{best_row['F1']*100:.2f}%"

_, active_filename = get_active_eeg()

# Top Header
render_header(
    title="Overview & Clinical Intelligence",
    subtitle="Enterprise neurological monitoring platform powered by Explainable Artificial Intelligence (XAI).",
    recording_name=active_filename,
    last_analysis=last_timestamp if history else None,
    dataset_context="CHB-MIT Scalp EEG Dataset"
)

# Research Safety Disclaimer Banner
st.markdown(f"""
    <div class="clinical-disclaimer">
        ⚠️ <strong>Investigative & Research Use Only:</strong> {RESEARCH_DISCLAIMER}
    </div>
""", unsafe_allow_html=True)

# Genuine KPI Cards Grid
kpis = [
    {
        "title": "Recordings Evaluated",
        "value": str(total_recordings),
        "subtitle": "Persistent historical sessions",
        "accent": "cyan",
        "tooltip": "Total patient sessions stored in SQLite repository"
    },
    {
        "title": "EEG Windows Processed",
        "value": f"{total_windows:,}",
        "subtitle": "Sliding window segments",
        "accent": "teal",
        "tooltip": "Multi-channel EEG sequences inferred across sessions"
    },
    {
        "title": "Top Model Accuracy",
        "value": best_acc,
        "subtitle": "DA-GRL Architecture",
        "accent": "emerald",
        "tooltip": "Empirically validated on CHB-MIT benchmark"
    },
    {
        "title": "Sensitivity (Recall)",
        "value": best_sens,
        "subtitle": "True positive seizure rate",
        "accent": "amber",
        "tooltip": "Validated seizure detection sensitivity"
    },
    {
        "title": "F1-Score",
        "value": best_f1,
        "subtitle": "Harmonic mean metric",
        "accent": "cyan",
        "tooltip": "Precision-recall balance across cross-validation"
    }
]
render_kpi_grid(kpis)

st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

# Main Grid: Platform Modules & System Architecture
col_left, col_right = st.columns([1.5, 1])

with col_left:
    st.markdown("### 🚀 Diagnostic Capabilities")
    
    cap_col1, cap_col2 = st.columns(2)
    with cap_col1:
        st.markdown("""
            <div class="kpi-card" style="padding: 1rem; margin-bottom: 0.75rem;">
                <div style="font-weight: 600; color: #22D3EE; font-size: 0.92rem;">🌊 EEG Signal Explorer</div>
                <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 4px;">
                    Full-bandwidth multi-channel clinical montage viewer, Welch PSD spectral density, and 5-band rhythm decomposition.
                </div>
            </div>
            <div class="kpi-card" style="padding: 1rem;">
                <div style="font-weight: 600; color: #14B8A6; font-size: 0.92rem;">🤖 Seizure Prediction</div>
                <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 4px;">
                    Multi-model deep neural inference with temporal sliding windows, threshold calibration, and ground-truth validation.
                </div>
            </div>
        """, unsafe_allow_html=True)
    with cap_col2:
        st.markdown("""
            <div class="kpi-card" style="padding: 1rem; margin-bottom: 0.75rem;">
                <div style="font-weight: 600; color: #F59E0B; font-size: 0.92rem;">🔍 Explainable AI (SHAP & LIME)</div>
                <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 4px;">
                    Model-specific gradient SHAP attribution, top-K channel biomarker selection, and comparative local feature attributions.
                </div>
            </div>
            <div class="kpi-card" style="padding: 1rem;">
                <div style="font-weight: 600; color: #38BDF8; font-size: 0.92rem;">📊 Model Performance</div>
                <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 4px;">
                    Empirical validation results comparing SCT, GRL, DA-SCT, and DA-GRL across accuracy, ROC AUC, and confusion matrices.
                </div>
            </div>
        """, unsafe_allow_html=True)

with col_right:
    st.markdown("### 🧠 Benchmark Model Overview")
    if not metrics_df.empty:
        disp_df = metrics_df[["Model", "Accuracy", "Sensitivity", "Specificity", "F1"]].copy()
        for col in ["Accuracy", "Sensitivity", "Specificity", "F1"]:
            disp_df[col] = (disp_df[col] * 100).round(2).astype(str) + "%"
        st.dataframe(disp_df, use_container_width=True, hide_index=True)
    else:
        st.info("Evaluation metrics file not found.")

st.markdown("---")

# Recent Analysis History Section
st.markdown("### 📜 Recent Analysis History")
if not history:
    st.info("No prior sessions recorded. Head to 'Seizure Prediction' to evaluate an EEG recording.")
else:
    for item in history[:5]:
        with st.expander(f"📁 {item['filename']} — {item['model_name']} ({item['timestamp']})"):
            c1, c2, c3 = st.columns([2, 1, 1])
            with c1:
                st.markdown(f"**Max Seizure Risk:** `{item['risk_score']*100:.2f}%`")
                st.markdown(f"**Segments:** `{item['total_segments']}` | **Detections:** `{item['detections']}`")
                st.markdown(f"**Top Biomarker Channels:** `{', '.join(item['top_channels'])}`")
            with c2:
                st.download_button(
                    "📥 Download Markdown Report",
                    item['report_content'],
                    file_name=f"report_{item['filename']}_{item['id']}.md",
                    key=f"dl_md_{item['id']}",
                    use_container_width=True
                )
            with c3:
                if st.button("🗑️ Delete Record", key=f"del_rec_{item['id']}", use_container_width=True):
                    delete_analysis(item['id'])
                    st.rerun()

