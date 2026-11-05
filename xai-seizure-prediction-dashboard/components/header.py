import streamlit as st
from typing import Optional
from datetime import datetime

def render_header(
    title: str = "EEG Analytics Dashboard",
    subtitle: str = "Advanced EEG seizure detection with SHAP-driven explainability and multi-model inference.",
    recording_name: Optional[str] = None,
    last_analysis: Optional[str] = None,
    dataset_context: str = "CHB-MIT Scalp EEG Database"
):
    """
    Renders the unified healthcare SaaS dashboard header.
    """
    meta_pills = []
    if dataset_context:
        meta_pills.append(f"<span class='meta-pill'>📁 {dataset_context}</span>")
    if recording_name:
        meta_pills.append(f"<span class='meta-pill' style='color:#34D399; border-color:rgba(52,211,153,0.3); background:rgba(52,211,153,0.08);'>🔬 {recording_name}</span>")
    if last_analysis:
        meta_pills.append(f"<span class='meta-pill' style='color:#94A3B8; border-color:rgba(148,163,184,0.25);'>🕒 Last: {last_analysis}</span>")
        
    pills_html = " ".join(meta_pills)
    
    html = f"""
    <div class="dashboard-header">
        <div class="title-row">
            <div>
                <h1 class="title">🧠 {title}</h1>
                <p class="subtitle">{subtitle}</p>
            </div>
            <div>
                {pills_html}
            </div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

