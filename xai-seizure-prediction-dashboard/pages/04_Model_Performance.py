import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

from components.sidebar import render_sidebar
from components.header import render_header
from components.metric_cards import render_kpi_card, render_kpi_grid
from src.evaluation.metrics import load_model_comparison_metrics, get_confusion_matrix_data
from utils.config import ASSETS_DIR, RESULTS_DIR, MODEL_REGISTRY
from utils.session import init_session_state, get_active_eeg

init_session_state()

css_path = ASSETS_DIR / "styles.css"
if css_path.exists():
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

render_sidebar()
_, active_filename = get_active_eeg()

render_header(
    title="Model Performance & Evaluation",
    subtitle="Empirically validated performance metrics, confusion matrices, and ROC curves across neural architectures.",
    recording_name=active_filename
)

# 1. Load Genuine Metrics Artifact
metrics_df = load_model_comparison_metrics()

if metrics_df.empty:
    st.warning("Model evaluation metrics artifact `model_comparison.csv` was not found.")
    st.stop()

# 2. Executive KPI Header (Best Performing Architecture)
best_model = metrics_df.sort_values(by="Accuracy", ascending=False).iloc[0]

kpis = [
    {"title": "Benchmark Champion", "value": str(best_model['Model']), "subtitle": "Top overall architecture", "accent": "cyan"},
    {"title": "Peak Accuracy", "value": f"{best_model['Accuracy']*100:.2f}%", "subtitle": "CHB-MIT cross-validated", "accent": "emerald"},
    {"title": "Peak Sensitivity", "value": f"{best_model['Sensitivity']*100:.2f}%", "subtitle": "True seizure detection rate", "accent": "teal"},
    {"title": "Peak Specificity", "value": f"{best_model['Specificity']*100:.2f}%", "subtitle": "Non-seizure specificity", "accent": "amber"},
    {"title": "ROC AUC", "value": f"{best_model['AUC']:.5f}", "subtitle": "Area under ROC curve", "accent": "rose"}
]
render_kpi_grid(kpis)

st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

# 3. Model Architecture Comparison Table
st.markdown("### 📋 Cross-Model Evaluation Matrix")

formatted_df = metrics_df.copy()
pct_cols = ["Accuracy", "Precision", "Recall", "F1", "Sensitivity", "Specificity"]
for col in pct_cols:
    if col in formatted_df.columns:
        formatted_df[col] = (formatted_df[col] * 100).round(2).astype(str) + "%"
if "AUC" in formatted_df.columns:
    formatted_df["AUC"] = formatted_df["AUC"].round(5)

st.dataframe(formatted_df, use_container_width=True, hide_index=True)

# 4. Interactive Comparative Charts
st.markdown("---")
st.markdown("### 📊 Performance Metric Comparisons")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    fig_bar = px.bar(
        metrics_df,
        x="Model",
        y=["Accuracy", "Sensitivity", "Specificity", "F1"],
        barmode="group",
        title="Accuracy, Sensitivity, Specificity, & F1-Score Comparison",
        color_discrete_sequence=["#22D3EE", "#14B8A6", "#F59E0B", "#C084FC"],
        template="plotly_dark"
    )
    fig_bar.update_layout(
        paper_bgcolor='#151F33',
        plot_bgcolor='#0B1020',
        yaxis=dict(range=[0.95, 1.002], tickformat=".2%", gridcolor='rgba(255,255,255,0.06)'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.06)'),
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with chart_col2:
    # Radar Chart
    categories = ['Accuracy', 'Precision', 'Recall', 'F1', 'Specificity']
    fig_radar = go.Figure()
    
    colors = ['#22D3EE', '#14B8A6', '#F59E0B', '#EF4444']
    for idx, row in metrics_df.iterrows():
        vals = [row[c] for c in categories]
        vals.append(vals[0]) # close radar loop
        fig_radar.add_trace(go.Scatterpolar(
            r=vals,
            theta=categories + [categories[0]],
            fill='toself',
            name=row['Model'],
            line=dict(color=colors[idx % len(colors)])
        ))
        
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0.97, 1.0], gridcolor='rgba(255,255,255,0.1)'),
            bgcolor='#0B1020'
        ),
        paper_bgcolor='#151F33',
        height=380,
        title="Multi-Dimensional Performance Radar",
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_radar, use_container_width=True)

# 5. Deep Architecture Performance & Artifact Inspection
st.markdown("---")
st.markdown("### 🔬 Architecture Deep-Dive & Training Artifacts")

selected_model_inspect = st.selectbox(
    "Select Model to Inspect Evaluation Curves",
    options=metrics_df["Model"].tolist(),
    index=0
)

col_info, col_cm = st.columns([1.5, 1])

with col_info:
    model_meta = MODEL_REGISTRY.get(selected_model_inspect, {})
    st.markdown(f"#### Architecture: `{selected_model_inspect}`")
    st.write(f"**Description:** {model_meta.get('description', 'Neuromorphic seizure detection architecture.')}")
    st.write(f"**Structural Formulation:** {model_meta.get('architecture', 'Deep ConvNet + Transformer')}")
    st.write(f"**Input Specification:** 18 bipolar channels (10-20 system), 4s windows with 50% overlap.")

with col_cm:
    st.markdown("#### Estimated Confusion Matrix")
    cm_data = get_confusion_matrix_data(selected_model_inspect)
    if cm_data:
        cm_matrix = np.array([[cm_data['TN'], cm_data['FP']], [cm_data['FN'], cm_data['TP']]])
        fig_cm = px.imshow(
            cm_matrix,
            labels=dict(x="Predicted Class", y="True Class", color="Count"),
            x=["Normal", "Seizure"],
            y=["Normal", "Seizure"],
            text_auto=True,
            color_continuous_scale="Teal",
            template="plotly_dark"
        )
        fig_cm.update_layout(
            paper_bgcolor='#151F33',
            plot_bgcolor='#0B1020',
            height=280,
            margin=dict(l=40, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_cm, use_container_width=True)

# 6. Saved Research Figures Gallery
st.markdown("#### Saved Empirical Training & ROC Figures")
img_map = {
    "DA-GRL": RESULTS_DIR / "dagrl_results.png",
    "DA-SCT": RESULTS_DIR / "dasct_results.png",
    "GRL": RESULTS_DIR / "grl_results.png",
    "SCT": RESULTS_DIR / "sct_results.png"
}

target_img = img_map.get(selected_model_inspect)
if target_img and target_img.exists():
    st.image(str(target_img), caption=f"Validation Curves, Loss Trajectory, & Confusion Matrix: {selected_model_inspect}", use_column_width=True)
else:
    radar_img = RESULTS_DIR / "radar_comparison.png"
    if radar_img.exists():
        st.image(str(radar_img), caption="Comparative Multi-Model Evaluation Radar", use_column_width=True)

