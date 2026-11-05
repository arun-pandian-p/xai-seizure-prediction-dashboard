import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

from components.sidebar import render_sidebar
from components.header import render_header
from components.metric_cards import render_kpi_card, render_kpi_grid
from src.explainability.lime_explainer import get_lime_artifacts, has_lime_support
from utils.config import ASSETS_DIR, RESULTS_DIR
from utils.session import init_session_state, get_active_eeg, get_prediction_results

init_session_state()

css_path = ASSETS_DIR / "styles.css"
if css_path.exists():
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

render_sidebar()
_, active_filename = get_active_eeg()
pred_results = get_prediction_results()

render_header(
    title="Explainable AI (XAI) Intelligence Hub",
    subtitle="Model interpretability via SHAP gradient attribution, spatial biomarker selection, and LIME local explanations.",
    recording_name=active_filename
)

# Workspace Navigation Tabs
tab_active, tab_global_shap, tab_lime, tab_attention = st.tabs([
    "🔍 Session Prediction XAI",
    "🐝 Global SHAP Attributions",
    "🍋 LIME & Comparative XAI",
    "👁️ Transformer Attention Maps"
])

with tab_active:
    st.markdown("### 🔬 Session Interpretability: Active Recording")
    
    if pred_results is None:
        st.info("💡 Run an analysis on the **'Seizure Prediction'** page with SHAP enabled to generate live per-window explanations.")
    else:
        st.markdown(f"**Evaluated Model:** `{pred_results['model_name']}` | **Top Identified Biomarkers:** `{', '.join(pred_results['top_channels'])}`")
        
        ch_importance = pred_results.get('channel_importance')
        shap_vals = pred_results.get('shap_vals')
        test_sample = pred_results.get('test_sample')
        
        if ch_importance is not None:
            col_x1, col_x2 = st.columns(2)
            
            with col_x1:
                st.markdown("#### Global Channel Importance (SHAP)")
                valid_channels = pred_results.get('top_channels', [])
                # If valid_channels length != ch_importance, fallback to indexed channel list
                from src.data.loader import STANDARD_CHANNELS
                ch_labels = STANDARD_CHANNELS[:len(ch_importance)]
                df_imp = pd.DataFrame({
                    'Channel': ch_labels,
                    'Importance': ch_importance
                }).sort_values(by='Importance', ascending=True)
                
                fig_imp = px.bar(
                    df_imp,
                    x='Importance',
                    y='Channel',
                    orientation='h',
                    title="Mean |SHAP Value| Across Channels",
                    color='Importance',
                    color_continuous_scale="Viridis",
                    template="plotly_dark"
                )
                fig_imp.update_layout(
                    paper_bgcolor='#151F33',
                    plot_bgcolor='#0B1020',
                    height=400,
                    margin=dict(l=40, r=20, t=40, b=30)
                )
                st.plotly_chart(fig_imp, use_container_width=True)
                
            with col_x2:
                st.markdown("#### Temporal Feature Impact on Risk")
                if shap_vals is not None and test_sample is not None:
                    try:
                        sv_agg = np.mean(shap_vals, axis=1) # average over time
                        data_agg = np.mean(test_sample.numpy() if hasattr(test_sample, 'numpy') else test_sample, axis=1)
                        
                        rows = []
                        for i in range(min(len(ch_labels), sv_agg.shape[1])):
                            for b in range(sv_agg.shape[0]):
                                rows.append({
                                    "Channel": ch_labels[i],
                                    "SHAP": float(sv_agg[b, i]),
                                    "Signal": float(data_agg[b, i])
                                })
                        df_shap = pd.DataFrame(rows)
                        fig_bee = px.scatter(
                            df_shap,
                            x="SHAP",
                            y="Channel",
                            color="Signal",
                            title="SHAP Value vs. Raw Amplitude",
                            color_continuous_scale="RdBu_r",
                            template="plotly_dark"
                        )
                        fig_bee.update_layout(
                            paper_bgcolor='#151F33',
                            plot_bgcolor='#0B1020',
                            height=400,
                            margin=dict(l=40, r=20, t=40, b=30)
                        )
                        st.plotly_chart(fig_bee, use_container_width=True)
                    except Exception as e_bee:
                        st.warning(f"Could not construct beeswarm plot: {e_bee}")
                else:
                    st.info("Detailed sample attributions not cached.")
        else:
            st.info("SHAP attribution was not computed for the current session. Enable 'Compute SHAP Attributions' in the prediction workspace.")

with tab_global_shap:
    st.markdown("### 🐝 Benchmark Global SHAP Research Findings")
    st.write(
        "Global explainability results from the CHB-MIT benchmark evaluation. "
        "The models consistently reveal focal temporal lobe activations as primary biomarkers of ictal onset."
    )
    
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        img_ch_imp = RESULTS_DIR / "shap_channel_importance.png"
        if img_ch_imp.exists():
            st.image(str(img_ch_imp), caption="Global Channel Ranking Across Patients", use_column_width=True)
        else:
            st.info("Saved channel importance artifact not found.")
            
    with col_g2:
        img_temporal = RESULTS_DIR / "shap_beeswarm_temporal.png"
        if img_temporal.exists():
            st.image(str(img_temporal), caption="Temporal SHAP Beeswarm Distribution", use_column_width=True)
        else:
            st.info("Saved temporal beeswarm artifact not found.")

with tab_lime:
    st.markdown("### 🍋 LIME (Local Interpretable Model-agnostic Explanations)")
    st.write(
        "Comparative evaluation between gradient-based SHAP and perturbation-based LIME local surrogates. "
        "LIME validates that non-linear spiking activations focus on synchronous spike-and-wave discharges."
    )
    
    lime_files = get_lime_artifacts()
    lime_path = Path(lime_files["lime_explanation"])
    comp_path = Path(lime_files["shap_vs_lime"])
    unified_path = Path(lime_files["unified_xai"])
    
    if lime_path.exists():
        col_l1, col_l2 = st.columns(2)
        with col_l1:
            st.image(str(lime_path), caption="LIME Local Feature Contributions for Ictal Windows", use_column_width=True)
        with col_l2:
            if comp_path.exists():
                st.image(str(comp_path), caption="SHAP vs LIME Agreement Report", use_column_width=True)
                
        if unified_path.exists():
            st.markdown("#### Unified Multi-Method XAI Report")
            st.image(str(unified_path), caption="Unified XAI Synthesis (SHAP, LIME, Integrated Gradients)", use_column_width=True)
    else:
        st.warning("LIME research artifacts are not present on disk.")

with tab_attention:
    st.markdown("### 👁️ Spiking Transformer Self-Attention Maps")
    st.write(
        "Self-attention weights extracted from the Spiking ConvNet-Transformer (SCT & DA-SCT) transformer encoder layers. "
        "Demonstrates temporal head focusing during pre-ictal and ictal transitions."
    )
    attn_path = RESULTS_DIR / "attention_map.png"
    if attn_path.exists():
        st.image(str(attn_path), caption="Multi-Head Transformer Attention Projection across EEG Windows", use_column_width=True)
    else:
        st.info("Attention map research artifact not found.")

