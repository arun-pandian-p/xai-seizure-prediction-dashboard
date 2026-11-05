import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

from components.sidebar import render_sidebar
from components.header import render_header
from components.metric_cards import render_kpi_card, render_kpi_grid
from src.database import get_history
from src.features.extraction import extract_features, compute_psd, extract_band_power
from src.data.loader import align_channels
from src.prediction.inference import load_seizure_intervals, is_seizure
from utils.config import ASSETS_DIR, RESULTS_DIR, RESEARCH_DISCLAIMER
from utils.session import init_session_state, get_active_eeg, get_prediction_results

init_session_state()

css_path = ASSETS_DIR / "styles.css"
if css_path.exists():
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

render_sidebar()
eeg_data, active_filename = get_active_eeg()
pred_results = get_prediction_results()

render_header(
    title="Research Insights & Clinical Analytics",
    subtitle="Exploratory Data Analysis (EDA), patient cohort statistics, feature distributions, and cross-session trends.",
    recording_name=active_filename
)

# Tabs
tab_eda, tab_feats, tab_cohort, tab_history = st.tabs([
    "📊 Dataset & Cohort EDA",
    "🧮 Interactive Biomarker Extractor",
    "👥 Patient Demographics & Class Balance",
    "📈 Historical Database Analytics"
])

with tab_eda:
    st.markdown("### 📊 CHB-MIT Scalp EEG Exploratory Data Analysis")
    st.write(
        "Empirical distributions and physiological signatures documented across 24 pediatric and adult epilepsy subjects "
        "comprising the clinical benchmark."
    )
    
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        img_norm_sz = RESULTS_DIR / "eda_normal_vs_seizure.png"
        if img_norm_sz.exists():
            st.image(str(img_norm_sz), caption="Waveform Dynamics: Physiological Baseline vs. Ictal Synchrony", use_column_width=True)
            
        img_psd = RESULTS_DIR / "eda_psd.png"
        if img_psd.exists():
            st.image(str(img_psd), caption="Power Spectral Density Shift During Seizure Onset", use_column_width=True)
            
    with col_e2:
        img_corr = RESULTS_DIR / "eda_channel_correlation.png"
        if img_corr.exists():
            st.image(str(img_corr), caption="Inter-Electrode Spatial Cross-Correlation Matrix", use_column_width=True)
            
        img_wavelet = RESULTS_DIR / "eda_wavelet.png"
        if img_wavelet.exists():
            st.image(str(img_wavelet), caption="Continuous Wavelet Transform (CWT) Scalogram", use_column_width=True)

with tab_feats:
    st.markdown("### 🧮 Interactive Multi-Feature Biomarker Extraction")
    st.write("Extract and correlate custom statistical and spectral features on the active EEG recording.")
    
    if eeg_data is None:
        st.info("💡 Load an EEG file on the 'EEG Explorer' or 'Seizure Prediction' page to run dynamic feature extraction.")
    else:
        available_feats = ['Mean', 'Variance', 'Std Dev', 'Skewness', 'Kurtosis', 'Entropy', 'RMS', 'Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
        selected_feats = st.multiselect("Select Biomarkers for Feature Matrix", available_feats, default=['Variance', 'Entropy', 'Alpha', 'Beta', 'Gamma'])
        
        aligned_data, valid_channels = align_channels(eeg_data['raw'], 18)
        sfreq = eeg_data['sfreq']
        
        if st.button("🚀 Compute Feature Covariance & Distributions", type="primary"):
            with st.spinner("Extracting multi-domain biomarker features across windows..."):
                try:
                    feat_list = []
                    window_sec = 4.0
                    step_sec = 2.0
                    w_samples = int(window_sec * sfreq)
                    step_samples = int(step_sec * sfreq)
                    
                    subsample = max(1, (aligned_data.shape[1] // w_samples) // 150)
                    window_idx = 0
                    
                    for s in range(0, aligned_data.shape[1] - w_samples + 1, step_samples * subsample):
                        seg = aligned_data[0, s:s + w_samples]
                        f_dict = extract_features(seg, sfreq)
                        freqs_s, psd_s = compute_psd(seg, sfreq)
                        b_dict = {k.split(' ')[0]: v for k, v in extract_band_power(freqs_s, psd_s).items()}
                        combined = {**f_dict, **b_dict}
                        
                        entry = {k: combined[k] for k in selected_feats if k in combined}
                        entry['Time'] = s / sfreq
                        feat_list.append(entry)
                        
                    df_feats = pd.DataFrame(feat_list)
                    if not df_feats.empty:
                        f_col1, f_col2 = st.columns(2)
                        with f_col1:
                            st.markdown("#### Biomarker Correlation Matrix")
                            corr_matrix = df_feats.drop(columns=['Time']).corr()
                            fig_corr = px.imshow(
                                corr_matrix,
                                text_auto=".2f",
                                color_continuous_scale="RdBu_r",
                                template="plotly_dark"
                            )
                            fig_corr.update_layout(paper_bgcolor='#151F33', plot_bgcolor='#0B1020')
                            st.plotly_chart(fig_corr, use_container_width=True)
                        with f_col2:
                            st.markdown("#### Biomarker Distributions")
                            df_melt = df_feats.melt(id_vars=['Time'], value_vars=selected_feats)
                            fig_box = px.box(
                                df_melt,
                                x="variable",
                                y="value",
                                color="variable",
                                template="plotly_dark",
                                points=False
                            )
                            fig_box.update_layout(paper_bgcolor='#151F33', plot_bgcolor='#0B1020', showlegend=False)
                            st.plotly_chart(fig_box, use_container_width=True)
                except Exception as e_feat:
                    st.error(f"Biomarker extraction failed: {e_feat}")

with tab_cohort:
    st.markdown("### 👥 Patient Demographics & Class Balance")
    st.write("Clinical cohort distributions across the 24 CHB-MIT subjects.")
    
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        img_imbal = RESULTS_DIR / "eda_class_imbalance.png"
        if img_imbal.exists():
            st.image(str(img_imbal), caption="Extreme Class Imbalance (Ictal vs Non-Ictal Hours)", use_column_width=True)
            
        img_patient = RESULTS_DIR / "eda_patient_statistics.png"
        if img_patient.exists():
            st.image(str(img_patient), caption="Total Duration and Seizure Burden per Subject", use_column_width=True)
            
    with col_c2:
        img_sz_stats = RESULTS_DIR / "eda_seizure_stats.png"
        if img_sz_stats.exists():
            st.image(str(img_sz_stats), caption="Seizure Duration and Onset Frequency Statistics", use_column_width=True)
            
        img_stat_feats = RESULTS_DIR / "eda_stat_features.png"
        if img_stat_feats.exists():
            st.image(str(img_stat_feats), caption="Cross-Subject Statistical Feature Distributions", use_column_width=True)

with tab_history:
    st.markdown("### 📈 Historical Database Analytics")
    history = get_history()
    
    if not history:
        st.info("No analyses recorded in database yet.")
    else:
        df_hist = pd.DataFrame(history)
        h_col1, h_col2 = st.columns(2)
        with h_col1:
            st.markdown("#### Max Risk Score Distribution")
            fig_hist_risk = px.histogram(
                df_hist,
                x="risk_score",
                nbins=15,
                title="Distribution of Max Risk Scores",
                color_discrete_sequence=["#22D3EE"],
                template="plotly_dark"
            )
            fig_hist_risk.update_layout(paper_bgcolor='#151F33', plot_bgcolor='#0B1020')
            st.plotly_chart(fig_hist_risk, use_container_width=True)
            
        with h_col2:
            st.markdown("#### Model Usage Frequency")
            fig_models = px.pie(
                df_hist,
                names="model_name",
                title="Evaluated Architectures in Sessions",
                color_discrete_sequence=["#22D3EE", "#14B8A6", "#F59E0B", "#EF4444", "#C084FC"],
                template="plotly_dark"
            )
            fig_models.update_layout(paper_bgcolor='#151F33', plot_bgcolor='#0B1020')
            st.plotly_chart(fig_models, use_container_width=True)

