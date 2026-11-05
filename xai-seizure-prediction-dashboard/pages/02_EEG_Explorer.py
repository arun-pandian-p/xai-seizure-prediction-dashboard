import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path

from components.sidebar import render_sidebar
from components.header import render_header
from components.metric_cards import render_kpi_card, render_kpi_grid
from components.eeg_charts import (
    plot_interactive_signal,
    plot_multichannel_signals,
    plot_psd_interactive,
    plot_band_power_distribution,
    plot_frequency_bands_stacked,
    plot_energy_heatmap,
    plot_topomap_mne
)
from src.data.loader import load_edf, align_channels, STANDARD_CHANNELS
from src.data.preprocessing import preprocess_signal, normalize_signal, extract_bands
from src.features.extraction import extract_features, compute_psd, extract_band_power
from utils.config import ASSETS_DIR, DATA_DIR
from utils.session import init_session_state, set_active_eeg, get_active_eeg, get_prediction_results

init_session_state()

css_path = ASSETS_DIR / "styles.css"
if css_path.exists():
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

render_sidebar()

eeg_data, active_filename = get_active_eeg()

render_header(
    title="EEG Signal Explorer",
    subtitle="Interactive multi-channel waveform exploration, spectral density analysis, and frequency decomposition.",
    recording_name=active_filename
)

# 1. File Upload / Ingestion Section
with st.expander("📁 Ingest EEG Recording (.EDF)", expanded=(eeg_data is None)):
    col_up1, col_up2 = st.columns([2, 1])
    with col_up1:
        uploaded_edf = st.file_uploader("Upload Clinical EDF Recording", type=['edf'], key="explorer_uploader")
        if uploaded_edf is not None:
            with st.spinner("Parsing and indexing EDF recording..."):
                try:
                    loaded = load_edf(uploaded_edf)
                    set_active_eeg(loaded, uploaded_edf.name)
                    eeg_data = loaded
                    active_filename = uploaded_edf.name
                    st.success(f"Successfully loaded '{uploaded_edf.name}' ({loaded['duration']:.1f}s, {loaded['sfreq']:.0f}Hz)")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to load EDF: {e}")
    with col_up2:
        st.markdown("#### Sample Clinical Data")
        st.write("Or load one of the repository benchmark recordings:")
        sample_path = DATA_DIR / "chb02_16+.edf"
        if sample_path.exists():
            if st.button("Load 'chb02_16+.edf' (CHB-MIT)", use_container_width=True):
                with st.spinner("Loading sample CHB-MIT recording..."):
                    try:
                        loaded = load_edf(str(sample_path))
                        set_active_eeg(loaded, "chb02_16+.edf")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Sample load error: {e}")

if eeg_data is None:
    st.info("👈 Please upload an EDF file or load a clinical sample above to explore the EEG signal.")
    st.stop()

# 2. Recording Metadata KPI Grid
raw_mne = eeg_data['raw']
sfreq = eeg_data['sfreq']
duration = eeg_data['duration']
total_channels = len(eeg_data['ch_names'])

meta_kpis = [
    {"title": "Recording Duration", "value": f"{duration:.1f}s", "subtitle": f"{duration/60:.1f} minutes", "accent": "cyan"},
    {"title": "Sampling Frequency", "value": f"{sfreq:.0f} Hz", "subtitle": f"{1000/sfreq:.2f} ms resolution", "accent": "teal"},
    {"title": "Available Channels", "value": str(total_channels), "subtitle": "Electrodes indexed", "accent": "amber"},
    {"title": "Total Samples", "value": f"{eeg_data['n_samples']:,}", "subtitle": "Temporal points", "accent": "emerald"}
]
render_kpi_grid(meta_kpis)

st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

# 3. Channel Alignment & Controls
aligned_data, valid_channels = align_channels(raw_mne, 18)

# Visualization Controls in Columns
ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1.5, 1, 1])

with ctrl_col1:
    selected_ch = st.selectbox(
        "Select Primary Channel",
        options=valid_channels,
        index=0,
        help="Select electrode channel for single-trace deep dive and spectral analysis."
    )
    ch_idx = valid_channels.index(selected_ch) if selected_ch in valid_channels else 0

with ctrl_col2:
    apply_bp = st.checkbox("Bandpass Filter (0.5 - 45 Hz)", value=True, help="Attenuates slow drift and high-frequency EMG noise.")
    apply_notch = st.checkbox("Notch Filter (50 Hz)", value=True, help="Attenuates AC power-line hum.")

with ctrl_col3:
    apply_zscore = st.checkbox("Z-Score Normalization", value=False, help="Centers mean to 0 with unit variance.")

# Zoom Time Window Slider
max_window = min(duration, 60.0)
zoom_range = st.slider(
    "Temporal Zoom Window (seconds)",
    min_value=0.0,
    max_value=float(duration),
    value=(0.0, float(max_window)),
    step=1.0,
    help="Select the visible time range for waveforms."
)

start_idx = int(zoom_range[0] * sfreq)
end_idx = int(zoom_range[1] * sfreq)

# Raw and Processed Zoomed Traces
raw_zoomed = aligned_data[ch_idx, start_idx:end_idx]
time_zoomed = eeg_data['times'][start_idx:end_idx]

processed_zoomed = raw_zoomed.copy()
if apply_bp:
    processed_zoomed = preprocess_signal(processed_zoomed, sfreq, lowcut=0.5, highcut=45.0, apply_notch=apply_notch)
if apply_zscore:
    processed_zoomed = normalize_signal(processed_zoomed)

# Check if prediction results exist to highlight detected seizure windows
pred_results = get_prediction_results()

# Workspace Tabs
tab_wave, tab_multi, tab_spectral, tab_topo, tab_heatmap = st.tabs([
    "📈 Single Channel Waveform",
    "📊 Multi-Channel Montage",
    "🌊 Spectral & Frequency Bands",
    "🗺️ Brain Topography",
    "🔥 Spatial Activity Heatmap"
])

with tab_wave:
    st.markdown(f"#### Waveform Analysis: `{selected_ch}`")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            plot_interactive_signal(time_zoomed, raw_zoomed, title=f"Raw Signal: {selected_ch}", color="#38BDF8"),
            use_container_width=True
        )
    with c2:
        st.plotly_chart(
            plot_interactive_signal(time_zoomed, processed_zoomed, title=f"Filtered Signal: {selected_ch}", color="#22D3EE"),
            use_container_width=True
        )
        
    # Statistical Summary
    feats = extract_features(processed_zoomed, sfreq)
    stat_kpis = [
        {"title": "Mean Amplitude", "value": f"{feats['Mean']:.2f} µV", "accent": "cyan"},
        {"title": "Signal Variance", "value": f"{feats['Variance']:.2f}", "accent": "teal"},
        {"title": "Standard Deviation", "value": f"{feats['Std Dev']:.2f} µV", "accent": "amber"},
        {"title": "Shannon Entropy", "value": f"{feats['Entropy']:.2f}", "accent": "emerald"}
    ]
    render_kpi_grid(stat_kpis)

with tab_multi:
    st.markdown("#### Clinical 18-Channel Bipolar Montage")
    offset = st.slider("Channel Vertical Spacing (µV)", min_value=20.0, max_value=300.0, value=80.0, step=10.0)
    
    # Sliced multi-channel segment
    multi_slice = aligned_data[:, start_idx:end_idx]
    fig_multi = plot_multichannel_signals(time_zoomed, multi_slice, valid_channels, offset=offset)
    st.plotly_chart(fig_multi, use_container_width=True)

with tab_spectral:
    st.markdown(f"#### Spectral Decomposition: `{selected_ch}`")
    freqs, psd = compute_psd(processed_zoomed, sfreq)
    band_powers = extract_band_power(freqs, psd)
    
    s_col1, s_col2 = st.columns(2)
    with s_col1:
        st.plotly_chart(plot_psd_interactive(freqs, psd, title=f"Power Spectral Density (Welch) - {selected_ch}"), use_container_width=True)
    with s_col2:
        st.plotly_chart(plot_band_power_distribution(band_powers), use_container_width=True)
        
    st.markdown("#### Rhythm Waveforms (Bandpass Decomposed)")
    bands_dict = extract_bands(processed_zoomed, sfreq)
    st.plotly_chart(plot_frequency_bands_stacked(time_zoomed, bands_dict), use_container_width=True)

with tab_topo:
    st.markdown("#### 🗺️ Brain Surface Topography (10-20 Standard Montage)")
    st.write("Calculates topographic spectral power across Delta, Theta, Alpha, and Beta frequency bands.")
    with st.spinner("Generating topographic scalp projections..."):
        t_center = (zoom_range[0] + zoom_range[1]) / 2.0
        fig_topo = plot_topomap_mne(raw_mne, t_center, sfreq, duration)
        if fig_topo is not None:
            st.pyplot(fig_topo)
        else:
            st.warning("Standard 10-20 montage coordinates could not be mapped for this electrode configuration.")

with tab_heatmap:
    st.markdown("#### Spatial Variance Heatmap")
    st.write("Visualizes multi-channel energy distribution across 2-second windows.")
    fig_hm = plot_energy_heatmap(aligned_data, valid_channels, sfreq)
    if fig_hm is not None:
        st.plotly_chart(fig_hm, use_container_width=True)
    else:
        st.warning("Insufficient recording duration to compute spatial heatmap.")

