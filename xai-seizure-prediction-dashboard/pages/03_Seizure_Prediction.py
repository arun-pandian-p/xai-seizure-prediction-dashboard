import streamlit as st
import numpy as np
import pandas as pd
import time
from pathlib import Path

from components.sidebar import render_sidebar
from components.header import render_header
from components.metric_cards import render_kpi_card, render_kpi_grid, render_status_badge
from components.prediction_cards import plot_prediction_gauge, plot_prediction_timeline, render_prediction_summary
from src.data.loader import load_edf, align_channels
from src.prediction.inference import prepare_sequences, run_inference, load_seizure_intervals, is_seizure, MODEL_CONFIGS
from src.prediction.postprocessing import smooth_probabilities, compute_seizure_intervals, generate_clinical_summary
from src.database import save_analysis
from src.report import generate_pdf_report
from utils.config import ASSETS_DIR, DATA_DIR, RESEARCH_DISCLAIMER
from utils.session import init_session_state, get_active_eeg, set_prediction_results, get_prediction_results

init_session_state()

css_path = ASSETS_DIR / "styles.css"
if css_path.exists():
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

render_sidebar()

eeg_data, active_filename = get_active_eeg()

render_header(
    title="Seizure Prediction Workspace",
    subtitle="Deep learning inference engine with sliding temporal windows, threshold tuning, and ground-truth validation.",
    recording_name=active_filename
)

# 1. Verification of Active Recording
if eeg_data is None:
    st.info("👈 No active EEG recording found. Upload an EDF recording below or load a clinical benchmark sample.")
    col_up1, col_up2 = st.columns([2, 1])
    with col_up1:
        uploaded_edf = st.file_uploader("Upload Clinical EDF File", type=['edf'], key="pred_uploader")
        if uploaded_edf is not None:
            with st.spinner("Loading EDF recording..."):
                try:
                    loaded = load_edf(uploaded_edf)
                    from utils.session import set_active_eeg
                    set_active_eeg(loaded, uploaded_edf.name)
                    st.success("EDF loaded successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to load: {e}")
    with col_up2:
        sample_path = DATA_DIR / "chb02_16+.edf"
        if sample_path.exists():
            if st.button("Load CHB-02 Sample (EDF)", use_container_width=True, key="pred_load_sample"):
                from utils.session import set_active_eeg
                loaded = load_edf(str(sample_path))
                set_active_eeg(loaded, "chb02_16+.edf")
                st.rerun()
    st.stop()

# 2. Pipeline Configuration Section
with st.container():
    st.markdown("### ⚙️ Inference Configuration")
    cfg_col1, cfg_col2, cfg_col3 = st.columns([1.5, 1, 1])
    
    with cfg_col1:
        model_name = st.selectbox(
            "Select Deep Learning Model",
            options=list(MODEL_CONFIGS.keys()),
            index=0,
            help="Select the deep neural network architecture to run inference on."
        )
        config = MODEL_CONFIGS[model_name]
        
    with cfg_col2:
        threshold = st.slider(
            "Decision Threshold",
            min_value=0.10,
            max_value=0.90,
            value=0.50,
            step=0.05,
            help="Probability cutoff for classifying a window as ictal (seizure)."
        )
        
    with cfg_col3:
        enable_shap = st.checkbox("Compute SHAP Attributions", value=True, help="Derives channel biomarker attributions.")
        top_k = st.slider("Top Biomarker Channels", 1, 18, 5) if enable_shap else 5

    param_col1, param_col2, param_col3 = st.columns(3)
    with param_col1:
        window_sec = st.slider("Window Size (seconds)", 1.0, 10.0, 4.0, 0.5, help="Temporal sequence length.")
    with param_col2:
        step_sec = st.slider("Step Size (seconds)", 0.5, 5.0, 2.0, 0.5, help="Overlap step between windows.")
    with param_col3:
        seizures_file = st.file_uploader("Upload Ground-Truth (.seizures)", type=['seizures', 'txt'], key="seizures_uploader", help="Optional clinical annotations for accuracy calculation.")

# 3. Channel Alignment Check
raw_mne = eeg_data['raw']
sfreq = eeg_data['sfreq']
aligned_data, valid_channels = align_channels(raw_mne, config['channels'])

st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

# 4. Run Prediction Button
run_pred = st.button("🚀 Run Seizure Prediction", use_container_width=True, type="primary")

model_path = config["filename"]
results = get_prediction_results()

if run_pred:
    if not Path(model_path).exists():
        st.error(f"Model checkpoint `{model_path}` was not found on disk.")
    else:
        with st.spinner(f"Executing {model_name} neural network inference & XAI pipeline..."):
            try:
                window_size = int(window_sec * sfreq)
                sequences, seq_times = prepare_sequences(
                    aligned_data, sfreq, config, window_size=window_size, step_sec=step_sec
                )
                
                predictions, probabilities, input_shape, top_ch_indices, ch_importance, shap_vals, test_sample = run_inference(
                    sequences, model_path, config, shap_enabled=enable_shap, top_k=top_k, threshold=threshold
                )
                
                # Ground truth comparison if provided
                intervals_gt = None
                accuracy = None
                if seizures_file is not None:
                    intervals_gt = load_seizure_intervals(seizures_file.getbuffer())
                    if intervals_gt and len(intervals_gt) > 0 and len(predictions) > 0:
                        correct = sum(1 for t, p in zip(seq_times, predictions) if p == is_seizure(t, intervals_gt))
                        accuracy = float(correct / len(predictions) * 100)
                
                # Smoothed risk score
                smoothed_probs = smooth_probabilities(probabilities)
                risk_score = float(np.max(smoothed_probs)) if len(smoothed_probs) > 0 else 0.0
                
                top_ch_names = [valid_channels[int(i)] for i in top_ch_indices] if (enable_shap and top_ch_indices is not None) else []
                detected_intervals = compute_seizure_intervals(predictions, seq_times)
                
                report_content = generate_clinical_summary(
                    filename=active_filename,
                    model_name=model_name,
                    risk_score=risk_score,
                    detections=int(np.sum(predictions)),
                    total_segments=len(sequences),
                    top_channels=top_ch_names,
                    intervals=detected_intervals,
                    accuracy=accuracy
                )
                
                # Store in database
                save_analysis(
                    filename=active_filename,
                    model_name=model_name,
                    risk_score=risk_score,
                    detections=int(np.sum(predictions)),
                    total_segments=len(sequences),
                    accuracy=accuracy,
                    top_channels=top_ch_names,
                    report_content=report_content
                )
                
                results = {
                    "model_name": model_name,
                    "threshold": threshold,
                    "sequences": sequences,
                    "seq_times": seq_times,
                    "predictions": predictions,
                    "probabilities": probabilities,
                    "smoothed_probs": smoothed_probs,
                    "risk_score": risk_score,
                    "top_channels": top_ch_names,
                    "channel_importance": ch_importance,
                    "shap_vals": shap_vals,
                    "test_sample": test_sample,
                    "intervals": detected_intervals,
                    "accuracy": accuracy,
                    "report_content": report_content
                }
                set_prediction_results(results)
                st.success("Inference completed successfully!")
            except Exception as e:
                st.error(f"Inference Pipeline Error: {e}")

# 5. Render Prediction Dashboard Output
if results is not None:
    st.markdown("---")
    st.markdown("### 📊 Prediction Results & Diagnostic Evidence")
    
    # Summary Cards
    render_prediction_summary(
        risk_score=results['risk_score'],
        threshold=results['threshold'],
        detections=int(np.sum(results['predictions'])),
        total_segments=len(results['sequences']),
        intervals=results['intervals'],
        top_channels=results['top_channels']
    )
    
    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
    
    # Gauge and Key Metrics Row
    res_col1, res_col2 = st.columns([1, 1.5])
    with res_col1:
        st.plotly_chart(
            plot_prediction_gauge(results['risk_score'], threshold=results['threshold']),
            use_container_width=True
        )
    with res_col2:
        kpi_res = [
            {"title": "Total Segments", "value": f"{len(results['sequences'])}", "subtitle": "Evaluated windows", "accent": "cyan"},
            {"title": "Ictal Detections", "value": f"{int(np.sum(results['predictions']))}", "subtitle": "Flagged positive", "accent": "rose" if np.sum(results['predictions']) > 0 else "emerald"},
            {"title": "Validation Accuracy", "value": f"{results['accuracy']:.2f}%" if results['accuracy'] is not None else "N/A", "subtitle": "Against ground-truth", "accent": "teal"}
        ]
        render_kpi_grid(kpi_res)
        
        if results['intervals']:
            st.markdown("##### Detected Ictal Onset Intervals")
            df_int = pd.DataFrame(results['intervals'])
            df_int.columns = ["Onset (s)", "Offset (s)", "Duration (s)"]
            st.dataframe(df_int, use_container_width=True, hide_index=True)
            
    # Timeline
    st.markdown("#### Temporal Probability Trajectory")
    st.plotly_chart(
        plot_prediction_timeline(
            results['seq_times'],
            results['probabilities'],
            results['predictions'],
            threshold=results['threshold']
        ),
        use_container_width=True
    )
    
    # Report Export Actions
    st.markdown("---")
    st.markdown("### 📥 Export Clinical Diagnostic Report")
    exp_col1, exp_col2 = st.columns(2)
    with exp_col1:
        st.download_button(
            "📥 Download Markdown Report",
            results['report_content'],
            file_name=f"clinical_report_{active_filename}_{int(time.time())}.md",
            mime="text/markdown",
            use_container_width=True
        )
    with exp_col2:
        try:
            pdf_bytes = generate_pdf_report({
                'filename': active_filename,
                'model_name': results['model_name'],
                'risk_score': results['risk_score'],
                'total_segments': len(results['sequences']),
                'detections': int(np.sum(results['predictions'])),
                'accuracy': f"{results['accuracy']:.2f}%" if results['accuracy'] is not None else "N/A",
                'top_channels': results['top_channels'],
                'clinical_notes': "Seizure activity detected." if results['risk_score'] >= results['threshold'] else "Physiological / Non-seizure baseline."
            })
            st.download_button(
                "📄 Download Official PDF Report",
                bytes(pdf_bytes),
                file_name=f"clinical_report_{active_filename}_{int(time.time())}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        except Exception as pdf_err:
            st.error(f"PDF generation error: {pdf_err}")
else:
    st.info("💡 Adjust inference parameters above and click **'Run Seizure Prediction'** to generate results.")

