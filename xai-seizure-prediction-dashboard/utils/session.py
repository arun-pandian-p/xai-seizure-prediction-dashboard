"""
Streamlit session state management for cross-page persistence.
"""
import streamlit as st

def init_session_state():
    """Initializes standard session state keys if not present."""
    defaults = {
        "eeg_data": None,
        "eeg_filename": None,
        "seizures_data": None,
        "seizures_filename": None,
        "selected_model": "DA-GRL",
        "threshold": 0.5,
        "prediction_results": None,
        "current_page": "Overview",
        "theme_toggle": "dark",
        "shap_enabled": True,
        "top_k": 5,
        "active_channel": None,
        "zoom_range": None
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

def set_active_eeg(eeg_dict, filename):
    st.session_state.eeg_data = eeg_dict
    st.session_state.eeg_filename = filename
    # Clear previous predictions on new file
    st.session_state.prediction_results = None

def get_active_eeg():
    return st.session_state.get("eeg_data"), st.session_state.get("eeg_filename")

def set_prediction_results(results):
    st.session_state.prediction_results = results

def get_prediction_results():
    return st.session_state.get("prediction_results")

