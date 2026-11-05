"""
NeuroGuard XAI - Neurological Analytics & Seizure Prediction Platform
Streamlit Application Entry Point
"""
import streamlit as st
from pathlib import Path

from src.database import init_db
from utils.session import init_session_state
from utils.config import ASSETS_DIR

# Set Top-Level Streamlit Page Configuration
st.set_page_config(
    page_title="NeuroGuard XAI | EEG Seizure Analytics",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize persistence and session state
init_db()
init_session_state()

# Global CSS Injection
css_path = ASSETS_DIR / "styles.css"
if css_path.exists():
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Define Platform Navigation
nav_pages = [
    st.Page("pages/01_Overview.py", title="Overview", icon="🏠", default=True),
    st.Page("pages/02_EEG_Explorer.py", title="EEG Signal Explorer", icon="🌊"),
    st.Page("pages/03_Seizure_Prediction.py", title="Seizure Prediction", icon="🤖"),
    st.Page("pages/04_Model_Performance.py", title="Model Performance", icon="📊"),
    st.Page("pages/05_Explainable_AI.py", title="Explainable AI", icon="🔍"),
    st.Page("pages/06_Research_Insights.py", title="Research Insights", icon="🔬")
]

pg = st.navigation(nav_pages)
pg.run()
