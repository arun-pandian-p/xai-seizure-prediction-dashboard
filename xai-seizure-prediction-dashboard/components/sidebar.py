import streamlit as st
import os
from pathlib import Path
from utils.config import APP_VERSION, RESEARCH_DISCLAIMER, ASSETS_DIR, DATA_DIR
from src.data.loader import load_edf
from utils.session import set_active_eeg, get_active_eeg

def render_sidebar():
    """
    Renders custom sidebar navigation, active file status, and footer.
    """
    with st.sidebar:
        # 1. Branding Header
        logo_path = ASSETS_DIR / "logo.png"
        col_logo, col_text = st.columns([1, 3.5])
        with col_logo:
            if logo_path.exists():
                st.image(str(logo_path), width=48)
            else:
                st.markdown("<h2 style='margin:0;'>🧠</h2>", unsafe_allow_html=True)
        with col_text:
            st.markdown("""
                <div style='padding-top: 2px;'>
                    <div style='font-weight: 700; font-size: 1.15rem; color: #FFFFFF; line-height: 1.2;'>NEUROGUARD</div>
                    <div style='font-size: 0.72rem; color: #22D3EE; font-weight: 600; letter-spacing: 0.08em;'>XAI ANALYTICS</div>
                </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        st.markdown("<hr style='margin: 0.5rem 0 1rem 0; border: 0; border-top: 1px solid rgba(255,255,255,0.08);'>", unsafe_allow_html=True)
        
        # 2. Recording Status & Quick Load
        current_data, current_filename = get_active_eeg()
        st.markdown("<div style='font-size: 0.78rem; font-weight: 600; text-transform: uppercase; color: #94A3B8; letter-spacing: 0.05em; margin-bottom: 0.5rem;'>Active Recording</div>", unsafe_allow_html=True)
        
        if current_filename:
            st.markdown(f"""
                <div style='background: #151F33; border: 1px solid rgba(34,211,238,0.3); border-radius: 8px; padding: 8px 12px; margin-bottom: 12px;'>
                    <div style='font-size: 0.82rem; font-weight: 600; color: #F1F5F9; word-break: break-all;'>📄 {current_filename}</div>
                    <div style='font-size: 0.72rem; color: #34D399; margin-top: 2px;'>✓ Loaded ({current_data['duration']:.1f}s, {current_data['sfreq']:.0f}Hz)</div>
                </div>
            """, unsafe_allow_html=True)
            if st.button("Clear Active Recording", key="btn_clear_rec", use_container_width=True):
                set_active_eeg(None, None)
                st.rerun()
        else:
            st.markdown("""
                <div style='background: #151F33; border: 1px dashed rgba(255,255,255,0.15); border-radius: 8px; padding: 10px; margin-bottom: 10px; text-align: center;'>
                    <div style='font-size: 0.78rem; color: #94A3B8;'>No recording active</div>
                </div>
            """, unsafe_allow_html=True)
            
            # Sample quick loaders if sample files exist in data/
            sample_file = DATA_DIR / "chb02_16+.edf"
            alt_sample = Path("seizure") / "chb01_03.edf"
            
            target_sample = None
            sample_label = None
            if sample_file.exists():
                target_sample = sample_file
                sample_label = "Load CHB-02 Sample (EDF)"
            elif alt_sample.exists():
                target_sample = alt_sample
                sample_label = "Load CHB-01 Sample (EDF)"
                
            if target_sample and st.button(sample_label, use_container_width=True, key="btn_load_sample"):
                with st.spinner("Loading clinical EEG sample..."):
                    try:
                        eeg_dict = load_edf(str(target_sample))
                        set_active_eeg(eeg_dict, target_sample.name)
                        st.success(f"Loaded {target_sample.name}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to load sample: {e}")

        st.markdown("<hr style='margin: 1rem 0; border: 0; border-top: 1px solid rgba(255,255,255,0.08);'>", unsafe_allow_html=True)
        
        # 3. Footer with version and research disclaimer
        st.markdown(f"""
            <div class="sidebar-footer">
                <div style="font-weight: 600; color: #F1F5F9; margin-bottom: 4px;">NeuroGuard {APP_VERSION}</div>
                <div style="color: #64748B; font-size: 0.72rem; line-height: 1.4;">
                    Built with PyTorch, MNE-Python, and SHAP.
                </div>
                <div style="margin-top: 8px; padding: 6px 8px; background: rgba(245, 158, 11, 0.08); border: 1px solid rgba(245, 158, 11, 0.2); border-radius: 6px; font-size: 0.68rem; color: #FBBF24; line-height: 1.3;">
                    ⚠️ <strong>Research Disclaimer:</strong> Educational/investigative tool only. Not a medical device.
                </div>
            </div>
        """, unsafe_allow_html=True)

