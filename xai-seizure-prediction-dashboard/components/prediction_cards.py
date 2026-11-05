import plotly.graph_objects as go
import streamlit as st
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional

PLOT_LAYOUT = dict(
    paper_bgcolor='#151F33',
    plot_bgcolor='#0B1020',
    font=dict(color='#F1F5F9', family='Inter, sans-serif'),
    margin=dict(l=45, r=25, t=40, b=35)
)

def plot_prediction_gauge(risk_score: float, threshold: float = 0.5, height: int = 280) -> go.Figure:
    """
    Renders semi-circle radial gauge for peak seizure risk percentage.
    """
    val_pct = float(risk_score * 100)
    bar_color = "#EF4444" if risk_score >= threshold else ("#F59E0B" if risk_score >= 0.3 else "#10B981")
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=val_pct,
        number={'suffix': "%", 'font': {'size': 36, 'color': '#FFFFFF'}},
        title={'text': "Peak Seizure Risk", 'font': {'size': 14, 'color': '#94A3B8'}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': '#94A3B8'},
            'bar': {'color': bar_color, 'thickness': 0.28},
            'bgcolor': '#0B1020',
            'borderwidth': 1,
            'bordercolor': 'rgba(255,255,255,0.08)',
            'steps': [
                {'range': [0, 40], 'color': 'rgba(16, 185, 129, 0.12)'},
                {'range': [40, 70], 'color': 'rgba(245, 158, 11, 0.12)'},
                {'range': [70, 100], 'color': 'rgba(239, 68, 68, 0.12)'}
            ],
            'threshold': {
                'line': {'color': '#EF4444', 'width': 3},
                'thickness': 0.8,
                'value': threshold * 100
            }
        }
    ))
    
    layout = PLOT_LAYOUT.copy()
    layout.update(height=height, margin=dict(l=30, r=30, t=50, b=20))
    fig.update_layout(layout)
    return fig

def plot_prediction_timeline(
    seq_times: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    threshold: float = 0.5,
    height: int = 320
) -> go.Figure:
    """
    Interactive prediction probability timeline with threshold guide and state coloring.
    """
    fig = go.Figure()
    
    bar_colors = ['#EF4444' if p == 1 else '#10B981' for p in predictions]
    
    # 1. Bar traces for each window
    fig.add_trace(go.Bar(
        x=seq_times.tolist(),
        y=probabilities.tolist(),
        marker=dict(color=bar_colors, line=dict(color='rgba(255,255,255,0.05)', width=0.5)),
        name='Seizure Probability',
        hoverinfo='x+y'
    ))
    
    # 2. Decision Threshold guide line
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="#F59E0B",
        annotation_text=f"Threshold ({threshold:.2f})",
        annotation_position="bottom right",
        annotation_font_color="#F59E0B"
    )
    
    layout = PLOT_LAYOUT.copy()
    layout.update(
        title=dict(text="Temporal Prediction Trajectory (Per-Window Seizure Probability)", font=dict(size=14, color='#F1F5F9')),
        xaxis_title="Time (seconds)",
        yaxis_title="Probability",
        yaxis=dict(range=[0, 1.05], gridcolor='rgba(255, 255, 255, 0.06)'),
        xaxis=dict(gridcolor='rgba(255, 255, 255, 0.06)'),
        height=height,
        showlegend=False
    )
    fig.update_layout(layout)
    return fig

def render_prediction_summary(
    risk_score: float,
    threshold: float,
    detections: int,
    total_segments: int,
    intervals: List[Dict[str, Any]],
    top_channels: List[str]
):
    """Renders structured summary cards for the prediction result."""
    is_seizure = risk_score >= threshold
    status_label = "SEIZURE DETECTED (HIGH RISK)" if is_seizure else "PHYSIOLOGICAL / NON-SEIZURE (LOW RISK)"
    accent_class = "accent-rose" if is_seizure else "accent-emerald"
    status_badge = "critical" if is_seizure else "normal"
    
    col1, col2 = st.columns([1.2, 1.8])
    with col1:
        st.markdown(f"""
            <div class="kpi-card {accent_class}" style="padding: 1.5rem; text-align: center;">
                <div class="kpi-label">Diagnostic Classification</div>
                <div style="font-size: 1.45rem; font-weight: 700; color: {'#EF4444' if is_seizure else '#10B981'}; margin: 0.5rem 0;">
                    {'⚠️ SEIZURE DETECTED' if is_seizure else '✅ NORMAL / LOW RISK'}
                </div>
                <div style="font-size: 0.85rem; color: #94A3B8;">
                    Peak Probability: <strong style="color: #FFFFFF;">{risk_score*100:.2f}%</strong> (Threshold: {threshold:.2f})
                </div>
            </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
            <div class="feature-card" style="margin-bottom: 0;">
                <div class="feature-card-header">Clinical Evidence Summary</div>
                <div style="font-size: 0.86rem; color: #F1F5F9; line-height: 1.6;">
                    <div>• <strong>Analyzed Windows:</strong> {total_segments} segments ({detections} ictal flags)</div>
                    <div>• <strong>Identified Biomarker Channels:</strong> {', '.join(top_channels) if top_channels else 'Standard 18 Channels'}</div>
                    <div>• <strong>Temporal Events:</strong> {len(intervals)} distinct seizure interval(s) detected</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

