import streamlit as st
from typing import Optional

def render_kpi_card(
    title: str,
    value: str,
    subtitle: Optional[str] = None,
    accent: str = "cyan",
    tooltip: Optional[str] = None
):
    """
    Renders a premium healthcare KPI card with restrained styling and high contrast.
    
    Args:
        title: Metric label (uppercase, muted).
        value: Primary number or metric string.
        subtitle: Additional contextual metadata.
        accent: Border accent color ("cyan", "teal", "amber", "rose", "emerald").
        tooltip: Optional hover text.
    """
    accent_class = f"accent-{accent}"
    tooltip_html = f"<span class='info-tooltip' title='{tooltip}'>ⓘ</span>" if tooltip else ""
    subtitle_html = f"<div class='kpi-subtitle'>{subtitle}</div>" if subtitle else ""
    
    html = f"""
    <div class="kpi-card {accent_class}">
        <div>
            <div class="kpi-label">{title} {tooltip_html}</div>
            <div class="kpi-value">{value}</div>
        </div>
        {subtitle_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def render_status_badge(status: str, label: Optional[str] = None) -> str:
    """
    Generates HTML for a pill-style status badge.
    Status can be: 'critical', 'normal', 'warning', 'neutral'.
    """
    display_text = label if label else status.upper()
    status_lower = status.lower()
    
    if "seizure" in status_lower or "high" in status_lower or "critical" in status_lower:
        badge_class = "critical"
        icon = "⚠️"
    elif "normal" in status_lower or "safe" in status_lower or "low" in status_lower:
        badge_class = "normal"
        icon = "✅"
    elif "warn" in status_lower or "moderate" in status_lower:
        badge_class = "warning"
        icon = "⚡"
    else:
        badge_class = "neutral"
        icon = "ℹ️"
        
    return f"<span class='status-badge {badge_class}'>{icon} {display_text}</span>"

def render_kpi_grid(cards_data):
    """
    Renders an evenly spaced grid of KPI cards.
    cards_data is a list of dicts with keys: title, value, subtitle, accent.
    """
    cols = st.columns(len(cards_data))
    for col, data in zip(cols, cards_data):
        with col:
            render_kpi_card(
                title=data.get("title", ""),
                value=data.get("value", ""),
                subtitle=data.get("subtitle", None),
                accent=data.get("accent", "cyan"),
                tooltip=data.get("tooltip", None)
            )

