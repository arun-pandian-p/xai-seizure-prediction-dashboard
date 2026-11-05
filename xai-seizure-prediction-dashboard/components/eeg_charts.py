import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from typing import Dict, List, Optional

PLOT_LAYOUT = dict(
    paper_bgcolor='#151F33',
    plot_bgcolor='#0B1020',
    font=dict(color='#F1F5F9', family='Inter, sans-serif'),
    margin=dict(l=45, r=25, t=40, b=35),
    xaxis=dict(gridcolor='rgba(255, 255, 255, 0.06)', zerolinecolor='rgba(255, 255, 255, 0.1)'),
    yaxis=dict(gridcolor='rgba(255, 255, 255, 0.06)', zerolinecolor='rgba(255, 255, 255, 0.1)')
)

def plot_interactive_signal(
    time: np.ndarray,
    data: np.ndarray,
    title: str = "EEG Signal",
    color: str = "#22D3EE",
    height: int = 350
) -> go.Figure:
    """Plots interactive single-channel EEG signal with zoom and pan."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time,
        y=data,
        mode='lines',
        name='Amplitude',
        line=dict(color=color, width=1.2)
    ))
    
    layout = PLOT_LAYOUT.copy()
    layout.update(
        title=dict(text=title, font=dict(size=14, color='#F1F5F9')),
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude (µV)",
        height=height,
        hovermode="x unified"
    )
    fig.update_layout(layout)
    return fig

def plot_multichannel_signals(
    time: np.ndarray,
    data_matrix: np.ndarray,
    ch_names: List[str],
    offset: float = 100.0,
    height: int = 600
) -> go.Figure:
    """
    Renders stacked multi-channel EEG montage traces (standard clinical presentation).
    """
    fig = go.Figure()
    n_ch = min(len(ch_names), data_matrix.shape[0])
    
    # Calculate offsets so channels don't overlap excessively
    tick_vals = []
    tick_text = []
    
    for i in range(n_ch):
        y_offset = (n_ch - 1 - i) * offset
        tick_vals.append(y_offset)
        tick_text.append(ch_names[i])
        
        fig.add_trace(go.Scatter(
            x=time,
            y=data_matrix[i, :] + y_offset,
            mode='lines',
            name=ch_names[i],
            line=dict(width=1, color='#22D3EE' if i % 2 == 0 else '#14B8A6'),
            hoverinfo='skip'
        ))
        
    layout = PLOT_LAYOUT.copy()
    layout.update(
        title=dict(text=f"Multi-Channel EEG Montage ({n_ch} Channels)", font=dict(size=14, color='#F1F5F9')),
        xaxis_title="Time (seconds)",
        yaxis=dict(
            tickmode='array',
            tickvals=tick_vals,
            ticktext=tick_text,
            showgrid=True,
            gridcolor='rgba(255, 255, 255, 0.05)'
        ),
        height=height,
        showlegend=False,
        hovermode=False
    )
    fig.update_layout(layout)
    return fig

def plot_psd_interactive(
    freqs: np.ndarray,
    psd: np.ndarray,
    title: str = "Power Spectral Density (Welch)",
    color: str = "#F59E0B",
    height: int = 350
) -> go.Figure:
    """Plots Welch Power Spectral Density on logarithmic scale (dB/Hz)."""
    fig = go.Figure()
    psd_db = 10 * np.log10(np.maximum(psd, 1e-15))
    
    fig.add_trace(go.Scatter(
        x=freqs,
        y=psd_db,
        mode='lines',
        name='PSD (dB/Hz)',
        line=dict(color=color, width=1.5),
        fill='tozeroy',
        fillcolor='rgba(245, 158, 11, 0.08)'
    ))
    
    layout = PLOT_LAYOUT.copy()
    layout.update(
        title=dict(text=title, font=dict(size=14, color='#F1F5F9')),
        xaxis_title="Frequency (Hz)",
        yaxis_title="Power / Frequency (dB/Hz)",
        xaxis=dict(range=[0, 45], gridcolor='rgba(255, 255, 255, 0.06)'),
        height=height,
        hovermode="x unified"
    )
    fig.update_layout(layout)
    return fig

def plot_band_power_distribution(band_powers: Dict[str, float], height: int = 350) -> go.Figure:
    """Plots power distribution bar chart across EEG frequency bands."""
    df = pd.DataFrame(list(band_powers.items()), columns=['Band', 'Power'])
    total = df['Power'].sum()
    df['Relative'] = (df['Power'] / total * 100) if total > 0 else 0
    
    color_map = {
        'Delta (0.5-4 Hz)': '#38BDF8',
        'Theta (4-8 Hz)': '#2DD4BF',
        'Alpha (8-13 Hz)': '#FBBF24',
        'Beta (13-30 Hz)': '#F87171',
        'Gamma (30-45 Hz)': '#C084FC'
    }
    colors = [color_map.get(b, '#22D3EE') for b in df['Band']]
    
    fig = go.Figure(go.Bar(
        x=df['Band'],
        y=df['Relative'],
        marker=dict(color=colors, line=dict(color='rgba(255,255,255,0.1)', width=1)),
        text=[f"{v:.1f}%" for v in df['Relative']],
        textposition='auto',
        hoverinfo='x+y'
    ))
    
    layout = PLOT_LAYOUT.copy()
    layout.update(
        title=dict(text="Relative Frequency Band Power (%)", font=dict(size=14, color='#F1F5F9')),
        xaxis_title="Frequency Band",
        yaxis_title="Relative Power (%)",
        height=height
    )
    fig.update_layout(layout)
    return fig

def plot_frequency_bands_stacked(time: np.ndarray, bands_dict: Dict[str, np.ndarray], height: int = 650) -> go.Figure:
    """Renders individual rhythm bands in shared-timeline subplots."""
    num_bands = len(bands_dict)
    fig = make_subplots(
        rows=num_bands,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=list(bands_dict.keys())
    )
    
    colors = ['#38BDF8', '#2DD4BF', '#FBBF24', '#F87171', '#C084FC']
    for i, ((band_name, band_data), color) in enumerate(zip(bands_dict.items(), colors)):
        fig.add_trace(
            go.Scatter(
                x=time,
                y=band_data,
                mode='lines',
                name=band_name,
                line=dict(color=color, width=1)
            ),
            row=i + 1, col=1
        )
        fig.update_yaxes(gridcolor='rgba(255, 255, 255, 0.05)', row=i + 1, col=1)
        fig.update_xaxes(gridcolor='rgba(255, 255, 255, 0.05)', row=i + 1, col=1)
        
    layout = PLOT_LAYOUT.copy()
    layout.update(
        title=dict(text="EEG Rhythm Decomposition", font=dict(size=14, color='#F1F5F9')),
        height=height,
        showlegend=False,
        margin=dict(l=45, r=25, t=50, b=35)
    )
    fig.update_layout(layout)
    return fig

def plot_energy_heatmap(aligned_data: np.ndarray, valid_channels: List[str], sfreq: float, height: int = 380) -> Optional[go.Figure]:
    """Generates spatial activity heatmap based on logarithmic signal variance."""
    window_size_hm = int(2.0 * sfreq)
    total_samples_hm = aligned_data.shape[1]
    num_windows_hm = total_samples_hm // window_size_hm
    if num_windows_hm <= 1:
        return None
        
    hm_data = aligned_data[:, :num_windows_hm * window_size_hm].reshape(aligned_data.shape[0], num_windows_hm, window_size_hm)
    energy_hm = np.log10(np.var(hm_data, axis=2) + 1e-7)
    
    fig = px.imshow(
        energy_hm,
        labels=dict(x="Time Step (2s Windows)", y="EEG Channel", color="Log10 Var"),
        y=valid_channels,
        aspect="auto",
        color_continuous_scale="Viridis",
        template="plotly_dark"
    )
    fig.update_layout(
        paper_bgcolor='#151F33',
        plot_bgcolor='#0B1020',
        height=height,
        margin=dict(l=45, r=25, t=35, b=35),
        font=dict(color='#F1F5F9', family='Inter, sans-serif')
    )
    return fig

def plot_topomap_mne(raw, t_mid: float, sfreq: float, duration: float):
    """
    Renders 4-band MNE topographic brain maps (Delta, Theta, Alpha, Beta).
    Returns matplotlib figure.
    """
    from mne.viz import plot_topomap
    from src.data.preprocessing import ensure_montage
    
    raw_topo, success = ensure_montage(raw.copy())
    if not success:
        return None
        
    info = raw_topo.info
    chs_with_pos = [ch['ch_name'] for ch in info['chs'] if (ch['loc'][0] != 0 or ch['loc'][1] != 0) and "_" not in ch['ch_name']]
    if len(chs_with_pos) < 4:
        return None
        
    raw_topo.pick_channels(chs_with_pos)
    start_sample = max(0, int((t_mid - 15) * sfreq))
    end_sample = min(int(duration * sfreq), int((t_mid + 15) * sfreq))
    data_topo, _ = raw_topo[:, start_sample:end_sample]
    
    topo_bands = {'Delta (0.5-4Hz)': (0.5, 4.0), 'Theta (4-8Hz)': (4.0, 8.0), 'Alpha (8-13Hz)': (8.0, 13.0), 'Beta (13-30Hz)': (13.0, 30.0)}
    fig, axes = plt.subplots(1, 4, figsize=(12, 3), facecolor='#151F33')
    
    for idx, (b_name, (low, high)) in enumerate(topo_bands.items()):
        powers = []
        for ch in range(data_topo.shape[0]):
            f, p = signal.welch(data_topo[ch], sfreq, nperseg=int(min(len(data_topo[ch]), 2 * sfreq)))
            powers.append(np.mean(p[np.logical_and(f >= low, f <= high)]))
        plot_topomap(np.array(powers), raw_topo.info, axes=axes[idx], show=False, cmap='RdBu_r', contours=5)
        axes[idx].set_title(b_name, color='#F1F5F9', fontsize=10, pad=8)
        
    plt.tight_layout()
    return fig

