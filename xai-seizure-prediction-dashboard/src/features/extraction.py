import numpy as np
from scipy import signal, stats
from typing import Dict, Tuple

def extract_features(data: np.ndarray, sfreq: float) -> Dict[str, float]:
    """
    Extracts time-domain statistical features from an EEG segment.
    """
    def shannon_entropy(x: np.ndarray) -> float:
        hist, _ = np.histogram(x, bins=20, density=True)
        return float(stats.entropy(hist + 1e-9))

    features = {
        'Mean': float(np.mean(data)),
        'Variance': float(np.var(data)),
        'Std Dev': float(np.std(data)),
        'Min': float(np.min(data)),
        'Max': float(np.max(data)),
        'Skewness': float(stats.skew(data)),
        'Kurtosis': float(stats.kurtosis(data)),
        'Entropy': shannon_entropy(data),
        'RMS': float(np.sqrt(np.mean(data**2)))
    }
    return features

def compute_psd(data: np.ndarray, sfreq: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes Power Spectral Density using Welch's method.
    """
    nperseg = min(int(4 * sfreq), len(data))
    if nperseg <= 0:
        nperseg = len(data)
    freqs, psd = signal.welch(data, sfreq, nperseg=nperseg)
    return freqs, psd

def extract_band_power(freqs: np.ndarray, psd: np.ndarray) -> Dict[str, float]:
    """
    Calculates absolute power across standard EEG frequency bands.
    """
    bands = {
        'Delta (0.5-4 Hz)': (0.5, 4.0),
        'Theta (4-8 Hz)': (4.0, 8.0),
        'Alpha (8-13 Hz)': (8.0, 13.0),
        'Beta (13-30 Hz)': (13.0, 30.0),
        'Gamma (30-45 Hz)': (30.0, 45.0)
    }
    
    band_powers = {}
    for band_name, (low, high) in bands.items():
        idx = np.logical_and(freqs >= low, freqs <= high)
        band_power = float(np.sum(psd[idx])) if np.any(idx) else 0.0
        band_powers[band_name] = band_power
        
    return band_powers

