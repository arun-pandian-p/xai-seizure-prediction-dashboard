import numpy as np
from scipy import signal
from typing import Dict, Tuple

def butter_bandpass(lowcut: float, highcut: float, fs: float, order: int = 5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data: np.ndarray, lowcut: float, highcut: float, fs: float, order: int = 5) -> np.ndarray:
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def preprocess_signal(data: np.ndarray, sfreq: float, lowcut: float = 0.5, highcut: float = 45.0, apply_notch: bool = True) -> np.ndarray:
    """
    Preprocesses the raw EEG signal by applying bandpass and optional notch filtering.
    Preserves existing filter characteristics.
    """
    filtered_data = butter_bandpass_filter(data, lowcut, highcut, sfreq, order=4)
    
    if apply_notch and sfreq > 100:
        nyq = 0.5 * sfreq
        freq = 50.0 / nyq
        if freq < 1.0:
            b, a = signal.iirnotch(freq, 30.0)
            filtered_data = signal.filtfilt(b, a, filtered_data)
        
    return filtered_data

def normalize_signal(data: np.ndarray) -> np.ndarray:
    """Normalizes the signal to zero mean and unit variance (Z-score normalization)."""
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return data
    return (data - mean) / std

def extract_bands(data: np.ndarray, sfreq: float) -> Dict[str, np.ndarray]:
    """Extracts standard neurological EEG frequency bands."""
    bands = {
        'Delta (0.5-4 Hz)': (0.5, 4.0),
        'Theta (4-8 Hz)': (4.0, 8.0),
        'Alpha (8-13 Hz)': (8.0, 13.0),
        'Beta (13-30 Hz)': (13.0, 30.0),
        'Gamma (30-45 Hz)': (30.0, 45.0)
    }
    
    band_signals = {}
    for band_name, (low, high) in bands.items():
        band_signals[band_name] = butter_bandpass_filter(data, low, high, sfreq, order=3)
        
    return band_signals

def ensure_montage(raw):
    """
    Ensures the raw MNE object has standard 10-20 electrode positions for topographic mapping.
    Handles bipolar names (FP1-F7 -> Fp1) cleanly.
    """
    import mne
    try:
        def clean_name(ch):
            name = str(ch).upper().split('-')[0]
            name = name.replace("EEG", "").replace("REF", "").replace("LE", "").strip()
            mapping = {
                'FP1': 'Fp1', 'FP2': 'Fp2', 'FZ': 'Fz', 'CZ': 'Cz', 'PZ': 'Pz', 'OZ': 'Oz',
                'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8'
            }
            return mapping.get(name, name)

        new_names = []
        seen = set()
        for ch in raw.ch_names:
            base = clean_name(ch)
            final = base
            counter = 2
            while final in seen:
                final = f"{base}_{counter}"
                counter += 1
            new_names.append(final)
            seen.add(final)
            
        raw.rename_channels(dict(zip(raw.ch_names, new_names)))
        montage = mne.channels.make_standard_montage("standard_1020")
        raw.set_montage(montage, on_missing='ignore')
        return raw, True
    except Exception:
        return raw, False

