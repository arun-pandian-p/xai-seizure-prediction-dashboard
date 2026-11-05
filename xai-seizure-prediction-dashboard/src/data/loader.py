import mne
import numpy as np
import tempfile
import os
from pathlib import Path
from typing import Dict, Any, Tuple, List, Union

STANDARD_CHANNELS = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FZ-CZ", "CZ-PZ"
]

def normalize_channel(ch: str) -> str:
    """Normalizes channel names to a standard format (uppercase, no spaces)."""
    return ch.strip().upper().replace(" ", "")

def load_edf(file_source: Union[Any, str, Path]) -> Dict[str, Any]:
    """
    Loads an EDF file using mne-python from either a Streamlit UploadedFile 
    or a local filesystem path.
    
    Args:
        file_source: Streamlit UploadedFile or file path.
        
    Returns:
        dict: Raw MNE object, channel names, sampling rate, duration, data array, and time array.
    """
    temp_file_path = None
    if isinstance(file_source, (str, Path)):
        edf_path = str(file_source)
    else:
        with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as temp_file:
            temp_file.write(file_source.getbuffer())
            temp_file_path = temp_file.name
        edf_path = temp_file_path

    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        
        # 1. Normalize Channel Names while strictly guaranteeing uniqueness
        normalized_names = {}
        seen = set()
        for ch in raw.ch_names:
            norm = normalize_channel(ch)
            final_norm = norm
            counter = 1
            while final_norm in seen:
                final_norm = f"{norm}_{counter}"
                counter += 1
            normalized_names[ch] = final_norm
            seen.add(final_norm)
            
        raw.rename_channels(normalized_names)
        
        # 2. Remove Duplicates
        unique_channels = list(dict.fromkeys(raw.ch_names))
        raw.pick_channels(unique_channels)
        
        # Extract basic info
        ch_names = raw.ch_names
        sfreq = float(raw.info['sfreq'])
        n_samples = raw.n_times
        duration = float(n_samples / sfreq)
        
        data, times = raw[:, :]
        
        return {
            'raw': raw,
            'ch_names': ch_names,
            'sfreq': sfreq,
            'duration': duration,
            'data': data,
            'times': times,
            'n_samples': n_samples
        }
    except Exception as e:
        raise ValueError(f"Failed to load EDF file: {str(e)}")
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception:
                pass

def align_channels(raw: mne.io.Raw, required_channels_count: int = 18) -> Tuple[np.ndarray, List[str]]:
    """
    Aligns the raw data to standard channels, filling missing channels with zeros.
    Preserves exact alignment required by pre-trained neural networks.
    """
    data = raw.get_data()
    ch_names = [ch.upper().strip() for ch in raw.ch_names]
    
    target_channels = STANDARD_CHANNELS.copy()
    if required_channels_count > len(target_channels):
        extra_needed = required_channels_count - len(target_channels)
        extra_channels = [ch for ch in ch_names if ch not in target_channels]
        target_channels.extend(extra_channels[:extra_needed])
        
    aligned_matrix = np.zeros((required_channels_count, data.shape[1]), dtype=np.float32)
    valid_channels = []
    
    for i, target_ch in enumerate(target_channels[:required_channels_count]):
        # Exact match or normalized match
        match_idx = None
        for orig_idx, orig_ch in enumerate(ch_names):
            if orig_ch == target_ch or orig_ch.replace("-", "") == target_ch.replace("-", ""):
                match_idx = orig_idx
                break
                
        if match_idx is not None:
            aligned_matrix[i, :] = data[match_idx, :]
            valid_channels.append(target_ch)
        else:
            valid_channels.append(target_ch)
            
    return aligned_matrix, valid_channels

