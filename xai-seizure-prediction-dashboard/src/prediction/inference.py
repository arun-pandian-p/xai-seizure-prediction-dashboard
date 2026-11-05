import torch
import numpy as np
import os
import re
import gc
from typing import Dict, Any, Tuple, Optional, List
import streamlit as st

from src.models import SCT, GRLSeizureModel, FederatedModel, DASCT, DAGRLModel
from src.data.preprocessing import preprocess_signal, normalize_signal, butter_bandpass_filter

class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        out = self.model(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        if out.dim() == 1:
            out = out.unsqueeze(1)
        return out

MODEL_CONFIGS = {
    "DA-GRL": {
        "filename": "models/dagrl_model.pt",
        "channels": 18,
        "features_per_channel": 5,
        "requires_sequence": True,
        "class": DAGRLModel
    },
    "DA-SCT": {
        "filename": "models/dasct_model.pt",
        "channels": 18,
        "features_per_channel": 5,
        "requires_sequence": True,
        "class": DASCT
    },
    "GRL": {
        "filename": "models/grl_model.pt",
        "channels": 18,
        "features_per_channel": 5,
        "requires_sequence": True,
        "class": GRLSeizureModel
    },
    "SCT": {
        "filename": "models/sct_model.pt",
        "channels": 18,
        "features_per_channel": 5,
        "requires_sequence": False,
        "class": SCT
    },
    "Federated": {
        "filename": "models/federated_model.pt",
        "channels": 18,
        "features_per_channel": 5,
        "requires_sequence": True,
        "class": FederatedModel
    },
    "Model combined": {
        "filename": "models/model_best_xai_combined.pt",
        "channels": 18,
        "features_per_channel": 5,
        "requires_sequence": True,
        "class": None # Dynamic
    }
}

def _load_model_uncached(model_path: str, config: Dict[str, Any]) -> torch.nn.Module:
    """Instantiates and loads model state dictionary on CPU."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found.")
        
    loaded = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    
    if isinstance(loaded, dict):
        model_class = config.get("class")
        if model_class is None and "best_model_type" in loaded:
            m_type = loaded["best_model_type"]
            if m_type == "dagrl": model_class = DAGRLModel
            elif m_type == "dasct": model_class = DASCT
            elif m_type == "grl": model_class = GRLSeizureModel
            elif m_type == "sct": model_class = SCT
            elif m_type == "federated": model_class = FederatedModel
        
        if model_class is None:
            raise ValueError(f"Could not resolve model class for {model_path}")
            
        n_ch = config.get("channels", 18)
        model = model_class(n_ch=n_ch)
        model.load_state_dict(loaded["model_state_dict"])
    else:
        model = loaded
        
    model.eval()
    return model

# Cached version for Streamlit runtime
def load_model(model_name: str) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """Cached loader for PyTorch neural network checkpoints."""
    if model_name not in MODEL_CONFIGS:
        raise KeyError(f"Unknown model name: {model_name}")
    config = MODEL_CONFIGS[model_name]
    model = _load_model_uncached(config["filename"], config)
    return model, config

def prepare_sequences(data: np.ndarray, sfreq: float, config: Dict[str, Any], window_size: int = 1024, step_sec: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Slices multi-channel aligned EEG into overlapping temporal windows.
    Preserves existing preprocessing sequence structure.
    """
    num_channels, total_samples = data.shape
    
    processed_data = np.zeros_like(data)
    for ch in range(num_channels):
        ch_data = preprocess_signal(data[ch], sfreq, lowcut=0.5, highcut=45.0)
        processed_data[ch] = normalize_signal(ch_data)
        
    sequences = []
    sequence_times = []
    step = max(1, int(step_sec * sfreq))
    
    for i in range(0, total_samples - window_size + 1, step):
        segment = processed_data[:, i:i + window_size]
        sequences.append(segment.T)
        sequence_times.append((i + window_size / 2) / sfreq)
        
    return np.array(sequences, dtype=np.float32), np.array(sequence_times, dtype=np.float32)

def run_inference(
    sequences: np.ndarray,
    model_path: str,
    config: Dict[str, Any],
    shap_enabled: bool = False,
    top_k: int = 5,
    threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, Tuple, Optional[List[int]], Optional[np.ndarray], Optional[Any], Optional[Any]]:
    """
    Runs batch inference on EEG sequences using CPU execution with explicit memory handling.
    """
    from src.explainability.shap_explainer import compute_shap_importance
    
    model = _load_model_uncached(model_path, config)
    
    selected_channels = None
    channel_importance = None
    shap_vals = None
    test_sample = None
    
    if shap_enabled and len(sequences) > 0:
        selected_channels, channel_importance, shap_vals, test_sample = compute_shap_importance(
            model, sequences, config, top_k=top_k
        )
        
    if len(sequences) == 0:
        return np.array([]), np.array([]), (0, 0, 0), None, None, None, None

    X = torch.tensor(sequences).float()
    probs_list = []
    batch_size = 32
    
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch_X = X[i:i + batch_size]
            outputs = model(batch_X)
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]
            
            p = outputs.squeeze().numpy()
            if p.ndim == 0:
                p = np.array([p])
            probs_list.append(p)
            
    probs = np.concatenate(probs_list) if len(probs_list) > 0 else np.array([])
    predictions = (probs >= threshold).astype(int)
    sample_shape = tuple(X[0].unsqueeze(0).shape)
    
    del X
    gc.collect()
    
    return predictions, probs, sample_shape, selected_channels, channel_importance, shap_vals, test_sample

def load_seizure_intervals(file_content: bytes) -> Optional[List[Tuple[int, int]]]:
    """Parses .seizures clinical annotation files for ground-truth validation."""
    try:
        text = file_content.decode("utf-8", errors="ignore")
    except Exception:
        return None
        
    starts = re.findall(r'Start Time:\s*(\d+)', text)
    ends = re.findall(r'End Time:\s*(\d+)', text)
    
    if not starts or not ends or len(starts) != len(ends):
        return None
        
    intervals = []
    for s, e in zip(starts, ends):
        intervals.append((int(s), int(e)))
        
    return intervals

def is_seizure(time_sec: float, intervals: List[Tuple[int, int]]) -> int:
    """Checks whether a given timestamp falls into any annotated seizure interval."""
    for start, end in intervals:
        if start <= time_sec <= end:
            return 1
    return 0

