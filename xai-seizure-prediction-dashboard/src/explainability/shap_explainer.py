import torch
import numpy as np
import shap
from typing import Dict, Any, Tuple, Optional, List

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

def compute_shap_importance(
    model: torch.nn.Module,
    sequences: np.ndarray,
    config: Dict[str, Any],
    top_k: int = 5
) -> Tuple[Optional[List[int]], Optional[np.ndarray], Optional[np.ndarray], Optional[torch.Tensor]]:
    """
    Computes SHAP values using DeepExplainer or GradientExplainer on CPU.
    Aggregates temporal attributions to channel-level biological biomarkers.
    """
    model.eval()
    wrapper = ModelWrapper(model)
    
    if len(sequences) == 0:
        return None, None, None, None
        
    bg_size = min(5, len(sequences))
    background = torch.tensor(sequences[:bg_size]).float()
    
    test_sample_indices = np.linspace(0, len(sequences) - 1, min(3, len(sequences)), dtype=int)
    test_sample = torch.tensor(sequences[test_sample_indices]).float()
    
    shap_values = None
    try:
        explainer = shap.DeepExplainer(wrapper, background)
        shap_values = explainer.shap_values(test_sample)
    except Exception:
        try:
            explainer = shap.GradientExplainer(wrapper, background)
            shap_values = explainer.shap_values(test_sample)
        except Exception as e2:
            print(f"[ERROR] SHAP explainer failed: {e2}")
            return None, None, None, None

    if isinstance(shap_values, list):
        shap_vals = shap_values[0] if len(shap_values) > 0 else shap_values
    else:
        shap_vals = shap_values
        
    if shap_vals is None:
        return None, None, None, None
        
    # shap_vals shape: (batch, T, channels) or (batch, T, channels, 1)
    if shap_vals.ndim == 4:
        shap_vals = shap_vals.squeeze(-1)
        
    # Channel-level importance
    importance = np.abs(shap_vals).mean(axis=(0, 1))
    
    # Sort top channels
    selected_channels = [int(np.asarray(i).item()) for i in np.argsort(importance)[-top_k:][::-1]]
    
    return selected_channels, importance, shap_vals, test_sample

