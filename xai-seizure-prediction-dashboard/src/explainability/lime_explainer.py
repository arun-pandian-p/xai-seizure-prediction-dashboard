"""
LIME (Local Interpretable Model-agnostic Explanations) evaluation module.
Handles access to research artifacts and comparative local explanation summaries.
"""
from pathlib import Path
from typing import Dict, Any

from utils.config import RESULTS_DIR

def get_lime_artifacts() -> Dict[str, str]:
    """Returns filesystem paths to saved LIME and unified XAI figures."""
    artifacts = {
        "lime_explanation": str(RESULTS_DIR / "lime_explanation.png"),
        "shap_vs_lime": str(RESULTS_DIR / "shap_vs_lime_report.png"),
        "unified_xai": str(RESULTS_DIR / "unified_xai_report.png")
    }
    return artifacts

def has_lime_support() -> bool:
    """Checks whether LIME research artifacts exist."""
    lime_file = RESULTS_DIR / "lime_explanation.png"
    return lime_file.exists()

