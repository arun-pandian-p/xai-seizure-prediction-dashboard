import numpy as np
import pytest
from pathlib import Path

from src.explainability.lime_explainer import get_lime_artifacts, has_lime_support
from utils.config import RESULTS_DIR

def test_lime_artifacts():
    artifacts = get_lime_artifacts()
    assert "lime_explanation" in artifacts
    assert "shap_vs_lime" in artifacts
    assert "unified_xai" in artifacts
    
    # If the documentation result folder exists in the project
    if RESULTS_DIR.exists():
        assert Path(artifacts["lime_explanation"]).exists()

def test_has_lime_support():
    if (RESULTS_DIR / "lime_explanation.png").exists():
        assert has_lime_support() is True

