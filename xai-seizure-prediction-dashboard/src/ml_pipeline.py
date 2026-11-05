"""Backward compatibility wrapper for src.prediction and src.explainability."""
from src.prediction.inference import (
    ModelWrapper,
    MODEL_CONFIGS,
    prepare_sequences,
    run_inference,
    load_seizure_intervals,
    is_seizure
)
from src.explainability.shap_explainer import compute_shap_importance
from src.prediction.postprocessing import smooth_probabilities, compute_seizure_intervals, generate_clinical_summary

__all__ = [
    "ModelWrapper",
    "MODEL_CONFIGS",
    "prepare_sequences",
    "run_inference",
    "load_seizure_intervals",
    "is_seizure",
    "compute_shap_importance",
    "smooth_probabilities",
    "compute_seizure_intervals",
    "generate_clinical_summary"
]
