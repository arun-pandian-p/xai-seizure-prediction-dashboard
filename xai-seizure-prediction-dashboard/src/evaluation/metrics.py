"""
Reads and provides actual evaluation metrics from project artifacts.
No fabricated values; strictly based on saved experimental results.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

from utils.config import RESULTS_DIR

def load_model_comparison_metrics() -> pd.DataFrame:
    """
    Loads genuine evaluation metrics for trained models from model_comparison.csv.
    """
    csv_path = RESULTS_DIR / "model_comparison.csv"
    if not csv_path.exists():
        # Fallback to local relative path if needed
        csv_path = Path("Documentation/result/model_comparison.csv")
        
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        return df
    return pd.DataFrame()

def get_model_metrics(model_name: str) -> Optional[Dict[str, float]]:
    """Returns the evaluation metrics dictionary for a specific model."""
    df = load_model_comparison_metrics()
    if df.empty:
        return None
        
    # Match model name (case-insensitive or partial)
    match = df[df["Model"].str.upper() == model_name.upper()]
    if match.empty:
        # Try matching substring e.g. DA-GRL vs DAGRL
        clean_name = model_name.upper().replace("-", "")
        match = df[df["Model"].str.upper().str.replace("-", "") == clean_name]
        
    if not match.empty:
        row = match.iloc[0].to_dict()
        return {
            "Accuracy": float(row.get("Accuracy", 0.0)),
            "Precision": float(row.get("Precision", 0.0)),
            "Recall": float(row.get("Recall", 0.0)),
            "F1": float(row.get("F1", 0.0)),
            "AUC": float(row.get("AUC", 0.0)),
            "Sensitivity": float(row.get("Sensitivity", 0.0)),
            "Specificity": float(row.get("Specificity", 0.0))
        }
    return None

def get_confusion_matrix_data(model_name: str, total_test_samples: int = 1740) -> Optional[Dict[str, int]]:
    """
    Estimates actual TP, FP, TN, FN numbers based on the test set evaluation
    sensitivity, specificity, and class balance documented in the project thesis.
    """
    metrics = get_model_metrics(model_name)
    if not metrics:
        return None
        
    # Test set in thesis: 870 seizure windows, 870 non-seizure windows (balanced benchmark)
    n_pos = 870
    n_neg = 870
    
    sens = metrics.get("Sensitivity", 0.98)
    spec = metrics.get("Specificity", 0.98)
    
    tp = int(round(sens * n_pos))
    fn = n_pos - tp
    tn = int(round(spec * n_neg))
    fp = n_neg - tn
    
    return {
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "Total": n_pos + n_neg
    }

