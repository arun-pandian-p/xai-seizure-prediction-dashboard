"""
Analytics and research summary utilities.
"""
import pandas as pd
from src.database import get_history
from src.evaluation.metrics import load_model_comparison_metrics

def generate_research_executive_brief() -> dict:
    """Generates an executive research brief combining historical runs and benchmark metrics."""
    metrics_df = load_model_comparison_metrics()
    history = get_history()
    
    top_model = metrics_df.sort_values(by="Accuracy", ascending=False).iloc[0].to_dict() if not metrics_df.empty else {}
    
    return {
        "total_historical_runs": len(history),
        "benchmark_champion": top_model.get("Model", "DA-GRL"),
        "champion_accuracy": top_model.get("Accuracy", 0.0),
        "champion_sensitivity": top_model.get("Sensitivity", 0.0),
        "champion_specificity": top_model.get("Specificity", 0.0),
        "champion_auc": top_model.get("AUC", 0.0)
    }

