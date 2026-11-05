"""
Centralized Configuration and Constants for NeuroGuard Dashboard.
"""
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "Documentation" / "result"
ASSETS_DIR = PROJECT_ROOT / "assets"
DB_PATH = PROJECT_ROOT / "neuroguard.db"

# Visual Identity Palette
COLORS = {
    "bg_main": "#0B1020",
    "bg_secondary": "#111A2E",
    "card_bg": "#151F33",
    "card_hover": "#1A2640",
    "accent_primary": "#22D3EE",   # Cyan
    "accent_secondary": "#14B8A6", # Teal
    "text_primary": "#F1F5F9",
    "text_muted": "#94A3B8",
    "warning": "#F59E0B",          # Amber
    "critical": "#EF4444",         # Rose
    "success": "#10B981",          # Emerald
    "border": "rgba(255, 255, 255, 0.08)",
}

# Standard 10-20 EEG Bipolar Channels (18 channels)
STANDARD_CHANNELS = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FZ-CZ", "CZ-PZ"
]

# Frequency Bands Definition
EEG_BANDS = {
    "Delta": (0.5, 4.0),
    "Theta": (4.0, 8.0),
    "Alpha": (8.0, 13.0),
    "Beta": (13.0, 30.0),
    "Gamma": (30.0, 45.0)
}

# Model Registry
MODEL_REGISTRY = {
    "DA-GRL": {
        "filename": str(MODELS_DIR / "dagrl_model.pt"),
        "channels": 18,
        "features_per_channel": 5,
        "requires_sequence": True,
        "name": "Domain-Adversarial GRL",
        "description": "Domain-adversarial gradient reversal architecture for cross-patient generalization.",
        "architecture": "GRL + Feature Extractor"
    },
    "DA-SCT": {
        "filename": str(MODELS_DIR / "dasct_model.pt"),
        "channels": 18,
        "features_per_channel": 5,
        "requires_sequence": True,
        "name": "Domain-Adversarial Spiking Conv-Transformer",
        "description": "Combines neuromorphic spike encoding with transformer temporal self-attention.",
        "architecture": "Spiking ConvNet + Transformer"
    },
    "GRL": {
        "filename": str(MODELS_DIR / "grl_model.pt"),
        "channels": 18,
        "features_per_channel": 5,
        "requires_sequence": True,
        "name": "Gradient Reversal Layer (GRL)",
        "description": "Adversarial domain adaptation model mitigating subject-to-subject distribution shifts.",
        "architecture": "Deep ConvNet + GRL"
    },
    "SCT": {
        "filename": str(MODELS_DIR / "sct_model.pt"),
        "channels": 18,
        "features_per_channel": 5,
        "requires_sequence": False,
        "name": "Spiking Conv-Transformer (SCT)",
        "description": "Spike-driven convolutional encoder with multi-head self-attention.",
        "architecture": "SpikeEncoder + Conv + Transformer"
    },
    "Federated": {
        "filename": str(MODELS_DIR / "federated_model.pt"),
        "channels": 18,
        "features_per_channel": 5,
        "requires_sequence": True,
        "name": "Federated Learning Model",
        "description": "Decentralized privacy-preserving trained model architecture.",
        "architecture": "Federated Multi-Site"
    },
    "Model combined": {
        "filename": str(MODELS_DIR / "model_best_xai_combined.pt"),
        "channels": 18,
        "features_per_channel": 5,
        "requires_sequence": True,
        "name": "Unified Best Ensemble",
        "description": "High-confidence composite model selected for unified explainability.",
        "architecture": "Ensemble"
    }
}

APP_VERSION = "2.4.0 (Research Edition)"
APP_TITLE = "NeuroGuard XAI"
APP_SUBTITLE = "EEG Seizure Detection & Neurological Analytics Dashboard"
RESEARCH_DISCLAIMER = (
    "For research and educational purposes only. This application does not constitute a clinical medical device "
    "and should not be used as a primary diagnostic tool."
)

