"""Backward compatibility wrapper for src.features.extraction."""
from src.features.extraction import (
    extract_features,
    compute_psd,
    extract_band_power
)

__all__ = ["extract_features", "compute_psd", "extract_band_power"]
