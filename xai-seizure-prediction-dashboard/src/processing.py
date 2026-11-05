"""Backward compatibility wrapper for src.data.preprocessing."""
from src.data.preprocessing import (
    butter_bandpass,
    butter_bandpass_filter,
    preprocess_signal,
    normalize_signal,
    extract_bands,
    ensure_montage
)

__all__ = [
    "butter_bandpass",
    "butter_bandpass_filter",
    "preprocess_signal",
    "normalize_signal",
    "extract_bands",
    "ensure_montage"
]
