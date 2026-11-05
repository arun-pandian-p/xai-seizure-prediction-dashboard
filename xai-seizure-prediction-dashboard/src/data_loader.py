"""Backward compatibility wrapper for src.data.loader."""
from src.data.loader import load_edf, align_channels, normalize_channel, STANDARD_CHANNELS

__all__ = ["load_edf", "align_channels", "normalize_channel", "STANDARD_CHANNELS"]
