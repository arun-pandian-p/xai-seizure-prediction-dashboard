import numpy as np
import pytest

from src.data.preprocessing import (
    butter_bandpass_filter,
    preprocess_signal,
    normalize_signal,
    extract_bands
)
from src.data.loader import normalize_channel, STANDARD_CHANNELS

def test_normalize_channel():
    assert normalize_channel("  fp1-f7  ") == "FP1-F7"
    assert normalize_channel("c3-p3") == "C3-P3"
    assert normalize_channel("Cz - Pz") == "CZ-PZ"

def test_normalize_signal():
    data = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32)
    norm = normalize_signal(data)
    assert np.isclose(np.mean(norm), 0.0, atol=1e-5)
    assert np.isclose(np.std(norm), 1.0, atol=1e-5)

def test_butter_bandpass_filter():
    sfreq = 256.0
    t = np.linspace(0, 2, int(2 * sfreq), endpoint=False)
    # 10 Hz pure sine wave (within Alpha band)
    sig = np.sin(2 * np.pi * 10.0 * t)
    filtered = butter_bandpass_filter(sig, 0.5, 45.0, sfreq)
    assert len(filtered) == len(sig)
    # Energy should be preserved
    assert np.std(filtered) > 0.5

def test_extract_bands():
    sfreq = 256.0
    sig = np.random.randn(int(4 * sfreq)).astype(np.float32)
    bands = extract_bands(sig, sfreq)
    expected_bands = [
        'Delta (0.5-4 Hz)',
        'Theta (4-8 Hz)',
        'Alpha (8-13 Hz)',
        'Beta (13-30 Hz)',
        'Gamma (30-45 Hz)'
    ]
    for b in expected_bands:
        assert b in bands
        assert len(bands[b]) == len(sig)

def test_standard_channels_count():
    assert len(STANDARD_CHANNELS) == 18

