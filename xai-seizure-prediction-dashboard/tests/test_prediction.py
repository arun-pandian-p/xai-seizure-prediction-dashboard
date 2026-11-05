import numpy as np
import os
import torch
import pytest
from pathlib import Path

from src.prediction.inference import (
    prepare_sequences,
    run_inference,
    load_seizure_intervals,
    is_seizure,
    MODEL_CONFIGS
)
from src.prediction.postprocessing import (
    smooth_probabilities,
    compute_seizure_intervals
)

def test_prepare_sequences():
    sfreq = 256.0
    num_channels = 18
    duration_sec = 10.0
    total_samples = int(duration_sec * sfreq)
    data = np.random.randn(num_channels, total_samples).astype(np.float32)
    
    config = MODEL_CONFIGS["DA-GRL"]
    window_size = int(4.0 * sfreq)
    step_sec = 2.0
    
    sequences, seq_times = prepare_sequences(data, sfreq, config, window_size=window_size, step_sec=step_sec)
    assert len(sequences) > 0
    assert len(sequences) == len(seq_times)
    # Shape of each sequence: (window_size, channels)
    assert sequences.shape[1] == window_size
    assert sequences.shape[2] == num_channels

def test_seizure_intervals_parsing():
    content = b"Start Time: 10\nEnd Time: 25\nStart Time: 40\nEnd Time: 60"
    intervals = load_seizure_intervals(content)
    assert intervals == [(10, 25), (40, 60)]
    assert is_seizure(15, intervals) == 1
    assert is_seizure(30, intervals) == 0
    assert is_seizure(55, intervals) == 1

def test_smooth_probabilities():
    probs = np.array([0.1, 0.2, 0.9, 0.8, 0.2])
    smoothed = smooth_probabilities(probs, window=3)
    assert len(smoothed) == len(probs)
    assert np.all(smoothed >= 0.0)
    assert np.all(smoothed <= 1.0)

def test_compute_seizure_intervals():
    preds = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 0])
    times = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18])
    intervals = compute_seizure_intervals(preds, times, min_duration_sec=2.0)
    assert len(intervals) >= 1
    assert intervals[0]["start_time"] == 4.0

