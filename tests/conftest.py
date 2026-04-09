"""Shared test fixtures for torch-aac."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch


@pytest.fixture
def device() -> torch.device:
    """Return available device (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def sine_wave_mono() -> np.ndarray:
    """Generate a 1-second mono 440Hz sine wave at 48kHz."""
    sr = 48000
    t = np.arange(sr, dtype=np.float32) / sr
    return (0.5 * np.sin(2 * math.pi * 440 * t)).astype(np.float32)


@pytest.fixture
def sine_wave_stereo() -> np.ndarray:
    """Generate a 1-second stereo sine wave at 48kHz."""
    sr = 48000
    t = np.arange(sr, dtype=np.float32) / sr
    left = (0.5 * np.sin(2 * math.pi * 1000 * t)).astype(np.float32)
    right = (0.5 * np.sin(2 * math.pi * 440 * t)).astype(np.float32)
    return np.stack([left, right], axis=-1)


@pytest.fixture
def short_mono() -> np.ndarray:
    """Generate 0.1 seconds of mono audio at 48kHz."""
    sr = 48000
    t = np.arange(int(sr * 0.1), dtype=np.float32) / sr
    return (0.3 * np.sin(2 * math.pi * 1000 * t)).astype(np.float32)
