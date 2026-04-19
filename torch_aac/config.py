"""Configuration and constants for AAC-LC encoding.

This module defines the encoder configuration, AAC-LC constants,
and sample-rate-dependent parameters used throughout the library.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch

# ---------------------------------------------------------------------------
# AAC-LC constants
# ---------------------------------------------------------------------------

AAC_FRAME_LENGTH = 1024
"""Number of spectral lines per AAC-LC frame."""

AAC_WINDOW_LENGTH = 2048
"""Window length for long blocks (2 × frame_length)."""

AAC_SHORT_WINDOW_LENGTH = 256
"""Window length for short blocks (8 short blocks per frame)."""

AAC_SHORT_FRAME_LENGTH = 128
"""Number of spectral lines per short block."""

AAC_NUM_SHORT_WINDOWS = 8
"""Number of short windows per EIGHT_SHORT_SEQUENCE frame."""

MAX_SFB = 51
"""Maximum number of scalefactor bands across all sample rates (32 kHz has 51)."""

ADTS_HEADER_SIZE = 7
"""Size of ADTS fixed header in bytes (no CRC)."""

# MPEG-4 Audio Object Types
AOT_AAC_LC = 2

# Perceptual Noise Substitution (PNS)
NOISE_BT = 13
"""Band type for PNS: decoder generates white noise at a specified energy."""

NOISE_OFFSET = 90
"""Decoder subtracts this from global_gain for the initial noise sf offset."""

NOISE_PRE = 256
"""Added to the first noise band's offset before writing as 9 raw bits."""

NOISE_PRE_BITS = 9
"""Bit width of the first noise band's absolute offset field."""

# ADTS sampling frequency index mapping
SAMPLE_RATE_INDEX: dict[int, int] = {
    96000: 0,
    88200: 1,
    64000: 2,
    48000: 3,
    44100: 4,
    32000: 5,
    24000: 6,
    22050: 7,
    16000: 8,
    12000: 9,
    11025: 10,
    8000: 11,
}

SUPPORTED_SAMPLE_RATES = frozenset(SAMPLE_RATE_INDEX.keys())

# Channel configuration for ADTS
CHANNEL_CONFIG: dict[int, int] = {
    1: 1,  # mono: center
    2: 2,  # stereo: left + right
}


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------


class QuantMode(Enum):
    """Quantization mode for the encoder."""

    HARD = "hard"
    """Hard rounding — no gradient. Used in encode mode."""

    STE = "ste"
    """Straight-through estimator — gradient passes through round()."""

    NOISE = "noise"
    """Additive uniform noise approximation of quantization."""

    CUBIC = "cubic"
    """Cubic soft-rounding: ``round(x) + (x - round(x))^3``.

    Provides real gradients (unlike STE) with small deviation from hard
    rounding (max 0.125 quantized units). Useful for warm-up training
    before switching to STE for artifact-faithful fine-tuning."""


# ---------------------------------------------------------------------------
# Encoder configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EncoderConfig:
    """Configuration for the AAC-LC encoder.

    Args:
        sample_rate: Audio sample rate in Hz. Must be one of the supported rates.
        channels: Number of audio channels (1 for mono, 2 for stereo).
        bitrate: Target bitrate in bits per second (e.g. 128000 for 128 kbps).
        device: PyTorch device string. "auto" selects CUDA, then MPS
            (Apple Silicon), then CPU. Explicit values: "cuda", "mps", "cpu".
        batch_size: Number of frames per GPU batch. 0 = auto-detect from VRAM.
        quant_mode: Quantization mode (hard, ste, or noise).
    """

    sample_rate: int = 48000
    channels: int = 2
    bitrate: int = 128000
    device: str = "auto"
    batch_size: int = 0
    quant_mode: QuantMode = QuantMode.HARD

    def __post_init__(self) -> None:
        if self.sample_rate not in SUPPORTED_SAMPLE_RATES:
            raise ValueError(
                f"Unsupported sample rate {self.sample_rate}. "
                f"Supported: {sorted(SUPPORTED_SAMPLE_RATES)}"
            )
        if self.channels not in CHANNEL_CONFIG:
            raise ValueError(
                f"Unsupported channel count {self.channels}. Supported: 1 (mono), 2 (stereo)"
            )
        if not 16000 <= self.bitrate <= 576000:
            raise ValueError(f"Bitrate {self.bitrate} out of range. Must be 16000-576000 bps.")

    @property
    def sample_rate_index(self) -> int:
        """ADTS sampling frequency index for this sample rate."""
        return SAMPLE_RATE_INDEX[self.sample_rate]

    @property
    def channel_config(self) -> int:
        """ADTS channel configuration index."""
        return CHANNEL_CONFIG[self.channels]

    @property
    def frame_length(self) -> int:
        """Number of new samples per frame."""
        return AAC_FRAME_LENGTH

    @property
    def window_length(self) -> int:
        """Window length for long blocks."""
        return AAC_WINDOW_LENGTH

    @property
    def bits_per_frame(self) -> float:
        """Target bits per frame based on bitrate and sample rate."""
        return self.bitrate * self.frame_length / self.sample_rate

    @property
    def bytes_per_frame(self) -> int:
        """Target bytes per frame (rounded down)."""
        return int(self.bits_per_frame / 8)

    @property
    def resolved_device(self) -> torch.device:
        """Resolve 'auto' to actual device.

        Selection order:
          1. CUDA (NVIDIA GPUs) if available
          2. MPS (Apple Silicon via Metal Performance Shaders) if available
          3. CPU fallback
        """
        if self.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(self.device)
