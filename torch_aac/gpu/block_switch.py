"""Block switching / transient detection for AAC-LC.

Determines whether to use long blocks (1024 spectral lines) or short blocks
(8 × 128 spectral lines) based on transient detection.

V1: Always returns ONLY_LONG_SEQUENCE. Short block support is planned for V2.
"""

from __future__ import annotations

import torch

# Window sequence types (ISO 14496-3)
ONLY_LONG_SEQUENCE = 0
LONG_START_SEQUENCE = 1
EIGHT_SHORT_SEQUENCE = 2
LONG_STOP_SEQUENCE = 3


def detect_transients(
    frames: torch.Tensor,
    threshold: float = 10.0,
) -> torch.Tensor:
    """Detect transients in audio frames.

    Computes energy ratios between adjacent sub-windows. A high ratio
    indicates a transient (e.g., drum hit) that benefits from short blocks.

    Args:
        frames: Audio frames, shape ``(..., window_length)``.
        threshold: Energy ratio threshold for transient detection.

    Returns:
        Boolean tensor, True where transient is detected. Shape matches
        all dims except the last.
    """
    window_length = frames.shape[-1]
    num_sub = 8
    sub_len = window_length // num_sub

    # Reshape into sub-windows
    sub_frames = frames.unflatten(-1, (num_sub, sub_len))  # (..., 8, sub_len)

    # Energy per sub-window
    energy = (sub_frames ** 2).sum(dim=-1)  # (..., 8)

    # Ratio of adjacent sub-window energies
    ratios = energy[..., 1:] / (energy[..., :-1] + 1e-10)  # (..., 7)

    # Transient if any ratio exceeds threshold
    return (ratios > threshold).any(dim=-1)


def get_window_sequence(
    frames: torch.Tensor,
) -> torch.Tensor:
    """Determine window sequence type per frame.

    V1: Always returns ONLY_LONG_SEQUENCE.

    Args:
        frames: Audio frames, shape ``(B, C, window_length)`` or similar.

    Returns:
        Window sequence type per frame, shape ``(B,)`` or scalar.
    """
    # V1: long blocks only
    batch_shape = frames.shape[:-1]
    if len(batch_shape) > 1:
        batch_shape = batch_shape[:1]  # Use first dim as batch
    return torch.full(batch_shape, ONLY_LONG_SEQUENCE, device=frames.device, dtype=torch.int64)
