"""Block switching / transient detection for AAC-LC.

Determines whether to use long blocks (1024 spectral lines) or short blocks
(8 x 128 spectral lines) based on transient detection.

The state machine handles transitions:
  ONLY_LONG → LONG_START → EIGHT_SHORT → LONG_STOP → ONLY_LONG
LONG_START and LONG_STOP use the same 1024-coefficient MDCT as ONLY_LONG
but with modified window shapes for smooth overlap-add transitions.
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
    energy = (sub_frames**2).sum(dim=-1)  # (..., 8)

    # Ratio of adjacent sub-window energies
    ratios = energy[..., 1:] / (energy[..., :-1] + 1e-10)  # (..., 7)

    # Transient if any ratio exceeds threshold
    return (ratios > threshold).any(dim=-1)


def get_window_sequence(
    is_transient: torch.Tensor,
    prev_window_seq: torch.Tensor,
) -> torch.Tensor:
    """Determine window sequence type per frame using the AAC state machine.

    Implements the required transitions for smooth overlap-add:
      - Non-transient after ONLY_LONG/LONG_STOP → ONLY_LONG
      - Transient after ONLY_LONG/LONG_STOP → LONG_START
      - Transient after LONG_START/EIGHT_SHORT → EIGHT_SHORT
      - Non-transient after EIGHT_SHORT/LONG_START → LONG_STOP

    Args:
        is_transient: Bool tensor per frame, True = transient detected.
        prev_window_seq: Int tensor per frame, previous frame's window_sequence.

    Returns:
        Window sequence type per frame (0-3).
    """
    result = torch.full_like(prev_window_seq, ONLY_LONG_SEQUENCE)

    # Previous was long-type (ONLY_LONG or LONG_STOP)
    was_long = (prev_window_seq == ONLY_LONG_SEQUENCE) | (prev_window_seq == LONG_STOP_SEQUENCE)
    # Previous was short-type (LONG_START or EIGHT_SHORT)
    was_short = (prev_window_seq == LONG_START_SEQUENCE) | (
        prev_window_seq == EIGHT_SHORT_SEQUENCE
    )

    # Transitions
    result = torch.where(
        was_long & is_transient,
        torch.full_like(result, LONG_START_SEQUENCE),
        result,
    )
    result = torch.where(
        was_long & ~is_transient,
        torch.full_like(result, ONLY_LONG_SEQUENCE),
        result,
    )
    result = torch.where(
        was_short & is_transient,
        torch.full_like(result, EIGHT_SHORT_SEQUENCE),
        result,
    )
    result = torch.where(
        was_short & ~is_transient,
        torch.full_like(result, LONG_STOP_SEQUENCE),
        result,
    )

    return result
