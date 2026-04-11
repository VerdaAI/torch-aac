"""Pre-computed window functions for AAC-LC encoding.

AAC-LC uses a sine window for overlap-add. KBD (Kaiser-Bessel Derived) windows
are used in future versions for improved frequency selectivity.
"""

from __future__ import annotations

import math
from functools import lru_cache

import torch


@lru_cache(maxsize=8)
def sine_window(length: int, device: torch.device | None = None) -> torch.Tensor:
    """Compute the sine window used in AAC-LC.

    W(n) = sin(pi/N * (n + 0.5))  for n = 0, 1, ..., N-1

    Args:
        length: Window length (typically 2048 for long blocks).
        device: Target device.

    Returns:
        1-D float32 tensor of shape ``(length,)``.
    """
    n = torch.arange(length, dtype=torch.float32, device=device)
    return torch.sin(math.pi / length * (n + 0.5))


@lru_cache(maxsize=8)
def kbd_window(
    length: int, alpha: float = 4.0, device: torch.device | None = None
) -> torch.Tensor:
    """Compute the Kaiser-Bessel Derived (KBD) window.

    Args:
        length: Window length.
        alpha: Kaiser window shape parameter.
        device: Target device.

    Returns:
        1-D float32 tensor of shape ``(length,)``.
    """
    half = length // 2
    # Kaiser window of length half+1
    n = torch.arange(half + 1, dtype=torch.float64, device=device)
    beta = math.pi * alpha
    kaiser = torch.i0(beta * torch.sqrt(1.0 - (2.0 * n / half - 1.0) ** 2)) / torch.i0(
        torch.tensor(beta, dtype=torch.float64, device=device)
    )
    # Cumulative sum
    cumsum = torch.cumsum(kaiser, dim=0)
    # Normalize and take square root
    kbd_left = torch.sqrt(cumsum[:-1] / cumsum[-1])
    # Mirror for right half
    kbd_right = kbd_left.flip(0)
    window = torch.cat([kbd_left, kbd_right])
    return window.to(torch.float32)
