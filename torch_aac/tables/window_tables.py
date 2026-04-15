"""Pre-computed window functions for AAC-LC encoding.

AAC-LC uses a sine window for overlap-add. KBD (Kaiser-Bessel Derived) windows
are used in future versions for improved frequency selectivity.

Transition windows (LONG_START, LONG_STOP) bridge between long and short block
windows during block switching.
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
def long_start_window(length: int = 2048, device: torch.device | None = None) -> torch.Tensor:
    """Transition window for LONG_START_SEQUENCE (ws=1).

    Layout (N=2048, N_short=256):
      [0..1023]  normal sine window (left half of long window)
      [1024..1471]  flat 1.0 (448 samples)
      [1472..1599]  descending half of 256-sample sine window (128 samples)
      [1600..2047]  zeros (448 samples)

    Derived from ISO 14496-3, Table 4.148.
    """
    N = length
    N_short = 256
    w = torch.zeros(N, dtype=torch.float32, device=device)

    # Left half: standard long sine window
    n = torch.arange(N // 2, dtype=torch.float32, device=device)
    w[: N // 2] = torch.sin(math.pi / N * (n + 0.5))

    # Flat region
    trans_start = (3 * N) // 4 - N_short // 4  # 1472
    w[N // 2 : trans_start] = 1.0

    # Descending short sine (right half of N_short-point sine window)
    n_s = torch.arange(N_short // 2, dtype=torch.float32, device=device)
    w[trans_start : trans_start + N_short // 2] = torch.sin(
        math.pi / N_short * (n_s + N_short // 2 + 0.5)
    )

    # Trailing zeros already 0.0
    return w


@lru_cache(maxsize=8)
def long_stop_window(length: int = 2048, device: torch.device | None = None) -> torch.Tensor:
    """Transition window for LONG_STOP_SEQUENCE (ws=3).

    Time-reverse of LONG_START:
      [0..447]    zeros (448 samples)
      [448..575]  ascending half of 256-sample sine window (128 samples)
      [576..1023] flat 1.0 (448 samples)
      [1024..2047] normal sine window (right half of long window)
    """
    return long_start_window(length, device).flip(0)


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
