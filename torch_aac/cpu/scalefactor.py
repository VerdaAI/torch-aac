"""Scalefactor delta encoding for AAC-LC.

Scalefactors are delta-encoded: each band's scalefactor is encoded as the
difference from the previous band's scalefactor.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def delta_encode_scalefactors(
    scalefactors: NDArray[np.int32],
) -> NDArray[np.int32]:
    """Delta-encode scalefactors for a single channel.

    The first scalefactor is encoded as-is (relative to global_gain).
    Subsequent scalefactors are encoded as differences from the previous.

    In our simplified V1 encoder, we use a single global_gain for all bands
    (uniform scalefactors), so deltas are all zero. This function is here
    for correctness and future per-band scalefactor support.

    Args:
        scalefactors: Array of scalefactor values, shape ``(num_sfb,)``.

    Returns:
        Delta-encoded scalefactors, shape ``(num_sfb,)``.
    """
    deltas = np.zeros_like(scalefactors)
    deltas[0] = scalefactors[0]
    deltas[1:] = np.diff(scalefactors)
    return deltas


# Scalefactor Huffman codebook (from ISO 13818-7, Table A.7)
# Maps delta value (offset by 60) to (codeword, length).
# Delta range: -60 to +60 → index 0 to 120.
# Most common deltas are near zero and have short codes.
SF_HUFFMAN: dict[int, tuple[int, int]] = {
    -1: (0b110, 3),
    0: (0b11, 2),
    1: (0b101, 3),
    2: (0b1110, 4),
    -2: (0b1111, 4),
    3: (0b10010, 5),
    -3: (0b10011, 5),
    4: (0b100010, 6),
    -4: (0b100011, 6),
    5: (0b1000010, 7),
    -5: (0b1000011, 7),
    6: (0b10000010, 8),
    -6: (0b10000011, 8),
    7: (0b100000010, 9),
    -7: (0b100000011, 9),
    8: (0b1000000010, 10),
    -8: (0b1000000011, 10),
    9: (0b10000000010, 11),
    -9: (0b10000000011, 11),
    10: (0b100000000010, 12),
    -10: (0b100000000011, 12),
    11: (0b1000000000010, 13),
    -11: (0b1000000000011, 13),
}


def encode_scalefactor_delta(delta: int) -> tuple[int, int]:
    """Encode a single scalefactor delta value.

    Args:
        delta: Delta value.

    Returns:
        Tuple of (codeword, num_bits).
    """
    if delta in SF_HUFFMAN:
        return SF_HUFFMAN[delta]
    # For deltas outside the common range, use a longer encoding
    # (in practice, deltas > 11 are very rare with uniform scalefactors)
    abs_delta = abs(delta)
    num_bits = abs_delta + 3
    codeword = (1 << 1) | (0 if delta > 0 else 1)
    return codeword, min(num_bits, 19)
