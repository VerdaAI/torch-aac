"""Scalefactor delta encoding for AAC-LC.

Scalefactors are delta-encoded: each band's scalefactor is stored as
the difference from the previous band's scalefactor, then Huffman-coded
using a dedicated scalefactor codebook (ISO 14496-3, Table 4.A.1).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from torch_aac.tables.huffman_tables import SCALEFACTOR_BITS, SCALEFACTOR_CODE

if TYPE_CHECKING:
    from torch_aac.cpu.bitstream import BitWriter

# Delta range: -60 to +60, mapped to indices 0-120.
_SF_DELTA_OFFSET = 60


def delta_encode_scalefactors(
    scalefactors: NDArray[np.int32],
) -> NDArray[np.int32]:
    """Delta-encode scalefactors for a single channel.

    The first scalefactor is relative to global_gain.
    Subsequent scalefactors are differences from the previous.

    Args:
        scalefactors: Array of scalefactor values, shape ``(num_sfb,)``.

    Returns:
        Delta-encoded scalefactors, shape ``(num_sfb,)``.
    """
    deltas = np.zeros_like(scalefactors)
    deltas[0] = scalefactors[0]
    deltas[1:] = np.diff(scalefactors)
    return deltas


def encode_scalefactor_delta(writer: BitWriter, delta: int) -> None:
    """Encode a single scalefactor delta value using the AAC scalefactor codebook.

    Args:
        writer: BitWriter to append encoded bits to.
        delta: Delta value in range [-60, 60].
    """
    # Clamp to valid range
    delta = max(-60, min(60, delta))
    idx = delta + _SF_DELTA_OFFSET
    codeword = SCALEFACTOR_CODE[idx]
    num_bits = SCALEFACTOR_BITS[idx]
    writer.write_bits(codeword, num_bits)
