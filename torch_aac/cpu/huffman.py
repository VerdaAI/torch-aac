"""Huffman bit packing for AAC-LC spectral data.

Encodes quantized spectral coefficients using the 11 AAC Huffman codebooks.
Codebook selection is done on GPU (gpu/huffman_select.py); this module
does the serial bit packing into the bitstream.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from torch_aac.tables.huffman_tables import (
    CODEBOOK_DIMENSION,
    CODEBOOK_MAX_ABS,
    CODEBOOK_UNSIGNED,
    get_codebook_entry,
)

if TYPE_CHECKING:
    from torch_aac.cpu.bitstream import BitWriter


def encode_spectral_band(
    writer: BitWriter,
    band_data: NDArray[np.int32],
    codebook: int,
) -> None:
    """Encode a scalefactor band's spectral data using the specified codebook.

    Args:
        writer: BitWriter to append encoded bits to.
        band_data: Quantized spectral coefficients for this band.
        codebook: Huffman codebook index (1-11).
    """
    if codebook == 0:
        return

    dim = CODEBOOK_DIMENSION[codebook]
    is_unsigned = CODEBOOK_UNSIGNED[codebook]

    # Pad band_data to be divisible by dim
    remainder = len(band_data) % dim
    if remainder != 0:
        band_data = np.pad(band_data, (0, dim - remainder))

    for i in range(0, len(band_data), dim):
        group = band_data[i : i + dim]

        if is_unsigned:
            # Unsigned codebooks: encode absolute values, then sign bits
            abs_group = np.abs(group)
            _encode_unsigned_group(writer, group, abs_group, codebook, dim)
        else:
            # Signed codebooks: encode values directly (with offset)
            _encode_signed_group(writer, group, codebook)


def _encode_signed_group(
    writer: BitWriter,
    values: NDArray[np.int32],
    codebook: int,
) -> None:
    """Encode a group using a signed codebook (1-2, 5-6)."""
    key = tuple(int(v) for v in values)
    try:
        codeword, length = get_codebook_entry(codebook, key)
        writer.write_bits(codeword, length)
    except (KeyError, ValueError):
        # Fallback: clamp values to codebook range and retry
        max_abs = CODEBOOK_MAX_ABS[codebook]
        clamped = tuple(max(-max_abs, min(max_abs, int(v))) for v in values)
        codeword, length = get_codebook_entry(codebook, clamped)
        writer.write_bits(codeword, length)


def _encode_unsigned_group(
    writer: BitWriter,
    values: NDArray[np.int32],
    abs_values: NDArray[np.int32],
    codebook: int,
    dim: int,
) -> None:
    """Encode a group using an unsigned codebook (3-4, 7-11).

    Writes the Huffman code for the absolute values, then appends
    a sign bit (0=positive, 1=negative) for each non-zero value.
    """
    max_abs = CODEBOOK_MAX_ABS[codebook]

    if codebook == 11:
        # Codebook 11: escape codes for values > 15
        lookup_vals = np.minimum(abs_values, 16)
    else:
        lookup_vals = np.minimum(abs_values, max_abs)

    key = tuple(int(v) for v in lookup_vals)
    try:
        codeword, length = get_codebook_entry(codebook, key)
        writer.write_bits(codeword, length)
    except (KeyError, ValueError):
        # Zero fallback
        zero_key = tuple(0 for _ in range(dim))
        codeword, length = get_codebook_entry(codebook, zero_key)
        writer.write_bits(codeword, length)
        return

    # Append sign bits for non-zero values
    for j in range(dim):
        if abs_values[j] != 0:
            writer.write_bits(1 if values[j] < 0 else 0, 1)

    # Escape codes for codebook 11 values > 15
    if codebook == 11:
        for j in range(dim):
            if abs_values[j] > 15:
                _encode_escape(writer, int(abs_values[j]))


MAX_ESCAPE_N = 8
"""Maximum escape sequence length. FFmpeg's decoder limits N to 8, allowing values up to 4095."""

MAX_ESCAPE_VAL = (1 << (MAX_ESCAPE_N + 4)) - 1  # 4095


def _encode_escape(writer: BitWriter, abs_val: int) -> None:
    """Encode an escape code for codebook 11 (values > 15).

    Escape format per ISO 14496-3:
        N ones + one zero + (N+4) bit mantissa
    where N = floor(log2(abs_val)) - 3

    Values are clamped to MAX_ESCAPE_VAL (4095) to avoid decoder overflow.
    """
    abs_val = min(abs_val, MAX_ESCAPE_VAL)
    n = max(0, min(abs_val.bit_length() - 4, MAX_ESCAPE_N))
    # N ones
    for _ in range(n):
        writer.write_bits(1, 1)
    # Terminating zero
    writer.write_bits(0, 1)
    # (N+4) bit mantissa
    writer.write_bits(abs_val, n + 4)
