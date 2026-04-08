"""Huffman bit packing for AAC-LC spectral data.

This module handles the actual variable-length code assembly for quantized
spectral coefficients. Codebook selection is done on GPU (gpu/huffman_select.py);
this module only does the serial bit packing.

For V1, we use a simplified encoding approach: each non-zero coefficient is
encoded with a code whose length approximates the Huffman table. A future
version will use the exact ISO 13818-7 Huffman tables.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from torch_aac.cpu.bitstream import BitWriter

# Codebook dimensions
CODEBOOK_DIMS = [0, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2]
CODEBOOK_IS_UNSIGNED = [
    False,  # 0
    True, True,     # 1-2
    True, True,     # 3-4
    False, False,   # 5-6
    True, True,     # 7-8
    True, True,     # 9-10
    True,           # 11
]
CODEBOOK_MAX_ABS = [0, 1, 1, 2, 2, 4, 4, 7, 7, 12, 12, 16]


def encode_spectral_band(
    writer: BitWriter,
    band_data: NDArray[np.int32],
    codebook: int,
) -> None:
    """Encode a scalefactor band's spectral data using the specified codebook.

    For V1, this uses a simplified encoding:
    - Codebooks 1-4 (quads): encode groups of 4 coefficients
    - Codebooks 5-11 (pairs): encode groups of 2 coefficients
    - Unsigned codebooks: encode absolute values + sign bits
    - Codebook 11: escape codes for values > 15

    Args:
        writer: BitWriter to append encoded bits to.
        band_data: Quantized spectral coefficients for this band.
        codebook: Huffman codebook index (1-11).
    """
    if codebook == 0:
        return

    dim = CODEBOOK_DIMS[codebook]
    is_unsigned = CODEBOOK_IS_UNSIGNED[codebook]
    max_abs = CODEBOOK_MAX_ABS[codebook]

    # Pad band_data to be divisible by dim
    remainder = len(band_data) % dim
    if remainder != 0:
        band_data = np.pad(band_data, (0, dim - remainder))

    for i in range(0, len(band_data), dim):
        group = band_data[i : i + dim]

        if is_unsigned:
            abs_group = np.abs(group)
            signs = group < 0

            # Encode the absolute values
            _encode_group_values(writer, abs_group, codebook, max_abs)

            # Append sign bits for non-zero values
            for j in range(dim):
                if abs_group[j] != 0:
                    writer.write_bits(1 if signs[j] else 0, 1)
        else:
            # Signed codebooks: values are offset
            _encode_group_values(writer, group, codebook, max_abs)


def _encode_group_values(
    writer: BitWriter,
    values: NDArray[np.int32],
    codebook: int,
    max_abs: int,
) -> None:
    """Encode a group of values using a simplified Huffman scheme.

    For V1, we approximate the Huffman codes. The exact tables from
    huffman_tables.py will be integrated in a future version for
    bit-exact compliance.

    Args:
        writer: BitWriter.
        values: Group of quantized values.
        codebook: Codebook index.
        max_abs: Maximum absolute value for this codebook.
    """
    # Try to use the exact Huffman tables if available
    try:
        from torch_aac.tables.huffman_tables import get_codebook_entry

        key = tuple(int(v) for v in values)
        codeword, length = get_codebook_entry(codebook, key)
        writer.write_bits(codeword, length)
        return
    except (ImportError, KeyError, ValueError):
        pass

    # Fallback: simplified encoding
    # Encode each value with a variable-length code
    for val in values:
        abs_val = abs(int(val))
        if abs_val == 0:
            # Zero: 1-bit code
            writer.write_bits(1, 1)
        elif abs_val <= max_abs:
            # Small value: unary-like code
            # Length approximates Huffman table entry
            num_bits = int(np.log2(abs_val + 1)) + 2
            writer.write_bits(abs_val, num_bits)
        else:
            # Escape (codebook 11 only): encode escape flag + value
            if codebook == 11:
                _encode_escape(writer, abs_val)
            else:
                # Clamp to max_abs (shouldn't happen if codebook selection is correct)
                clamped = min(abs_val, max_abs)
                num_bits = int(np.log2(clamped + 1)) + 2
                writer.write_bits(clamped, num_bits)


def _encode_escape(writer: BitWriter, abs_val: int) -> None:
    """Encode an escape code for codebook 11 (values > 15).

    Escape format: N ones + one zero + (N+4) bit value
    where N = floor(log2(abs_val)) - 3

    Args:
        writer: BitWriter.
        abs_val: Absolute value to encode (must be > 15).
    """
    if abs_val <= 15:
        # Not actually an escape; encode normally
        writer.write_bits(abs_val, 5)
        return

    # Calculate escape parameters
    n = max(0, abs_val.bit_length() - 4)

    # Write N ones
    for _ in range(n):
        writer.write_bits(1, 1)
    # Write terminating zero
    writer.write_bits(0, 1)
    # Write (N+4) bit value
    writer.write_bits(abs_val, n + 4)
