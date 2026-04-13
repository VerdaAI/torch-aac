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
    CODEBOOKS,
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

    # Convert to a Python list of ints once — list iteration + indexing is
    # much cheaper than numpy element access in a tight loop.
    values_list = band_data.tolist()
    n = len(values_list)

    if is_unsigned:
        max_abs = CODEBOOK_MAX_ABS[codebook]
        clamp_to = 16 if codebook == 11 else max_abs
        is_cb11 = codebook == 11
        for i in range(0, n, dim):
            _encode_unsigned_group_fast(writer, values_list, i, dim, codebook, clamp_to, is_cb11)
    else:
        for i in range(0, n, dim):
            _encode_signed_group_fast(writer, values_list, i, dim, codebook)


def _encode_signed_group_fast(
    writer: BitWriter,
    values_list: list,
    start: int,
    dim: int,
    codebook: int,
) -> None:
    """Encode a signed-codebook group directly from a Python list slice."""
    key = tuple(values_list[start : start + dim])
    entry = CODEBOOKS[codebook].get(key) if CODEBOOKS[codebook] is not None else None  # type: ignore[union-attr]
    if entry is None:
        max_abs = CODEBOOK_MAX_ABS[codebook]
        clamped = tuple(max(-max_abs, min(max_abs, v)) for v in key)
        entry = get_codebook_entry(codebook, clamped)
    writer.write_bits(entry[0], entry[1])


def _encode_unsigned_group_fast(
    writer: BitWriter,
    values_list: list,
    start: int,
    dim: int,
    codebook: int,
    clamp_to: int,
    is_cb11: bool,
) -> None:
    """Encode an unsigned-codebook group directly from a Python list slice.

    Packs all non-zero sign bits into a single ``write_bits`` call. For
    codebook 11, escape mantissas are emitted after the sign bits.
    """
    # Build abs-tuple (for dict lookup) and collect sign-bit info in one pass.
    # Also tracks whether any value exceeds 15 (triggers cb11 escape).
    signs = 0
    nnz = 0
    abs_key: list[int] = [0] * dim
    any_escape = False
    for j in range(dim):
        v = values_list[start + j]
        av = -v if v < 0 else v
        if av > clamp_to:
            abs_key[j] = clamp_to
        else:
            abs_key[j] = av
        if av != 0:
            signs = (signs << 1) | (1 if v < 0 else 0)
            nnz += 1
        if is_cb11 and av > 15:
            any_escape = True

    key = tuple(abs_key)
    entry = CODEBOOKS[codebook].get(key) if CODEBOOKS[codebook] is not None else None  # type: ignore[union-attr]
    if entry is None:
        # Zero fallback — emit the all-zeros codeword and bail.
        entry = get_codebook_entry(codebook, (0,) * dim)
        writer.write_bits(entry[0], entry[1])
        return

    writer.write_bits(entry[0], entry[1])
    if nnz:
        writer.write_bits(signs, nnz)

    if any_escape:
        for j in range(dim):
            v = values_list[start + j]
            av = -v if v < 0 else v
            if av > 15:
                _encode_escape(writer, av)


MAX_ESCAPE_N = 8
"""Maximum escape sequence length. FFmpeg's decoder limits N to 8, allowing values up to 4095."""

MAX_ESCAPE_VAL = (1 << (MAX_ESCAPE_N + 4)) - 1  # 4095


def _encode_escape(writer: BitWriter, abs_val: int) -> None:
    """Encode an escape code for codebook 11 (values > 15).

    Escape format per ISO 14496-3:
        N ones + one zero + (N+4) bit mantissa
    where N = floor(log2(abs_val)) - 4

    The decoded value is reconstructed as ``n = 2^(N+4) + mantissa``, so the
    mantissa stored is ``abs_val - 2^(N+4)``.

    Values are clamped to MAX_ESCAPE_VAL (8191) to avoid decoder overflow.
    """
    abs_val = min(abs_val, MAX_ESCAPE_VAL)
    # N = bit_length - 5 (= floor(log2(abs_val)) - 4), since bit_length == floor(log2)+1.
    n = max(0, min(abs_val.bit_length() - 5, MAX_ESCAPE_N))
    mantissa_bits = n + 4
    mantissa = abs_val - (1 << mantissa_bits)
    # Write "N ones + zero + mantissa" in as few write_bits calls as possible.
    # Prefix = N ones followed by 1 zero = ((1 << n) - 1) << 1 in (n+1) bits.
    if n > 0:
        writer.write_bits(((1 << n) - 1) << 1, n + 1)  # N ones + zero
    else:
        writer.write_bits(0, 1)  # just the terminating zero
    writer.write_bits(mantissa, mantissa_bits)
