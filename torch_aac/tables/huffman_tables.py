"""AAC Huffman codebook tables for ISO 14496-3 AAC-LC spectral coding.

There are 12 codebooks (numbered 0-11).  Codebook 0 is the "zero" codebook
(no bits transmitted).  Codebooks 1-4 encode 4-tuples ("quads"), and
codebooks 5-11 encode 2-tuples ("pairs").

Some codebooks transmit unsigned magnitudes with separate sign bits
(``CODEBOOK_UNSIGNED[book] == True``); others transmit signed values.

STATUS (V1):
    The metadata arrays (``CODEBOOK_UNSIGNED``, ``CODEBOOK_DIMENSION``,
    ``CODEBOOK_MAX_ABS``) are correct per ISO 14496-3:2005.

    The actual Huffman codeword tables (``HCB_1`` through ``HCB_11``) are
    **not yet populated**.  The encoder's bit-packing layer
    (``cpu/huffman.py``) contains a fallback encoding path that is used
    when ``get_codebook_entry()`` raises ``ValueError``.

    TODO(v2): Populate the codebook dictionaries from an authoritative
    source (e.g. FFmpeg ``libavcodec/aactab.h``, FDK-AAC, or the ISO spec
    directly).  A generation script should be added at
    ``scripts/extract_huffman_tables.py``.

Reference:
    ISO/IEC 14496-3:2005, subpart 4, Tables 4.A.2 through 4.A.12.
    https://wiki.multimedia.cx/index.php/AAC_Huffman_Tables
"""

from __future__ import annotations

from typing import Final

# ============================================================================
# Codebook metadata -- VERIFIED against ISO 14496-3:2005
# ============================================================================

CODEBOOK_UNSIGNED: Final[list[bool]] = [
    False,  # 0 -- placeholder (zero codebook)
    False,  # 1 -- signed 4-tuple, values -1..+1
    False,  # 2 -- signed 4-tuple, values -1..+1
    True,   # 3 -- unsigned 4-tuple + sign bits, values 0..2
    True,   # 4 -- unsigned 4-tuple + sign bits, values 0..2
    False,  # 5 -- signed 2-tuple, values -4..+4
    False,  # 6 -- signed 2-tuple, values -4..+4
    True,   # 7 -- unsigned 2-tuple + sign bits, values 0..7
    True,   # 8 -- unsigned 2-tuple + sign bits, values 0..7
    True,   # 9 -- unsigned 2-tuple + sign bits, values 0..12
    True,   # 10 -- unsigned 2-tuple + sign bits, values 0..12
    True,   # 11 -- unsigned 2-tuple + sign bits, values 0..16 (escape)
]
"""Whether each codebook uses unsigned values (sign bits transmitted separately).

For unsigned codebooks, the encoder first transmits the Huffman code for the
tuple of absolute values, then appends one sign bit per non-zero value.
"""

CODEBOOK_DIMENSION: Final[list[int]] = [
    0,  # 0 -- placeholder
    4,  # 1
    4,  # 2
    4,  # 3
    4,  # 4
    2,  # 5
    2,  # 6
    2,  # 7
    2,  # 8
    2,  # 9
    2,  # 10
    2,  # 11
]
"""Number of spectral coefficients per Huffman symbol.

Codebooks 1-4 encode groups of 4 ("quads"); codebooks 5-11 encode pairs.
"""

CODEBOOK_MAX_ABS: Final[list[int]] = [
    0,   # 0 -- placeholder
    1,   # 1
    1,   # 2
    2,   # 3
    2,   # 4
    4,   # 5
    4,   # 6
    7,   # 7
    7,   # 8
    12,  # 9
    12,  # 10
    16,  # 11 (values > 15 use escape coding)
]
"""Maximum absolute value representable by each codebook.

For codebook 11, values exceeding 15 are transmitted via escape coding:
``N`` ones + zero + ``(N+4)``-bit mantissa, where ``N = floor(log2(|v|)) - 3``.
"""

CODEBOOK_LAV: Final[list[int]] = CODEBOOK_MAX_ABS
"""Largest Absolute Value -- alias for ``CODEBOOK_MAX_ABS``."""

# Number of entries per codebook.
# Quads (signed):   (2*max+1)^4; Quads (unsigned): (max+1)^4
# Pairs (signed):   (2*max+1)^2; Pairs (unsigned): (max+1)^2
CODEBOOK_NUM_ENTRIES: Final[list[int]] = [
    0,    # 0 -- placeholder
    81,   # 1: 3^4
    81,   # 2: 3^4
    81,   # 3: 3^4 (unsigned: values 0,1,2)
    81,   # 4: 3^4
    81,   # 5: 9^2
    81,   # 6: 9^2
    64,   # 7: 8^2
    64,   # 8: 8^2
    169,  # 9: 13^2
    169,  # 10: 13^2
    289,  # 11: 17^2
]
"""Number of entries (spectrum indices) in each codebook."""


# ============================================================================
# Codebook dictionaries
# ============================================================================
# Each maps tuple-of-values -> (codeword_bits, codeword_length).
#
# V1: These are empty stubs.  The encoder falls back to a simplified
# encoding scheme when entries are not found (see cpu/huffman.py).
#
# V2 TODO: Populate from ISO 14496-3 reference.
# ============================================================================

HCB_1: dict[tuple[int, ...], tuple[int, int]] = {}
"""Codebook 1: signed 4-tuple, values -1..+1, 81 entries."""

HCB_2: dict[tuple[int, ...], tuple[int, int]] = {}
"""Codebook 2: signed 4-tuple, values -1..+1, 81 entries."""

HCB_3: dict[tuple[int, ...], tuple[int, int]] = {}
"""Codebook 3: unsigned 4-tuple, values 0..2, 81 entries."""

HCB_4: dict[tuple[int, ...], tuple[int, int]] = {}
"""Codebook 4: unsigned 4-tuple, values 0..2, 81 entries."""

HCB_5: dict[tuple[int, ...], tuple[int, int]] = {}
"""Codebook 5: signed 2-tuple, values -4..+4, 81 entries."""

HCB_6: dict[tuple[int, ...], tuple[int, int]] = {}
"""Codebook 6: signed 2-tuple, values -4..+4, 81 entries."""

HCB_7: dict[tuple[int, ...], tuple[int, int]] = {}
"""Codebook 7: unsigned 2-tuple, values 0..7, 64 entries."""

HCB_8: dict[tuple[int, ...], tuple[int, int]] = {}
"""Codebook 8: unsigned 2-tuple, values 0..7, 64 entries."""

HCB_9: dict[tuple[int, ...], tuple[int, int]] = {}
"""Codebook 9: unsigned 2-tuple, values 0..12, 169 entries."""

HCB_10: dict[tuple[int, ...], tuple[int, int]] = {}
"""Codebook 10: unsigned 2-tuple, values 0..12, 169 entries."""

HCB_11: dict[tuple[int, ...], tuple[int, int]] = {}
"""Codebook 11: unsigned 2-tuple, values 0..16 (escape), 289 entries."""


# ============================================================================
# Master codebook list
# ============================================================================

CODEBOOKS: Final[list[dict[tuple[int, ...], tuple[int, int]] | None]] = [
    None,   # 0 -- "zero" section, no codebook
    HCB_1,  # 1
    HCB_2,  # 2
    HCB_3,  # 3
    HCB_4,  # 4
    HCB_5,  # 5
    HCB_6,  # 6
    HCB_7,  # 7
    HCB_8,  # 8
    HCB_9,  # 9
    HCB_10,  # 10
    HCB_11,  # 11
]
"""Codebook lookup list indexed by codebook number (0-11).

``CODEBOOKS[0]`` is ``None`` (zero codebook).  ``CODEBOOKS[n]`` for n=1..11
is the dictionary for codebook *n*.
"""


# ============================================================================
# Public API
# ============================================================================

def get_codebook_entry(
    book: int,
    values: tuple[int, ...],
) -> tuple[int, int]:
    """Look up the Huffman codeword for a tuple of quantized spectral values.

    Args:
        book: Codebook number (1-11).
        values: Tuple of quantized values.  Length must match the codebook
            dimension (4 for books 1-4, 2 for books 5-11).

    Returns:
        A ``(codeword_bits, codeword_length)`` tuple where *codeword_bits*
        is an integer whose *codeword_length* least-significant bits form
        the Huffman code (MSB-first).

    Raises:
        ValueError: If *book* is out of range, *values* has wrong dimension,
            or the entry is not found in the codebook (e.g. tables not yet
            populated).
    """
    if book < 1 or book > 11:
        raise ValueError(f"Codebook number must be 1-11, got {book}")

    codebook = CODEBOOKS[book]
    if codebook is None:
        raise ValueError(f"Codebook {book} is not available")

    expected_dim = CODEBOOK_DIMENSION[book]
    if len(values) != expected_dim:
        raise ValueError(
            f"Codebook {book} expects {expected_dim}-tuples, "
            f"got {len(values)}-tuple"
        )

    entry = codebook.get(values)
    if entry is None:
        raise ValueError(
            f"Values {values} not found in codebook {book}. "
            f"Tables may not be populated yet (V1 stub)."
        )
    return entry


def is_codebook_populated(book: int) -> bool:
    """Check whether a codebook has its Huffman table entries populated.

    Args:
        book: Codebook number (1-11).

    Returns:
        ``True`` if the codebook dictionary contains entries, ``False`` if
        it is empty (stub).
    """
    if book < 1 or book > 11:
        return False
    codebook = CODEBOOKS[book]
    return codebook is not None and len(codebook) > 0


def get_num_populated() -> int:
    """Return the number of codebooks that have their tables populated.

    Returns:
        Count of codebooks (1-11) that contain at least one entry.
    """
    return sum(1 for i in range(1, 12) if is_codebook_populated(i))
