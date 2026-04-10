"""AAC Huffman codebook tables for ISO 14496-3 AAC-LC spectral coding.

There are 12 codebooks (numbered 0-11).  Codebook 0 is the "zero" codebook
(no bits transmitted).  Codebooks 1-4 encode 4-tuples ("quads"), and
codebooks 5-11 encode 2-tuples ("pairs").

Some codebooks transmit unsigned magnitudes with separate sign bits
(``CODEBOOK_UNSIGNED[book] == True``); others transmit signed values.

The raw Huffman codeword/bit-length data originates from the ISO/IEC 14496-3
standard (Tables 4.A.2 through 4.A.12).  The numeric arrays below were
extracted from FFmpeg's ``libavcodec/aactab.c`` (LGPL 2.1).  Only the raw
table data is reproduced here — this constitutes factual information defined
by the ISO specification and is not copyrightable expression.

Reference:
    ISO/IEC 14496-3:2005, subpart 4, Tables 4.A.2 through 4.A.12.
    FFmpeg ``libavcodec/aactab.c`` (for the numeric values).
"""

from __future__ import annotations

from typing import Final

# ============================================================================
# Raw Huffman codeword arrays (codes) — indexed by spectrum index
# ============================================================================
# Codebook N: codes_N[i] is the Huffman codeword (as an integer, MSB-first)
# for the spectrum index i.  The corresponding bit length is bits_N[i].

CODES_1: Final[list[int]] = [
    0x7F8, 0x1F1, 0x7FD, 0x3F5, 0x068, 0x3F0, 0x7F7, 0x1EC,
    0x7F5, 0x3F1, 0x072, 0x3F4, 0x074, 0x011, 0x076, 0x1EB,
    0x06C, 0x3F6, 0x7FC, 0x1E1, 0x7F1, 0x1F0, 0x061, 0x1F6,
    0x7F2, 0x1EA, 0x7FB, 0x1F2, 0x069, 0x1ED, 0x077, 0x017,
    0x06F, 0x1E6, 0x064, 0x1E5, 0x067, 0x015, 0x062, 0x012,
    0x000, 0x014, 0x065, 0x016, 0x06D, 0x1E9, 0x063, 0x1E4,
    0x06B, 0x013, 0x071, 0x1E3, 0x070, 0x1F3, 0x7FE, 0x1E7,
    0x7F3, 0x1EF, 0x060, 0x1EE, 0x7F0, 0x1E2, 0x7FA, 0x3F3,
    0x06A, 0x1E8, 0x075, 0x010, 0x073, 0x1F4, 0x06E, 0x3F7,
    0x7F6, 0x1E0, 0x7F9, 0x3F2, 0x066, 0x1F5, 0x7FF, 0x1F7,
    0x7F4,
]
"""Codebook 1 codewords: signed 4-tuple, values {-1, 0, 1}, 81 entries."""

BITS_1: Final[list[int]] = [
    11,  9, 11, 10,  7, 10, 11,  9, 11, 10,  7, 10,  7,  5,  7,  9,
     7, 10, 11,  9, 11,  9,  7,  9, 11,  9, 11,  9,  7,  9,  7,  5,
     7,  9,  7,  9,  7,  5,  7,  5,  1,  5,  7,  5,  7,  9,  7,  9,
     7,  5,  7,  9,  7,  9, 11,  9, 11,  9,  7,  9, 11,  9, 11, 10,
     7,  9,  7,  5,  7,  9,  7, 10, 11,  9, 11, 10,  7,  9, 11,  9,
    11,
]
"""Codebook 1 bit lengths."""

CODES_2: Final[list[int]] = [
    0x1F3, 0x06F, 0x1FD, 0x0EB, 0x023, 0x0EA, 0x1F7, 0x0E8,
    0x1FA, 0x0F2, 0x02D, 0x070, 0x020, 0x006, 0x02B, 0x06E,
    0x028, 0x0E9, 0x1F9, 0x066, 0x0F8, 0x0E7, 0x01B, 0x0F1,
    0x1F4, 0x06B, 0x1F5, 0x0EC, 0x02A, 0x06C, 0x02C, 0x00A,
    0x027, 0x067, 0x01A, 0x0F5, 0x024, 0x008, 0x01F, 0x009,
    0x000, 0x007, 0x01D, 0x00B, 0x030, 0x0EF, 0x01C, 0x064,
    0x01E, 0x00C, 0x029, 0x0F3, 0x02F, 0x0F0, 0x1FC, 0x071,
    0x1F2, 0x0F4, 0x021, 0x0E6, 0x0F7, 0x068, 0x1F8, 0x0EE,
    0x022, 0x065, 0x031, 0x002, 0x026, 0x0ED, 0x025, 0x06A,
    0x1FB, 0x072, 0x1FE, 0x069, 0x02E, 0x0F6, 0x1FF, 0x06D,
    0x1F6,
]
"""Codebook 2 codewords: signed 4-tuple, values {-1, 0, 1}, 81 entries."""

BITS_2: Final[list[int]] = [
    9, 7, 9, 8, 6, 8, 9, 8, 9, 8, 6, 7, 6, 5, 6, 7,
    6, 8, 9, 7, 8, 8, 6, 8, 9, 7, 9, 8, 6, 7, 6, 5,
    6, 7, 6, 8, 6, 5, 6, 5, 3, 5, 6, 5, 6, 8, 6, 7,
    6, 5, 6, 8, 6, 8, 9, 7, 9, 8, 6, 8, 8, 7, 9, 8,
    6, 7, 6, 4, 6, 8, 6, 7, 9, 7, 9, 7, 6, 8, 9, 7,
    9,
]
"""Codebook 2 bit lengths."""

CODES_3: Final[list[int]] = [
    0x0000, 0x0009, 0x00EF, 0x000B, 0x0019, 0x00F0, 0x01EB, 0x01E6,
    0x03F2, 0x000A, 0x0035, 0x01EF, 0x0034, 0x0037, 0x01E9, 0x01ED,
    0x01E7, 0x03F3, 0x01EE, 0x03ED, 0x1FFA, 0x01EC, 0x01F2, 0x07F9,
    0x07F8, 0x03F8, 0x0FF8, 0x0008, 0x0038, 0x03F6, 0x0036, 0x0075,
    0x03F1, 0x03EB, 0x03EC, 0x0FF4, 0x0018, 0x0076, 0x07F4, 0x0039,
    0x0074, 0x03EF, 0x01F3, 0x01F4, 0x07F6, 0x01E8, 0x03EA, 0x1FFC,
    0x00F2, 0x01F1, 0x0FFB, 0x03F5, 0x07F3, 0x0FFC, 0x00EE, 0x03F7,
    0x7FFE, 0x01F0, 0x07F5, 0x7FFD, 0x1FFB, 0x3FFA, 0xFFFF, 0x00F1,
    0x03F0, 0x3FFC, 0x01EA, 0x03EE, 0x3FFB, 0x0FF6, 0x0FFA, 0x7FFC,
    0x07F2, 0x0FF5, 0xFFFE, 0x03F4, 0x07F7, 0x7FFB, 0x0FF7, 0x0FF9,
    0x7FFA,
]
"""Codebook 3 codewords: unsigned 4-tuple, values {0, 1, 2}, 81 entries."""

BITS_3: Final[list[int]] = [
     1,  4,  8,  4,  5,  8,  9,  9, 10,  4,  6,  9,  6,  6,  9,  9,
     9, 10,  9, 10, 13,  9,  9, 11, 11, 10, 12,  4,  6, 10,  6,  7,
    10, 10, 10, 12,  5,  7, 11,  6,  7, 10,  9,  9, 11,  9, 10, 13,
     8,  9, 12, 10, 11, 12,  8, 10, 15,  9, 11, 15, 13, 14, 16,  8,
    10, 14,  9, 10, 14, 12, 12, 15, 11, 12, 16, 10, 11, 15, 12, 12,
    15,
]
"""Codebook 3 bit lengths."""

CODES_4: Final[list[int]] = [
    0x007, 0x016, 0x0F6, 0x018, 0x008, 0x0EF, 0x1EF, 0x0F3,
    0x7F8, 0x019, 0x017, 0x0ED, 0x015, 0x001, 0x0E2, 0x0F0,
    0x070, 0x3F0, 0x1EE, 0x0F1, 0x7FA, 0x0EE, 0x0E4, 0x3F2,
    0x7F6, 0x3EF, 0x7FD, 0x005, 0x014, 0x0F2, 0x009, 0x004,
    0x0E5, 0x0F4, 0x0E8, 0x3F4, 0x006, 0x002, 0x0E7, 0x003,
    0x000, 0x06B, 0x0E3, 0x069, 0x1F3, 0x0EB, 0x0E6, 0x3F6,
    0x06E, 0x06A, 0x1F4, 0x3EC, 0x1F0, 0x3F9, 0x0F5, 0x0EC,
    0x7FB, 0x0EA, 0x06F, 0x3F7, 0x7F9, 0x3F3, 0x0FFF, 0x0E9,
    0x06D, 0x3F8, 0x06C, 0x068, 0x1F5, 0x3EE, 0x1F2, 0x7F4,
    0x7F7, 0x3F1, 0x0FFE, 0x3ED, 0x1F1, 0x7F5, 0x7FE, 0x3F5,
    0x7FC,
]
"""Codebook 4 codewords: unsigned 4-tuple, values {0, 1, 2}, 81 entries."""

BITS_4: Final[list[int]] = [
     4,  5,  8,  5,  4,  8,  9,  8, 11,  5,  5,  8,  5,  4,  8,  8,
     7, 10,  9,  8, 11,  8,  8, 10, 11, 10, 11,  4,  5,  8,  4,  4,
     8,  8,  8, 10,  4,  4,  8,  4,  4,  7,  8,  7,  9,  8,  8, 10,
     7,  7,  9, 10,  9, 10,  8,  8, 11,  8,  7, 10, 11, 10, 12,  8,
     7, 10,  7,  7,  9, 10,  9, 11, 11, 10, 12, 10,  9, 11, 11, 10,
    11,
]
"""Codebook 4 bit lengths."""

CODES_5: Final[list[int]] = [
    0x1FFF, 0x0FF7, 0x07F4, 0x07E8, 0x03F1, 0x07EE, 0x07F9, 0x0FF8,
    0x1FFD, 0x0FFD, 0x07F1, 0x03E8, 0x01E8, 0x00F0, 0x01EC, 0x03EE,
    0x07F2, 0x0FFA, 0x0FF4, 0x03EF, 0x01F2, 0x00E8, 0x0070, 0x00EC,
    0x01F0, 0x03EA, 0x07F3, 0x07EB, 0x01EB, 0x00EA, 0x001A, 0x0008,
    0x0019, 0x00EE, 0x01EF, 0x07ED, 0x03F0, 0x00F2, 0x0073, 0x000B,
    0x0000, 0x000A, 0x0071, 0x00F3, 0x07E9, 0x07EF, 0x01EE, 0x00EF,
    0x0018, 0x0009, 0x001B, 0x00EB, 0x01E9, 0x07EC, 0x07F6, 0x03EB,
    0x01F3, 0x00ED, 0x0072, 0x00E9, 0x01F1, 0x03ED, 0x07F7, 0x0FF6,
    0x07F0, 0x03E9, 0x01ED, 0x00F1, 0x01EA, 0x03EC, 0x07F8, 0x0FF9,
    0x1FFC, 0x0FFC, 0x0FF5, 0x07EA, 0x03F3, 0x03F2, 0x07F5, 0x0FFB,
    0x1FFE,
]
"""Codebook 5 codewords: signed 2-tuple, values {-4..4}, 81 entries."""

BITS_5: Final[list[int]] = [
    13, 12, 11, 11, 10, 11, 11, 12, 13, 12, 11, 10,  9,  8,  9, 10,
    11, 12, 12, 10,  9,  8,  7,  8,  9, 10, 11, 11,  9,  8,  5,  4,
     5,  8,  9, 11, 10,  8,  7,  4,  1,  4,  7,  8, 11, 11,  9,  8,
     5,  4,  5,  8,  9, 11, 11, 10,  9,  8,  7,  8,  9, 10, 11, 12,
    11, 10,  9,  8,  9, 10, 11, 12, 13, 12, 12, 11, 10, 10, 11, 12,
    13,
]
"""Codebook 5 bit lengths."""

CODES_6: Final[list[int]] = [
    0x7FE, 0x3FD, 0x1F1, 0x1EB, 0x1F4, 0x1EA, 0x1F0, 0x3FC,
    0x7FD, 0x3F6, 0x1E5, 0x0EA, 0x06C, 0x071, 0x068, 0x0F0,
    0x1E6, 0x3F7, 0x1F3, 0x0EF, 0x032, 0x027, 0x028, 0x026,
    0x031, 0x0EB, 0x1F7, 0x1E8, 0x06F, 0x02E, 0x008, 0x004,
    0x006, 0x029, 0x06B, 0x1EE, 0x1EF, 0x072, 0x02D, 0x002,
    0x000, 0x003, 0x02F, 0x073, 0x1FA, 0x1E7, 0x06E, 0x02B,
    0x007, 0x001, 0x005, 0x02C, 0x06D, 0x1EC, 0x1F9, 0x0EE,
    0x030, 0x024, 0x02A, 0x025, 0x033, 0x0EC, 0x1F2, 0x3F8,
    0x1E4, 0x0ED, 0x06A, 0x070, 0x069, 0x074, 0x0F1, 0x3FA,
    0x7FF, 0x3F9, 0x1F6, 0x1ED, 0x1F8, 0x1E9, 0x1F5, 0x3FB,
    0x7FC,
]
"""Codebook 6 codewords: signed 2-tuple, values {-4..4}, 81 entries."""

BITS_6: Final[list[int]] = [
    11, 10,  9,  9,  9,  9,  9, 10, 11, 10,  9,  8,  7,  7,  7,  8,
     9, 10,  9,  8,  6,  6,  6,  6,  6,  8,  9,  9,  7,  6,  4,  4,
     4,  6,  7,  9,  9,  7,  6,  4,  4,  4,  6,  7,  9,  9,  7,  6,
     4,  4,  4,  6,  7,  9,  9,  8,  6,  6,  6,  6,  6,  8,  9, 10,
     9,  8,  7,  7,  7,  7,  8, 10, 11, 10,  9,  9,  9,  9,  9, 10,
    11,
]
"""Codebook 6 bit lengths."""

CODES_7: Final[list[int]] = [
    0x000, 0x005, 0x037, 0x074, 0x0F2, 0x1EB, 0x3ED, 0x7F7,
    0x004, 0x00C, 0x035, 0x071, 0x0EC, 0x0EE, 0x1EE, 0x1F5,
    0x036, 0x034, 0x072, 0x0EA, 0x0F1, 0x1E9, 0x1F3, 0x3F5,
    0x073, 0x070, 0x0EB, 0x0F0, 0x1F1, 0x1F0, 0x3EC, 0x3FA,
    0x0F3, 0x0ED, 0x1E8, 0x1EF, 0x3EF, 0x3F1, 0x3F9, 0x7FB,
    0x1ED, 0x0EF, 0x1EA, 0x1F2, 0x3F3, 0x3F8, 0x7F9, 0x7FC,
    0x3EE, 0x1EC, 0x1F4, 0x3F4, 0x3F7, 0x7F8, 0xFFD, 0xFFE,
    0x7F6, 0x3F0, 0x3F2, 0x3F6, 0x7FA, 0x7FD, 0xFFC, 0xFFF,
]
"""Codebook 7 codewords: unsigned 2-tuple, values {0..7}, 64 entries."""

BITS_7: Final[list[int]] = [
     1,  3,  6,  7,  8,  9, 10, 11,  3,  4,  6,  7,  8,  8,  9,  9,
     6,  6,  7,  8,  8,  9,  9, 10,  7,  7,  8,  8,  9,  9, 10, 10,
     8,  8,  9,  9, 10, 10, 10, 11,  9,  8,  9,  9, 10, 10, 11, 11,
    10,  9,  9, 10, 10, 11, 12, 12, 11, 10, 10, 10, 11, 11, 12, 12,
]
"""Codebook 7 bit lengths."""

CODES_8: Final[list[int]] = [
    0x00E, 0x005, 0x010, 0x030, 0x06F, 0x0F1, 0x1FA, 0x3FE,
    0x003, 0x000, 0x004, 0x012, 0x02C, 0x06A, 0x075, 0x0F8,
    0x00F, 0x002, 0x006, 0x014, 0x02E, 0x069, 0x072, 0x0F5,
    0x02F, 0x011, 0x013, 0x02A, 0x032, 0x06C, 0x0EC, 0x0FA,
    0x071, 0x02B, 0x02D, 0x031, 0x06D, 0x070, 0x0F2, 0x1F9,
    0x0EF, 0x068, 0x033, 0x06B, 0x06E, 0x0EE, 0x0F9, 0x3FC,
    0x1F8, 0x074, 0x073, 0x0ED, 0x0F0, 0x0F6, 0x1F6, 0x1FD,
    0x3FD, 0x0F3, 0x0F4, 0x0F7, 0x1F7, 0x1FB, 0x1FC, 0x3FF,
]
"""Codebook 8 codewords: unsigned 2-tuple, values {0..7}, 64 entries."""

BITS_8: Final[list[int]] = [
     5,  4,  5,  6,  7,  8,  9, 10,  4,  3,  4,  5,  6,  7,  7,  8,
     5,  4,  4,  5,  6,  7,  7,  8,  6,  5,  5,  6,  6,  7,  8,  8,
     7,  6,  6,  6,  7,  7,  8,  9,  8,  7,  6,  7,  7,  8,  8, 10,
     9,  7,  7,  8,  8,  8,  9,  9, 10,  8,  8,  8,  9,  9,  9, 10,
]
"""Codebook 8 bit lengths."""

CODES_9: Final[list[int]] = [
    0x0000, 0x0005, 0x0037, 0x00E7, 0x01DE, 0x03CE, 0x03D9, 0x07C8,
    0x07CD, 0x0FC8, 0x0FDD, 0x1FE4, 0x1FEC, 0x0004, 0x000C, 0x0035,
    0x0072, 0x00EA, 0x00ED, 0x01E2, 0x03D1, 0x03D3, 0x03E0, 0x07D8,
    0x0FCF, 0x0FD5, 0x0036, 0x0034, 0x0071, 0x00E8, 0x00EC, 0x01E1,
    0x03CF, 0x03DD, 0x03DB, 0x07D0, 0x0FC7, 0x0FD4, 0x0FE4, 0x00E6,
    0x0070, 0x00E9, 0x01DD, 0x01E3, 0x03D2, 0x03DC, 0x07CC, 0x07CA,
    0x07DE, 0x0FD8, 0x0FEA, 0x1FDB, 0x01DF, 0x00EB, 0x01DC, 0x01E6,
    0x03D5, 0x03DE, 0x07CB, 0x07DD, 0x07DC, 0x0FCD, 0x0FE2, 0x0FE7,
    0x1FE1, 0x03D0, 0x01E0, 0x01E4, 0x03D6, 0x07C5, 0x07D1, 0x07DB,
    0x0FD2, 0x07E0, 0x0FD9, 0x0FEB, 0x1FE3, 0x1FE9, 0x07C4, 0x01E5,
    0x03D7, 0x07C6, 0x07CF, 0x07DA, 0x0FCB, 0x0FDA, 0x0FE3, 0x0FE9,
    0x1FE6, 0x1FF3, 0x1FF7, 0x07D3, 0x03D8, 0x03E1, 0x07D4, 0x07D9,
    0x0FD3, 0x0FDE, 0x1FDD, 0x1FD9, 0x1FE2, 0x1FEA, 0x1FF1, 0x1FF6,
    0x07D2, 0x03D4, 0x03DA, 0x07C7, 0x07D7, 0x07E2, 0x0FCE, 0x0FDB,
    0x1FD8, 0x1FEE, 0x3FF0, 0x1FF4, 0x3FF2, 0x07E1, 0x03DF, 0x07C9,
    0x07D6, 0x0FCA, 0x0FD0, 0x0FE5, 0x0FE6, 0x1FEB, 0x1FEF, 0x3FF3,
    0x3FF4, 0x3FF5, 0x0FE0, 0x07CE, 0x07D5, 0x0FC6, 0x0FD1, 0x0FE1,
    0x1FE0, 0x1FE8, 0x1FF0, 0x3FF1, 0x3FF8, 0x3FF6, 0x7FFC, 0x0FE8,
    0x07DF, 0x0FC9, 0x0FD7, 0x0FDC, 0x1FDC, 0x1FDF, 0x1FED, 0x1FF5,
    0x3FF9, 0x3FFB, 0x7FFD, 0x7FFE, 0x1FE7, 0x0FCC, 0x0FD6, 0x0FDF,
    0x1FDE, 0x1FDA, 0x1FE5, 0x1FF2, 0x3FFA, 0x3FF7, 0x3FFC, 0x3FFD,
    0x7FFF,
]
"""Codebook 9 codewords: unsigned 2-tuple, values {0..12}, 169 entries."""

BITS_9: Final[list[int]] = [
     1,  3,  6,  8,  9, 10, 10, 11, 11, 12, 12, 13, 13,  3,  4,  6,
     7,  8,  8,  9, 10, 10, 10, 11, 12, 12,  6,  6,  7,  8,  8,  9,
    10, 10, 10, 11, 12, 12, 12,  8,  7,  8,  9,  9, 10, 10, 11, 11,
    11, 12, 12, 13,  9,  8,  9,  9, 10, 10, 11, 11, 11, 12, 12, 12,
    13, 10,  9,  9, 10, 11, 11, 11, 12, 11, 12, 12, 13, 13, 11,  9,
    10, 11, 11, 11, 12, 12, 12, 12, 13, 13, 13, 11, 10, 10, 11, 11,
    12, 12, 13, 13, 13, 13, 13, 13, 11, 10, 10, 11, 11, 11, 12, 12,
    13, 13, 14, 13, 14, 11, 10, 11, 11, 12, 12, 12, 12, 13, 13, 14,
    14, 14, 12, 11, 11, 12, 12, 12, 13, 13, 13, 14, 14, 14, 15, 12,
    11, 12, 12, 12, 13, 13, 13, 13, 14, 14, 15, 15, 13, 12, 12, 12,
    13, 13, 13, 13, 14, 14, 14, 14, 15,
]
"""Codebook 9 bit lengths."""

CODES_10: Final[list[int]] = [
    0x022, 0x008, 0x01D, 0x026, 0x05F, 0x0D3, 0x1CF, 0x3D0,
    0x3D7, 0x3ED, 0x7F0, 0x7F6, 0xFFD, 0x007, 0x000, 0x001,
    0x009, 0x020, 0x054, 0x060, 0x0D5, 0x0DC, 0x1D4, 0x3CD,
    0x3DE, 0x7E7, 0x01C, 0x002, 0x006, 0x00C, 0x01E, 0x028,
    0x05B, 0x0CD, 0x0D9, 0x1CE, 0x1DC, 0x3D9, 0x3F1, 0x025,
    0x00B, 0x00A, 0x00D, 0x024, 0x057, 0x061, 0x0CC, 0x0DD,
    0x1CC, 0x1DE, 0x3D3, 0x3E7, 0x05D, 0x021, 0x01F, 0x023,
    0x027, 0x059, 0x064, 0x0D8, 0x0DF, 0x1D2, 0x1E2, 0x3DD,
    0x3EE, 0x0D1, 0x055, 0x029, 0x056, 0x058, 0x062, 0x0CE,
    0x0E0, 0x0E2, 0x1DA, 0x3D4, 0x3E3, 0x7EB, 0x1C9, 0x05E,
    0x05A, 0x05C, 0x063, 0x0CA, 0x0DA, 0x1C7, 0x1CA, 0x1E0,
    0x3DB, 0x3E8, 0x7EC, 0x1E3, 0x0D2, 0x0CB, 0x0D0, 0x0D7,
    0x0DB, 0x1C6, 0x1D5, 0x1D8, 0x3CA, 0x3DA, 0x7EA, 0x7F1,
    0x1E1, 0x0D4, 0x0CF, 0x0D6, 0x0DE, 0x0E1, 0x1D0, 0x1D6,
    0x3D1, 0x3D5, 0x3F2, 0x7EE, 0x7FB, 0x3E9, 0x1CD, 0x1C8,
    0x1CB, 0x1D1, 0x1D7, 0x1DF, 0x3CF, 0x3E0, 0x3EF, 0x7E6,
    0x7F8, 0xFFA, 0x3EB, 0x1DD, 0x1D3, 0x1D9, 0x1DB, 0x3D2,
    0x3CC, 0x3DC, 0x3EA, 0x7ED, 0x7F3, 0x7F9, 0xFF9, 0x7F2,
    0x3CE, 0x1E4, 0x3CB, 0x3D8, 0x3D6, 0x3E2, 0x3E5, 0x7E8,
    0x7F4, 0x7F5, 0x7F7, 0xFFB, 0x7FA, 0x3EC, 0x3DF, 0x3E1,
    0x3E4, 0x3E6, 0x3F0, 0x7E9, 0x7EF, 0xFF8, 0xFFE, 0xFFC,
    0xFFF,
]
"""Codebook 10 codewords: unsigned 2-tuple, values {0..12}, 169 entries."""

BITS_10: Final[list[int]] = [
     6,  5,  6,  6,  7,  8,  9, 10, 10, 10, 11, 11, 12,  5,  4,  4,
     5,  6,  7,  7,  8,  8,  9, 10, 10, 11,  6,  4,  5,  5,  6,  6,
     7,  8,  8,  9,  9, 10, 10,  6,  5,  5,  5,  6,  7,  7,  8,  8,
     9,  9, 10, 10,  7,  6,  6,  6,  6,  7,  7,  8,  8,  9,  9, 10,
    10,  8,  7,  6,  7,  7,  7,  8,  8,  8,  9, 10, 10, 11,  9,  7,
     7,  7,  7,  8,  8,  9,  9,  9, 10, 10, 11,  9,  8,  8,  8,  8,
     8,  9,  9,  9, 10, 10, 11, 11,  9,  8,  8,  8,  8,  8,  9,  9,
    10, 10, 10, 11, 11, 10,  9,  9,  9,  9,  9,  9, 10, 10, 10, 11,
    11, 12, 10,  9,  9,  9,  9, 10, 10, 10, 10, 11, 11, 11, 12, 11,
    10,  9, 10, 10, 10, 10, 10, 11, 11, 11, 11, 12, 11, 10, 10, 10,
    10, 10, 10, 11, 11, 12, 12, 12, 12,
]
"""Codebook 10 bit lengths."""

CODES_11: Final[list[int]] = [
    0x000, 0x006, 0x019, 0x03D, 0x09C, 0x0C6, 0x1A7, 0x390,
    0x3C2, 0x3DF, 0x7E6, 0x7F3, 0xFFB, 0x7EC, 0xFFA, 0xFFE,
    0x38E, 0x005, 0x001, 0x008, 0x014, 0x037, 0x042, 0x092,
    0x0AF, 0x191, 0x1A5, 0x1B5, 0x39E, 0x3C0, 0x3A2, 0x3CD,
    0x7D6, 0x0AE, 0x017, 0x007, 0x009, 0x018, 0x039, 0x040,
    0x08E, 0x0A3, 0x0B8, 0x199, 0x1AC, 0x1C1, 0x3B1, 0x396,
    0x3BE, 0x3CA, 0x09D, 0x03C, 0x015, 0x016, 0x01A, 0x03B,
    0x044, 0x091, 0x0A5, 0x0BE, 0x196, 0x1AE, 0x1B9, 0x3A1,
    0x391, 0x3A5, 0x3D5, 0x094, 0x09A, 0x036, 0x038, 0x03A,
    0x041, 0x08C, 0x09B, 0x0B0, 0x0C3, 0x19E, 0x1AB, 0x1BC,
    0x39F, 0x38F, 0x3A9, 0x3CF, 0x093, 0x0BF, 0x03E, 0x03F,
    0x043, 0x045, 0x09E, 0x0A7, 0x0B9, 0x194, 0x1A2, 0x1BA,
    0x1C3, 0x3A6, 0x3A7, 0x3BB, 0x3D4, 0x09F, 0x1A0, 0x08F,
    0x08D, 0x090, 0x098, 0x0A6, 0x0B6, 0x0C4, 0x19F, 0x1AF,
    0x1BF, 0x399, 0x3BF, 0x3B4, 0x3C9, 0x3E7, 0x0A8, 0x1B6,
    0x0AB, 0x0A4, 0x0AA, 0x0B2, 0x0C2, 0x0C5, 0x198, 0x1A4,
    0x1B8, 0x38C, 0x3A4, 0x3C4, 0x3C6, 0x3DD, 0x3E8, 0x0AD,
    0x3AF, 0x192, 0x0BD, 0x0BC, 0x18E, 0x197, 0x19A, 0x1A3,
    0x1B1, 0x38D, 0x398, 0x3B7, 0x3D3, 0x3D1, 0x3DB, 0x7DD,
    0x0B4, 0x3DE, 0x1A9, 0x19B, 0x19C, 0x1A1, 0x1AA, 0x1AD,
    0x1B3, 0x38B, 0x3B2, 0x3B8, 0x3CE, 0x3E1, 0x3E0, 0x7D2,
    0x7E5, 0x0B7, 0x7E3, 0x1BB, 0x1A8, 0x1A6, 0x1B0, 0x1B2,
    0x1B7, 0x39B, 0x39A, 0x3BA, 0x3B5, 0x3D6, 0x7D7, 0x3E4,
    0x7D8, 0x7EA, 0x0BA, 0x7E8, 0x3A0, 0x1BD, 0x1B4, 0x38A,
    0x1C4, 0x392, 0x3AA, 0x3B0, 0x3BC, 0x3D7, 0x7D4, 0x7DC,
    0x7DB, 0x7D5, 0x7F0, 0x0C1, 0x7FB, 0x3C8, 0x3A3, 0x395,
    0x39D, 0x3AC, 0x3AE, 0x3C5, 0x3D8, 0x3E2, 0x3E6, 0x7E4,
    0x7E7, 0x7E0, 0x7E9, 0x7F7, 0x190, 0x7F2, 0x393, 0x1BE,
    0x1C0, 0x394, 0x397, 0x3AD, 0x3C3, 0x3C1, 0x3D2, 0x7DA,
    0x7D9, 0x7DF, 0x7EB, 0x7F4, 0x7FA, 0x195, 0x7F8, 0x3BD,
    0x39C, 0x3AB, 0x3A8, 0x3B3, 0x3B9, 0x3D0, 0x3E3, 0x3E5,
    0x7E2, 0x7DE, 0x7ED, 0x7F1, 0x7F9, 0x7FC, 0x193, 0xFFD,
    0x3DC, 0x3B6, 0x3C7, 0x3CC, 0x3CB, 0x3D9, 0x3DA, 0x7D3,
    0x7E1, 0x7EE, 0x7EF, 0x7F5, 0x7F6, 0xFFC, 0xFFF, 0x19D,
    0x1C2, 0x0B5, 0x0A1, 0x096, 0x097, 0x095, 0x099, 0x0A0,
    0x0A2, 0x0AC, 0x0A9, 0x0B1, 0x0B3, 0x0BB, 0x0C0, 0x18F,
    0x004,
]
"""Codebook 11 codewords: unsigned 2-tuple, values {0..16} (escape), 289 entries."""

BITS_11: Final[list[int]] = [
     4,  5,  6,  7,  8,  8,  9, 10, 10, 10, 11, 11, 12, 11, 12, 12,
    10,  5,  4,  5,  6,  7,  7,  8,  8,  9,  9,  9, 10, 10, 10, 10,
    11,  8,  6,  5,  5,  6,  7,  7,  8,  8,  8,  9,  9,  9, 10, 10,
    10, 10,  8,  7,  6,  6,  6,  7,  7,  8,  8,  8,  9,  9,  9, 10,
    10, 10, 10,  8,  8,  7,  7,  7,  7,  8,  8,  8,  8,  9,  9,  9,
    10, 10, 10, 10,  8,  8,  7,  7,  7,  7,  8,  8,  8,  9,  9,  9,
     9, 10, 10, 10, 10,  8,  9,  8,  8,  8,  8,  8,  8,  8,  9,  9,
     9, 10, 10, 10, 10, 10,  8,  9,  8,  8,  8,  8,  8,  8,  9,  9,
     9, 10, 10, 10, 10, 10, 10,  8, 10,  9,  8,  8,  9,  9,  9,  9,
     9, 10, 10, 10, 10, 10, 10, 11,  8, 10,  9,  9,  9,  9,  9,  9,
     9, 10, 10, 10, 10, 10, 10, 11, 11,  8, 11,  9,  9,  9,  9,  9,
     9, 10, 10, 10, 10, 10, 11, 10, 11, 11,  8, 11, 10,  9,  9, 10,
     9, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11,  8, 11, 10, 10, 10,
    10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11,  9, 11, 10,  9,
     9, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11,  9, 11, 10,
    10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11,  9, 12,
    10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12,  9,
     9,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  9,
     5,
]
"""Codebook 11 bit lengths."""

# ============================================================================
# Scalefactor Huffman table (codebook for scalefactor indices)
# ============================================================================

SCALEFACTOR_CODE: Final[list[int]] = [
    0x3FFE8, 0x3FFE6, 0x3FFE7, 0x3FFE5, 0x7FFF5, 0x7FFF1, 0x7FFED, 0x7FFF6,
    0x7FFEE, 0x7FFEF, 0x7FFF0, 0x7FFFC, 0x7FFFD, 0x7FFFF, 0x7FFFE, 0x7FFF7,
    0x7FFF8, 0x7FFFB, 0x7FFF9, 0x3FFE4, 0x7FFFA, 0x3FFE3, 0x1FFEF, 0x1FFF0,
    0x0FFF5, 0x1FFEE, 0x0FFF2, 0x0FFF3, 0x0FFF4, 0x0FFF1, 0x07FF6, 0x07FF7,
    0x03FF9, 0x03FF5, 0x03FF7, 0x03FF3, 0x03FF6, 0x03FF2, 0x01FF7, 0x01FF5,
    0x00FF9, 0x00FF7, 0x00FF6, 0x007F9, 0x00FF4, 0x007F8, 0x003F9, 0x003F7,
    0x003F5, 0x001F8, 0x001F7, 0x000FA, 0x000F8, 0x000F6, 0x00079, 0x0003A,
    0x00038, 0x0001A, 0x0000B, 0x00004, 0x00000, 0x0000A, 0x0000C, 0x0001B,
    0x00039, 0x0003B, 0x00078, 0x0007A, 0x000F7, 0x000F9, 0x001F6, 0x001F9,
    0x003F4, 0x003F6, 0x003F8, 0x007F5, 0x007F4, 0x007F6, 0x007F7, 0x00FF5,
    0x00FF8, 0x01FF4, 0x01FF6, 0x01FF8, 0x03FF8, 0x03FF4, 0x0FFF0, 0x07FF4,
    0x0FFF6, 0x07FF5, 0x3FFE2, 0x7FFD9, 0x7FFDA, 0x7FFDB, 0x7FFDC, 0x7FFDD,
    0x7FFDE, 0x7FFD8, 0x7FFD2, 0x7FFD3, 0x7FFD4, 0x7FFD5, 0x7FFD6, 0x7FFF2,
    0x7FFDF, 0x7FFE7, 0x7FFE8, 0x7FFE9, 0x7FFEA, 0x7FFEB, 0x7FFE6, 0x7FFE0,
    0x7FFE1, 0x7FFE2, 0x7FFE3, 0x7FFE4, 0x7FFE5, 0x7FFD7, 0x7FFEC, 0x7FFF4,
    0x7FFF3,
]
"""Scalefactor Huffman codewords (121 entries, index 0-120).

Index 60 corresponds to a scalefactor difference of 0 (the most common case,
encoded as a single '0' bit).
"""

SCALEFACTOR_BITS: Final[list[int]] = [
    18, 18, 18, 18, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
    19, 19, 19, 18, 19, 18, 17, 17, 16, 17, 16, 16, 16, 16, 15, 15,
    14, 14, 14, 14, 14, 14, 13, 13, 12, 12, 12, 11, 12, 11, 10, 10,
    10,  9,  9,  8,  8,  8,  7,  6,  6,  5,  4,  3,  1,  4,  4,  5,
     6,  6,  7,  7,  8,  8,  9,  9, 10, 10, 10, 11, 11, 11, 11, 12,
    12, 13, 13, 13, 14, 14, 16, 15, 16, 15, 18, 19, 19, 19, 19, 19,
    19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
    19, 19, 19, 19, 19, 19, 19, 19, 19,
]
"""Scalefactor Huffman bit lengths (121 entries)."""

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

# Indexing parameters for each codebook.
# For signed codebooks: offset = max_abs, base = 2*max_abs+1
# For unsigned codebooks: offset = 0, base = max_abs+1
_CODEBOOK_OFFSET: Final[list[int]] = [
    0,  # 0 -- placeholder
    1,  # 1: signed, values {-1,0,1}, offset=1
    1,  # 2: signed, values {-1,0,1}, offset=1
    0,  # 3: unsigned, values {0,1,2}, offset=0
    0,  # 4: unsigned, values {0,1,2}, offset=0
    4,  # 5: signed, values {-4..4}, offset=4
    4,  # 6: signed, values {-4..4}, offset=4
    0,  # 7: unsigned, values {0..7}, offset=0
    0,  # 8: unsigned, values {0..7}, offset=0
    0,  # 9: unsigned, values {0..12}, offset=0
    0,  # 10: unsigned, values {0..12}, offset=0
    0,  # 11: unsigned, values {0..16}, offset=0
]

_CODEBOOK_BASE: Final[list[int]] = [
    0,   # 0 -- placeholder
    3,   # 1: 2*1+1
    3,   # 2: 2*1+1
    3,   # 3: 2+1
    3,   # 4: 2+1
    9,   # 5: 2*4+1
    9,   # 6: 2*4+1
    8,   # 7: 7+1
    8,   # 8: 7+1
    13,  # 9: 12+1
    13,  # 10: 12+1
    17,  # 11: 16+1
]

# ============================================================================
# Master code/bits lists indexed by codebook number (1-based)
# ============================================================================

_ALL_CODES: Final[list[list[int] | None]] = [
    None,       # 0
    CODES_1,    # 1
    CODES_2,    # 2
    CODES_3,    # 3
    CODES_4,    # 4
    CODES_5,    # 5
    CODES_6,    # 6
    CODES_7,    # 7
    CODES_8,    # 8
    CODES_9,    # 9
    CODES_10,   # 10
    CODES_11,   # 11
]

_ALL_BITS: Final[list[list[int] | None]] = [
    None,      # 0
    BITS_1,    # 1
    BITS_2,    # 2
    BITS_3,    # 3
    BITS_4,    # 4
    BITS_5,    # 5
    BITS_6,    # 6
    BITS_7,    # 7
    BITS_8,    # 8
    BITS_9,    # 9
    BITS_10,   # 10
    BITS_11,   # 11
]

# ============================================================================
# Codebook dictionaries: tuple-of-values -> (codeword, bit_length)
# ============================================================================
# Built at import time from the raw arrays above.

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


def _index_to_values(book: int, index: int) -> tuple[int, ...]:
    """Convert a flat spectrum index to the tuple of values it represents.

    Args:
        book: Codebook number (1-11).
        index: Flat index into the codebook arrays.

    Returns:
        Tuple of spectral values encoded by this index.
    """
    dim = CODEBOOK_DIMENSION[book]
    base = _CODEBOOK_BASE[book]
    offset = _CODEBOOK_OFFSET[book]

    values: list[int] = []
    remaining = index
    for _ in range(dim):
        values.append(remaining % base - offset)
        remaining //= base
    # Verified against FFmpeg's ff_aac_codebook_vector_idx table:
    # Index 67 → codebook_vector02_idx[67] = 0xf456 → positions (2,1,1,1)
    # → values (+1, 0, 0, 0) where coeff[0]=+1.
    # Mathematically: 67 = 2*27 + 1*9 + 1*3 + 1
    # So coeff[0] = (67/27)%3-1 = 2-1 = 1 (MSB-first ordering)
    # Our extraction gives [LSB, ..., MSB], so reverse for coeff[0..dim-1]:
    values.reverse()
    return tuple(values)


def _build_codebook_dicts() -> None:
    """Populate all HCB_* dictionaries from the raw code/bits arrays."""
    hcb_list = [None, HCB_1, HCB_2, HCB_3, HCB_4, HCB_5, HCB_6,
                HCB_7, HCB_8, HCB_9, HCB_10, HCB_11]

    for book in range(1, 12):
        codes = _ALL_CODES[book]
        bits = _ALL_BITS[book]
        hcb = hcb_list[book]
        if codes is None or bits is None or hcb is None:
            continue
        num_entries = CODEBOOK_NUM_ENTRIES[book]
        for idx in range(num_entries):
            values = _index_to_values(book, idx)
            hcb[values] = (codes[idx], bits[idx])


_build_codebook_dicts()


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
            or the entry is not found in the codebook.
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
            f"Values {values} not found in codebook {book}."
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
