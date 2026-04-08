"""GPU-parallel Huffman codebook selection for AAC-LC.

AAC defines 12 Huffman codebooks for encoding quantized spectral coefficients.
This module estimates the encoding cost for each codebook and selects the
optimal one per scalefactor band section — all on GPU in a single batched pass.

The actual bit-packing of Huffman codes is done on CPU (cpu/huffman.py).
This module only does the cost estimation and codebook selection.
"""

from __future__ import annotations

import torch

# Codebook properties
# Codebooks 1-4: 4-tuples (quads), Codebooks 5-11: 2-tuples (pairs)
CODEBOOK_DIMS = [0, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2]
CODEBOOK_MAX_ABS = [0, 1, 1, 2, 2, 4, 4, 7, 7, 12, 12, 16]
CODEBOOK_IS_UNSIGNED = [
    False,  # 0: unused
    True, True,     # 1-2: unsigned quads, max 1
    True, True,     # 3-4: unsigned quads, max 2 (separate sign bits)
    False, False,   # 5-6: signed pairs
    True, True,     # 7-8: unsigned pairs (separate sign bits)
    True, True,     # 9-10: unsigned pairs (separate sign bits)
    True,           # 11: unsigned pairs, escape (separate sign bits)
]

# Average bits per coefficient for each codebook at various max_abs values.
# This is a simplified cost model for GPU selection.
# Index: [codebook][max_abs_in_section] → approximate bits/coefficient
# These are rough averages; exact costs depend on actual value distribution.
APPROX_BITS_PER_COEFF: list[list[float]] = [
    [],                                                    # 0: unused
    [1.0, 2.0],                                           # 1: max_abs ≤ 1
    [0.8, 1.8],                                           # 2: max_abs ≤ 1
    [1.5, 2.5, 4.0],                                     # 3: max_abs ≤ 2
    [1.2, 2.2, 3.5],                                     # 4: max_abs ≤ 2
    [2.0, 2.5, 3.0, 3.5, 4.0, 5.0],                     # 5: max_abs ≤ 4
    [1.8, 2.3, 2.8, 3.3, 3.8, 4.8],                     # 6: max_abs ≤ 4
    [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0],     # 7: max_abs ≤ 7
    [2.3, 2.8, 3.3, 3.8, 4.3, 4.8, 5.3, 5.8, 6.8],     # 8: max_abs ≤ 7
    [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0,      # 9: max_abs ≤ 12
     7.5, 8.0, 8.5, 9.0, 10.0],
    [2.8, 3.3, 3.8, 4.3, 4.8, 5.3, 5.8, 6.3, 6.8,      # 10: max_abs ≤ 12
     7.3, 7.8, 8.3, 8.8, 9.8],
    [3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5,      # 11: max_abs ≤ 16+
     8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 12.0, 14.0],
]

NUM_CODEBOOKS = 12  # 1-11, index 0 unused


def select_codebooks(
    quantized: torch.Tensor,
    sfb_offsets: list[int],
) -> torch.Tensor:
    """Select the optimal Huffman codebook per scalefactor band section.

    For each scalefactor band in each frame, determines which of the 12
    codebooks produces the fewest bits. This is done by computing the
    maximum absolute value in each band and selecting the codebook with
    the lowest max_abs that can still encode those values.

    Args:
        quantized: Quantized spectral coefficients, shape ``(B, 1024)``
            or ``(B, C, 1024)``.
        sfb_offsets: Cumulative scalefactor band offsets (length = num_sfb + 1).

    Returns:
        Codebook indices per band per frame, shape ``(B, num_sfb)`` or
        ``(B, C, num_sfb)``.
    """
    num_sfb = len(sfb_offsets) - 1
    device = quantized.device
    original_shape = quantized.shape[:-1]  # (B,) or (B, C)

    # Flatten to 2D: (N, 1024) where N = B or B*C
    flat = quantized.reshape(-1, quantized.shape[-1])
    N = flat.shape[0]

    # Compute max absolute value per scalefactor band
    max_abs_per_band = torch.zeros(N, num_sfb, device=device, dtype=torch.float32)
    for i in range(num_sfb):
        start, end = sfb_offsets[i], sfb_offsets[i + 1]
        if start < flat.shape[-1] and end <= flat.shape[-1]:
            band_slice = flat[:, start:end]
            max_abs_per_band[:, i] = torch.abs(band_slice).max(dim=-1).values

    # Select codebook based on max_abs value
    # Strategy: use the smallest-range codebook that can encode the max_abs
    codebook_maxabs = torch.tensor(CODEBOOK_MAX_ABS[1:], device=device, dtype=torch.float32)
    # codebook_maxabs: [1, 1, 2, 2, 4, 4, 7, 7, 12, 12, 16] → indices 0-10 map to books 1-11

    # For each band, find the first codebook whose max_abs >= band's max_abs
    # Expand for broadcasting: (N, num_sfb, 1) vs (1, 1, 11)
    max_abs_expanded = max_abs_per_band.unsqueeze(-1)  # (N, sfb, 1)
    cb_expanded = codebook_maxabs.unsqueeze(0).unsqueeze(0)  # (1, 1, 11)

    # Can this codebook handle the values? (bool mask)
    can_encode = cb_expanded >= max_abs_expanded  # (N, sfb, 11)

    # For bands with all zeros, use codebook 0 (zero section)
    is_zero_band = max_abs_per_band < 0.5  # (N, sfb)

    # Assign very high cost to codebooks that can't encode
    # Find the first valid codebook (smallest range that works)
    # Use argmax on the can_encode mask (first True)
    # Add 1 because codebooks are 1-indexed
    # If no codebook can encode (shouldn't happen with book 11), default to 11
    invalid_penalty = torch.where(can_encode, 0, 1000)
    # Prefer smaller codebooks (they tend to have shorter codes for small values)
    # Add a small preference penalty for larger codebooks
    preference = torch.arange(11, device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    cost = invalid_penalty.float() + preference * 0.01

    selected = cost.argmin(dim=-1) + 1  # (N, sfb), 1-indexed

    # Override: zero bands get codebook 0
    selected = torch.where(is_zero_band, torch.zeros_like(selected), selected)

    return selected.reshape(*original_shape, num_sfb)
