"""GPU-parallel rate control via binary search on global gain.

Rate control finds the optimal global_gain per frame. The search balances
two constraints:
1. Bit budget: estimated bits must fit within the target frame size
2. Value range: quantized values must stay within ±4095 (escape code limit)

The search finds the LOWEST gain (best quality) satisfying both constraints.
"""

from __future__ import annotations

import torch

from torch_aac.config import QuantMode
from torch_aac.gpu.quantizer import estimate_bit_count, quantize

# Search parameters
MIN_GAIN = 0
MAX_GAIN = 255
MAX_ITERATIONS = 12
MAX_QUANTIZED_VALUE = 4095  # AAC escape code limit


def find_global_gain(
    mdct_coeffs: torch.Tensor,
    target_bits: torch.Tensor | float,
    quant_mode: QuantMode = QuantMode.HARD,
    min_gain: int = MIN_GAIN,
    max_gain: int = MAX_GAIN,
    max_iterations: int = MAX_ITERATIONS,
) -> torch.Tensor:
    """Find optimal global_gain per frame via parallel binary search.

    Finds the lowest gain (finest quantization, best quality) where:
    - Estimated bit count ≤ target_bits
    - Max quantized value ≤ MAX_QUANTIZED_VALUE (4095)

    Args:
        mdct_coeffs: MDCT coefficients, shape ``(B, C, 1024)`` or ``(B, 1024)``.
        target_bits: Target bits per frame.
        quant_mode: Quantization mode.
        min_gain: Minimum gain search bound.
        max_gain: Maximum gain search bound.
        max_iterations: Maximum binary search iterations.

    Returns:
        Optimal global_gain per frame, shape ``(B,)``, dtype int64.
    """
    B = mdct_coeffs.shape[0]
    device = mdct_coeffs.device

    if isinstance(target_bits, (int, float)):
        target_bits = torch.full((B,), target_bits, device=device, dtype=torch.float32)
    elif target_bits.dim() == 0:
        target_bits = target_bits.expand(B)

    lo = torch.full((B,), min_gain, device=device, dtype=torch.float32)
    hi = torch.full((B,), max_gain, device=device, dtype=torch.float32)

    for _ in range(max_iterations):
        mid = torch.floor((lo + hi) / 2.0)

        q = quantize(mdct_coeffs, mid, mode=quant_mode)

        # Check bit budget
        bits = estimate_bit_count(q)
        if bits.dim() > 1:
            bits = bits.sum(dim=-1)

        # Check value range (max absolute quantized value per frame)
        if q.dim() == 3:
            max_q = q.abs().reshape(B, -1).max(dim=-1).values
        else:
            max_q = q.abs().max(dim=-1).values

        # A gain is too low if: too many bits OR values overflow
        too_low = (bits > target_bits) | (max_q > MAX_QUANTIZED_VALUE)
        lo = torch.where(too_low, mid + 1.0, lo)
        hi = torch.where(~too_low, mid, hi)

    return lo.to(torch.int64)
