"""GPU-parallel rate control via binary search on global gain.

Rate control finds the global_gain value per frame such that the quantized
coefficients fit within the target bit budget. This implementation runs the
entire binary search on GPU for all frames simultaneously, avoiding GPU↔CPU
round-trips.
"""

from __future__ import annotations

import torch

from torch_aac.config import QuantMode
from torch_aac.gpu.quantizer import estimate_bit_count, quantize

# Default search parameters
# MIN_GAIN must be high enough that quantized values stay within escape code
# limits (max ~4095 for AAC). For typical audio (MDCT max ~500), gain ≥ 80
# keeps max_q under 4095. We use 60 as a safe minimum.
MIN_GAIN = 60
MAX_GAIN = 255
MAX_ITERATIONS = 12  # log2(255-60) ≈ 8, use 12 for safety


def find_global_gain(
    mdct_coeffs: torch.Tensor,
    target_bits: torch.Tensor | float,
    quant_mode: QuantMode = QuantMode.HARD,
    min_gain: int = MIN_GAIN,
    max_gain: int = MAX_GAIN,
    max_iterations: int = MAX_ITERATIONS,
) -> torch.Tensor:
    """Find optimal global_gain per frame via parallel binary search.

    For each frame, searches for the smallest global_gain that produces a
    quantized representation fitting within the target bit budget.

    Args:
        mdct_coeffs: MDCT coefficients, shape ``(B, C, 1024)`` or ``(B, 1024)``.
        target_bits: Target bits per frame. Can be a scalar or per-frame tensor
            of shape ``(B,)``.
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

    # Initialize binary search bounds for all frames
    lo = torch.full((B,), min_gain, device=device, dtype=torch.float32)
    hi = torch.full((B,), max_gain, device=device, dtype=torch.float32)

    for _ in range(max_iterations):
        mid = torch.floor((lo + hi) / 2.0)

        # Quantize all frames with current gain candidates
        q = quantize(mdct_coeffs, mid, mode=quant_mode)

        # Estimate bit counts — sum over channels if multi-channel
        bits = estimate_bit_count(q)
        if bits.dim() > 1:
            # Sum across channels: (B, C) → (B,)
            bits = bits.sum(dim=-1)

        # Update bounds: if too many bits, increase gain (coarser quantization)
        too_many = bits > target_bits
        lo = torch.where(too_many, mid + 1.0, lo)
        hi = torch.where(~too_many, mid, hi)

    return lo.to(torch.int64)
