"""GPU-parallel rate control and per-band scalefactor computation.

Finds the optimal global_gain and per-band scalefactors for each frame.
The global_gain sets the baseline quantization, and per-band scalefactors
adjust it so each band gets the right level of quantization.
"""

from __future__ import annotations

import torch

from torch_aac.config import QuantMode
from torch_aac.gpu.quantizer import SF_OFFSET, estimate_bit_count, quantize

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
    - Estimated bit count <= target_bits
    - Max quantized value <= MAX_QUANTIZED_VALUE (4095)

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

        bits = estimate_bit_count(q)
        if bits.dim() > 1:
            bits = bits.sum(dim=-1)

        if q.dim() == 3:
            max_q = q.abs().reshape(B, -1).max(dim=-1).values
        else:
            max_q = q.abs().max(dim=-1).values

        too_low = (bits > target_bits) | (max_q > MAX_QUANTIZED_VALUE)
        lo = torch.where(too_low, mid + 1.0, lo)
        hi = torch.where(~too_low, mid, hi)

    return lo.to(torch.int64)


def compute_scalefactors(
    mdct_coeffs: torch.Tensor,
    global_gain: torch.Tensor,
    sfb_offsets: list[int],
) -> torch.Tensor:
    """Compute per-band scalefactors for optimal reconstruction.

    Each band's scalefactor is set so that the quantizer step produces
    the best reconstruction for that band's energy level. The scalefactor
    is relative to global_gain: sf[band] = global_gain + delta[band].

    The optimal scalefactor for a band makes step ≈ 1.0 for that band's
    peak coefficient: sf = SF_OFFSET + 4 * log2(max_coeff^0.75) ≈
    SF_OFFSET for coefficients near 1.0.

    Args:
        mdct_coeffs: MDCT coefficients, shape ``(B, C, 1024)`` or ``(B, 1024)``.
        global_gain: Per-frame global gain, shape ``(B,)``.
        sfb_offsets: Cumulative SFB offsets.

    Returns:
        Per-band scalefactors, shape ``(B, num_sfb)`` or ``(B, C, num_sfb)``.
        Values are absolute scalefactors (not deltas).
    """
    num_sfb = len(sfb_offsets) - 1
    device = mdct_coeffs.device

    # Flatten to (N, 1024) for processing
    original_shape = mdct_coeffs.shape[:-1]
    flat = mdct_coeffs.reshape(-1, mdct_coeffs.shape[-1])
    N = flat.shape[0]

    # Compute max absolute value per band
    max_abs = torch.zeros(N, num_sfb, device=device)
    for i in range(num_sfb):
        s, e = sfb_offsets[i], sfb_offsets[i + 1]
        if e <= flat.shape[-1]:
            max_abs[:, i] = flat[:, s:e].abs().max(dim=-1).values

    # Optimal scalefactor per band: makes quantizer step = max_coeff^0.75 / target_q
    # We want quantized values in range [0, ~50] for best reconstruction.
    # q = |x|^0.75 / step, step = 2^(0.25*(sf - SF_OFFSET))
    # For q ≈ target_q: step = |x|^0.75 / target_q
    # sf = SF_OFFSET + 4 * log2(|x|^0.75 / target_q)
    # sf = SF_OFFSET + 3 * log2(|x|) - 4 * log2(target_q)
    target_q = 30.0  # target max quantized value per band
    log2_max = torch.log2(max_abs.clamp(min=1e-10))
    optimal_sf = SF_OFFSET + 3.0 * log2_max - 4.0 * torch.log2(torch.tensor(target_q, device=device))

    scalefactors = optimal_sf.clamp(0, 255).round().to(torch.int64)

    # For zero bands, use global_gain
    # Expand global_gain to (N, 1) to broadcast with (N, num_sfb)
    gg_flat = global_gain.float()
    # If mdct_coeffs was (B, C, 1024), N = B*C but global_gain is (B,)
    # Repeat each gain C times to match flattened shape
    if N > gg_flat.shape[0]:
        repeat_factor = N // gg_flat.shape[0]
        gg_flat = gg_flat.repeat_interleave(repeat_factor)
    gg_expanded = gg_flat.unsqueeze(-1).expand_as(scalefactors).to(torch.int64)

    zero_bands = max_abs < 1e-10
    scalefactors = torch.where(zero_bands, gg_expanded, scalefactors)

    return scalefactors.reshape(*original_shape, num_sfb)
