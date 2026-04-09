"""GPU-accelerated non-uniform quantizer for AAC-LC.

AAC uses a non-uniform quantizer defined by:
    q = nint(|x|^(3/4) / quantizer_step)
    quantizer_step = 2^(0.25 * (global_gain - SF_OFFSET))

where SF_OFFSET is a constant (typically 100 in the spec).

This module supports three quantization modes:
- HARD: Standard rounding (no gradient).
- STE: Straight-through estimator (gradient passes through round).
- NOISE: Additive uniform noise approximation (smooth gradient).
"""

from __future__ import annotations

import torch

from torch_aac.config import QuantMode

SF_OFFSET = 140
"""Scalefactor offset (SCALE_ONE_POS) from FFmpeg/ISO 14496-3.

This is the scalefactor index where the reconstructed scale = 1.0:
    step = 2^(0.25 * (global_gain - SF_OFFSET))

At global_gain = 140 (SCALE_ONE_POS), the encoder's quantizer step
matches the decoder's dequantization scale of 1.0.

Note: FFmpeg defines POW_SF2_ZERO=200 (index in pow2sf_tab where
pow(2,0)=1.0) and SCALE_ONE_POS=140 (scalefactor for unity scale).
The decoder computes: scale = pow2sf_tab[sf + 200 - 140] = pow2sf_tab[sf + 60].
"""


def compute_quantizer_step(global_gain: torch.Tensor) -> torch.Tensor:
    """Compute the quantizer step size from global gain.

    Args:
        global_gain: Integer gain values, shape ``(B,)`` or scalar.

    Returns:
        Step sizes, same shape as input.
    """
    return torch.pow(2.0, 0.25 * (global_gain.float() - SF_OFFSET))


def quantize(
    mdct_coeffs: torch.Tensor,
    global_gain: torch.Tensor,
    mode: QuantMode = QuantMode.HARD,
) -> torch.Tensor:
    """Quantize MDCT coefficients using the AAC non-uniform quantizer.

    q = round(sign(x) * |x|^(3/4) / step)

    Args:
        mdct_coeffs: MDCT coefficients, shape ``(B, C, 1024)`` or ``(B, 1024)``.
        global_gain: Per-frame global gain, shape ``(B,)`` or ``(B, 1)`` or ``(B, C)``.
        mode: Quantization mode (HARD, STE, or NOISE).

    Returns:
        Quantized (integer) coefficients, same shape as mdct_coeffs.
    """
    step = compute_quantizer_step(global_gain)

    # Broadcast step to match coefficients shape
    while step.dim() < mdct_coeffs.dim():
        step = step.unsqueeze(-1)

    signs = torch.sign(mdct_coeffs)
    abs_coeffs = torch.abs(mdct_coeffs)

    # AAC non-uniform compression: |x|^(3/4)
    # Add small epsilon to avoid gradient issues at zero
    compressed = torch.pow(abs_coeffs + 1e-10, 0.75)
    scaled = compressed / step

    if mode == QuantMode.HARD:
        quantized = torch.round(scaled)
    elif mode == QuantMode.STE:
        # Straight-through estimator: round in forward, identity in backward
        quantized = scaled + (torch.round(scaled) - scaled).detach()
    elif mode == QuantMode.NOISE:
        # Additive uniform noise approximation
        if mdct_coeffs.requires_grad:
            noise = torch.rand_like(scaled) - 0.5
            quantized = scaled + noise
        else:
            quantized = torch.round(scaled)
    else:
        raise ValueError(f"Unknown quantization mode: {mode}")

    return signs * quantized


def dequantize(
    quantized: torch.Tensor,
    global_gain: torch.Tensor,
) -> torch.Tensor:
    """Inverse quantize: reconstruct approximate MDCT coefficients.

    x_hat = sign(q) * |q|^(4/3) * step

    Args:
        quantized: Quantized coefficients.
        global_gain: Per-frame global gain.

    Returns:
        Reconstructed MDCT coefficients, same shape as quantized.
    """
    step = compute_quantizer_step(global_gain)
    while step.dim() < quantized.dim():
        step = step.unsqueeze(-1)

    signs = torch.sign(quantized)
    abs_q = torch.abs(quantized)

    # Inverse of |x|^(3/4): |q|^(4/3)
    decompressed = torch.pow(abs_q + 1e-10, 4.0 / 3.0)

    return signs * decompressed * step


def quantize_per_band(
    mdct_coeffs: torch.Tensor,
    scalefactors: torch.Tensor,
    sfb_offsets: list[int],
    mode: QuantMode = QuantMode.HARD,
) -> torch.Tensor:
    """Quantize MDCT coefficients using per-band scalefactors.

    Each scalefactor band uses its own quantizer step:
        step[band] = 2^(0.25 * (sf[band] - SF_OFFSET))

    Args:
        mdct_coeffs: Shape ``(B, C, 1024)`` or ``(B, 1024)``.
        scalefactors: Per-band scalefactors, shape ``(B, num_sfb)`` or ``(B, C, num_sfb)``.
        sfb_offsets: Cumulative SFB offsets.
        mode: Quantization mode.

    Returns:
        Quantized coefficients, same shape as mdct_coeffs.
    """
    device = mdct_coeffs.device
    result = torch.zeros_like(mdct_coeffs)
    num_sfb = len(sfb_offsets) - 1

    # Flatten for processing
    flat_coeffs = mdct_coeffs.reshape(-1, 1024)
    flat_result = result.reshape(-1, 1024)
    flat_sf = scalefactors.reshape(-1, num_sfb)
    N = flat_coeffs.shape[0]

    for i in range(num_sfb):
        s, e = sfb_offsets[i], sfb_offsets[i + 1]
        if e > 1024:
            break
        band_sf = flat_sf[:, i]  # (N,)
        step = torch.pow(2.0, 0.25 * (band_sf.float() - SF_OFFSET)).unsqueeze(-1)  # (N, 1)

        band_coeffs = flat_coeffs[:, s:e]  # (N, band_width)
        signs = torch.sign(band_coeffs)
        abs_coeffs = torch.abs(band_coeffs)
        compressed = torch.pow(abs_coeffs + 1e-10, 0.75)
        scaled = compressed / step

        if mode == QuantMode.HARD:
            quantized = torch.round(scaled)
        elif mode == QuantMode.STE:
            quantized = scaled + (torch.round(scaled) - scaled).detach()
        elif mode == QuantMode.NOISE:
            if mdct_coeffs.requires_grad:
                quantized = scaled + torch.rand_like(scaled) - 0.5
            else:
                quantized = torch.round(scaled)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        flat_result[:, s:e] = signs * quantized

    return flat_result.reshape_as(mdct_coeffs)


def estimate_bit_count(
    quantized: torch.Tensor,
) -> torch.Tensor:
    """Estimate the number of bits needed to Huffman-encode quantized coefficients.

    Uses a simple approximation: each non-zero coefficient costs approximately
    log2(|q| + 1) + 1 bits (sign bit). Zero coefficients cost ~1 bit in most
    codebooks (the zero codeword is short).

    This is used by the GPU rate control loop for fast bit estimation without
    running actual Huffman encoding.

    Args:
        quantized: Quantized coefficients, shape ``(..., 1024)``.

    Returns:
        Estimated bit count per frame, shape ``(...,)`` — all dims except last.
    """
    abs_q = torch.abs(quantized)

    # Approximate Huffman cost per coefficient.
    # This drives the rate control binary search. Should be reasonably
    # accurate to avoid under/over-quantization.
    #
    # Cost model based on typical AAC Huffman code lengths:
    # - Zero: ~0.25 bits (most bands are zero, section overhead is small)
    # - Small (1-4): ~4-6 bits (Huffman code + sign)
    # - Medium (5-16): ~6-10 bits
    # - Large (>16): escape codes ~2*log2(val) + 8 bits
    log_q = torch.log2(abs_q + 1.0)
    cost = torch.where(
        abs_q < 0.5,
        torch.full_like(abs_q, 0.25),
        torch.where(
            abs_q <= 16.0,
            log_q + 4.0,
            log_q * 2.0 + 8.0,
        ),
    )

    return cost.sum(dim=-1)
