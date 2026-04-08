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

SF_OFFSET = 100
"""Scalefactor offset constant from ISO 14496-3."""


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

    # Approximate Huffman cost per coefficient
    # Zero: ~1 bit, small values: ~3-5 bits, larger: ~log2(val)+2
    cost = torch.where(
        abs_q < 0.5,
        torch.ones_like(abs_q),  # zero cost ≈ 1 bit
        torch.log2(abs_q + 1.0) + 2.0,  # non-zero cost
    )

    return cost.sum(dim=-1)
