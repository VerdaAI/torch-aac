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

SF_OFFSET = 164
"""Scalefactor "unity point" — empirically verified against FFmpeg's decoder.

Determined by feeding trivial bitstreams (q=1 at a known bin, fixed sf) to
FFmpeg and observing output: sf=200 produces unity peak amplitude. The
relationship is:
    decoder_output_per_coef = |q|^(4/3) * 2^((sf - 200) / 4)

The AAC encoder quantizes as:
    q = nint(|x|^(3/4) / pow34sf(sf))
where pow34sf(sf) = 2^(3/16 * (sf - 200))

The AAC decoder dequantizes as:
    x_hat = |q|^(4/3) * pow2sf(sf)
where pow2sf(sf) = 2^(1/4 * (sf - 200))

For lossless roundtrip, step_dec = step_enc^(4/3):
    (2^(3/16*(sf-200)))^(4/3) = 2^(1/4*(sf-200)) ✓
"""


def compute_quantizer_step(global_gain: torch.Tensor) -> torch.Tensor:
    """Compute the ENCODER quantizer step from scalefactor.

    Uses pow34sf: step_enc = 2^(3/16 * (sf - 100))

    The AAC decoder reconstructs: x_hat = |q|^(4/3) * step_dec
    where step_dec = 2^(1/4 * (sf - 100)).

    For lossless roundtrip, we need step_dec = step_enc^(4/3):
        step_enc^(4/3) = 2^(3/16 * (sf-100) * 4/3) = 2^(1/4 * (sf-100)) = step_dec ✓

    Args:
        global_gain: Scalefactor values, shape ``(B,)`` or scalar.

    Returns:
        Encoder step sizes, same shape as input.
    """
    return torch.pow(2.0, (3.0 / 16.0) * (global_gain.float() - SF_OFFSET))


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


def compute_decoder_scale(scalefactor: torch.Tensor) -> torch.Tensor:
    """Compute the decoder dequantization scale (pow2sf).

    pow2sf(sf) = 2^(1/4 * (sf - 100))

    This is what the DECODER uses for reconstruction.
    """
    return torch.pow(2.0, 0.25 * (scalefactor.float() - SF_OFFSET))


def dequantize(
    quantized: torch.Tensor,
    global_gain: torch.Tensor,
) -> torch.Tensor:
    """Inverse quantize: reconstruct approximate MDCT coefficients.

    Uses the DECODER formula: x_hat = sign(q) * |q|^(4/3) * pow2sf(sf)
    where pow2sf(sf) = 2^(1/4 * (sf - 100))

    Args:
        quantized: Quantized coefficients.
        global_gain: Per-frame global gain (scalefactor).

    Returns:
        Reconstructed MDCT coefficients, same shape as quantized.
    """
    scale = compute_decoder_scale(global_gain)
    while scale.dim() < quantized.dim():
        scale = scale.unsqueeze(-1)

    signs = torch.sign(quantized)
    abs_q = torch.abs(quantized)
    decompressed = torch.pow(abs_q + 1e-10, 4.0 / 3.0)

    return signs * decompressed * scale


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
    result = torch.zeros_like(mdct_coeffs)
    num_sfb = len(sfb_offsets) - 1

    # Flatten for processing
    flat_coeffs = mdct_coeffs.reshape(-1, 1024)
    flat_result = result.reshape(-1, 1024)
    flat_sf = scalefactors.reshape(-1, num_sfb)

    for i in range(num_sfb):
        s, e = sfb_offsets[i], sfb_offsets[i + 1]
        if e > 1024:
            break
        band_sf = flat_sf[:, i]  # (N,)
        # Encoder step uses pow34sf = 2^(3/16 * (sf - 100)) for correct
        # roundtrip through FFmpeg's pow2sf-based decoder.
        step = torch.pow(2.0, (3.0 / 16.0) * (band_sf.float() - SF_OFFSET)).unsqueeze(-1)

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


_BITS_PER_COEF_TABLE: torch.Tensor | None = None


def _build_bits_table(device: torch.device) -> torch.Tensor:
    """Build a lookup table: bits_per_coefficient[|q|] for q in [0, 8192].

    Derived from actual AAC Huffman code lengths for pair codebooks (cb5-11).
    For each |q|, computes the cost of encoding the pair (±q, 0) in the
    optimal codebook, then divides by 2 for per-coefficient cost.

    This replaces the rough log-based formula that overestimated by 1.2-2.5x,
    causing rate control to waste 30-50% of the bit budget.
    """
    from torch_aac.tables.huffman_tables import CODEBOOK_MAX_ABS as CMA
    from torch_aac.tables.huffman_tables import CODEBOOK_UNSIGNED as CU
    from torch_aac.tables.huffman_tables import CODEBOOKS as CBS

    table = torch.zeros(8193, dtype=torch.float32, device=device)

    # Available pair codebooks in pairs_only mode: 5, 7, 9, 11
    pair_cbs = [(5, CMA[5]), (7, CMA[7]), (9, CMA[9]), (11, CMA[11])]

    for q_abs in range(8193):
        # Find optimal codebook
        best_cb = 11
        for cb_num, ma in pair_cbs:
            if q_abs <= ma:
                best_cb = cb_num
                break

        cb_dict = CBS[best_cb]
        is_unsigned = CU[best_cb]

        if is_unsigned:
            lookup_q = min(q_abs, 16 if best_cb == 11 else CMA[best_cb])
            entry = cb_dict.get((lookup_q, 0))  # type: ignore[union-attr]
            code_bits = entry[1] if entry else 14
            sign_bits = 1 if q_abs > 0 else 0
            escape_bits = 0
            if best_cb == 11 and q_abs > 15:
                av = min(q_abs, 8191)
                n = max(0, min(av.bit_length() - 5, 8))
                escape_bits = (n + 1) + (n + 4)
            total = code_bits + sign_bits + escape_bits
        else:
            entry = cb_dict.get((q_abs, 0)) or cb_dict.get((-q_abs, 0))  # type: ignore[union-attr]
            total = entry[1] if entry else 14

        table[q_abs] = total / 2.0  # per-coefficient (pair covers 2 coefs)

    return table


def estimate_bit_count(
    quantized: torch.Tensor,
) -> torch.Tensor:
    """Estimate the number of bits to Huffman-encode quantized coefficients.

    Uses a precomputed lookup table derived from actual AAC Huffman code
    lengths. For each |q|, the table gives the expected bits per coefficient
    when encoding a pair (±q, 0) in the optimal codebook.

    Args:
        quantized: Quantized coefficients, shape ``(..., 1024)``.

    Returns:
        Estimated bit count per frame, shape ``(...,)`` — all dims except last.
    """
    global _BITS_PER_COEF_TABLE
    device = quantized.device
    if _BITS_PER_COEF_TABLE is None or _BITS_PER_COEF_TABLE.device != device:
        _BITS_PER_COEF_TABLE = _build_bits_table(device)

    abs_q = quantized.abs().long().clamp(0, 8192)
    cost = _BITS_PER_COEF_TABLE[abs_q]

    return cost.sum(dim=-1)
