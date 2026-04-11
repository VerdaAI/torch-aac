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
    original_shape = mdct_coeffs.shape[:-1]

    # Without a psychoacoustic model, uniform per-band scalefactor (= global_gain)
    # is better than trying to equalize q across bands. The previous approach
    # (sf chosen so q≈30 per band) forced every band — including pure noise —
    # into the escape codebook (cb11), which is bit-wasteful and degrades SNR.
    # With a uniform sf, noise-floor bands naturally quantize to q≈0 and the
    # codebook selector picks small books for them.
    gg = global_gain.to(torch.int64).reshape(-1)
    # Expand to match flattened frame count (e.g. stereo: B*C frames from B gains)
    flat_count = int(torch.tensor(original_shape).prod().item())
    if flat_count > gg.shape[0]:
        gg = gg.repeat_interleave(flat_count // gg.shape[0])
    scalefactors = gg.unsqueeze(-1).expand(flat_count, num_sfb).contiguous()

    return scalefactors.reshape(*original_shape, num_sfb)
