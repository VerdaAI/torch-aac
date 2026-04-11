"""GPU-parallel rate control and per-band scalefactor computation.

Finds the optimal global_gain and per-band scalefactors for each frame.
The global_gain sets the baseline quantization, and per-band scalefactors
adjust it so each band gets the right level of quantization.
"""

from __future__ import annotations

import torch

from torch_aac.config import QuantMode
from torch_aac.gpu.psychoacoustic import compute_masking_thresholds
from torch_aac.gpu.quantizer import estimate_bit_count, quantize, quantize_per_band

# Search parameters
MIN_GAIN = 0
MAX_GAIN = 255
MAX_ITERATIONS = 12
MAX_QUANTIZED_VALUE = 4095  # AAC escape code limit

# Maximum per-band sf shift above the global gain (how much coarser a masked
# band can be). Capped to avoid extreme shifts that overflow the 8-bit sf
# field or blow through the ±60 delta encoding range between adjacent bands.
MAX_MASK_SHIFT = 40


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


def compute_mask_shifts(
    mdct_coeffs: torch.Tensor,
    sample_rate: int,
    sfb_offsets: list[int],
) -> torch.Tensor:
    """Compute per-band sf offsets (in sf units) above a reference, driven by
    the psychoacoustic masking threshold.

    For each band we derive ``sf_max``, the coarsest scalefactor that keeps
    the quantization noise power at or below the masking threshold:

        sf_max = 164 + 2*log2(12 * mask / band_width)

    (derived from ``noise_power = width * step² / 12`` and
    ``step = 2^((sf-164)/4)``.)

    The returned shifts are ``sf_max[i] - min(sf_max)``, so the tightest-
    headroom band has shift=0 and drives rate control, while all other
    bands are coarser by an integer offset. This keeps the budget fully
    spent on the perceptually important band.

    Args:
        mdct_coeffs: MDCT coefficients, shape ``(..., 1024)``.
        sample_rate: Sample rate in Hz (for the psychoacoustic model).
        sfb_offsets: Cumulative SFB offsets.

    Returns:
        Per-band shifts, shape ``(..., num_sfb)``, dtype int64, in [0, MAX_MASK_SHIFT].
    """
    num_sfb = len(sfb_offsets) - 1
    mask = compute_masking_thresholds(mdct_coeffs, sample_rate, sfb_offsets)
    # (..., num_sfb)

    widths = torch.tensor(
        [max(1, sfb_offsets[i + 1] - sfb_offsets[i]) for i in range(num_sfb)],
        device=mdct_coeffs.device,
        dtype=torch.float32,
    )

    # sf_max such that noise power ≈ masking threshold.
    sf_max = 164.0 + 2.0 * torch.log2((12.0 * mask / widths) + 1e-20)
    sf_max = sf_max.clamp(0.0, 255.0)

    # Normalize to the tightest-headroom band (the one that drives rate control).
    shift = sf_max - sf_max.min(dim=-1, keepdim=True).values
    shift = shift.clamp(0.0, float(MAX_MASK_SHIFT))

    return shift.round().to(torch.int64)


def find_scalefactors(
    mdct_coeffs: torch.Tensor,
    target_bits: torch.Tensor | float,
    sfb_offsets: list[int],
    sample_rate: int,
    quant_mode: QuantMode = QuantMode.HARD,
    min_gain: int = MIN_GAIN,
    max_gain: int = MAX_GAIN,
    max_iterations: int = MAX_ITERATIONS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Joint rate control + psychoacoustic per-band scalefactor selection.

    **Experimental.** Computes mask-driven per-band shifts, then runs parallel
    binary search for the global baseline gain ``G`` such that quantizing with
    ``sf[band] = G + shift[band]`` fits the target bit budget.

    Currently the encoder uses ``find_global_gain`` + ``compute_scalefactors``
    (uniform sf) because the uniform path accidentally does the right thing
    for tonal signals: the rate control binary search over-quantizes, which
    zeros out noise-floor bands and gives 60-73 dB SNR on sines/chords at
    128 kbps. The mask-shift variant produces lower SNR on tonal signals at
    low bitrate (~34 dB @ 48 kbps) because it keeps noise-floor bands partially
    reconstructed. Proper tonality detection + PNS is needed for this path
    to beat uniform across all signal types — tracked in the Phase 3 plan.

    Args:
        mdct_coeffs: MDCT coefficients, shape ``(B, C, 1024)`` or ``(B, 1024)``.
        target_bits: Target bits per frame.
        sfb_offsets: Cumulative SFB offsets.
        sample_rate: Sample rate in Hz for the psychoacoustic model.
        quant_mode: Quantization mode.
        min_gain: Minimum gain search bound.
        max_gain: Maximum gain search bound.
        max_iterations: Maximum binary search iterations.

    Returns:
        Tuple of (global_gain, scalefactors).
            - global_gain: shape ``(B,)``, dtype int64. The baseline gain
              for the tightest-headroom band per frame.
            - scalefactors: shape ``(B, num_sfb)`` or ``(B, C, num_sfb)``,
              dtype int64. Absolute per-band sf values in [0, 255].
    """
    B = mdct_coeffs.shape[0]
    device = mdct_coeffs.device

    if isinstance(target_bits, (int, float)):
        target_bits = torch.full((B,), target_bits, device=device, dtype=torch.float32)
    elif target_bits.dim() == 0:
        target_bits = target_bits.expand(B)

    # Compute per-band psychoacoustic shifts once — they only depend on the
    # MDCT coefficients, not on the gain we're searching for.
    shifts = compute_mask_shifts(mdct_coeffs, sample_rate, sfb_offsets)  # (..., num_sfb)

    lo = torch.full((B,), min_gain, device=device, dtype=torch.float32)
    hi = torch.full((B,), max_gain, device=device, dtype=torch.float32)

    for _ in range(max_iterations):
        mid = torch.floor((lo + hi) / 2.0)  # (B,)
        # Build per-band sf for this trial gain: sf = mid + shift
        if shifts.dim() == 3:
            # Broadcast gain to (B, C, num_sfb). Shift is (B, C, num_sfb).
            trial_sf = mid.view(B, 1, 1) + shifts.float()
        else:
            trial_sf = mid.view(B, 1) + shifts.float()
        trial_sf = trial_sf.clamp(0, 255).to(torch.int64)

        q = quantize_per_band(mdct_coeffs, trial_sf, sfb_offsets, mode=quant_mode)

        bits = estimate_bit_count(q)
        if bits.dim() > 1:
            bits = bits.sum(dim=tuple(range(1, bits.dim())))

        if q.dim() == 3:
            max_q = q.abs().reshape(B, -1).max(dim=-1).values
        else:
            max_q = q.abs().max(dim=-1).values

        too_low = (bits > target_bits) | (max_q > MAX_QUANTIZED_VALUE)
        lo = torch.where(too_low, mid + 1.0, lo)
        hi = torch.where(~too_low, mid, hi)

    global_gain = lo.to(torch.int64)

    # Final scalefactors: gain + per-band shift, clamped.
    if shifts.dim() == 3:
        scalefactors = global_gain.view(B, 1, 1) + shifts
    else:
        scalefactors = global_gain.view(B, 1) + shifts
    scalefactors = scalefactors.clamp(0, 255)

    return global_gain, scalefactors


def compute_scalefactors(
    mdct_coeffs: torch.Tensor,
    global_gain: torch.Tensor,
    sfb_offsets: list[int],
) -> torch.Tensor:
    """Return uniform per-band scalefactors set to ``global_gain``.

    Kept for the differentiable mode and tests. The production encoding path
    uses :func:`find_scalefactors` instead, which combines rate control with
    psychoacoustic per-band allocation.
    """
    num_sfb = len(sfb_offsets) - 1
    original_shape = mdct_coeffs.shape[:-1]

    gg = global_gain.to(torch.int64).reshape(-1)
    flat_count = int(torch.tensor(original_shape).prod().item())
    if flat_count > gg.shape[0]:
        gg = gg.repeat_interleave(flat_count // gg.shape[0])
    scalefactors = gg.unsqueeze(-1).expand(flat_count, num_sfb).contiguous()

    return scalefactors.reshape(*original_shape, num_sfb)
