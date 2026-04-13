"""Perceptual Noise Substitution (PNS) detection and energy computation.

PNS replaces noise-like scalefactor bands with band_type=13 in the AAC
bitstream. The decoder then generates white noise at the specified energy
level instead of trying to reconstruct exact coefficients. This dramatically
improves quality for broadband signals (speech, noise) at low bitrate.

Detection is GPU-batched: all bands across all frames are evaluated in
parallel.
"""

from __future__ import annotations

import torch

# Empirical correction added to sfo = 2*log2(E) for PNS noise energy.
# Compensates for the MDCT normalization mismatch between our unnormalized
# forward MDCT and FFmpeg's PNS noise generator. Set to 0 and sweep
# until rms_ratio ≈ 1.0 on white noise to calibrate.
PNS_ENERGY_CORRECTION: int = 0


def detect_noise_bands(
    mdct_coeffs: torch.Tensor,
    quantized: torch.Tensor,
    codebook_indices: torch.Tensor,
    sfb_offsets: list[int],
    max_abs_threshold: int = 3,
    density_threshold: float = 0.3,
) -> torch.Tensor:
    """Detect noise-like bands suitable for PNS.

    A band is classified as "noise-like" (suitable for PNS) when:
    1. It has a non-zero codebook (band is active).
    2. Its max quantized value is small (≤ ``max_abs_threshold``), meaning the
       quantizer couldn't resolve meaningful structure.
    3. It has spread energy (non-zero density ≥ ``density_threshold``) — not a
       single isolated coefficient but broadband noise.

    Args:
        mdct_coeffs: MDCT coefficients, shape ``(B, C, 1024)``.
        quantized: Quantized coefficients, shape ``(B, C, 1024)``.
        codebook_indices: Codebook per band, shape ``(B, C, num_sfb)``.
        sfb_offsets: Cumulative SFB offsets.
        max_abs_threshold: Maximum abs quantized value to qualify as noise.
        density_threshold: Minimum fraction of non-zero coefs in the band.

    Returns:
        Boolean mask, shape ``(B, C, num_sfb)``, True where PNS should be used.
    """
    device = mdct_coeffs.device
    num_sfb = len(sfb_offsets) - 1
    flat_m = mdct_coeffs.reshape(-1, 1024)
    flat_q = quantized.reshape(-1, 1024)
    flat_cb = codebook_indices.reshape(-1, num_sfb)
    N = flat_m.shape[0]

    # Per-band statistics
    band_energy = torch.zeros(N, num_sfb, device=device)
    band_max_q = torch.zeros(N, num_sfb, device=device)
    band_nnz = torch.zeros(N, num_sfb, device=device)
    band_width = torch.zeros(num_sfb, device=device)

    for i in range(num_sfb):
        s, e = sfb_offsets[i], sfb_offsets[i + 1]
        if e > 1024:
            break
        w = e - s
        band_width[i] = w
        band_energy[:, i] = (flat_m[:, s:e] ** 2).sum(dim=-1)
        band_max_q[:, i] = flat_q[:, s:e].abs().max(dim=-1).values
        band_nnz[:, i] = (flat_q[:, s:e] != 0).sum(dim=-1).float()

    # Conditions
    is_active = flat_cb > 0  # non-zero codebook
    is_low_q = band_max_q <= max_abs_threshold
    density = band_nnz / band_width.clamp(min=1).unsqueeze(0)
    is_spread = density >= density_threshold

    mask = is_active & is_low_q & is_spread

    return mask.reshape(codebook_indices.shape)


def compute_noise_energy_sf(
    mdct_coeffs: torch.Tensor,
    sfb_offsets: list[int],
    noise_mask: torch.Tensor,
    global_gain: torch.Tensor,
) -> torch.Tensor:
    """Compute noise scalefactor offsets (sfo) for PNS bands.

    The decoder reconstructs noise as:
        noise_scaled = random_noise * sf[idx] / sqrt(random_energy)
    where sf[idx] = pow2sf_tab[sfo + 200] = 2^(sfo/4).

    So the decoded band energy per coefficient = sf². We want this to equal
    the original MDCT energy per coefficient: E/W.

    Thus: 2^(sfo/2) = E/W → sfo = 2*log2(E/W).

    Args:
        mdct_coeffs: MDCT coefficients, shape ``(B, C, 1024)``.
        sfb_offsets: Cumulative SFB offsets.
        noise_mask: Boolean mask from detect_noise_bands.
        global_gain: Per-frame global gain, shape ``(B,)``.

    Returns:
        Noise sfo per band, shape matching noise_mask. For non-PNS bands,
        the value is 0 (unused). Values are integers in [0, 255].
    """
    device = mdct_coeffs.device
    num_sfb = len(sfb_offsets) - 1
    flat_m = mdct_coeffs.reshape(-1, 1024)
    N = flat_m.shape[0]

    # Compute total energy per band (sum of mdct² over band coefficients)
    band_energy = torch.zeros(N, num_sfb, device=device)
    for i in range(num_sfb):
        s, e = sfb_offsets[i], sfb_offsets[i + 1]
        if e > 1024:
            break
        band_energy[:, i] = (flat_m[:, s:e] ** 2).sum(dim=-1)

    # The decoder generates noise with total energy sf², where
    # sf = 2^(sfo/4). We want sf² = band_energy, so:
    # 2^(sfo/2) = band_energy → sfo = 2*log2(band_energy)
    #
    # Empirical correction +C accounts for the mismatch between our
    # unnormalized forward MDCT and FFmpeg's PNS noise reconstruction.
    # Set via pns.PNS_ENERGY_CORRECTION; calibrated by sweeping C on
    # white noise until rms_ratio ≈ 1.0.
    sfo = 2.0 * torch.log2(band_energy.clamp(min=1e-20)) + PNS_ENERGY_CORRECTION

    # Clamp to valid range (decoder clips to [-100, 155])
    sfo = sfo.clamp(-100, 155).round().to(torch.int64)

    # Zero out non-PNS bands
    flat_mask = noise_mask.reshape(N, num_sfb)
    sfo = torch.where(flat_mask, sfo, torch.zeros_like(sfo))

    return sfo.reshape(noise_mask.shape)
