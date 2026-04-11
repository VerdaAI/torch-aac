"""GPU-accelerated psychoacoustic model for AAC-LC.

Computes masking thresholds that determine how much quantization noise
is perceptually acceptable in each scalefactor band. This drives the
bit allocation: bands where the signal masks noise get fewer bits.

V1 implements a simplified model based on ISO 11172-3 psychoacoustic
model 1 (adapted for AAC frequency bands):

1. FFT → power spectrum
2. Map to Bark scale critical bands
3. Apply spreading function (frequency-domain masking)
4. Compute signal-to-mask ratio (SMR) per scalefactor band
5. Return masking thresholds

All operations are batched on GPU for efficient processing of many
frames simultaneously.
"""

from __future__ import annotations

from functools import lru_cache

import torch


def hz_to_bark(hz: torch.Tensor) -> torch.Tensor:
    """Convert frequency in Hz to Bark scale.

    Uses Traunmüller's formula: z = 26.81 / (1 + 1960/f) - 0.53

    Args:
        hz: Frequency values in Hz.

    Returns:
        Bark-scale values.
    """
    return 26.81 * hz / (1960.0 + hz) - 0.53


def bark_to_hz(bark: torch.Tensor) -> torch.Tensor:
    """Convert Bark scale to Hz."""
    return 1960.0 * (bark + 0.53) / (26.81 - bark - 0.53)


@lru_cache(maxsize=4)
def _absolute_threshold_of_hearing(
    num_fft_bins: int,
    sample_rate: int,
    device: torch.device,
) -> torch.Tensor:
    """Compute the absolute threshold of hearing (ATH) in dB SPL.

    Uses the ISO 226 approximation:
        T(f) = 3.64 * (f/1000)^-0.8
             - 6.5 * exp(-0.6 * (f/1000 - 3.3)^2)
             + 1e-3 * (f/1000)^4

    Args:
        num_fft_bins: Number of FFT bins (typically N/2 + 1).
        sample_rate: Sample rate in Hz.
        device: Target device.

    Returns:
        ATH in dB SPL, shape ``(num_fft_bins,)``.
    """
    freqs = torch.linspace(0, sample_rate / 2, num_fft_bins, device=device)
    # Avoid division by zero at DC
    freqs = torch.clamp(freqs, min=20.0)
    f_khz = freqs / 1000.0

    ath = (
        3.64 * torch.pow(f_khz, -0.8)
        - 6.5 * torch.exp(-0.6 * (f_khz - 3.3) ** 2)
        + 1e-3 * torch.pow(f_khz, 4.0)
    )

    return ath


@lru_cache(maxsize=4)
def _spreading_function_matrix(
    num_bands: int,
    sample_rate: int,
    sfb_offsets: tuple[int, ...],
    device: torch.device,
) -> torch.Tensor:
    """Compute the spreading function matrix in Bark domain.

    The spreading function models frequency-domain masking: a loud tone
    in one critical band partially masks noise in nearby bands.

    Args:
        num_bands: Number of scalefactor bands.
        sample_rate: Sample rate in Hz.
        sfb_offsets: Tuple of SFB offsets (for hashability in lru_cache).
        device: Target device.

    Returns:
        Spreading matrix of shape ``(num_bands, num_bands)`` in linear scale.
    """
    offsets = list(sfb_offsets)
    # Compute center frequency of each band in Bark
    center_freqs = torch.zeros(num_bands, device=device)
    for i in range(num_bands):
        lo = offsets[i] * sample_rate / 2048  # FFT bin to Hz
        hi = offsets[i + 1] * sample_rate / 2048
        center_freqs[i] = (lo + hi) / 2.0

    center_bark = hz_to_bark(center_freqs)

    # Spreading function: dB attenuation as function of Bark distance
    # S(dz) = 15.81 + 7.5*(dz+0.474) - 17.5*sqrt(1 + (dz+0.474)^2)
    # where dz = bark_masker - bark_maskee
    dz = center_bark.unsqueeze(1) - center_bark.unsqueeze(0)  # (bands, bands)

    spreading_db = 15.81 + 7.5 * (dz + 0.474) - 17.5 * torch.sqrt(1.0 + (dz + 0.474) ** 2)

    # Clamp to reasonable range and convert to linear
    spreading_db = torch.clamp(spreading_db, min=-100.0, max=0.0)
    return torch.pow(10.0, spreading_db / 10.0)


def compute_masking_thresholds(
    mdct_coeffs: torch.Tensor,
    sample_rate: int,
    sfb_offsets: list[int],
) -> torch.Tensor:
    """Compute perceptual masking thresholds per scalefactor band.

    Args:
        mdct_coeffs: MDCT coefficients, shape ``(B, 1024)`` or ``(B, C, 1024)``.
        sample_rate: Sample rate in Hz.
        sfb_offsets: Cumulative SFB offsets.

    Returns:
        Masking thresholds per SFB, shape ``(B, num_sfb)`` or ``(B, C, num_sfb)``.
        Values represent the maximum quantization noise power that is
        perceptually masked in each band.
    """
    num_sfb = len(sfb_offsets) - 1
    device = mdct_coeffs.device
    original_shape = mdct_coeffs.shape

    # Flatten to 2-D: (N, 1024)
    flat = mdct_coeffs.reshape(-1, mdct_coeffs.shape[-1])
    N = flat.shape[0]

    # 1. Power spectrum from MDCT coefficients
    # MDCT coefficients approximate the power spectrum in each band
    power = flat**2  # (N, 1024)

    # 2. Compute energy per scalefactor band
    band_energy = torch.zeros(N, num_sfb, device=device)
    for i in range(num_sfb):
        start, end = sfb_offsets[i], sfb_offsets[i + 1]
        if end <= power.shape[-1]:
            band_energy[:, i] = power[:, start:end].sum(dim=-1)

    # 3. Convert to dB (with floor to avoid log(0))
    10.0 * torch.log10(band_energy + 1e-10)

    # 4. Apply spreading function
    spreading = _spreading_function_matrix(num_sfb, sample_rate, tuple(sfb_offsets), device)
    # Spread energy: (N, num_sfb) @ (num_sfb, num_sfb) → (N, num_sfb)
    spread_energy = torch.matmul(band_energy, spreading)

    # 5. Compute masking threshold
    # The threshold is the spread energy minus a masking offset
    # (tonality-dependent; we use a fixed offset for V1)
    masking_offset_db = 6.0  # dB below the masked signal
    threshold = spread_energy * (10.0 ** (-masking_offset_db / 10.0))

    # 6. Apply absolute threshold of hearing
    ath = _absolute_threshold_of_hearing(1025, sample_rate, device)
    # Map ATH to SFB bands
    ath_per_band = torch.zeros(num_sfb, device=device)
    for i in range(num_sfb):
        start, end = sfb_offsets[i], min(sfb_offsets[i + 1], 1025)
        if start < end:
            ath_per_band[i] = 10.0 ** (ath[start:end].mean() / 10.0)

    # Threshold is max of masking threshold and ATH
    threshold = torch.maximum(threshold, ath_per_band.unsqueeze(0).expand_as(threshold))

    # Reshape to original batch dims
    return threshold.reshape(*original_shape[:-1], num_sfb)


def compute_smr(
    mdct_coeffs: torch.Tensor,
    masking_thresholds: torch.Tensor,
    sfb_offsets: list[int],
) -> torch.Tensor:
    """Compute Signal-to-Mask Ratio (SMR) per scalefactor band.

    SMR = signal_energy / masking_threshold (in linear scale)

    Higher SMR means the signal is more audible above the masking threshold,
    so more bits should be allocated to that band.

    Args:
        mdct_coeffs: MDCT coefficients, shape ``(B, 1024)`` or ``(B, C, 1024)``.
        masking_thresholds: From compute_masking_thresholds, shape ``(B, num_sfb)``.
        sfb_offsets: Cumulative SFB offsets.

    Returns:
        SMR per band, shape matching masking_thresholds.
    """
    num_sfb = len(sfb_offsets) - 1
    flat = mdct_coeffs.reshape(-1, mdct_coeffs.shape[-1])
    N = flat.shape[0]

    band_energy = torch.zeros(N, num_sfb, device=flat.device)
    for i in range(num_sfb):
        start, end = sfb_offsets[i], sfb_offsets[i + 1]
        if end <= flat.shape[-1]:
            band_energy[:, i] = (flat[:, start:end] ** 2).sum(dim=-1)

    flat_thresh = masking_thresholds.reshape(N, num_sfb)
    smr = band_energy / (flat_thresh + 1e-10)

    return smr.reshape(*mdct_coeffs.shape[:-1], num_sfb)
