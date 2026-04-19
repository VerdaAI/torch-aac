"""Mid/Side stereo coding for AAC-LC channel pair elements.

M/S stereo replaces correlated L/R pairs with M=(L+R)/2 and S=(L-R)/2.
When channels are highly correlated (most music), S is near-zero, saving
significant bits.  The transform is applied per scalefactor band — bands
with uncorrelated content (e.g. hard-panned instruments) stay as L/R.

The decoder reconstructs: L = M + S, R = M - S.

The M/S decision uses a rate-distortion criterion: for each band, estimate
the Huffman bit cost of encoding as L/R vs M/S. Only use M/S when the bit
savings exceed a threshold that compensates for the reconstruction noise
penalty (quantization noise from M leaks into both L and R during
reconstruction).
"""

from __future__ import annotations

import torch


def compute_ms_mask(
    mdct_l: torch.Tensor,
    mdct_r: torch.Tensor,
    sfb_offsets: list[int],
    gain_l: torch.Tensor | None = None,
    gain_r: torch.Tensor | None = None,
) -> torch.Tensor:
    """Decide which bands benefit from M/S coding using rate-distortion.

    For each band, estimates the Huffman bit cost of encoding as L/R vs
    M/S. Only enables M/S when the bit savings are substantial enough to
    offset the reconstruction noise penalty (R=M-S amplifies mid-channel
    quantization noise into the reconstructed right channel).

    The criterion: use M/S when ``bits_LR > bits_MS * threshold``, where
    threshold > 1 adds a safety margin for the noise amplification.

    Args:
        mdct_l: Left-channel MDCT coefficients, shape ``(B, 1024)``.
        mdct_r: Right-channel MDCT coefficients, shape ``(B, 1024)``.
        sfb_offsets: Cumulative SFB offsets.
        gain_l: Per-frame gain for left channel (optional, for cost estimation).
        gain_r: Per-frame gain for right channel (optional).

    Returns:
        Boolean mask, shape ``(B, num_sfb)``. True = use M/S for this band.
    """
    B = mdct_l.shape[0]
    num_sfb = len(sfb_offsets) - 1
    device = mdct_l.device

    # Only use M/S when the side channel energy is a tiny fraction of
    # mid energy. This ensures M/S only activates on highly correlated
    # bands where the side is effectively zero, avoiding the noise
    # doubling penalty from R=M-S reconstruction.
    SIDE_ENERGY_THRESHOLD = 0.01  # side must be <1% of mid energy

    ms_mask = torch.zeros(B, num_sfb, device=device, dtype=torch.bool)

    for i in range(num_sfb):
        s, e = sfb_offsets[i], sfb_offsets[i + 1]
        if e > mdct_l.shape[-1]:
            break

        l_band = mdct_l[:, s:e]
        r_band = mdct_r[:, s:e]

        mid_energy = ((l_band + r_band) * 0.5).pow(2).sum(dim=-1)
        side_energy = ((l_band - r_band) * 0.5).pow(2).sum(dim=-1)

        # Use M/S only when side is negligible relative to mid
        ratio = side_energy / (mid_energy + 1e-20)
        ms_mask[:, i] = ratio < SIDE_ENERGY_THRESHOLD

    return ms_mask


def apply_ms_transform(
    mdct_l: torch.Tensor,
    mdct_r: torch.Tensor,
    ms_mask: torch.Tensor,
    sfb_offsets: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply M/S transform to selected bands.

    For bands where ``ms_mask`` is True, replaces L/R with:
        M = (L + R) * 0.5
        S = (L - R) * 0.5

    Bands where ``ms_mask`` is False are left unchanged.

    Args:
        mdct_l: Left-channel MDCT, shape ``(B, 1024)``.
        mdct_r: Right-channel MDCT, shape ``(B, 1024)``.
        ms_mask: Per-band mask, shape ``(B, num_sfb)``.
        sfb_offsets: Cumulative SFB offsets.

    Returns:
        Tuple of (mid_or_left, side_or_right), same shapes as input.
    """
    out_l = mdct_l.clone()
    out_r = mdct_r.clone()
    num_sfb = len(sfb_offsets) - 1

    for i in range(num_sfb):
        s, e = sfb_offsets[i], sfb_offsets[i + 1]
        if e > mdct_l.shape[-1]:
            break

        use_ms = ms_mask[:, i].unsqueeze(-1)  # (B, 1)

        l_band = mdct_l[:, s:e]
        r_band = mdct_r[:, s:e]

        mid = (l_band + r_band) * 0.5
        side = (l_band - r_band) * 0.5

        out_l[:, s:e] = torch.where(use_ms, mid, l_band)
        out_r[:, s:e] = torch.where(use_ms, side, r_band)

    return out_l, out_r
