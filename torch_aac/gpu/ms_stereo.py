"""Mid/Side stereo coding for AAC-LC channel pair elements.

M/S stereo replaces correlated L/R pairs with M=(L+R)/2 and S=(L-R)/2.
When channels are highly correlated (most music), S is near-zero, saving
significant bits.  The transform is applied per scalefactor band — bands
with uncorrelated content (e.g. hard-panned instruments) stay as L/R.

The decoder reconstructs: L = M + S, R = M - S.
"""

from __future__ import annotations

import torch


def compute_ms_mask(
    mdct_l: torch.Tensor,
    mdct_r: torch.Tensor,
    sfb_offsets: list[int],
) -> torch.Tensor:
    """Decide which bands benefit from M/S coding.

    For each band, compares energy of (L-R)/2 (side) against (L+R)/2 (mid).
    If side energy < mid energy, M/S saves bits because the side channel
    is cheaper to code.

    Args:
        mdct_l: Left-channel MDCT coefficients, shape ``(B, 1024)``.
        mdct_r: Right-channel MDCT coefficients, shape ``(B, 1024)``.
        sfb_offsets: Cumulative SFB offsets.

    Returns:
        Boolean mask, shape ``(B, num_sfb)``. True = use M/S for this band.
    """
    B = mdct_l.shape[0]
    num_sfb = len(sfb_offsets) - 1
    device = mdct_l.device

    ms_mask = torch.zeros(B, num_sfb, device=device, dtype=torch.bool)

    for i in range(num_sfb):
        s, e = sfb_offsets[i], sfb_offsets[i + 1]
        if e > mdct_l.shape[-1]:
            break

        l_band = mdct_l[:, s:e]
        r_band = mdct_r[:, s:e]

        mid_energy = ((l_band + r_band) * 0.5).pow(2).sum(dim=-1)
        side_energy = ((l_band - r_band) * 0.5).pow(2).sum(dim=-1)

        # Use M/S when side is smaller than mid (channels correlated in this band)
        ms_mask[:, i] = side_energy < mid_energy

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

        # Mask: (B,) bool → (B, 1) for broadcasting
        use_ms = ms_mask[:, i].unsqueeze(-1)  # (B, 1)

        l_band = mdct_l[:, s:e]
        r_band = mdct_r[:, s:e]

        mid = (l_band + r_band) * 0.5
        side = (l_band - r_band) * 0.5

        out_l[:, s:e] = torch.where(use_ms, mid, l_band)
        out_r[:, s:e] = torch.where(use_ms, side, r_band)

    return out_l, out_r
