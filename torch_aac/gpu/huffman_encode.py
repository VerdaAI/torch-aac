"""GPU-parallel Huffman encoding for AAC-LC spectral data.

Replaces the CPU-side Python per-group Huffman lookup loop with vectorized
GPU tensor operations. For each frame, the pipeline is:

1. Compute a flat Huffman index per coefficient group via weighted sums.
2. Gather (code, length) from pre-built GPU lookup tables.
3. Compute sign bits for unsigned codebooks.
4. Compute escape prefixes + mantissas for codebook 11 values > 15.
5. Return flat (codes, lengths) arrays ready for C bitwriter_pack.

All operations are batched across frames and groups for maximum GPU
utilization.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch

from torch_aac.tables.huffman_tables import (
    CODEBOOK_DIMENSION,
    CODEBOOK_MAX_ABS,
    CODEBOOK_NUM_ENTRIES,
    CODEBOOK_UNSIGNED,
    CODEBOOKS,
    SCALEFACTOR_BITS,
    SCALEFACTOR_CODE,
)

# Codebook properties as plain lists for fast indexing
_DIM = CODEBOOK_DIMENSION  # [0, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2]
# Base = range of values per position in the codebook index computation.
# Signed (cb1,2,5,6): base = max_abs*2+1 (values -max..+max)
# Unsigned (cb3,4,7-11): base = max_abs+1 (values 0..max)
_BASE = [0, 3, 3, 3, 3, 9, 9, 8, 8, 13, 13, 17]
_OFFSET = [0, 1, 1, 0, 0, 4, 4, 0, 0, 0, 0, 0]  # value offset for signed cbooks
_MAX_ENTRIES = max(CODEBOOK_NUM_ENTRIES[1:12])  # 289 for cb11


@lru_cache(maxsize=1)
def _build_gpu_tables(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Build padded (12, max_entries) lookup tables on GPU.

    Returns (all_codes, all_lengths) where:
        all_codes[cb, flat_idx] = Huffman codeword (uint32)
        all_lengths[cb, flat_idx] = codeword bit length (uint8)

    Codebook 0 is unused (zero-filled).
    """
    # Pad to 14 rows to safely handle NOISE_BT=13 indices (row 13 stays zero)
    all_codes = torch.zeros(14, _MAX_ENTRIES, dtype=torch.int64, device=device)
    all_lengths = torch.zeros(14, _MAX_ENTRIES, dtype=torch.int64, device=device)

    for cb in range(1, 12):
        codebook = CODEBOOKS[cb]
        if codebook is None:
            continue
        dim = _DIM[cb]
        base = _BASE[cb]
        offset = _OFFSET[cb]
        for values, (code, length) in codebook.items():
            # Compute flat index from values tuple (MSB-first ordering)
            flat_idx = 0
            for d in range(dim):
                flat_idx = flat_idx * base + (values[d] + offset)
            if 0 <= flat_idx < _MAX_ENTRIES:
                all_codes[cb, flat_idx] = code
                all_lengths[cb, flat_idx] = length

    return all_codes, all_lengths


@lru_cache(maxsize=1)
def _build_sf_table(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Build scalefactor VLC table on GPU. Index = delta + 60."""
    sf_codes = torch.tensor(SCALEFACTOR_CODE, dtype=torch.int64, device=device)
    sf_lengths = torch.tensor(SCALEFACTOR_BITS, dtype=torch.int64, device=device)
    return sf_codes, sf_lengths


def _compute_group_indices(
    quantized: torch.Tensor,
    codebook_per_coeff: torch.Tensor,
    dim_per_coeff: torch.Tensor,
    base_per_coeff: torch.Tensor,
    offset_per_coeff: torch.Tensor,
    is_unsigned_per_coeff: torch.Tensor,
    group_starts: torch.Tensor,
    group_dims: torch.Tensor,
    group_cbs: torch.Tensor,
) -> torch.Tensor:
    """Compute flat Huffman indices for each group.

    For a group starting at position s with dim d, codebook cb:
        For signed: idx = sum((q[s+j] + offset) * base^(d-1-j) for j in range(d))
        For unsigned: idx = sum(|q[s+j]| * base^(d-1-j) for j in range(d))
                     but clamp to codebook max_abs (or 16 for cb11)
    """
    device = quantized.device
    B = quantized.shape[0]
    G = group_starts.shape[1]

    # Gather coefficient values for each group position
    # group_starts: (B, G), group_dims: (G,), group_cbs: (B, G)
    max_dim = 4
    # Build indices for all positions in each group
    offsets = torch.arange(max_dim, device=device)  # (4,)
    # (B, G, 4): index of each coeff in the group
    coeff_indices = group_starts.unsqueeze(-1) + offsets.unsqueeze(0).unsqueeze(0)
    coeff_indices = coeff_indices.clamp(0, 1023)  # safety

    # Gather values: (B, G, 4)
    vals = quantized.unsqueeze(1).expand(B, G, -1).gather(2, coeff_indices)

    # For unsigned codebooks, take absolute values
    is_unsigned = CODEBOOK_UNSIGNED  # list of bool per cb
    unsigned_mask = torch.tensor(
        [is_unsigned[cb] if cb < len(is_unsigned) else False for cb in range(12)],
        device=device,
        dtype=torch.bool,
    )
    group_unsigned = unsigned_mask[group_cbs]  # (B, G)

    abs_vals = vals.abs()
    # Clamp for cb11 (escape): lookup values capped at 16
    # For other unsigned: capped at their max_abs
    max_abs_table = torch.tensor(
        [0, 1, 1, 2, 2, 4, 4, 7, 7, 12, 12, 16], device=device, dtype=torch.long
    )
    cb_max = max_abs_table[group_cbs].unsqueeze(-1)  # (B, G, 1)
    clamped_abs = abs_vals.clamp(max=cb_max.expand_as(abs_vals))

    # Choose between signed values (with offset) and unsigned (abs + clamped)
    offset_table = torch.tensor(_OFFSET, device=device, dtype=torch.long)
    base_table = torch.tensor(_BASE, device=device, dtype=torch.long)
    dim_table = torch.tensor(_DIM, device=device, dtype=torch.long)

    group_offset = offset_table[group_cbs].unsqueeze(-1)  # (B, G, 1)
    group_base = base_table[group_cbs].unsqueeze(-1)  # (B, G, 1)
    group_dim = dim_table[group_cbs]  # (B, G)

    # For signed: use (val + offset)
    signed_vals = vals + group_offset
    # For unsigned: use clamped_abs (offset is 0 for all unsigned books)
    lookup_vals = torch.where(group_unsigned.unsqueeze(-1), clamped_abs, signed_vals)

    # Zero out positions beyond the group's dimension
    dim_mask = offsets.unsqueeze(0).unsqueeze(0) < group_dim.unsqueeze(-1)  # (B, G, 4)
    lookup_vals = lookup_vals * dim_mask.long()

    # Compute flat index: MSB-first weighted sum
    # idx = v[0]*base^(d-1) + v[1]*base^(d-2) + ... + v[d-1]*base^0
    # With max_dim=4: idx = v[0]*base^3 + v[1]*base^2 + v[2]*base + v[3]
    # For dim=2: v[2] and v[3] are 0, base^3 and base^2 terms contribute 0
    # BUT we need to shift: for dim=2, it should be v[0]*base + v[1]
    # The values at positions >= dim are already zeroed, but the base powers
    # should be base^(dim-1-j) not base^(3-j). Fix: compute relative to dim.
    #
    # Actually simpler: Horner's method per group.
    # idx = ((v[0] * base + v[1]) * base + v[2]) * base + v[3]
    # For dim=2 with v[2]=v[3]=0: idx = v[0]*base^3 + v[1]*base^2
    # That's WRONG — should be v[0]*base + v[1].
    #
    # Fix: only iterate up to dim positions.
    # Since dim varies per group, we compute Horner for all 4 and adjust.

    # Compute with Horner's method for each position
    gb = group_base.squeeze(-1)  # (B, G)
    idx = torch.zeros(B, G, device=device, dtype=torch.long)
    for j in range(max_dim):
        active = j < group_dim  # (B, G) bool
        idx = torch.where(active, idx * gb + lookup_vals[:, :, j], idx)

    return idx


def encode_spectral_gpu(
    quantized: torch.Tensor,
    codebook_indices: torch.Tensor,
    scalefactors: torch.Tensor,
    global_gain: torch.Tensor,
    sfb_offsets: list[int],
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Encode spectral + scalefactor + section data on GPU.

    Returns a list of (codes, lengths) numpy arrays per frame, ready for
    C bitwriter_pack. Each pair includes ICS header, section data,
    scalefactor data, and spectral data — everything after the ADTS header
    and syntactic element ID.

    Args:
        quantized: (B, 1024) int quantized coefficients.
        codebook_indices: (B, num_sfb) int codebook per band.
        scalefactors: (B, num_sfb) int per-band scalefactors.
        global_gain: (B,) int.
        sfb_offsets: SFB offset table.

    Returns:
        List of B tuples (codes_np, lengths_np) for C packing.
    """
    device = quantized.device
    B = quantized.shape[0]
    num_sfb = len(sfb_offsets) - 1

    all_codes_table, all_lengths_table = _build_gpu_tables(device)
    sf_codes_table, sf_lengths_table = _build_sf_table(device)

    results: list[tuple[np.ndarray, np.ndarray]] = []

    # Process per-frame (could batch further but frame structure varies)
    for b in range(B):
        frame_codes: list[int] = []
        frame_lengths: list[int] = []
        q = quantized[b]  # (1024,)
        cbs = codebook_indices[b]  # (num_sfb,)
        sfs = scalefactors[b]  # (num_sfb,)
        gg = int(global_gain[b].item())

        # --- Determine max_sfb ---
        max_sfb = 0
        for i in range(num_sfb):
            if int(cbs[i].item()) != 0:
                max_sfb = i + 1
        max_sfb = min(max_sfb, 51)

        # --- ICS header (fixed fields) ---
        frame_codes.append(gg)
        frame_lengths.append(8)  # global_gain
        frame_codes.append(0)
        frame_lengths.append(1)  # ics_reserved_bit
        frame_codes.append(0)
        frame_lengths.append(2)  # window_sequence = ONLY_LONG
        frame_codes.append(0)
        frame_lengths.append(1)  # window_shape = sine
        frame_codes.append(max_sfb)
        frame_lengths.append(6)  # max_sfb
        frame_codes.append(0)
        frame_lengths.append(1)  # predictor_data_present

        if max_sfb > 0:
            # --- Section data ---
            i = 0
            while i < max_sfb:
                cb = int(cbs[i].item())
                run = 1
                while i + run < max_sfb and int(cbs[i + run].item()) == cb:
                    run += 1
                frame_codes.append(cb)
                frame_lengths.append(4)
                remaining = run
                while remaining >= 31:
                    frame_codes.append(31)
                    frame_lengths.append(5)
                    remaining -= 31
                frame_codes.append(remaining)
                frame_lengths.append(5)
                i += run

            # --- Scalefactor data ---
            prev_sf = gg
            for i in range(max_sfb):
                if int(cbs[i].item()) == 0:
                    continue
                sf = int(sfs[i].item())
                delta = max(-60, min(60, sf - prev_sf))
                idx = delta + 60
                frame_codes.append(int(sf_codes_table[idx].item()))
                frame_lengths.append(int(sf_lengths_table[idx].item()))
                prev_sf = prev_sf + delta

        # --- Pulse/TNS/gain flags ---
        frame_codes.append(0)
        frame_lengths.append(1)  # pulse_data_present
        frame_codes.append(0)
        frame_lengths.append(1)  # tns_data_present
        frame_codes.append(0)
        frame_lengths.append(1)  # gain_control_data_present

        if max_sfb > 0:
            # --- Spectral data (GPU-accelerated lookup) ---
            _encode_spectral_bands_gpu(
                q,
                cbs,
                max_sfb,
                sfb_offsets,
                all_codes_table,
                all_lengths_table,
                frame_codes,
                frame_lengths,
            )

        results.append(
            (
                np.array(frame_codes, dtype=np.uint32),
                np.array(frame_lengths, dtype=np.uint8),
            )
        )

    return results


def _encode_spectral_bands_gpu(
    q: torch.Tensor,
    cbs: torch.Tensor,
    max_sfb: int,
    sfb_offsets: list[int],
    all_codes_table: torch.Tensor,
    all_lengths_table: torch.Tensor,
    frame_codes: list[int],
    frame_lengths: list[int],
) -> None:
    """Spectral Huffman encoding with GPU lookup + CPU post-processing.

    Converts the quantized coefficients to a flat Python list once, then
    iterates groups purely in Python (no GPU kernel launches per band).
    The GPU tables are used via a single pre-transfer to CPU numpy at init.
    """
    # Pull tables to CPU once (cached globally — see _get_cpu_tables)
    cpu_codes, cpu_lengths = _get_cpu_tables(all_codes_table, all_lengths_table)
    q_list: list[int] = [int(v) for v in q.cpu().tolist()]

    for band_idx in range(max_sfb):
        cb = int(cbs[band_idx].item())
        if cb == 0:
            continue

        start = sfb_offsets[band_idx]
        end = sfb_offsets[band_idx + 1]
        dim = _DIM[cb]
        base = _BASE[cb]
        offset = _OFFSET[cb]
        is_unsigned = CODEBOOK_UNSIGNED[cb]

        # Iterate groups in pure Python (list indexing, no numpy/torch)
        pos = start
        while pos + dim <= end:
            # Compute flat Huffman index via Horner's method
            idx = 0
            if is_unsigned:
                for j in range(dim):
                    v = q_list[pos + j]
                    av = -v if v < 0 else v
                    clamp = 16 if cb == 11 else CODEBOOK_MAX_ABS[cb]
                    av = av if av <= clamp else clamp
                    idx = idx * base + av
            else:
                for j in range(dim):
                    idx = idx * base + (q_list[pos + j] + offset)

            if idx < 0 or idx >= _MAX_ENTRIES:
                idx = 0

            frame_codes.append(cpu_codes[cb][idx])
            frame_lengths.append(cpu_lengths[cb][idx])

            if is_unsigned:
                # Sign bits for non-zero values
                sign_val = 0
                nnz = 0
                for j in range(dim):
                    v = q_list[pos + j]
                    av = -v if v < 0 else v
                    if av != 0:
                        sign_val = (sign_val << 1) | (1 if v < 0 else 0)
                        nnz += 1
                if nnz > 0:
                    frame_codes.append(sign_val)
                    frame_lengths.append(nnz)

                # Escape codes for cb11
                if cb == 11:
                    for j in range(dim):
                        v = q_list[pos + j]
                        av = -v if v < 0 else v
                        if av > 15:
                            av = min(av, 8191)
                            n = max(0, min(av.bit_length() - 5, 8))
                            mantissa_bits = n + 4
                            mantissa = av - (1 << mantissa_bits)
                            if n > 0:
                                frame_codes.append(((1 << n) - 1) << 1)
                                frame_lengths.append(n + 1)
                            else:
                                frame_codes.append(0)
                                frame_lengths.append(1)
                            frame_codes.append(mantissa)
                            frame_lengths.append(mantissa_bits)
            pos += dim

        # Handle leftover coefficients (band width not multiple of dim)
        if pos < end:
            # Pad remainder with zeros
            remainder = list(q_list[pos:end]) + [0] * (dim - (end - pos))
            idx = 0
            if is_unsigned:
                for j in range(dim):
                    av = abs(remainder[j])
                    clamp = 16 if cb == 11 else CODEBOOK_MAX_ABS[cb]
                    av = min(av, clamp)
                    idx = idx * base + av
            else:
                for j in range(dim):
                    idx = idx * base + (remainder[j] + offset)
            idx = max(0, min(idx, _MAX_ENTRIES - 1))
            frame_codes.append(cpu_codes[cb][idx])
            frame_lengths.append(cpu_lengths[cb][idx])
            if is_unsigned:
                sign_val = 0
                nnz = 0
                for j in range(dim):
                    v = remainder[j]
                    av = abs(v)
                    if av != 0:
                        sign_val = (sign_val << 1) | (1 if v < 0 else 0)
                        nnz += 1
                if nnz:
                    frame_codes.append(sign_val)
                    frame_lengths.append(nnz)


# Cache CPU-side tables to avoid repeated GPU→CPU transfers
_cpu_tables_cache: dict[int, tuple[list[list[int]], list[list[int]]]] = {}


def _get_cpu_tables(
    all_codes: torch.Tensor, all_lengths: torch.Tensor
) -> tuple[list[list[int]], list[list[int]]]:
    """Transfer GPU Huffman tables to CPU lists (cached)."""
    key = id(all_codes)
    if key not in _cpu_tables_cache:
        codes = all_codes.cpu().tolist()
        lengths = all_lengths.cpu().tolist()
        _cpu_tables_cache[key] = (codes, lengths)
    return _cpu_tables_cache[key]


# ---------------------------------------------------------------------------
# Batch-flattened GPU spectral encoding (pairs-only, all frames at once)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=4)
def _build_pair_group_map(
    sfb_offsets_tuple: tuple[int, ...], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Build static group map assuming dim=2 for all bands.

    Returns:
        group_band_idx: (G,) int — which SFB each group belongs to.
        group_coef_start: (G,) int — starting coefficient index.
        num_groups: total groups per frame.
    """
    band_indices: list[int] = []
    coef_starts: list[int] = []
    offsets = list(sfb_offsets_tuple)
    num_sfb = len(offsets) - 1
    for i in range(num_sfb):
        s, e = offsets[i], offsets[i + 1]
        if e > 1024:
            break
        for g_start in range(s, e, 2):
            band_indices.append(i)
            coef_starts.append(g_start)
    return (
        torch.tensor(band_indices, device=device, dtype=torch.long),
        torch.tensor(coef_starts, device=device, dtype=torch.long),
        len(band_indices),
    )


def encode_spectral_batched(
    quantized: torch.Tensor,
    codebooks: torch.Tensor,
    sfb_offsets: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Batch-vectorized spectral Huffman lookup for ALL frames at once.

    All frames × all groups are processed in a single GPU gather. Only pair
    codebooks (cb5-11) are expected — use ``select_codebooks(pairs_only=True)``
    or remap cb1-4 before calling.

    Args:
        quantized: (N, 1024) int tensor — quantized spectral coefficients
            (N = B*C, already flattened across batch and channels).
        codebooks: (N, num_sfb) int — codebook per band.
        sfb_offsets: cumulative SFB offsets.

    Returns:
        Tuple of numpy arrays, each shape (N, G, S) where G = groups/frame
        and S = max entries per group (6: huff code, sign bits, 2× escape
        prefix+mantissa):
            - spectral_codes: uint32
            - spectral_lengths: uint8
            - active_mask: bool — True for groups that should be emitted
    """
    device = quantized.device
    N = quantized.shape[0]
    num_sfb = len(sfb_offsets) - 1

    group_band, group_start, G = _build_pair_group_map(tuple(sfb_offsets), device)

    # Property tables on device — pad to 14 entries to handle NOISE_BT=13 safely
    def _pad(lst, n):
        return lst + [lst[0]] * (n - len(lst))

    base_t = torch.tensor(_pad(_BASE, 14), device=device, dtype=torch.long)
    offset_t = torch.tensor(_pad(_OFFSET, 14), device=device, dtype=torch.long)
    maxabs_t = torch.tensor(_pad(CODEBOOK_MAX_ABS, 14), device=device, dtype=torch.long)
    unsigned_t = torch.tensor(_pad(CODEBOOK_UNSIGNED, 14), device=device, dtype=torch.bool)

    all_codes_t, all_lengths_t = _build_gpu_tables(device)

    quantized = quantized.long()  # ensure integer dtype for indexing

    # --- Per-group codebook and active mask ---
    # group_cb: (N, G) — codebook for each group
    group_cb = codebooks[:, group_band]  # (N, G)
    # Active = non-zero AND non-PNS (cb=13 has no spectral data)
    active = (group_cb > 0) & (group_cb <= 11)  # (N, G)

    # --- Gather coefficient pairs ---
    coef_a_idx = group_start.unsqueeze(0).expand(N, -1)  # (N, G)
    coef_b_idx = (group_start + 1).clamp(max=1023).unsqueeze(0).expand(N, -1)
    coef_a = quantized.gather(1, coef_a_idx)  # (N, G)
    coef_b = quantized.gather(1, coef_b_idx)  # (N, G)
    # Zero out coef_b for odd-width band remainder (start+1 >= band_end)
    band_end = torch.tensor(
        [sfb_offsets[min(i + 1, num_sfb)] for i in range(num_sfb)],
        device=device,
        dtype=torch.long,
    )
    past_end = (group_start + 1) >= band_end[group_band]
    coef_b = coef_b.masked_fill(past_end.unsqueeze(0).expand(N, -1), 0)

    abs_a = coef_a.abs()
    abs_b = coef_b.abs()

    # --- Compute flat Huffman index ---
    gb = base_t[group_cb]  # (N, G)
    go = offset_t[group_cb]
    gm = maxabs_t[group_cb]
    is_unsigned = unsigned_t[group_cb]  # (N, G)

    # Signed: idx = (a + offset) * base + (b + offset)
    signed_idx = (coef_a + go) * gb + (coef_b + go)
    # Unsigned: idx = clamp(|a|, max) * base + clamp(|b|, max)
    ua = torch.minimum(abs_a, gm)
    ub = torch.minimum(abs_b, gm)
    unsigned_idx = ua * gb + ub

    flat_idx = torch.where(is_unsigned, unsigned_idx, signed_idx)
    flat_idx = flat_idx.clamp(0, _MAX_ENTRIES - 1)

    # --- Gather Huffman codes/lengths ---
    huff_codes = all_codes_t[group_cb, flat_idx]  # (N, G)
    huff_lengths = all_lengths_t[group_cb, flat_idx]  # (N, G)

    # --- Sign bits for unsigned codebooks ---
    nz_a = abs_a > 0
    nz_b = abs_b > 0
    sign_a = (coef_a < 0).long()
    sign_b = (coef_b < 0).long()
    # Pack: both nonzero → 2 bits, one nonzero → 1 bit, none → 0 bits
    num_sign = (nz_a.long() + nz_b.long()) * is_unsigned.long()
    sign_val = (
        torch.where(
            nz_a & nz_b,
            sign_a * 2 + sign_b,
            torch.where(nz_a, sign_a, torch.where(nz_b, sign_b, torch.zeros_like(sign_a))),
        )
        * is_unsigned.long()
    )

    # --- Escape codes for cb11 values > 15 ---
    is_cb11 = group_cb == 11
    esc_a = is_cb11 & (abs_a > 15)
    esc_b = is_cb11 & (abs_b > 15)

    def _escape_fields(
        av: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute escape prefix code/length and mantissa code/length."""
        av = av.clamp(max=8191)
        # N_esc = floor(log2(val)) - 4 = bit_length - 5
        # Use log2 approximation: bit_length ≈ floor(log2(x)) + 1
        bl = torch.floor(torch.log2(av.float().clamp(min=1))) + 1
        n_esc = (bl.long() - 5).clamp(0, 8)
        mantissa_bits = n_esc + 4
        mantissa = av - (1 << mantissa_bits.clamp(max=12))  # avoid huge shifts
        # Prefix: N ones + zero = ((1<<N)-1)<<1 in N+1 bits, or just 0 in 1 bit for N=0
        prefix_code = torch.where(
            n_esc > 0,
            ((1 << n_esc) - 1) << 1,
            torch.zeros_like(n_esc),
        )
        prefix_len = torch.where(n_esc > 0, n_esc + 1, torch.ones_like(n_esc))
        # Zero out non-escape entries
        prefix_code = prefix_code * mask.long()
        prefix_len = prefix_len * mask.long()
        mantissa = mantissa * mask.long()
        mantissa_bits = mantissa_bits * mask.long()
        return prefix_code, prefix_len, mantissa, mantissa_bits

    esc_a_prefix, esc_a_plen, esc_a_mant, esc_a_mlen = _escape_fields(abs_a, esc_a)
    esc_b_prefix, esc_b_plen, esc_b_mant, esc_b_mlen = _escape_fields(abs_b, esc_b)

    # --- Pack into (N, G, 6) output: [huff, sign, esc_a_prefix, esc_a_mant, esc_b_prefix, esc_b_mant] ---
    S = 6
    out_codes = torch.zeros(N, G, S, device=device, dtype=torch.long)
    out_lengths = torch.zeros(N, G, S, device=device, dtype=torch.long)

    out_codes[:, :, 0] = huff_codes
    out_lengths[:, :, 0] = huff_lengths
    out_codes[:, :, 1] = sign_val
    out_lengths[:, :, 1] = num_sign
    out_codes[:, :, 2] = esc_a_prefix
    out_lengths[:, :, 2] = esc_a_plen
    out_codes[:, :, 3] = esc_a_mant
    out_lengths[:, :, 3] = esc_a_mlen
    out_codes[:, :, 4] = esc_b_prefix
    out_lengths[:, :, 4] = esc_b_plen
    out_codes[:, :, 5] = esc_b_mant
    out_lengths[:, :, 5] = esc_b_mlen

    # Transfer to CPU
    codes_np = out_codes.cpu().numpy().astype(np.uint32)
    lengths_np = out_lengths.cpu().numpy().astype(np.uint8)
    active_np = active.cpu().numpy()

    return codes_np, lengths_np, active_np
