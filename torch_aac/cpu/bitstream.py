"""ADTS bitstream writer for AAC-LC.

Assembles complete AAC-LC ADTS frames from quantized spectral data.
Handles ADTS header construction, section data, scalefactor data,
and spectral data packing.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from torch_aac.config import (
    ADTS_HEADER_SIZE,
    AOT_AAC_LC,
    EncoderConfig,
)


class _PyBitWriter:
    """Pure-Python bit-level writer (fallback when C extension unavailable)."""

    __slots__ = ("_accum", "_buffer", "_byte_pos", "_nbits", "_total_bits")

    def __init__(self, capacity: int = 8192) -> None:
        self._buffer = bytearray(capacity)
        self._byte_pos = 0
        self._accum = 0
        self._nbits = 0
        self._total_bits = 0

    def write_bits(self, value: int, num_bits: int) -> None:
        mask = (1 << num_bits) - 1
        self._accum = (self._accum << num_bits) | (value & mask)
        self._nbits += num_bits
        self._total_bits += num_bits
        while self._nbits >= 8:
            self._nbits -= 8
            byte = (self._accum >> self._nbits) & 0xFF
            if self._byte_pos >= len(self._buffer):
                self._buffer.extend(b"\x00" * 1024)
            self._buffer[self._byte_pos] = byte
            self._byte_pos += 1
            self._accum &= (1 << self._nbits) - 1 if self._nbits else 0

    @property
    def bits_written(self) -> int:
        return self._total_bits

    def to_bytes(self) -> bytes:
        if self._nbits == 0:
            return bytes(self._buffer[: self._byte_pos])
        if self._byte_pos >= len(self._buffer):
            self._buffer.extend(b"\x00" * 1)
        self._buffer[self._byte_pos] = (self._accum << (8 - self._nbits)) & 0xFF
        return bytes(self._buffer[: self._byte_pos + 1])

    def align_to_byte(self) -> None:
        if self._nbits > 0:
            self.write_bits(0, 8 - self._nbits)


class _CBitWriter:
    """Deferred-packing BitWriter backed by the C extension.

    Collects ``(code, length)`` pairs in Python lists (O(1) amortized
    append), then packs them all at once via the C ``bitwriter_pack`` in
    ``to_bytes()``. This eliminates the per-call Python accumulator math
    that dominated the profile.
    """

    __slots__ = ("_codes", "_lengths", "_total_bits")

    def __init__(self, capacity: int = 8192) -> None:
        self._codes: list[int] = []
        self._lengths: list[int] = []
        self._total_bits = 0

    def write_bits(self, value: int, num_bits: int) -> None:
        self._codes.append(value)
        self._lengths.append(num_bits)
        self._total_bits += num_bits

    @property
    def bits_written(self) -> int:
        return self._total_bits

    def to_bytes(self) -> bytes:
        from torch_aac.cpu._bitwriter_native import bitwriter_pack

        codes = np.array(self._codes, dtype=np.uint32)
        lengths = np.array(self._lengths, dtype=np.uint8)
        out_size = max(1024, (self._total_bits + 7) // 8 + 16)
        output = np.zeros(out_size, dtype=np.uint8)
        total = bitwriter_pack(codes, lengths, output)
        return bytes(output[: (total + 7) // 8])

    def align_to_byte(self) -> None:
        if self._total_bits % 8 != 0:
            self.write_bits(0, 8 - (self._total_bits % 8))


def _select_bitwriter() -> type:
    """Pick the fastest available BitWriter implementation."""
    try:
        from torch_aac.cpu._bitwriter_native import is_available

        if is_available():
            return _CBitWriter
    except ImportError:
        pass
    return _PyBitWriter


BitWriter = _select_bitwriter()


def write_adts_header(
    writer: BitWriter,
    config: EncoderConfig,
    frame_length_bytes: int,
) -> None:
    """Write a 7-byte ADTS fixed header (no CRC).

    ADTS header format (56 bits = 7 bytes):
        - syncword:           12 bits (0xFFF)
        - ID:                  1 bit  (0 = MPEG-4)
        - layer:               2 bits (0b00)
        - protection_absent:   1 bit  (1 = no CRC)
        - profile:             2 bits (1 = AAC-LC, 0-indexed from AOT)
        - sampling_freq_index: 4 bits
        - private_bit:         1 bit  (0)
        - channel_config:      3 bits
        - originality:         1 bit  (0)
        - home:                1 bit  (0)
        - copyright_id:        1 bit  (0)
        - copyright_start:     1 bit  (0)
        - frame_length:       13 bits (header + payload)
        - buffer_fullness:    11 bits (0x7FF = VBR)
        - num_raw_data_blocks: 2 bits (0 = 1 block per frame)

    Args:
        writer: BitWriter to write to.
        config: Encoder configuration.
        frame_length_bytes: Total frame length including header.
    """
    # Syncword
    writer.write_bits(0xFFF, 12)
    # ID: 0 = MPEG-4
    writer.write_bits(0, 1)
    # Layer: always 0
    writer.write_bits(0, 2)
    # Protection absent: 1 = no CRC
    writer.write_bits(1, 1)
    # Profile: AAC-LC = AOT 2, but ADTS profile = AOT - 1 = 1
    writer.write_bits(AOT_AAC_LC - 1, 2)
    # Sampling frequency index
    writer.write_bits(config.sample_rate_index, 4)
    # Private bit
    writer.write_bits(0, 1)
    # Channel configuration
    writer.write_bits(config.channel_config, 3)
    # Originality, home, copyright_id, copyright_start
    writer.write_bits(0, 4)
    # Frame length (13 bits): includes header
    writer.write_bits(frame_length_bytes, 13)
    # Buffer fullness: 0x7FF = VBR
    writer.write_bits(0x7FF, 11)
    # Number of raw data blocks minus 1
    writer.write_bits(0, 2)


def write_single_channel_element(
    writer: BitWriter,
    quantized: NDArray[np.int32],
    global_gain: int,
    codebook_indices: NDArray[np.int32],
    sfb_offsets: list[int],
    huffman_encode_fn: object,
    scalefactors: NDArray[np.int32] | None = None,
    noise_scalefactors: NDArray[np.int32] | None = None,
    window_sequence: int = 0,
) -> None:
    """Write a single_channel_element (SCE) to the bitstream."""
    writer.write_bits(0b000, 3)
    writer.write_bits(0, 4)
    _write_ics(
        writer,
        quantized,
        global_gain,
        codebook_indices,
        sfb_offsets,
        huffman_encode_fn,
        scalefactors,
        noise_scalefactors,
        window_sequence=window_sequence,
    )


def write_channel_pair_element(
    writer: BitWriter,
    quantized_l: NDArray[np.int32],
    quantized_r: NDArray[np.int32],
    global_gain_l: int,
    global_gain_r: int,
    codebook_indices_l: NDArray[np.int32],
    codebook_indices_r: NDArray[np.int32],
    sfb_offsets: list[int],
    huffman_encode_fn: object,
    scalefactors_l: NDArray[np.int32] | None = None,
    scalefactors_r: NDArray[np.int32] | None = None,
    noise_scalefactors_l: NDArray[np.int32] | None = None,
    noise_scalefactors_r: NDArray[np.int32] | None = None,
    window_sequence: int = 0,
) -> None:
    """Write a channel_pair_element (CPE) to the bitstream."""
    writer.write_bits(0b001, 3)
    writer.write_bits(0, 4)
    writer.write_bits(0, 1)  # common_window=0

    _write_ics(
        writer,
        quantized_l,
        global_gain_l,
        codebook_indices_l,
        sfb_offsets,
        huffman_encode_fn,
        scalefactors_l,
        noise_scalefactors=noise_scalefactors_l,
        window_sequence=window_sequence,
    )
    _write_ics(
        writer,
        quantized_r,
        global_gain_r,
        codebook_indices_r,
        sfb_offsets,
        huffman_encode_fn,
        scalefactors_r,
        noise_scalefactors=noise_scalefactors_r,
        window_sequence=window_sequence,
    )


def _write_ics(
    writer: BitWriter,
    quantized: NDArray[np.int32],
    global_gain: int,
    codebook_indices: NDArray[np.int32],
    sfb_offsets: list[int],
    huffman_encode_fn: object,
    scalefactors: NDArray[np.int32] | None = None,
    noise_scalefactors: NDArray[np.int32] | None = None,
    window_sequence: int = 0,
    num_window_groups: int = 1,
    group_len: list[int] | None = None,
) -> None:
    """Write an individual_channel_stream (ICS).

    Supports both long blocks (window_sequence=0,1,3) and short blocks
    (window_sequence=2). For short blocks, the ICS header includes
    scale_factor_grouping and section data uses 3-bit length fields.
    """
    is_short = window_sequence == 2
    num_sfb_total = len(codebook_indices)

    if is_short:
        # Short blocks: codebook_indices has num_groups * num_sfb_per_win entries.
        # max_sfb is the highest active band across ALL groups.
        max_sfb_limit = 15
        num_sfb_per_win = num_sfb_total // max(num_window_groups, 8)
        max_sfb = 0
        for g in range(max(num_window_groups, 8)):
            for b in range(min(num_sfb_per_win, max_sfb_limit)):
                idx = g * num_sfb_per_win + b
                if idx < num_sfb_total and int(codebook_indices[idx]) != 0:
                    max_sfb = max(max_sfb, b + 1)
        max_sfb = min(max_sfb, max_sfb_limit)
    else:
        max_sfb_limit = 51  # max long-window bands across all sample rates
        max_sfb = 0
        for i in range(min(num_sfb_total, max_sfb_limit)):
            if int(codebook_indices[i]) != 0:
                max_sfb = i + 1
        max_sfb = min(max_sfb, max_sfb_limit)

    # global_gain (8 bits)
    writer.write_bits(global_gain & 0xFF, 8)

    # --- ics_info ---
    writer.write_bits(0, 1)  # ics_reserved_bit
    writer.write_bits(window_sequence, 2)  # window_sequence
    writer.write_bits(0, 1)  # window_shape: 0=sine

    if is_short:
        writer.write_bits(max_sfb, 4)  # max_sfb (4 bits for short)
        # scale_factor_grouping: 7 bits, one per window boundary.
        # Bit=1 means this window is grouped with the next.
        # For no grouping (8 independent groups): 0b0000000 = 0.
        # For all-grouped (1 group of 8): 0b1111111 = 127.
        if group_len is None:
            # Default: no grouping (8 groups of 1 window each)
            writer.write_bits(0, 7)
            num_window_groups = 8
            group_len = [1] * 8
        else:
            # Encode grouping: bit i = 1 if window i+1 is in same group as window i
            grouping_bits = 0
            win_idx = 0
            for g in range(len(group_len)):
                for w in range(1, group_len[g]):
                    grouping_bits |= 1 << (6 - (win_idx + w - 1))
                win_idx += group_len[g]
            writer.write_bits(grouping_bits, 7)
            num_window_groups = len(group_len)
    else:
        writer.write_bits(max_sfb, 6)  # max_sfb (6 bits for long)
        writer.write_bits(0, 1)  # predictor_data_present: 0
        num_window_groups = 1
        group_len = [1]

    # For short blocks, codebook_indices has num_window_groups * num_sfb_per_win
    # entries.  Slice per-group codebooks: group g → cb[g*nspw : g*nspw + max_sfb].
    num_sfb_per_win = num_sfb_total // num_window_groups if is_short else num_sfb_total

    if max_sfb > 0:
        sect_bits = 3 if is_short else 5
        sect_escape = (1 << sect_bits) - 1

        for g in range(num_window_groups):
            if is_short:
                grp_cb = codebook_indices[g * num_sfb_per_win : g * num_sfb_per_win + max_sfb]
            else:
                grp_cb = codebook_indices[:max_sfb]
            _write_section_data(writer, grp_cb, max_sfb, sect_bits, sect_escape)

        # --- scalefactor_data (all groups, sequential) ---
        active_sf = scalefactors if scalefactors is not None else None
        active_nsf = noise_scalefactors if noise_scalefactors is not None else None
        if is_short:
            total_sfb = num_window_groups * max_sfb
            # Build interleaved cb/sf arrays: [g0_b0..g0_bM, g1_b0..g1_bM, ...]
            cb_for_write = np.concatenate(
                [
                    codebook_indices[g * num_sfb_per_win : g * num_sfb_per_win + max_sfb]
                    for g in range(num_window_groups)
                ]
            )
            sf_for_write = (
                np.concatenate(
                    [
                        active_sf[g * num_sfb_per_win : g * num_sfb_per_win + max_sfb]
                        for g in range(num_window_groups)
                    ]
                )
                if active_sf is not None
                else None
            )
            _write_scalefactor_data(writer, total_sfb, cb_for_write, global_gain, sf_for_write)
        else:
            _write_scalefactor_data(
                writer,
                max_sfb,
                codebook_indices[:max_sfb],
                global_gain,
                active_sf[:max_sfb] if active_sf is not None else None,
                active_nsf[:max_sfb] if active_nsf is not None else None,
            )

    # --- pulse, TNS, gain control flags ---
    # pulse_data_present is ALWAYS written (ISO 14496-3); must be 0 for short blocks.
    writer.write_bits(0, 1)  # pulse_data_present: 0
    writer.write_bits(0, 1)  # tns_data_present: 0
    writer.write_bits(0, 1)  # gain_control_data_present: 0

    if max_sfb > 0:
        # --- spectral_data ---
        if is_short:
            for g in range(num_window_groups):
                grp_cb = codebook_indices[g * num_sfb_per_win : g * num_sfb_per_win + max_sfb]
                for w in range(group_len[g]):
                    win_idx = sum(group_len[:g]) + w
                    offset = win_idx * 128
                    win_data = quantized[offset : offset + 128]
                    _write_spectral_data(writer, win_data, grp_cb, sfb_offsets, huffman_encode_fn)
        else:
            _write_spectral_data(
                writer, quantized, codebook_indices[:max_sfb], sfb_offsets, huffman_encode_fn
            )


def _write_zero_frame_data(
    writer: BitWriter,
    num_sfb: int,
) -> None:
    """Write ICS data for an all-silent frame (max_sfb=0 or all-zero codebooks).

    For max_sfb=0: no section data, scalefactor data, or spectral data.
    For max_sfb>0 with all-zero codebooks: one section covering all bands.
    """
    if num_sfb == 0:
        # No section/scalefactor/spectral data needed.
        # However, the bit-reader in FFmpeg's decoder expects byte alignment
        # padding after the ICS header. These are effectively "don't care" bits
        # that get consumed during parsing. Write zeros for compatibility.
        return

    # One section: codebook 0, length = num_sfb
    writer.write_bits(0, 4)  # codebook 0 (ZERO_HCB)

    # Encode section length (5-bit fields, escape at 31)
    remaining = num_sfb
    while remaining >= 31:
        writer.write_bits(31, 5)
        remaining -= 31
    writer.write_bits(remaining, 5)


def _write_section_data(
    writer: BitWriter,
    codebook_indices: NDArray[np.int32],
    num_sfb: int,
    sect_bits: int = 5,
    sect_escape: int = 31,
) -> None:
    """Write section data — groups consecutive bands with the same codebook.

    For long windows, section_len uses 5-bit fields (escape=31).
    For short windows, section_len uses 3-bit fields (escape=7).
    """
    max_section_len = sect_escape
    i = 0
    while i < num_sfb:
        cb = int(codebook_indices[i])
        # Find run of bands with same codebook
        run = 1
        while i + run < num_sfb and int(codebook_indices[i + run]) == cb:
            run += 1

        # Write section: codebook (4 bits) + length encoding
        writer.write_bits(cb, 4)

        remaining = run
        while remaining >= max_section_len:
            writer.write_bits(max_section_len, sect_bits)
            remaining -= max_section_len
        writer.write_bits(remaining, sect_bits)

        i += run


def _write_scalefactor_data(
    writer: BitWriter,
    num_sfb: int,
    codebook_indices: NDArray[np.int32],
    global_gain: int = 0,
    scalefactors: NDArray[np.int32] | None = None,
    noise_scalefactors: NDArray[np.int32] | None = None,
) -> None:
    """Write scalefactor data using Huffman-encoded deltas.

    Spectral bands and noise (PNS) bands use independent delta chains.
    The first noise band writes 9 raw bits; subsequent noise bands use
    VLC deltas from the scalefactor Huffman table.
    """
    from torch_aac.config import NOISE_BT, NOISE_OFFSET, NOISE_PRE, NOISE_PRE_BITS
    from torch_aac.cpu.scalefactor import encode_scalefactor_delta

    prev_sf = global_gain
    prev_noise_sf = global_gain - NOISE_OFFSET
    is_first_noise = True

    for i in range(num_sfb):
        cb = int(codebook_indices[i])
        if cb == 0:
            continue
        if cb == NOISE_BT:
            # PNS band: write noise energy scalefactor.
            # noise_scalefactors[i] is the TARGET offset[1] value — the
            # index into pow2sf_tab (via sfo + 200) that gives the right
            # noise energy: sf = 2^(target_sfo/4), sf² = band_energy.
            target_sfo = int(noise_scalefactors[i]) if noise_scalefactors is not None else 0
            if is_first_noise:
                # First noise band: 9 raw bits.
                # Decoder: offset[1] = initial + raw - NOISE_PRE
                # where initial = global_gain - NOISE_OFFSET.
                # We want offset[1] = target_sfo, so:
                # raw = target_sfo - initial + NOISE_PRE
                initial = global_gain - NOISE_OFFSET
                raw = target_sfo - initial + NOISE_PRE
                writer.write_bits(max(0, min(511, raw)), NOISE_PRE_BITS)
                prev_noise_sf = target_sfo
                is_first_noise = False
            else:
                delta = target_sfo - prev_noise_sf
                delta = max(-60, min(60, delta))
                encode_scalefactor_delta(writer, delta)
                prev_noise_sf = prev_noise_sf + delta
        else:
            # Normal spectral band
            sf = int(scalefactors[i]) if scalefactors is not None else global_gain
            delta = sf - prev_sf
            delta = max(-60, min(60, delta))
            encode_scalefactor_delta(writer, delta)
            prev_sf = prev_sf + delta


def _write_spectral_data(
    writer: BitWriter,
    quantized: NDArray[np.int32],
    codebook_indices: NDArray[np.int32],
    sfb_offsets: list[int],
    huffman_encode_fn: object,
) -> None:
    """Write Huffman-encoded spectral coefficients.

    For each section/band, encodes the quantized coefficients using the
    selected Huffman codebook.

    Args:
        writer: BitWriter.
        quantized: Quantized spectral data, shape ``(1024,)``.
        codebook_indices: Codebook per SFB.
        sfb_offsets: SFB offset table.
        huffman_encode_fn: Callable that encodes spectral values.
    """
    num_sfb = len(codebook_indices)

    from torch_aac.config import NOISE_BT

    for i in range(num_sfb):
        cb = int(codebook_indices[i])
        if cb == 0 or cb == NOISE_BT:
            # Zero section or PNS — no spectral data
            continue

        start = sfb_offsets[i]
        end = sfb_offsets[i + 1]
        band_data = quantized[start:end]

        # Encode using the Huffman function
        # huffman_encode_fn is expected to be cpu.huffman.encode_spectral_band
        if callable(huffman_encode_fn):
            huffman_encode_fn(writer, band_data, cb)


def _write_fill_element(writer: BitWriter, num_bytes: int) -> None:
    """Write a FIL (fill) element to pad the frame.

    FIL element structure:
        - id_syn_ele:  3 bits (ID_FIL = 6)
        - count:       4 bits (0-14 payload bytes, 15 = escape)
        - [esc_count:  8 bits if count==15]
        - fill_data:   count × 8 bits (zero-filled)
    """
    # Account for the 3+4 bit overhead of the FIL element itself
    # Each fill element can carry up to 269 payload bytes (15 + 255 - 1)
    remaining = num_bytes
    while remaining > 0:
        # id_syn_ele: ID_FIL = 6
        writer.write_bits(0b110, 3)

        # Determine payload count
        if remaining <= 14:
            count = remaining
            writer.write_bits(count, 4)
        elif remaining <= 269:
            count = remaining
            writer.write_bits(15, 4)  # escape
            writer.write_bits(count - 14, 8)
        else:
            count = 269
            writer.write_bits(15, 4)
            writer.write_bits(255, 8)

        # Write fill data (zeros)
        for _ in range(count):
            writer.write_bits(0, 8)

        remaining -= count


def write_end_element(writer: BitWriter) -> None:
    """Write the END element (ID_END = 0b111, 3 bits)."""
    writer.write_bits(0b111, 3)


def build_adts_frame(
    config: EncoderConfig,
    quantized: NDArray[np.int32],
    global_gain: int,
    codebook_indices: NDArray[np.int32],
    sfb_offsets: list[int],
    huffman_encode_fn: object,
    quantized_r: NDArray[np.int32] | None = None,
    global_gain_r: int | None = None,
    codebook_indices_r: NDArray[np.int32] | None = None,
    scalefactors: NDArray[np.int32] | None = None,
    scalefactors_r: NDArray[np.int32] | None = None,
    noise_scalefactors: NDArray[np.int32] | None = None,
    noise_scalefactors_r: NDArray[np.int32] | None = None,
    window_sequence: int = 0,
) -> bytes:
    """Build a complete ADTS frame.

    Writes the ADTS header, channel element(s), end element, and byte-aligns.
    The header's frame_length field is patched after the payload is known.
    """
    # Write channel element(s) + END to measure payload size
    payload_writer = BitWriter(4096)

    if config.channels == 2 and quantized_r is not None:
        assert global_gain_r is not None
        assert codebook_indices_r is not None
        write_channel_pair_element(
            payload_writer,
            quantized,
            quantized_r,
            global_gain,
            global_gain_r,
            codebook_indices,
            codebook_indices_r,
            sfb_offsets,
            huffman_encode_fn,
            scalefactors_l=scalefactors,
            scalefactors_r=scalefactors_r,
            noise_scalefactors_l=noise_scalefactors,
            noise_scalefactors_r=noise_scalefactors_r,
            window_sequence=window_sequence,
        )
    else:
        write_single_channel_element(
            payload_writer,
            quantized,
            global_gain,
            codebook_indices,
            sfb_offsets,
            huffman_encode_fn,
            scalefactors=scalefactors,
            noise_scalefactors=noise_scalefactors,
            window_sequence=window_sequence,
        )

    write_end_element(payload_writer)
    payload_writer.align_to_byte()

    payload_bytes = payload_writer.to_bytes()
    frame_length = ADTS_HEADER_SIZE + len(payload_bytes)

    # Build complete frame: ADTS header + payload
    frame_writer = BitWriter(frame_length + 8)
    write_adts_header(frame_writer, config, frame_length)

    for byte in payload_bytes:
        frame_writer.write_bits(byte, 8)

    return frame_writer.to_bytes()[:frame_length]
