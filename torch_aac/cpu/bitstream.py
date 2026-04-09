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


class BitWriter:
    """Bit-level writer for constructing AAC bitstreams.

    Accumulates bits and flushes to a byte buffer.
    """

    def __init__(self, capacity: int = 8192) -> None:
        self._buffer = bytearray(capacity)
        self._byte_pos = 0
        self._bit_pos = 0  # bits written in current byte (0-7)

    def write_bits(self, value: int, num_bits: int) -> None:
        """Write ``num_bits`` bits from ``value`` (MSB first).

        Args:
            value: Integer value to write. Only the lowest ``num_bits`` are used.
            num_bits: Number of bits to write (1-32).
        """
        for i in range(num_bits - 1, -1, -1):
            bit = (value >> i) & 1
            if self._byte_pos >= len(self._buffer):
                self._buffer.extend(b"\x00" * 1024)
            self._buffer[self._byte_pos] |= bit << (7 - self._bit_pos)
            self._bit_pos += 1
            if self._bit_pos == 8:
                self._bit_pos = 0
                self._byte_pos += 1

    @property
    def bits_written(self) -> int:
        """Total number of bits written."""
        return self._byte_pos * 8 + self._bit_pos

    def to_bytes(self) -> bytes:
        """Return the accumulated bytes (including partial last byte)."""
        total_bytes = self._byte_pos + (1 if self._bit_pos > 0 else 0)
        return bytes(self._buffer[:total_bytes])

    def align_to_byte(self) -> None:
        """Pad with zero bits to the next byte boundary."""
        if self._bit_pos > 0:
            self._bit_pos = 0
            self._byte_pos += 1


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
) -> None:
    """Write a single_channel_element (SCE) to the bitstream.

    SCE structure:
        - element ID:          3 bits (ID_SCE = 0x01 → actually 0b000 for first SCE)
        - element instance tag: 4 bits (0)
        - individual_channel_stream (ICS)

    Args:
        writer: BitWriter to write to.
        quantized: Quantized spectral coefficients, shape ``(1024,)``.
        global_gain: Global gain value (0-255).
        codebook_indices: Huffman codebook index per SFB, shape ``(num_sfb,)``.
        sfb_offsets: Cumulative SFB offsets.
        huffman_encode_fn: Function to encode spectral data with Huffman codes.
    """
    len(sfb_offsets) - 1

    # id_syn_ele: ID_SCE = 0 (3 bits)
    writer.write_bits(0b000, 3)
    # element_instance_tag
    writer.write_bits(0, 4)

    # --- individual_channel_stream ---
    _write_ics(writer, quantized, global_gain, codebook_indices, sfb_offsets, huffman_encode_fn)


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
) -> None:
    """Write a channel_pair_element (CPE) to the bitstream.

    CPE structure:
        - element ID:           3 bits (ID_CPE = 0b010)
        - element instance tag:  4 bits
        - common_window:         1 bit (0 = independent L/R)
        - ICS for left
        - ICS for right

    Args:
        writer: BitWriter to write to.
        quantized_l: Left channel quantized coefficients.
        quantized_r: Right channel quantized coefficients.
        global_gain_l: Left channel global gain.
        global_gain_r: Right channel global gain.
        codebook_indices_l: Left channel codebook indices.
        codebook_indices_r: Right channel codebook indices.
        sfb_offsets: Cumulative SFB offsets.
        huffman_encode_fn: Function to encode spectral data.
    """
    len(sfb_offsets) - 1

    # id_syn_ele: ID_CPE = 1 (3 bits)
    writer.write_bits(0b001, 3)
    # element_instance_tag
    writer.write_bits(0, 4)
    # common_window: 0 = independent L/R (no M/S stereo)
    writer.write_bits(0, 1)

    # Left channel ICS
    _write_ics(
        writer, quantized_l, global_gain_l, codebook_indices_l, sfb_offsets, huffman_encode_fn
    )
    # Right channel ICS
    _write_ics(
        writer, quantized_r, global_gain_r, codebook_indices_r, sfb_offsets, huffman_encode_fn
    )


def _write_ics(
    writer: BitWriter,
    quantized: NDArray[np.int32],
    global_gain: int,
    codebook_indices: NDArray[np.int32],
    sfb_offsets: list[int],
    huffman_encode_fn: object,
) -> None:
    """Write an individual_channel_stream (ICS).

    ICS structure:
        - global_gain:          8 bits
        - ics_info
        - section_data
        - scalefactor_data
        - spectral_data
    """
    num_sfb_total = len(sfb_offsets) - 1

    # Determine effective max_sfb: highest band with SIGNIFICANT content.
    # Cap at 40 bands to avoid high-frequency bands where small quantization
    # residuals produce unreliable Huffman codes. FFmpeg typically uses 22-34.
    MAX_SFB_LIMIT = 35
    max_sfb = 0
    for i in range(min(num_sfb_total, MAX_SFB_LIMIT)):
        if int(codebook_indices[i]) != 0:
            max_sfb = i + 1
    max_sfb = min(max_sfb, MAX_SFB_LIMIT)

    # global_gain (8 bits)
    writer.write_bits(global_gain & 0xFF, 8)

    # --- ics_info ---
    writer.write_bits(0, 1)       # ics_reserved_bit
    writer.write_bits(0, 2)       # window_sequence: ONLY_LONG_SEQUENCE
    writer.write_bits(1, 1)       # window_shape: sine

    writer.write_bits(max_sfb, 6) # max_sfb (only bands with content)
    writer.write_bits(0, 1)       # predictor_data_present: 0

    if max_sfb == 0:
        # No spectral content: just write trailer bits
        pass
    else:
        # Use only the first max_sfb bands for section/sf/spectral data
        active_codebooks = codebook_indices[:max_sfb]

        # --- section_data ---
        _write_section_data(writer, active_codebooks, max_sfb)

        # --- scalefactor_data ---
        _write_scalefactor_data(writer, max_sfb, active_codebooks)

    # --- pulse, TNS, gain control flags (BEFORE spectral data!) ---
    # Per ISO 14496-3 and FFmpeg's aacdec.c, these flags come after
    # scalefactor_data but BEFORE spectral_data.
    writer.write_bits(0, 1)       # pulse_data_present: 0
    writer.write_bits(0, 1)       # tns_data_present: 0
    writer.write_bits(0, 1)       # gain_control_data_present: 0

    if max_sfb > 0:
        # --- spectral_data (AFTER pulse/tns/gain flags) ---
        _write_spectral_data(writer, quantized, active_codebooks, sfb_offsets, huffman_encode_fn)


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
) -> None:
    """Write section data — groups consecutive bands with the same codebook.

    For long windows, section_len uses 5-bit fields. A section_len value of
    31 (0b11111) is an escape meaning "continue to next field."
    """
    max_section_len = 31  # 2^5 - 1 for long blocks
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
            writer.write_bits(max_section_len, 5)
            remaining -= max_section_len
        writer.write_bits(remaining, 5)

        i += run


def _write_scalefactor_data(
    writer: BitWriter,
    num_sfb: int,
    codebook_indices: NDArray[np.int32],
) -> None:
    """Write scalefactor data using Huffman-encoded deltas.

    With uniform global gain, all scalefactors are equal, so all deltas are 0.
    The scalefactor Huffman code for delta=0 is a single "0" bit.
    """
    from torch_aac.cpu.scalefactor import encode_scalefactor_delta

    for i in range(num_sfb):
        if int(codebook_indices[i]) == 0:
            # Zero section — no scalefactor needed
            continue
        # Delta = 0 for uniform gain
        encode_scalefactor_delta(writer, 0)


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

    for i in range(num_sfb):
        cb = int(codebook_indices[i])
        if cb == 0:
            # Zero section — no spectral data
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
) -> bytes:
    """Build a complete ADTS frame.

    Writes the ADTS header, channel element(s), end element, and byte-aligns.
    The header's frame_length field is patched after the payload is known.

    Args:
        config: Encoder configuration.
        quantized: Left (or mono) quantized spectral data, shape ``(1024,)``.
        global_gain: Left/mono global gain.
        codebook_indices: Left/mono codebook indices per SFB.
        sfb_offsets: SFB offsets.
        huffman_encode_fn: Function to encode spectral data.
        quantized_r: Right channel data (for stereo).
        global_gain_r: Right channel gain (for stereo).
        codebook_indices_r: Right channel codebooks (for stereo).

    Returns:
        Complete ADTS frame as bytes.
    """
    # Write channel element(s) + END to measure payload size
    payload_writer = BitWriter(4096)

    if config.channels == 2 and quantized_r is not None:
        assert global_gain_r is not None
        assert codebook_indices_r is not None
        write_channel_pair_element(
            payload_writer,
            quantized, quantized_r,
            global_gain, global_gain_r,
            codebook_indices, codebook_indices_r,
            sfb_offsets,
            huffman_encode_fn,
        )
    else:
        write_single_channel_element(
            payload_writer,
            quantized, global_gain, codebook_indices,
            sfb_offsets, huffman_encode_fn,
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
