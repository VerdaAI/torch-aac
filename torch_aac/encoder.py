"""AACEncoder — the main encode-mode orchestrator.

Coordinates GPU stages (windowing, MDCT, quantization, rate control,
codebook selection) with CPU stages (Huffman packing, bitstream assembly)
to produce valid AAC-LC ADTS bitstreams.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from torch_aac.config import (
    AAC_FRAME_LENGTH,
    AAC_WINDOW_LENGTH,
    ADTS_HEADER_SIZE,
    AOT_AAC_LC,
    EncoderConfig,
    QuantMode,
)
from torch_aac.cpu.bitstream import build_adts_frame
from torch_aac.cpu.huffman import encode_spectral_band
from torch_aac.gpu.filterbank import apply_window, frame_audio, mdct
from torch_aac.gpu.huffman_encode import encode_spectral_gpu
from torch_aac.gpu.huffman_select import select_codebooks
from torch_aac.gpu.quantizer import quantize_per_band
from torch_aac.gpu.rate_control import compute_scalefactors, find_global_gain
from torch_aac.tables.sfb_tables import get_sfb_offsets


class AACEncoder:
    """GPU-accelerated AAC-LC encoder.

    Produces valid AAC-LC bitstreams in ADTS container format. Supports
    mono and stereo audio at standard sample rates.

    Example::

        with AACEncoder(sample_rate=48000, channels=2, bitrate=128000) as enc:
            aac_bytes = enc.encode(pcm_float32)
            enc.encode_file("input.wav", "output.aac")

    Args:
        sample_rate: Audio sample rate in Hz.
        channels: Number of audio channels (1 or 2).
        bitrate: Target bitrate in bps.
        device: PyTorch device ("auto", "cuda", "cpu").
        batch_size: Frames per GPU batch (0 = auto).
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        channels: int = 2,
        bitrate: int = 128000,
        device: str = "auto",
        batch_size: int = 0,
    ) -> None:
        self.config = EncoderConfig(
            sample_rate=sample_rate,
            channels=channels,
            bitrate=bitrate,
            device=device,
            batch_size=batch_size,
        )
        self._device = self.config.resolved_device
        self._sfb_offsets = get_sfb_offsets(sample_rate)
        self._target_bits = self.config.bits_per_frame

        # Auto batch size
        if batch_size == 0:
            if self._device.type == "cuda":
                props = torch.cuda.get_device_properties(self._device)
                vram_gb = props.total_mem / (1024**3)
                self._batch_size = 4096 if vram_gb < 20 else 8192
            elif self._device.type == "mps":
                # Apple Silicon unified memory. No cheap query for total
                # available memory; 1024 is a safe default on M1/M2/M3.
                self._batch_size = 1024
            else:
                self._batch_size = 256
        else:
            self._batch_size = batch_size

        # GPU Huffman path exists (gpu/huffman_encode.py) but is currently
        # slower than the CPU+C path for single-stream encoding due to Python
        # per-group overhead. It would win for multi-stream batching (100+
        # streams simultaneously) where the GPU gather amortizes launch cost.
        # For now, default to the fast CPU+C path.
        self._use_gpu_huffman = False

    def encode(self, pcm: np.ndarray | torch.Tensor) -> bytes:
        """Encode PCM audio to AAC-LC ADTS bytes.

        Args:
            pcm: Float32 audio. Shape ``(num_samples,)`` for mono or
                ``(num_samples, channels)`` / ``(channels, num_samples)``
                for stereo. Values should be in range [-1.0, 1.0].

        Returns:
            Complete AAC-LC bitstream in ADTS container.
        """
        # Convert to torch tensor
        if isinstance(pcm, np.ndarray):
            pcm_tensor = torch.from_numpy(pcm.astype(np.float32))
        else:
            pcm_tensor = pcm.float()

        # Normalize shape to (channels, num_samples)
        pcm_tensor = self._normalize_input(pcm_tensor)
        pcm_tensor = pcm_tensor.to(self._device)

        C, _T = pcm_tensor.shape
        assert self.config.channels == C, f"Expected {self.config.channels} channels, got {C}"

        # Frame the audio: (C, num_frames, window_length)
        frames = frame_audio(pcm_tensor, AAC_FRAME_LENGTH, AAC_WINDOW_LENGTH)
        # frames shape: (C, num_frames, 2048)

        num_frames = frames.shape[-2]

        # Process in batches
        all_adts_frames: list[bytes] = []

        for batch_start in range(0, num_frames, self._batch_size):
            batch_end = min(batch_start + self._batch_size, num_frames)
            batch_frames = frames[:, batch_start:batch_end, :]  # (C, B, 2048)

            adts_bytes = self._encode_batch(batch_frames)
            all_adts_frames.extend(adts_bytes)

        return b"".join(all_adts_frames)

    def encode_file(
        self,
        input_path: str | Path,
        output_path: str | Path,
    ) -> None:
        """Encode a WAV file to AAC.

        Args:
            input_path: Path to input WAV file.
            output_path: Path to output AAC file.
        """
        from torch_aac.utils.audio_io import read_audio

        pcm, _sr = read_audio(str(input_path), target_sr=self.config.sample_rate)
        aac_bytes = self.encode(pcm)

        with open(output_path, "wb") as f:
            f.write(aac_bytes)

    def _normalize_input(self, pcm: torch.Tensor) -> torch.Tensor:
        """Normalize PCM tensor to shape (channels, num_samples)."""
        if pcm.dim() == 1:
            # Mono: (T,) → (1, T)
            if self.config.channels == 1:
                return pcm.unsqueeze(0)
            raise ValueError("Got 1-D input but encoder expects stereo")

        if pcm.dim() == 2:
            # Could be (T, C) or (C, T)
            if pcm.shape[0] == self.config.channels:
                return pcm  # Already (C, T)
            if pcm.shape[1] == self.config.channels:
                return pcm.T  # (T, C) → (C, T)
            raise ValueError(
                f"Cannot infer channel layout from shape {pcm.shape} "
                f"with {self.config.channels} channels"
            )

        raise ValueError(f"Expected 1-D or 2-D PCM tensor, got {pcm.dim()}-D")

    def _encode_batch(self, batch_frames: torch.Tensor) -> list[bytes]:
        """Encode a batch of frames through the GPU+CPU pipeline.

        Args:
            batch_frames: Shape (C, B, window_length).

        Returns:
            List of ADTS frame bytes, one per frame.
        """
        C, B, _W = batch_frames.shape

        # === GPU stages ===

        # 1. Apply window
        windowed = apply_window(batch_frames)  # (C, B, 2048)

        # 2. MDCT
        mdct_coeffs = mdct(windowed)  # (C, B, 1024)

        # Rearrange to (B, C, 1024) for quantizer
        mdct_coeffs = mdct_coeffs.permute(1, 0, 2)  # (B, C, 1024)

        # 3. Rate control: find global gain per frame (uniform sf across bands).
        # The psychoacoustic-driven per-band variant in find_scalefactors
        # is experimental and not yet on the critical path — see A23 notes.
        target = torch.full(
            (B,), self._target_bits / max(C, 1), device=self._device, dtype=torch.float32
        )
        global_gains = find_global_gain(mdct_coeffs, target)
        scalefactors = compute_scalefactors(mdct_coeffs, global_gains, self._sfb_offsets)

        # 5. Quantize using per-band scalefactors
        quantized = quantize_per_band(
            mdct_coeffs, scalefactors, self._sfb_offsets, mode=QuantMode.HARD
        )
        quantized = quantized.clamp(-4095, 4095)

        # 6. Codebook selection per channel
        q_flat = quantized.reshape(B * C, -1)
        codebooks = select_codebooks(q_flat, self._sfb_offsets)
        codebooks = codebooks.reshape(B, C, -1)

        # === Mono fast path: GPU Huffman + C bit-packing ===
        if C == 1 and self._use_gpu_huffman:
            return self._encode_batch_gpu_huffman(
                quantized, codebooks, scalefactors, global_gains, B
            )

        # === Fallback: CPU Huffman (stereo, or when C extension unavailable) ===
        quantized_np = quantized.cpu().numpy().astype(np.int32)
        codebooks_np = codebooks.cpu().numpy().astype(np.int32)
        gains_np = global_gains.cpu().numpy().astype(np.int32)
        sf_np = scalefactors.cpu().numpy().astype(np.int32)

        adts_frames: list[bytes] = []

        for i in range(B):
            if C == 1:
                frame_bytes = build_adts_frame(
                    config=self.config,
                    quantized=quantized_np[i, 0],
                    global_gain=int(gains_np[i]),
                    codebook_indices=codebooks_np[i, 0],
                    sfb_offsets=self._sfb_offsets,
                    huffman_encode_fn=encode_spectral_band,
                    scalefactors=sf_np[i, 0],
                )
            else:
                frame_bytes = build_adts_frame(
                    config=self.config,
                    quantized=quantized_np[i, 0],
                    global_gain=int(gains_np[i]),
                    codebook_indices=codebooks_np[i, 0],
                    sfb_offsets=self._sfb_offsets,
                    huffman_encode_fn=encode_spectral_band,
                    quantized_r=quantized_np[i, 1],
                    global_gain_r=int(gains_np[i]),
                    codebook_indices_r=codebooks_np[i, 1],
                    scalefactors=sf_np[i, 0],
                    scalefactors_r=sf_np[i, 1],
                )
            adts_frames.append(frame_bytes)

        return adts_frames

    def _encode_batch_gpu_huffman(
        self,
        quantized: torch.Tensor,
        codebooks: torch.Tensor,
        scalefactors: torch.Tensor,
        global_gains: torch.Tensor,
        B: int,
    ) -> list[bytes]:
        """Fast path: GPU Huffman lookup + C bit-packing (mono only).

        The GPU produces (codes, lengths) arrays for each frame's ICS payload
        via vectorized table gathers. The C bitwriter_pack then packs these
        into bytes. ADTS header is prepended on CPU.
        """
        from torch_aac.cpu._bitwriter_native import bitwriter_pack

        # GPU Huffman lookup for all frames: returns list of (codes_np, lengths_np)
        ics_pairs = encode_spectral_gpu(
            quantized[:, 0, :],  # (B, 1024) mono channel
            codebooks[:, 0, :],  # (B, num_sfb)
            scalefactors[:, 0, :],  # (B, num_sfb)
            global_gains,
            self._sfb_offsets,
        )

        # ADTS header constants
        sr_idx = {
            96000: 0,
            88200: 1,
            64000: 2,
            48000: 3,
            44100: 4,
            32000: 5,
            24000: 6,
            22050: 7,
            16000: 8,
            12000: 9,
            11025: 10,
            8000: 11,
        }.get(self.config.sample_rate, 3)
        profile = AOT_AAC_LC - 1  # 1 for LC in ADTS

        adts_frames: list[bytes] = []

        for i in range(B):
            codes_np, lengths_np = ics_pairs[i]

            # Prepend element framing: ID_SCE(0, 3 bits) + tag(0, 4 bits)
            # Append: ID_END(7, 3 bits)
            n = len(codes_np)
            full_codes = np.empty(n + 2, dtype=np.uint32)
            full_lengths = np.empty(n + 2, dtype=np.uint8)
            full_codes[0] = 0  # SCE=0 (3 bits) + tag=0 (4 bits) = 7 bits as one field
            full_lengths[0] = 7
            full_codes[1 : n + 1] = codes_np
            full_lengths[1 : n + 1] = lengths_np
            full_codes[n + 1] = 7  # ID_END
            full_lengths[n + 1] = 3

            # Total payload bits → pad to byte boundary
            total_bits = int(full_lengths.sum())
            pad = (8 - total_bits % 8) % 8
            if pad > 0:
                full_codes = np.append(full_codes, np.uint32(0))
                full_lengths = np.append(full_lengths, np.uint8(pad))
                total_bits += pad

            payload_bytes = (total_bits + 7) // 8
            frame_len = ADTS_HEADER_SIZE + payload_bytes

            # Pack payload via C
            output = np.zeros(payload_bytes + 16, dtype=np.uint8)
            bitwriter_pack(full_codes, full_lengths, output)
            payload = bytes(output[:payload_bytes])

            # Build ADTS header
            header = bytearray(ADTS_HEADER_SIZE)
            header[0] = 0xFF
            header[1] = 0xF1  # sync + MPEG-4 + layer 0 + no CRC
            header[2] = (profile << 6) | (sr_idx << 2) | (0 << 1) | (1 >> 2)
            header[3] = ((1 & 0x3) << 6) | ((frame_len >> 11) & 0x3)
            header[4] = (frame_len >> 3) & 0xFF
            header[5] = ((frame_len & 0x7) << 5) | 0x1F
            header[6] = 0xFC  # buffer fullness VBR + 0 raw_data_blocks

            adts_frames.append(bytes(header) + payload)

        return adts_frames

    def close(self) -> None:
        """Release GPU resources."""
        if self._device.type == "cuda":
            torch.cuda.empty_cache()
        elif self._device.type == "mps":
            torch.mps.empty_cache()

    def __enter__(self) -> AACEncoder:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
