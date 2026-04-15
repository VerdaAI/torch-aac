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

        # Use GPU Huffman when C BitWriter is available (for the batch packing).
        try:
            from torch_aac.cpu._bitwriter_native import is_available

            self._use_gpu_huffman = is_available()
        except ImportError:
            self._use_gpu_huffman = False
        # PNS: calibrated (C=64) and structurally correct, but disabled by
        # default because it replaces correlated signal with uncorrelated noise,
        # which degrades speech SNR (11.8 → 6.0 dB at 128k). Needs a spectral
        # flatness criterion to distinguish true noise from speech formants.
        # Enable for pure-noise workloads via encoder._enable_pns = True.
        self._enable_pns = False

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

    @staticmethod
    def _zero_insignificant_bands(
        mdct_coeffs: torch.Tensor, sfb_offsets: list[int], threshold: float = 0.01
    ) -> torch.Tensor:
        """Zero out MDCT bands with peak amplitude < threshold of the frame peak.

        This concentrates the bit budget on active bands. For a 10-harmonic
        signal with 49 bands, only ~10 bands are significant — the other 39
        are zeroed, freeing ~4x more bits per active band.
        """
        flat = mdct_coeffs.reshape(-1, mdct_coeffs.shape[-1])
        num_sfb = len(sfb_offsets) - 1

        # Frame-level peak amplitude
        frame_peak = flat.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-20)

        result = flat.clone()
        for i in range(num_sfb):
            s, e = sfb_offsets[i], sfb_offsets[i + 1]
            if e > 1024:
                break
            band_peak = flat[:, s:e].abs().max(dim=-1).values  # (N,)
            insignificant = band_peak < (frame_peak.squeeze(-1) * threshold)
            result[:, s:e] = result[:, s:e] * (~insignificant).unsqueeze(-1).float()

        return result.reshape_as(mdct_coeffs)

    def _encode_batch(self, batch_frames: torch.Tensor) -> list[bytes]:
        """Encode a batch of frames through the GPU+CPU pipeline.

        Args:
            batch_frames: Shape (C, B, window_length).

        Returns:
            List of ADTS frame bytes, one per frame.
        """
        C, B, _W = batch_frames.shape

        # === GPU stages ===

        # 1. Transient detection for block switching
        from torch_aac.gpu.block_switch import (
            EIGHT_SHORT_SEQUENCE,
            ONLY_LONG_SEQUENCE,
            detect_transients,
            get_window_sequence,
        )

        # Detect per-frame (use channel 0 for detection)
        is_transient = detect_transients(batch_frames[0])  # (B,)
        # State machine: run sequentially since each frame depends on previous.
        if not hasattr(self, "_prev_win_seq_state"):
            self._prev_win_seq_state = ONLY_LONG_SEQUENCE
        prev = torch.tensor([self._prev_win_seq_state], device=self._device, dtype=torch.int64)
        win_seq = torch.zeros(B, device=self._device, dtype=torch.int64)
        for i in range(B):
            ws = get_window_sequence(is_transient[i : i + 1], prev)
            win_seq[i] = ws[0]
            prev = ws
        self._prev_win_seq_state = int(win_seq[-1].item())
        # Separate long and short frames
        is_short = win_seq == EIGHT_SHORT_SEQUENCE
        short_indices = torch.where(is_short)[0]
        long_indices = torch.where(~is_short)[0]

        # 2. Apply window + MDCT
        windowed = apply_window(batch_frames, window_sequence=win_seq)  # (C, B, 2048)
        mdct_coeffs = mdct(windowed)  # (C, B, 1024)
        mdct_coeffs = mdct_coeffs.permute(1, 0, 2)  # (B, C, 1024)

        # For short blocks: compute short MDCT and replace long MDCT
        from torch_aac.gpu.filterbank import mdct_short

        if len(short_indices) > 0:
            short_frames = batch_frames[:, short_indices, :]  # (C, n_short, 2048)
            short_mdct_raw = mdct_short(short_frames)  # (C, n_short, 8, 128)
            # Flatten 8x128 to 1024 for the pipeline
            short_mdct = short_mdct_raw.reshape(C, len(short_indices), 1024)
            short_mdct = short_mdct.permute(1, 0, 2)  # (n_short, C, 1024)
            mdct_coeffs[short_indices] = short_mdct

        # 3. Rate control per-channel target
        target = torch.full(
            (B,), self._target_bits / max(C, 1), device=self._device, dtype=torch.float32
        )

        # --- Process LONG and SHORT frames with different SFB offsets ---
        from torch_aac.tables.sfb_tables import (
            get_sfb_offsets_short,
            get_sfb_offsets_short_tiled,
        )

        sfb_short = get_sfb_offsets_short(self.config.sample_rate)
        sfb_short_tiled = get_sfb_offsets_short_tiled(self.config.sample_rate)
        num_sfb_long = len(self._sfb_offsets) - 1
        num_sfb_short_tiled = len(sfb_short_tiled) - 1  # 8 * num_sfb_per_win

        # Allocate per-frame arrays (padded to max SFB count)
        max_num_sfb = max(num_sfb_long, num_sfb_short_tiled)
        global_gains = torch.zeros(B, device=self._device, dtype=torch.int64)
        scalefactors = torch.zeros(B, C, max_num_sfb, device=self._device, dtype=torch.int64)
        quantized = torch.zeros(B, C, 1024, device=self._device, dtype=mdct_coeffs.dtype)
        codebooks = torch.zeros(B, C, max_num_sfb, device=self._device, dtype=torch.int64)

        # Process LONG frames
        if len(long_indices) > 0:
            long_coeffs = mdct_coeffs[long_indices]
            long_gains = find_global_gain(long_coeffs, target[long_indices])
            long_sf = compute_scalefactors(long_coeffs, long_gains, self._sfb_offsets)
            long_q = quantize_per_band(
                long_coeffs, long_sf, self._sfb_offsets, mode=QuantMode.HARD
            ).clamp(-8191, 8191)
            long_cb = select_codebooks(
                long_q.reshape(len(long_indices) * C, -1),
                self._sfb_offsets,
                pairs_only=self._use_gpu_huffman,
            ).reshape(len(long_indices), C, -1)

            global_gains[long_indices] = long_gains
            scalefactors[long_indices, :, :num_sfb_long] = long_sf
            quantized[long_indices] = long_q
            codebooks[long_indices, :, :num_sfb_long] = long_cb

        # Process SHORT frames (tiled SFB offsets over 1024 coefficients)
        if len(short_indices) > 0:
            short_coeffs = mdct_coeffs[short_indices]
            short_gains = find_global_gain(short_coeffs, target[short_indices])
            short_sf = compute_scalefactors(short_coeffs, short_gains, sfb_short_tiled)
            short_q = quantize_per_band(
                short_coeffs, short_sf, sfb_short_tiled, mode=QuantMode.HARD
            ).clamp(-8191, 8191)
            short_cb = select_codebooks(
                short_q.reshape(len(short_indices) * C, -1),
                sfb_short_tiled,
                pairs_only=False,  # short blocks use CPU Huffman path
            ).reshape(len(short_indices), C, -1)

            global_gains[short_indices] = short_gains
            scalefactors[short_indices, :, :num_sfb_short_tiled] = short_sf
            quantized[short_indices] = short_q
            codebooks[short_indices, :, :num_sfb_short_tiled] = short_cb

        # 7. PNS: detect noise-like bands (long blocks only)
        noise_sf_np = None
        if self._enable_pns and len(long_indices) > 0:
            from torch_aac.config import NOISE_BT
            from torch_aac.gpu.pns import compute_noise_energy_sf, detect_noise_bands

            long_q_slice = quantized[long_indices]
            long_cb_slice = codebooks[long_indices, :, :num_sfb_long]
            noise_mask = detect_noise_bands(
                mdct_coeffs[long_indices], long_q_slice, long_cb_slice, self._sfb_offsets
            )
            noise_sf = compute_noise_energy_sf(
                mdct_coeffs[long_indices],
                self._sfb_offsets,
                noise_mask,
                global_gains[long_indices],
            )
            # Write back updated codebooks for long frames
            long_cb_updated = torch.where(noise_mask, NOISE_BT, long_cb_slice)
            codebooks[long_indices, :, :num_sfb_long] = long_cb_updated
            noise_sf_np = noise_sf.cpu().numpy().astype(np.int32)

        # === GPU Huffman + C bit-packing (long blocks only) ===
        has_short = is_short.any().item()
        if self._use_gpu_huffman and not has_short:
            return self._encode_batch_gpu_huffman(
                quantized,
                codebooks[:, :, :num_sfb_long],
                scalefactors[:, :, :num_sfb_long],
                global_gains,
                B,
                noise_sf_np=noise_sf_np,
            )

        # === CPU Huffman path (handles both long and short blocks) ===
        quantized_np = quantized.cpu().numpy().astype(np.int32)
        codebooks_np = codebooks.cpu().numpy().astype(np.int32)
        gains_np = global_gains.cpu().numpy().astype(np.int32)
        sf_np = scalefactors.cpu().numpy().astype(np.int32)
        win_seq_np = win_seq.cpu().numpy().astype(np.int32)

        adts_frames: list[bytes] = []

        for i in range(B):
            nsf = None
            nsf_r = None
            if noise_sf_np is not None:
                # Map long_indices back: noise_sf_np[j] corresponds to long_indices[j]
                long_idx_list = long_indices.tolist()
                if i in long_idx_list:
                    j = long_idx_list.index(i)
                    nsf = noise_sf_np[j, 0]
                    if C == 2:
                        nsf_r = noise_sf_np[j, 1]

            ws = int(win_seq_np[i])
            is_short_frame = ws == EIGHT_SHORT_SEQUENCE

            if is_short_frame:
                frame_sfb = sfb_short  # per-window offsets for bitstream spectral data
                frame_cb = codebooks_np[i, 0, :num_sfb_short_tiled]
                frame_sf = sf_np[i, 0, :num_sfb_short_tiled]
            else:
                frame_sfb = self._sfb_offsets
                frame_cb = codebooks_np[i, 0, :num_sfb_long]
                frame_sf = sf_np[i, 0, :num_sfb_long]

            if C == 1:
                frame_bytes = build_adts_frame(
                    config=self.config,
                    quantized=quantized_np[i, 0],
                    global_gain=int(gains_np[i]),
                    codebook_indices=frame_cb,
                    sfb_offsets=frame_sfb,
                    huffman_encode_fn=encode_spectral_band,
                    scalefactors=frame_sf,
                    noise_scalefactors=nsf,
                    window_sequence=ws,
                )
            else:
                if is_short_frame:
                    frame_cb_r = codebooks_np[i, 1, :num_sfb_short_tiled]
                    frame_sf_r = sf_np[i, 1, :num_sfb_short_tiled]
                else:
                    frame_cb_r = codebooks_np[i, 1, :num_sfb_long]
                    frame_sf_r = sf_np[i, 1, :num_sfb_long]

                frame_bytes = build_adts_frame(
                    config=self.config,
                    quantized=quantized_np[i, 0],
                    global_gain=int(gains_np[i]),
                    codebook_indices=frame_cb,
                    sfb_offsets=frame_sfb,
                    huffman_encode_fn=encode_spectral_band,
                    quantized_r=quantized_np[i, 1],
                    global_gain_r=int(gains_np[i]),
                    codebook_indices_r=frame_cb_r,
                    scalefactors=frame_sf,
                    scalefactors_r=frame_sf_r,
                    noise_scalefactors=nsf,
                    noise_scalefactors_r=nsf_r,
                    window_sequence=ws,
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
        noise_sf_np: np.ndarray | None = None,
    ) -> list[bytes]:
        """Fast path: batch GPU Huffman lookup + C bit-packing.

        All frames' spectral data is computed in ONE GPU gather via
        ``encode_spectral_batched``. Per-frame ICS header, section data,
        and scalefactor data are computed on CPU (fast), then everything
        is packed via the C bitwriter.
        """
        from torch_aac.cpu._bitwriter_native import bitwriter_pack
        from torch_aac.gpu.huffman_encode import encode_spectral_batched
        from torch_aac.tables.huffman_tables import SCALEFACTOR_BITS, SCALEFACTOR_CODE

        sf_code_table = SCALEFACTOR_CODE
        sf_bits_table = SCALEFACTOR_BITS

        C = quantized.shape[1]

        # --- GPU batch: spectral codes for all frames x channels at once ---
        q_flat = quantized.reshape(B * C, 1024)
        cb_flat = codebooks.reshape(B * C, -1)
        spec_codes, spec_lengths, spec_active = encode_spectral_batched(
            q_flat, cb_flat, self._sfb_offsets
        )
        # spec_codes: (N, G, 6) uint32, spec_lengths: (N, G, 6) uint8
        # spec_active: (N, G) bool

        # --- CPU: per-frame ICS header + section + sf + ADTS assembly ---
        gains_np = global_gains.cpu().numpy().astype(np.int32)
        sf_np = scalefactors.cpu().numpy().astype(np.int32)
        cb_np = codebooks.cpu().numpy().astype(np.int32)
        num_sfb = len(self._sfb_offsets) - 1
        sr_idx = self.config.sample_rate_index
        profile = AOT_AAC_LC - 1
        # Pre-allocate reusable output buffer for C bitwriter
        pack_buf = np.zeros(8192, dtype=np.uint8)

        adts_frames: list[bytes] = []

        for i in range(B):
            for ch in range(C):
                frame_idx = i * C + ch

                # Collect (code, length) pairs for this frame
                fc: list[int] = []
                fl: list[int] = []

                gg = int(gains_np[i])
                cbs = cb_np[i, ch]
                sfs = sf_np[i, ch]

                # --- ICS header ---
                max_sfb = 0
                for b in range(num_sfb):
                    if cbs[b] != 0:
                        max_sfb = b + 1
                max_sfb = min(max_sfb, 49)

                fc.append(gg)
                fl.append(8)  # global_gain
                fc.append(0)
                fl.append(1)  # reserved
                fc.append(0)
                fl.append(2)  # window_seq
                fc.append(0)
                fl.append(1)  # window_shape
                fc.append(max_sfb)
                fl.append(6)  # max_sfb
                fc.append(0)
                fl.append(1)  # predictor

                if max_sfb > 0:
                    # --- Section data ---
                    si = 0
                    while si < max_sfb:
                        cb = int(cbs[si])
                        run = 1
                        while si + run < max_sfb and int(cbs[si + run]) == cb:
                            run += 1
                        fc.append(cb)
                        fl.append(4)
                        rem = run
                        while rem >= 31:
                            fc.append(31)
                            fl.append(5)
                            rem -= 31
                        fc.append(rem)
                        fl.append(5)
                        si += run

                    # --- Scalefactor data (direct VLC code/length) ---
                    prev_sf = gg
                    prev_noise_sf = gg - 90  # NOISE_OFFSET
                    is_first_noise = True
                    nsf_arr = noise_sf_np[i, ch] if noise_sf_np is not None else None
                    for b in range(max_sfb):
                        cb_b = int(cbs[b])
                        if cb_b == 0:
                            continue
                        if cb_b == 13 and nsf_arr is not None:
                            # PNS noise band
                            target_sfo = int(nsf_arr[b])
                            if is_first_noise:
                                raw = target_sfo - (gg - 90) + 256
                                fc.append(max(0, min(511, raw)))
                                fl.append(9)
                                prev_noise_sf = target_sfo
                                is_first_noise = False
                            else:
                                delta = max(-60, min(60, target_sfo - prev_noise_sf))
                                idx = delta + 60
                                fc.append(sf_code_table[idx])
                                fl.append(sf_bits_table[idx])
                                prev_noise_sf += delta
                        else:
                            sf = int(sfs[b])
                            delta = max(-60, min(60, sf - prev_sf))
                            idx = delta + 60
                            fc.append(sf_code_table[idx])
                            fl.append(sf_bits_table[idx])
                            prev_sf += delta

                # --- Pulse/TNS/gain flags ---
                fc.append(0)
                fl.append(1)
                fc.append(0)
                fl.append(1)
                fc.append(0)
                fl.append(1)

                if max_sfb > 0:
                    # --- Spectral data (from GPU batch, numpy-vectorized extraction) ---
                    frame_c = spec_codes[frame_idx]  # (G, 6)
                    frame_l = spec_lengths[frame_idx]  # (G, 6)
                    frame_a = spec_active[frame_idx]  # (G,)
                    # Mask inactive groups, flatten, filter non-zero lengths
                    active_c = frame_c[frame_a]  # (A, 6) where A = active groups
                    active_l = frame_l[frame_a]  # (A, 6)
                    flat_c = active_c.ravel()
                    flat_l = active_l.ravel()
                    nonzero = flat_l > 0
                    fc.extend(flat_c[nonzero].tolist())
                    fl.extend(flat_l[nonzero].tolist())

                # Store this channel's ICS data
                if ch == 0:
                    ics_codes_l = fc
                    ics_lengths_l = fl
                else:
                    ics_codes_r = fc
                    ics_lengths_r = fl

            # --- Assemble ADTS frame ---
            all_codes: list[int] = []
            all_lengths: list[int] = []

            if C == 1:
                all_codes.append(0)
                all_lengths.append(7)  # SCE + tag
                all_codes.extend(ics_codes_l)
                all_lengths.extend(ics_lengths_l)
            else:
                all_codes.append(1)
                all_lengths.append(3)  # CPE id
                all_codes.append(0)
                all_lengths.append(4)  # tag
                all_codes.append(0)
                all_lengths.append(1)  # common_window=0
                all_codes.extend(ics_codes_l)
                all_lengths.extend(ics_lengths_l)
                all_codes.extend(ics_codes_r)
                all_lengths.extend(ics_lengths_r)

            all_codes.append(7)
            all_lengths.append(3)  # ID_END
            # Byte-align
            total_bits = sum(all_lengths)
            pad = (8 - total_bits % 8) % 8
            if pad:
                all_codes.append(0)
                all_lengths.append(pad)
                total_bits += pad

            payload_bytes = total_bits // 8
            frame_len = ADTS_HEADER_SIZE + payload_bytes

            codes_arr = np.array(all_codes, dtype=np.uint32)
            lengths_arr = np.array(all_lengths, dtype=np.uint8)
            # Reuse pre-allocated buffer if large enough
            if payload_bytes + 16 <= len(pack_buf):
                pack_buf[: payload_bytes + 16] = 0
                bitwriter_pack(codes_arr, lengths_arr, pack_buf)
                payload = bytes(pack_buf[:payload_bytes])
            else:
                tmp = np.zeros(payload_bytes + 16, dtype=np.uint8)
                bitwriter_pack(codes_arr, lengths_arr, tmp)
                payload = bytes(tmp[:payload_bytes])

            # ADTS header (7 bytes, inline construction)
            adts_frames.append(
                bytes(
                    [
                        0xFF,
                        0xF1,
                        (profile << 6) | (sr_idx << 2) | (C >> 2),
                        ((C & 0x3) << 6) | ((frame_len >> 11) & 0x3),
                        (frame_len >> 3) & 0xFF,
                        ((frame_len & 0x7) << 5) | 0x1F,
                        0xFC,
                    ]
                )
                + payload
            )

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
