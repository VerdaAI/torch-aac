"""DifferentiableAAC — differentiable mode for training codec-robust models.

Simulates AAC encoding/decoding as a differentiable PyTorch operation.
Gradients flow through the entire pipeline via straight-through estimator
(STE), noise injection, or cubic soft-rounding for the quantization step.

Matches the real encoder's behavior including short-block switching for
transient audio (silence→impulse, drum hits, plosives).

Example::

    codec = DifferentiableAAC(sample_rate=48000, bitrate=128000)
    audio = model(input)
    decoded = codec(audio)
    loss = F.mse_loss(decoded, target)
    loss.backward()  # gradients flow through AAC simulation
"""

from __future__ import annotations

import torch
import torch.nn as nn

from torch_aac.config import AAC_FRAME_LENGTH, AAC_WINDOW_LENGTH, EncoderConfig, QuantMode
from torch_aac.gpu.filterbank import apply_window, frame_audio, imdct, mdct, overlap_add
from torch_aac.gpu.quantizer import dequantize, estimate_bit_count, quantize
from torch_aac.gpu.rate_control import find_global_gain


class DifferentiableAAC(nn.Module):
    """Differentiable AAC-LC simulation for training.

    Simulates the encode→decode pipeline as a differentiable operation.
    No actual bitstream is produced; instead, the quantization and
    reconstruction artifacts are applied to the audio tensor.

    Includes transient detection and short-block MDCT switching to match
    the real encoder's behavior on impulsive audio.

    Args:
        sample_rate: Audio sample rate in Hz.
        bitrate: Target bitrate in bps (affects quantization coarseness).
        channels: Number of audio channels.
        quant_mode: Quantization mode (``"ste"``, ``"noise"``, or ``"cubic"``).
        device: PyTorch device.
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        bitrate: int = 128000,
        channels: int = 1,
        quant_mode: str = "ste",
        device: str = "auto",
    ) -> None:
        super().__init__()
        self.config = EncoderConfig(
            sample_rate=sample_rate,
            channels=channels,
            bitrate=bitrate,
            device=device,
            quant_mode={"ste": QuantMode.STE, "noise": QuantMode.NOISE, "cubic": QuantMode.CUBIC}[
                quant_mode
            ],
        )
        self._device = self.config.resolved_device
        self._target_bits = self.config.bits_per_frame

    def forward(
        self,
        audio: torch.Tensor,
        return_rate_loss: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Simulate AAC encode→decode on audio.

        Args:
            audio: Input audio tensor, shape ``(batch, channels, num_samples)``
                or ``(channels, num_samples)`` or ``(num_samples,)``.
            return_rate_loss: If True, also return a differentiable rate loss
                that encourages the output to fit the target bitrate.

        Returns:
            Reconstructed audio tensor (same shape as input), or
            tuple of (reconstructed, rate_loss) if return_rate_loss=True.
        """
        original_dim = audio.dim()

        # Normalize to (B, C, T)
        if audio.dim() == 1:
            audio = audio.unsqueeze(0).unsqueeze(0)  # (1, 1, T)
        elif audio.dim() == 2:
            audio = audio.unsqueeze(0)  # (1, C, T)

        B, C, T = audio.shape

        # frame_audio expects (C, T) per batch item; process per batch
        all_reconstructed = []
        all_q: list[torch.Tensor] = []

        for b in range(B):
            frames = frame_audio(audio[b], AAC_FRAME_LENGTH, AAC_WINDOW_LENGTH)
            # frames: (C, num_frames, 2048)
            num_frames = frames.shape[1]

            # --- Transient detection & block switching (non-differentiable) ---
            with torch.no_grad():
                from torch_aac.gpu.block_switch import (
                    EIGHT_SHORT_SEQUENCE,
                    LONG_START_SEQUENCE,
                    LONG_STOP_SEQUENCE,
                    ONLY_LONG_SEQUENCE,
                    detect_transients,
                )

                # Detect on all channels, OR together
                is_transient = detect_transients(frames[0])
                for ch in range(1, C):
                    is_transient = is_transient | detect_transients(frames[ch])

                # State machine (pure Python)
                prev = ONLY_LONG_SEQUENCE
                ws_list = []
                for trans in is_transient.tolist():
                    was_long = prev in (ONLY_LONG_SEQUENCE, LONG_STOP_SEQUENCE)
                    if was_long:
                        ws = LONG_START_SEQUENCE if trans else ONLY_LONG_SEQUENCE
                    else:
                        ws = EIGHT_SHORT_SEQUENCE if trans else LONG_STOP_SEQUENCE
                    ws_list.append(ws)
                    prev = ws
                win_seq = torch.tensor(ws_list, device=audio.device, dtype=torch.int64)
                is_short = win_seq == EIGHT_SHORT_SEQUENCE
                short_indices = torch.where(is_short)[0]

            # --- Window (with transition windows for START/STOP) ---
            windowed = apply_window(frames, window_sequence=win_seq)

            # --- MDCT ---
            coeffs = mdct(windowed)  # (C, num_frames, 1024)

            # Replace long MDCT with short MDCT for transient frames
            if len(short_indices) > 0:
                from torch_aac.gpu.filterbank import mdct_short

                short_frames = frames[:, short_indices, :]
                short_mdct = mdct_short(short_frames)  # (C, n_short, 8, 128)
                short_flat = short_mdct.reshape(C, len(short_indices), 1024)
                coeffs[:, short_indices, :] = short_flat

            # Rearrange to (num_frames, C, 1024)
            coeffs_bc = coeffs.permute(1, 0, 2)

            # --- Rate control (detached) ---
            with torch.no_grad():
                target = torch.full(
                    (num_frames,),
                    self._target_bits / max(C, 1),
                    device=audio.device,
                    dtype=torch.float32,
                )
                gains = find_global_gain(coeffs_bc, target, quant_mode=QuantMode.HARD)

            # --- Differentiable quantize + dequantize ---
            q = quantize(coeffs_bc, gains, mode=self.config.quant_mode)
            reconstructed_coeffs = dequantize(q, gains)
            if return_rate_loss:
                all_q.append(q)

            # Back to (C, num_frames, 1024)
            reconstructed_coeffs = reconstructed_coeffs.permute(1, 0, 2)

            # --- Inverse MDCT ---
            # Long frames: standard 1024-point IMDCT → 2048 samples
            time_domain = imdct(reconstructed_coeffs)  # (C, num_frames, 2048)

            # --- Synthesis windowing ---
            # Long/transition frames get the synthesis window (sine or transition).
            # Short frames use imdct_short which applies its own sub-window OLA —
            # no additional long window (the real decoder works the same way).
            rewindowed = apply_window(time_domain, window_sequence=win_seq)

            # Short frames: replace with 8x128-point IMDCT + sub-window OLA
            # (done AFTER windowing so the long window doesn't corrupt them)
            if len(short_indices) > 0:
                from torch_aac.gpu.filterbank import imdct_short

                short_rc = reconstructed_coeffs[:, short_indices, :]  # (C, n, 1024)
                short_rc_8x128 = short_rc.reshape(C, len(short_indices), 8, 128)
                short_td = imdct_short(short_rc_8x128)  # (C, n, 2048)
                rewindowed[:, short_indices, :] = short_td

            reconstructed = overlap_add(rewindowed, frame_length=AAC_FRAME_LENGTH)

            # Trim to original length
            reconstructed = reconstructed[:, :T]
            all_reconstructed.append(reconstructed)

        result = torch.stack(all_reconstructed, dim=0)  # (B, C, T)

        # Restore original shape
        if original_dim == 1:
            result = result.squeeze(0).squeeze(0)
        elif original_dim == 2:
            result = result.squeeze(0)

        if return_rate_loss:
            q_cat = torch.cat([qb.reshape(-1, qb.shape[-1]) for qb in all_q], dim=0)
            bits_per_frame = estimate_bit_count(q_cat)
            target = max(self._target_bits / max(C, 1), 1.0)
            rate_loss = bits_per_frame.mean() / target
            return result, rate_loss

        return result
