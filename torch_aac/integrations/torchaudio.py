"""torchaudio-compatible transforms and utilities for torch-aac.

Drop-in augmentation for training pipelines:

    from torch_aac.integrations.torchaudio import AACSimulation

    augment = AACSimulation(sample_rate=48000, bitrate=128000)
    coded = augment(waveform)   # differentiable!
    loss = F.mse_loss(coded, target)
    loss.backward()             # gradients flow through AAC

Save/load AAC files with torchaudio-like API:

    from torch_aac.integrations.torchaudio import save_aac, load_aac

    save_aac("output.aac", waveform, 48000, bitrate=128000)
    waveform, sr = load_aac("output.aac")
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import torch_aac
from torch_aac.differentiable import DifferentiableAAC


class AACSimulation(nn.Module):
    """Differentiable AAC compression as a torchaudio-compatible transform.

    Simulates the full AAC encode-decode pipeline with gradient flow.
    Use as a data augmentation in training to make models robust to AAC.

    Compatible with ``torch.nn.Sequential`` and ``torchaudio.transforms``.

    Args:
        sample_rate: Audio sample rate in Hz.
        bitrate: Target AAC bitrate in bps.
        quant_mode: ``"ste"`` (straight-through estimator), ``"noise"``
            (additive uniform noise), or ``"cubic"`` (soft-rounding with real
            gradients). STE is recommended for most training; cubic is useful
            for warm-up before switching to STE.
        device: PyTorch device. ``"auto"`` selects CUDA > MPS > CPU.

    Example::

        import torchaudio.transforms as T
        from torch_aac.integrations.torchaudio import AACSimulation

        augment = torch.nn.Sequential(
            T.Resample(16000, 48000),
            AACSimulation(sample_rate=48000, bitrate=128000),
            T.Resample(48000, 16000),
        )
        robust_audio = augment(waveform)
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        bitrate: int = 128000,
        quant_mode: str = "ste",
        device: str = "auto",
    ) -> None:
        super().__init__()
        self.codec = DifferentiableAAC(
            sample_rate=sample_rate,
            bitrate=bitrate,
            channels=1,
            quant_mode=quant_mode,
            device=device,
        )
        self._sample_rate = sample_rate
        self._bitrate = bitrate

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply AAC compression simulation.

        Args:
            waveform: Audio tensor. Accepts shapes:
                - ``(num_samples,)`` — mono
                - ``(channels, num_samples)`` — multi-channel
                - ``(batch, channels, num_samples)`` — batched

        Returns:
            Compressed audio, same shape as input. Gradients flow through.
        """
        return self.codec(waveform)

    def extra_repr(self) -> str:
        return f"sr={self._sample_rate}, bitrate={self._bitrate // 1000}k"


class AACEncode(nn.Module):
    """Non-differentiable AAC encoding as a torchaudio-compatible transform.

    Produces actual AAC bytes (not a simulation). Useful for evaluation
    pipelines where you want real codec artifacts, not a differentiable
    approximation.

    Args:
        sample_rate: Audio sample rate in Hz.
        bitrate: Target AAC bitrate in bps.
        device: PyTorch device for the encoder.

    Example::

        from torch_aac.integrations.torchaudio import AACEncode

        encoder = AACEncode(sample_rate=48000, bitrate=128000)
        decoded = encoder(waveform)  # encode → FFmpeg decode
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        bitrate: int = 128000,
        device: str = "auto",
    ) -> None:
        super().__init__()
        self._sample_rate = sample_rate
        self._bitrate = bitrate
        self._device = device

    @torch.no_grad()
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Encode to AAC and decode back.

        Args:
            waveform: ``(num_samples,)`` or ``(channels, num_samples)``.
                Values in [-1, 1].

        Returns:
            Decoded audio tensor, same shape (may differ slightly in length
            due to MDCT framing).
        """
        pcm = waveform.detach().cpu().numpy()
        aac_bytes = torch_aac.encode(
            pcm,
            sample_rate=self._sample_rate,
            bitrate=self._bitrate,
            device=self._device,
        )

        # Decode with FFmpeg
        with tempfile.NamedTemporaryFile(suffix=".aac", delete=False) as f:
            f.write(aac_bytes)
            p = f.name
        try:
            channels = 1 if waveform.dim() == 1 else waveform.shape[0]
            r = subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-v",
                    "error",
                    "-i",
                    p,
                    "-f",
                    "f32le",
                    "-ar",
                    str(self._sample_rate),
                    "-ac",
                    str(channels),
                    "pipe:1",
                ],
                capture_output=True,
            )
        finally:
            Path(p).unlink(missing_ok=True)

        decoded = np.frombuffer(r.stdout, dtype=np.float32)
        result = torch.from_numpy(decoded.copy())

        # Reshape to match input
        if waveform.dim() == 2:
            result = result.reshape(-1, channels).T if channels > 1 else result.unsqueeze(0)

        # Trim/pad to match input length
        target_len = waveform.shape[-1]
        if result.shape[-1] > target_len:
            result = result[..., :target_len]
        elif result.shape[-1] < target_len:
            pad = target_len - result.shape[-1]
            result = torch.nn.functional.pad(result, (0, pad))

        return result.to(waveform.device)

    def extra_repr(self) -> str:
        return f"sr={self._sample_rate}, bitrate={self._bitrate // 1000}k"


def save_aac(
    filepath: str,
    waveform: torch.Tensor,
    sample_rate: int,
    bitrate: int = 128000,
    device: str = "auto",
) -> None:
    """Save a waveform to an AAC file (torchaudio.save-like API).

    Args:
        filepath: Output path (should end in ``.aac``).
        waveform: Audio tensor, shape ``(channels, num_samples)`` or
            ``(num_samples,)``. Values in [-1, 1].
        sample_rate: Sample rate in Hz.
        bitrate: Target bitrate in bps.
        device: PyTorch device for encoding.

    Example::

        save_aac("output.aac", waveform, 48000, bitrate=128000)
    """
    pcm = waveform.detach().cpu().numpy()
    channels = 1 if waveform.dim() == 1 else waveform.shape[0]
    aac_bytes = torch_aac.encode(
        pcm,
        sample_rate=sample_rate,
        channels=channels,
        bitrate=bitrate,
        device=device,
    )
    Path(filepath).write_bytes(aac_bytes)


def load_aac(
    filepath: str,
    sample_rate: int | None = None,
) -> tuple[torch.Tensor, int]:
    """Load an AAC file to a waveform tensor (torchaudio.load-like API).

    Uses FFmpeg for decoding. Returns mono audio at the file's native sample
    rate (or resampled if ``sample_rate`` is specified).

    Args:
        filepath: Path to ``.aac`` file.
        sample_rate: Target sample rate. ``None`` = use file's native rate.

    Returns:
        Tuple of (waveform, sample_rate). Waveform shape: ``(1, num_samples)``.

    Example::

        waveform, sr = load_aac("input.aac")
    """
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", filepath]
    if sample_rate:
        cmd += ["-ar", str(sample_rate)]
    cmd += ["-f", "f32le", "-ac", "1", "pipe:1"]

    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"FFmpeg decode failed: {r.stderr.decode()[:200]}")

    pcm = np.frombuffer(r.stdout, dtype=np.float32)
    waveform = torch.from_numpy(pcm.copy()).unsqueeze(0)  # (1, T)

    # Detect sample rate if not specified
    if sample_rate is None:
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "stream=sample_rate",
                "-of",
                "csv=p=0",
                filepath,
            ],
            capture_output=True,
        )
        sample_rate = int(probe.stdout.decode().strip()) if probe.returncode == 0 else 48000

    return waveform, sample_rate
