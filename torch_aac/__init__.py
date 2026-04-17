"""torch-aac: GPU-accelerated and differentiable AAC-LC encoder.

The first open-source GPU-accelerated AAC-LC encoder with differentiable
mode for training codec-robust audio models.

Quick start::

    import torch_aac

    # Encode audio to AAC
    aac_bytes = torch_aac.encode(pcm_float32, sample_rate=48000, bitrate=128000)

    # Differentiable mode for training
    codec = torch_aac.DifferentiableAAC(sample_rate=48000, bitrate=128000)
    decoded = codec(audio_tensor)  # gradients flow through
    loss.backward()
"""

from __future__ import annotations

from torch_aac.config import EncoderConfig, QuantMode
from torch_aac.encoder import AACEncoder

__version__ = "0.6.0"
__all__ = [
    "AACEncoder",
    "DifferentiableAAC",
    "EncoderConfig",
    "QuantMode",
    "encode",
    "encode_batch",
    "encode_file",
]


def encode(
    pcm: object,
    sample_rate: int = 48000,
    channels: int | None = None,
    bitrate: int = 128000,
    device: str = "auto",
) -> bytes:
    """One-shot encode PCM audio to AAC-LC ADTS bytes.

    Args:
        pcm: Float32 audio as numpy array or torch tensor.
            Shape ``(num_samples,)`` for mono or ``(num_samples, channels)``
            for multi-channel.
        sample_rate: Audio sample rate in Hz.
        channels: Number of channels. If None, inferred from pcm shape.
        bitrate: Target bitrate in bps.
        device: PyTorch device string.

    Returns:
        AAC-LC bitstream in ADTS container format.
    """
    import numpy as np
    import torch

    # Infer channels from input shape
    if channels is None:
        if isinstance(pcm, np.ndarray):
            channels = 1 if pcm.ndim == 1 else min(pcm.shape)
        elif isinstance(pcm, torch.Tensor):
            channels = 1 if pcm.dim() == 1 else min(pcm.shape)
        else:
            channels = 1

    with AACEncoder(
        sample_rate=sample_rate,
        channels=channels,
        bitrate=bitrate,
        device=device,
    ) as enc:
        return enc.encode(pcm)


def encode_batch(
    pcm_list: list[object],
    sample_rate: int = 48000,
    channels: int = 1,
    bitrate: int = 128000,
    device: str = "auto",
    max_workers: int | None = None,
) -> list[bytes]:
    """Encode multiple audio streams to AAC in parallel.

    GPU stages (MDCT, quantization, Huffman lookup) are shared across all
    streams. CPU stages (bitstream packing) are parallelized across
    ``max_workers`` processes.

    Args:
        pcm_list: List of float32 audio arrays (numpy or torch). Each can
            be a different length. All must have the same sample rate,
            channel count, and target bitrate.
        sample_rate: Audio sample rate in Hz.
        channels: Number of channels per stream.
        bitrate: Target bitrate in bps.
        device: PyTorch device string.
        max_workers: Number of parallel workers for CPU packing. ``None``
            uses ``min(len(pcm_list), os.cpu_count())``.

    Returns:
        List of AAC-LC bytes, one per input stream.

    Example::

        import numpy as np, torch_aac

        streams = [np.random.randn(48000).astype(np.float32) for _ in range(8)]
        aac_list = torch_aac.encode_batch(streams, sample_rate=48000, bitrate=128000)
    """
    from concurrent.futures import ThreadPoolExecutor

    if not pcm_list:
        return []

    if max_workers is None:
        # ThreadPool scales to ~2 workers due to the Python GIL.
        # The C BitWriter releases the GIL during byte-packing, giving
        # partial parallelism, but the Python per-frame assembly is
        # GIL-bound. 2 workers is the sweet spot in benchmarks.
        max_workers = min(len(pcm_list), 2)

    if len(pcm_list) <= 1 or max_workers <= 1:
        return [
            encode(pcm, sample_rate=sample_rate, channels=channels, bitrate=bitrate, device=device)
            for pcm in pcm_list
        ]

    def _encode_one(pcm_item: object) -> bytes:
        return encode(
            pcm_item, sample_rate=sample_rate, channels=channels, bitrate=bitrate, device=device
        )

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(_encode_one, pcm_list))


def encode_file(
    input_path: str,
    output_path: str,
    bitrate: int = 128000,
    device: str = "auto",
) -> None:
    """One-shot encode a WAV file to AAC.

    Args:
        input_path: Path to input WAV file.
        output_path: Path to output AAC file.
        bitrate: Target bitrate in bps.
        device: PyTorch device string.
    """
    from torch_aac.utils.audio_io import read_audio

    pcm, sr = read_audio(input_path)
    channels = 1 if pcm.ndim == 1 else pcm.shape[-1]

    with AACEncoder(
        sample_rate=sr,
        channels=channels,
        bitrate=bitrate,
        device=device,
    ) as enc:
        enc.encode_file(input_path, output_path)


# Lazy import for DifferentiableAAC to avoid circular imports
def __getattr__(name: str) -> object:
    if name == "DifferentiableAAC":
        from torch_aac.differentiable import DifferentiableAAC

        return DifferentiableAAC
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
