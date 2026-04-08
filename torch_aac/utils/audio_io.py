"""Audio I/O utilities for reading WAV/FLAC/PCM files.

Uses soundfile (libsndfile) for robust format support.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def read_audio(
    path: str,
    target_sr: int | None = None,
) -> tuple[NDArray[np.float32], int]:
    """Read an audio file and return float32 PCM.

    Args:
        path: Path to audio file (WAV, FLAC, etc.).
        target_sr: If provided and different from file's sample rate,
            raises an error (resampling not included to keep dependencies light).

    Returns:
        Tuple of (pcm, sample_rate) where pcm is float32 array
        of shape ``(num_samples,)`` for mono or ``(num_samples, channels)``
        for multi-channel.
    """
    import soundfile as sf

    data, sr = sf.read(path, dtype="float32")

    if target_sr is not None and sr != target_sr:
        raise ValueError(
            f"Audio sample rate {sr} != target {target_sr}. "
            f"Resample before encoding or use a matching sample rate."
        )

    return data, sr


def write_audio(
    path: str,
    pcm: NDArray[np.float32],
    sample_rate: int,
) -> None:
    """Write float32 PCM to a WAV file.

    Args:
        path: Output file path.
        pcm: Float32 audio array.
        sample_rate: Sample rate in Hz.
    """
    import soundfile as sf

    sf.write(path, pcm, sample_rate, subtype="FLOAT")
