"""GPU device detection and memory management utilities."""

from __future__ import annotations

import torch


def get_device(device: str = "auto") -> torch.device:
    """Resolve device string to a torch.device.

    Args:
        device: Device string. "auto" selects CUDA if available.

    Returns:
        Resolved torch.device.
    """
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def get_gpu_name(device: torch.device | None = None) -> str:
    """Get the GPU name for the given device.

    Args:
        device: CUDA device. If None, uses current device.

    Returns:
        GPU name string, or "CPU" if not a CUDA device.
    """
    if device is None or device.type != "cuda":
        return "CPU"
    return torch.cuda.get_device_name(device)


def estimate_max_batch_size(
    device: torch.device,
    channels: int = 2,
    window_length: int = 2048,
    reserve_mb: int = 512,
) -> int:
    """Estimate the maximum batch size for the available GPU memory.

    Args:
        device: Target device.
        channels: Number of audio channels.
        window_length: MDCT window length.
        reserve_mb: VRAM to reserve for PyTorch overhead.

    Returns:
        Estimated maximum batch size.
    """
    if device.type != "cuda":
        return 256

    props = torch.cuda.get_device_properties(device)
    available = props.total_mem - (reserve_mb * 1024 * 1024)

    # Memory per frame estimate:
    # input(W*4) + windowed(W*4) + fft_complex(W*8) + mdct(W/2*4) + quantized(W/2*4)
    bytes_per_frame = channels * (
        window_length * 4  # input
        + window_length * 4  # windowed
        + window_length * 8  # FFT complex
        + (window_length // 2) * 4  # MDCT
        + (window_length // 2) * 4  # quantized
    )

    max_batch = int(available / bytes_per_frame)
    # Clamp to reasonable range
    return max(64, min(max_batch, 16384))
