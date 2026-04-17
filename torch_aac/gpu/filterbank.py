"""GPU-accelerated windowing, MDCT, and inverse MDCT for AAC-LC.

The MDCT (Modified Discrete Cosine Transform) is the core transform in AAC.
For long blocks, it transforms 2048 time-domain samples into 1024 frequency
coefficients, with 50% overlap between consecutive frames.

Implementation uses a pre-computed cosine basis matrix for batch processing
on GPU. The matmul approach is efficient for large batches and correct by
construction. A future version may use FFT-based tricks for even higher
throughput.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from torch_aac.tables.window_tables import sine_window


def frame_audio(
    pcm: torch.Tensor,
    frame_length: int = 1024,
    window_length: int = 2048,
) -> torch.Tensor:
    """Split PCM audio into overlapping frames for MDCT.

    Uses 50% overlap: each frame is ``window_length`` samples, advancing by
    ``frame_length`` samples.

    Args:
        pcm: Audio tensor of shape ``(channels, num_samples)`` or
            ``(batch, channels, num_samples)``.
        frame_length: Hop size (number of new samples per frame). Default 1024.
        window_length: Total window length. Default 2048.

    Returns:
        Framed audio of shape ``(..., num_frames, window_length)``.
    """
    if pcm.dim() == 2:
        pcm = pcm.unsqueeze(0)  # (1, C, T)

    B, C, T = pcm.shape
    # Pad to at least one full window, then to complete frames
    overlap = window_length - frame_length
    if window_length > T:
        pcm = F.pad(pcm, (0, window_length - T))
        T = window_length
    pad_len = (frame_length - ((T - overlap) % frame_length)) % frame_length
    if pad_len > 0:
        pcm = F.pad(pcm, (0, pad_len))
    T_padded = pcm.shape[-1]

    num_frames = (T_padded - overlap) // frame_length

    # Use unfold for efficient overlapping frame extraction
    # Reshape to (B*C, T) for unfold
    flat = pcm.reshape(B * C, T_padded)
    frames = flat.unfold(dimension=-1, size=window_length, step=frame_length)
    # frames: (B*C, num_frames, window_length)
    frames = frames.reshape(B, C, num_frames, window_length)

    return frames.squeeze(0) if B == 1 else frames


def apply_window(
    frames: torch.Tensor,
    window_type: str = "sine",
    window_sequence: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply a window function to framed audio.

    When ``window_sequence`` is provided, applies per-frame transition windows
    for LONG_START (ws=1) and LONG_STOP (ws=3) frames. EIGHT_SHORT (ws=2) and
    ONLY_LONG (ws=0) frames use the regular sine window.

    Args:
        frames: Framed audio of shape ``(C, B, window_length)`` or
            ``(..., window_length)``.
        window_type: Window type. Currently ``"sine"`` is supported.
        window_sequence: Per-frame window sequence, shape ``(B,)``. If None,
            applies the standard sine window to all frames.

    Returns:
        Windowed frames, same shape as input.
    """
    window_length = frames.shape[-1]
    if window_type != "sine":
        raise ValueError(f"Unknown window type: {window_type!r}")

    device = frames.device
    window = sine_window(window_length, device=device)

    if window_sequence is None:
        return frames * window

    # Fast path: skip per-frame logic when no transition frames
    has_transitions = ((window_sequence == 1) | (window_sequence == 3)).any().item()
    if not has_transitions:
        return frames * window

    from torch_aac.tables.window_tables import long_start_window, long_stop_window

    result = frames * window  # default: sine window for all
    win_start = long_start_window(window_length, device=device)
    win_stop = long_stop_window(window_length, device=device)

    # Apply transition windows per-frame.
    # frames is (C, B, W), window_sequence is (B,).
    for ws_val, win in [(1, win_start), (3, win_stop)]:
        mask = window_sequence == ws_val  # (B,)
        if mask.any():
            # Broadcast: (C, B, W) * (1, B, W) where mask selects B
            result[:, mask, :] = frames[:, mask, :] * win.unsqueeze(0)

    return result


def _get_mdct_basis(N: int, device: torch.device) -> torch.Tensor:
    """Get or compute the MDCT cosine basis matrix.

    Args:
        N: Window length.
        device: Target device.

    Returns:
        Cosine basis matrix of shape ``(N, N/2)``.
    """
    M = N // 2
    n0 = M / 2.0 + 0.5  # = (M + 1) / 2
    n = torch.arange(N, device=device, dtype=torch.float32)
    k = torch.arange(M, device=device, dtype=torch.float32)
    # basis[n, k] = cos(pi/M * (n + n0) * (k + 0.5))
    return torch.cos(math.pi / M * torch.outer(n + n0, k + 0.5))


# Cache basis matrices to avoid recomputation
_basis_cache: dict[tuple[int, torch.device], torch.Tensor] = {}


def _cached_basis(N: int, device: torch.device) -> torch.Tensor:
    """Return cached MDCT basis matrix, computing on first call."""
    key = (N, device)
    if key not in _basis_cache:
        _basis_cache[key] = _get_mdct_basis(N, device)
    return _basis_cache[key]


def mdct(windowed_frames: torch.Tensor) -> torch.Tensor:
    """Compute the MDCT of windowed frames.

    Uses the standard MDCT definition:
        X(k) = sum_{n=0}^{N-1} x(n) * cos(pi/M * (n + n0) * (k + 0.5))
    where M = N/2 and n0 = (M+1)/2.

    Implemented as a matrix multiply with a pre-computed cosine basis,
    which is efficient for large batches on GPU.

    Args:
        windowed_frames: Windowed audio of shape ``(..., N)`` where N is the
            window length (must be even).

    Returns:
        MDCT coefficients of shape ``(..., N/2)``.
    """
    N = windowed_frames.shape[-1]
    basis = _cached_basis(N, windowed_frames.device)
    # (..., N) @ (N, M) → (..., M)
    return torch.matmul(windowed_frames, basis)


def mdct_short(frames_2048: torch.Tensor) -> torch.Tensor:
    """Compute 8 short-block MDCTs from a 2048-sample frame.

    Extracts 8 overlapping 256-sample sub-windows from the center of the
    2048-sample frame, applies a sine window, and computes 128-point MDCT
    for each.

    Args:
        frames_2048: Input frames, shape ``(..., 2048)``.

    Returns:
        Short-block MDCT coefficients, shape ``(..., 8, 128)``.
    """
    device = frames_2048.device
    # Sub-window positions within the 2048-sample frame.
    # Per AAC spec: short windows are at offset 448 + w*128 for w=0..7
    # Each sub-window is 256 samples with 128-sample overlap.
    short_win = sine_window(256, device=device)
    basis = _cached_basis(256, device)  # (256, 128)

    results = []
    for w in range(8):
        start = 448 + w * 128
        sub = frames_2048[..., start : start + 256]  # (..., 256)
        windowed = sub * short_win
        coeffs = torch.matmul(windowed, basis)  # (..., 128)
        results.append(coeffs)

    return torch.stack(results, dim=-2)  # (..., 8, 128)


def imdct(mdct_coeffs: torch.Tensor) -> torch.Tensor:
    """Compute the inverse MDCT.

    Produces N time-domain samples from N/2 spectral coefficients.
    The output must be windowed and overlap-added for perfect reconstruction.

    IMDCT(n) = (2/M) * sum_{k=0}^{M-1} X(k) * cos(pi/M * (n + n0) * (k + 0.5))

    Args:
        mdct_coeffs: MDCT coefficients of shape ``(..., M)`` where M = N/2.

    Returns:
        Time-domain signal of shape ``(..., N)`` where N = 2*M.
    """
    M = mdct_coeffs.shape[-1]
    N = M * 2
    basis = _cached_basis(N, mdct_coeffs.device)  # (N, M)
    # (..., M) @ (M, N) → (..., N)
    return torch.matmul(mdct_coeffs, basis.T) * (2.0 / M)


def imdct_short(short_coeffs: torch.Tensor) -> torch.Tensor:
    """Inverse MDCT for short blocks: 8 × 128 coefficients → 2048-sample frame.

    Performs 8 separate 128-point IMDCTs, windows each sub-block with the
    short sine window, and overlap-adds them within the 2048-sample frame
    at the same positions used by ``mdct_short`` (offset 448 + w*128).

    This matches the real AAC decoder's short-block reconstruction.

    Args:
        short_coeffs: Shape ``(..., 8, 128)``.

    Returns:
        Time-domain frame, shape ``(..., 2048)``.
    """
    batch_shape = short_coeffs.shape[:-2]
    device = short_coeffs.device
    short_win = sine_window(256, device=device)
    basis = _cached_basis(256, device)  # (256, 128)

    output = torch.zeros(*batch_shape, 2048, device=device, dtype=short_coeffs.dtype)

    for w in range(8):
        coeffs_w = short_coeffs[..., w, :]  # (..., 128)
        # 128-point IMDCT → 256 samples
        td = torch.matmul(coeffs_w, basis.T) * (2.0 / 128)  # (..., 256)
        # Window
        td = td * short_win
        # Overlap-add at the sub-window position
        start = 448 + w * 128
        output[..., start : start + 256] += td

    return output


def overlap_add(
    windowed_frames: torch.Tensor,
    frame_length: int = 1024,
) -> torch.Tensor:
    """Reconstruct audio from overlapping windowed frames via overlap-add.

    Args:
        windowed_frames: Windowed frames of shape ``(..., num_frames, window_length)``.
        frame_length: Hop size.

    Returns:
        Reconstructed audio of shape ``(..., num_samples)``.
    """
    *batch_dims, num_frames, window_length = windowed_frames.shape
    overlap = window_length - frame_length
    total_length = num_frames * frame_length + overlap

    # Flatten batch dims for processing
    flat = windowed_frames.reshape(-1, num_frames, window_length)
    B = flat.shape[0]

    output = torch.zeros(B, total_length, device=flat.device, dtype=flat.dtype)
    for i in range(num_frames):
        start = i * frame_length
        output[:, start : start + window_length] += flat[:, i, :]

    return output.reshape(*batch_dims, total_length)
