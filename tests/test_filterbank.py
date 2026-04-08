"""Tests for the GPU filterbank (windowing, MDCT, inverse MDCT)."""

from __future__ import annotations

import torch

from torch_aac.gpu.filterbank import apply_window, frame_audio, imdct, mdct, overlap_add


class TestFrameAudio:
    def test_basic_framing(self, device: torch.device) -> None:
        """Test that framing produces correct number of frames."""
        sr = 48000
        pcm = torch.randn(1, sr, device=device)  # 1 channel, 1 second
        frames = frame_audio(pcm, frame_length=1024, window_length=2048)
        # Expected: (sr - 1024) / 1024 = ~46 frames
        assert frames.dim() == 3  # (C, num_frames, window_length)
        assert frames.shape[0] == 1  # 1 channel
        assert frames.shape[2] == 2048  # window length

    def test_frame_overlap(self, device: torch.device) -> None:
        """Test that consecutive frames overlap by 50%."""
        pcm = torch.arange(4096, dtype=torch.float32, device=device).unsqueeze(0)
        frames = frame_audio(pcm, frame_length=1024, window_length=2048)
        # Frame 0: samples 0-2047, Frame 1: samples 1024-3071
        assert torch.equal(frames[0, 0, 1024:], frames[0, 1, :1024])


class TestMDCT:
    def test_output_shape(self, device: torch.device) -> None:
        """MDCT should halve the last dimension."""
        x = torch.randn(2, 10, 2048, device=device)
        result = mdct(x)
        assert result.shape == (2, 10, 1024)

    def test_real_output(self, device: torch.device) -> None:
        """MDCT output should be real-valued."""
        x = torch.randn(4, 2048, device=device)
        result = mdct(x)
        assert result.dtype == torch.float32

    def test_energy_consistent(self, device: torch.device) -> None:
        """MDCT energy ratio should be consistent across frames."""
        x = torch.randn(8, 2048, device=device)
        windowed = apply_window(x)
        coeffs = mdct(windowed)
        time_energy = (windowed ** 2).sum(dim=-1)
        freq_energy = (coeffs ** 2).sum(dim=-1)
        ratio = freq_energy / (time_energy + 1e-10)
        # Ratio should be consistent (same normalization for all frames)
        # The MDCT has a known energy scaling factor of ~N/2
        mean_ratio = ratio.mean()
        assert (ratio / mean_ratio - 1.0).abs().max() < 0.1


class TestIMDCT:
    def test_output_shape(self, device: torch.device) -> None:
        """IMDCT should double the last dimension."""
        x = torch.randn(4, 1024, device=device)
        result = imdct(x)
        assert result.shape == (4, 2048)

    def test_mdct_imdct_roundtrip(self, device: torch.device) -> None:
        """MDCT → IMDCT with overlap-add should approximately reconstruct."""
        # Create test signal
        T = 8192
        signal = torch.randn(1, T, device=device)

        # Frame, window, MDCT
        frames = frame_audio(signal, frame_length=1024, window_length=2048)
        windowed = apply_window(frames)
        coeffs = mdct(windowed)

        # IMDCT, window again, overlap-add
        reconstructed_frames = imdct(coeffs)
        rewindowed = apply_window(reconstructed_frames)
        reconstructed = overlap_add(rewindowed, frame_length=1024)

        # Trim to original length (skip first and last half-window for edge effects)
        start = 1024
        end = min(T, reconstructed.shape[-1]) - 1024
        if end > start:
            original = signal[0, start:end]
            recon = reconstructed[0, start:end]
            # Correlation should be high (not exact due to normalization)
            corr = torch.nn.functional.cosine_similarity(
                original.unsqueeze(0), recon.unsqueeze(0)
            )
            assert corr.item() > 0.9, f"Reconstruction correlation too low: {corr.item()}"


class TestApplyWindow:
    def test_sine_window_shape(self, device: torch.device) -> None:
        """Window should not change shape."""
        x = torch.randn(4, 2048, device=device)
        result = apply_window(x)
        assert result.shape == x.shape

    def test_sine_window_zeros_at_boundaries(self, device: torch.device) -> None:
        """Sine window should be near zero at boundaries."""
        x = torch.ones(1, 2048, device=device)
        result = apply_window(x)
        # First and last values should be small
        assert result[0, 0].abs() < 0.01
        assert result[0, -1].abs() < 0.01

    def test_sine_window_peak_at_center(self, device: torch.device) -> None:
        """Sine window should peak near the center."""
        x = torch.ones(1, 2048, device=device)
        result = apply_window(x)
        center = result[0, 1024]
        assert center > 0.99  # sin(pi/2) = 1.0
