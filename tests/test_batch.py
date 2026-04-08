"""Tests for batch encoding functionality."""

from __future__ import annotations

import numpy as np

from torch_aac import AACEncoder, encode


class TestBatchEncoding:
    def test_multiple_files_same_encoder(self) -> None:
        """Encoder should handle multiple sequential encode calls."""
        with AACEncoder(sample_rate=48000, channels=1, bitrate=128000, device="cpu") as enc:
            results = []
            for _ in range(5):
                pcm = np.random.randn(48000).astype(np.float32) * 0.3
                aac = enc.encode(pcm)
                results.append(aac)
                assert len(aac) > 0

            # All should produce valid ADTS
            for aac in results:
                assert aac[0] == 0xFF
                assert (aac[1] & 0xF0) == 0xF0

    def test_various_lengths(self) -> None:
        """Encoder should handle various audio lengths."""
        for duration_sec in [0.05, 0.1, 0.5, 1.0, 2.0]:
            num_samples = int(48000 * duration_sec)
            pcm = np.random.randn(num_samples).astype(np.float32) * 0.3
            aac = encode(pcm, sample_rate=48000, bitrate=128000, device="cpu")
            assert len(aac) > 0

    def test_stereo_encoding(self) -> None:
        """Stereo encoding should work."""
        pcm = np.random.randn(48000, 2).astype(np.float32) * 0.3
        aac = encode(pcm, sample_rate=48000, channels=2, bitrate=128000, device="cpu")
        assert len(aac) > 0

    def test_different_sample_rates(self) -> None:
        """Should handle 16kHz and 44.1kHz."""
        for sr in [16000, 44100, 48000]:
            pcm = np.random.randn(sr).astype(np.float32) * 0.3  # 1 second
            aac = encode(pcm, sample_rate=sr, bitrate=128000, device="cpu")
            assert len(aac) > 0

            # Verify sample rate in ADTS header
            sf_idx = (aac[2] >> 2) & 0x0F
            expected_idx = {16000: 8, 44100: 4, 48000: 3}[sr]
            assert sf_idx == expected_idx, f"Wrong sf_idx for {sr}Hz: {sf_idx}"

    def test_different_bitrates(self) -> None:
        """Should handle various bitrates."""
        pcm = np.random.randn(48000).astype(np.float32) * 0.3
        for bitrate in [48000, 96000, 128000, 192000, 256000, 320000]:
            aac = encode(pcm, sample_rate=48000, bitrate=bitrate, device="cpu")
            assert len(aac) > 0

    def test_deterministic(self) -> None:
        """Same input should always produce same output."""
        pcm = np.random.RandomState(42).randn(48000).astype(np.float32) * 0.3
        result1 = encode(pcm, sample_rate=48000, bitrate=128000, device="cpu")
        result2 = encode(pcm, sample_rate=48000, bitrate=128000, device="cpu")
        assert result1 == result2


class TestEdgeCases:
    def test_very_short_audio(self) -> None:
        """Audio shorter than one frame should still produce output."""
        pcm = np.random.randn(100).astype(np.float32) * 0.3
        aac = encode(pcm, sample_rate=48000, bitrate=128000, device="cpu")
        assert len(aac) > 0

    def test_silence(self) -> None:
        """Pure silence should produce valid output."""
        pcm = np.zeros(48000, dtype=np.float32)
        aac = encode(pcm, sample_rate=48000, bitrate=128000, device="cpu")
        assert len(aac) > 0

    def test_clipped_audio(self) -> None:
        """Audio with values > 1.0 should not crash."""
        pcm = np.random.randn(48000).astype(np.float32) * 2.0  # clipped
        aac = encode(pcm, sample_rate=48000, bitrate=128000, device="cpu")
        assert len(aac) > 0

    def test_dc_offset(self) -> None:
        """Audio with DC offset should not crash."""
        pcm = np.ones(48000, dtype=np.float32) * 0.5
        aac = encode(pcm, sample_rate=48000, bitrate=128000, device="cpu")
        assert len(aac) > 0
