"""End-to-end encoder tests.

Tests that the encoder produces valid AAC-LC ADTS bitstreams that can
be decoded by FFmpeg.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from torch_aac import AACEncoder, encode


class TestEncoderBasic:
    def test_encode_mono_returns_bytes(self, sine_wave_mono: np.ndarray) -> None:
        """Encoding mono audio should return non-empty bytes."""
        result = encode(sine_wave_mono, sample_rate=48000, bitrate=128000, device="cpu")
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_encode_stereo_returns_bytes(self, sine_wave_stereo: np.ndarray) -> None:
        """Encoding stereo audio should return non-empty bytes."""
        result = encode(sine_wave_stereo, sample_rate=48000, bitrate=128000, device="cpu")
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_adts_sync_word(self, sine_wave_mono: np.ndarray) -> None:
        """Output should start with ADTS sync word 0xFFF."""
        result = encode(sine_wave_mono, sample_rate=48000, bitrate=128000, device="cpu")
        # First 12 bits should be 0xFFF
        assert len(result) >= 2
        sync = (result[0] << 4) | (result[1] >> 4)
        assert sync == 0xFFF, f"Expected ADTS sync 0xFFF, got 0x{sync:03X}"

    def test_adts_header_fields(self, sine_wave_mono: np.ndarray) -> None:
        """Verify key ADTS header fields."""
        result = encode(sine_wave_mono, sample_rate=48000, bitrate=128000, device="cpu")
        assert len(result) >= 7

        # Byte 0-1: sync word (12 bits) + ID (1) + layer (2) + protection (1)
        # ID=0 (MPEG-4), layer=0, protection_absent=1
        assert (result[1] & 0x0F) == 0x01  # ID=0, layer=00, protection=1

        # Byte 2: profile(2) + sf_index(4) + private(1) + channel_config_hi(1)
        profile = (result[2] >> 6) & 0x03
        assert profile == 1, f"Expected AAC-LC profile (1), got {profile}"

        sf_index = (result[2] >> 2) & 0x0F
        assert sf_index == 3, f"Expected sf_index 3 (48kHz), got {sf_index}"

    def test_short_audio(self, short_mono: np.ndarray) -> None:
        """Should handle very short audio without errors."""
        result = encode(short_mono, sample_rate=48000, bitrate=128000, device="cpu")
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_silence(self) -> None:
        """Should handle silence input."""
        silence = np.zeros(48000, dtype=np.float32)
        result = encode(silence, sample_rate=48000, bitrate=128000, device="cpu")
        assert isinstance(result, bytes)
        assert len(result) > 0


class TestEncoderWithFFmpeg:
    """Tests that require FFmpeg to be installed."""

    @pytest.fixture
    def ffmpeg_available(self) -> bool:
        """Check if FFmpeg is available."""
        try:
            subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                check=True,
            )
            return True
        except (FileNotFoundError, subprocess.CalledProcessError):
            return False

    def test_ffmpeg_can_decode(
        self,
        sine_wave_mono: np.ndarray,
        ffmpeg_available: bool,
    ) -> None:
        """FFmpeg should be able to decode our output."""
        if not ffmpeg_available:
            pytest.skip("FFmpeg not available")

        aac_bytes = encode(sine_wave_mono, sample_rate=48000, bitrate=128000, device="cpu")

        with tempfile.NamedTemporaryFile(suffix=".aac", delete=False) as f:
            f.write(aac_bytes)
            aac_path = f.name

        try:
            # Try to decode with FFmpeg
            result = subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", aac_path,
                    "-f", "f32le",
                    "-acodec", "pcm_f32le",
                    "pipe:1",
                ],
                capture_output=True,
                timeout=30,
            )
            # FFmpeg should not error out
            assert result.returncode == 0, (
                f"FFmpeg decode failed: {result.stderr.decode()}"
            )
            # Should produce some PCM output
            assert len(result.stdout) > 0, "FFmpeg produced no output"

            # Parse decoded PCM
            decoded = np.frombuffer(result.stdout, dtype=np.float32)
            assert len(decoded) > 0, "Decoded PCM is empty"
        finally:
            Path(aac_path).unlink(missing_ok=True)

    def test_ffprobe_validates_format(
        self,
        sine_wave_mono: np.ndarray,
        ffmpeg_available: bool,
    ) -> None:
        """ffprobe should recognize the output as valid AAC."""
        if not ffmpeg_available:
            pytest.skip("FFmpeg not available")

        aac_bytes = encode(sine_wave_mono, sample_rate=48000, bitrate=128000, device="cpu")

        with tempfile.NamedTemporaryFile(suffix=".aac", delete=False) as f:
            f.write(aac_bytes)
            aac_path = f.name

        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v", "error",
                    "-show_entries", "stream=codec_name,sample_rate,channels",
                    "-of", "csv=p=0",
                    aac_path,
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            assert result.returncode == 0, f"ffprobe failed: {result.stderr}"
            output = result.stdout.strip()
            assert "aac" in output.lower(), f"Not recognized as AAC: {output}"
        finally:
            Path(aac_path).unlink(missing_ok=True)


class TestEncoderContext:
    def test_context_manager(self, sine_wave_mono: np.ndarray) -> None:
        """Encoder should work as context manager."""
        with AACEncoder(sample_rate=48000, channels=1, bitrate=128000, device="cpu") as enc:
            result = enc.encode(sine_wave_mono)
            assert len(result) > 0

    def test_multiple_encodes(self, sine_wave_mono: np.ndarray) -> None:
        """Encoder should support multiple encode calls."""
        with AACEncoder(sample_rate=48000, channels=1, bitrate=128000, device="cpu") as enc:
            result1 = enc.encode(sine_wave_mono)
            result2 = enc.encode(sine_wave_mono)
            assert len(result1) > 0
            assert len(result2) > 0
            # Same input should produce same output (deterministic)
            assert result1 == result2
