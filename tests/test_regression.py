"""Regression tests for bitstream correctness and quality baselines.

These tests use synthetic reference signals to verify:
1. Strict FFmpeg decode passes (no bitstream errors)
2. SNR meets minimum quality baselines
3. Rate control stays within tolerance

Run with: pytest tests/test_regression.py -v
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from torch_aac import AACEncoder


def _encode_decode_strict(
    pcm: np.ndarray, sr: int, bitrate: int, channels: int = 1
) -> tuple[np.ndarray, int]:
    """Encode with torch-aac, strict decode with FFmpeg.

    Returns (decoded_audio, aac_size_bytes).
    Raises AssertionError if FFmpeg strict decode fails.
    """
    enc = AACEncoder(sample_rate=sr, channels=channels, bitrate=bitrate)
    aac = enc.encode(pcm)

    with tempfile.NamedTemporaryFile(suffix=".aac", delete=False) as f:
        f.write(aac)
        aac_path = Path(f.name)
    wav_path = aac_path.with_suffix(".wav")

    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-xerror",
                "-i",
                str(aac_path),
                "-f",
                "wav",
                str(wav_path),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"FFmpeg strict decode failed at {bitrate // 1000}k: {result.stderr[:300]}"
        )
        import soundfile as sf

        decoded, _ = sf.read(str(wav_path), dtype="float32")
        if decoded.ndim > 1:
            decoded = decoded.mean(axis=1)
        return decoded, len(aac)
    finally:
        aac_path.unlink(missing_ok=True)
        wav_path.unlink(missing_ok=True)


def _compute_snr(original: np.ndarray, decoded: np.ndarray, skip: int = 2048) -> float:
    """Compute SNR in dB, skipping edge samples."""
    n = min(len(original), len(decoded)) - skip
    if n <= 0:
        return 0.0
    o = original[skip : skip + n]
    d = decoded[skip : skip + n]
    sig = np.mean(o**2)
    noise = np.mean((o - d) ** 2)
    if noise < 1e-20:
        return 100.0
    return float(10 * np.log10(sig / noise))


# ---------------------------------------------------------------------------
# Reference signals (synthetic, deterministic)
# ---------------------------------------------------------------------------
SR = 48000
DURATION = 2.0  # seconds
T = np.linspace(0, DURATION, int(SR * DURATION), dtype=np.float32)

# 1kHz sine — pure tone baseline
SINE_1K = (np.sin(2 * np.pi * 1000 * T) * 0.5).astype(np.float32)

# Multi-tone chord — tests harmonic content
CHORD = (
    np.sin(2 * np.pi * 440 * T) * 0.3
    + np.sin(2 * np.pi * 880 * T) * 0.2
    + np.sin(2 * np.pi * 1320 * T) * 0.15
    + np.sin(2 * np.pi * 1760 * T) * 0.1
).astype(np.float32)

# Speech formants — tests voiced speech-like content
SPEECH = (
    np.sin(2 * np.pi * 200 * T) * 0.3
    + np.sin(2 * np.pi * 400 * T) * 0.2
    + np.sin(2 * np.pi * 800 * T) * 0.15
    + np.sin(2 * np.pi * 1600 * T) * 0.1
    + np.sin(2 * np.pi * 3200 * T) * 0.05
).astype(np.float32)

# Transient — silence then click (triggers short blocks)
TRANSIENT = np.zeros(int(SR * DURATION), dtype=np.float32)
TRANSIENT[SR : SR + 256] = 0.95

# Deterministic noise (seeded)
_RNG = np.random.RandomState(42)
NOISE = (_RNG.randn(int(SR * DURATION)) * 0.3).astype(np.float32)


# ---------------------------------------------------------------------------
# Strict decode tests — catch bitstream corruption
# ---------------------------------------------------------------------------
class TestStrictDecode:
    """Every encoded signal must pass FFmpeg -xerror at all bitrates."""

    @pytest.mark.parametrize("bitrate", [64000, 128000, 192000])
    @pytest.mark.parametrize(
        "signal,name",
        [
            (SINE_1K, "sine"),
            (CHORD, "chord"),
            (SPEECH, "speech"),
            (TRANSIENT, "transient"),
            (NOISE, "noise"),
        ],
    )
    def test_strict_decode(self, signal: np.ndarray, name: str, bitrate: int) -> None:
        _encode_decode_strict(signal, SR, bitrate)

    @pytest.mark.parametrize("bitrate", [64000, 128000])
    def test_strict_decode_stereo(self, bitrate: int) -> None:
        stereo = np.stack([CHORD, SPEECH], axis=-1)  # (samples, 2)
        enc = AACEncoder(sample_rate=SR, channels=2, bitrate=bitrate)
        aac = enc.encode(stereo)

        with tempfile.NamedTemporaryFile(suffix=".aac", delete=False) as f:
            f.write(aac)
            aac_path = Path(f.name)
        wav_path = aac_path.with_suffix(".wav")
        try:
            r = subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-v",
                    "error",
                    "-xerror",
                    "-i",
                    str(aac_path),
                    "-f",
                    "wav",
                    str(wav_path),
                ],
                capture_output=True,
                text=True,
            )
            assert r.returncode == 0, f"Stereo strict decode failed: {r.stderr[:300]}"
        finally:
            aac_path.unlink(missing_ok=True)
            wav_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Quality baselines — catch SNR regressions
# ---------------------------------------------------------------------------
class TestQualityBaselines:
    """SNR must meet minimum thresholds on reference signals."""

    def test_sine_128k(self) -> None:
        decoded, _ = _encode_decode_strict(SINE_1K, SR, 128000)
        snr = _compute_snr(SINE_1K, decoded)
        assert snr > 40, f"Sine 128k SNR {snr:.1f} dB < 40 dB minimum"

    def test_chord_128k(self) -> None:
        decoded, _ = _encode_decode_strict(CHORD, SR, 128000)
        snr = _compute_snr(CHORD, decoded)
        assert snr > 35, f"Chord 128k SNR {snr:.1f} dB < 35 dB minimum"

    def test_speech_128k(self) -> None:
        decoded, _ = _encode_decode_strict(SPEECH, SR, 128000)
        snr = _compute_snr(SPEECH, decoded)
        assert snr > 35, f"Speech 128k SNR {snr:.1f} dB < 35 dB minimum"

    def test_noise_128k(self) -> None:
        decoded, _ = _encode_decode_strict(NOISE, SR, 128000)
        snr = _compute_snr(NOISE, decoded)
        assert snr > 5, f"Noise 128k SNR {snr:.1f} dB < 5 dB minimum"

    def test_transient_128k(self) -> None:
        """Transient signal triggers short blocks — must still meet quality bar."""
        decoded, _ = _encode_decode_strict(TRANSIENT, SR, 128000)
        # Measure SNR only in the active region
        active = TRANSIENT[SR : SR + 256]
        dec_active = decoded[SR : SR + 256] if len(decoded) > SR + 256 else decoded[-256:]
        sig = np.mean(active**2)
        noise = np.mean((active - dec_active) ** 2)
        snr = 10 * np.log10(sig / max(noise, 1e-20))
        assert snr > 5, f"Transient 128k active-region SNR {snr:.1f} dB < 5 dB"


# ---------------------------------------------------------------------------
# Rate control — catch bitrate drift
# ---------------------------------------------------------------------------
class TestRateControl:
    """Actual bitrate must be within tolerance of target."""

    @pytest.mark.parametrize("bitrate", [64000, 128000])
    def test_bitrate_accuracy(self, bitrate: int) -> None:
        """Actual bitrate should be within 30% of target on tonal signals."""
        _, aac_size = _encode_decode_strict(CHORD, SR, bitrate)
        actual_kbps = aac_size * 8 / DURATION / 1000
        target_kbps = bitrate / 1000
        error_pct = abs(actual_kbps - target_kbps) / target_kbps * 100
        assert error_pct < 30, (
            f"Bitrate error {error_pct:.0f}% at {bitrate // 1000}k "
            f"(actual={actual_kbps:.1f}k, target={target_kbps:.0f}k)"
        )
