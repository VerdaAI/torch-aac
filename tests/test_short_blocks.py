"""Tests for short block (EIGHT_SHORT_SEQUENCE) support.

Validates transient detection, block switching, transition windows,
short block encoding, and pre-echo reduction.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch

from torch_aac import AACEncoder
from torch_aac.gpu.block_switch import (
    EIGHT_SHORT_SEQUENCE,
    LONG_START_SEQUENCE,
    LONG_STOP_SEQUENCE,
    ONLY_LONG_SEQUENCE,
    detect_transients,
    get_window_sequence,
)
from torch_aac.tables.sfb_tables import get_sfb_offsets_short, get_sfb_offsets_short_tiled
from torch_aac.tables.window_tables import long_start_window, long_stop_window, sine_window


class TestTransientDetection:
    """Test transient detection on crafted signals."""

    def test_silence_no_transient(self) -> None:
        frames = torch.zeros(10, 2048)
        assert not detect_transients(frames).any()

    def test_steady_tone_no_transient(self) -> None:
        t = torch.linspace(0, 2 * torch.pi * 440 * 10 * 2048 / 48000, 10 * 2048)
        frames = torch.sin(t).reshape(10, 2048)
        assert not detect_transients(frames).any()

    def test_impulse_triggers_transient(self) -> None:
        frames = torch.zeros(5, 2048)
        # Put energy in later sub-windows of frame 2
        frames[2, 1024:1280] = 0.95
        assert detect_transients(frames)[2].item()


class TestBlockSwitching:
    """Test window sequence state machine transitions."""

    def test_long_to_short_transition(self) -> None:
        """Transient after ONLY_LONG should produce LONG_START."""
        is_trans = torch.tensor([True])
        prev = torch.tensor([ONLY_LONG_SEQUENCE])
        ws = get_window_sequence(is_trans, prev)
        assert ws.item() == LONG_START_SEQUENCE

    def test_long_start_to_eight_short(self) -> None:
        """Continued transient after LONG_START should produce EIGHT_SHORT."""
        is_trans = torch.tensor([True])
        prev = torch.tensor([LONG_START_SEQUENCE])
        ws = get_window_sequence(is_trans, prev)
        assert ws.item() == EIGHT_SHORT_SEQUENCE

    def test_eight_short_to_long_stop(self) -> None:
        """Non-transient after EIGHT_SHORT should produce LONG_STOP."""
        is_trans = torch.tensor([False])
        prev = torch.tensor([EIGHT_SHORT_SEQUENCE])
        ws = get_window_sequence(is_trans, prev)
        assert ws.item() == LONG_STOP_SEQUENCE

    def test_long_stop_to_only_long(self) -> None:
        """Non-transient after LONG_STOP should return to ONLY_LONG."""
        is_trans = torch.tensor([False])
        prev = torch.tensor([LONG_STOP_SEQUENCE])
        ws = get_window_sequence(is_trans, prev)
        assert ws.item() == ONLY_LONG_SEQUENCE

    def test_full_transition_sequence(self) -> None:
        """Test ONLY_LONG -> LONG_START -> EIGHT_SHORT -> LONG_STOP -> ONLY_LONG."""
        transients = [True, True, False, False]
        expected = [
            LONG_START_SEQUENCE,
            EIGHT_SHORT_SEQUENCE,
            LONG_STOP_SEQUENCE,
            ONLY_LONG_SEQUENCE,
        ]
        prev = torch.tensor([ONLY_LONG_SEQUENCE])
        for is_trans, exp in zip(transients, expected, strict=True):
            ws = get_window_sequence(torch.tensor([is_trans]), prev)
            assert ws.item() == exp
            prev = ws


class TestTransitionWindows:
    """Test LONG_START and LONG_STOP window functions."""

    def test_long_start_left_half_matches_sine(self) -> None:
        sw = sine_window(2048)
        lsw = long_start_window(2048)
        assert torch.allclose(lsw[:1024], sw[:1024])

    def test_long_start_flat_region(self) -> None:
        lsw = long_start_window(2048)
        assert (lsw[1024:1472] == 1.0).all()

    def test_long_start_trailing_zeros(self) -> None:
        lsw = long_start_window(2048)
        assert (lsw[1600:] == 0.0).all()

    def test_long_stop_is_mirror_of_start(self) -> None:
        lsw = long_start_window(2048)
        lstop = long_stop_window(2048)
        assert torch.allclose(lstop, lsw.flip(0))

    def test_long_stop_leading_zeros(self) -> None:
        lstop = long_stop_window(2048)
        assert (lstop[:448] == 0.0).all()


class TestTiledSFBOffsets:
    """Test the tiled SFB offset tables for short blocks."""

    def test_tiled_offsets_count(self) -> None:
        short = get_sfb_offsets_short(48000)
        tiled = get_sfb_offsets_short_tiled(48000)
        num_sfb_per_win = len(short) - 1
        assert len(tiled) - 1 == 8 * num_sfb_per_win

    def test_tiled_offsets_range(self) -> None:
        tiled = get_sfb_offsets_short_tiled(48000)
        assert tiled[0] == 0
        assert tiled[-1] == 1024

    def test_tiled_offsets_monotonic(self) -> None:
        tiled = get_sfb_offsets_short_tiled(48000)
        for i in range(len(tiled) - 1):
            assert tiled[i] < tiled[i + 1]

    def test_tiled_offsets_all_sample_rates(self) -> None:
        for sr in [8000, 16000, 22050, 24000, 32000, 44100, 48000]:
            tiled = get_sfb_offsets_short_tiled(sr)
            assert tiled[-1] == 1024


class TestShortBlockEncoding:
    """End-to-end tests for short block encoding."""

    @staticmethod
    def _make_transient_signal(sr: int = 48000) -> np.ndarray:
        """Create a signal with a clear transient: silence then click."""
        samples = sr  # 1 second
        pcm = np.zeros(samples, dtype=np.float32)
        click_pos = samples // 2
        pcm[click_pos : click_pos + 256] = 0.95
        return pcm

    def test_short_block_ffmpeg_decode(self) -> None:
        """Encoded short blocks must be decodable by FFmpeg."""
        pcm = self._make_transient_signal()
        enc = AACEncoder(sample_rate=48000, channels=1, bitrate=128000)
        aac_bytes = enc.encode(pcm)

        with tempfile.NamedTemporaryFile(suffix=".aac", delete=False) as f:
            f.write(aac_bytes)
            aac_path = Path(f.name)
        wav_path = aac_path.with_suffix(".wav")

        try:
            result = subprocess.run(
                ["ffmpeg", "-y", "-i", str(aac_path), "-f", "wav", str(wav_path)],
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, f"FFmpeg failed: {result.stderr[-300:]}"
        finally:
            aac_path.unlink(missing_ok=True)
            wav_path.unlink(missing_ok=True)

    def test_short_block_strict_decode(self) -> None:
        """Short blocks must pass FFmpeg strict decode (-xerror) with no errors."""
        pcm = self._make_transient_signal()
        enc = AACEncoder(sample_rate=48000, channels=1, bitrate=128000)
        aac_bytes = enc.encode(pcm)

        with tempfile.NamedTemporaryFile(suffix=".aac", delete=False) as f:
            f.write(aac_bytes)
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
            assert result.returncode == 0, f"Strict decode failed: {result.stderr[-300:]}"
        finally:
            aac_path.unlink(missing_ok=True)
            wav_path.unlink(missing_ok=True)

    def test_short_block_stereo_ffmpeg_decode(self) -> None:
        """Stereo short blocks must be decodable by FFmpeg."""
        mono = self._make_transient_signal()
        stereo = np.stack([mono, mono * 0.8], axis=-1)  # (samples, 2)
        enc = AACEncoder(sample_rate=48000, channels=2, bitrate=128000)
        aac_bytes = enc.encode(stereo)

        with tempfile.NamedTemporaryFile(suffix=".aac", delete=False) as f:
            f.write(aac_bytes)
            aac_path = Path(f.name)
        wav_path = aac_path.with_suffix(".wav")

        try:
            result = subprocess.run(
                ["ffmpeg", "-y", "-i", str(aac_path), "-f", "wav", str(wav_path)],
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, f"FFmpeg failed: {result.stderr[-300:]}"
        finally:
            aac_path.unlink(missing_ok=True)
            wav_path.unlink(missing_ok=True)

    def test_pre_echo_reduction(self) -> None:
        """Short blocks should reduce pre-echo compared to long blocks."""
        import soundfile as sf

        sr = 48000
        pcm = np.zeros(sr * 2, dtype=np.float32)
        click_pos = sr
        pcm[click_pos] = 1.0
        pcm[click_pos + 1] = -1.0

        def _encode_decode(force_long: bool) -> np.ndarray:
            enc = AACEncoder(sample_rate=sr, channels=1, bitrate=64000)
            if force_long:
                import torch_aac.gpu.block_switch as bs

                old = bs.detect_transients
                bs.detect_transients = lambda f, threshold=10.0: torch.zeros(
                    f.shape[:-1], dtype=torch.bool, device=f.device
                )
                aac = enc.encode(pcm)
                bs.detect_transients = old
            else:
                aac = enc.encode(pcm)

            with tempfile.NamedTemporaryFile(suffix=".aac", delete=False) as f:
                f.write(aac)
                path = Path(f.name)
            wav = path.with_suffix(".wav")
            try:
                subprocess.run(
                    ["ffmpeg", "-y", "-i", str(path), "-f", "wav", str(wav)],
                    capture_output=True,
                    check=True,
                )
                decoded, _ = sf.read(str(wav))
                return decoded
            finally:
                path.unlink(missing_ok=True)
                wav.unlink(missing_ok=True)

        dec_long = _encode_decode(force_long=True)
        dec_short = _encode_decode(force_long=False)

        # Measure peak pre-echo in 1024 samples before click
        pe_start = click_pos - 1024
        pe_end = click_pos - 10
        peak_long = np.max(np.abs(dec_long[pe_start:pe_end]))
        peak_short = np.max(np.abs(dec_short[pe_start:pe_end]))

        # Short blocks should have less pre-echo (confine noise to shorter windows)
        assert peak_short < peak_long * 0.75, (
            f"Short block pre-echo ({peak_short:.6f}) should be less than "
            f"long block ({peak_long:.6f})"
        )
