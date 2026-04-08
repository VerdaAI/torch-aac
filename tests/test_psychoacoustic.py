"""Tests for the psychoacoustic model."""

from __future__ import annotations

import pytest
import torch

from torch_aac.gpu.psychoacoustic import (
    bark_to_hz,
    compute_masking_thresholds,
    compute_smr,
    hz_to_bark,
)
from torch_aac.tables.sfb_tables import get_sfb_offsets


class TestBarkConversion:
    def test_roundtrip(self) -> None:
        """Hz → Bark → Hz should be identity."""
        hz = torch.tensor([100.0, 500.0, 1000.0, 4000.0, 10000.0])
        bark = hz_to_bark(hz)
        recovered = bark_to_hz(bark)
        assert torch.allclose(hz, recovered, atol=1.0)

    def test_monotonic(self) -> None:
        """Higher Hz should give higher Bark."""
        hz = torch.tensor([100.0, 500.0, 1000.0, 4000.0, 10000.0])
        bark = hz_to_bark(hz)
        assert (bark[1:] > bark[:-1]).all()

    def test_known_values(self) -> None:
        """Check approximate Bark values for known frequencies."""
        hz = torch.tensor([1000.0])
        bark = hz_to_bark(hz)
        # 1kHz ≈ 8.5 Bark
        assert 7.5 < bark.item() < 9.5


class TestMaskingThresholds:
    @pytest.fixture
    def sfb_offsets_48k(self) -> list[int]:
        return get_sfb_offsets(48000)

    def test_output_shape(self, sfb_offsets_48k: list[int]) -> None:
        """Output should have one threshold per SFB."""
        num_sfb = len(sfb_offsets_48k) - 1
        coeffs = torch.randn(4, 1024)
        thresholds = compute_masking_thresholds(coeffs, 48000, sfb_offsets_48k)
        assert thresholds.shape == (4, num_sfb)

    def test_output_shape_multichannel(self, sfb_offsets_48k: list[int]) -> None:
        """Should handle (B, C, 1024) input."""
        num_sfb = len(sfb_offsets_48k) - 1
        coeffs = torch.randn(2, 2, 1024)
        thresholds = compute_masking_thresholds(coeffs, 48000, sfb_offsets_48k)
        assert thresholds.shape == (2, 2, num_sfb)

    def test_positive_thresholds(self, sfb_offsets_48k: list[int]) -> None:
        """All thresholds should be positive."""
        coeffs = torch.randn(4, 1024) * 0.5
        thresholds = compute_masking_thresholds(coeffs, 48000, sfb_offsets_48k)
        assert (thresholds > 0).all()

    def test_louder_signal_higher_threshold(self, sfb_offsets_48k: list[int]) -> None:
        """Louder signal should produce higher masking thresholds in audible bands."""
        quiet = torch.randn(4, 1024) * 0.01
        loud = torch.randn(4, 1024) * 1.0

        thresh_quiet = compute_masking_thresholds(quiet, 48000, sfb_offsets_48k)
        thresh_loud = compute_masking_thresholds(loud, 48000, sfb_offsets_48k)

        # Compare only first 30 bands (audible range, before ATH dominates)
        assert thresh_loud[:, :30].mean() > thresh_quiet[:, :30].mean()

    def test_silence_gives_ath(self, sfb_offsets_48k: list[int]) -> None:
        """Near-silent signal should give thresholds near the ATH floor."""
        silence = torch.zeros(1, 1024)
        thresholds = compute_masking_thresholds(silence, 48000, sfb_offsets_48k)
        # All thresholds should be at the ATH floor (positive but small)
        assert (thresholds > 0).all()


class TestSMR:
    @pytest.fixture
    def sfb_offsets_48k(self) -> list[int]:
        return get_sfb_offsets(48000)

    def test_output_shape(self, sfb_offsets_48k: list[int]) -> None:
        """SMR shape should match threshold shape."""
        coeffs = torch.randn(4, 1024) * 0.5
        thresholds = compute_masking_thresholds(coeffs, 48000, sfb_offsets_48k)
        smr = compute_smr(coeffs, thresholds, sfb_offsets_48k)
        assert smr.shape == thresholds.shape

    def test_smr_positive(self, sfb_offsets_48k: list[int]) -> None:
        """SMR should be non-negative."""
        coeffs = torch.randn(4, 1024) * 0.5
        thresholds = compute_masking_thresholds(coeffs, 48000, sfb_offsets_48k)
        smr = compute_smr(coeffs, thresholds, sfb_offsets_48k)
        assert (smr >= 0).all()
