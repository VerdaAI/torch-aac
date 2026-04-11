"""Tests for the differentiable AAC mode.

Verifies that gradients flow through the AAC simulation pipeline
and that the output is a reasonable approximation of real AAC.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from torch_aac.differentiable import DifferentiableAAC


class TestDifferentiableForward:
    def test_output_shape_mono(self) -> None:
        """Output shape should match input for mono."""
        codec = DifferentiableAAC(sample_rate=48000, bitrate=128000, channels=1, device="cpu")
        audio = torch.randn(1, 1, 48000)
        output = codec(audio)
        assert output.shape == audio.shape

    def test_output_shape_1d(self) -> None:
        """Should handle 1-D input."""
        codec = DifferentiableAAC(sample_rate=48000, bitrate=128000, channels=1, device="cpu")
        audio = torch.randn(4800)
        output = codec(audio)
        assert output.shape == audio.shape

    def test_output_shape_2d(self) -> None:
        """Should handle 2-D input (channels, samples)."""
        codec = DifferentiableAAC(sample_rate=48000, bitrate=128000, channels=1, device="cpu")
        audio = torch.randn(1, 48000)
        output = codec(audio)
        assert output.shape == audio.shape

    def test_output_shape_stereo(self) -> None:
        """Output shape should match input for stereo."""
        codec = DifferentiableAAC(sample_rate=48000, bitrate=128000, channels=2, device="cpu")
        audio = torch.randn(1, 2, 48000)
        output = codec(audio)
        assert output.shape == audio.shape

    def test_output_is_finite(self) -> None:
        """Output should not contain NaN or Inf."""
        codec = DifferentiableAAC(sample_rate=48000, bitrate=128000, channels=1, device="cpu")
        audio = torch.randn(1, 1, 48000) * 0.5
        output = codec(audio)
        assert torch.isfinite(output).all()

    def test_silence_produces_near_silence(self) -> None:
        """Silent input should produce near-silent output."""
        codec = DifferentiableAAC(sample_rate=48000, bitrate=128000, channels=1, device="cpu")
        audio = torch.zeros(1, 1, 4800)
        output = codec(audio)
        assert output.abs().max() < 0.01


class TestDifferentiableGradients:
    def test_gradient_flows_ste(self) -> None:
        """Gradients should flow through in STE mode."""
        codec = DifferentiableAAC(
            sample_rate=48000,
            bitrate=128000,
            channels=1,
            quant_mode="ste",
            device="cpu",
        )
        audio = torch.randn(1, 1, 4800, requires_grad=True)
        output = codec(audio)
        loss = output.sum()
        loss.backward()
        assert audio.grad is not None
        assert audio.grad.shape == audio.shape
        assert not torch.all(audio.grad == 0), "Gradients should be non-zero"

    def test_gradient_flows_noise(self) -> None:
        """Gradients should flow through in noise mode."""
        codec = DifferentiableAAC(
            sample_rate=48000,
            bitrate=128000,
            channels=1,
            quant_mode="noise",
            device="cpu",
        )
        audio = torch.randn(1, 1, 4800, requires_grad=True)
        output = codec(audio)
        loss = output.sum()
        loss.backward()
        assert audio.grad is not None
        assert not torch.all(audio.grad == 0), "Gradients should be non-zero"

    def test_gradient_magnitude_reasonable(self) -> None:
        """Gradients should not explode or vanish."""
        codec = DifferentiableAAC(
            sample_rate=48000,
            bitrate=128000,
            channels=1,
            quant_mode="ste",
            device="cpu",
        )
        audio = (torch.randn(4800) * 0.5).detach().requires_grad_(True)
        output = codec(audio)
        loss = (output**2).mean()
        loss.backward()
        assert audio.grad is not None, "Gradient is None"
        grad_mean = audio.grad.abs().mean().item()
        assert 1e-8 < grad_mean < 100, f"Gradient magnitude {grad_mean} out of range"

    def test_gradient_changes_with_input(self) -> None:
        """Different inputs should produce different gradients."""
        codec = DifferentiableAAC(
            sample_rate=48000,
            bitrate=128000,
            channels=1,
            quant_mode="ste",
            device="cpu",
        )

        audio1 = (torch.randn(4800) * 0.1).detach().requires_grad_(True)
        output1 = codec(audio1)
        output1.sum().backward()

        audio2 = (torch.randn(4800) * 0.8).detach().requires_grad_(True)
        output2 = codec(audio2)
        output2.sum().backward()

        assert audio1.grad is not None and audio2.grad is not None
        assert not torch.allclose(audio1.grad, audio2.grad)


class TestDifferentiableTraining:
    def test_optimization_step_changes_params(self) -> None:
        """An optimization step through DifferentiableAAC should update parameters."""
        codec = DifferentiableAAC(
            sample_rate=48000,
            bitrate=128000,
            channels=1,
            quant_mode="ste",
            device="cpu",
        )

        # Simple learnable gain
        gain = nn.Parameter(torch.ones(1))
        optimizer = torch.optim.SGD([gain], lr=0.01)

        audio = torch.randn(4800) * 0.3
        initial_gain = gain.item()

        optimizer.zero_grad()
        scaled = audio * gain
        coded = codec(scaled)
        loss = (coded**2).mean()
        loss.backward()
        optimizer.step()

        # Parameter should have changed
        assert gain.item() != initial_gain, "Parameter didn't update"
        # Gradient should exist
        assert gain.grad is not None
        assert gain.grad.item() != 0.0

    def test_rate_loss(self) -> None:
        """return_rate_loss should return a second tensor."""
        codec = DifferentiableAAC(
            sample_rate=48000,
            bitrate=128000,
            channels=1,
            quant_mode="ste",
            device="cpu",
        )
        audio = torch.randn(1, 1, 4800)
        result = codec(audio, return_rate_loss=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        decoded, rate_loss = result
        assert decoded.shape == audio.shape
        assert rate_loss.dim() == 0  # scalar
