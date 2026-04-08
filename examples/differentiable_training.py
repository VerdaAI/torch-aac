"""Example: Train a model to be robust to AAC compression.

Demonstrates using DifferentiableAAC in a training loop where
gradients flow through the AAC simulation.

Usage:
    python examples/differentiable_training.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_aac import DifferentiableAAC


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create differentiable AAC codec
    codec = DifferentiableAAC(
        sample_rate=48000,
        bitrate=128000,
        channels=1,
        quant_mode="ste",  # straight-through estimator
        device=device,
    )

    # Simple model: a learnable audio transform (3-layer MLP on short segments)
    segment_len = 4800  # 0.1 seconds
    model = nn.Sequential(
        nn.Linear(segment_len, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, segment_len),
        nn.Tanh(),  # keep output in [-1, 1]
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("\nTraining a model robust to AAC compression...")
    print("The model learns to produce audio that survives AAC quantization.\n")

    for step in range(50):
        optimizer.zero_grad()

        # Generate random input audio
        input_audio = torch.randn(segment_len, device=device) * 0.3

        # Model produces output audio
        output_audio = model(input_audio)

        # Pass through differentiable AAC simulation
        # Gradients flow through quantization + MDCT
        coded_audio = codec(output_audio)

        # Loss: minimize distortion from AAC compression
        reconstruction_loss = F.mse_loss(coded_audio, output_audio)

        # Optional: rate loss to encourage efficient encoding
        # coded_audio, rate_loss = codec(output_audio, return_rate_loss=True)
        # total_loss = reconstruction_loss + 0.1 * rate_loss

        reconstruction_loss.backward()
        optimizer.step()

        if step % 10 == 0:
            print(
                f"Step {step:3d} | "
                f"Loss: {reconstruction_loss.item():.6f} | "
                f"Output range: [{output_audio.min().item():.3f}, {output_audio.max().item():.3f}]"
            )

    print("\nDone! The model has learned to produce AAC-robust audio.")
    print("Key insight: gradients flowed through the AAC quantization via STE.")


if __name__ == "__main__":
    main()
