"""Example: Basic audio encoding with torch-aac.

Usage:
    python examples/basic_encode.py
"""

import numpy as np

import torch_aac


def main() -> None:
    # Generate a 1-second 440Hz sine wave at 48kHz
    sample_rate = 48000
    t = np.arange(sample_rate, dtype=np.float32) / sample_rate
    pcm = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    print(f"Input: {len(pcm)} samples, {sample_rate}Hz, mono")

    # Encode to AAC
    aac_bytes = torch_aac.encode(pcm, sample_rate=sample_rate, bitrate=128000)
    print(f"Encoded: {len(aac_bytes)} bytes of AAC-LC")

    # Or use the encoder class for more control
    with torch_aac.AACEncoder(sample_rate=48000, channels=1, bitrate=128000) as enc:
        aac_bytes = enc.encode(pcm)
        print(f"Encoded again: {len(aac_bytes)} bytes")

    # Write to file
    with open("/tmp/test_output.aac", "wb") as f:
        f.write(aac_bytes)
    print("Written to /tmp/test_output.aac")
    print("Decode with: ffmpeg -i /tmp/test_output.aac /tmp/test_output.wav")


if __name__ == "__main__":
    main()
