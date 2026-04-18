#!/usr/bin/env python3
"""
Throughput test: pure-GPU path vs mixed GPU/CPU path.

Compares a pure tone (all long frames, pure GPU path) vs music with
transients (mix of long + short frames, GPU/CPU split path) to verify
the split doesn't cause excessive overhead.
"""

import time

import numpy as np

from torch_aac import AACEncoder


def main():
    sr = 48000
    duration = 60.0
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)

    # Pure tone: no transients, all long frames -> pure GPU path
    tone = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)

    # Music-like with periodic transients: triggers short-block switching
    # -> mixed long+short frames -> GPU/CPU split path
    music = (
        np.sin(2 * np.pi * 440 * t) * 0.3 + np.sin(2 * np.pi * 880 * t) * 0.2
    ).astype(np.float32)
    for pos in range(sr, int(sr * duration), sr * 5):
        music[pos : pos + 256] = 0.95

    # Warmup
    enc = AACEncoder(sample_rate=sr, channels=1, bitrate=128000, device="cuda")
    enc.encode(tone[: sr * 5])

    for name, pcm in [
        ("Pure tone (all GPU)", tone),
        ("Music+transients (split)", music),
    ]:
        times = []
        for _ in range(3):
            enc2 = AACEncoder(sample_rate=sr, channels=1, bitrate=128000, device="cuda")
            t0 = time.perf_counter()
            enc2.encode(pcm)
            times.append(time.perf_counter() - t0)
        avg = np.mean(times)
        print(f"{name}: {avg*1000:.0f} ms = {duration/avg:.0f}x realtime")


if __name__ == "__main__":
    main()
