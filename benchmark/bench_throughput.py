"""Throughput benchmark for torch-aac.

Measures encoding speed (frames/sec, realtime factor) at various audio
durations. Runs on the best available device (CUDA if available, else CPU).

Compares against FFmpeg's native AAC encoder as a baseline.

Usage:
    python benchmark/bench_throughput.py
    python benchmark/bench_throughput.py --quick  # 10s audio only
    python benchmark/bench_throughput.py --durations 10 60 300
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

import torch_aac


@dataclass
class BenchResult:
    label: str
    duration_sec: float
    encode_time_sec: float
    output_bytes: int

    @property
    def realtime_factor(self) -> float:
        return self.duration_sec / self.encode_time_sec

    @property
    def frames_per_sec(self) -> float:
        num_frames = int(self.duration_sec * 48000 / 1024)
        return num_frames / self.encode_time_sec


def generate_test_audio(duration_sec: float, sample_rate: int = 48000) -> np.ndarray:
    """Generate a test signal: mix of sine waves to approximate real audio."""
    n = int(duration_sec * sample_rate)
    t = np.arange(n, dtype=np.float32) / sample_rate
    # Mix of frequencies to exercise the encoder
    pcm = (
        0.3 * np.sin(2 * np.pi * 440 * t)
        + 0.2 * np.sin(2 * np.pi * 880 * t)
        + 0.1 * np.sin(2 * np.pi * 1760 * t)
    ).astype(np.float32)
    return pcm


def benchmark_torch_aac(
    pcm: np.ndarray,
    sample_rate: int = 48000,
    bitrate: int = 128000,
    device: str = "auto",
    runs: int = 3,
) -> BenchResult:
    """Benchmark torch-aac encoding."""
    duration = len(pcm) / sample_rate
    encoder = torch_aac.AACEncoder(
        sample_rate=sample_rate, channels=1, bitrate=bitrate, device=device
    )

    # Warmup
    _ = encoder.encode(pcm)

    times = []
    for _ in range(runs):
        if device != "cpu" and torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        aac_bytes = encoder.encode(pcm)
        if device != "cpu" and torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    encoder.close()

    return BenchResult(
        label="torch-aac",
        duration_sec=duration,
        encode_time_sec=float(np.median(times)),
        output_bytes=len(aac_bytes),
    )


def benchmark_ffmpeg(
    pcm: np.ndarray,
    sample_rate: int = 48000,
    bitrate: int = 128000,
    runs: int = 3,
) -> BenchResult | None:
    """Benchmark FFmpeg native AAC encoder."""
    duration = len(pcm) / sample_rate

    # Check FFmpeg is available
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    times = []
    output_size = 0
    for _ in range(runs):
        with tempfile.NamedTemporaryFile(suffix=".aac", delete=False) as f:
            aac_path = f.name
        try:
            t0 = time.perf_counter()
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "f32le",
                    "-ar",
                    str(sample_rate),
                    "-ac",
                    "1",
                    "-i",
                    "pipe:0",
                    "-c:a",
                    "aac",
                    "-b:a",
                    f"{bitrate // 1000}k",
                    aac_path,
                ],
                input=pcm.tobytes(),
                capture_output=True,
                check=True,
            )
            t1 = time.perf_counter()
            times.append(t1 - t0)
            output_size = os.path.getsize(aac_path)
        finally:
            Path(aac_path).unlink(missing_ok=True)

    return BenchResult(
        label="ffmpeg-aac",
        duration_sec=duration,
        encode_time_sec=float(np.median(times)),
        output_bytes=output_size,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Throughput benchmark for torch-aac")
    parser.add_argument("--quick", action="store_true", help="Quick test (10s audio only)")
    parser.add_argument(
        "--durations",
        type=float,
        nargs="+",
        default=[10.0, 60.0, 300.0, 3600.0],
        help="Audio durations to test (seconds)",
    )
    parser.add_argument("--bitrate", type=int, default=128000, help="Target bitrate in bps")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs per config")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    args = parser.parse_args()

    if args.quick:
        args.durations = [10.0]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = "CPU"
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)

    print(f"Device: {device} ({gpu_name})")
    print(f"torch_aac version: {torch_aac.__version__}")
    print(f"Bitrate: {args.bitrate} bps")
    print(f"Runs per config: {args.runs}")
    print()
    print(f"{'Duration':<12} {'Encoder':<15} {'Time (s)':<12} {'Realtime':<12} {'frames/s':<12}")
    print("-" * 65)

    results = []
    for duration in args.durations:
        pcm = generate_test_audio(duration)

        # Benchmark torch-aac
        ta_result = benchmark_torch_aac(pcm, bitrate=args.bitrate, device=device, runs=args.runs)
        print(
            f"{duration:>8.0f}s    {ta_result.label:<15} "
            f"{ta_result.encode_time_sec:>10.4f}   "
            f"{ta_result.realtime_factor:>8.1f}x    "
            f"{ta_result.frames_per_sec:>10.0f}"
        )
        results.append(
            {
                "label": ta_result.label,
                "duration_sec": ta_result.duration_sec,
                "encode_time_sec": ta_result.encode_time_sec,
                "realtime_factor": ta_result.realtime_factor,
                "frames_per_sec": ta_result.frames_per_sec,
                "output_bytes": ta_result.output_bytes,
                "device": device,
                "gpu_name": gpu_name,
            }
        )

        # Benchmark FFmpeg
        ff_result = benchmark_ffmpeg(pcm, bitrate=args.bitrate, runs=args.runs)
        if ff_result is not None:
            print(
                f"{duration:>8.0f}s    {ff_result.label:<15} "
                f"{ff_result.encode_time_sec:>10.4f}   "
                f"{ff_result.realtime_factor:>8.1f}x    "
                f"{ff_result.frames_per_sec:>10.0f}"
            )
            results.append(
                {
                    "label": ff_result.label,
                    "duration_sec": ff_result.duration_sec,
                    "encode_time_sec": ff_result.encode_time_sec,
                    "realtime_factor": ff_result.realtime_factor,
                    "frames_per_sec": ff_result.frames_per_sec,
                    "output_bytes": ff_result.output_bytes,
                }
            )

        print()

    if args.output:
        with open(args.output, "w") as f:
            json.dump(
                {
                    "device": device,
                    "gpu_name": gpu_name,
                    "bitrate": args.bitrate,
                    "runs": args.runs,
                    "results": results,
                },
                f,
                indent=2,
            )
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
