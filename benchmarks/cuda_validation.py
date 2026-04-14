#!/usr/bin/env python3
"""
torch-aac CUDA Validation & Benchmark Script

Tests correctness + throughput on CUDA GPUs.
Prerequisites: pip install -e ".[dev]" && ffmpeg available on PATH

Usage:
    # Run on default GPU (cuda:0)
    python benchmarks/cuda_validation.py

    # Run on a specific GPU
    CUDA_VISIBLE_DEVICES=0 python benchmarks/cuda_validation.py
    CUDA_VISIBLE_DEVICES=1 python benchmarks/cuda_validation.py
"""

import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

import torch_aac


def device_check():
    print("=" * 60)
    print("torch-aac CUDA Validation")
    print("=" * 60)
    print(f"\nPyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("ERROR: No CUDA device found. Aborting.")
        raise SystemExit(1)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    total_bytes = getattr(props, "total_memory", None) or getattr(props, "total_mem", 0)
    print(f"VRAM: {total_bytes / 1024**3:.1f} GB")
    print(f"torch-aac: {torch_aac.__version__}")


def correctness_test(sr=48000):
    print("\n--- Correctness ---")
    signals = {
        "sine_1khz": (0.5 * np.sin(2 * np.pi * 1000 * np.arange(2 * sr) / sr)).astype(np.float32),
        "noise": (0.3 * np.random.RandomState(42).randn(2 * sr)).astype(np.float32),
    }
    failures = []
    for name, pcm in signals.items():
        aac = torch_aac.encode(pcm, sample_rate=sr, bitrate=128000, device="cuda")
        with tempfile.NamedTemporaryFile(suffix=".aac", delete=False) as f:
            f.write(aac)
            p = f.name
        r = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-i",
                p,
                "-f",
                "f32le",
                "-ar",
                str(sr),
                "-ac",
                "1",
                "pipe:1",
            ],
            capture_output=True,
        )
        Path(p).unlink()
        assert r.returncode == 0, f"{name}: FFmpeg decode failed"
        d = np.frombuffer(r.stdout, dtype=np.float32)
        ratio = np.abs(d).max() / np.abs(pcm).max()
        ok = 0.9 < ratio < 1.1
        status = "OK" if ok else "FAIL"
        print(f"  {name}: {len(aac)} bytes, peak ratio {ratio:.4f} {status}")
        if not ok:
            failures.append(name)
    return failures


def differentiable_test():
    print("\n--- Differentiable Mode (CUDA) ---")
    codec = torch_aac.DifferentiableAAC(sample_rate=48000, bitrate=128000, device="cuda")
    x = torch.randn(4096, device="cuda", requires_grad=True)
    y = codec(x)
    loss = y.pow(2).mean()
    loss.backward()
    grad_ok = x.grad.norm() > 0 and not x.grad.isnan().any()
    print(f"  grad norm: {x.grad.norm().item():.4f}, NaN: {x.grad.isnan().any().item()}")
    print("  OK" if grad_ok else "  FAIL")
    return grad_ok


def throughput_sweep(sr=48000):
    print("\n--- Throughput (CUDA) ---")
    results = {}
    for dur in [1, 10, 30, 60]:
        pcm = (0.5 * np.sin(2 * np.pi * 440 * np.arange(dur * sr) / sr)).astype(np.float32)
        for _ in range(3):
            torch_aac.encode(pcm, sample_rate=sr, bitrate=128000, device="cuda")
        torch.cuda.synchronize()
        times = []
        for _ in range(5):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            torch_aac.encode(pcm, sample_rate=sr, bitrate=128000, device="cuda")
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        best = min(times)
        results[dur] = best
        print(f"  {dur:3d}s mono 128k: {best * 1000:7.1f} ms = {dur / best:6.1f}x realtime")
    return results


def cuda_vs_cpu(sr=48000):
    print("\n--- CUDA vs CPU ---")
    pcm = (0.5 * np.sin(2 * np.pi * 440 * np.arange(10 * sr) / sr)).astype(np.float32)
    results = {}
    for device in ["cuda", "cpu"]:
        for _ in range(3):
            torch_aac.encode(pcm, sample_rate=sr, bitrate=128000, device=device)
        if device == "cuda":
            torch.cuda.synchronize()
        times = []
        for _ in range(5):
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            torch_aac.encode(pcm, sample_rate=sr, bitrate=128000, device=device)
            if device == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        best = min(times)
        results[device] = best
        print(f"  {device:5}: {best * 1000:.1f} ms = {10 / best:.1f}x realtime")
    return results


def ffmpeg_comparison(sr=48000):
    """Compare torch-aac (CUDA, CPU) vs FFmpeg's native AAC encoder."""
    print("\n--- torch-aac vs FFmpeg AAC ---")
    results = {}
    for dur in [1, 10, 30, 60]:
        pcm = (0.5 * np.sin(2 * np.pi * 440 * np.arange(dur * sr) / sr)).astype(np.float32)
        raw_bytes = pcm.tobytes()

        # FFmpeg: pipe raw PCM in, pipe AAC out
        # Warmup
        for _ in range(2):
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-v",
                    "error",
                    "-f",
                    "f32le",
                    "-ar",
                    str(sr),
                    "-ac",
                    "1",
                    "-i",
                    "pipe:0",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "128k",
                    "-f",
                    "adts",
                    "pipe:1",
                ],
                input=raw_bytes,
                capture_output=True,
            )
        ffmpeg_times = []
        for _ in range(5):
            t0 = time.perf_counter()
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-v",
                    "error",
                    "-f",
                    "f32le",
                    "-ar",
                    str(sr),
                    "-ac",
                    "1",
                    "-i",
                    "pipe:0",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "128k",
                    "-f",
                    "adts",
                    "pipe:1",
                ],
                input=raw_bytes,
                capture_output=True,
            )
            ffmpeg_times.append(time.perf_counter() - t0)
        ffmpeg_best = min(ffmpeg_times)

        # torch-aac CUDA
        for _ in range(3):
            torch_aac.encode(pcm, sample_rate=sr, bitrate=128000, device="cuda")
        torch.cuda.synchronize()
        cuda_times = []
        for _ in range(5):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            torch_aac.encode(pcm, sample_rate=sr, bitrate=128000, device="cuda")
            torch.cuda.synchronize()
            cuda_times.append(time.perf_counter() - t0)
        cuda_best = min(cuda_times)

        # torch-aac CPU
        for _ in range(3):
            torch_aac.encode(pcm, sample_rate=sr, bitrate=128000, device="cpu")
        cpu_times = []
        for _ in range(5):
            t0 = time.perf_counter()
            torch_aac.encode(pcm, sample_rate=sr, bitrate=128000, device="cpu")
            cpu_times.append(time.perf_counter() - t0)
        cpu_best = min(cpu_times)

        speedup_cuda = ffmpeg_best / cuda_best
        speedup_cpu = ffmpeg_best / cpu_best
        results[dur] = {
            "ffmpeg": ffmpeg_best,
            "cuda": cuda_best,
            "cpu": cpu_best,
        }
        print(
            f"  {dur:3d}s | ffmpeg {ffmpeg_best * 1000:7.1f} ms ({dur / ffmpeg_best:6.1f}x rt)"
            f"  cuda {cuda_best * 1000:7.1f} ms ({dur / cuda_best:6.1f}x rt)"
            f"  cpu {cpu_best * 1000:7.1f} ms ({dur / cpu_best:6.1f}x rt)"
            f"  | cuda/ffmpeg {speedup_cuda:.2f}x  cpu/ffmpeg {speedup_cpu:.2f}x"
        )
    return results


def batch_throughput(sr=48000):
    print("\n--- Batch Throughput (CUDA) ---")
    streams = [
        (0.3 * np.random.RandomState(i).randn(10 * sr)).astype(np.float32) for i in range(8)
    ]
    for _ in range(2):
        torch_aac.encode_batch(streams, sample_rate=sr, bitrate=128000, device="cuda")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    torch_aac.encode_batch(streams, sample_rate=sr, bitrate=128000, device="cuda")
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print(f"  8x10s batch: {dt * 1000:.1f} ms = {80 / dt:.1f}x aggregate realtime")
    return dt


def memory_usage(sr=48000):
    print("\n--- GPU Memory ---")
    torch.cuda.reset_peak_memory_stats()
    pcm60 = (0.5 * np.sin(2 * np.pi * 440 * np.arange(60 * sr) / sr)).astype(np.float32)
    torch_aac.encode(pcm60, sample_rate=sr, bitrate=128000, device="cuda")
    peak_mb = torch.cuda.max_memory_allocated() / 1024**2
    print(f"  Peak VRAM (60s encode): {peak_mb:.0f} MB")
    return peak_mb


def main():
    device_check()
    failures = correctness_test()
    grad_ok = differentiable_test()
    throughput_sweep()
    cuda_vs_cpu()
    ffmpeg_comparison()
    batch_throughput()
    memory_usage()

    print("\n" + "=" * 60)
    print("CUDA Validation Complete")
    if failures:
        print(f"FAILURES: {failures}")
    if not grad_ok:
        print("FAILURE: differentiable mode")
    print("=" * 60)


if __name__ == "__main__":
    main()
