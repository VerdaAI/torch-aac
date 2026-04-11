"""Quality benchmark for torch-aac.

Measures reconstruction quality (SNR, peak ratio, correlation) by
encoding audio, decoding with FFmpeg, and comparing with the original.

Usage:
    python benchmark/bench_quality.py
    python benchmark/bench_quality.py --bitrates 48000 128000 320000
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np

import torch_aac


def generate_signal(kind: str, duration_sec: float = 1.0, sr: int = 48000) -> np.ndarray:
    """Generate test signal of various types."""
    n = int(duration_sec * sr)
    t = np.arange(n, dtype=np.float32) / sr

    if kind == "sine_440":
        return (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    elif kind == "sine_1khz":
        return (0.5 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
    elif kind == "chord":
        return (
            0.3 * np.sin(2 * np.pi * 440 * t)
            + 0.2 * np.sin(2 * np.pi * 880 * t)
            + 0.1 * np.sin(2 * np.pi * 1320 * t)
        ).astype(np.float32)
    elif kind == "noise":
        rng = np.random.RandomState(42)
        return (0.3 * rng.randn(n)).astype(np.float32)
    elif kind == "silence":
        return np.zeros(n, dtype=np.float32)
    else:
        raise ValueError(f"Unknown signal kind: {kind}")


def encode_decode_roundtrip(
    pcm: np.ndarray, sample_rate: int, bitrate: int
) -> tuple[np.ndarray, int, int]:
    """Encode with torch-aac, decode with FFmpeg. Returns (decoded, aac_bytes, errors)."""
    aac = torch_aac.encode(pcm, sample_rate=sample_rate, bitrate=bitrate, device="cpu")

    with tempfile.NamedTemporaryFile(suffix=".aac", delete=False) as f:
        f.write(aac)
        aac_path = f.name

    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-err_detect",
                "ignore_err",
                "-max_error_rate",
                "1.0",
                "-i",
                aac_path,
                "-f",
                "f32le",
                "-ar",
                str(sample_rate),
                "-ac",
                "1",
                "pipe:1",
            ],
            capture_output=True,
            timeout=30,
        )
        decoded = (
            np.frombuffer(result.stdout, dtype=np.float32)
            if len(result.stdout) > 0
            else np.array([], dtype=np.float32)
        )
        errors = len([ln for ln in result.stderr.decode().split("\n") if ln.strip()])
    finally:
        Path(aac_path).unlink(missing_ok=True)

    return decoded, len(aac), errors


def compute_metrics(orig: np.ndarray, decoded: np.ndarray) -> dict[str, float]:
    """Compute quality metrics comparing original and decoded."""
    if len(decoded) == 0:
        return {
            "peak_ratio": 0.0,
            "rms_ratio": 0.0,
            "snr_db": float("-inf"),
            "correlation": 0.0,
        }

    # Align lengths (skip encoder latency)
    min_len = min(len(orig), len(decoded))
    if min_len < 2048:
        return {"peak_ratio": 0.0, "rms_ratio": 0.0, "snr_db": float("-inf"), "correlation": 0.0}

    skip = 1024
    o = orig[skip:min_len]
    d = decoded[skip:min_len]

    orig_peak = np.abs(o).max() if len(o) > 0 else 1.0
    dec_peak = np.abs(d).max() if len(d) > 0 else 0.0
    orig_rms = np.sqrt(np.mean(o**2)) if len(o) > 0 else 1.0
    dec_rms = np.sqrt(np.mean(d**2)) if len(d) > 0 else 0.0

    peak_ratio = dec_peak / (orig_peak + 1e-10)
    rms_ratio = dec_rms / (orig_rms + 1e-10)

    # SNR: normalize first
    if dec_rms > 1e-10 and orig_rms > 1e-10:
        d_normalized = d * (orig_rms / dec_rms)
        noise = d_normalized - o
        noise_rms = np.sqrt(np.mean(noise**2))
        snr_db = 20 * np.log10(orig_rms / (noise_rms + 1e-10))
    else:
        snr_db = float("-inf")

    # Correlation
    correlation = float(np.corrcoef(o, d)[0, 1]) if dec_rms > 1e-10 and orig_rms > 1e-10 else 0.0

    return {
        "peak_ratio": float(peak_ratio),
        "rms_ratio": float(rms_ratio),
        "snr_db": float(snr_db),
        "correlation": correlation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Quality benchmark for torch-aac")
    parser.add_argument(
        "--bitrates",
        type=int,
        nargs="+",
        default=[48000, 128000, 320000],
    )
    parser.add_argument(
        "--signals",
        type=str,
        nargs="+",
        default=["sine_440", "sine_1khz", "chord", "noise"],
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=1.0,
    )
    args = parser.parse_args()

    print(f"torch_aac version: {torch_aac.__version__}")
    print(f"Duration: {args.duration}s")
    print()
    print(
        f"{'Signal':<12} {'Bitrate':<10} {'Errors':<8} {'Peak':<8} "
        f"{'RMS':<8} {'SNR (dB)':<10} {'Corr':<8}"
    )
    print("-" * 72)

    for signal_kind in args.signals:
        pcm = generate_signal(signal_kind, args.duration)
        for bitrate in args.bitrates:
            decoded, _aac_size, errors = encode_decode_roundtrip(pcm, 48000, bitrate)
            metrics = compute_metrics(pcm, decoded)
            snr_str = f"{metrics['snr_db']:.1f}" if metrics["snr_db"] > -900 else "-inf"
            print(
                f"{signal_kind:<12} {bitrate // 1000:>4}k      "
                f"{errors:>6}   {metrics['peak_ratio']:>6.3f}  "
                f"{metrics['rms_ratio']:>6.3f}  {snr_str:>8}    "
                f"{metrics['correlation']:>6.3f}"
            )
        print()


if __name__ == "__main__":
    main()
