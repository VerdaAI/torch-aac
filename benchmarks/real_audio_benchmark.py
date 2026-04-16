#!/usr/bin/env python3
"""Real audio benchmark: torch-aac vs FFmpeg native AAC.

Finds audio files under a data root directory, encodes with both encoders,
decodes with FFmpeg, and compares SNR using cross-correlation alignment.

Usage:
    python benchmarks/real_audio_benchmark.py [--data-root /path/to/audio]
    python benchmarks/real_audio_benchmark.py --device cuda --data-root /data/audio
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import time
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_SR = 48000
DEFAULT_MAX_DUR = 30.0  # seconds
BITRATES = [64000, 128000, 192000]


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------
def load_audio(
    path: str, target_sr: int = DEFAULT_SR, max_dur: float = DEFAULT_MAX_DUR
) -> np.ndarray:
    """Load audio file, convert to mono float32 at target_sr, truncate."""
    info = sf.info(path)
    frames_to_read = min(int(info.samplerate * max_dur), info.frames)
    audio, file_sr = sf.read(path, frames=frames_to_read, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)  # to mono
    if file_sr != target_sr:
        try:
            from scipy.signal import resample_poly

            g = gcd(target_sr, file_sr)
            audio = resample_poly(audio, target_sr // g, file_sr // g).astype(np.float32)
        except ImportError:
            # Fallback: simple linear interpolation
            ratio = target_sr / file_sr
            new_len = int(len(audio) * ratio)
            audio = np.interp(
                np.linspace(0, len(audio) - 1, new_len),
                np.arange(len(audio)),
                audio,
            ).astype(np.float32)
    return audio[: int(target_sr * max_dur)]


def load_musdb_audio(
    path: str, target_sr: int = DEFAULT_SR, max_dur: float = DEFAULT_MAX_DUR
) -> np.ndarray:
    """Load mix from musdb18 stem.mp4 via ffmpeg."""
    r = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-t",
            str(max_dur),
            "-i",
            path,
            "-map",
            "0:0",
            "-f",
            "f32le",
            "-ar",
            str(target_sr),
            "-ac",
            "1",
            "pipe:1",
        ],
        capture_output=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {path}: {r.stderr.decode()[:200]}")
    return np.frombuffer(r.stdout, dtype=np.float32)[: int(target_sr * max_dur)]


# ---------------------------------------------------------------------------
# Encode / decode helpers
# ---------------------------------------------------------------------------
def encode_torch_aac(
    pcm: np.ndarray, sr: int, bitrate: int, device: str = "cpu"
) -> tuple[bytes, float]:
    """Encode with torch-aac, return (aac_bytes, elapsed_ms)."""
    from torch_aac import AACEncoder

    enc = AACEncoder(sample_rate=sr, channels=1, bitrate=bitrate, device=device)
    t0 = time.perf_counter()
    aac = enc.encode(pcm)
    elapsed = (time.perf_counter() - t0) * 1000
    return aac, elapsed


def encode_ffmpeg(pcm: np.ndarray, sr: int, bitrate: int, out_path: str) -> float:
    """Encode with FFmpeg native AAC, return elapsed_ms."""
    t0 = time.perf_counter()
    r = subprocess.run(
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
            str(bitrate),
            "-ar",
            str(sr),
            "-ac",
            "1",
            out_path,
        ],
        input=pcm.tobytes(),
        capture_output=True,
    )
    elapsed = (time.perf_counter() - t0) * 1000
    if r.returncode != 0:
        raise RuntimeError(f"FFmpeg encode failed: {r.stderr.decode()[:200]}")
    return elapsed


def decode_aac(aac_path: str, wav_path: str) -> np.ndarray:
    """Decode AAC to wav via FFmpeg, return float32 array."""
    r = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", aac_path, "-f", "wav", wav_path],
        capture_output=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"FFmpeg decode failed: {r.stderr.decode()[:200]}")
    audio, _ = sf.read(wav_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio


# ---------------------------------------------------------------------------
# Alignment and metrics
# ---------------------------------------------------------------------------
def align_xcorr(
    original: np.ndarray, decoded: np.ndarray, max_shift: int = 4096
) -> tuple[np.ndarray, np.ndarray]:
    """Align decoded to original using cross-correlation, return trimmed pair.

    Uses a chunk from the middle of both signals for robust correlation,
    avoiding edge effects and silence at the start.
    """
    min_len = min(len(original), len(decoded))
    if min_len < 4096:
        n = min(len(original), len(decoded))
        return original[:n], decoded[:n]

    # Take a chunk from the middle of the original
    mid = min_len // 2
    chunk_len = min(8000, mid // 2)
    o_chunk = original[mid : mid + chunk_len]

    # Search decoded around the same position ± max_shift
    search_start = max(0, mid - max_shift)
    search_end = min(len(decoded), mid + chunk_len + max_shift)
    d_search = decoded[search_start:search_end]

    # Cross-correlate (mode='valid' gives one value per offset)
    corr = np.correlate(d_search, o_chunk, mode="valid")
    best_offset = int(np.argmax(np.abs(corr)))
    delay = search_start + best_offset - mid  # positive = decoded shifted right

    # Apply alignment: shift decoded so decoded[delay] aligns with original[0]
    skip = 2048  # skip edges for cleaner measurement
    if delay >= 0:
        o_start = skip
        d_start = skip + delay
    else:
        o_start = skip - delay
        d_start = skip

    n = min(len(original) - o_start, len(decoded) - d_start) - skip
    if n <= 0:
        n = min(len(original), len(decoded))
        return original[:n], decoded[:n]
    return original[o_start : o_start + n], decoded[d_start : d_start + n]


def compute_snr(original: np.ndarray, decoded: np.ndarray) -> float:
    """Compute SNR in dB between aligned signals."""
    n = min(len(original), len(decoded))
    o, d = original[:n], decoded[:n]
    sig_power = np.mean(o**2)
    noise_power = np.mean((o - d) ** 2)
    if noise_power < 1e-20:
        return 100.0
    if sig_power < 1e-20:
        return 0.0
    return float(10 * np.log10(sig_power / noise_power))


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------
def find_audio_files(data_root: Path, max_files: int = 20) -> list[tuple[str, str, str]]:
    """Find audio files, return list of (dataset, path, genre)."""
    files: list[tuple[str, str, str]] = []
    extensions = {".wav", ".flac", ".mp3", ".mp4", ".m4a"}

    for ext in extensions:
        for p in sorted(data_root.rglob(f"*{ext}"))[:50]:
            path_str = str(p)
            # Detect dataset
            if "bach10" in path_str.lower():
                dataset, genre = "Bach10", "classical"
            elif "librispeech" in path_str.lower():
                dataset, genre = "LibriSpeech", "speech"
            elif "musdb18" in path_str.lower():
                if "test" in path_str.lower():
                    dataset = "musdb18/test"
                elif "train" in path_str.lower():
                    dataset = "musdb18/train"
                else:
                    dataset = "musdb18"
                genre = "music"
            else:
                dataset, genre = "other", "unknown"
            files.append((dataset, path_str, genre))

    # Deduplicate and limit
    seen = set()
    unique = []
    for ds, p, g in files:
        stem = Path(p).stem
        if stem not in seen:
            seen.add(stem)
            unique.append((ds, p, g))
    return unique[:max_files]


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Real audio benchmark")
    parser.add_argument(
        "--data-root", type=str, default=".", help="Root directory to scan for audio files"
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="PyTorch device (cpu, cuda, mps)"
    )
    parser.add_argument("--max-files", type=int, default=20, help="Max files to test")
    parser.add_argument(
        "--max-dur", type=float, default=DEFAULT_MAX_DUR, help="Max duration per file (seconds)"
    )
    parser.add_argument("--output", type=str, default="benchmarks/results/real_audio_results.json")
    args = parser.parse_args()

    sr = DEFAULT_SR
    data_root = Path(args.data_root)

    print(f"Scanning {data_root} for audio files...")
    files = find_audio_files(data_root, args.max_files)

    if not files:
        print("No audio files found! Generating synthetic test signals...")
        # Generate test signals
        t = np.linspace(0, 10, sr * 10, dtype=np.float32)
        synth = {
            "synth_speech": (
                np.sin(2 * np.pi * 200 * t) * 0.3
                + np.sin(2 * np.pi * 400 * t) * 0.2
                + np.sin(2 * np.pi * 800 * t) * 0.15
            ).astype(np.float32),
            "synth_music": (
                np.sin(2 * np.pi * 440 * t) * 0.3
                + np.sin(2 * np.pi * 880 * t) * 0.2
                + np.random.randn(len(t)).astype(np.float32) * 0.05
            ).astype(np.float32),
        }
        for name, pcm in synth.items():
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, pcm, sr)
                files.append(("synthetic", f.name, name))

    print(f"Found {len(files)} files across {len(set(d for d, _, _ in files))} datasets\n")

    results = []
    count = 0

    # Header
    print(
        f"{'File':<40} | {'kbps':>4} | {'torch SNR':>10} | {'ffmpeg SNR':>10} | "
        f"{'Δ SNR':>7} | {'torch kbps':>10} | {'ff kbps':>8} | "
        f"{'torch ms':>8} | {'ff ms':>7}"
    )
    print("-" * 130)

    for dataset, path, genre in files:
        fname = Path(path).name[:38]
        try:
            if "musdb" in dataset.lower() and path.endswith(".mp4"):
                pcm = load_musdb_audio(path, target_sr=sr, max_dur=args.max_dur)
            else:
                pcm = load_audio(path, target_sr=sr, max_dur=args.max_dur)
        except Exception as e:
            print(f"{'SKIP: ' + fname:<40} | Error loading: {e}")
            continue

        duration = len(pcm) / sr

        for bitrate in BITRATES:
            count += 1
            br_k = bitrate // 1000
            try:
                # --- torch-aac ---
                aac_bytes, t_ms = encode_torch_aac(pcm, sr, bitrate, device=args.device)
                t_kbps = len(aac_bytes) * 8 / duration / 1000

                with tempfile.NamedTemporaryFile(suffix=".aac", delete=False) as f:
                    f.write(aac_bytes)
                    t_aac = f.name
                t_wav = t_aac + ".wav"
                t_decoded = decode_aac(t_aac, t_wav)

                # --- FFmpeg ---
                with tempfile.NamedTemporaryFile(suffix=".aac", delete=False) as f:
                    f_aac = f.name
                f_ms = encode_ffmpeg(pcm, sr, bitrate, f_aac)
                f_kbps = Path(f_aac).stat().st_size * 8 / duration / 1000
                f_wav = f_aac + ".wav"
                f_decoded = decode_aac(f_aac, f_wav)

                # --- Align and measure ---
                o_t, d_t = align_xcorr(pcm, t_decoded)
                o_f, d_f = align_xcorr(pcm, f_decoded)

                t_snr = compute_snr(o_t, d_t)
                f_snr = compute_snr(o_f, d_f)
                delta = t_snr - f_snr

                entry = {
                    "file": Path(path).name,
                    "dataset": dataset,
                    "genre": genre,
                    "duration_s": round(duration, 2),
                    "bitrate": bitrate,
                    "torch_aac_snr": round(t_snr, 2),
                    "ffmpeg_snr": round(f_snr, 2),
                    "delta_snr": round(delta, 2),
                    "torch_aac_kbps": round(t_kbps, 1),
                    "ffmpeg_kbps": round(f_kbps, 1),
                    "torch_aac_time_ms": round(t_ms, 1),
                    "ffmpeg_time_ms": round(f_ms, 1),
                }
                results.append(entry)

                marker = "+" if delta > 0 else "-" if delta < -3 else "~"
                print(
                    f"{fname:<40} | {br_k:>4} | {t_snr:>10.1f} | {f_snr:>10.1f} | "
                    f"{delta:>+6.1f}{marker} | {t_kbps:>10.1f} | {f_kbps:>8.1f} | "
                    f"{t_ms:>8.0f} | {f_ms:>7.0f}"
                )

                # Cleanup
                import contextlib

                for p2 in [t_aac, t_wav, f_aac, f_wav]:
                    with contextlib.suppress(Exception):
                        Path(p2).unlink(missing_ok=True)

            except Exception as e:
                print(f"{fname:<40} | {br_k:>4} | ERROR: {e}")

    # --- Summary ---
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if results:
        t_snrs = [r["torch_aac_snr"] for r in results]
        f_snrs = [r["ffmpeg_snr"] for r in results]
        deltas = [r["delta_snr"] for r in results]
        t_wins = sum(1 for d in deltas if d > 0)
        f_wins = sum(1 for d in deltas if d < 0)

        print(f"Total comparisons: {len(results)}")
        print(f"torch-aac avg SNR:  {np.mean(t_snrs):.1f} dB")
        print(f"FFmpeg avg SNR:     {np.mean(f_snrs):.1f} dB")
        print(f"Average Δ SNR:      {np.mean(deltas):+.1f} dB")
        print(f"torch-aac wins:     {t_wins}")
        print(f"FFmpeg wins:        {f_wins}")

        # Per-dataset breakdown
        datasets = set(r["dataset"] for r in results)
        for ds in sorted(datasets):
            ds_results = [r for r in results if r["dataset"] == ds]
            ds_delta = np.mean([r["delta_snr"] for r in ds_results])
            ds_wins = sum(1 for r in ds_results if r["delta_snr"] > 0)
            print(f"  {ds:<20}: Δ={ds_delta:+.1f} dB, wins {ds_wins}/{len(ds_results)}")

        # Files where FFmpeg wins by >3 dB
        bad = [r for r in results if r["delta_snr"] < -3]
        if bad:
            print(f"\nFiles where FFmpeg wins by >3 dB ({len(bad)}):")
            for r in bad[:10]:
                print(f"  {r['file']:<40} {r['bitrate'] // 1000}k: Δ={r['delta_snr']:+.1f} dB")

    # Save results
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
