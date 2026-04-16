#!/usr/bin/env python3
"""
Real Audio Benchmark: torch-aac vs FFmpeg AAC

Compares encoding quality (SNR, spectral distortion, bitrate accuracy)
and throughput on real audio files.

Supports custom data directories, configurable bitrates, sample rates,
and max duration. Scans for .wav, .flac, .mp3, and .stem.mp4 files.

Usage:
    # Default: scan /data/raw_bench_datasets, 48kHz, 30s max, 64/128/192k
    python benchmarks/real_audio_benchmark.py

    # Custom data directory and settings
    python benchmarks/real_audio_benchmark.py --data-dir /path/to/audio \\
        --bitrates 128000 256000 --max-duration 60 --max-files 30

    # Specific GPU
    CUDA_VISIBLE_DEVICES=1 python benchmarks/real_audio_benchmark.py

    # Force CPU
    python benchmarks/real_audio_benchmark.py --device cpu
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch

import torch_aac

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_SR = 48000
DEFAULT_MAX_DUR = 30
DEFAULT_BITRATES = [64000, 128000, 192000]
DEFAULT_DATA_ROOT = "/data/raw_bench_datasets"
FFMPEG_ENCODER_DELAY = 1024  # FFmpeg AAC encoder priming samples


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark torch-aac vs FFmpeg AAC on real audio files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir", type=str, default=DEFAULT_DATA_ROOT,
        help=f"Root directory to scan for audio files (default: {DEFAULT_DATA_ROOT})",
    )
    parser.add_argument(
        "--output-dir", type=str, default="benchmarks/results",
        help="Directory for results output (default: benchmarks/results)",
    )
    parser.add_argument(
        "--bitrates", type=int, nargs="+", default=DEFAULT_BITRATES,
        help=f"Bitrates to test in bps (default: {DEFAULT_BITRATES})",
    )
    parser.add_argument(
        "--sample-rate", type=int, default=DEFAULT_SR,
        help=f"Target sample rate in Hz (default: {DEFAULT_SR})",
    )
    parser.add_argument(
        "--max-duration", type=float, default=DEFAULT_MAX_DUR,
        help=f"Max audio duration in seconds (default: {DEFAULT_MAX_DUR})",
    )
    parser.add_argument(
        "--max-files", type=int, default=20,
        help="Max total files to benchmark (default: 20)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device for torch-aac: cuda, cpu, or mps (default: auto-detect)",
    )
    parser.add_argument(
        "--no-spectrograms", action="store_true",
        help="Skip spectrogram generation",
    )
    parser.add_argument(
        "--file-list", type=str, default=None,
        help="Path to a text file listing audio paths (one per line). "
             "Overrides --data-dir and --max-files for reproducible runs.",
    )
    parser.add_argument(
        "--save-file-list", type=str, default=None,
        help="Save the selected file list to this path for future replay",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_audio(path: str, target_sr: int = DEFAULT_SR, max_dur: float = DEFAULT_MAX_DUR) -> np.ndarray:
    """Load audio file, convert to mono float32 at target_sr, truncate."""
    info = sf.info(path)
    frames_to_read = min(int(info.samplerate * max_dur), info.frames)
    audio, file_sr = sf.read(path, frames=frames_to_read, dtype="float32", always_2d=True)
    # to mono
    audio = audio.mean(axis=1)
    # resample if needed
    if file_sr != target_sr:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(target_sr, file_sr)
        audio = resample_poly(audio, target_sr // g, file_sr // g).astype(np.float32)
    return audio[:int(target_sr * max_dur)]


def load_musdb_audio(path: str, target_sr: int = DEFAULT_SR, max_dur: float = DEFAULT_MAX_DUR) -> np.ndarray:
    """Load first audio stream (mix) from musdb18 stem.mp4 via ffmpeg."""
    max_samples = int(target_sr * max_dur)
    r = subprocess.run(
        [
            "ffmpeg", "-v", "error",
            "-t", str(max_dur),
            "-i", path,
            "-map", "0:0",
            "-f", "f32le", "-ar", str(target_sr), "-ac", "1", "pipe:1",
        ],
        capture_output=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {path}: {r.stderr.decode()}")
    audio = np.frombuffer(r.stdout, dtype=np.float32)
    return audio[:max_samples]


def encode_ffmpeg(pcm: np.ndarray, sr: int, bitrate: int, out_path: str):
    """Encode PCM to AAC via FFmpeg subprocess."""
    r = subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "pipe:0",
            "-c:a", "aac", "-b:a", str(bitrate),
            "-ar", str(sr), "-ac", "1", out_path,
        ],
        input=pcm.tobytes(), capture_output=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg encode failed: {r.stderr.decode()}")


def decode_ffmpeg(aac_path: str, sr: int) -> np.ndarray:
    """Decode AAC file to PCM float32 via FFmpeg."""
    r = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-err_detect", "ignore_err",
            "-i", aac_path,
            "-f", "f32le", "-ar", str(sr), "-ac", "1", "pipe:1",
        ],
        capture_output=True,
    )
    if r.returncode != 0 or len(r.stdout) == 0:
        raise RuntimeError("ffmpeg decode failed (corrupt bitstream)")
    return np.frombuffer(r.stdout, dtype=np.float32)


def compute_snr(original: np.ndarray, decoded: np.ndarray) -> float:
    """Compute SNR in dB after aligning lengths."""
    n = min(len(original), len(decoded))
    orig = original[:n]
    dec = decoded[:n]
    noise = orig - dec
    sig_power = np.mean(orig ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power < 1e-20:
        return 100.0
    return 10 * np.log10(sig_power / noise_power)


def compute_spectral_distortion(original: np.ndarray, decoded: np.ndarray,
                                 sr: int = DEFAULT_SR, n_fft: int = 2048) -> float:
    """Mean absolute difference of log-magnitude spectrograms."""
    n = min(len(original), len(decoded))
    orig = original[:n]
    dec = decoded[:n]

    def log_mag_spec(x):
        # Compute STFT magnitude in dB
        hop = n_fft // 4
        num_frames = (len(x) - n_fft) // hop + 1
        if num_frames < 1:
            return np.array([0.0])
        frames = np.lib.stride_tricks.as_strided(
            x, shape=(num_frames, n_fft),
            strides=(x.strides[0] * hop, x.strides[0]),
        )
        window = np.hanning(n_fft).astype(np.float32)
        spec = np.abs(np.fft.rfft(frames * window, axis=1))
        return 20 * np.log10(np.maximum(spec, 1e-10))

    s_orig = log_mag_spec(orig)
    s_dec = log_mag_spec(dec)
    n_frames = min(s_orig.shape[0], s_dec.shape[0])
    return float(np.mean(np.abs(s_orig[:n_frames] - s_dec[:n_frames])))


def align_decoded(original: np.ndarray, decoded: np.ndarray,
                  encoder_delay: int = 0) -> tuple:
    """Trim encoder delay and align signals. Returns (orig_aligned, dec_aligned)."""
    if encoder_delay > 0 and len(decoded) > encoder_delay:
        decoded = decoded[encoder_delay:]
    # Trim both to same length, skip first 1024 samples (edge effects)
    skip = 1024
    n = min(len(original) - skip, len(decoded) - skip)
    if n <= 0:
        n = min(len(original), len(decoded))
        return original[:n], decoded[:n]
    return original[skip:skip + n], decoded[skip:skip + n]


# ---------------------------------------------------------------------------
# File selection
# ---------------------------------------------------------------------------
def select_files(data_root: Path, max_files: int = 20):
    """Pick files from each dataset found under data_root."""
    files = []

    # Bach10: mix WAVs (not instrument stems)
    bach_dir = data_root / "Bach10"
    if bach_dir.exists():
        bach_dirs = sorted([d for d in bach_dir.iterdir() if d.is_dir()])
        for d in bach_dirs[:5]:
            mix = d / f"{d.name}.wav"
            if mix.exists():
                files.append(("Bach10", str(mix), "classical"))

    # LibriSpeech: pick 5 longest files
    libri_dir = data_root / "LibriSpeech"
    if libri_dir.exists():
        flacs = []
        for flac in libri_dir.rglob("*.flac"):
            try:
                info = sf.info(str(flac))
                flacs.append((info.duration, str(flac)))
            except Exception:
                continue
        flacs.sort(reverse=True)
        for dur, path in flacs[:5]:
            files.append(("LibriSpeech", path, "speech"))

    # Also scan for any loose audio files in data_root
    for ext in ["*.wav", "*.flac", "*.mp3"]:
        for f in sorted(data_root.glob(ext))[:3]:
            files.append(("root", str(f), "unknown"))

    # musdb18: 5 from test, 5 from train
    musdb_dir = data_root / "musdb18"
    if musdb_dir.exists():
        for split in ["test", "train"]:
            split_dir = musdb_dir / split
            if not split_dir.exists():
                continue
            stems = sorted(split_dir.glob("*.stem.mp4"))
            # Pick evenly spaced
            step = max(1, len(stems) // 5)
            picked = stems[::step][:5]
            for s in picked:
                files.append((f"musdb18/{split}", str(s), "music"))

    return files[:max_files]


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------
def run_benchmark(args):
    sr = args.sample_rate
    max_dur = args.max_duration
    bitrates = args.bitrates
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path(args.data_dir)
    results_dir = Path(args.output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Real Audio Benchmark: torch-aac vs FFmpeg AAC")
    print("=" * 80)
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"torch-aac: {torch_aac.__version__}")
    print(f"Bitrates: {[f'{b//1000}k' for b in bitrates]}")
    print(f"Max duration: {max_dur}s, Sample rate: {sr} Hz")
    print(f"Data dir: {data_root}")

    if args.file_list:
        # Load file list for reproducible runs
        files = []
        with open(args.file_list) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) == 3:
                    dataset, path, genre = parts
                else:
                    path = parts[0]
                    dataset, genre = "custom", "unknown"
                files.append((dataset, path, genre))
        print(f"\nLoaded {len(files)} files from {args.file_list}")
    else:
        files = select_files(data_root, args.max_files)

    # Save file list for future replay
    if args.save_file_list or not args.file_list:
        list_path = args.save_file_list or str(results_dir / "file_list.txt")
        with open(list_path, "w") as f:
            f.write("# dataset\tpath\tgenre\n")
            for dataset, path, genre in files:
                f.write(f"{dataset}\t{path}\t{genre}\n")
        print(f"File list saved to {list_path}")

    print(f"\nSelected {len(files)} files:")
    for dataset, path, genre in files:
        print(f"  [{dataset}] {Path(path).name} ({genre})")

    results = []
    worst_snr = float("inf")
    worst_entry = None

    # Header
    print(f"\n{'File':<40} | {'BR':>4} | {'taac SNR':>9} | {'ffmpeg SNR':>10} | "
          f"{'d SNR':>6} | {'taac kbps':>9} | {'ff kbps':>8} | "
          f"{'taac SD':>7} | {'ff SD':>6} | {'taac ms':>8} | {'ff ms':>7}")
    print("-" * 160)

    for dataset, path, genre in files:
        fname = Path(path).stem[:38]
        try:
            if "musdb18" in dataset:
                pcm = load_musdb_audio(path, target_sr=sr, max_dur=max_dur)
            else:
                pcm = load_audio(path, target_sr=sr, max_dur=max_dur)

            dur_s = len(pcm) / sr
            if dur_s < 0.5:
                print(f"  {fname:<40} | SKIP (too short: {dur_s:.1f}s)")
                continue

        except Exception as e:
            print(f"  {fname:<40} | LOAD ERROR: {e}")
            continue

        for bitrate in bitrates:
            br_label = f"{bitrate // 1000}k"
            try:
                # --- torch-aac encode ---
                t0 = time.perf_counter()
                aac_bytes = torch_aac.encode(
                    pcm, sample_rate=sr, bitrate=bitrate, device=device
                )
                if device == "cuda":
                    torch.cuda.synchronize()
                taac_time = time.perf_counter() - t0

                # Write and decode torch-aac output
                with tempfile.NamedTemporaryFile(suffix=".aac", delete=False) as f:
                    f.write(aac_bytes)
                    taac_path = f.name
                taac_decoded = decode_ffmpeg(taac_path, sr)
                taac_size = os.path.getsize(taac_path)
                Path(taac_path).unlink()

                taac_kbps = taac_size * 8 / dur_s / 1000

                # --- FFmpeg encode ---
                with tempfile.NamedTemporaryFile(suffix=".aac", delete=False) as f:
                    ff_path = f.name
                t0 = time.perf_counter()
                encode_ffmpeg(pcm, sr, bitrate, ff_path)
                ff_time = time.perf_counter() - t0

                ff_decoded = decode_ffmpeg(ff_path, sr)
                ff_size = os.path.getsize(ff_path)
                Path(ff_path).unlink()

                ff_kbps = ff_size * 8 / dur_s / 1000

                # --- Align and measure ---
                orig_taac, dec_taac = align_decoded(pcm, taac_decoded, encoder_delay=0)
                orig_ff, dec_ff = align_decoded(pcm, ff_decoded,
                                                 encoder_delay=FFMPEG_ENCODER_DELAY)

                taac_snr = compute_snr(orig_taac, dec_taac)
                ff_snr = compute_snr(orig_ff, dec_ff)
                delta_snr = taac_snr - ff_snr

                taac_sd = compute_spectral_distortion(orig_taac, dec_taac)
                ff_sd = compute_spectral_distortion(orig_ff, dec_ff)

                entry = {
                    "file": Path(path).name,
                    "dataset": dataset,
                    "genre": genre,
                    "duration_s": round(dur_s, 2),
                    "bitrate": bitrate,
                    "torch_aac_snr": round(taac_snr, 2),
                    "ffmpeg_snr": round(ff_snr, 2),
                    "delta_snr": round(delta_snr, 2),
                    "torch_aac_kbps": round(taac_kbps, 1),
                    "ffmpeg_kbps": round(ff_kbps, 1),
                    "torch_aac_spectral_dist": round(taac_sd, 2),
                    "ffmpeg_spectral_dist": round(ff_sd, 2),
                    "torch_aac_time_ms": round(taac_time * 1000, 1),
                    "ffmpeg_time_ms": round(ff_time * 1000, 1),
                }
                results.append(entry)

                # Track worst torch-aac SNR for spectrogram
                if taac_snr < worst_snr:
                    worst_snr = taac_snr
                    worst_entry = {
                        "file": Path(path).name,
                        "bitrate": bitrate,
                        "original": pcm.copy(),
                        "taac_decoded": taac_decoded.copy(),
                        "ff_decoded": ff_decoded.copy(),
                    }

                print(f"  {fname:<40} | {br_label:>4} | {taac_snr:>8.1f}dB | "
                      f"{ff_snr:>9.1f}dB | {delta_snr:>+5.1f} | "
                      f"{taac_kbps:>8.1f} | {ff_kbps:>7.1f} | "
                      f"{taac_sd:>7.1f} | {ff_sd:>5.1f} | "
                      f"{taac_time * 1000:>7.1f} | {ff_time * 1000:>6.1f}")

            except Exception as e:
                print(f"  {fname:<40} | {br_label:>4} | ERROR: {e}")
                continue

    if not results:
        print("\nNo results collected!")
        return

    # -----------------------------------------------------------------------
    # Save raw results
    # -----------------------------------------------------------------------
    json_path = results_dir / "real_audio_results.json"
    # Convert numpy types for JSON serialization
    clean_results = []
    for r in results:
        clean_results.append({k: float(v) if hasattr(v, 'item') else v for k, v in r.items()})
    with open(json_path, "w") as f:
        json.dump(clean_results, f, indent=2)
    print(f"\nRaw results saved to {json_path}")

    # -----------------------------------------------------------------------
    # Generate spectrogram for worst-performing file
    # -----------------------------------------------------------------------
    if worst_entry is not None and not args.no_spectrograms:
        try:
            generate_spectrograms(worst_entry, results_dir, sr, max_dur)
        except Exception as e:
            print(f"Spectrogram generation failed: {e}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print_summary(results, bitrates, device)


def generate_spectrograms(entry, results_dir, sr=DEFAULT_SR, max_dur=DEFAULT_MAX_DUR):
    """Save 3-panel spectrogram PNG for the worst-performing file."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    n_fft = 2048
    hop = 512

    signals = [
        ("Original", entry["original"]),
        ("torch-aac decoded", entry["taac_decoded"]),
        ("FFmpeg decoded", entry["ff_decoded"]),
    ]

    for ax, (title, sig) in zip(axes, signals):
        sig = sig[:sr * int(max_dur)]
        num_frames = max(1, (len(sig) - n_fft) // hop + 1)
        frames = np.lib.stride_tricks.as_strided(
            sig, shape=(num_frames, n_fft),
            strides=(sig.strides[0] * hop, sig.strides[0]),
        )
        window = np.hanning(n_fft).astype(np.float32)
        spec = np.abs(np.fft.rfft(frames * window, axis=1))
        spec_db = 20 * np.log10(np.maximum(spec, 1e-10))

        im = ax.imshow(
            spec_db.T, aspect="auto", origin="lower",
            extent=[0, len(sig) / sr, 0, sr / 2 / 1000],
            vmin=-80, vmax=0, cmap="magma",
        )
        ax.set_title(f"{title} - {entry['file']} @ {entry['bitrate'] // 1000}k")
        ax.set_ylabel("Freq (kHz)")
        fig.colorbar(im, ax=ax, label="dB")

    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    out_path = results_dir / "spectrograms.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Spectrograms saved to {out_path}")


def print_summary(results, bitrates=DEFAULT_BITRATES, device="cpu"):
    """Print summary statistics."""
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    taac_snrs = [r["torch_aac_snr"] for r in results]
    ff_snrs = [r["ffmpeg_snr"] for r in results]
    deltas = [r["delta_snr"] for r in results]

    print(f"\n  Average torch-aac SNR:  {np.mean(taac_snrs):.1f} dB")
    print(f"  Average FFmpeg SNR:     {np.mean(ff_snrs):.1f} dB")
    print(f"  Average SNR delta:      {np.mean(deltas):+.1f} dB "
          f"({'torch-aac better' if np.mean(deltas) > 0 else 'FFmpeg better'})")

    # Bitrate accuracy
    for br in bitrates:
        br_results = [r for r in results if r["bitrate"] == br]
        if not br_results:
            continue
        taac_errs = [(r["torch_aac_kbps"] - br / 1000) / (br / 1000) * 100
                     for r in br_results]
        ff_errs = [(r["ffmpeg_kbps"] - br / 1000) / (br / 1000) * 100
                   for r in br_results]
        print(f"\n  Bitrate accuracy @ {br // 1000}k:")
        print(f"    torch-aac: {np.mean(taac_errs):+.1f}% avg error "
              f"(range {min(taac_errs):+.1f}% to {max(taac_errs):+.1f}%)")
        print(f"    FFmpeg:    {np.mean(ff_errs):+.1f}% avg error "
              f"(range {min(ff_errs):+.1f}% to {max(ff_errs):+.1f}%)")

    # Files where FFmpeg wins by >3 dB
    ff_wins = [r for r in results if r["delta_snr"] < -3.0]
    if ff_wins:
        print(f"\n  Files where FFmpeg wins by >3 dB ({len(ff_wins)}):")
        for r in ff_wins:
            print(f"    {r['file']} @ {r['bitrate'] // 1000}k: "
                  f"delta {r['delta_snr']:+.1f} dB")
    else:
        print(f"\n  No files where FFmpeg wins by >3 dB")

    # By genre
    genres = set(r["genre"] for r in results)
    print(f"\n  By genre:")
    for genre in sorted(genres):
        g_results = [r for r in results if r["genre"] == genre]
        avg_delta = np.mean([r["delta_snr"] for r in g_results])
        avg_taac = np.mean([r["torch_aac_snr"] for r in g_results])
        print(f"    {genre:<12}: avg SNR {avg_taac:.1f} dB, "
              f"avg delta vs FFmpeg {avg_delta:+.1f} dB")

    # Throughput
    if device == "cuda":
        print(f"\n  Throughput (CUDA):")
        for br in bitrates:
            br_results = [r for r in results if r["bitrate"] == br]
            if not br_results:
                continue
            taac_avg = np.mean([r["torch_aac_time_ms"] for r in br_results])
            ff_avg = np.mean([r["ffmpeg_time_ms"] for r in br_results])
            avg_dur = np.mean([r["duration_s"] for r in br_results])
            taac_rt = avg_dur / (taac_avg / 1000)
            ff_rt = avg_dur / (ff_avg / 1000)
            print(f"    {br // 1000}k: torch-aac {taac_avg:.0f} ms ({taac_rt:.0f}x rt), "
                  f"FFmpeg {ff_avg:.0f} ms ({ff_rt:.0f}x rt), "
                  f"speedup {ff_avg / taac_avg:.2f}x")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(args)
