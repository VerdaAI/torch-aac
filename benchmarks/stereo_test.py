#!/usr/bin/env python3
"""
Stereo M/S Coding Test for torch-aac

Tests stereo channel-pair-element encoding quality on real stereo audio.
Scans a directory for stereo audio files (any format ffmpeg can read),
encodes via torch-aac, strict-decodes via FFmpeg, and reports SNR +
actual bitrate.

Usage:
    python benchmarks/stereo_test.py [--data-root DIR]

Defaults to benchmarks/test_audio/stereo/ which contains 3 example files
extracted from musdb18.
"""

import argparse
import glob
import os
import subprocess
import tempfile

import numpy as np
import soundfile as sf

from torch_aac import AACEncoder


def load_stereo_via_ffmpeg(path, target_sr=48000, max_dur=30.0):
    """Load any audio file as stereo float32 at target_sr via ffmpeg."""
    r = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-t", str(max_dur), "-i", path,
            "-map", "0:a:0",
            "-f", "f32le", "-ar", str(target_sr), "-ac", "2", "pipe:1",
        ],
        capture_output=True,
    )
    if r.returncode != 0:
        return None
    raw = np.frombuffer(r.stdout, dtype=np.float32)
    if raw.size % 2 != 0:
        raw = raw[:-1]
    return raw.reshape(-1, 2)


def main():
    parser = argparse.ArgumentParser(description="torch-aac stereo M/S test")
    parser.add_argument(
        "--data-root", type=str, default="benchmarks/test_audio/stereo",
        help="Directory to scan for stereo audio files",
    )
    parser.add_argument(
        "--sample-rate", type=int, default=48000,
        help="Target sample rate (default: 48000)",
    )
    parser.add_argument(
        "--max-dur", type=float, default=30.0,
        help="Max duration per file in seconds (default: 30)",
    )
    parser.add_argument(
        "--bitrates", type=int, nargs="+", default=[64000, 128000],
        help="Bitrates to test in bps (default: 64000 128000)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device: cuda, cpu, or mps (default: cuda)",
    )
    args = parser.parse_args()

    sr = args.sample_rate

    # Find stereo audio files -- any format ffmpeg can read
    files = []
    for ext in ["wav", "flac", "mp3", "mp4", "m4a", "ogg", "webm"]:
        files.extend(glob.glob(f"{args.data_root}/**/*.{ext}", recursive=True))
        files.extend(glob.glob(f"{args.data_root}/**/*.stem.{ext}", recursive=True))
    files = sorted(set(files))

    print(f"Found {len(files)} files in {args.data_root}")
    for f in files:
        print(f"  {f}")
    print()
    print(f"{'File':<45} {'BR':>5} {'SNR (dB)':>10} {'kbps':>8}  Status")
    print("-" * 90)

    for path in files:
        name = os.path.basename(path)[:43]
        audio = load_stereo_via_ffmpeg(path, target_sr=sr, max_dur=args.max_dur)
        if audio is None or audio.ndim != 2 or audio.shape[1] != 2:
            print(f"{name:<45} -- LOAD FAILED")
            continue

        audio = audio[: sr * int(args.max_dur)]
        duration = len(audio) / sr

        for bitrate in args.bitrates:
            enc = AACEncoder(
                sample_rate=sr, channels=2, bitrate=bitrate, device=args.device
            )
            aac = enc.encode(audio)

            with tempfile.NamedTemporaryFile(suffix=".aac", delete=False) as f:
                f.write(aac)
                aac_path = f.name
            wav_path = aac_path + ".wav"

            r = subprocess.run(
                ["ffmpeg", "-y", "-v", "error", "-xerror",
                 "-i", aac_path, "-f", "wav", wav_path],
                capture_output=True, text=True,
            )

            if r.returncode == 0:
                decoded, _ = sf.read(wav_path, dtype="float32")
                n = min(len(audio), len(decoded))
                skip = 2048
                o, d = audio[skip:n - skip], decoded[skip:n - skip]
                snr = 10 * np.log10(
                    np.mean(o**2) / max(np.mean((o - d) ** 2), 1e-20)
                )
                kbps = len(aac) * 8 / duration / 1000
                print(f"{name:<45} {bitrate // 1000:>4}k {snr:>9.1f}  {kbps:>6.1f}k  OK")
            else:
                err = r.stderr.split("\n")[0][:60]
                print(f"{name:<45} {bitrate // 1000:>4}k      FAIL  --      {err}")

            os.unlink(aac_path)
            if os.path.exists(wav_path):
                os.unlink(wav_path)


if __name__ == "__main__":
    main()
