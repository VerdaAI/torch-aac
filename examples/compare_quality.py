"""Example: Encode a WAV file at multiple bitrates and compare quality.

Takes a user-provided WAV, encodes it at 48/128/320 kbps with torch-aac,
decodes each output with FFmpeg, and reports SNR vs the original. Also
writes each decoded result back to a .wav so you can listen for yourself.

Usage:
    python examples/compare_quality.py path/to/input.wav
    python examples/compare_quality.py path/to/input.wav --bitrates 64000 128000

Requires ffmpeg on PATH for decoding the output.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

import torch_aac


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    """Read a mono or stereo WAV file to float32 in [-1, 1]."""
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        n_channels = w.getnchannels()
        sampwidth = w.getsampwidth()
        n_frames = w.getnframes()
        raw = w.readframes(n_frames)

    if sampwidth == 2:
        pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        pcm = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth * 8} bits")

    if n_channels == 2:
        pcm = pcm.reshape(-1, 2).T  # (2, T)
    # Mono stays 1-D

    return pcm, sr


def write_wav(path: Path, pcm: np.ndarray, sr: int) -> None:
    """Write float32 PCM to a 16-bit WAV."""
    if pcm.ndim == 2:
        # (C, T) → interleaved
        interleaved = pcm.T.reshape(-1)
        n_channels = pcm.shape[0]
    else:
        interleaved = pcm
        n_channels = 1

    i16 = np.clip(interleaved * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(n_channels)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(i16.tobytes())


def decode_aac_to_pcm(aac_bytes: bytes, sr: int, n_channels: int) -> np.ndarray:
    """Decode AAC bytes back to float32 PCM using FFmpeg."""
    with tempfile.NamedTemporaryFile(suffix=".aac", delete=False) as f:
        f.write(aac_bytes)
        aac_path = f.name
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-i",
                aac_path,
                "-f",
                "f32le",
                "-ar",
                str(sr),
                "-ac",
                str(n_channels),
                "pipe:1",
            ],
            capture_output=True,
            check=True,
        )
    finally:
        Path(aac_path).unlink(missing_ok=True)

    decoded = np.frombuffer(result.stdout, dtype=np.float32)
    if n_channels == 2:
        decoded = decoded.reshape(-1, 2).T
    return decoded


def compute_snr(orig: np.ndarray, dec: np.ndarray) -> float:
    """Scale-invariant SNR in dB."""
    if orig.ndim == 2:
        orig = orig.mean(axis=0)
    if dec.ndim == 2:
        dec = dec.mean(axis=0)

    n = min(len(orig), len(dec))
    # Skip the first 1024 samples (MDCT warmup) so the delay doesn't
    # dominate the SNR for short signals.
    start = min(1024, n // 4)
    o = orig[start:n].astype(np.float64)
    d = dec[start:n].astype(np.float64)

    o_rms = np.sqrt(np.mean(o * o))
    d_rms = np.sqrt(np.mean(d * d))
    if d_rms < 1e-10 or o_rms < 1e-10:
        return float("-inf")
    # Normalize decoded to match original RMS (scale-invariant)
    d = d * (o_rms / d_rms)
    noise = d - o
    noise_rms = np.sqrt(np.mean(noise * noise))
    return 20.0 * float(np.log10(o_rms / (noise_rms + 1e-12)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare torch-aac quality across bitrates")
    parser.add_argument("input", type=Path, help="Input WAV file")
    parser.add_argument(
        "--bitrates",
        type=int,
        nargs="+",
        default=[48000, 128000, 320000],
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory to write decoded WAVs (default: same as input)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    out_dir = args.out_dir or args.input.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    pcm, sr = read_wav(args.input)
    n_channels = 1 if pcm.ndim == 1 else pcm.shape[0]
    duration = (pcm.shape[-1] / sr) if pcm.ndim >= 1 else 0.0
    print(f"Input: {args.input.name}  {duration:.2f}s  {sr} Hz  {n_channels} ch")
    print(f"torch_aac version: {torch_aac.__version__}")
    print()
    print(f"{'Bitrate':>10}  {'AAC bytes':>10}  {'Realized br':>13}  {'SNR (dB)':>10}")
    print("-" * 52)

    stem = args.input.stem
    for br in args.bitrates:
        aac_bytes = torch_aac.encode(pcm, sample_rate=sr, bitrate=br, channels=n_channels)
        realized_br = int(len(aac_bytes) * 8 / max(duration, 1e-6))
        decoded = decode_aac_to_pcm(aac_bytes, sr, n_channels)
        snr = compute_snr(pcm, decoded)

        out_wav = out_dir / f"{stem}__{br // 1000}k.wav"
        write_wav(out_wav, decoded, sr)

        print(f"{br // 1000:>8}k   {len(aac_bytes):>10}  {realized_br // 1000:>11}k  {snr:>10.1f}")
        print(f"            -> wrote {out_wav}")


if __name__ == "__main__":
    main()
