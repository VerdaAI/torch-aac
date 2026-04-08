"""CLI entry point: python -m torch_aac

Supports encoding audio files and pipe-based I/O for FFmpeg integration.

Usage:
    # File-based encoding
    python -m torch_aac -i input.wav -o output.aac -b 128k

    # Pipe mode (FFmpeg integration)
    ffmpeg -i input.mp4 -f f32le -ar 48000 -ac 1 pipe:1 | python -m torch_aac -b 128k > output.aac
"""

from __future__ import annotations

import argparse
import sys


def parse_bitrate(s: str) -> int:
    """Parse bitrate string like '128k' or '320000' to integer bps."""
    s = s.strip().lower()
    if s.endswith("k"):
        return int(float(s[:-1]) * 1000)
    return int(s)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="torch-aac",
        description="GPU-accelerated AAC-LC encoder",
    )
    parser.add_argument("-i", "--input", help="Input WAV file path")
    parser.add_argument("-o", "--output", help="Output AAC file path")
    parser.add_argument(
        "-b", "--bitrate", default="128k",
        help="Target bitrate (e.g., 128k, 320000). Default: 128k",
    )
    parser.add_argument(
        "--sample-rate", type=int, default=48000,
        help="Sample rate in Hz (for pipe mode). Default: 48000",
    )
    parser.add_argument(
        "--channels", type=int, default=1,
        help="Number of channels (for pipe mode). Default: 1",
    )
    parser.add_argument(
        "--device", default="auto",
        help="PyTorch device (auto, cuda, cpu). Default: auto",
    )
    parser.add_argument(
        "--version", action="version",
        version=f"%(prog)s {_get_version()}",
    )

    args = parser.parse_args()
    bitrate = parse_bitrate(args.bitrate)

    if args.input and args.output:
        # File mode
        from torch_aac import encode_file

        encode_file(args.input, args.output, bitrate=bitrate, device=args.device)
        print(f"Encoded {args.input} → {args.output}", file=sys.stderr)

    elif args.input:
        # Input file, output to stdout
        from torch_aac import AACEncoder
        from torch_aac.utils.audio_io import read_audio

        pcm, sr = read_audio(args.input)
        channels = 1 if pcm.ndim == 1 else pcm.shape[-1]
        with AACEncoder(
            sample_rate=sr, channels=channels, bitrate=bitrate, device=args.device
        ) as enc:
            aac_bytes = enc.encode(pcm)
            sys.stdout.buffer.write(aac_bytes)

    else:
        # Pipe mode: read f32le PCM from stdin, write ADTS to stdout
        import numpy as np

        from torch_aac import AACEncoder

        pcm_data = sys.stdin.buffer.read()
        if not pcm_data:
            print("Error: no input data", file=sys.stderr)
            sys.exit(1)

        pcm = np.frombuffer(pcm_data, dtype=np.float32)
        if args.channels > 1:
            pcm = pcm.reshape(-1, args.channels)

        with AACEncoder(
            sample_rate=args.sample_rate,
            channels=args.channels,
            bitrate=bitrate,
            device=args.device,
        ) as enc:
            aac_bytes = enc.encode(pcm)
            sys.stdout.buffer.write(aac_bytes)


def _get_version() -> str:
    try:
        from torch_aac import __version__

        return __version__
    except ImportError:
        return "unknown"


if __name__ == "__main__":
    main()
