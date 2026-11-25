"""Encode a 48 kHz mono/stereo PCM16 WAV file or directory to Ogg Opus."""

import argparse
from itertools import islice
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
import wave

import numpy as np

from . import encode_many


def _read_wav(path):
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        if (
            wav.getframerate() != 48000
            or wav.getsampwidth() != 2
            or channels not in (1, 2)
        ):
            raise ValueError(
                f"{path}: expected 48 kHz, mono/stereo, signed 16-bit PCM WAV"
            )
        frames = wav.getnframes()
        raw = wav.readframes(frames)
        if len(raw) != frames * channels * 2:
            raise ValueError(f"{path}: truncated WAV file")
        return (
            np.frombuffer(raw, dtype="<i2")
            .astype(np.int16, copy=False)
            .reshape(-1, channels)
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", required=True, type=Path, help="output directory")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--bitrate", type=int, default=64000)
    parser.add_argument("--complexity", type=int, default=5)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch size must be positive")
    directory = args.source.is_dir()
    paths = args.source.rglob("*.wav") if directory else iter([args.source])
    total = 0
    while group := list(islice(paths, args.batch_size)):
        blobs = encode_many(
            map(_read_wav, group),
            bitrate=args.bitrate,
            complexity=args.complexity,
            batch_size=args.batch_size,
        )
        for path, blob in zip(group, blobs, strict=True):
            rel = path.relative_to(args.source) if directory else Path(path.name)
            target = args.output / rel.with_suffix(".opus")
            if target.resolve() == path.resolve():
                raise ValueError(f"output would overwrite input: {path}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile(dir=target.parent, delete=False) as temp:
                try:
                    temp.write(blob)
                    temp.close()
                    os.replace(temp.name, target)
                finally:
                    Path(temp.name).unlink(missing_ok=True)
            total += 1
    if not total:
        parser.exit(1, "No WAV files found\n")
    print(f"Encoded {total} WAV files")


if __name__ == "__main__":
    main()
