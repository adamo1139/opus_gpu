"""Decode an Ogg Opus file or a directory tree."""

import argparse
import os
from tempfile import NamedTemporaryFile
from pathlib import Path
import sys
from time import perf_counter
import wave

from . import SAMPLE_RATE, decode_many


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source", type=Path, help="file or directory containing .opus or .ogg files"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="write PCM WAVs here (default: measure decoding only)",
    )
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    directory = args.source.is_dir()
    paths = (
        (
            path
            for path in args.source.rglob("*")
            if path.suffix.lower() in {".opus", ".ogg"} and path.is_file()
        )
        if directory
        else [args.source]
    )
    start = perf_counter()
    last_report = start
    count = samples = 0
    try:
        for path, pcm in decode_many(
            paths,
            backend=args.backend,
            workers=args.workers,
            batch_size=args.batch_size,
        ):
            if args.output:
                relative = (
                    path.relative_to(args.source) if directory else Path(path.name)
                )
                dest = args.output / relative.with_suffix(".wav")
                dest.parent.mkdir(parents=True, exist_ok=True)
                if dest.resolve() == path.resolve():
                    raise ValueError(f"output would overwrite input: {path}")
                with NamedTemporaryFile(dir=dest.parent, delete=False) as temp:
                    try:
                        with wave.open(temp, "wb") as wav:
                            wav.setparams((pcm.shape[1], 2, SAMPLE_RATE, 0, "NONE", ""))
                            wav.writeframes(
                                pcm.astype("<i2", copy=False).ravel().data.cast("B")
                            )
                        temp.close()
                        os.replace(temp.name, dest)
                    finally:
                        Path(temp.name).unlink(missing_ok=True)
            count += 1
            samples += len(pcm)
            if directory and count % args.batch_size == 0:
                now = perf_counter()
                if now - last_report >= 5:
                    print(
                        f"{args.backend}: {count} files decoded in {now - start:.1f}s",
                        file=sys.stderr,
                        flush=True,
                    )
                    last_report = now
    except (RuntimeError, ValueError, OSError) as error:
        parser.exit(1, f"{error}\n")
    if not count:
        parser.exit(1, "No .opus or .ogg files found\n")
    elapsed = perf_counter() - start
    print(
        f"{args.backend}: {count} files in {elapsed:.3f}s, {count / elapsed:.1f} files/s, "
        f"{samples / SAMPLE_RATE / elapsed:.1f}x realtime",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
