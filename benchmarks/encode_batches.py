"""Measure sustained CUDA Opus encoding from PCM already in host RAM."""

import argparse
from pathlib import Path
import sys
from time import perf_counter

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--bitrate", type=int, default=64000)
    parser.add_argument("--complexity", type=int, default=5)
    args = parser.parse_args()
    if args.batches < 1 or args.batch_size < 1:
        parser.error("batch count and size must be positive")
    if not 6000 <= args.bitrate <= 256000 or not 0 <= args.complexity <= 10:
        parser.error("use bitrate 6000..256000 and complexity 0..10")

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from opus_gpu import SAMPLE_RATE, encode_many

    try:
        import cupy as cp

        if not 0 <= args.device < cp.cuda.runtime.getDeviceCount():
            parser.error(f"CUDA device {args.device} is unavailable")
        cp.cuda.Device(args.device).use()
    except (ImportError, RuntimeError) as error:
        parser.error(str(error))

    seconds = 10
    t = np.arange(seconds * SAMPLE_RATE) / SAMPLE_RATE
    left = (10000 * np.sin(2 * np.pi * 331 * t)).astype(np.int16)
    right = (6000 * np.sin(2 * np.pi * 751 * t)).astype(np.int16)
    stereo = np.column_stack((left, right))
    clips = (left, left, stereo, right)
    batch = [clips[i % len(clips)] for i in range(args.batch_size)]
    options = dict(
        bitrate=args.bitrate, complexity=args.complexity, batch_size=args.batch_size
    )

    warmup = perf_counter()
    for _ in encode_many(batch, **options):
        pass
    print(f"cuda encode: warmup {perf_counter() - warmup:.3f}s", flush=True)

    start = perf_counter()
    count = encoded_bytes = 0
    inputs = (clip for _ in range(args.batches) for clip in batch)
    for ogg in encode_many(inputs, **options):
        count += 1
        encoded_bytes += len(ogg)
        if count % (10 * args.batch_size) == 0:
            print(
                f"cuda encode: {count // args.batch_size}/{args.batches} batches",
                flush=True,
            )
    elapsed = perf_counter() - start
    if count != args.batches * args.batch_size:
        raise RuntimeError(f"unexpected output count: {count}")
    audio = count * seconds
    print(
        f"cuda encode: {count} files in {elapsed:.3f}s, {count / elapsed:.1f} files/s, "
        f"{audio / elapsed:.1f}x realtime, RTF {elapsed / audio:.6f}, "
        f"{encoded_bytes} Ogg bytes"
    )


if __name__ == "__main__":
    main()
