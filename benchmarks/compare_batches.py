"""Compare sustained decode throughput on 10-second Opus clips."""

import argparse
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

from compare_durations import make_corpus


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--backend", choices=("cpu", "cuda", "both"), default="both")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    if args.batches < 1 or args.batch_size < 4 or args.batch_size % 4:
        parser.error("use a positive batch count and a batch size divisible by four")

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from opus_gpu import SAMPLE_RATE, decode_many

    backends = ("cpu", "cuda") if args.backend == "both" else (args.backend,)
    if "cuda" in backends:
        try:
            import cupy as cp

            if not 0 <= args.device < cp.cuda.runtime.getDeviceCount():
                parser.error(f"CUDA device {args.device} is unavailable")
        except (ImportError, RuntimeError) as error:
            parser.error(str(error))

    with TemporaryDirectory(prefix="opus-batches-") as tmp:
        clips = Path(tmp) / "clips"
        make_corpus(clips, 10, args.batch_size)
        paths = sorted(clips.glob("*.opus"))
        results = {}
        for backend in backends:
            affinity = os.sched_getaffinity(0)
            if backend == "cpu":
                os.sched_setaffinity(0, {min(affinity)})
            else:
                cp.cuda.Device(args.device).use()
            workers = 1 if backend == "cpu" else 4
            try:
                warmup = perf_counter()
                for _ in decode_many(
                    paths, backend=backend, workers=workers, batch_size=args.batch_size
                ):
                    pass
                print(f"{backend}: warmup {perf_counter() - warmup:.3f}s", flush=True)
                start = perf_counter()
                count = samples = 0
                inputs = (path for _ in range(args.batches) for path in paths)
                for _, pcm in decode_many(
                    inputs, backend=backend, workers=workers, batch_size=args.batch_size
                ):
                    count += 1
                    samples += len(pcm)
                    if count % (10 * args.batch_size) == 0:
                        print(
                            f"{backend}: {count // args.batch_size}/{args.batches} batches",
                            flush=True,
                        )
                elapsed = perf_counter() - start
            finally:
                os.sched_setaffinity(0, affinity)
            assert count == args.batches * args.batch_size
            assert samples == count * 10 * SAMPLE_RATE
            audio = samples / SAMPLE_RATE
            results[backend] = elapsed
            print(
                f"{backend}: {count} files in {elapsed:.3f}s, {count / elapsed:.1f} files/s, "
                f"{audio / elapsed:.1f}x realtime, RTF {elapsed / audio:.6f}",
                flush=True,
            )
        if len(results) == 2:
            print(f"CPU time / CUDA time = {results['cpu'] / results['cuda']:.2f}x")


if __name__ == "__main__":
    main()
