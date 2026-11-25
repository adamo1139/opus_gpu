"""Compare CPU and CUDA on matched sets of 1, 10, and 60 second clips."""

import argparse
import os
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory


def make_corpus(folder, seconds, count):
    folder.mkdir()
    for index, (rate, channels, mode, bitrate, frame) in enumerate(
        [
            (16000, 1, "voip", "9k", "20"),
            (48000, 1, "voip", "23k", "20"),
            (48000, 2, "lowdelay", "128k", "2.5"),
            (48000, 1, "audio", "64k", "20"),
        ]
    ):
        tone = "0.2*sin(2*PI*330*t)" + (
            "|0.12*sin(2*PI*770*t)" if channels == 2 else ""
        )
        path = folder / f"{index}.opus"
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                f"aevalsrc={tone}:s={rate}:d={seconds}",
                "-c:a",
                "libopus",
                "-application",
                mode,
                "-b:a",
                bitrate,
                "-frame_duration",
                frame,
                path,
            ],
            check=True,
        )
        for number in range(index + 4, count, 4):
            os.link(path, folder / f"{number}.opus")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--durations", nargs="+", type=int, choices=(1, 10, 60), default=(1, 10, 60)
    )
    parser.add_argument("--files", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--cpu-workers", type=int, default=4, help="workers for the CPU backend"
    )
    parser.add_argument("--backend", choices=("cpu", "cuda", "both"), default="both")
    args = parser.parse_args()
    if args.cpu_workers < 1:
        parser.error("--cpu-workers must be positive")
    if args.files < 4 or args.files % 4 or args.batch_size < 1:
        parser.error("use a positive batch size and a positive multiple of four files")
    with TemporaryDirectory(prefix="opus-durations-") as tmp:
        for seconds in args.durations:
            folder = Path(tmp) / str(seconds)
            make_corpus(folder, seconds, args.files)
            times = {}
            for backend in (
                ("cpu", "cuda") if args.backend == "both" else (args.backend,)
            ):
                workers = args.cpu_workers if backend == "cpu" else 4
                print(
                    f"{seconds}s clips, {args.files} files, batch {args.batch_size}, {backend}, workers={workers}",
                    flush=True,
                )
                command = [
                    sys.executable,
                    "-m",
                    "opus_gpu",
                    str(folder),
                    "--backend",
                    backend,
                    "--workers",
                    str(workers),
                    "--batch-size",
                    str(args.batch_size),
                ]
                run = subprocess.run(
                    command,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=Path(__file__).resolve().parents[1],
                )
                print(run.stderr, end="", flush=True)
                run.check_returncode()
                times[backend] = float(
                    re.search(
                        rf"{backend}: {args.files} files in ([\d.]+)s", run.stderr
                    )[1]
                )
                print(f"RTF {times[backend] / (args.files * seconds):.6f}", flush=True)
            if len(times) == 2:
                print(
                    f"CPU time / CUDA time = {times['cpu'] / times['cuda']:.2f}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
