"""Decode Opus files into 48 kHz, (frames, channels) signed 16-bit PCM."""

from concurrent.futures import ThreadPoolExecutor
from itertools import islice
import os
from pathlib import Path

import av
import numpy as np

from ._ogg import read_stream

SAMPLE_RATE = 48000
__all__ = ["SAMPLE_RATE", "decode", "decode_many"]


def _decode_cpu(path):
    with av.open(str(path)) as container:
        stream = next(
            (s for s in container.streams.audio if s.codec_context.name == "opus"), None
        )
        if stream is None:
            raise ValueError(f"{path}: no Opus audio stream")
        layout = stream.codec_context.layout
        channels = len(layout.channels)
        codec = av.CodecContext.create("libopus", "r")
        codec.extradata = stream.codec_context.extradata
        codec.format = "s16"
        codec.layout = layout
        codec.sample_rate = SAMPLE_RATE
        blocks = []
        for packet in container.demux(stream):
            for frame in codec.decode(packet):
                blocks.append(
                    np.frombuffer(
                        frame.planes[0], dtype=np.int16, count=frame.samples * channels
                    )
                )
    return (
        np.concatenate(blocks).reshape(-1, channels)
        if blocks
        else np.empty((0, channels), dtype=np.int16)
    )


def decode(path, *, backend="cpu"):
    """Decode one file. Use decode_many to share CUDA launches across files."""
    if backend == "cpu":
        return _decode_cpu(path)
    if backend == "cuda":
        from ._cuda import decode_group

        return next(decode_group([read_stream(path)], [path]))
    raise ValueError("backend must be 'cpu' or 'cuda'")


def decode_many(paths, *, backend="cpu", workers=None, batch_size=64):
    """Yield (path, PCM) in input order. CUDA prefetches one compressed group."""
    if isinstance(paths, (str, bytes, os.PathLike)):
        raise TypeError("paths must be an iterable of paths")
    if backend not in ("cpu", "cuda"):
        raise ValueError("backend must be 'cpu' or 'cuda'")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if workers is None:
        workers = min(
            4,
            len(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else os.cpu_count() or 1,
        )
    source = map(Path, paths)
    groups = iter(lambda: list(islice(source, batch_size)), [])
    with ThreadPoolExecutor(max_workers=workers) as pool:
        if backend == "cpu":
            for group in groups:
                yield from zip(group, pool.map(_decode_cpu, group))
            return

        from ._cuda import decode_group

        group = next(groups, [])
        pending = pool.map(read_stream, group)
        while group:
            streams = list(pending)
            following = next(groups, [])
            pending = pool.map(read_stream, following)
            yield from zip(group, decode_group(streams, group))
            group = following
