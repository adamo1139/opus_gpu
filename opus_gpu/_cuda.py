from functools import lru_cache
import gzip
from importlib.resources import files
from itertools import accumulate

import numpy as np


# The nested SILK/CELT calls (including recursive band splitting) use the
# CUDA thread's call stack in addition to the per-stream scratch allocation.
_DECODER_STACK_BYTES = 16 * 1024


def _ensure_stack(runtime, minimum=_DECODER_STACK_BYTES):
    limit = runtime.cudaLimitStackSize
    if runtime.deviceGetLimit(limit) < minimum:
        runtime.deviceSetLimit(limit, minimum)


@lru_cache(maxsize=8)
def _module(device, codec="decoder"):
    import cupy as cp

    info = cp.cuda.runtime.getDeviceProperties(device)
    if (info["major"], info["minor"]) < (8, 6):
        raise RuntimeError("CUDA decoding requires compute capability 8.6 or newer")
    ptx = gzip.decompress(files("opus_gpu").joinpath(f"{codec}.ptx.gz").read_bytes())
    module = cp.cuda.function.Module()
    module.load(ptx + b"\0")
    size = cp.empty(1, dtype=cp.int32)
    module.get_function(f"{codec}_size")((1,), (1,), (size,))
    return module, int(size.get()[0]), info["multiProcessorCount"]


def decode_group(streams, paths):
    try:
        import cupy as cp
    except ImportError as error:
        raise RuntimeError(
            "Install opus-gpu[cuda12] or a CuPy build for your CUDA version"
        ) from error

    module, stride, multiprocessors = _module(cp.cuda.runtime.getDevice())
    _ensure_stack(cp.cuda.runtime)
    # Leave room for driver allocations and other CUDA work. Retry a smaller
    # group if another allocator wins the race after the memory query.
    free, _ = cp.cuda.runtime.memGetInfo()
    required = sum(
        2 * s.frames * s.channels + len(s.data) + 8 * len(s.packet_sizes) + stride + 32
        for s in streams
    )
    if len(streams) > 1 and required > 0.85 * free:
        yield from _split_group(streams, paths)
        return
    try:
        output, offsets = _launch_decode(
            cp, module, stride, multiprocessors, streams, paths
        )
    except cp.cuda.memory.OutOfMemoryError:
        if len(streams) == 1:
            raise
    else:
        # Copy bounded groups of consecutive files once; a consumer keeping one
        # view retains at most this transfer (or one very large file).
        start = 0
        while start < len(streams):
            end = start + 1
            while (
                end < len(streams) and offsets[end + 1] - offsets[start] <= 16 * 1024**2
            ):
                end += 1
            chunk = output[offsets[start] : offsets[end]].get()
            for index in range(start, end):
                stream = streams[index]
                pos = offsets[index] - offsets[start]
                item = chunk[pos : pos + stream.frames * stream.channels].reshape(
                    -1, stream.channels
                )
                yield item[stream.skip : stream.end]
            start = end
        return
    # Exit the exception handler first to release the failed launch's traceback.
    yield from _split_group(streams, paths)


def _split_group(streams, paths):
    middle = len(streams) // 2
    yield from decode_group(streams[:middle], paths[:middle])
    yield from decode_group(streams[middle:], paths[middle:])


def _launch_decode(cp, module, stride, multiprocessors, streams, paths):
    count = len(streams)
    packet_starts = list(accumulate((len(s.packet_sizes) for s in streams), initial=0))
    packet_offsets = np.empty(packet_starts[-1] + 1, dtype=np.int64)
    packet_offsets[0] = 0
    np.cumsum(
        np.concatenate([s.packet_sizes for s in streams]),
        dtype=np.int64,
        out=packet_offsets[1:],
    )
    offsets = list(accumulate((s.frames * s.channels for s in streams), initial=0))
    packed = np.frombuffer(b"".join(s.data for s in streams), dtype=np.uint8)
    # Neighboring lanes follow similar codec paths and finish closer together.
    order = sorted(
        range(count),
        key=lambda i: (
            streams[i].data[0] >> 3,
            streams[i].channels,
            len(streams[i].packet_sizes),
        ),
    )
    output = cp.empty(offsets[-1], dtype=cp.int16)
    status = cp.empty(count, dtype=cp.int32)
    # Keep every transfer and its allocation alive until the kernel has finished.
    args = (
        np.int32(count),
        cp.asarray(packed),
        cp.asarray(packet_offsets),
        cp.asarray(packet_starts, dtype=cp.int32),
        cp.asarray([s.channels for s in streams], dtype=cp.int32),
        cp.asarray([s.gain for s in streams], dtype=cp.int32),
        cp.asarray(offsets, dtype=cp.int64),
        output,
        cp.empty((count, stride), dtype=cp.uint8),
        np.int32(stride),
        status,
        cp.asarray(order, dtype=cp.int32),
    )
    # Spread small batches across SMs before filling all 32 lanes of a warp.
    lanes = min(32, 1 << max(0, (count // multiprocessors).bit_length() - 1))
    module.get_function("decode_streams")(
        ((count + lanes - 1) // lanes,), (lanes,), args
    )
    errors = status.get()
    for index in np.flatnonzero(errors):
        raise ValueError(f"{paths[index]}: libopus error {errors[index]}")
    return output, offsets
