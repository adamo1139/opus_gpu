from functools import lru_cache
import gzip
from importlib.resources import files
from itertools import accumulate, chain

import numpy as np


# The nested SILK/CELT calls (including recursive band splitting) use the
# CUDA thread's call stack in addition to the per-stream scratch allocation.
_DECODER_STACK_BYTES = 16 * 1024


def _ensure_decoder_stack(runtime):
    limit = runtime.cudaLimitStackSize
    if runtime.deviceGetLimit(limit) < _DECODER_STACK_BYTES:
        runtime.deviceSetLimit(limit, _DECODER_STACK_BYTES)


@lru_cache(maxsize=8)
def _module(device):
    import cupy as cp

    info = cp.cuda.runtime.getDeviceProperties(device)
    if (info["major"], info["minor"]) < (8, 6):
        raise RuntimeError("CUDA decoding requires compute capability 8.6 or newer")
    ptx = gzip.decompress(files("opus_gpu").joinpath("decoder.ptx.gz").read_bytes())
    module = cp.cuda.function.Module()
    module.load(ptx + b"\0")
    size = cp.empty(1, dtype=cp.int32)
    module.get_function("decoder_size")((1,), (1,), (size,))
    return module, int(size.get()[0]), info["multiProcessorCount"]


def decode_group(streams, paths):
    try:
        import cupy as cp
    except ImportError as error:
        raise RuntimeError(
            "Install opus-gpu[cuda12] or a CuPy build for your CUDA version"
        ) from error

    module, stride, multiprocessors = _module(cp.cuda.runtime.getDevice())
    _ensure_decoder_stack(cp.cuda.runtime)
    count = len(streams)
    packet_starts = list(accumulate((len(s.packets) for s in streams), initial=0))
    packets = list(chain.from_iterable(s.packets for s in streams))
    packet_offsets = np.fromiter(
        accumulate(map(len, packets), initial=0), dtype=np.int64
    )
    offsets = list(accumulate((s.frames * s.channels for s in streams), initial=0))
    packed = np.frombuffer(b"".join(packets), dtype=np.uint8)
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
    )
    # Spread small batches across SMs before filling all 32 lanes of a warp.
    lanes = min(32, 1 << max(0, (count // multiprocessors).bit_length() - 1))
    module.get_function("decode_streams")(
        ((count + lanes - 1) // lanes,), (lanes,), args
    )
    errors = status.get()
    for index in np.flatnonzero(errors):
        raise ValueError(f"{paths[index]}: libopus error {errors[index]}")
    pcm = output.get()
    for index, stream in enumerate(streams):
        item = pcm[offsets[index] : offsets[index + 1]].reshape(-1, stream.channels)
        # Return independent arrays so retaining one file does not pin the batch.
        yield item[stream.skip : stream.end].copy()
