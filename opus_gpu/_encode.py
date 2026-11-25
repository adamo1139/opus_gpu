"""Encode 48 kHz signed 16-bit PCM to mono/stereo Ogg Opus on CUDA."""

from itertools import accumulate, islice
import operator
import secrets
import struct

import numpy as np

from ._cuda import _ensure_stack, _module
from ._ogg import _page_crc

_FRAME = 960
_SKIP = 312  # Verified by OPUS_GET_LOOKAHEAD in the 48 kHz AUDIO kernel.


def _page(serial, sequence, flags, granule, packets):
    laces = bytearray()
    for packet in packets:
        full, rest = divmod(len(packet), 255)
        laces.extend(b"\xff" * full)
        laces.append(rest)
    header = struct.pack(
        "<4sBBQIIIB", b"OggS", 0, flags, granule, serial, sequence, 0, len(laces)
    )
    page = bytearray(header)
    page.extend(laces)
    page.extend(b"".join(packets))
    struct.pack_into("<I", page, 22, _page_crc(page))
    return page


def _mux(packets, frames, channels):
    serial = secrets.randbits(32)
    header = struct.pack("<8sBBHIhB", b"OpusHead", 1, channels, _SKIP, 48000, 0, 0)
    vendor = b"opus_gpu"
    tags = b"OpusTags" + struct.pack("<I", len(vendor)) + vendor + bytes(4)
    result = _page(serial, 0, 2, 0, [header])
    result.extend(_page(serial, 1, 0, 0, [tags]))
    # At most 40 * 6 laces even at the maximum legal packet size.
    for start in range(0, len(packets), 40):
        end = min(start + 40, len(packets))
        last = end == len(packets)
        granule = _SKIP + frames if last else end * _FRAME
        result.extend(
            _page(
                serial, 2 + start // 40, 4 if last else 0, granule, packets[start:end]
            )
        )
    return bytes(result)


def _pcm_array(pcm):
    pcm = np.asarray(pcm)
    if pcm.ndim == 1:
        pcm = pcm[:, None]
    if pcm.dtype != np.int16 or pcm.ndim != 2 or pcm.shape[1] not in (1, 2):
        raise ValueError("expected 48 kHz int16 PCM with one or two channels")
    return pcm


def _encode_group(group, bitrate, complexity):
    try:
        import cupy as cp
    except ImportError as error:
        raise RuntimeError("CUDA encoding requires CuPy") from error
    module, stride, sms = _module(cp.cuda.runtime.getDevice(), "encoder")
    _ensure_stack(cp.cuda.runtime, 32 * 1024)
    count = len(group)
    capacity = (bitrate + 399) // 400
    channels = [pcm.shape[1] for pcm in group]
    counts = [(len(pcm) + _SKIP + _FRAME - 1) // _FRAME for pcm in group]
    starts = list(accumulate(counts, initial=0))
    offsets = list(
        accumulate((n * c * _FRAME for n, c in zip(counts, channels)), initial=0)
    )
    need = offsets[-1] * 2 + starts[-1] * (capacity + 4) + count * stride
    if count > 1 and need > 0.85 * cp.cuda.runtime.memGetInfo()[0]:
        mid = count // 2
        yield from _encode_group(group[:mid], bitrate, complexity)
        yield from _encode_group(group[mid:], bitrate, complexity)
        return
    padded = np.zeros(offsets[-1], dtype=np.int16)
    for i, pcm in enumerate(group):
        padded[offsets[i] : offsets[i] + pcm.size].reshape(pcm.shape)[:] = pcm
    packets = cp.empty((starts[-1], capacity), dtype=cp.uint8)
    lengths = cp.empty(starts[-1], dtype=cp.int32)
    errors = cp.empty(count, dtype=cp.int32)
    args = (
        np.int32(count),
        cp.asarray(padded),
        cp.asarray(offsets, dtype=cp.int64),
        cp.asarray(starts, dtype=cp.int32),
        cp.asarray(channels, dtype=cp.int32),
        np.int32(bitrate),
        np.int32(complexity),
        np.int32(capacity),
        packets,
        lengths,
        cp.empty((count, stride), dtype=cp.uint8),
        np.int32(stride),
        errors,
    )
    lanes = min(32, 1 << max(0, (count // sms).bit_length() - 1))
    module.get_function("encode_streams")(
        ((count + lanes - 1) // lanes,), (lanes,), args
    )
    status = errors.get()
    for i in np.flatnonzero(status):
        raise ValueError(f"Opus encode item {i}: libopus error {status[i]}")
    sizes = lengths.get()
    data = packets.get()
    for i, pcm in enumerate(group):
        parts = [
            memoryview(data[j, : sizes[j]]) for j in range(starts[i], starts[i + 1])
        ]
        yield _mux(parts, len(pcm), channels[i])


def encode_many(inputs, *, bitrate=64000, complexity=5, batch_size=64):
    """Yield one Ogg Opus byte string per 48 kHz int16 PCM array, in order."""
    bitrate, complexity, batch_size = map(
        operator.index, (bitrate, complexity, batch_size)
    )
    if not 6000 <= bitrate <= 256000 or not 0 <= complexity <= 10 or batch_size < 1:
        raise ValueError(
            "expected bitrate 6000..256000, complexity 0..10 and a positive batch size"
        )
    arrays = map(_pcm_array, inputs)
    while group := list(islice(arrays, batch_size)):
        yield from _encode_group(group, bitrate, complexity)


def encode(pcm, *, bitrate=64000, complexity=5):
    """Encode one 48 kHz mono/stereo int16 array to Ogg Opus bytes on CUDA."""
    return next(encode_many([pcm], bitrate=bitrate, complexity=complexity))
