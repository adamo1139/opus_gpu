import shutil
import subprocess
import struct
from unittest.mock import Mock

import numpy as np
import pytest

from opus_gpu import decode, decode_many
from opus_gpu._ogg import read_stream


def set_header_gain(path, gain):
    raw = bytearray(path.read_bytes())
    header = 27 + raw[26]
    page_end = header + sum(raw[27:header])
    struct.pack_into("<h", raw, header + 16, gain)
    raw[22:26] = bytes(4)
    crc = 0
    for byte in raw[:page_end]:
        crc ^= byte << 24
        for _ in range(8):
            crc = ((crc << 1) ^ (0x04C11DB7 if crc >> 31 else 0)) & 0xFFFFFFFF
    struct.pack_into("<I", raw, 22, crc)
    path.write_bytes(raw)


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg is required to generate comparison streams")
    root = tmp_path_factory.mktemp("opus")
    # Low-bitrate voice, wideband hybrid, and short CELT packets exercise all
    # three modes and include clips whose last packet needs sample trimming.
    cases = [
        ("silk", 16000, 1, "voip", "9k", "20", 0.793),
        ("hybrid", 48000, 1, "voip", "23k", "20", 2.137),
        ("celt", 48000, 2, "lowdelay", "128k", "2.5", 0.137),
        ("short", 48000, 1, "audio", "64k", "20", 0.001),
        ("long_packet", 48000, 2, "audio", "96k", "120", 0.63),
    ]
    records = []
    for name, rate, channels, mode, bitrate, frame_ms, duration in cases:
        path = root / (name + ".opus")
        tones = "0.2*sin(2*PI*330*t)" + (
            "|0.12*sin(2*PI*770*t)" if channels == 2 else ""
        )
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                f"aevalsrc={tones}:s={rate}:d={duration}",
                "-c:a",
                "libopus",
                "-application",
                mode,
                "-b:a",
                bitrate,
                "-frame_duration",
                frame_ms,
                # Force an OpusTags packet across an Ogg page boundary.
                *(
                    ["-metadata", "comment=" + "x" * 66000]
                    if name == "long_packet"
                    else []
                ),
                path,
            ],
            check=True,
        )
        if name == "celt":
            set_header_gain(path, -6 * 256)
        data = subprocess.check_output(
            [
                "ffmpeg",
                "-v",
                "error",
                "-c:a",
                "libopus",
                "-i",
                path,
                "-ar",
                "48000",
                "-f",
                "s16le",
                "-",
            ]
        )
        records.append((path, np.frombuffer(data, dtype="<i2").reshape(-1, channels)))
    return records


def check_pcm(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == np.int16
    diff = actual.astype(float) - expected
    # Fixed- and floating-point libopus have slightly different rounding.
    assert np.linalg.norm(diff) < 0.012 * np.linalg.norm(expected.astype(float))


def test_cpu(corpus):
    for path, expected in corpus:
        check_pcm(decode(path), expected)
        stream = read_stream(path)
        assert stream.end - stream.skip == len(expected)


def test_batch_order_and_partial_group(corpus):
    paths = [path for path, _ in corpus]
    results = list(decode_many(iter(paths), batch_size=3, workers=2))
    assert [path for path, _ in results] == paths
    for (_, pcm), (_, expected) in zip(results, corpus):
        check_pcm(pcm, expected)


def test_invalid_input(corpus, tmp_path):
    path = tmp_path / "bad.opus"
    raw = bytearray(corpus[0][0].read_bytes())
    raw[-1] ^= 1
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="checksum"):
        read_stream(path)
    with pytest.raises(ValueError, match="batch_size"):
        list(decode_many([], batch_size=0))
    assert list(decode_many([])) == []


def test_cuda(corpus):
    cp = pytest.importorskip("cupy")
    try:
        count = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError as error:
        if error.status != 100:
            raise
        count = 0
    if not count:
        pytest.skip("no CUDA device visible")
    info = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
    if (info["major"], info["minor"]) < (8, 6):
        pytest.skip("requires SM 8.6+")
    # Cross a warp boundary and leave an incomplete batch.
    batch = corpus * 10
    for (_, pcm), (_, expected) in zip(
        decode_many((p for p, _ in batch), backend="cuda", batch_size=37),
        batch,
        strict=True,
    ):
        check_pcm(pcm, expected)


@pytest.mark.parametrize("current", [1024, 16384, 32768])
def test_cuda_thread_stack(current):
    from opus_gpu._cuda import _ensure_decoder_stack

    runtime = Mock()
    runtime.cudaLimitStackSize = 0
    runtime.deviceGetLimit.return_value = current
    _ensure_decoder_stack(runtime)
    runtime.deviceGetLimit.assert_called_once_with(runtime.cudaLimitStackSize)
    if current < 16384:
        runtime.deviceSetLimit.assert_called_once_with(runtime.cudaLimitStackSize, 16384)
    else:
        runtime.deviceSetLimit.assert_not_called()


def test_prefetch_preserves_order_and_bounds(corpus, monkeypatch):
    from opus_gpu import _cuda

    paths = [p for p, _ in corpus] * 11
    consumed = []
    launched = []

    def source():
        for path in paths:
            consumed.append(path)
            yield path

    def collect(streams, group):
        assert len(streams) == len(group)
        launched.append(list(group))
        return (stream.end - stream.skip for stream in streams)

    monkeypatch.setattr(_cuda, "decode_group", collect)
    results = decode_many(source(), backend="cuda", batch_size=3)
    first = next(results)
    assert len(consumed) == 6
    assert [p for p, _ in [first, *results]] == paths
    assert max(map(len, launched)) == 3
