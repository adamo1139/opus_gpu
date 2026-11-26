import wave

import numpy as np
import pytest

from opus_gpu import decode, encode, encode_many
from opus_gpu._encode import _mux
from opus_gpu._ogg import _parse
from opus_gpu.encode_wav import _read_wav


def test_mux_continuity_crc_and_trimming():
    # A standard one-frame 20 ms Opus silence packet. Span two audio pages.
    packets = [b"\xf8\xff\xfe"] * 51
    frames = len(packets) * 960 - 312 - 77
    stream = _parse(_mux(packets, frames, 2))
    assert stream.end - stream.skip == frames
    assert stream.skip == 312
    assert stream.channels == 2
    assert stream.data == b"".join(packets)
    assert stream.packet_sizes.tolist() == [3] * 51


def test_encoder_input_validation():
    with pytest.raises(ValueError, match="int16"):
        encode(np.array([1.0], dtype=np.float32))
    with pytest.raises(ValueError, match="bitrate"):
        encode(np.zeros(100, dtype=np.int16), bitrate=0)
    assert list(encode_many([])) == []


def test_wav_input(tmp_path):
    path = tmp_path / "input.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setparams((1, 2, 48000, 0, "NONE", ""))
        wav.writeframes(b"\x12\x00" * 100)
    np.testing.assert_array_equal(
        _read_wav(path), np.full((100, 1), 18, dtype=np.int16)
    )


def test_cuda_encoder(tmp_path):
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
    t = np.arange(48017) / 48000
    stereo = np.column_stack(
        [10000 * np.sin(2 * np.pi * f * t) for f in (331, 751)]
    ).astype(np.int16)
    arrays = [stereo[:, :1], stereo, stereo[:1, :1], stereo[:0]]
    for i, (pcm, blob) in enumerate(
        zip(arrays, encode_many(arrays, batch_size=3), strict=True)
    ):
        stream = _parse(blob)
        assert stream.end - stream.skip == len(pcm)
        path = tmp_path / f"{i}.opus"
        path.write_bytes(blob)
        device_pcm = decode(path, backend="cuda", output="cupy")
        assert isinstance(device_pcm, cp.ndarray)
        assert device_pcm.shape == pcm.shape
        device_stream = _parse(encode(device_pcm))
        assert device_stream.end - device_stream.skip == len(pcm)
        if len(pcm):
            reference = decode(path)
            assert reference.shape == pcm.shape
            if len(pcm) > 1:
                for channel in range(pcm.shape[1]):
                    assert (
                        np.corrcoef(pcm[:, channel], reference[:, channel])[0, 1] > 0.98
                    )


def test_encoder_retries_smaller_batches(monkeypatch):
    import sys
    from types import SimpleNamespace

    from opus_gpu import _encode

    runtime = SimpleNamespace(
        getDevice=lambda: 0,
        memGetInfo=lambda: (2**30, 2**30),
    )
    cuda = SimpleNamespace(
        runtime=runtime, memory=SimpleNamespace(OutOfMemoryError=MemoryError)
    )
    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace(cuda=cuda))
    monkeypatch.setattr(_encode, "_module", lambda *args: (None, 1, 1))
    monkeypatch.setattr(_encode, "_ensure_stack", lambda *args: None)

    def launch(cp, module, stride, sms, group, *args):
        if len(group) > 1:
            raise MemoryError
        return np.array([[0xF8, 0xFF, 0xFE]], dtype=np.uint8), [3]

    monkeypatch.setattr(_encode, "_launch_encode", launch)
    inputs = [np.zeros(frames, dtype=np.int16) for frames in (8, 23, 71)]
    streams = [_parse(blob) for blob in encode_many(inputs)]
    assert [stream.end - stream.skip for stream in streams] == [8, 23, 71]
