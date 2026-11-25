# opus_gpu

WIP : encoding and decoding OPUS coded on CUDA-enabled GPUs

The first release decodes Opus into 48 kHz signed 16-bit PCM. The CUDA backend runs fixed-point libopus on NVIDIA GPUs with compute capability **8.6 or newer**. It accepts single-stream mono and stereo Ogg Opus files (channel mapping family 0). The CPU backend uses FFmpeg's libopus decoder through PyAV.

## Decode a directory

From this checkout, with Python 3.10+ and a CUDA 12 environment:

```bash
python -m pip install '.[cuda12]'
python -m opus_gpu ./clips --backend cuda --output ./wav
```

The command finds `.opus` files recursively and writes a WAV for each one, preserving the directory structure. A single file is also accepted. Omit `--output` to measure decode throughput including file reads, host/device transfers, and returning PCM to the host. To compare the CPU and GPU on your corpus:

```bash
python -m opus_gpu ./clips --backend cpu
python -m opus_gpu ./clips --backend cuda
```

CUDA loads the packaged PTX and the driver compiles it for the active GPU on first use. That startup cost is included in the CLI timing. A CPU-only install uses `python -m pip install .`. An existing CuPy installation compatible with your CUDA environment can also run the GPU backend.

## Python

```python
from pathlib import Path
from opus_gpu import decode, decode_many

pcm = decode("speech.opus")  # NumPy int16, shape (frames, channels)

for path, pcm in decode_many(Path("clips").rglob("*.opus"), backend="cuda", batch_size=64):
    print(path, pcm.shape)
```

Both entry points default to CPU. On CUDA, each file has its own decoder state. A launch processes multiple files in parallel and processes each file's packets in order. The reader verifies Ogg checksums, honors header gain and pre-skip, and trims the output to the final granule position.

The CUDA backend raises the current device context's per-thread call stack limit to at least 16 KiB for nested decoder calls.

`decode_many` consumes paths lazily and yields results in input order. `batch_size` sets the number of files per launch. The CUDA backend prefetches the next compressed batch while the current one decodes. PCM memory scales with the total duration and channel count of the current batch, so use smaller batches for long recordings. The CLI accepts `--batch-size` and `--workers` for corpus-specific tuning.

## Verification

Run `python -m pytest -q -rs` with pytest and the FFmpeg CLI installed. The tests generate SILK, hybrid, and CELT streams and compare sample counts and PCM against FFmpeg/libopus. The CUDA comparison runs when a supported device is visible.

The packaged kernel targets `sm_86` PTX. It has been assembled offline for SM 8.6, 8.7, 8.9, 9.0, 10.0, and 12.0.

On a corpus of 2,048 files made from four short Opus clips, the RTX 5080 completed a CUDA run in 3.223 s (635 files/s). The i7-11700K completed a four-worker CPU run in 1.717 s (1,193 files/s). Both CLI runs used batches of 64 and returned PCM to host memory without writing WAV files.

To rebuild the included kernel with Clang/LLVM 15:

```bash
curl -LO https://downloads.xiph.org/releases/opus/opus-1.5.2.tar.gz
python native/build_kernel.py opus-1.5.2.tar.gz --llvm-bin /usr/lib/llvm-15/bin
```

The builder verifies the release checksum, adapts the decoder controls and per-stream scratch storage to CUDA, and emits the PTX packaged in the wheel. The upstream terms are in [OPUS-LICENSE.txt](opus_gpu/OPUS-LICENSE.txt).
