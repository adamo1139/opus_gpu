# opus_gpu

WIP : encoding and decoding OPUS coded on CUDA-enabled GPUs

Decode Opus into 48 kHz signed 16-bit PCM, or encode mono/stereo PCM into Ogg Opus on the GPU. The CUDA backend runs fixed-point libopus on NVIDIA GPUs with compute capability **8.6 or newer**. It accepts single-stream mono and stereo Ogg Opus files (channel mapping family 0). The CPU backend uses FFmpeg's libopus decoder through PyAV.

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

## Encode a directory

The CUDA encoder accepts 48 kHz mono/stereo signed 16-bit PCM WAVs. It writes one Ogg Opus file per input with 20 ms packets, preserving relative paths. The default is 64 kbit/s CBR at libopus complexity 5.

```bash
python -m opus_gpu.encode_wav ./wav --output ./encoded --batch-size 64 --bitrate 64000
```

In Python, `encode(pcm)` returns Ogg Opus bytes and `encode_many(arrays)` yields one byte string per input in order. Set `bitrate` in bits/second and `complexity` from 0 to 10. The fixed-point libopus encoder runs on CUDA. The host handles WAV input and Ogg framing.

## Python

```python
from pathlib import Path
from opus_gpu import decode, decode_many

pcm = decode("speech.opus")  # NumPy int16, shape (frames, channels)

for path, pcm in decode_many(Path("clips").rglob("*.opus"), backend="cuda", batch_size=64):
    print(path, pcm.shape)
```

Both entry points default to CPU. On CUDA, each file has its own decoder state. A launch processes multiple files in parallel and processes each file's packets in order. The reader verifies Ogg checksums, honors header gain and pre-skip, and trims the output to the final granule position.

The decoder raises the CUDA context's per-thread stack limit to at least 16 KiB. The encoder uses at least 32 KiB.

`decode_many` consumes paths lazily and yields results in input order. `batch_size` is an upper bound. The decoder schedules up to 2,048 files per launch, groups similar streams in neighboring lanes, and splits groups when device memory is tight. The CUDA backend prefetches the next compressed batch while the current one decodes. PCM memory scales with the total duration and channel count of the current batch, so use smaller batches for long recordings. The CLI accepts `--batch-size` and `--workers` for corpus-specific tuning.

## Verification

Run `python -m pytest -q -rs` with pytest and the FFmpeg CLI installed. The tests generate SILK, hybrid, and CELT streams and compare sample counts and PCM against FFmpeg/libopus. The CUDA comparison runs when a supported device is visible.

The packaged kernel targets `sm_86` PTX. It has been assembled offline for SM 8.6, 8.7, 8.9, 9.0, 10.0, and 12.0.

## Decode benchmarks

### Effect of clip duration

### Single-thread CPU baseline

The CPU decoder ran with one worker. Linux CPU affinity restricted the process and all its threads to logical CPU 3 on the i7-11700K. Each row covers 256 files of the specified length with a batch size of 256. The RTX 5080 column reuses the earlier CUDA measurements with four host workers. The 1 s CUDA timing includes first use of the rebuilt kernel.

| Clip duration | CPU, 1 thread | CUDA wall time | CPU RTF | CUDA RTF | CPU time / CUDA time |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 s | 0.655 s | 11.279 s | 0.002559 | 0.044059 | 0.06× |
| 10 s | 5.490 s | 3.091 s | 0.002145 | 0.001207 | 1.78× |
| 60 s | 32.855 s | 15.330 s | 0.002139 | 0.000998 | 2.14× |

```bash
taskset -c 3 python benchmarks/compare_durations.py --backend cpu --cpu-workers 1
```

### Four-worker CPU baseline

One end-to-end run per backend and duration on an NVIDIA GeForce RTX 5080 and an Intel Core i7-11700K. Each run decoded 256 files in one requested batch of 256 with four host workers. The corpus contained 64 copies of each of four synthetic Opus profiles, generated at exactly the duration shown. Timing includes reading files and returning 48 kHz PCM to host memory. WAV output was disabled.

| Clip duration | CPU wall time | CUDA wall time | CPU RTF | CUDA RTF | CPU files/s | CUDA files/s | CPU time / CUDA time |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 s | 0.576 s | 11.279 s | 0.002250 | 0.044059 | 444.3 | 22.7 | 0.05× |
| 10 s | 5.062 s | 3.091 s | 0.001977 | 0.001207 | 50.6 | 82.8 | 1.64× |
| 60 s | 33.173 s | 15.330 s | 0.002160 | 0.000998 | 7.7 | 16.7 | 2.16× |

RTF is wall time divided by decoded audio duration. Lower is faster. The 1 s CUDA row was the first invocation after the kernel changed and includes startup and any driver JIT compilation. Each later row also started a fresh process, with the driver cache available. At 60 s, CUDA decoded 4 h 16 min of audio in 15.330 s (1,001.9× real time).

```bash
python benchmarks/compare_durations.py
```

### Earlier decoder baseline

On a corpus of 2,048 files made from four short Opus clips, the RTX 5080 completed a CUDA run in 3.223 s (635 files/s). The i7-11700K completed a four-worker CPU run in 1.717 s (1,193 files/s). Both CLI runs used batches of 64 and returned PCM to host memory without writing WAV files.

On 2,048 files averaging 15 s (512 copies each of 12, 14, 16 and 18 s clips), CUDA took 11.705 s with a batch of 2,048. A separate CPU run with four workers and a batch of 64 took 69.850 s. Both processed 8 h 32 min of audio. The GPU was 5.97× faster with those settings.

## Rebuild the kernels

Use Clang/LLVM 15 and the official Opus release archive.

```bash
curl -LO https://downloads.xiph.org/releases/opus/opus-1.5.2.tar.gz
python native/build_kernel.py opus-1.5.2.tar.gz --llvm-bin /usr/lib/llvm-15/bin
python native/build_kernel.py opus-1.5.2.tar.gz --llvm-bin /usr/lib/llvm-15/bin --codec encoder
```

The builder verifies the release checksum, adapts the decoder controls and per-stream scratch storage to CUDA, and emits the PTX packaged in the wheel. The upstream terms are in [OPUS-LICENSE.txt](opus_gpu/OPUS-LICENSE.txt).
