"""Rebuild the packaged PTX from the unmodified Xiph Opus 1.5.2 release."""

import argparse
import gzip
import hashlib
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re
import subprocess
import tarfile
from tempfile import TemporaryDirectory

HERE = Path(__file__).resolve().parent
SHA256 = "65c1d2f78b9f2fb20082c38cbe47c951ad5839345876e46941612ee87f9a7ce1"


def sources(tree, filename, group):
    makefile = (tree / filename).read_text().replace("\\\n", " ")
    value = re.search(rf"^{group}\s*=([^\n]+)", makefile, re.M)[1]
    return [tree / path for path in value.split()]


def adapt_controls(tree):
    # PTX device functions take fixed arguments. All decoder controls used here
    # take zero or one integer/pointer, so carry that operand in a uintptr_t.
    for header, source, name in [
        ("include/opus.h", "src/opus_decoder.c", "opus_decoder_ctl"),
        ("include/opus_custom.h", "celt/celt_decoder.c", "opus_custom_decoder_ctl"),
    ]:
        for item in (header, source):
            path = tree / item
            code = path.read_text()
            signature = rf"int {name}\(([^;{{}}\n]*?), int request, \.\.\.\)"
            code, count = re.subn(
                signature,
                rf"int {name}_fixed(\1, int request, uintptr_t operand)",
                code,
            )
            assert count == 1
            if item == header:
                code += f"""
#ifndef {name}
#define {name}(st, ...) {name}_args(st, __VA_ARGS__, 0)
#define {name}_args(st, request, value, ...) {name}_fixed(st, request, (uintptr_t)(value))
#endif
"""
            else:
                code = re.sub(
                    r"va_list ap;|va_start\(ap, request\);|va_end\(ap\);", "", code
                )
                code = re.sub(r"va_arg\(ap, ([^)]*)\)", r"((\1)operand)", code)
            path.write_text(code)


def build(archive, tools, output, work):
    if hashlib.sha256(archive.read_bytes()).hexdigest() != SHA256:
        raise ValueError("expected the official opus-1.5.2.tar.gz release")
    with tarfile.open(archive) as tar:
        tar.extractall(work, filter="data")
    tree = work / "opus-1.5.2"
    adapt_controls(tree)
    stack = tree / "celt/stack_alloc.h"
    code = re.sub(
        r"#ifdef CELT_C.*?#endif /\* CELT_C \*/",
        '#include "lane_stack.h"',
        stack.read_text(),
        count=1,
        flags=re.S,
    )
    code = re.sub(
        r"^#define ALLOC_STACK char \*_saved_stack; \(global_stack = .*?$",
        "#define ALLOC_STACK SAVE_STACK",
        code,
        flags=re.M,
    )
    stack.write_text(code)
    clang = str(tools / "clang")
    resource = subprocess.check_output(
        [clang, "-print-resource-dir"], text=True
    ).strip()
    flags = [
        "-target",
        "nvptx64-nvidia-cuda",
        "-march=sm_86",
        "-ffreestanding",
        "-fno-builtin",
        "-fwrapv",
        "-nostdinc",
        "-isystem",
        resource + "/include",
        "-std=gnu11",
        "-O2",
        "-DOPUS_BUILD",
        "-DFIXED_POINT",
        "-DDISABLE_FLOAT_API",
        "-DNONTHREADSAFE_PSEUDOSTACK",
        "-I" + str(HERE),
        "-I" + str(HERE / "freestanding"),
    ]
    flags += [
        "-I" + str(tree / sub)
        for sub in ("include", "celt", "silk", "silk/fixed", "src")
    ]
    files = [
        tree / "src/opus.c",
        tree / "src/opus_decoder.c",
        tree / "src/extensions.c",
        HERE / "decode.c",
        HERE / "memory.c",
    ]
    files += sources(tree, "silk_sources.mk", "SILK_SOURCES")
    files += [
        p
        for p in sources(tree, "celt_sources.mk", "CELT_SOURCES")
        if p.stem != "celt_encoder"
    ]

    def compile_one(path):
        obj = work / (path.stem + ".bc")
        subprocess.run([clang, *flags, "-emit-llvm", "-c", path, "-o", obj], check=True)
        return obj

    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as pool:
        objects = list(pool.map(compile_one, files))
    linked = work / "linked.bc"
    optimized = work / "optimized.ll"
    subprocess.run([tools / "llvm-link", *objects, "-o", linked], check=True)
    subprocess.run(
        [
            tools / "opt",
            "-enable-new-pm=0",
            "-internalize",
            "-internalize-public-api-list=decode_streams,decoder_size",
            "-globaldce",
            "-O2",
            "-S",
            linked,
            "-o",
            optimized,
        ],
        check=True,
    )
    # LLVM's PTX backend uses this calling convention for CUDA launch entries.
    ir = re.sub(
        r"define (\w+ )*void @(decode_streams|decoder_size)\(",
        r"define ptx_kernel void @\2(",
        optimized.read_text(),
    )
    optimized.write_text(ir)
    ptx = subprocess.check_output(
        [tools / "llc", "-mcpu=sm_86", "-mattr=+ptx75", optimized, "-o", "-"]
    )
    for name in (b"decode_streams", b"decoder_size"):
        if b".entry " + name + b"(" not in ptx:
            raise RuntimeError(f"missing CUDA entry {name!r}")
    output.write_bytes(gzip.compress(ptx, mtime=0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--llvm-bin", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=HERE.parent / "opus_gpu/decoder.ptx.gz"
    )
    args = parser.parse_args()
    with TemporaryDirectory() as directory:
        build(args.archive, args.llvm_bin.resolve(), args.output, Path(directory))
