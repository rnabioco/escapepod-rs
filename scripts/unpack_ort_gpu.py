#!/usr/bin/env python3
"""Unpack the onnxruntime-gpu wheel fetched by `pixi run -e gpu install-ort`.

The wheel is just a zip container for the CUDA-enabled libonnxruntime that
`ort` (built with `load-dynamic`) dlopens at run time; nothing is installed
into any Python environment. `scripts/gpu_env_activation.sh` points
ORT_DYLIB_PATH at the extracted library.
"""

import sys
import zipfile
from pathlib import Path


def main() -> None:
    dest = Path(sys.argv[1])
    wheels = sorted(dest.glob("onnxruntime_gpu-*.whl"))
    if not wheels:
        sys.exit(f"no onnxruntime_gpu-*.whl in {dest} — did `pip download` fail?")
    wheel = wheels[-1]
    zipfile.ZipFile(wheel).extractall(dest)
    dylibs = sorted((dest / "onnxruntime" / "capi").glob("libonnxruntime.so.*"))
    if not dylibs:
        sys.exit(f"{wheel.name} unpacked but contains no capi/libonnxruntime.so.*")
    print(f"unpacked {wheel.name}")
    print(f"ORT dylib: {dylibs[-1]}")


if __name__ == "__main__":
    main()
