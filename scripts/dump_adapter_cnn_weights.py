#!/usr/bin/env python3
"""Dump ADAPTed BoundariesCNN weights from ONNX to a flat little-endian f32 blob.

The GPU/batched CNN path (escapepod-demux, `gpu`/`cnn-detect`) needs the conv
weights as raw arrays; Rust has no ONNX-weight extractor, so this companion
file is produced alongside the ONNX. Layout (concatenated, in this order; all
shapes fixed by the architecture — 3x Conv1d + ConvTranspose1d, k=7, 64 ch):

    0.weight [64,1,7]  0.bias [64]
    2.weight [64,64,7] 2.bias [64]
    4.weight [64,64,7] 4.bias [64]
    6.weight [64,2,7]  6.bias [2]     (ConvTranspose: [Cin=64, Cout=2, K=7])

Usage:
    python dump_adapter_cnn_weights.py adapter_cnn_rna004.onnx adapter_cnn_rna004.weights
"""

import sys
import numpy as np
import onnx
from onnx import numpy_helper

ORDER = ["0.weight", "0.bias", "2.weight", "2.bias",
         "4.weight", "4.bias", "6.weight", "6.bias"]


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(f"usage: {sys.argv[0]} <model.onnx> <out.weights>")
    model = onnx.load(sys.argv[1])
    inits = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    missing = [n for n in ORDER if n not in inits]
    if missing:
        sys.exit(f"error: ONNX missing expected initializers: {missing}")
    with open(sys.argv[2], "wb") as f:
        for name in ORDER:
            arr = np.ascontiguousarray(inits[name], dtype="<f4")
            f.write(arr.tobytes())
            print(f"  {name}: {list(arr.shape)} ({arr.size} f32)")
    total = sum(inits[n].size for n in ORDER)
    print(f"Wrote {total} f32 ({4*total} bytes) -> {sys.argv[2]}")


if __name__ == "__main__":
    main()
