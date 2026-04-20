#!/usr/bin/env python3
"""Export ADAPTed's BoundariesCNN to ONNX.

The ADAPTed CNN weights (`rna004_130bps@v0.2.4.pth`) ship under CC BY-NC
4.0 and must not be redistributed in this MIT-licensed repo. This script
converts the local install's `.pth` into an ONNX graph that escpod can
load at runtime. The `.onnx` lives under `benchmarks/` and is
gitignored.

Run once after setting up the pixi `warpdemux-bench` env:

    pixi run -e warpdemux-bench python scripts/export_adapter_cnn_to_onnx.py

The resulting file (default: `benchmarks/adapter_cnn_rna004.onnx`) is
what `escpod demux detect --method cnn --cnn-model PATH` expects.

Notes
-----
* Input shape: `[batch, 1, length]`, float32. `length` is the downscaled
  signal window (by default the raw signal `[min_obs_adapter:]` pooled
  by `downscale_factor=10`).
* Output shape: `[batch, 2, length']`. Channel 0 scores adapter-end
  positions; channel 1 scores poly(A)-end.
* We export with `dynamic_axes` so the same model accepts any input
  length at runtime.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--pth",
        type=Path,
        default=None,
        help="Path to rna004_130bps@v0.2.4.pth (auto-resolved from the installed `adapted` package if unset).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("benchmarks/adapter_cnn_rna004.onnx"),
        help="Where to write the ONNX model (default: benchmarks/adapter_cnn_rna004.onnx).",
    )
    p.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17, which tract-onnx supports comfortably).",
    )
    p.add_argument(
        "--probe-length",
        type=int,
        default=5500 // 10,
        help="Dummy input length for the trace (max_obs_adapter - min_obs_adapter) / downscale_factor.",
    )
    args = p.parse_args()

    try:
        import torch
        from adapted.detect.cnn import BoundariesCNN, load_cnn_model
        from adapted import models as adapted_models
        import importlib.resources as pkg_resources
    except ImportError as e:
        print(f"error: {e}. Run inside `pixi run -e warpdemux-bench`.", file=sys.stderr)
        return 1

    pth_path = args.pth
    if pth_path is None:
        with pkg_resources.path(adapted_models, "rna004_130bps@v0.2.4.pth") as p:
            pth_path = Path(p)
    print(f"Loading weights from: {pth_path}", file=sys.stderr)

    model: BoundariesCNN = load_cnn_model(str(pth_path))
    model.eval()

    dummy = torch.zeros((1, 1, args.probe_length), dtype=torch.float32)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Stick to the legacy (TorchScript-based) exporter: torch 2.x's dynamo
    # exporter emits symbolic shape expressions like `((length - 1)//3) + 1`
    # that tract-onnx's shape parser can't handle. The legacy path produces
    # plain `Resize`/`Conv` shape dims that load fine.
    torch.onnx.export(
        model,
        dummy,
        str(args.out),
        input_names=["signal"],
        output_names=["scores"],
        dynamic_axes={
            "signal": {0: "batch", 2: "length"},
            "scores": {0: "batch", 2: "length_out"},
        },
        opset_version=args.opset,
        dynamo=False,
    )

    # The torch 2.x dynamo exporter defaults to writing weights to a sibling
    # `.onnx.data` file; inline them so downstream loaders only need one file.
    import onnx as _onnx

    loaded = _onnx.load(str(args.out), load_external_data=True)
    _onnx.save(loaded, str(args.out), save_as_external_data=False)
    sidecar = args.out.with_suffix(args.out.suffix + ".data")
    if sidecar.exists():
        sidecar.unlink()

    size_kb = args.out.stat().st_size / 1024
    print(f"Wrote {args.out} ({size_kb:.1f} KiB, opset {args.opset}, weights inlined)", file=sys.stderr)

    # Sanity: run once and print output shape.
    try:
        import onnxruntime as ort  # noqa: F401
        sess = __import__("onnxruntime").InferenceSession(str(args.out), providers=["CPUExecutionProvider"])
        out = sess.run(["scores"], {"signal": dummy.numpy()})[0]
        print(f"Sanity: output shape {out.shape}, dtype {out.dtype}", file=sys.stderr)
    except ImportError:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
