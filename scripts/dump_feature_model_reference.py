#!/usr/bin/env python3
"""Reference probabilities for a charging `feature_model` bundle, from the
bundle alone.

The oracle `examples/verify_feature_model.rs` is checked against. Given a
bundle and a corpus of per-base feature grids, it reproduces the model's input
tensor and runs the graph through **onnxruntime** — so a disagreement with
escpod isolates to tract or to the three rules escpod reproduces (fold,
standardisation, missingness), not to the feature computation, which both
sides read from the same file.

It deliberately imports **nothing from escapepod-models**. Every rule it
applies — which columns, in what order, folded which way, standardised by
what, and how a missing value is encoded — comes out of the bundle's own
`metadata.json`. That is the claim the format makes ("a consumer that computes
these differently gets a wrong answer, not an error, so the definition travels
with the weights"), and a script that reached for the training package instead
would not test it.

The corpus is a NumPy `[n_reads, n_offsets * 4]` float32 array of the
canonical `offsets x (dwell, mean, std, resid)` grid — escapepod-models writes
one as `<prefix>_F.npy`. Columns are selected from it by name, exactly as
`escapepod-classify` does.

Writes three files beside `--out`:

    <out>.json      metadata: shapes, channels, the bundle's identity
    <out>.grid.f32  the selected rows' FULL feature grid, row-major f32
    <out>.p.f64     P(classes[1]) per row, f64

The grid travels raw rather than as JSON because it carries NaN, which JSON
cannot represent, and because it is large. The Rust side reads the same bytes
and runs them through `ChargingBundle::select_columns`, so column selection is
under test too — the shipped models use a subset feature set, and the fixture
bundle does not.

Usage::

    python scripts/dump_feature_model_reference.py \\
        --bundle path/to/charging_fnn_rna004@v0.1.0 \\
        --corpus path/to/rep2_F.npy --n 4096 \\
        --out /tmp/fnn_ref
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

# The canonical per-base statistics, offsets-outer / stats-inner. This is the
# grid layout `<prefix>_F.npy` is written in and the one `features.order`
# names columns of; it is a property of the FORMAT, not of a model.
FEAT_STATS = ("dwell", "mean", "std", "resid")


def parse_column(name: str, offsets: list[int]) -> int:
    """`b<+/-offset>_<stat>` -> its index in the `offsets x FEAT_STATS` grid."""
    m = re.fullmatch(r"b([+-]?\d+)_(\w+)", name)
    if not m:
        raise SystemExit(f"feature name {name!r} is not b<offset>_<stat>")
    off, stat = int(m.group(1)), m.group(2)
    if off not in offsets:
        raise SystemExit(f"feature {name!r}: offset {off} not in recipe offsets")
    if stat not in FEAT_STATS:
        raise SystemExit(f"feature {name!r}: unknown stat {stat!r}")
    return offsets.index(off) * len(FEAT_STATS) + FEAT_STATS.index(stat)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, required=True,
                    help="<prefix>_F.npy: [n_reads, n_offsets*4] float32")
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--skip", type=int, default=0,
                    help="rows to skip, so a second run sees different reads")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    import onnxruntime as ort

    meta = json.loads((args.bundle / "metadata.json").read_text())
    fm = meta.get("feature_model")
    if fm is None:
        raise SystemExit(f"{args.bundle} carries no `feature_model` block")

    offsets = list(meta["features"]["offsets"])
    order = list(meta["features"]["order"])
    channels = list(fm["input"]["channels"])
    n_off = int(fm["input"]["n_offsets"])
    n_val = len(channels) // 2
    mu = np.asarray(fm["standardisation"]["mu"], dtype=np.float64)
    sd = np.asarray(fm["standardisation"]["sd"], dtype=np.float64)
    if len(order) != n_val * n_off:
        raise SystemExit(
            f"{len(order)} columns cannot fold into {n_val} x {n_off}")
    if len(mu) != n_val or len(sd) != n_val:
        raise SystemExit("one (mu, sd) per value channel is required")

    # --- the corpus rows ----------------------------------------------------
    F = np.load(args.corpus, mmap_mode="r")
    if F.shape[1] != len(offsets) * len(FEAT_STATS):
        raise SystemExit(
            f"corpus has {F.shape[1]} columns; the recipe's "
            f"{len(offsets)} offsets x {len(FEAT_STATS)} stats is "
            f"{len(offsets) * len(FEAT_STATS)} — wrong corpus for this bundle")
    grid = np.asarray(F[args.skip:args.skip + args.n], dtype=np.float32)
    print(f"{grid.shape[0]} rows x {grid.shape[1]} grid columns "
          f"({np.isnan(grid).mean():.1%} unresolved)")

    # --- select, fold, standardise, mask ------------------------------------
    idx = [parse_column(n, offsets) for n in order]
    Fs = grid[:, idx]
    # `features.order` is offsets-outer / channels-inner; the tensor is
    # [channel, offset].
    X = Fs.reshape(len(Fs), n_off, n_val).transpose(0, 2, 1)
    m = np.isfinite(X).astype(np.float32)
    Xv = np.nan_to_num(X, nan=0.0).astype(np.float32)
    for c in range(n_val):
        Xv[:, c] = (Xv[:, c] - float(mu[c])) / (float(sd[c]) if sd[c] > 0 else 1.0)
    Xv *= m
    Xn = np.concatenate([Xv, m], axis=1)

    # --- the graph ----------------------------------------------------------
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(
        str(args.bundle / fm["file"]), sess_options=so,
        providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    logits = sess.run(None, {in_name: Xn})[0].astype(np.float64)
    logits -= logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    p = e[:, 1] / e.sum(axis=1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    grid.astype("<f4").tofile(f"{args.out}.grid.f32")
    p.astype("<f8").tofile(f"{args.out}.p.f64")
    Path(f"{args.out}.json").write_text(json.dumps({
        "bundle": str(args.bundle),
        "model_id": meta["model"]["id"],
        "corpus": str(args.corpus),
        "skip": args.skip,
        "n_rows": int(grid.shape[0]),
        "n_grid_cols": int(grid.shape[1]),
        "n_selected": len(order),
        "channels": channels,
        "n_offsets": n_off,
        "onnxruntime": ort.__version__,
        "p_min": float(p.min()),
        "p_max": float(p.max()),
        "p_mean": float(p.mean()),
        "unresolved_fraction": float(np.isnan(grid).mean()),
    }, indent=1))
    print(f"wrote {args.out}.{{json,grid.f32,p.f64}}: "
          f"P in [{p.min():.4f}, {p.max():.4f}], mean {p.mean():.4f}")


if __name__ == "__main__":
    main()
