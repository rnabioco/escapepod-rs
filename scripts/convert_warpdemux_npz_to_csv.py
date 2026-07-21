#!/usr/bin/env python3
"""Convert WarpDemuX barcode fingerprints (.npz) to escpod classify CSV.

WarpDemuX writes per-shard fingerprint files ``barcode_fpts_*.npz`` containing:
    - ``read_ids``: object array of UUID strings, shape (N,)
    - ``signals``:  float32 array of fingerprints, shape (N, F)  (F == 25)
(see ext/WarpDemuX/warpdemux/file_proc.py).

`escpod demux classify` reads fingerprints as CSV/Parquet with the schema
``read_id,fp_0,...,fp_{F-1}``. Feeding WarpDemuX's own fingerprints into escpod
classify isolates the SVM/DTW/Platt stage from escpod's fingerprint extraction
(the "Layer A" parity test in benchmarks/README.md).

Usage:
    python convert_warpdemux_npz_to_csv.py <npz_file_or_dir> <out.csv>

If the input is a directory, all ``barcode_fpts_*.npz`` shards are concatenated.
"""

import csv
import sys
from pathlib import Path

import numpy as np


def iter_npz(path: Path):
    """Yield (read_ids, signals) for each npz shard under `path`."""
    if path.is_dir():
        shards = sorted(path.glob("barcode_fpts_*.npz"))
        if not shards:
            shards = sorted(path.glob("*.npz"))
        if not shards:
            sys.exit(f"error: no .npz fingerprint shards found under {path}")
    else:
        shards = [path]

    for shard in shards:
        with np.load(shard, allow_pickle=True) as data:
            yield np.asarray(data["read_ids"]), np.asarray(data["signals"])


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(f"usage: {sys.argv[0]} <npz_file_or_dir> <out.csv>")
    in_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])

    n_written = 0
    n_features = None
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        header_written = False
        for read_ids, signals in iter_npz(in_path):
            if signals.ndim != 2:
                sys.exit(f"error: expected 2-D signals, got shape {signals.shape}")
            if n_features is None:
                n_features = signals.shape[1]
            elif signals.shape[1] != n_features:
                sys.exit(
                    f"error: inconsistent feature width "
                    f"({signals.shape[1]} vs {n_features})"
                )
            if not header_written:
                writer.writerow(["read_id"] + [f"fp_{i}" for i in range(n_features)])
                header_written = True
            for rid, fpt in zip(read_ids, signals):
                writer.writerow([str(rid)] + [f"{v:.6f}" for v in fpt])
                n_written += 1

    print(f"Wrote {n_written:,} fingerprints ({n_features} features) -> {out_path}")


if __name__ == "__main__":
    main()
