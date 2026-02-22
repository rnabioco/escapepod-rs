#!/usr/bin/env python3
"""Compare escapepod demux predictions against WarpDemuX ground truth.

Computes:
- Overall agreement percentage
- Per-barcode precision/recall/F1
- Confusion matrix (barcodes + unclassified)
- Confidence score correlation
- Boundary difference distribution (if boundaries provided)

Usage:
    python compare_demux_results.py \
        --escapepod-a pred_a.csv \
        --escapepod-b pred_b.csv \
        --warpdemux barcode_predictions_0.csv.gz \
        [--boundaries-escapepod bounds_b.csv] \
        [--boundaries-warpdemux detected_boundaries_0.csv.gz]
"""

import argparse
import csv
import gzip
import sys
from collections import Counter, defaultdict
from pathlib import Path


def read_csv(path: Path, delimiter=",") -> list[dict]:
    """Read a CSV file (supports .gz), returning list of row dicts."""
    open_fn = gzip.open if str(path).endswith(".gz") else open
    rows = []
    with open_fn(path, "rt") as f:
        # Skip comment headers (WarpDemuX uses # prefix)
        first_line = f.readline()
        if first_line.startswith("#"):
            header = first_line.lstrip("#").strip()
        else:
            header = first_line.strip()

        reader = csv.DictReader(
            f, fieldnames=header.split(delimiter), delimiter=delimiter
        )
        for row in reader:
            rows.append(row)
    return rows


def parse_escapepod_predictions(path: Path) -> dict[str, dict]:
    """Parse escapepod classify output CSV.

    Expected columns: read_id, predicted_barcode, confidence, is_confident, [p03, p04, ...]
    """
    preds = {}
    open_fn = gzip.open if str(path).endswith(".gz") else open
    with open_fn(path, "rt") as f:
        reader = csv.DictReader(f)
        for row in reader:
            read_id = row["read_id"]
            barcode_str = row["predicted_barcode"]
            if barcode_str == "unclassified":
                barcode = -1
            elif barcode_str.startswith("BC"):
                barcode = int(barcode_str[2:])
            else:
                barcode = int(barcode_str)
            confidence = float(row["confidence"])
            preds[read_id] = {
                "barcode": barcode,
                "confidence": confidence,
            }
            # Collect probability columns
            for key, val in row.items():
                if key.startswith("p") and key[1:].lstrip("-").isdigit():
                    preds[read_id][key] = float(val)
    return preds


def parse_warpdemux_predictions(path: Path) -> dict[str, dict]:
    """Parse WarpDemuX barcode_predictions CSV.

    Format: #read_id,predicted_barcode,confidence_score,p03,p04,...
    """
    preds = {}
    open_fn = gzip.open if str(path).endswith(".gz") else open
    with open_fn(path, "rt") as f:
        first_line = f.readline()
        if first_line.startswith("#"):
            header = first_line.lstrip("#").strip().split(",")
        else:
            header = first_line.strip().split(",")
            f.seek(0)  # re-read if no comment

        reader = csv.DictReader(f, fieldnames=header)
        for row in reader:
            if row.get("read_id", "").startswith("#"):
                continue
            read_id = row["read_id"]
            barcode = int(row["predicted_barcode"])
            confidence = float(row.get("confidence_score", row.get("confidence", 0)))
            preds[read_id] = {
                "barcode": barcode,
                "confidence": confidence,
            }
            for key, val in row.items():
                if key and key.startswith("p") and key[1:].lstrip("-").isdigit():
                    try:
                        preds[read_id][key] = float(val)
                    except (ValueError, TypeError):
                        pass
    return preds


def parse_boundaries(path: Path) -> dict[str, tuple[int, int]]:
    """Parse boundary CSV, returning read_id -> (adapter_start, adapter_end)."""
    bounds = {}
    open_fn = gzip.open if str(path).endswith(".gz") else open
    with open_fn(path, "rt") as f:
        first_line = f.readline()
        if first_line.startswith("#"):
            header = first_line.lstrip("#").strip().split(",")
        else:
            header = first_line.strip().split(",")

        reader = csv.DictReader(f, fieldnames=header)
        for row in reader:
            read_id = row.get("read_id", "")
            if read_id.startswith("#"):
                continue
            try:
                start = int(float(row.get("adapter_start", 0)))
                end = int(float(row.get("adapter_end", 0)))
                bounds[read_id] = (start, end)
            except (ValueError, TypeError):
                pass
    return bounds


def load_warpdemux_predictions_dir(path: Path) -> dict[str, dict]:
    """Load all WarpDemuX prediction shards from a directory or single file."""
    if path.is_file():
        return parse_warpdemux_predictions(path)

    # Directory: load all shards
    preds = {}
    for shard in sorted(path.glob("barcode_predictions_*.csv.gz")):
        preds.update(parse_warpdemux_predictions(shard))
    if not preds:
        # Try uncompressed
        for shard in sorted(path.glob("barcode_predictions_*.csv")):
            preds.update(parse_warpdemux_predictions(shard))
    return preds


def compute_metrics(
    escapepod: dict[str, dict],
    warpdemux: dict[str, dict],
    label: str = "escapepod",
) -> None:
    """Compute and print comparison metrics."""
    # Find common read IDs
    common_ids = set(escapepod.keys()) & set(warpdemux.keys())
    print(f"\n{'='*60}")
    print(f"Comparison: {label} vs WarpDemuX")
    print(f"{'='*60}")
    print(f"  Reads in {label}: {len(escapepod):,}")
    print(f"  Reads in WarpDemuX: {len(warpdemux):,}")
    print(f"  Common reads: {len(common_ids):,}")

    if not common_ids:
        print("  No common reads found!")
        return

    # Collect all barcode labels
    all_barcodes = sorted(
        set(
            [escapepod[rid]["barcode"] for rid in common_ids]
            + [warpdemux[rid]["barcode"] for rid in common_ids]
        )
    )

    # Overall agreement
    agree = sum(
        1
        for rid in common_ids
        if escapepod[rid]["barcode"] == warpdemux[rid]["barcode"]
    )
    print(f"\n  Overall agreement: {agree:,}/{len(common_ids):,} ({100*agree/len(common_ids):.2f}%)")

    # Confusion matrix
    confusion = defaultdict(lambda: defaultdict(int))
    for rid in common_ids:
        ep_bc = escapepod[rid]["barcode"]
        wdx_bc = warpdemux[rid]["barcode"]
        confusion[wdx_bc][ep_bc] += 1

    # Per-barcode metrics
    print(f"\n  {'Barcode':>10s}  {'TP':>6s}  {'FP':>6s}  {'FN':>6s}  {'Prec':>6s}  {'Recall':>6s}  {'F1':>6s}  {'Support':>8s}")
    print(f"  {'-'*10}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*8}")

    for bc in all_barcodes:
        bc_label = f"BC{bc:02d}" if bc >= 0 else "unclass"
        tp = confusion[bc][bc]
        fp = sum(confusion[other][bc] for other in all_barcodes if other != bc)
        fn = sum(confusion[bc][other] for other in all_barcodes if other != bc)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        support = tp + fn

        print(f"  {bc_label:>10s}  {tp:6d}  {fp:6d}  {fn:6d}  {precision:6.3f}  {recall:6.3f}  {f1:6.3f}  {support:8d}")

    # Confusion matrix display
    print(f"\n  Confusion matrix (rows=WarpDemuX, cols={label}):")
    bc_labels = [f"BC{bc:02d}" if bc >= 0 else "uncl" for bc in all_barcodes]
    header = "  " + " " * 10 + "".join(f"{l:>8s}" for l in bc_labels)
    print(header)
    for i, wdx_bc in enumerate(all_barcodes):
        row_label = bc_labels[i]
        row_vals = "".join(f"{confusion[wdx_bc][ep_bc]:8d}" for ep_bc in all_barcodes)
        print(f"  {row_label:>10s}{row_vals}")

    # Confidence correlation
    ep_confs = []
    wdx_confs = []
    for rid in common_ids:
        ep_confs.append(escapepod[rid]["confidence"])
        wdx_confs.append(warpdemux[rid]["confidence"])

    if ep_confs and wdx_confs:
        # Pearson correlation (manual to avoid numpy dependency)
        n = len(ep_confs)
        mean_ep = sum(ep_confs) / n
        mean_wdx = sum(wdx_confs) / n
        cov = sum((a - mean_ep) * (b - mean_wdx) for a, b in zip(ep_confs, wdx_confs)) / n
        std_ep = (sum((a - mean_ep) ** 2 for a in ep_confs) / n) ** 0.5
        std_wdx = (sum((b - mean_wdx) ** 2 for b in wdx_confs) / n) ** 0.5
        if std_ep > 0 and std_wdx > 0:
            corr = cov / (std_ep * std_wdx)
            print(f"\n  Confidence correlation (Pearson r): {corr:.4f}")
        print(f"  {label} mean confidence: {mean_ep:.4f}")
        print(f"  WarpDemuX mean confidence: {mean_wdx:.4f}")

    # Agreement breakdown by WarpDemuX confidence
    bins = [(0.0, 0.5), (0.5, 0.8), (0.8, 0.95), (0.95, 1.01)]
    print(f"\n  Agreement by WarpDemuX confidence:")
    print(f"  {'Confidence':>15s}  {'Agree':>8s}  {'Total':>8s}  {'Rate':>8s}")
    for lo, hi in bins:
        ids_in_bin = [
            rid for rid in common_ids if lo <= warpdemux[rid]["confidence"] < hi
        ]
        if ids_in_bin:
            agree_bin = sum(
                1
                for rid in ids_in_bin
                if escapepod[rid]["barcode"] == warpdemux[rid]["barcode"]
            )
            print(f"  [{lo:.2f}, {hi:.2f})  {agree_bin:8d}  {len(ids_in_bin):8d}  {100*agree_bin/len(ids_in_bin):7.2f}%")


def compare_boundaries(
    bounds_ep: dict[str, tuple[int, int]],
    bounds_wdx: dict[str, tuple[int, int]],
) -> None:
    """Compare boundary predictions between escapepod and WarpDemuX."""
    common_ids = set(bounds_ep.keys()) & set(bounds_wdx.keys())
    print(f"\n{'='*60}")
    print("Boundary comparison: escapepod LLR vs WarpDemuX CNN")
    print(f"{'='*60}")
    print(f"  Common reads: {len(common_ids):,}")

    if not common_ids:
        return

    start_diffs = []
    end_diffs = []
    for rid in common_ids:
        ep_start, ep_end = bounds_ep[rid]
        wdx_start, wdx_end = bounds_wdx[rid]
        start_diffs.append(ep_start - wdx_start)
        end_diffs.append(ep_end - wdx_end)

    def stats(diffs: list[int], label: str) -> None:
        abs_diffs = [abs(d) for d in diffs]
        mean_d = sum(diffs) / len(diffs)
        mean_abs = sum(abs_diffs) / len(abs_diffs)
        sorted_abs = sorted(abs_diffs)
        median_abs = sorted_abs[len(sorted_abs) // 2]
        p95 = sorted_abs[int(0.95 * len(sorted_abs))]
        print(f"  {label}:")
        print(f"    Mean diff: {mean_d:+.1f} samples")
        print(f"    Mean |diff|: {mean_abs:.1f} samples")
        print(f"    Median |diff|: {median_abs} samples")
        print(f"    95th percentile |diff|: {p95} samples")

    stats(start_diffs, "adapter_start")
    stats(end_diffs, "adapter_end")


def main():
    parser = argparse.ArgumentParser(
        description="Compare escapepod demux predictions against WarpDemuX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--escapepod-a",
        type=Path,
        help="Escapepod predictions using WarpDemuX boundaries (Layer A)",
    )
    parser.add_argument(
        "--escapepod-b",
        type=Path,
        help="Escapepod predictions using LLR boundaries (Layer B)",
    )
    parser.add_argument(
        "--warpdemux",
        type=Path,
        required=True,
        help="WarpDemuX predictions (file or directory with shards)",
    )
    parser.add_argument(
        "--boundaries-escapepod",
        type=Path,
        help="Escapepod LLR boundaries CSV (for boundary comparison)",
    )
    parser.add_argument(
        "--boundaries-warpdemux",
        type=Path,
        help="WarpDemuX detected boundaries CSV/dir (for boundary comparison)",
    )

    args = parser.parse_args()

    # Load WarpDemuX predictions
    print(f"Loading WarpDemuX predictions from {args.warpdemux}...")
    wdx_preds = load_warpdemux_predictions_dir(args.warpdemux)
    print(f"  Loaded {len(wdx_preds):,} predictions")

    # Layer A: same boundaries
    if args.escapepod_a:
        print(f"\nLoading Layer A predictions from {args.escapepod_a}...")
        ep_a = parse_escapepod_predictions(args.escapepod_a)
        print(f"  Loaded {len(ep_a):,} predictions")
        compute_metrics(ep_a, wdx_preds, label="Layer A (WDX boundaries)")

    # Layer B: full pipeline
    if args.escapepod_b:
        print(f"\nLoading Layer B predictions from {args.escapepod_b}...")
        ep_b = parse_escapepod_predictions(args.escapepod_b)
        print(f"  Loaded {len(ep_b):,} predictions")
        compute_metrics(ep_b, wdx_preds, label="Layer B (LLR boundaries)")

    # Boundary comparison
    if args.boundaries_escapepod and args.boundaries_warpdemux:
        print(f"\nLoading boundaries...")
        bounds_ep = parse_boundaries(args.boundaries_escapepod)
        # WarpDemuX boundaries may be sharded
        if args.boundaries_warpdemux.is_dir():
            bounds_wdx = {}
            for shard in sorted(args.boundaries_warpdemux.glob("detected_boundaries_*.csv.gz")):
                bounds_wdx.update(parse_boundaries(shard))
        else:
            bounds_wdx = parse_boundaries(args.boundaries_warpdemux)
        compare_boundaries(bounds_ep, bounds_wdx)


if __name__ == "__main__":
    main()
