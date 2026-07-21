#!/usr/bin/env python
"""Benchmark the escapepod Python library against the official pod5 library.

Unlike ``benchmark.sh`` (which times the ``escpod`` CLI vs the ``pod5`` CLI with
hyperfine), this script exercises the two *Python libraries* in-process, so the
numbers reflect library work rather than interpreter startup + import overhead.

Each benchmark defines an escapepod callable and a pod5 callable that perform the
same logical operation. We warm up, run several timed repetitions, and report the
median wall time for each library plus the speedup (pod5 / escapepod). Where a
callable returns a checksum we also assert the two libraries agree, so a "win"
can't come from doing less work.

Usage:
    pixi run -e python-test python benchmarks/benchmark_python.py <pod5_file_or_dir>
    pixi run -e python-test python benchmarks/benchmark_python.py reads.pod5 \
        --runs 5 --warmup 1 --limit 5000 --json out.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

import escapepod
import pod5


# --------------------------------------------------------------------------- #
# Timing harness
# --------------------------------------------------------------------------- #
def _parity_ok(a: object, b: object) -> bool:
    """Compare two checksums, tolerating last-ULP float drift.

    int16 signal is bit-identical between the libraries, but pA calibration
    sums differ in the last float32 ULP (~1e-5 relative) because the two
    implementations evaluate the same formula in a different order.
    """
    if a is None or b is None:
        return False
    if isinstance(a, float) or isinstance(b, float):
        fa, fb = float(a), float(b)
        return abs(fa - fb) <= 1e-4 * max(1.0, abs(fa), abs(fb))
    return a == b


@dataclass
class Result:
    name: str
    esc_times: list[float] = field(default_factory=list)
    pod5_times: list[float] = field(default_factory=list)
    esc_check: object = None
    pod5_check: object = None
    note: str = ""

    @property
    def esc_median(self) -> Optional[float]:
        return statistics.median(self.esc_times) if self.esc_times else None

    @property
    def pod5_median(self) -> Optional[float]:
        return statistics.median(self.pod5_times) if self.pod5_times else None

    @property
    def speedup(self) -> Optional[float]:
        e, p = self.esc_median, self.pod5_median
        if e and p and e > 0:
            return p / e
        return None


def _time_one(fn: Callable[[], object], runs: int, warmup: int) -> tuple[list[float], object]:
    """Run ``fn`` warmup+runs times; return (timed_seconds, last_return_value)."""
    check = None
    for _ in range(warmup):
        check = fn()
    times: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        check = fn()
        times.append(time.perf_counter() - t0)
    return times, check


def run_benchmark(
    name: str,
    esc_fn: Optional[Callable[[], object]],
    pod5_fn: Optional[Callable[[], object]],
    runs: int,
    warmup: int,
    note: str = "",
) -> Result:
    res = Result(name=name, note=note)
    if esc_fn is not None:
        res.esc_times, res.esc_check = _time_one(esc_fn, runs, warmup)
    if pod5_fn is not None:
        res.pod5_times, res.pod5_check = _time_one(pod5_fn, runs, warmup)
    return res


# --------------------------------------------------------------------------- #
# Benchmark cases
# --------------------------------------------------------------------------- #
def build_cases(path: str, ids: list[str], sel_ids: list[str]):
    """Yield (name, esc_fn, pod5_fn, note) tuples.

    ``ids`` is the full read-id list; ``sel_ids`` is a random-access subset.
    Readers are opened fresh inside each callable so open/close cost is part of
    the measurement where that is the point of the benchmark, and amortized
    where it isn't (we reuse a persistent reader for the per-read loops).
    """

    # Persistent readers for the heavy iteration benchmarks (open cost measured
    # separately in the "open + metadata" case).
    esc_reader = escapepod.Reader(path)
    pod5_reader = pod5.Reader(path)

    def esc_open_meta():
        r = escapepod.Reader(path)
        n = r.read_count
        _ = r.run_infos
        return n

    def pod5_open_meta():
        r = pod5.Reader(path)
        n = r.num_reads
        r.close()
        return n

    def esc_read_ids():
        return len(esc_reader.read_ids())

    def pod5_read_ids():
        return len(pod5_reader.read_ids)

    def esc_iter_meta():
        total = 0
        for rd in esc_reader.reads():
            total += rd.read_number
        return total

    def pod5_iter_meta():
        total = 0
        for rec in pod5_reader.reads():
            total += rec.read_number
        return total

    def esc_signal_int16():
        acc = 0
        for rd in esc_reader.reads():
            acc += int(esc_reader.get_signal(rd).sum())
        return acc

    def pod5_signal_int16():
        acc = 0
        for rec in pod5_reader.reads():
            acc += int(rec.signal.sum())
        return acc

    def esc_signal_pa():
        acc = 0.0
        for rd in esc_reader.reads():
            acc += float(esc_reader.get_signal_pa(rd).sum())
        return round(acc, 1)

    def pod5_signal_pa():
        acc = 0.0
        for rec in pod5_reader.reads():
            acc += float(rec.signal_pa.sum())
        return round(acc, 1)

    def esc_to_pandas():
        # Row count is the parity check; column schemas legitimately differ.
        return esc_reader.to_pandas().shape[0]

    def pod5_to_pandas():
        # pod5 exposes read metadata as an Arrow table; materialize to pandas.
        return pod5_reader.read_table.read_all().to_pandas().shape[0]

    def esc_selection():
        return len(esc_reader.reads(selection=sel_ids))

    def pod5_selection():
        return len(list(pod5_reader.reads(selection=sel_ids)))

    def esc_signal_batched():
        # escapepod-specific batched signal fetch (no pod5 equivalent).
        reads = esc_reader.reads()
        acc = 0
        for _rid, sig in esc_reader.get_signals(reads):
            acc += int(sig.sum())
        return acc

    cases = [
        ("open + metadata", esc_open_meta, pod5_open_meta, "fresh open, read_count/num_reads"),
        ("read_ids", esc_read_ids, pod5_read_ids, ""),
        ("iterate read metadata", esc_iter_meta, pod5_iter_meta, ""),
        ("read all signal (int16)", esc_signal_int16, pod5_signal_int16, "VBZ decode, per-read"),
        ("read all signal (pA)", esc_signal_pa, pod5_signal_pa, "decode + calibrate"),
        ("metadata -> pandas", esc_to_pandas, pod5_to_pandas, "esc to_pandas vs pod5 arrow->pandas"),
        ("random-access selection", esc_selection, pod5_selection, f"{len(sel_ids)} reads"),
        ("read all signal (batched)", esc_signal_batched, None, "escapepod get_signals(), no pod5 equiv"),
    ]
    return cases, esc_reader, pod5_reader


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def fmt_time(t: Optional[float]) -> str:
    if t is None:
        return "     —   "
    if t < 1e-3:
        return f"{t * 1e6:7.1f} µs"
    if t < 1.0:
        return f"{t * 1e3:7.2f} ms"
    return f"{t:7.3f} s "


def report(results: list[Result], meta: dict) -> None:
    print()
    print("=" * 78)
    print("escapepod vs pod5 — Python library benchmark")
    print("=" * 78)
    print(f"  file          : {meta['file']}")
    print(f"  file size     : {meta['file_size_mb']:.1f} MB")
    print(f"  reads         : {meta['reads']:,} (signal loops use {meta['bench_reads']:,})")
    print(f"  escapepod     : {meta['escapepod_version']}")
    print(f"  pod5          : {meta['pod5_version']}")
    print(f"  runs/warmup   : {meta['runs']}/{meta['warmup']} (median reported)")
    print("-" * 78)
    print(f"  {'benchmark':<30}{'escapepod':>12}{'pod5':>12}{'speedup':>10}  parity")
    print("-" * 78)
    for r in results:
        speed = ""
        if r.speedup is not None:
            speed = f"{r.speedup:6.2f}x"
        parity = ""
        if r.esc_check is not None and r.pod5_check is not None:
            parity = "ok" if _parity_ok(r.esc_check, r.pod5_check) else "MISMATCH"
        elif r.pod5_median is None:
            parity = "esc-only"
        print(
            f"  {r.name:<30}{fmt_time(r.esc_median):>12}{fmt_time(r.pod5_median):>12}"
            f"{speed:>10}  {parity}"
        )
    print("-" * 78)
    print("  speedup = pod5 median / escapepod median  (>1 means escapepod faster)")
    mism = [r.name for r in results if r.esc_check is not None
            and r.pod5_check is not None and not _parity_ok(r.esc_check, r.pod5_check)]
    if mism:
        print(f"  WARNING: parity mismatch in: {', '.join(mism)}")
    print("=" * 78)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="POD5 file, or a directory (first *.pod5 is used)")
    ap.add_argument("--runs", type=int, default=5, help="timed repetitions (default 5)")
    ap.add_argument("--warmup", type=int, default=1, help="warmup repetitions (default 1)")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap reads used in signal loops (default: all)")
    ap.add_argument("--select", type=int, default=100,
                    help="reads in the random-access selection case (default 100)")
    ap.add_argument("--json", type=str, default=None, help="write results as JSON here")
    args = ap.parse_args()

    p = Path(args.path)
    if p.is_dir():
        pod5s = sorted(p.glob("*.pod5"))
        if not pod5s:
            print(f"error: no *.pod5 files in {p}", file=sys.stderr)
            return 1
        p = pod5s[0]
    if not p.exists():
        print(f"error: {p} not found", file=sys.stderr)
        return 1
    path = str(p)

    # Gather read ids once and pick a deterministic selection subset.
    r0 = escapepod.Reader(path)
    all_ids = r0.read_ids()
    total_reads = len(all_ids)
    if args.limit is not None and args.limit < total_reads:
        # A stable stride keeps the subset spread across the file (and across
        # signal batches) rather than clustered at the front.
        stride = max(1, total_reads // args.limit)
        bench_ids = all_ids[::stride][: args.limit]
    else:
        bench_ids = all_ids
    n_sel = min(args.select, len(bench_ids))
    sel_stride = max(1, len(bench_ids) // n_sel)
    sel_ids = bench_ids[::sel_stride][:n_sel]

    # If we're limiting, write a subset file so the per-read loops only touch the
    # capped set for BOTH libraries (fair + faster on huge inputs).
    work_path = path
    tmp_dir = None
    tmp_path = None
    if args.limit is not None and len(bench_ids) < total_reads:
        import tempfile
        # pod5.Writer refuses to overwrite, so hand it a path that does not yet
        # exist (mkdtemp gives us a private dir; the .pod5 inside is fresh).
        tmp_dir = tempfile.mkdtemp(prefix="escpod_pybench_")
        tmp_path = str(Path(tmp_dir) / "subset.pod5")
        want = set(bench_ids)
        with pod5.Reader(path) as src, pod5.Writer(tmp_path) as w:
            for rec in src.reads(selection=list(want), missing_ok=True):
                w.add_read(rec.to_read())
        work_path = tmp_path
        # recompute ids on the subset file
        rw = escapepod.Reader(work_path)
        bench_ids = rw.read_ids()
        n_sel = min(args.select, len(bench_ids))
        sel_stride = max(1, len(bench_ids) // n_sel)
        sel_ids = bench_ids[::sel_stride][:n_sel]

    cases, esc_reader, pod5_reader = build_cases(work_path, bench_ids, sel_ids)

    results: list[Result] = []
    for name, esc_fn, pod5_fn, note in cases:
        print(f"  running: {name} ...", file=sys.stderr, flush=True)
        results.append(run_benchmark(name, esc_fn, pod5_fn, args.runs, args.warmup, note))

    meta = {
        "file": work_path,
        "file_size_mb": Path(work_path).stat().st_size / 1e6,
        "reads": total_reads,
        "bench_reads": len(bench_ids),
        "escapepod_version": escapepod.__version__,
        "pod5_version": pod5.__version__,
        "runs": args.runs,
        "warmup": args.warmup,
    }
    report(results, meta)

    if args.json:
        out = {
            "meta": meta,
            "results": [
                {
                    "name": r.name,
                    "note": r.note,
                    "escapepod_median_s": r.esc_median,
                    "pod5_median_s": r.pod5_median,
                    "escapepod_times_s": r.esc_times,
                    "pod5_times_s": r.pod5_times,
                    "speedup": r.speedup,
                    "parity": (None if r.esc_check is None or r.pod5_check is None
                               else _parity_ok(r.esc_check, r.pod5_check)),
                }
                for r in results
            ],
        }
        Path(args.json).write_text(json.dumps(out, indent=2))
        print(f"  wrote {args.json}", file=sys.stderr)

    if tmp_dir:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
