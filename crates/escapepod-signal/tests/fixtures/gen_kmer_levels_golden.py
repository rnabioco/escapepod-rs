#!/usr/bin/env python3
"""Regenerate kmer_levels_golden.json from leech's NumPy implementations.

The golden vectors pin escapepod-signal's ported k-mer level primitives
(rnabioco/escapepod-rs#204) to leech's reference numerics bit-for-bit.
Floats travel as IEEE-754 bit patterns (u32/u64 ints) so no precision is
lost in the JSON round trip.

Run (needs numpy and a leech checkout, no leech_core build required):

    LEECH_SRC=~/devel/rnabioco/leech/src python gen_kmer_levels_golden.py

writes kmer_levels_golden.json and kmer_table_synthetic.tsv next to itself.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

LEECH_SRC = Path(os.environ.get("LEECH_SRC", Path.home() / "devel/rnabioco/leech/src"))
sys.path.insert(0, str(LEECH_SRC))

from leech.signal_refine import (  # noqa: E402
    extract_levels,
    load_kmer_table,
    rough_rescale_quantile,
)

HERE = Path(__file__).resolve().parent


def f64_bits(a):
    return [int(x) for x in np.asarray(a, dtype=np.float64).view(np.uint64).ravel()]


def f32_bits(a):
    return [int(x) for x in np.asarray(a, dtype=np.float32).view(np.uint32).ravel()]


# --- synthetic k-mer table exercising every lenient-parse branch -----------
TABLE = (
    "# comment line\n"
    "kmer\tlevel_mean\n"  # header: level fails float() -> skipped
    "AAAAA\t0.95838\n"
    "AAAAC\t-1.25\n"
    "AAACG\t0.75\n"
    "AACGT\t1e-3\n"
    "ACGTA\t2.5\n"
    "CGTAC\t-0.125\n"
    "acgtt\t0.3333333333333333\n"  # lowercase: stored uppercased
    "AAAAA\t1.5\n"  # duplicate: last wins
    "GGGGG 0.125\n"  # whitespace fallback split
    "AAAAT\t9.0\n"
    "onlyonefield\n"  # skipped
    "\n"
    "CCCCC\t-0.5\n"
)

golden = {"table_file": "kmer_table_synthetic.tsv"}
(HERE / golden["table_file"]).write_text(TABLE)

# Load via a temp copy so leech's .pkl cache lands outside the fixtures dir.
with tempfile.TemporaryDirectory() as td:
    tmp_table = Path(td) / "table.tsv"
    tmp_table.write_text(TABLE)
    kmer_to_level, kmer_len = load_kmer_table(tmp_table)

golden["load_kmer_table"] = {
    "k": kmer_len,
    "levels": {k: f64_bits([v])[0] for k, v in sorted(kmer_to_level.items())},
}

# --- np.arange(0.05, 1, 0.05), the quantile grid ---------------------------
quants = np.arange(0.05, 1, 0.05)
assert quants.size == 19
golden["quants_arange_bits"] = f64_bits(quants)

# --- quantile probes: pin NumPy's dtype-dependent interpolation ------------
rng = np.random.default_rng(42)
probe = rng.normal(size=7)
probes = []
for name, kind, arr in [
    ("f64", "f64", probe.astype(np.float64)),
    ("f32", "f32", probe.astype(np.float32)),
    # Sorted-adjacent pairs spanning binades: the f32 endpoint subtraction
    # rounds here, distinguishing NumPy's dtype-faithful bracketing values
    # from a promote-to-f64-first implementation.
    ("f32_cross_binade", "f32",
     np.array([0.001, 0.9, -0.0625, 7.25, -3.1, 0.017, 100.0], dtype=np.float32)),
]:
    out = np.quantile(arr, quants)
    probes.append(
        {
            "name": name,
            "kind": kind,
            "data_bits": f64_bits(arr) if kind == "f64" else f32_bits(arr),
            "out_dtype": str(out.dtype),
            "out_bits": f64_bits(out) if out.dtype == np.float64 else f32_bits(out),
        }
    )
golden["quantile_probes"] = probes

# --- extract_levels (Python reference returns float32) ---------------------
cases = []
for seq, center in [
    ("AAAAACGTACGTT", None),  # chained hits, default center
    ("AAAAACGTACGTT", 0),
    ("AAAAACGTACGTT", 4),
    ("aaaaacguacguu", None),  # lowercase + U->T
    ("AANAACGT", None),  # N windows miss -> zeros
    ("ACG", None),  # shorter than k -> zeros
    ("GGGGG", None),  # exactly k
    ("UUUUUAAAAT", None),  # leading U's (ACGTT hit via U->T)
]:
    out = extract_levels(seq, kmer_to_level, kmer_len, center_idx=center)
    assert out.dtype == np.float32, out.dtype
    cases.append({"seq": seq, "center": center, "out_f32_bits": f32_bits(out)})
golden["extract_levels"] = cases

# --- rough_rescale_quantile ------------------------------------------------
def make_case(name, n_bases, *, dwell_lo=8, dwell_hi=14, tail=7, levels_mode="corr",
              degenerate=False):
    dwells = rng.integers(dwell_lo, dwell_hi, size=n_bases)
    m = np.zeros(n_bases + 1, dtype=np.int64)
    m[1:] = np.cumsum(dwells)
    sig = (rng.normal(size=int(m[-1]) + tail) * 1.2 + 0.3).astype(np.float32)
    centers = (m[:-1] + m[1:]) // 2
    if levels_mode == "corr":
        lv = (0.9 * sig[centers].astype(np.float64) + rng.normal(scale=0.3, size=n_bases)).astype(
            np.float32
        )
    elif levels_mode == "const":
        lv = np.full(n_bases, 0.5, dtype=np.float32)
    else:
        raise ValueError(levels_mode)
    out = rough_rescale_quantile(sig, lv, m)
    assert out.dtype == np.float32, out.dtype
    return {
        "name": name,
        "signal_bits": f32_bits(sig),
        "levels_bits": f32_bits(lv),
        "map": [int(x) for x in m],
        "clip_bases": 10,
        "degenerate": degenerate,
        "out_bits": f32_bits(out),
    }


golden["rough_rescale_quantile"] = [
    make_case("typical_60_bases", 60),
    make_case("no_clip_20_bases", 20),
    make_case("tiny_2_bases", 2),
    make_case("constant_levels", 30, levels_mode="const"),
    # 21 bases clip to a single post-clip point: the fit is singular and
    # NumPy's lstsq returns a minimum-norm fit; the Rust port documents that
    # it returns the signal unchanged instead. Recorded for reference, the
    # parity test only checks the Rust degenerate behaviour.
    make_case("degenerate_21_bases", 21, degenerate=True),
]

out_path = HERE / "kmer_levels_golden.json"
out_path.write_text(json.dumps(golden, indent=1))
print(f"wrote {out_path}")
print("level_qs probe dtypes:", [p["out_dtype"] for p in probes])
