#!/usr/bin/env python3
"""Regenerate signal_kmer_golden.json from leech's NumPy implementations.

The golden vectors pin `escapepod_signal::seq_encoding` to leech's reference
numerics (rnabioco/escapepod-rs#271). The encoding is exactly zeros and ones,
so each case's array travels as a string of '0'/'1' in row-major order --
exact, compact, and readable next to its `shape`.

The reference is deliberately the **NumPy** path in `leech.features`, not the
`leech_core` extension the same function dispatches to when it is importable.
The two disagree on one branch: a span whose start is negative. NumPy clamps it
to 0 and fills the surviving tail; the extension casts to `usize` before
clamping, so the start lands on `signal_len`, the span comes out empty and the
base disappears. `HAS_RUST` is forced off below so this file always pins the
readable definition -- which is the one escapepod-signal implements, and the
disagreement is the whole reason the primitive should have one home.

Run (needs numpy and a leech checkout; no leech_core build required):

    LEECH_SRC=~/devel/rnabioco/leech/src python gen_signal_kmer_golden.py

writes signal_kmer_golden.json next to itself.
"""

import json
import os
import sys
import types
from pathlib import Path

import numpy as np

LEECH_SRC = Path(os.environ.get("LEECH_SRC", Path.home() / "devel/rnabioco/leech/src"))
sys.path.insert(0, str(LEECH_SRC))

# `leech.features` imports pysam and escapepod at module scope, and annotates
# functions this file never calls with pysam types. Stubbing both keeps the
# generator's dependencies at numpy, so it runs without a maturin build of the
# bindings or a compiler.
class _Stub(types.ModuleType):
    def __getattr__(self, name):
        return type(name, (), {})


for absent in ("pysam", "escapepod"):
    sys.modules.setdefault(absent, _Stub(absent))

import leech.features as lf  # noqa: E402

# Pin the reference to the NumPy definition even where the extension is built.
lf.HAS_RUST = False

HERE = Path(__file__).resolve().parent
RNG = np.random.default_rng(271)
BASES = "ACGT"


def case(name, sequence, seq_to_sig_map, signal_len, kmer_context, note=None):
    """Run leech's reference encoder and record inputs alongside its output."""
    seq_ints = lf.sequence_to_int(sequence)
    sig_map = np.asarray(seq_to_sig_map, dtype=np.int64)
    enc = lf.encode_signal_kmer(seq_ints, sig_map, signal_len, kmer_context)
    assert enc.dtype == np.float32
    assert set(np.unique(enc)) <= {0.0, 1.0}, "the encoding is one-hot, not weights"
    out = {
        "name": name,
        # The sequence INCLUDING k-mer context, i.e. seq_len + before + after.
        "sequence": sequence,
        "seq_to_sig_map": [int(v) for v in sig_map],
        "signal_len": int(signal_len),
        "kmer_before": int(kmer_context[0]),
        "kmer_after": int(kmer_context[1]),
        "shape": [int(enc.shape[0]), int(enc.shape[1])],
        "encoding": "".join("1" if v else "0" for v in enc.ravel()),
    }
    if note:
        out["note"] = note
    return out


cases = []

# --- the shapes a caller actually passes ----------------------------------
cases.append(
    case(
        "three_bases_symmetric_context",
        "AACGG",  # 1 before + ACG + 1 after
        [0, 10, 20, 30],
        30,
        (1, 1),
        "leech's own fixture: the centre block tiles the window",
    )
)
cases.append(
    case(
        "default_36_channel_context",
        # 8 core bases with (4, 4) of context either side.
        "GTCAACGTTAGCCATG",
        [0, 7, 13, 26, 31, 39, 44, 52, 60],
        60,
        (4, 4),
        "the 36-channel input a signal_kmer model is trained on",
    )
)
cases.append(
    case("asymmetric_context", "CCCAGTA", [0, 5, 11, 16], 16, (3, 1))
)

# --- the branches that hide ------------------------------------------------
cases.append(
    case(
        "empty_span_in_the_middle",
        "AACGG",
        [0, 10, 10, 30],  # base 1 owns no samples at all
        30,
        (1, 1),
        "an unresolved base contributes nothing, its neighbours are unaffected",
    )
)
cases.append(
    case(
        "span_runs_past_the_end",
        "AACGG",
        [0, 10, 20, 45],
        30,
        (1, 1),
        "the head of the last base still lands",
    )
)
cases.append(
    case(
        "negative_start",
        "AACGG",
        [-8, 10, 20, 30],
        30,
        (1, 1),
        "NumPy clamps to 0 and keeps the tail; leech_core drops the base -- "
        "escapepod-signal follows NumPy",
    )
)
cases.append(
    case(
        "wholly_outside_the_window",
        "AACGG",
        [-30, -20, 40, 60],
        30,
        (1, 1),
        "first and last spans miss the window entirely; the middle one spans it",
    )
)
cases.append(
    case(
        "unknown_bases",
        "ANCGN",  # unknowns in both the context and the core
        [0, 10, 20, 30],
        30,
        (1, 1),
        "an ambiguity code leaves an all-zero column, not a fifth channel",
    )
)
cases.append(case("rna_uracil", "AAUGG", [0, 10, 20, 30], 30, (1, 1), "U shares T's channel"))
cases.append(case("no_context", "C", [0, 5], 5, (0, 0)))
cases.append(case("zero_bases", "", [0], 8, (0, 0), "map is just the closing boundary"))

# --- randomised sweep ------------------------------------------------------
for i in range(24):
    before, after = (int(x) for x in RNG.integers(0, 5, size=2))
    n_bases = int(RNG.integers(1, 15))
    # Dwells of 0 are deliberate: they are the empty-span branch.
    dwells = RNG.integers(0, 12, size=n_bases)
    sig_map = np.concatenate([[0], np.cumsum(dwells)]).astype(np.int64)
    # Shift the whole map so some cases start before the window, and size the
    # window so some spans run off the end.
    sig_map -= int(RNG.integers(0, 6))
    signal_len = int(sig_map[-1]) + int(RNG.integers(-4, 8))
    signal_len = max(signal_len, 0)
    seq = "".join(
        RNG.choice(list(BASES + "N"), p=[0.24, 0.24, 0.24, 0.24, 0.04])
        for _ in range(n_bases + before + after)
    )
    cases.append(case(f"random_{i:02d}", seq, sig_map, signal_len, (before, after)))

golden = {
    "sequence_to_int": [
        {"sequence": s, "ints": [int(v) for v in lf.sequence_to_int(s)]}
        for s in ["ACGT", "acgt", "ACGU", "acgu", "NX-.", "AcGuNt"]
    ],
    "cases": cases,
}

(HERE / "signal_kmer_golden.json").write_text(json.dumps(golden, indent=1) + "\n")
print(f"wrote signal_kmer_golden.json: {len(cases)} cases")
