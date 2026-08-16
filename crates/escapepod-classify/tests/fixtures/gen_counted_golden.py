#!/usr/bin/env python3
"""Golden vectors for the COUNTING span anchor.

`charging_golden.json` pins the aligner-derived anchor -- the one the shipped
`charging_cnn_rna004@v0.1.0` bundle was trained with, and the only one
`escapepod-classify` implemented. Every charging config written since
2026-08-13 (`9a37d46`, "adopt counting extraction as the default") instead sets
`count_arm_bases` to 8 or 24, which reaches the arm offsets by counting along
the *query* rather than asking the aligner.

That is a different feature space, so a model trained with it and scored
without it gets a confident wrong answer. This file pins the counting side.

It deliberately does NOT regenerate the existing fixtures: the bundle carries a
GBM trained on the aligner features and re-fitting it would churn every golden
probability for no reason. Only the feature grid and mask boundary are pinned
here, because the fixture GBM's *probabilities* are meaningless on features it
was not trained on.

Reuses the committed `trna_mappings_padded.bam`, `trna_reads.pod5` and
`trna_reference.fa`. Run from the escapepod-models worktree::

    CHARGING_PY=src/escapepod_models/charging.py \
    pixi run -e boundary python \
      ext/escapepod-rs/crates/escapepod-classify/tests/fixtures/gen_counted_golden.py
"""

import importlib.util
import json
import os
import sys
import struct
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

# The reference implementation, imported standalone so this does not depend on
# the training package being installed.
CHARGING_PY = Path(os.environ.get("CHARGING_PY", "src/escapepod_models/charging.py")).resolve()
spec = importlib.util.spec_from_file_location("charging_ref", CHARGING_PY)
charging = importlib.util.module_from_spec(spec)
# Register before exec: `charging.JunctionRecord` is a `@dataclass(slots=True)`,
# and slots rebuilds the class, which makes dataclasses look `cls.__module__`
# up in sys.modules. A spec-loaded module that is not registered there fails
# with an opaque AttributeError on None.
sys.modules[spec.name] = charging
spec.loader.exec_module(charging)

# The counting depth the current configs use. 8 (charging_cnn_rna004, ldx16)
# and 24 (ldx16r/w/x) are both in service; 24 is the wider of the two and
# exercises the cap on more offsets.
COUNT_ARM_BASES = 24

geometry = charging.junction_positions(str(HERE / "trna_reference.fa"))
remap = {}
records, qc = charging.collect_junctions(
    str(HERE / "trna_mappings_padded.bam"),
    geometry,
    min_mapq=1,
    remap_sink=remap,
    count_arm_bases=COUNT_ARM_BASES,
)
print("orientation:", qc["orientation"], qc["orientation_votes"])

best = {}
for r in records:
    prev = best.get(r.read_id)
    if prev is None or r.mapq > prev.mapq:
        best[r.read_id] = r

refiner = charging._Refiner(str(HERE / "bundle" / "kmer_levels.tsv"))

import escapepod  # noqa: E402

reads = []
with escapepod.Reader(str(HERE / "trna_reads.pod5")) as reader:
    meta = reader.reads()
    for rid, sig in reader.get_signals_pa(meta):
        r = best.get(rid)
        if r is None:
            continue
        sig = np.asarray(sig, dtype=np.float32)
        r = charging._refine_record(r, remap[rid], sig, refiner)
        seq, _s2s, nb, _rev, _qj, _qa, qf, _qcs = remap[rid]
        exp = refiner.expected_at(seq, qf, nb)
        F = charging._base_features(sig, r, exp)
        reads.append(
            {
                "read_id": rid,
                "reference": r.reference,
                # f32 bit patterns: NaN is part of the contract (an unresolved
                # base must stay unresolved) and does not survive a decimal
                # round-trip.
                "f_bits": [struct.unpack("<I", struct.pack("<f", v))[0] for v in F],
                "common_start_sig": int(r.common_start_sig),
                "junction_sig": int(r.junction_sig),
                "cs_source": r.cs_source,
                "feat_spans": [[int(a), int(b)] for a, b in r.feat_spans],
            }
        )

reads.sort(key=lambda d: d["read_id"])
assert len(reads) >= 15, f"fixture unexpectedly small: {len(reads)}"

n_nan = sum(1 for d in reads for b in d["f_bits"] if np.isnan(struct.unpack("<f", struct.pack("<I", b))[0]))
print(f"{len(reads)} reads; {n_nan} NaN features; cs_source counts:")
for k in sorted({d["cs_source"] for d in reads}):
    print(f"  {k}: {sum(1 for d in reads if d['cs_source'] == k)}")

out = {
    "count_arm_bases": COUNT_ARM_BASES,
    "orientation": qc["orientation"],
    "offsets": list(charging.FEAT_OFFSETS),
    "stats": list(charging.FEAT_STATS),
    "reads": reads,
}
(HERE / "charging_golden_counted.json").write_text(json.dumps(out, indent=1))
print("wrote charging_golden_counted.json")
