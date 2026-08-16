#!/usr/bin/env python3
"""Regenerate the charging-classifier parity fixtures and golden vectors.

Builds, from leech's tRNA test fixtures (rnabioco/leech, MIT):

- ``trna_mappings_padded.bam`` — the 20 fixture alignments plus two renamed
  copies each (fresh seeded UUIDs), so the move-table orientation vote has
  >= 50 informative reads and runs through the REAL code path on both sides
  (the copies anchor and vote but have no POD5 signal, exercising that skip
  path too);
- ``trna_reads.pod5`` / ``trna_reference.fa`` — copied verbatim;
- ``bundle/`` — a complete, hash-pinned model bundle: a synthetic (full 4^5)
  k-mer level table, a tiny binary HistGradientBoosting model trained on the
  reference features and exported via ``scripts/export_gbm_model.py``, and
  ``metadata.json`` carrying the full recipe;
- ``charging_golden.json`` — per-read feature vectors (f32 bit patterns),
  probabilities (f64 bit patterns) and ``cl`` bytes computed by the
  REFERENCE implementation (``escapepod_models.charging``, imported
  standalone from CHARGING_PY), which built the training corpus.

Run on a machine with the leech venv (numpy, pysam, sklearn, escapepod)::

    LEECH_SRC=~/devel/rnabioco/leech/src \
    CHARGING_PY=~/devel/rnabioco/escapepod-models/.claude/worktrees/charging-cca/src/escapepod_models/charging.py \
    LEECH_FIXTURES=~/devel/rnabioco/leech/tests/fixtures \
    python gen_charging_golden.py
"""

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import uuid as uuidlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LEECH_SRC = Path(os.environ.get("LEECH_SRC", Path.home() / "devel/rnabioco/leech/src"))
CHARGING_PY = Path(
    os.environ.get(
        "CHARGING_PY",
        Path.home()
        / "devel/rnabioco/escapepod-models/.claude/worktrees/charging-cca/src/escapepod_models/charging.py",
    )
)
LEECH_FIXTURES = Path(
    os.environ.get("LEECH_FIXTURES", Path.home() / "devel/rnabioco/leech/tests/fixtures")
)
EXPORT_SCRIPT = HERE.parents[3] / "scripts" / "export_gbm_model.py"

sys.path.insert(0, str(LEECH_SRC))

# Import charging.py standalone (its top-level imports are stdlib + numpy;
# pysam/leech are function-local), avoiding the escapepod_models package.
spec = importlib.util.spec_from_file_location("charging", CHARGING_PY)
charging = importlib.util.module_from_spec(spec)
sys.modules["charging"] = charging  # dataclasses resolve via sys.modules
# Register before exec: `charging.JunctionRecord` is a `@dataclass(slots=True)`,
# and slots rebuilds the class, which makes dataclasses look `cls.__module__`
# up in sys.modules. A spec-loaded module that is not registered there fails
# with an opaque AttributeError on None.
sys.modules[spec.name] = charging
spec.loader.exec_module(charging)

import pysam  # noqa: E402

rng = np.random.default_rng(20260810)


def f32_bits(a):
    return [int(x) for x in np.asarray(a, dtype=np.float32).view(np.uint32).ravel()]


def f64_bits_scalar(x):
    return int(np.float64(x).view(np.uint64))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- fixture inputs ---------------------------------------------------------
shutil.copy2(LEECH_FIXTURES / "trna_reads.pod5", HERE / "trna_reads.pod5")
shutil.copy2(LEECH_FIXTURES / "trna_reference.fa", HERE / "trna_reference.fa")

padded = HERE / "trna_mappings_padded.bam"
with pysam.AlignmentFile(LEECH_FIXTURES / "trna_mappings.bam", "rb", check_sq=False) as src:
    with pysam.AlignmentFile(padded, "wb", template=src) as dst:
        for aln in src:
            dst.write(aln)
            for _ in range(2):
                dup = pysam.AlignedSegment(dst.header)
                dup = aln.__copy__()
                dup.query_name = str(uuidlib.UUID(bytes=rng.bytes(16), version=4))
                dst.write(dup)
pysam.index(str(padded))

# --- synthetic complete 4^5 k-mer table ------------------------------------
BUNDLE = HERE / "bundle"
BUNDLE.mkdir(exist_ok=True)
bases = "ACGT"
kmers = ["".join(k) for k in __import__("itertools").product(bases, repeat=5)]
levels = rng.normal(100.0, 15.0, size=len(kmers))
table_path = BUNDLE / "kmer_levels.tsv"
with open(table_path, "w") as fh:
    fh.write("kmer\tlevel_mean\n")
    for k, v in zip(kmers, levels):
        fh.write(f"{k}\t{v:.6f}\n")

# --- reference feature computation (the charging.py path) -------------------
geometry = charging.junction_positions(str(HERE / "trna_reference.fa"))
remap = {}
records, qc = charging.collect_junctions(
    str(padded), geometry, min_mapq=1, remap_sink=remap
)
print("orientation:", qc["orientation"], qc["orientation_votes"])

# One record per read, best alignment wins (mirrors both charging.py's dedup
# and escpod's).
best = {}
for r in records:
    prev = best.get(r.read_id)
    if prev is None or r.mapq > prev.mapq:
        best[r.read_id] = r

refiner = charging._Refiner(str(table_path))  # scale_iters=-1: rescale-only

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
        reads.append({"read_id": rid, "reference": r.reference, "F": F})

reads.sort(key=lambda d: d["read_id"])
print(f"{len(reads)} reads with signal + features")
assert len(reads) >= 15, "fixture unexpectedly small"

# --- tiny binary GBM on the reference features ------------------------------
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402

X = np.stack([d["F"] for d in reads])
y = np.arange(len(reads)) % 2  # arbitrary but deterministic labels
clf = HistGradientBoostingClassifier(
    max_iter=20, learning_rate=0.3, random_state=0, early_stopping=False
)
clf.fit(X, y)
probs = clf.predict_proba(X)[:, 1]

import joblib  # noqa: E402

joblib.dump(clf, BUNDLE / "model.joblib")
gbm_json = BUNDLE / "model.gbm.json"
subprocess.run(
    [sys.executable, str(EXPORT_SCRIPT), str(BUNDLE / "model.joblib"), str(gbm_json)],
    check=True,
)
(BUNDLE / "model.joblib").unlink()

# --- bundle metadata --------------------------------------------------------
meta = {
    "format": "escapepod-charging-classifier/1",
    "model": {
        "id": "charging_fixture",
        "version": "0.0.1",
        "chemistry": "rna004",
        "task": "tRNA aminoacylation state (charged vs uncharged) — TEST FIXTURE",
    },
    "classes": list(charging.LABELS),
    "gbm": {"file": gbm_json.name, "sha256": sha256(gbm_json)},
    "anchor": {
        "motif": charging.MOTIF,
        "motif_offset": 3,
        "common_arm": charging.COMMON_ARM,
    },
    "features": {
        "order": charging.feature_names(),
        "offsets": list(charging.FEAT_OFFSETS),
        "stats": list(charging.FEAT_STATS),
    },
    "kmer_table": {
        "file": table_path.name,
        "sha256": sha256(table_path),
        "center_idx": None,
    },
    "operating_point": {
        "probability": 0.6,
        "cl": 153,
        "source": "arbitrary fixture operating point (labels are synthetic)",
    },
}
(BUNDLE / "metadata.json").write_text(json.dumps(meta, indent=1))

# --- golden -----------------------------------------------------------------
golden = {
    "orientation": qc["orientation"],
    "numpy": np.__version__,
    "sklearn": __import__("sklearn").__version__,
    "reads": [
        {
            "read_id": d["read_id"],
            "reference": d["reference"],
            "features_bits": f32_bits(d["F"]),
            "p_bits": f64_bits_scalar(p),
            "cl": int(np.round(p * 255)),
        }
        for d, p in zip(reads, probs)
    ],
}
(HERE / "charging_golden.json").write_text(json.dumps(golden, indent=1))
print(f"wrote {HERE / 'charging_golden.json'} ({len(reads)} reads)")

# leech's load_kmer_table writes a .pkl cache beside the table; it is not
# part of the bundle.
(BUNDLE / f"{table_path.name}.pkl").unlink(missing_ok=True)
