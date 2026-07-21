#!/usr/bin/env python3
"""Generate the committed GBM parity fixture for escapepod-demux tests.

Fits a tiny multiclass HistGradientBoostingClassifier on synthetic data, exports
it with export_gbm_model.export_model, and dumps sklearn's predict_proba on a
handful of rows. The Rust test `crates/escapepod-demux/tests/gbm_parity.rs`
loads these files and asserts GbmPredictor reproduces sklearn within 1e-5 — so
escpod's native tree walk is verified against the reference implementation
without needing sklearn at `cargo test` time.

Run from the repo root in an env with scikit-learn:
    pixi run -e warpdemux-bench python scripts/_gen_gbm_fixture.py
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_gbm_model import export_model  # noqa: E402

import json  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "crates/escapepod-demux/tests/fixtures/gbm"
N_FEATURES = 6
N_CLASSES = 4
N_EVAL = 12

rng = np.random.RandomState(0)
X, y = make_classification(
    n_samples=400, n_features=N_FEATURES, n_informative=5, n_redundant=0,
    n_classes=N_CLASSES, n_clusters_per_class=1, random_state=0,
)
# String class labels exercise the trailing-digit barcode-id derivation.
labels = np.array([f"barcode{int(c) + 1:02d}" for c in y])

clf = HistGradientBoostingClassifier(
    max_iter=25, learning_rate=0.1, max_leaf_nodes=15,
    l2_regularization=1.0, random_state=0,
)
clf.fit(X, labels)

model = export_model(clf)

# Evaluation rows: a deterministic slice of the training matrix.
Xe = X[:N_EVAL]
proba = clf.predict_proba(Xe)  # columns follow clf.classes_ order == class index

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "model.json").write_text(json.dumps(model))

with (OUT / "inputs.csv").open("w") as f:
    f.write(",".join(f"fp_{i}" for i in range(N_FEATURES)) + "\n")
    for row in Xe:
        f.write(",".join(f"{v:.9g}" for v in row) + "\n")

with (OUT / "expected_probs.csv").open("w") as f:
    f.write(",".join(f"p{i}" for i in range(N_CLASSES)) + "\n")
    for row in proba:
        f.write(",".join(f"{v:.12g}" for v in row) + "\n")

print(f"wrote fixture to {OUT}: {N_CLASSES} classes, {N_FEATURES} features, "
      f"{len(model['trees'])} iterations, {N_EVAL} eval rows", file=sys.stderr)
