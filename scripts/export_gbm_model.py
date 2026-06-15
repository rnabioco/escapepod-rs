#!/usr/bin/env python3
"""Export a fitted sklearn HistGradientBoostingClassifier to escapepod GbmModel JSON.

escpod's runtime has no ONNX tree-ensemble path (tract lacks `ai.onnx.ml`
`TreeEnsembleClassifier`), so a gradient-boosted barcode head ships as a small
JSON dump of the fitted trees that escpod walks natively (see
`crates/escapepod-demux/src/gbm.rs`). This converter produces that JSON.

The model whose head this reproduces (`WDX4_tRNA_rna004_v1_0`) is itself
gradient-boosted (`fpt_boost`); the bake-off arm `_wdx_gbm.py` (escapepod-models)
fits the HistGradientBoostingClassifier this script consumes.

Only **multiclass** models are supported (one tree per class per boosting
iteration, combined with softmax) — the 5-barcode bake-off case. Binary models
(1 tree/iter + sigmoid) and categorical splits are rejected explicitly rather
than silently mis-exported.

Usage:
    python export_gbm_model.py model.joblib output.json
    python export_gbm_model.py model.joblib output.json --thresholds 0.1,0.1,0.1

`model.joblib` is a `joblib.dump` of a *fitted* HistGradientBoostingClassifier.
Class index -> barcode id is derived from `clf.classes_` (trailing digits of each
class label, e.g. "barcode03" -> 3, or an integer label used as-is); override
with --label-map mapping.json ({"<class label>": <barcode id>, ...}).
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np


def parse_barcode_id(label) -> int:
    """Map a class label (e.g. "barcode03", "BC3", 3) to an integer barcode id."""
    if isinstance(label, (int, np.integer)):
        return int(label)
    m = re.search(r"(\d+)\s*$", str(label))
    if not m:
        raise SystemExit(f"Cannot derive a barcode id from class label {label!r}; "
                         f"pass --label-map to specify it explicitly.")
    return int(m.group(1))


def export_tree(nodes) -> list:
    """Convert one sklearn TreePredictor.nodes structured array to a node list."""
    names = nodes.dtype.names
    # Threshold field is `num_threshold` in modern sklearn, `threshold` in older.
    thr_field = "num_threshold" if "num_threshold" in names else "threshold"
    has_categorical = "is_categorical" in names
    out = []
    for nd in nodes:
        is_leaf = bool(nd["is_leaf"])
        if not is_leaf and has_categorical and bool(nd["is_categorical"]):
            raise SystemExit("Categorical splits are unsupported (fingerprints are "
                             "numeric); refusing to export.")
        out.append({
            "feature": int(nd["feature_idx"]) if not is_leaf else -1,
            "threshold": float(nd[thr_field]) if not is_leaf else 0.0,
            "left": int(nd["left"]),
            "right": int(nd["right"]),
            "missing_left": bool(nd["missing_go_to_left"]),
            "value": float(nd["value"]),
            "leaf": is_leaf,
        })
    return out


def export_model(clf, label_map_override=None, thresholds=None) -> dict:
    if not hasattr(clf, "_predictors"):
        raise SystemExit(f"{type(clf).__name__} is not a fitted "
                         "HistGradientBoostingClassifier (no _predictors).")

    classes = list(clf.classes_)
    n_classes = len(classes)
    if n_classes < 2:
        raise SystemExit(f"Need >= 2 classes, got {n_classes}.")

    predictors = clf._predictors  # [n_iterations][n_trees_per_iteration]
    n_trees_per_iter = len(predictors[0])
    if n_trees_per_iter != n_classes:
        raise SystemExit(
            f"Model has {n_trees_per_iter} trees/iteration but {n_classes} classes; "
            "binary/sigmoid models are unsupported (multiclass softmax only).")

    # baseline raw score per class (raveled to length n_classes, class order).
    baseline = np.asarray(clf._baseline_prediction, dtype=float).ravel().tolist()
    if len(baseline) != n_classes:
        raise SystemExit(
            f"baseline has {len(baseline)} entries, expected {n_classes}.")

    n_features = int(getattr(clf, "n_features_in_", getattr(clf, "_n_features", 0)))
    if n_features <= 0:
        raise SystemExit("Could not determine n_features_in_ from the model.")

    trees = [[export_tree(predictors[it][k].nodes) for k in range(n_classes)]
             for it in range(len(predictors))]

    if label_map_override is not None:
        label_mapper = {str(i): int(label_map_override[str(c)])
                        for i, c in enumerate(classes)}
    else:
        label_mapper = {str(i): parse_barcode_id(c) for i, c in enumerate(classes)}

    model = {
        "model_type": "gbm",
        "n_classes": n_classes,
        "n_features": n_features,
        "baseline": baseline,
        "trees": trees,
        "label_mapper": label_mapper,
        "classes": [label_mapper[str(i)] for i in range(n_classes)],
    }
    if thresholds is not None:
        if len(thresholds) != n_classes:
            raise SystemExit(
                f"--thresholds has {len(thresholds)} values, expected {n_classes}.")
        model["thresholds"] = list(thresholds)
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", type=Path, help="fitted HGB classifier (.joblib)")
    ap.add_argument("output", type=Path, help="output GbmModel JSON path")
    ap.add_argument("--thresholds", type=str, default=None,
                    help="comma-separated per-class confidence-margin thresholds")
    ap.add_argument("--label-map", type=Path, default=None,
                    help="JSON {class_label: barcode_id} overriding the default "
                         "trailing-digit derivation")
    args = ap.parse_args()

    try:
        import joblib
    except ImportError:
        sys.exit("Error: joblib required (pip install joblib).")

    clf = joblib.load(args.model)
    label_map_override = (json.loads(args.label_map.read_text())
                          if args.label_map else None)
    thresholds = ([float(x) for x in args.thresholds.split(",")]
                  if args.thresholds else None)

    model = export_model(clf, label_map_override, thresholds)
    args.output.write_text(json.dumps(model))
    print(f"wrote {args.output}: {model['n_classes']} classes, "
          f"{model['n_features']} features, {len(model['trees'])} iterations",
          file=sys.stderr)


if __name__ == "__main__":
    main()
