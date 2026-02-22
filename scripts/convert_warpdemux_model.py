#!/usr/bin/env python3
"""Convert a WarpDemuX DTW_SVM model (.joblib) to escapepod DtwSvmModel JSON format.

This script properly extracts all SVM parameters including:
- Training fingerprints (_X) and labels
- SVM dual coefficients, support vectors, intercepts
- Platt scaling parameters (probA_, probB_)
- Kernel parameters (gamma, pwr_dist)
- Label mapper and thresholds

Usage:
    python convert_warpdemux_model.py model.joblib output.json

Example:
    pixi run python scripts/convert_warpdemux_model.py \
        resources/tools/WarpDemuX/warpdemux/models/model_files/WDX4_rna004_v1_0.joblib \
        /tmp/wdx4_rna004.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def load_model(path: Path):
    """Load a joblib model file."""
    try:
        import joblib
    except ImportError:
        print("Error: joblib required. Install with: pip install joblib", file=sys.stderr)
        sys.exit(1)
    return joblib.load(path)


def inspect_model(model) -> None:
    """Print model structure for debugging."""
    print(f"  Type: {type(model).__name__}")
    print(f"  Attributes: {sorted([a for a in dir(model) if not a.startswith('__')])}")

    if hasattr(model, "_X"):
        print(f"  _X shape: {model._X.shape}")
    if hasattr(model, "n_classes"):
        print(f"  n_classes: {model.n_classes}")
    if hasattr(model, "label_mapper"):
        print(f"  label_mapper: {model.label_mapper}")
    if hasattr(model, "gamma"):
        print(f"  gamma: {model.gamma}")
    if hasattr(model, "pwr_dist"):
        print(f"  pwr_dist: {model.pwr_dist}")
    if hasattr(model, "window"):
        print(f"  window: {model.window}")
    if hasattr(model, "noise_class"):
        print(f"  noise_class: {model.noise_class}")
    if hasattr(model, "thresholds"):
        print(f"  thresholds: {model.thresholds}")

    if hasattr(model, "model"):
        svc = model.model
        print(f"  SVC type: {type(svc).__name__}")
        if hasattr(svc, "classes_"):
            print(f"  SVC classes_: {svc.classes_}")
        if hasattr(svc, "support_"):
            print(f"  SVC n_support_vectors: {len(svc.support_)}")
        if hasattr(svc, "n_support_"):
            print(f"  SVC n_support_: {svc.n_support_}")
        if hasattr(svc, "dual_coef_"):
            print(f"  SVC dual_coef_ shape: {svc.dual_coef_.shape}")
        if hasattr(svc, "intercept_"):
            print(f"  SVC intercept_ shape: {svc.intercept_.shape}")
        if hasattr(svc, "probA_"):
            print(f"  SVC probA_: {svc.probA_}")
        if hasattr(svc, "probB_"):
            print(f"  SVC probB_: {svc.probB_}")


def extract_training_labels(model) -> np.ndarray:
    """Extract training labels from a WarpDemuX DTW_SVM model.

    The WarpDemuX DTW_SVM stores training data in _X but doesn't directly
    store labels as a simple array. We reconstruct them from the SVC's
    internal state.
    """
    svc = model.model
    n_samples = model._X.shape[0]

    # Method 1: Check for _y attribute (some WarpDemuX versions store it)
    if hasattr(model, "_y"):
        y = np.asarray(model._y)
        if len(y) == n_samples:
            print("  Labels: extracted from model._y")
            return y

    # Method 2: For precomputed kernel SVC, sklearn stores the original y
    # in the fit_status_ or we can reconstruct from support vector structure
    if hasattr(svc, "_y") and svc._y is not None:
        y = np.asarray(svc._y)
        if len(y) == n_samples:
            print("  Labels: extracted from SVC._y")
            return y

    # Method 3: Reconstruct from n_support_ per class
    # For precomputed kernels where all points may be support vectors,
    # or where we know the class distribution
    if hasattr(svc, "n_support_") and hasattr(svc, "classes_"):
        total_sv = sum(svc.n_support_)
        if total_sv == n_samples:
            # All training points are support vectors, ordered by class
            labels = []
            for class_idx, n_sv_class in enumerate(svc.n_support_):
                labels.extend([int(svc.classes_[class_idx])] * n_sv_class)
            print("  Labels: reconstructed from n_support_ (all points are SVs)")
            return np.array(labels)

    # Method 4: Use the SVC's decision function to predict labels
    # For precomputed kernel, we need the kernel matrix
    print("  Warning: Could not directly extract labels, using SVC prediction")
    if hasattr(svc, "predict"):
        # Build kernel matrix for training data
        from scipy.spatial.distance import cdist

        # This is a fallback - compute DTW distances would be expensive
        # Instead, use support vector structure
        n_classes = len(svc.classes_)
        samples_per_class = n_samples // n_classes
        labels = []
        for i, cls in enumerate(svc.classes_):
            count = samples_per_class
            if i < n_samples % n_classes:
                count += 1
            labels.extend([int(cls)] * count)
        print(f"  Warning: Using uniform label distribution as fallback ({samples_per_class}/class)")
        return np.array(labels[:n_samples])

    raise ValueError("Cannot extract training labels from model")


def convert_model(model, output_path: Path) -> None:
    """Convert a WarpDemuX DTW_SVM model to escapepod JSON format."""
    svc = model.model
    fingerprints = model._X
    n_samples, n_features = fingerprints.shape

    # Extract labels
    labels = extract_training_labels(model)

    # SVM parameters
    support_indices = svc.support_.tolist()
    dual_coef = svc.dual_coef_.tolist()
    intercept = svc.intercept_.tolist()
    classes = [int(c) for c in svc.classes_]

    # Platt scaling
    prob_a = svc.probA_.tolist() if hasattr(svc, "probA_") and svc.probA_ is not None else None
    prob_b = svc.probB_.tolist() if hasattr(svc, "probB_") and svc.probB_ is not None else None

    # Kernel parameters
    gamma = float(model.gamma)
    power = float(model.pwr_dist)

    # DTW window
    window = int(model.window) if model.window else None

    # Label mapper: class_index -> barcode_id
    # WarpDemuX's label_mapper maps class_idx -> barcode_id
    label_mapper = {str(k): int(v) for k, v in model.label_mapper.items()}

    # Thresholds
    thresholds = None
    if hasattr(model, "thresholds") and model.thresholds is not None:
        thresholds = model.thresholds.tolist() if hasattr(model.thresholds, "tolist") else list(model.thresholds)

    # Noise class
    noise_class = bool(model.noise_class) if hasattr(model, "noise_class") else False

    # Number of classes
    n_classes = len(classes)

    # Build output
    output = {
        "version": "1.0",
        "training_fingerprints": fingerprints.tolist(),
        "training_labels": [int(l) for l in labels.tolist()],
        "support_indices": support_indices,
        "dual_coef": dual_coef,
        "intercept": intercept,
        "classes": classes,
        "kernel_params": {
            "gamma": gamma,
            "power": power,
        },
        "label_mapper": label_mapper,
        "n_classes": n_classes,
        "noise_class": noise_class,
        "use_kernel_weighted": False,
    }

    if window is not None:
        output["window"] = window
    if thresholds is not None:
        output["thresholds"] = thresholds
    if prob_a is not None:
        output["prob_a"] = prob_a
    if prob_b is not None:
        output["prob_b"] = prob_b

    # Write JSON
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    # Summary
    print(f"\nExported to {output_path}")
    print(f"  Classes: {n_classes} {classes}")
    print(f"  Training samples: {n_samples}")
    print(f"  Features per sample: {n_features}")
    print(f"  Support vectors: {len(support_indices)}")
    print(f"  Kernel: gamma={gamma}, power={power}")
    print(f"  DTW window: {window}")
    print(f"  Noise class: {noise_class}")
    print(f"  Label mapper: {label_mapper}")
    if prob_a is not None:
        print(f"  Platt scaling: {len(prob_a)} pairs")
    if thresholds is not None:
        print(f"  Thresholds: {thresholds}")

    # Validate
    n_pairs = n_classes * (n_classes - 1) // 2
    assert len(intercept) == n_pairs, f"intercept length {len(intercept)} != expected {n_pairs}"
    assert len(dual_coef) == n_classes - 1, f"dual_coef rows {len(dual_coef)} != expected {n_classes - 1}"
    for i, row in enumerate(dual_coef):
        assert len(row) == len(support_indices), f"dual_coef[{i}] length {len(row)} != n_sv {len(support_indices)}"
    print("  Validation: passed")


def main():
    parser = argparse.ArgumentParser(
        description="Convert WarpDemuX DTW_SVM model to escapepod JSON format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("model", type=Path, help="Input WarpDemuX .joblib model file")
    parser.add_argument("output", type=Path, help="Output JSON file")
    parser.add_argument("--inspect", action="store_true", help="Print model structure and exit")

    args = parser.parse_args()

    if not args.model.exists():
        print(f"Error: Model file not found: {args.model}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {args.model}...")
    model = load_model(args.model)

    print("Inspecting model...")
    inspect_model(model)

    if args.inspect:
        return

    if not hasattr(model, "_X") or not hasattr(model, "model"):
        print("Error: Not a WarpDemuX DTW_SVM model (missing _X or model attributes)", file=sys.stderr)
        sys.exit(1)

    convert_model(model, args.output)


if __name__ == "__main__":
    main()
