#!/usr/bin/env python3
"""Export WarpDemuX models to JSON format for use with escapepod-rs.

This script converts WarpDemuX's Python/sklearn models (stored as joblib files)
to the JSON format that escapepod-rs can load for inference.

Usage:
    python export_warpdemux_models.py --model model.joblib --output model.json
    python export_warpdemux_models.py --model model.joblib --fingerprints fingerprints.csv --output model.json

Supported model types:
    - DTW-SVM (sklearn.svm.SVC)

The output JSON contains:
    - training_fingerprints: Reference fingerprints for DTW computation
    - training_labels: Class labels for each fingerprint
    - support_indices: Indices of support vectors
    - dual_coef: SVM dual coefficients
    - intercept: SVM intercept terms
    - classes: Unique class labels
    - kernel_params: RBF kernel parameters (gamma, power)
    - label_mapper: Class index to barcode ID mapping
    - prob_a/prob_b: Platt scaling parameters (if available)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def load_joblib_model(model_path: Path):
    """Load a model from a joblib file."""
    try:
        import joblib
    except ImportError:
        print("Error: joblib is required. Install with: pip install joblib", file=sys.stderr)
        sys.exit(1)

    return joblib.load(model_path)


def load_fingerprints(fingerprints_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load fingerprints and labels from a CSV file.

    Expected format: read_id,barcode,feat1,feat2,...,featN

    Returns:
        fingerprints: Array of shape (n_samples, n_features)
        labels: Array of shape (n_samples,)
    """
    fingerprints = []
    labels = []

    with open(fingerprints_path) as f:
        # Skip header
        next(f)
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 3:
                barcode = int(parts[1]) if parts[1].isdigit() else hash(parts[1]) % 1000
                features = [float(x) for x in parts[2:]]
                fingerprints.append(features)
                labels.append(barcode)

    return np.array(fingerprints), np.array(labels)


def export_svm_model(
    model,
    fingerprints: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    kernel_gamma: float = 1.0,
    kernel_power: float = 1.0,
    window: int = None,
    thresholds: list = None,
):
    """Export an sklearn SVM model to JSON format for escapepod-rs.

    Args:
        model: sklearn.svm.SVC model
        fingerprints: Training fingerprints (n_samples, n_features)
        labels: Training labels (n_samples,)
        output_path: Output JSON file path
        kernel_gamma: RBF kernel gamma parameter
        kernel_power: Power to raise distances before exponential
        window: DTW window constraint (Sakoe-Chiba band)
        thresholds: Per-class confidence thresholds
    """
    # Get unique classes
    classes = sorted(np.unique(labels).tolist())
    n_classes = len(classes)

    # Create label mapper: class_index -> barcode_id
    label_mapper = {i: int(c) for i, c in enumerate(classes)}

    # Get support vector indices
    if hasattr(model, 'support_'):
        support_indices = model.support_.tolist()
    else:
        # If model doesn't expose support vectors, use all training points
        support_indices = list(range(len(fingerprints)))

    # Get dual coefficients
    if hasattr(model, 'dual_coef_'):
        dual_coef = model.dual_coef_.tolist()
    else:
        # Placeholder if not available
        n_sv = len(support_indices)
        dual_coef = [[1.0 / n_sv] * n_sv for _ in range(n_classes - 1)]

    # Get intercepts
    if hasattr(model, 'intercept_'):
        intercept = model.intercept_.tolist()
    else:
        n_pairs = n_classes * (n_classes - 1) // 2
        intercept = [0.0] * n_pairs

    # Get Platt scaling parameters if available
    prob_a = None
    prob_b = None
    if hasattr(model, 'probA_') and model.probA_ is not None:
        prob_a = model.probA_.tolist()
    if hasattr(model, 'probB_') and model.probB_ is not None:
        prob_b = model.probB_.tolist()

    # Build the model dictionary
    model_dict = {
        "version": "1.0",
        "training_fingerprints": fingerprints.tolist(),
        "training_labels": [int(l) for l in labels.tolist()],
        "support_indices": support_indices,
        "dual_coef": dual_coef,
        "intercept": intercept,
        "classes": [int(c) for c in classes],
        "kernel_params": {
            "gamma": kernel_gamma,
            "power": kernel_power,
        },
        "label_mapper": {str(k): v for k, v in label_mapper.items()},
        "n_classes": n_classes,
        "noise_class": False,
    }

    # Add optional fields
    if window is not None:
        model_dict["window"] = window
    if thresholds is not None:
        model_dict["thresholds"] = thresholds
    if prob_a is not None:
        model_dict["prob_a"] = prob_a
    if prob_b is not None:
        model_dict["prob_b"] = prob_b

    # Write to JSON
    with open(output_path, 'w') as f:
        json.dump(model_dict, f, indent=2)

    print(f"Exported SVM model to {output_path}")
    print(f"  Classes: {n_classes}")
    print(f"  Training samples: {len(fingerprints)}")
    print(f"  Support vectors: {len(support_indices)}")
    print(f"  Features: {fingerprints.shape[1]}")


def main():
    parser = argparse.ArgumentParser(
        description="Export WarpDemuX models to JSON format for escapepod-rs"
    )
    parser.add_argument(
        "--model",
        type=Path,
        help="Input WarpDemuX model file (joblib format)",
    )
    parser.add_argument(
        "--fingerprints",
        type=Path,
        required=True,
        help="Training fingerprints CSV file (read_id,barcode,feat1,...)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Output JSON model file",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="RBF kernel gamma parameter (default: 1.0)",
    )
    parser.add_argument(
        "--power",
        type=float,
        default=1.0,
        help="Power to raise distances before exponential (default: 1.0)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=None,
        help="DTW window constraint (Sakoe-Chiba band)",
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default=None,
        help="Per-class confidence thresholds (comma-separated)",
    )

    args = parser.parse_args()

    # Load fingerprints
    print(f"Loading fingerprints from {args.fingerprints}...")
    fingerprints, labels = load_fingerprints(args.fingerprints)
    print(f"  Loaded {len(fingerprints)} fingerprints with {fingerprints.shape[1]} features")

    # Load model if provided
    model = None
    if args.model:
        print(f"Loading model from {args.model}...")
        model = load_joblib_model(args.model)
        print(f"  Model type: {type(model).__name__}")

    # Parse thresholds
    thresholds = None
    if args.thresholds:
        thresholds = [float(x) for x in args.thresholds.split(',')]

    # Export
    export_svm_model(
        model=model,
        fingerprints=fingerprints,
        labels=labels,
        output_path=args.output,
        kernel_gamma=args.gamma,
        kernel_power=args.power,
        window=args.window,
        thresholds=thresholds,
    )


if __name__ == "__main__":
    main()
