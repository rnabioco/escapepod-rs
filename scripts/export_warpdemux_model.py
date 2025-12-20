#!/usr/bin/env python3
"""
Export WarpDemuX model to JSON format for use with escapepod.

This script reads a trained WarpDemuX model (saved as .joblib) and exports it
to a portable JSON format that can be loaded by the escapepod demux classifier.

The exported JSON contains:
- training_fingerprints: The training data (X) used to fit the model
- training_labels: The class labels (y) for each training fingerprint
- kernel_params: gamma and power parameters for the RBF kernel
- label_map: mapping from barcode names to integer IDs
- threshold: classification threshold value
- threshold_type: type of threshold used (e.g., "ratio", "kernel")

Usage:
    python export_warpdemux_model.py model.joblib -o model.json
    python export_warpdemux_model.py model.joblib -o model.json --gamma 0.5 --power 2.0

The .joblib file should contain a trained WarpDemuX classifier with:
- _X: training fingerprints (numpy array of shape [n_samples, n_features])
- _y: training labels (numpy array of shape [n_samples])
- gamma: RBF kernel gamma parameter (optional, can be overridden)
- power: RBF kernel power parameter (optional, can be overridden)
- threshold: classification threshold
- threshold_type: type of threshold

Example WarpDemuX model structure:
    model = {
        '_X': np.array([[...], [...], ...]),  # Training fingerprints
        '_y': np.array([0, 1, 2, ...]),       # Training labels
        'gamma': 0.1,                         # Kernel parameter
        'power': 1.0,                         # Kernel parameter
        'threshold': 0.8,                     # Classification threshold
        'threshold_type': 'ratio',            # Threshold type
        'label_map': {'BC01': 0, 'BC02': 1, ...}  # Label mapping
    }
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import joblib
    import numpy as np
except ImportError as e:
    print(f"Error: Required Python packages not found: {e}", file=sys.stderr)
    print("Please install: pip install joblib numpy", file=sys.stderr)
    sys.exit(1)


def export_warpdemux_model(
    model_path: Path,
    output_path: Path,
    gamma: float = None,
    power: float = None,
    threshold: float = None,
    threshold_type: str = None,
) -> None:
    """
    Export a WarpDemuX model to JSON format.

    Args:
        model_path: Path to the .joblib model file
        output_path: Path to write the JSON output
        gamma: Override gamma parameter (if None, use model's gamma)
        power: Override power parameter (if None, use model's power)
        threshold: Override threshold (if None, use model's threshold)
        threshold_type: Override threshold type (if None, use model's type)
    """
    # Load the model
    print(f"Loading model from {model_path}...")
    try:
        model = joblib.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        sys.exit(1)

    # Extract training data
    if hasattr(model, '_X') or '_X' in model:
        X = model._X if hasattr(model, '_X') else model['_X']
    else:
        print("Error: Model does not contain '_X' (training fingerprints)", file=sys.stderr)
        sys.exit(1)

    if hasattr(model, '_y') or '_y' in model:
        y = model._y if hasattr(model, '_y') else model['_y']
    else:
        print("Error: Model does not contain '_y' (training labels)", file=sys.stderr)
        sys.exit(1)

    # Convert numpy arrays to lists for JSON serialization
    X_list = X.tolist() if isinstance(X, np.ndarray) else list(X)
    y_list = y.tolist() if isinstance(y, np.ndarray) else list(y)

    print(f"  Training samples: {len(X_list)}")
    print(f"  Feature dimension: {len(X_list[0]) if X_list else 0}")
    print(f"  Unique labels: {len(set(y_list))}")

    # Extract or use provided kernel parameters
    model_gamma = None
    if hasattr(model, 'gamma'):
        model_gamma = model.gamma
    elif 'gamma' in model:
        model_gamma = model['gamma']

    model_power = None
    if hasattr(model, 'power'):
        model_power = model.power
    elif 'power' in model:
        model_power = model['power']

    final_gamma = gamma if gamma is not None else model_gamma if model_gamma is not None else 1.0
    final_power = power if power is not None else model_power if model_power is not None else 1.0

    print(f"  Kernel gamma: {final_gamma}")
    print(f"  Kernel power: {final_power}")

    # Extract or use provided threshold
    model_threshold = None
    if hasattr(model, 'threshold'):
        model_threshold = model.threshold
    elif 'threshold' in model:
        model_threshold = model['threshold']

    final_threshold = threshold if threshold is not None else model_threshold if model_threshold is not None else 0.8

    # Extract or use provided threshold type
    model_threshold_type = None
    if hasattr(model, 'threshold_type'):
        model_threshold_type = model.threshold_type
    elif 'threshold_type' in model:
        model_threshold_type = model['threshold_type']

    final_threshold_type = threshold_type if threshold_type is not None else model_threshold_type if model_threshold_type is not None else "ratio"

    print(f"  Threshold: {final_threshold}")
    print(f"  Threshold type: {final_threshold_type}")

    # Extract label mapping
    label_map = {}
    if hasattr(model, 'label_map'):
        label_map = model.label_map
    elif 'label_map' in model:
        label_map = model['label_map']
    elif hasattr(model, 'classes_'):
        # Scikit-learn style classifier
        label_map = {str(label): idx for idx, label in enumerate(model.classes_)}
    else:
        # Create a default mapping from unique labels
        unique_labels = sorted(set(y_list))
        label_map = {f"barcode_{label}": label for label in unique_labels}

    print(f"  Label map: {label_map}")

    # Create the output dictionary
    output_data = {
        "training_fingerprints": X_list,
        "training_labels": y_list,
        "kernel_params": {
            "gamma": final_gamma,
            "power": final_power,
        },
        "label_map": label_map,
        "threshold": final_threshold,
        "threshold_type": final_threshold_type,
    }

    # Write to JSON
    print(f"Writing model to {output_path}...")
    try:
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"Successfully exported model to {output_path}")
    except Exception as e:
        print(f"Error writing output: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Export WarpDemuX model to JSON format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "model",
        type=Path,
        help="Path to WarpDemuX .joblib model file"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        required=True,
        help="Output JSON file path"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        help="Override RBF kernel gamma parameter (default: use model's gamma or 1.0)"
    )
    parser.add_argument(
        "--power",
        type=float,
        help="Override RBF kernel power parameter (default: use model's power or 1.0)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="Override classification threshold (default: use model's threshold or 0.8)"
    )
    parser.add_argument(
        "--threshold-type",
        type=str,
        choices=["ratio", "kernel"],
        help="Override threshold type (default: use model's type or 'ratio')"
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.model.exists():
        print(f"Error: Model file not found: {args.model}", file=sys.stderr)
        sys.exit(1)

    export_warpdemux_model(
        args.model,
        args.output,
        gamma=args.gamma,
        power=args.power,
        threshold=args.threshold,
        threshold_type=args.threshold_type,
    )


if __name__ == "__main__":
    main()
