#!/usr/bin/env python3
"""Regenerate the `feature_model` (per-base-feature ONNX) parity fixtures.

The companion to ``gen_charging_golden.py``, for the *other* scorer over the
same features. That script pins the feature chain and the GBM; this one pins
the three rules that stand between the flat feature vector and an ONNX graph —
the fold, the per-channel standardisation, and the explicit encoding of
missingness — plus the graph itself.

It deliberately does **not** recompute the features. It reads them out of
``charging_golden.json``, i.e. the reference implementation's own f32 output,
and folds *those*. That is the whole point of the decomposition: escpod's
feature grid differs from NumPy's in the last bits (reduction order, bounded
at 1e-4 by the existing test), so a golden built end to end could not tell a
wrong fold from a rounding difference. Feeding both sides the identical
feature vector makes the tensor comparison **exact**, and any residue in the
final probability is then attributable to the features alone.

Builds, beside this script:

- ``bundle_fnn/`` — a complete, hash-pinned bundle in the `fnn` shape
  ``workflow/scripts/build_charging_bundle.py`` emits: the ONNX, a
  ``feature_model`` block declaring channels / offsets / mu / sd, the same
  ``features`` block the GBM fixture bundle carries, and the same k-mer table;
- ``charging_fnn_golden.json`` — per read: the reference feature vector, the
  network input tensor ``feature_nn_input(feature_nn_fold(...))`` produced,
  the ONNX logits, ``P(charged)`` and the ``cl`` byte.

Weights are random but seeded, and the BatchNorm buffers are randomised too:
left at their defaults those layers are the identity in eval mode, and a
fixture whose graph is half no-ops tests less than it appears to.

Run with the escapepod-models pixi env (torch, onnxruntime, numpy)::

    CHARGING_PY=~/devel/rnabioco/escapepod-models/src/escapepod_models/charging.py \
    pixi run -e boundary python gen_charging_fnn_golden.py
"""

import hashlib
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
CHARGING_PY = Path(
    os.environ.get(
        "CHARGING_PY",
        Path.home() / "devel/rnabioco/escapepod-models/src/escapepod_models/charging.py",
    )
)

spec = importlib.util.spec_from_file_location("charging", CHARGING_PY)
charging = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = charging
spec.loader.exec_module(charging)

# The fixture GBM bundle — and therefore the golden features — is the full
# -8..+16 grid: 25 offsets x 4 stats, feature set "all".
FEAT_LO, FEAT_HI = -8, 16
FEATURE_SET = "all"
ARCH, HIDDEN = "cnn", 16
# See the head-scaling comment below: an untrained net barely separates 19 reads.
HEAD_GAIN = 40.0
charging.set_feature_window(FEAT_LO, FEAT_HI)

SRC_BUNDLE = HERE / "bundle"
BUNDLE = HERE / "bundle_fnn"
MODEL_ID = "charging_fnn_fixture"


def f32_bits(a):
    return [int(x) for x in np.asarray(a, dtype=np.float32).view(np.uint32).ravel()]


def f64_bits(x):
    return int(np.float64(x).view(np.uint64))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- the reference features, straight out of the GBM golden ------------------
gbm_golden = json.loads((HERE / "charging_golden.json").read_text())
reads = gbm_golden["reads"]
F = np.stack(
    [
        np.array(r["features_bits"], dtype=np.uint32).view(np.float32)
        for r in reads
    ]
)
names = charging.feature_names()
assert F.shape[1] == len(names) == 100, F.shape

# --- fold, gauge, input ------------------------------------------------------
channels = charging.feature_nn_channels(FEATURE_SET)
n_val = len(channels) // 2
n_off = F.shape[1] // n_val
assert channels[:n_val] == list(charging.FEAT_STATS)

Fs = charging.apply_feature_set(F, FEATURE_SET)
X = charging.feature_nn_fold(Fs, n_val)
assert X.shape == (len(reads), n_val, n_off), X.shape

# Fitted on these reads. A shipped model fits on its TRAIN split alone; the
# fixture has one split, and what is under test is that the constants travel
# and are applied, not how they were chosen.
mu, sd = charging.feature_nn_gauge(X)
Xn = charging.feature_nn_input(X, mu, sd)
assert Xn.shape == (len(reads), 2 * n_val, n_off), Xn.shape
# The fixture must actually exercise the missingness path, or the mask
# channels and the zeroing rule are untested.
assert (Xn[:, n_val:] == 0.0).any(), "no unresolved bases in the fixture features"

# --- a small, fully randomised network --------------------------------------
torch.manual_seed(20260817)
model = charging.build_feature_nn(ARCH, len(channels), n_off, HIDDEN)
with torch.no_grad():
    # Torch's own init for the weights — a hand-rolled N(0, 0.5) blows up
    # through two convolutions and a 400-wide head, saturating the softmax so
    # every golden probability is ~0 and the `cl` bytes stop distinguishing
    # anything. Only the buffers are overridden, because BatchNorm in eval
    # mode with default buffers is the identity and would leave those layers
    # untested.
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm1d):
            m.weight.normal_(1.0, 0.2)
            m.bias.normal_(0.0, 0.2)
            m.running_mean.normal_(0.0, 0.5)
            m.running_var.uniform_(0.5, 2.0)
    # An untrained head separates these 19 reads by ~0.06 in logit, i.e. every
    # golden probability within 0.03 of a half and every `cl` byte within
    # three of 128 — a golden that would pass against a broken model almost as
    # well as against a correct one. Scaling the head spreads them over the
    # range without touching the arithmetic under test.
    model[-1].weight.mul_(HEAD_GAIN)
    model[-1].bias.mul_(HEAD_GAIN)
model.eval()

ckpt = HERE / "_fnn_fixture.pt"
torch.save(
    {
        "arch": ARCH,
        "hidden": HIDDEN,
        "channels": channels,
        "n_off": n_off,
        "feature_set": FEATURE_SET,
        "feat_lo": FEAT_LO,
        "feat_hi": FEAT_HI,
        "state_dict": model.state_dict(),
        "standardisation": {
            "method": "per-channel affine over observed values",
            "mu": [float(v) for v in mu],
            "sd": [float(v) for v in sd],
        },
    },
    ckpt,
)

BUNDLE.mkdir(exist_ok=True)
onnx_path = BUNDLE / f"{MODEL_ID}.onnx"
# Exported through the package's own function, so the fixture is produced by
# the code path that produces shipped models.
report = charging.export_feature_nn_onnx(str(ckpt), str(onnx_path))
ckpt.unlink()
print("export:", report)

import onnxruntime as ort  # noqa: E402

# Single-threaded: the default pool tries to pin affinities and floods a
# SLURM allocation with pthread_setaffinity_np errors.
_so = ort.SessionOptions()
_so.intra_op_num_threads = 1
_so.inter_op_num_threads = 1
sess = ort.InferenceSession(
    str(onnx_path), sess_options=_so, providers=["CPUExecutionProvider"]
)
logits = sess.run(None, {"features": Xn.astype(np.float32)})[0]
with torch.no_grad():
    want = model(torch.from_numpy(Xn.astype(np.float32))).numpy()
max_abs = float(np.abs(logits - want).max())
print(f"onnx vs torch logits: max |diff| = {max_abs:.3e}")
assert max_abs < 1e-4, max_abs

# P(charged) in f64, matching what the runtime computes from f32 logits.
lo = logits.astype(np.float64)
lo = lo - lo.max(axis=1, keepdims=True)
e = np.exp(lo)
p = e[:, 1] / e.sum(axis=1)

# --- the bundle --------------------------------------------------------------
shutil.copy2(SRC_BUNDLE / "kmer_levels.tsv", BUNDLE / "kmer_levels.tsv")
src_meta = json.loads((SRC_BUNDLE / "metadata.json").read_text())

meta = {
    "format": "escapepod-charging-classifier/1",
    "model": {
        "id": MODEL_ID,
        "version": "0.0.1",
        "chemistry": "rna004",
        "task": "tRNA aminoacylation state (charged vs uncharged) — TEST FIXTURE",
    },
    "classes": list(charging.LABELS),
    "anchor": src_meta["anchor"],
    "features": src_meta["features"] | {"feature_set": FEATURE_SET},
    "feature_model": {
        "file": onnx_path.name,
        "sha256": sha256(onnx_path),
        "opset": 17,
        "arch": ARCH,
        "input": {
            "name": "features",
            "shape": [None, len(channels), n_off],
            "dtype": "float32",
            "layout": "batch, channel, offset",
            "channels": channels,
            "n_offsets": n_off,
            "fold": (
                "`features.order` is offsets-outer / channels-inner, so the "
                f"k-th selected column is offset k // {n_val}, value channel "
                f"k % {n_val}. Transpose to [channel, offset]."
            ),
        },
        "output": {
            "name": "logits",
            "shape": [None, 2],
            "classes": list(charging.LABELS),
            "activation": "softmax over dim 1; P(charged) = softmax(logits)[:, 1]",
        },
        "standardisation": {
            "method": "per-channel affine over observed values",
            "mu": [float(v) for v in mu],
            "sd": [float(v) for v in sd],
            "apply": (
                "value channel c = observed ? (x - mu[c]) / sd[c] : 0.0; "
                "observed channel c = observed ? 1.0 : 0.0, where `observed` "
                "means the selected feature is finite"
            ),
        },
        "missing": (
            "NaN is NOT passed to the network: the value channel is zeroed and "
            "the paired observed channel carries the indicator."
        ),
    },
    "kmer_table": src_meta["kmer_table"],
    "operating_point": {
        "probability": 0.5,
        "cl": 128,
        "source": "arbitrary fixture operating point (weights are random)",
    },
}
(BUNDLE / "metadata.json").write_text(json.dumps(meta, indent=1))

# --- golden ------------------------------------------------------------------
golden = {
    "numpy": np.__version__,
    "torch": torch.__version__,
    "onnxruntime": ort.__version__,
    "channels": channels,
    "n_offsets": n_off,
    "mu": [float(v) for v in mu],
    "sd": [float(v) for v in sd],
    "onnx_vs_torch_max_abs_logit": max_abs,
    "reads": [
        {
            "read_id": r["read_id"],
            # The reference features this row was folded from — the runtime
            # feeds these to its own fold so the tensor check is exact.
            "features_bits": r["features_bits"],
            "input_bits": f32_bits(Xn[i]),
            "logits_bits": f32_bits(logits[i]),
            "p_bits": f64_bits(p[i]),
            "cl": int(np.round(p[i] * 255)),
        }
        for i, r in enumerate(reads)
    ],
}
(HERE / "charging_fnn_golden.json").write_text(json.dumps(golden, indent=1))
print(
    f"wrote {BUNDLE}/ and charging_fnn_golden.json "
    f"({len(reads)} reads, {len(channels)}x{n_off} input, "
    f"P range {p.min():.4f}..{p.max():.4f})"
)
