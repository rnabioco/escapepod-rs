#!/usr/bin/env python3
"""Reference tensors and logits for a charging `waveform_model` bundle.

The oracle `examples/verify_waveform_model.rs` is checked against. It takes a
leech-prepared corpus (the `.npz` the training chain writes) and, for a slice
of its chunks, emits the *three input tensors exactly as the corpus holds
them* plus the logit onnxruntime produces from them.

That split is the whole point, and it is what
`rnabioco/escapepod-rs#306` asks for. Two things can be wrong in a Rust
consumer of this model and they need separating:

* the **graph** — does escpod's runtime get the same number out of the same
  tensors? The ONNX itself already round-trips against torch at 1.4e-05 with
  zero decision disagreements over 4,096 real chunks, so a residue here is the
  runtime, not the export.
* the **assembly** — does escpod build the same tensors from POD5 and BAM?
  That is the part with no reference implementation in Rust and every
  opportunity to differ silently: a window justified to the wrong side of a
  base, a channel list permuted, a k-mer context split `(4, 4)` the other way
  round. Each produces a correctly shaped tensor full of plausible numbers.

So this writes the tensors, not just the scores, and the Rust side compares
both.

It imports **nothing from leech or escapepod-models**: the corpus arrays are
read positionally out of the `.npz` and the graph is run through onnxruntime
from the bundle's own declared input names. An agreement is therefore evidence
that the bundle describes itself, not that two copies of one library agree.

Writes beside ``--out``:

    <out>.json          shapes, the read ids, the base indices, the bundle id
    <out>.signal.f32    [n, C, L] row-major f32
    <out>.sequence.f32  [n, R, L] row-major f32
    <out>.features.f32  [n, F, W] row-major f32
    <out>.logit.f32     [n] f32
    <out>.focus.i64     [n] i64 -- the sample the window was centred on
    <out>.dwell.f32     [n, W] f32 -- per-base dwells over the feature window

The last two are diagnostics, not model inputs. They separate *where* the
window was cut from *what* was in it: if the focus sample and the dwells agree
but the tensors do not, the map is right and the values are wrong; if the focus
disagrees, everything downstream of it is displaced and the tensors will differ
in a way that looks like noise.

Usage::

    python scripts/dump_waveform_reference.py \\
        --bundle models/charging_tcn_rna004@v0.1.0 \\
        --corpus benchmarks/.../test/test.npz --n 512 \\
        --out /tmp/tcn_ref
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


def encode_signal_kmer(
    seq_ints: np.ndarray, seq_to_sig: np.ndarray, signal_len: int, before: int, after: int
) -> np.ndarray:
    """The `sequence` tensor: one 4-row one-hot block per k-mer position,
    scattered along the signal axis by the base-to-signal map.

    A transcription of the rule the bundle names
    (`escapepod_signal::seq_encoding::encode_signal_kmer`), written here rather
    than imported so that the comparison is against the *contract* and not
    against the same code twice. Spans are intersected with `[0, signal_len)`,
    so a base that starts before the window still contributes its tail.
    """
    kmer_len = before + 1 + after
    out = np.zeros((4 * kmer_len, signal_len), dtype=np.float32)
    n_bases = len(seq_to_sig) - 1
    for kmer_pos in range(kmer_len):
        block = 4 * kmer_pos
        for seq_pos in range(n_bases):
            idx = seq_pos + kmer_pos
            if idx >= len(seq_ints):
                continue
            base = int(seq_ints[idx])
            if base < 0:
                continue
            lo = max(int(seq_to_sig[seq_pos]), 0)
            hi = min(int(seq_to_sig[seq_pos + 1]), signal_len)
            if hi > lo:
                out[block + base, lo:hi] = 1.0
    return out


BASE_TO_INT = {"A": 0, "C": 1, "G": 2, "T": 3, "U": 3}


def sequence_to_int(seq: str) -> np.ndarray:
    return np.array([BASE_TO_INT.get(c.upper(), -1) for c in seq], dtype=np.int8)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, required=True, help="leech-prepared .npz")
    ap.add_argument("--n", type=int, default=512, help="chunks to take from the front")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    meta = json.loads((args.bundle / "metadata.json").read_text())
    wm = meta["waveform_model"]
    pre = wm["preprocessing"]
    sig_len = int(pre["signal_len"])
    n_sig_ch = int(pre["signal_channels"])
    n_feat = int(pre["n_feature_channels"])
    feat_w = int(pre["feature_width"])
    before, after = pre["signal_kmer_context"]

    # `allow_pickle` because the per-chunk base-to-signal maps are ragged and
    # travel as an object array in the per-class files.
    z = np.load(args.corpus, allow_pickle=True)
    n = min(args.n, len(z["read_ids"]))

    read_ids = [str(x) for x in z["read_ids"][:n]]
    base_indices = [int(x) for x in z["base_indices"][:n]]
    focus = np.asarray(z["focus_signal_pos"][:n], dtype=np.int64)
    dwells = z["dwells_flat"].reshape(-1, feat_w)[:n].astype(np.float32)

    # Two layouts exist: the per-class files store each per-chunk array with
    # its own trailing axes, the merged file flattens them. Reshaping to the
    # geometry the bundle declares reads both, and disagrees loudly with a
    # corpus that is not this model's.
    signals = z["signals_flat"].reshape(-1, sig_len)[:n]
    residuals = z["signal_residuals_flat"].reshape(-1, sig_len)[:n]
    features = z["features_flat"].reshape(-1, n_feat, feat_w)[:n]

    # `sequence` is not stored as a tensor: the corpus keeps the context string
    # and the chunk-local map it is scattered by, and the dataset encodes at
    # load. Same here.
    if "seq_to_sig_maps" in z.files:
        maps = [np.asarray(m, dtype=np.int64) for m in z["seq_to_sig_maps"][:n]]
    else:
        offsets = z["seq_to_sig_offsets"]
        values = z["seq_to_sig_values"]
        maps = [values[offsets[i] : offsets[i + 1]] for i in range(n)]
    ctx_seqs = z["sequences_with_kmer_context"][:n]

    signal_t = np.stack(
        [signals, residuals][:n_sig_ch], axis=1
    ).astype(np.float32)  # [n, C, L]
    seq_t = np.zeros((n, 4 * (before + 1 + after), sig_len), dtype=np.float32)
    for i in range(n):
        seq_t[i] = encode_signal_kmer(
            sequence_to_int(str(ctx_seqs[i])), maps[i], sig_len, before, after
        )
    feat_t = features.astype(np.float32)

    sess = ort.InferenceSession(
        str(args.bundle / wm["file"]), providers=["CPUExecutionProvider"]
    )
    names = [i.name for i in sess.get_inputs()]
    by_name = {"signal": signal_t, "sequence": seq_t, "features": feat_t}
    missing = [x for x in names if x not in by_name]
    if missing:
        raise SystemExit(f"graph wants inputs this script does not build: {missing}")

    logits = np.empty(n, dtype=np.float32)
    step = 64
    for lo in range(0, n, step):
        hi = min(lo + step, n)
        feed = {name: by_name[name][lo:hi] for name in names}
        out = sess.run(None, feed)[0]
        logits[lo:hi] = np.asarray(out, dtype=np.float32).reshape(-1)

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    signal_t.tofile(f"{out}.signal.f32")
    seq_t.tofile(f"{out}.sequence.f32")
    feat_t.tofile(f"{out}.features.f32")
    logits.tofile(f"{out}.logit.f32")
    focus.tofile(f"{out}.focus.i64")
    # The k-mer window and the chunk-local context sequence, as text. These
    # probe the reference slice and the chunk's base range WITHOUT involving
    # the signal at all, which is what splits a sequence-side disagreement
    # from a signal-side one.
    Path(f"{out}.seq.txt").write_text(
        "\n".join(str(x) for x in z["sequences"][:n]) + "\n"
    )
    Path(f"{out}.ctx.txt").write_text(
        "\n".join(str(x) for x in z["sequences_with_kmer_context"][:n]) + "\n"
    )
    dwells.tofile(f"{out}.dwell.f32")
    Path(f"{out}.json").write_text(
        json.dumps(
            {
                "bundle": f"{meta['model']['id']}@{meta['model']['version']}",
                "n": n,
                "signal_shape": list(signal_t.shape),
                "sequence_shape": list(seq_t.shape),
                "features_shape": list(feat_t.shape),
                "read_ids": read_ids,
                "base_indices": base_indices,
                "input_names": names,
            },
            indent=2,
        )
    )
    print(f"wrote {n} chunks to {out}.*")
    print(f"logit range [{logits.min():.4f}, {logits.max():.4f}]")


if __name__ == "__main__":
    main()
