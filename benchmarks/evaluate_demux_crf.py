#!/usr/bin/env python3
"""Score `escpod demux` classification CSVs against a labelled truth table, and
against each other.

usage: evaluate_demux_crf.py [--truth truth.csv] [--near-tie NATS]
                             [--solo-axis AXIS] LABEL=CSV [LABEL=CSV ...]

Each positional argument is one arm: a classification CSV written by
`escpod demux --classifications`, either the fused form (`read_id, ldx,
ldx_confidence, [ldx_crf_logp, ldx_crf_margin, ldx_crf_best,
ldx_mean_logpost,] fdx, ...`) or the single-model form (`read_id, barcode,
confidence, [crf_logp, crf_margin, crf_best, mean_logpost]`), whose axis is
taken from --solo-axis (default `ldx`). A bare path is labelled by its
basename.

Against the truth (a CSV with `read_id` and one `<axis>_truth` column per
axis; rows with an empty truth are not scored on that axis): reads called,
yield, accuracy over the labelled reads, and precision when called.

Against arm 0, read for read on the reads both arms hold: identical calls,
differing calls, and — when both CSVs carry `<axis>_crf_margin`, i.e. were
written with `--ref-scores` — the margin distribution of the differing calls
and how many sit under --near-tie (default 0.25 nats). That attribution is the
contract for an encoder change: CPU and GPU differ on 4 of 20,000 calls, all
under 0.25 nats, and a native kernel is held to the same bar
(rnabioco/escapepod-rs#331).

Needs pandas (the `python-test` pixi env has it).
"""

import argparse
import os
import sys

import pandas as pd

UNCLASSIFIED = "unclassified"


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--truth", default=None, help="truth CSV; omit to skip accuracy")
    p.add_argument("--near-tie", type=float, default=0.25, metavar="NATS")
    p.add_argument("--solo-axis", default="ldx", metavar="AXIS")
    p.add_argument("arms", nargs="+", metavar="LABEL=CSV")
    return p.parse_args(argv)


def load_arm(spec, solo_axis):
    """Return `(label, frame, axes)`; a single-model CSV is renamed onto `solo_axis`."""
    label, _, path = spec.rpartition("=") if "=" in spec else ("", "", spec)
    label = label or os.path.splitext(os.path.basename(path))[0]
    df = pd.read_csv(path, dtype={"read_id": str}).set_index("read_id")
    if "barcode" in df.columns:
        rename = {c: f"{solo_axis}_{c}" for c in df.columns if c != "barcode"}
        rename["barcode"] = solo_axis
        df = df.rename(columns=rename)
    axes = [c for c in df.columns if f"{c}_confidence" in df.columns]
    if not axes:
        sys.exit(f"{path}: no axis column found (expected `barcode` or `<axis>` + `<axis>_confidence`)")
    return label, df, axes


def score_against_truth(label, df, axes, truth):
    for ax in axes:
        col = f"{ax}_truth"
        if col not in truth.columns:
            print(f"  {label:>10s} {ax}: truth has no `{col}` column, not scored")
            continue
        t = truth[truth[col].notna()][[col]].join(df[[ax]], how="inner")
        n = len(t)
        called = (t[ax] != UNCLASSIFIED) & t[ax].notna()
        correct = (t[ax] == t[col]).sum()
        print(
            f"  {label:>10s} {ax}: labelled {n:6d}  called {called.sum():6d} "
            f"({called.mean():6.1%})  accuracy {correct / max(n, 1):6.1%}  "
            f"precision-when-called {correct / max(called.sum(), 1):6.1%}"
        )


def compare(ref_label, ref, arm_label, arm, axes, near_tie):
    common = ref.index.intersection(arm.index)
    only_ref = len(ref.index.difference(arm.index))
    only_arm = len(arm.index.difference(ref.index))
    print(f"  {arm_label} vs {ref_label}: {len(common)} reads in both"
          + (f", {only_ref} only in {ref_label}, {only_arm} only in {arm_label}" if only_ref or only_arm else ""))
    a, b = ref.loc[common], arm.loc[common]
    for ax in axes:
        if ax not in b.columns:
            print(f"    {ax}: absent from {arm_label}")
            continue
        same = (a[ax] == b[ax]) | (a[ax].isna() & b[ax].isna())
        diff = ~same
        conf = f"{ax}_confidence"
        conf_same = ((a[conf] - b[conf]).abs() < 1e-6).sum() if conf in b.columns else None
        line = f"    {ax}: identical {same.sum():6d}  differing {diff.sum():4d}"
        if conf_same is not None:
            line += f"  confidence identical {conf_same}"
        print(line)
        if diff.sum() == 0:
            continue
        d_a, d_b = a[diff], b[diff]
        one_unc = ((d_a[ax] == UNCLASSIFIED) ^ (d_b[ax] == UNCLASSIFIED)).sum()
        print(f"      of which one side unclassified: {one_unc}, both called differently: {diff.sum() - one_unc}")
        mcol = f"{ax}_crf_margin"
        if mcol in a.columns and mcol in b.columns:
            m = pd.concat([d_a[mcol].abs(), d_b[mcol].abs()], axis=1).min(axis=1)
            q = m.quantile([0, 0.5, 1.0])
            print(
                f"      |crf_margin| (smaller side): min {q[0]:.3f}  median {q[0.5]:.3f}  max {q[1.0]:.3f}"
                f"  under {near_tie} nats: {(m < near_tie).sum()} of {len(m)}"
            )
        else:
            print(f"      no `{mcol}` column (run with --ref-scores to attribute disagreements)")


def main(argv):
    args = parse_args(argv)
    pd.set_option("display.width", 200)
    arms = [load_arm(s, args.solo_axis) for s in args.arms]
    truth = None
    if args.truth:
        truth = pd.read_csv(args.truth, dtype={"read_id": str}).set_index("read_id")

    print("=== arms")
    for label, df, axes in arms:
        for ax in axes:
            called = (df[ax] != UNCLASSIFIED) & df[ax].notna()
            print(f"  {label:>10s} {ax}: {len(df):6d} rows  called {called.sum():6d} ({called.mean():6.1%})")

    if truth is not None:
        print(f"=== against truth ({len(truth)} labelled reads)")
        for label, df, axes in arms:
            score_against_truth(label, df, axes, truth)

    if len(arms) > 1:
        ref_label, ref, ref_axes = arms[0]
        print(f"=== against {ref_label}, read for read")
        for label, df, _ in arms[1:]:
            compare(ref_label, ref, label, df, ref_axes, args.near_tie)


if __name__ == "__main__":
    main(sys.argv[1:])
