#!/usr/bin/env python3
"""Generate align_golden.json: parasail's scores and coordinates, the oracle
for escapepod-align's scalar Gotoh (tests/parasail_golden.rs).

parasail's `sw_trace` (local) and `sg_trace` (semi-global: leading and
trailing gaps of both sequences free) are the reference implementations. Its
open/extend convention is escapepod-align's with the signs flipped: a gap of
length k costs `open + (k - 1) * extend`, so `Scoring { gap_open: -10,
gap_extend: -1 }` is parasail's `open=10, extend=1`.

The substitution matrix is ACGTN with N as a wildcard (N pairs with anything
as a match), which is escapepod-align's rule for every non-ACGT letter.

Recorded per (pair, mode, scheme): score, and 0-based half-open
query/reference coordinates of the aligned region. parasail reports the end
cell; the start is read off its CIGAR, whose leading I/D runs (emitted when a
local path meets the matrix border, and for sg's free leading gaps) are
offsets, not alignment.

Usage, from aa-tRNA-seq-pipeline (parasail-python 1.3.x in its pixi env):

    cd ~/devel/rnabioco/aa-tRNA-seq-pipeline
    pixi run python <this> --samtools .pixi/envs/default/bin/samtools \
        > crates/escapepod-align/tests/fixtures/align_golden.json
"""
import argparse, json, os, random, re, subprocess, sys

import parasail

SCHEMES = {
    # gpu-tRNA-mapper's default, and escpod align's.
    "default": (2, -1, -10, -1),
    # bwa mem -x ont2d's scores in this convention (bwa's O=1,E=1 is a
    # 1-base gap of -2): gap-lenient.
    "gap_lenient": (1, -1, -2, -1),
}
MODES = {"local": parasail.sw_trace, "semiglobal": parasail.sg_trace}
HERE = os.path.dirname(os.path.abspath(__file__))
CLASSIFY_FIXTURES = os.path.join(HERE, "../../../escapepod-classify/tests/fixtures")


def matrix(m, x):
    mat = parasail.matrix_create("ACGTN", m, x)
    for k in range(5):
        mat.set_value(4, k, m)
        mat.set_value(k, 4, m)
    return mat


def mutate(rng, s):
    out = []
    for b in s:
        if rng.random() < 0.04:  # deletion
            continue
        out.append(rng.choice("ACGT") if rng.random() < 0.05 else b)
        if rng.random() < 0.03:  # insertion
            out.append(rng.choice("ACGT"))
    return "".join(out)


def rand_seq(rng, n, n_rate=0.0):
    return "".join("N" if rng.random() < n_rate else rng.choice("ACGT") for _ in range(n))


def read_fasta(path):
    refs, name = {}, None
    for line in open(path):
        line = line.strip()
        if line.startswith(">"):
            name = line[1:].split()[0]
            refs[name] = []
        elif line:
            refs[name].append(line)
    return {k: "".join(v).upper() for k, v in refs.items()}


def coords(res):
    cig = res.cigar
    s = cig.decode.decode()
    ops = re.findall(r"(\d+)([MIDNSHP=X])", s)
    qs, rs = cig.beg_query, cig.beg_ref
    for n, op in ops:
        if op == "I":
            qs += int(n)
        elif op == "D":
            rs += int(n)
        else:
            break
    return qs, res.end_query + 1, rs, res.end_ref + 1


def results(query, ref):
    out = []
    for scheme, (m, x, o, e) in SCHEMES.items():
        mat = matrix(m, x)
        for mode, fn in MODES.items():
            res = fn(query, ref, -o, -e, mat)
            qs, qe, rs, re_ = coords(res)
            out.append({"mode": mode, "scheme": scheme, "score": res.score,
                        "query_start": qs, "query_end": qe,
                        "ref_start": rs, "ref_end": re_})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--samtools", default="samtools")
    ap.add_argument("--seed", type=int, default=395)
    ap.add_argument("--n-random", type=int, default=40)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    pairs = []
    for k in range(a.n_random):
        ref = rand_seq(rng, rng.randint(20, 300), n_rate=0.01)
        if k % 8 == 7:
            # Unrelated read: exercises the zero floor / negative sg cells.
            query = rand_seq(rng, rng.randint(20, 300))
        else:
            lo = rng.randint(0, len(ref) // 3)
            hi = len(ref) - rng.randint(0, len(ref) // 3)
            query = (rand_seq(rng, rng.randint(0, 30)) + mutate(rng, ref[lo:hi])
                     + rand_seq(rng, rng.randint(0, 30)))
            if len(query) < 20:
                query += rand_seq(rng, 20 - len(query))
            query = query[:300]
        pairs.append({"source": "random", "name": f"random_{k}", "query": query,
                      "reference": ref, "results": results(query, ref)})

    # Real reads from escapepod-classify's fixture BAM, against the reference
    # bwa assigned them, from both the plain and the N-carrying FASTA.
    bam = os.path.join(CLASSIFY_FIXTURES, "trna_mappings_padded.bam")
    sam = subprocess.run([a.samtools, "view", bam], capture_output=True, text=True,
                         check=True).stdout.splitlines()
    plain = read_fasta(os.path.join(CLASSIFY_FIXTURES, "trna_reference.fa"))
    amb = read_fasta(os.path.join(CLASSIFY_FIXTURES, "trna_reference_ambiguous.fa"))
    seen = set()
    for line in sam:
        f = line.split("\t")
        seq = f[9]
        if seq in seen:
            continue
        seen.add(seq)
        k = len(seen)
        refs = plain if k % 2 else amb
        pairs.append({"source": "fixture" if k % 2 else "fixture_ambiguous",
                      "name": f[0], "query": seq, "reference": refs[f[2]],
                      "reference_name": f[2], "results": results(seq, refs[f[2]])})
        if k >= 8:
            break

    json.dump({"generator": "gen_align_golden.py", "parasail": parasail.__version__,
               "seed": a.seed,
               "schemes": {k: list(v) for k, v in SCHEMES.items()},
               "pairs": pairs}, sys.stdout, separators=(",", ":"))
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
