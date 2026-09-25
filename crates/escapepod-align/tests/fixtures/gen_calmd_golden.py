#!/usr/bin/env python3
"""Measure what `samtools calmd` writes for MD/NM, ambiguity codes included.

`escpod align` must reproduce calmd's MD/NM byte for byte, because the
waveform charging bundle rebuilds each read's reference from MD
(`escapepod_classify::waveform::reference_from_md`). What calmd does at a
reference `N` is not documented anywhere we could rely on, so it is measured
here on a synthetic reference and committed as `calmd_golden.json`.

Pinned by tests/sam_fields.rs::md_nm_match_samtools_calmd. Usage (samtools
from aa-tRNA-seq-pipeline's pixi env; the committed golden is samtools 1.23.1):

    python gen_calmd_golden.py \
        --samtools ~/devel/rnabioco/aa-tRNA-seq-pipeline/.pixi/envs/default/bin/samtools \
        > calmd_golden.json
"""
import argparse, json, os, subprocess, sys, tempfile

REF_NAME = "t1"
# Upper-case N, an IUPAC R, a soft-masked (lower-case) stretch with an n,
# and an RNA-alphabet U.
REF = "ACGTANCGTAGCNATCGGACRTACGTacgtnacgtAAAAGGUU"

# (name, 1-based POS, CIGAR, SEQ). Indices in comments are 0-based on REF.
CASES = [
    ("n_under_read_base", 1, "10M", "ACGTAACGTA"),         # ref N @5 vs A
    ("n_under_read_n", 1, "10M", "ACGTANCGTA"),            # ref N @5 vs N
    ("read_n_over_ref_base", 1, "10M", "ACNTACCGTA"),      # read N @2 vs G
    ("mismatch_beside_n", 1, "10M", "ACGTAATGTA"),         # N @5, C>T @6
    ("mismatch_both_sides_of_n", 3, "8M", "GAACGTAG"),     # T>A @3... see golden
    ("n_inside_deletion", 1, "4M3D4M", "ACGTGTAG"),        # del A N C @4..6
    ("n_at_deletion_edge", 1, "5M2D4M", "ACGTAGTAG"),      # del N C @5..6
    ("insertion_beside_n", 1, "5M1I5M", "ACGTAGACGTA"),    # ins G before N
    ("two_ns_one_read", 4, "12M", "TACCGTAGCCAT"),         # N @5 and @12
    ("alignment_starts_on_n", 6, "2S6M", "GGACGTAG"),      # first ref base is N
    ("alignment_ends_on_n", 8, "6M3S", "GTAGCATTT"),       # last ref base is N @12
    ("iupac_r_vs_a", 19, "5M", "ACATA"),                   # R @20 vs A
    ("iupac_r_vs_c", 19, "5M", "ACCTA"),                   # R @20 vs C
    ("lowercase_ref_match", 27, "4M", "ACGT"),             # a c g t @26..29
    ("lowercase_ref_mismatch", 27, "4M", "ACTT"),          # g>T @28
    ("lowercase_n_vs_a", 27, "9M", "ACGTAACGT"),           # n @30 vs A
    ("all_match", 1, "5M", "ACGTA"),
    ("deletion_then_mismatch", 1, "3M2D3M", "ACGTCG"),
    ("lowercase_deletion", 27, "2M2D2M", "ACAA"),          # del g t @28..29, n @30
    ("ref_u_vs_t", 40, "4M", "GGTT"),                      # U @41..42 vs T
    ("ref_u_vs_c", 40, "4M", "GGCT"),                      # U @41 vs C
]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--samtools", default="samtools")
    a = ap.parse_args()
    ver = subprocess.run([a.samtools, "--version"], capture_output=True, text=True,
                         check=True).stdout.splitlines()[0]
    with tempfile.TemporaryDirectory() as d:
        fa = os.path.join(d, "ref.fa")
        with open(fa, "w") as f:
            f.write(f">{REF_NAME}\n{REF}\n")
        subprocess.run([a.samtools, "faidx", fa], check=True)
        sam = os.path.join(d, "in.sam")
        with open(sam, "w") as f:
            f.write(f"@HD\tVN:1.6\tSO:unsorted\n@SQ\tSN:{REF_NAME}\tLN:{len(REF)}\n")
            for name, pos, cigar, seq in CASES:
                f.write(f"{name}\t0\t{REF_NAME}\t{pos}\t60\t{cigar}\t*\t0\t0\t{seq}\t*\n")
        out = subprocess.run([a.samtools, "calmd", "-Q", sam, fa], capture_output=True,
                             text=True, check=True).stdout
    by_name = {}
    for line in out.splitlines():
        if line.startswith("@"):
            continue
        f = line.split("\t")
        tags = dict((t[:2], t[5:]) for t in f[11:])
        by_name[f[0]] = {"md": tags["MD"], "nm": int(tags["NM"])}
    records = []
    for name, pos, cigar, seq in CASES:
        records.append({"name": name, "pos": pos, "cigar": cigar, "seq": seq,
                        **by_name[name]})
    json.dump({"generator": "gen_calmd_golden.py", "samtools": ver,
               "reference": {"name": REF_NAME, "seq": REF}, "records": records},
              sys.stdout, indent=1)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
