"""compare.py WORKDIR : for each bam id list in WORKDIR/ids/*.bam.ids and pod5 ids in WORKDIR/ids/*.pod5.ids,
report parents missing vs POD5 and vs controls' union BAM parents."""
import sys, glob, os
w = sys.argv[1]
rd = lambda p: set(open(p).read().split())
ctrl = [os.path.basename(p)[:-8] for p in glob.glob(f"{w}/ids/ctl_*.bam.ids")]
union = set().union(*[rd(f"{w}/ids/{c}.bam.ids") for c in ctrl]) if ctrl else set()
print(f"controls={sorted(ctrl)} union_parents_in_controls_bams={len(union)}")
print("name\tpod5_reads\tbam_parents\tmissing_vs_pod5\tmissing_vs_ctl_union_among_own_reads\textra_not_in_pod5")
for p in sorted(glob.glob(f"{w}/ids/*.pod5.ids")):
    n = os.path.basename(p)[:-9]
    pod = rd(p)
    bp = f"{w}/ids/{n}.bam.ids"
    if not os.path.exists(bp):
        print(f"{n}\t{len(pod)}\tNOBAM"); continue
    b = rd(bp)
    exp = pod & union
    print(f"{n}\t{len(pod)}\t{len(b)}\t{len(pod-b)}\t{len(exp-b)}\t{len(b-pod)}")
