"""bam_parents.py BAM OUT_IDS.txt : write unique PARENT read ids (pi tag if present, else qname); print records/parents."""
import sys, pysam
recs = 0; ids = set()
with pysam.AlignmentFile(sys.argv[1], check_sq=False, require_index=False) as b:
    for a in b.fetch(until_eof=True):
        recs += 1
        ids.add(a.get_tag("pi") if a.has_tag("pi") else a.query_name)
open(sys.argv[2], "w").write("\n".join(sorted(ids)) + ("\n" if ids else ""))
print(f"{sys.argv[1]}\trecords={recs}\tparents={len(ids)}")
