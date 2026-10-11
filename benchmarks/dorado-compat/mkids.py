"""mkids.py INPUT.pod5 N OUT.txt : write first N read ids."""
import sys, pod5
r = pod5.Reader(sys.argv[1]); n = int(sys.argv[2])
with open(sys.argv[3], "w") as o:
    for i, rd in enumerate(r.reads()):
        if i >= n: break
        o.write(str(rd.read_id) + "\n")
