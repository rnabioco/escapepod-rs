"""pod5_ids.py OUT.txt POD5... : sorted read ids of the POD5s."""
import sys, pod5
ids = sorted(str(rd.read_id) for f in sys.argv[2:] for rd in pod5.Reader(f).reads())
open(sys.argv[1], "w").write("\n".join(ids) + "\n")
print(len(ids))
