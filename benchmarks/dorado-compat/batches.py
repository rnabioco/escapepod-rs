"""Print per-file signal-table batch row counts (compressed as run-length) and read count.
usage: batches.py FILE...   (run with the pod5 python, e.g. `pixi run python`)"""
import sys, itertools, pod5
for f in sys.argv[1:]:
    r = pod5.Reader(f)
    sb = r._signal_handle.reader
    rows = [sb.get_batch(i).num_rows for i in range(sb.num_record_batches)]
    rle = ",".join(f"{k}x{len(list(g))}" for k, g in itertools.groupby(rows))
    # uniform stride = no batch smaller than the max except possibly the last
    m = max(rows) if rows else 0
    nonuni = any(x != m for x in rows[:-1])
    print(f"{f}\treads={r.num_reads}\tsignal_batches={len(rows)}\trows={rle}\tlast={rows[-1] if rows else 0}\tshort_last={rows[-1]<m if rows else False}\tNONUNIFORM_BEFORE_LAST={nonuni}")
