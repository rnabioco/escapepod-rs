"""total_rows.py FILE : print the total number of rows in FILE's signal table (sum over record batches)."""
import sys, pod5
r = pod5.Reader(sys.argv[1]); r.num_reads  # touch the reads table first, as batches.py does
sb = r._signal_handle.reader
print(sum(sb.get_batch(i).num_rows for i in range(sb.num_record_batches)))
