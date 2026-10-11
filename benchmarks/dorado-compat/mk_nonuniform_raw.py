"""mk_nonuniform_raw.py IN.pod5 OUT.pod5 R1,R2,... : re-chunk the SIGNAL table's record batches to the given row counts
(sum must equal total rows; row order unchanged, so the reads table's signal-row indices stay valid) and re-emit the
POD5 container with patched footer offsets/lengths. Uses only pyarrow + bytes; escpod is not involved."""
import sys, struct, io, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pyarrow as pa, pyarrow.ipc as ipc
from pod5_layout import parse_footer
src, dst, spec = sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3].split(",")]
b = open(src, "rb").read()
fstart, flen, ents = parse_footer(b)
ents = sorted(ents, key=lambda e: e["off"])
marker = b[8:24]
sig = next(e for e in ents if e["ct"] == 1)
rdr = ipc.open_file(pa.BufferReader(b[sig["off"]:sig["off"] + sig["len"]]))
tbl = rdr.read_all(); assert sum(spec) == tbl.num_rows, (sum(spec), tbl.num_rows)
sink = io.BytesIO()
with ipc.new_file(sink, tbl.schema) as w:
    o = 0
    for n in spec:
        w.write_batch(pa.RecordBatch.from_struct_array(tbl.slice(o, n).combine_chunks().to_struct_array().combine_chunks()) if False
                      else pa.Table.from_batches(tbl.slice(o, n).to_batches()).combine_chunks().to_batches()[0]); o += n
new_sig = sink.getvalue()
out = bytearray(b[:24]); newents = []
pad = lambda n: (-n) % 8
for e in ents:
    data = new_sig if e is sig else b[e["off"]:e["off"] + e["len"]]
    newents.append((e, len(out), len(data)))
    out += data; out += b"\0" * pad(len(data)); out += marker
out += b[fstart - 8:fstart]            # FOOTER magic
nf_start = len(out); out += bytearray(b[fstart:fstart + flen])
for e, off, ln in newents:
    struct.pack_into("<q", out, nf_start + (e["off_pos"] - fstart), off)
    struct.pack_into("<q", out, nf_start + (e["len_pos"] - fstart), ln)
out += b[fstart + flen:]               # footer length, marker, signature
assert len(out) % 8 == 0
open(dst, "wb").write(out)
print("wrote", dst, len(out), "signal bytes", len(new_sig), "vs", sig["len"])
