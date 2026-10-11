"""pod5_layout.py FILE : dump container layout (footer entries via a minimal flatbuffer reader)."""
import sys, struct
def parse_footer(b):
    # file tail: ... FOOTER_MAGIC(8) <flatbuf> <len i64> <marker16> <sig8>
    flen = struct.unpack_from("<q", b, len(b) - 8 - 16 - 8)[0]
    fstart = len(b) - 8 - 16 - 8 - flen
    root = fstart + struct.unpack_from("<I", b, fstart)[0]
    def field(tab, idx):
        vt = tab - struct.unpack_from("<i", b, tab)[0]
        vsz = struct.unpack_from("<H", b, vt)[0]
        o = 4 + 2 * idx
        if o >= vsz: return None
        fo = struct.unpack_from("<H", b, vt + o)[0]
        return tab + fo if fo else None
    contents = field(root, 3)
    vec = contents + struct.unpack_from("<I", b, contents)[0]
    n = struct.unpack_from("<I", b, vec)[0]; ents = []
    for i in range(n):
        p = vec + 4 + 4 * i; tab = p + struct.unpack_from("<I", b, p)[0]
        off_p = field(tab, 0); len_p = field(tab, 1); ct_p = field(tab, 3)
        ents.append(dict(off=struct.unpack_from("<q", b, off_p)[0], len=struct.unpack_from("<q", b, len_p)[0],
                         ct=struct.unpack_from("<h", b, ct_p)[0] if ct_p else 0, off_pos=off_p, len_pos=len_p))
    return fstart, flen, ents
if __name__ == "__main__":
    b = open(sys.argv[1], "rb").read()
    print("head", b[:8], b[8:24].hex()); fs, fl, ents = parse_footer(b)
    print("footer at", fs, "len", fl, "magic", b[fs-8:fs], "size", len(b))
    for e in sorted(ents, key=lambda e: e["off"]): print(e, "tail-after:", b[e["off"]+e["len"]:e["off"]+e["len"]+24].hex())
