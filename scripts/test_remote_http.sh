#!/usr/bin/env bash
#
# End-to-end check for lazy remote POD5 reading (`remote` feature, issue #83).
#
# Serves a POD5 file over HTTP with Range support, then asserts that
# `escpod inspect`/`view` against the URL produce the same output as against
# the local path — and reports how many bytes the reader actually pulled.
#
# The point is the byte count: a correct lazy read fetches the file tail, the
# footer, and the reads table, and never touches the signal table. `python -m
# http.server` cannot be used here because it ignores the Range header.
#
# Usage: scripts/test_remote_http.sh [path/to/file.pod5]
#   Requires a build with the feature:
#     cargo build --features remote -p escapepod-cli

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POD5="${1:-$ROOT/data/drna/yeast_trna_reads.pod5}"
ESCPOD="${ESCPOD:-$ROOT/target/debug/escpod}"
PORT="${PORT:-8731}"

[ -f "$POD5" ] || { echo "no such POD5: $POD5" >&2; exit 1; }
[ -x "$ESCPOD" ] || { echo "build first: cargo build --features remote -p escapepod-cli" >&2; exit 1; }

WORK="$(mktemp -d)"
trap 'kill %1 2>/dev/null; rm -rf "$WORK"' EXIT

cat > "$WORK/server.py" <<'PYEOF'
"""Static file server with HTTP Range support; logs each served range."""
import os, re, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT, PORT = sys.argv[1], int(sys.argv[2])


class H(BaseHTTPRequestHandler):
    def _f(self):
        p = os.path.join(ROOT, self.path.split("?")[0].lstrip("/"))
        if not os.path.isfile(p):
            self.send_error(404)
            return None
        return p

    def _common(self, p, size):
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Last-Modified", self.date_time_string(os.path.getmtime(p)))
        self.send_header("ETag", '"%d-%d"' % (size, int(os.path.getmtime(p))))

    def do_HEAD(self):
        p = self._f()
        if not p:
            return
        size = os.path.getsize(p)
        self.send_response(200)
        self.send_header("Content-Length", str(size))
        self._common(p, size)
        self.end_headers()

    def do_GET(self):
        p = self._f()
        if not p:
            return
        size = os.path.getsize(p)
        m = re.match(r"bytes=(\d+)-(\d*)", self.headers.get("Range") or "")
        if m:
            start = int(m.group(1))
            end = min(int(m.group(2)) if m.group(2) else size - 1, size - 1)
            length = end - start + 1
            self.send_response(206)
            self.send_header("Content-Range", "bytes %d-%d/%d" % (start, end, size))
        else:
            start, length = 0, size
            self.send_response(200)
        sys.stderr.write("%d\n" % length)
        sys.stderr.flush()
        self.send_header("Content-Length", str(length))
        self._common(p, size)
        self.end_headers()
        with open(p, "rb") as f:
            f.seek(start)
            self.wfile.write(f.read(length))

    def log_message(self, *a):
        pass


ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
PYEOF

SIZE=$(stat -c%s "$POD5")
URL="http://127.0.0.1:$PORT/$(basename "$POD5")"
python "$WORK/server.py" "$(dirname "$POD5")" "$PORT" 2>"$WORK/bytes.log" &
for _ in $(seq 1 40); do
    curl -sf -I "$URL" >/dev/null 2>&1 && break
    sleep 0.25
done

echo "file: $POD5 ($SIZE bytes)"
FAIL=0
# `inspect summary` prints the input location, so its output legitimately
# differs between a path and a URL; compare it with that line dropped.
for CMD in "inspect summary" "inspect reads" "view"; do
    : > "$WORK/bytes.log"
    # shellcheck disable=SC2086
    "$ESCPOD" $CMD "$POD5" 2>/dev/null | grep -v '^File:' > "$WORK/local.txt"
    # shellcheck disable=SC2086
    "$ESCPOD" $CMD "$URL" 2>"$WORK/err.txt" | grep -v '^File:' > "$WORK/remote.txt"
    # escpod's status, not grep's — `$?` on a pipeline reports the last stage,
    # which would silently mask a failed remote run.
    RC=${PIPESTATUS[0]}

    BYTES=$(awk '{s+=$1} END {print s+0}' "$WORK/bytes.log")
    REQS=$(wc -l < "$WORK/bytes.log")
    PCT=$(awk -v b="$BYTES" -v s="$SIZE" 'BEGIN{printf "%.1f", 100*b/s}')

    if [ $RC -ne 0 ]; then
        echo "FAIL  escpod $CMD: remote run failed"
        sed 's/^/      /' "$WORK/err.txt" | head -3
        FAIL=1
    elif cmp -s "$WORK/local.txt" "$WORK/remote.txt"; then
        echo "ok    escpod $CMD: identical output; $BYTES bytes in $REQS requests (${PCT}% of file)"
    else
        echo "FAIL  escpod $CMD: local and remote output differ"
        diff "$WORK/local.txt" "$WORK/remote.txt" | head -10
        FAIL=1
    fi
done

exit $FAIL
