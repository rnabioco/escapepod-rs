"""verify_pod5.py FILE... : read every read's signal with official pod5; check len==sample_count.
prints FILE<TAB>reads<TAB>bad<TAB>OK|FAIL ; ids to stdout not printed."""
import sys, pod5
for f in sys.argv[1:]:
    n = bad = 0; err = ""
    try:
        for rd in pod5.Reader(f).reads():
            n += 1
            if len(rd.signal) != rd.sample_count: bad += 1
    except Exception as e:
        err = repr(e)[:200]
    print(f"{f}\t{n}\t{bad}\t{'OK' if not bad and not err else 'FAIL '+err}")
