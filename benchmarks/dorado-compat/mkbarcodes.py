"""mkbarcodes.py OUT.csv POD5... : read_id,barcode CSV, barcode alternates BC01/BC02/BC03 by read order."""
import sys, pod5
with open(sys.argv[1], "w") as o:
    o.write("read_id,barcode\n")
    i = 0
    for f in sys.argv[2:]:
        for rd in pod5.Reader(f).reads():
            o.write(f"{rd.read_id},BC0{i%3+1}\n"); i += 1
