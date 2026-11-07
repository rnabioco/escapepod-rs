"""Rust <-> Python bit-exact parity for ADC -> pA calibration (#501).

`crates/escapepod-pod5/tests/fixtures/calibration_pa.tsv` is the golden the
Rust test `test_calibration_golden` holds `escapepod_pod5::calibrate_pa` to.
Here every Python entry point that calibrates is held to the same bits, so
the binding cannot drift from the Rust definition without one side failing.
"""

import tempfile
import uuid
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

import escapepod

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
GOLDEN = (
    REPO_ROOT
    / "crates"
    / "escapepod-pod5"
    / "tests"
    / "fixtures"
    / "calibration_pa.tsv"
)


def _f32(bits: str) -> float:
    return float(np.array([int(bits, 16)], dtype=np.uint32).view(np.float32)[0])


def _load_golden():
    """{(offset_bits, scale_bits): (adc int16 array, pA uint32 bits array)}"""
    groups = defaultdict(lambda: ([], []))
    for line in GOLDEN.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        off, scale, adc, pa = line.split("\t")
        adcs, pas = groups[(off, scale)]
        adcs.append(int(adc))
        pas.append(int(pa, 16))
    return {
        k: (np.array(a, dtype=np.int16), np.array(p, dtype=np.uint32))
        for k, (a, p) in groups.items()
    }


GROUPS = _load_golden()


def test_golden_is_nontrivial():
    assert len(GROUPS) >= 2
    assert sum(len(a) for a, _ in GROUPS.values()) > 100


@pytest.mark.parametrize("key", sorted(GROUPS))
def test_calibrate_signal_array_bit_exact(key):
    adc, want = GROUPS[key]
    rd = escapepod.ReadData(
        str(uuid.UUID(int=1)),
        calibration_offset=_f32(key[0]),
        calibration_scale=_f32(key[1]),
    )
    # The binding stores f32: the value must survive the float round trip.
    assert np.float32(rd.calibration_offset).view(np.uint32) == int(key[0], 16)
    assert np.float32(rd.calibration_scale).view(np.uint32) == int(key[1], 16)
    got = rd.calibrate_signal_array(adc)
    assert got.dtype == np.float32
    np.testing.assert_array_equal(got.view(np.uint32), want)


def test_reader_signal_pa_bit_exact():
    """Write each calibration as a read, read it back through every
    Reader/DatasetReader pA path, and compare bits against the golden."""
    keys = sorted(GROUPS)
    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "calibration.pod5")
        ids = {}
        with escapepod.Writer(path) as w:
            w.add_run_info(escapepod.RunInfo("acq-calibration"))
            for i, key in enumerate(keys):
                rid = str(uuid.UUID(int=i + 1))
                ids[rid] = key
                adc, _ = GROUPS[key]
                rd = escapepod.ReadData(
                    rid,
                    read_number=i,
                    calibration_offset=_f32(key[0]),
                    calibration_scale=_f32(key[1]),
                    num_samples=len(adc),
                )
                w.add_read_data(rd, adc)

        with escapepod.Reader(path) as r:
            reads = r.reads()
            assert len(reads) == len(keys)
            for read in reads:
                adc, want = GROUPS[ids[read.read_id]]
                np.testing.assert_array_equal(r.get_signal(read), adc)
                np.testing.assert_array_equal(
                    r.get_signal_pa(read).view(np.uint32), want
                )
            for rid, sig in r.get_signals_pa(reads):
                _, want = GROUPS[ids[rid]]
                np.testing.assert_array_equal(
                    np.asarray(sig, dtype=np.float32).view(np.uint32), want
                )

        ds = escapepod.DatasetReader(d)
        for read in ds.reads():
            _, want = GROUPS[ids[read.read_id]]
            np.testing.assert_array_equal(ds.get_signal_pa(read).view(np.uint32), want)
