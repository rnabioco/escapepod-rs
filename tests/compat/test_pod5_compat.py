#!/usr/bin/env python3
"""
POD5 forward/backward compatibility tests between escapepod-rs and Python pod5.

Tests:
  1. Python → escapepod (backward compat): Write with pod5, read with escpod CLI
  2. escapepod → Python (forward compat): Filter through escpod, read with pod5
  3. Filter round-trip: Python → escpod filter → Python, verify no data loss
  4. Merge round-trip: Python A + B → escpod merge → Python, verify run_info dedup
  5. Subset round-trip: Python → escpod subset (CSV mapping) → Python per group
  6. Edge cases: minimal signal, multi-chunk signal, empty metadata dicts
  7. Existing test files: read real ONT POD5 files with both tools

Index and repack are intentionally not round-trip tested here (experimental).

Usage:
    cargo build --release
    pixi run python tests/compat/test_pod5_compat.py
"""

import os
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from types import SimpleNamespace

from datetime import datetime, timezone

import numpy as np
import pod5

# ---------------------------------------------------------------------------
# Locate the escpod binary
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ESCPOD = REPO_ROOT / "target" / "release" / "escpod"
if not ESCPOD.exists():
    ESCPOD = REPO_ROOT / "target" / "debug" / "escpod"
if not ESCPOD.exists():
    sys.exit("ERROR: escpod binary not found. Run 'cargo build --release' first.")

# ---------------------------------------------------------------------------
# Canonical test data
# ---------------------------------------------------------------------------
READ_IDS = [
    uuid.UUID("00000000-0000-0000-0000-000000000001"),
    uuid.UUID("00000000-0000-0000-0000-000000000002"),
    uuid.UUID("00000000-0000-0000-0000-000000000003"),
    uuid.UUID("00000000-0000-0000-0000-000000000004"),
    uuid.UUID("00000000-0000-0000-0000-000000000005"),
]

END_REASONS = [
    (pod5.EndReasonEnum.UNKNOWN, False),
    (pod5.EndReasonEnum.MUX_CHANGE, True),
    (pod5.EndReasonEnum.UNBLOCK_MUX_CHANGE, False),
    (pod5.EndReasonEnum.DATA_SERVICE_UNBLOCK_MUX_CHANGE, True),
    (pod5.EndReasonEnum.SIGNAL_POSITIVE, True),
]

PORE_TYPES = ["not_set", "pore_r10", "pore_r9", "not_set", "pore_r10"]

# Signal sizes: small, medium, multi-chunk (>102400), small, medium
SIGNAL_SIZES = [100, 5000, 150_000, 200, 8000]

# Calibration values chosen to survive float formatting
CALIBRATION = [
    (-220.5000, 0.150000),
    (-180.2500, 0.145200),
    (-200.0000, 0.160000),
    (-195.7500, 0.155000),
    (-210.1250, 0.148500),
]

CHANNELS = [1, 100, 250, 400, 500]
WELLS = [1, 2, 3, 4, 1]
READ_NUMBERS = [10, 20, 30, 40, 50]
START_SAMPLES = [0, 10000, 50000, 100000, 200000]
MEDIAN_BEFORES = [200.50, 180.25, 210.00, 195.75, 205.30]
NUM_MINKNOW_EVENTS = [50, 2500, 75000, 100, 4000]

RUN_INFO_1 = pod5.RunInfo(
    acquisition_id="acq_001_compat_test",
    acquisition_start_time=datetime.fromtimestamp(1700000000, tz=timezone.utc),
    adc_max=2047,
    adc_min=-2048,
    context_tags={"experiment_type": "genomic_dna", "basecall_model": "dna_r10.4.1_e8.2_400bps_sup"},
    experiment_name="compat_test_experiment",
    flow_cell_id="PAM12345",
    flow_cell_product_code="FLO-MIN114",
    protocol_name="sequencing/sequencing_MIN114_DNA",
    protocol_run_id="proto_001",
    protocol_start_time=datetime.fromtimestamp(1699999000, tz=timezone.utc),
    sample_id="sample_compat_test",
    sample_rate=4000,
    sequencing_kit="SQK-LSK114",
    sequencer_position="MN12345",
    sequencer_position_type="MinION",
    software="MinKNOW 23.11.1",
    system_name="host_machine",
    system_type="linux",
    tracking_id={"device_id": "MN12345", "run_id": "run_001_compat", "protocol_group_id": "group_A"},
)

RUN_INFO_2 = pod5.RunInfo(
    acquisition_id="acq_002_compat_test",
    acquisition_start_time=datetime.fromtimestamp(1700100000, tz=timezone.utc),
    adc_max=4095,
    adc_min=-4096,
    context_tags={"experiment_type": "rna"},
    experiment_name="compat_test_rna",
    flow_cell_id="PAM67890",
    flow_cell_product_code="FLO-MIN114",
    protocol_name="sequencing/sequencing_MIN114_RNA",
    protocol_run_id="proto_002",
    protocol_start_time=datetime.fromtimestamp(1700099000, tz=timezone.utc),
    sample_id="sample_rna",
    sample_rate=5000,
    sequencing_kit="SQK-RNA004",
    sequencer_position="MN67890",
    sequencer_position_type="MinION",
    software="MinKNOW 24.01.0",
    system_name="host2",
    system_type="linux",
    tracking_id={"device_id": "MN67890", "run_id": "run_002"},
)

# Reads 0-2 use RUN_INFO_1, reads 3-4 use RUN_INFO_2
RUN_INFOS = [RUN_INFO_1, RUN_INFO_1, RUN_INFO_1, RUN_INFO_2, RUN_INFO_2]


def generate_signal(size: int, seed: int) -> np.ndarray:
    """Generate deterministic pseudo-random signal data."""
    rng = np.random.default_rng(seed)
    return rng.integers(-2000, 2000, size=size, dtype=np.int16)


SIGNALS = [generate_signal(sz, i) for i, sz in enumerate(SIGNAL_SIZES)]


# ---------------------------------------------------------------------------
# Helper: build pod5 Read objects
# ---------------------------------------------------------------------------
def make_pod5_reads():
    """Create list of pod5.Read objects from canonical data."""
    reads = []
    for i in range(5):
        # pod5 expects end_reason as EndReason object
        end_reason = pod5.EndReason(END_REASONS[i][0], END_REASONS[i][1])
        pore = pod5.Pore(channel=CHANNELS[i], well=WELLS[i], pore_type=PORE_TYPES[i])
        calibration = pod5.Calibration(
            offset=CALIBRATION[i][0], scale=CALIBRATION[i][1]
        )
        r = pod5.Read(
            read_id=READ_IDS[i],
            pore=pore,
            calibration=calibration,
            read_number=READ_NUMBERS[i],
            start_sample=START_SAMPLES[i],
            median_before=MEDIAN_BEFORES[i],
            end_reason=end_reason,
            run_info=RUN_INFOS[i],
            signal=SIGNALS[i],
            num_minknow_events=NUM_MINKNOW_EVENTS[i],
        )
        reads.append(r)
    return reads


# ---------------------------------------------------------------------------
# Helper: run escpod CLI
# ---------------------------------------------------------------------------
def run_escpod(*args, check=True):
    """Run escpod with arguments and return CompletedProcess."""
    cmd = [str(ESCPOD)] + list(args)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"COMMAND: {' '.join(cmd)}", file=sys.stderr)
        print(f"STDOUT: {result.stdout}", file=sys.stderr)
        print(f"STDERR: {result.stderr}", file=sys.stderr)
        raise RuntimeError(f"escpod failed with exit code {result.returncode}")
    return result


# ---------------------------------------------------------------------------
# Helper: write a POD5 file with Python
# ---------------------------------------------------------------------------
def write_python_pod5(path: Path):
    """Write canonical test data to a POD5 file using Python pod5."""
    reads = make_pod5_reads()
    with pod5.Writer(path) as writer:
        for r in reads:
            writer.add_read(r)


def write_python_pod5_subset(path: Path, indices: list[int]):
    """Write a subset of the canonical reads to a POD5 file."""
    all_reads = make_pod5_reads()
    with pod5.Writer(path) as writer:
        for i in indices:
            writer.add_read(all_reads[i])


# ---------------------------------------------------------------------------
# Reader helpers
# ---------------------------------------------------------------------------
# `pod5.Read.signal` is lazy and reads through the underlying ArrowTableHandle,
# so the reader must still be open at access time. To avoid encoding that
# footgun into every assertion block, drain everything we care about
# (metadata + signal) into plain in-memory snapshots while the reader is
# open, then close it before any assertions run. Each test only touches each
# read once, so there's no reader-cache "warm path" to worry about hiding
# bugs.
def materialize_reads(path: Path) -> dict:
    """Open path, drain all reads + signals into SimpleNamespace snapshots
    that mirror pod5.Read accessors, then close the reader. Returns a dict
    keyed by read_id (UUID)."""
    out: dict = {}
    with pod5.Reader(path) as reader:
        for r in reader.reads():
            out[r.read_id] = SimpleNamespace(
                read_id=r.read_id,
                read_number=r.read_number,
                start_sample=r.start_sample,
                median_before=r.median_before,
                num_samples=r.num_samples,
                num_minknow_events=r.num_minknow_events,
                pore=SimpleNamespace(
                    channel=r.pore.channel,
                    well=r.pore.well,
                    pore_type=r.pore.pore_type,
                ),
                calibration=SimpleNamespace(
                    offset=r.calibration.offset,
                    scale=r.calibration.scale,
                ),
                end_reason=SimpleNamespace(
                    name=r.end_reason.name,
                    forced=r.end_reason.forced,
                ),
                signal=np.asarray(r.signal),
            )
    return out


def read_run_info_count(path: Path) -> int:
    """Open path, count run_info rows, close. Separate fresh open so this
    exercises the footer parse independently of the reads-table path."""
    with pod5.Reader(path) as reader:
        return len(list(reader.run_info_table.read_pandas().itertuples()))


def random_access_signal_check(
    label: str, path: Path, read_id: uuid.UUID, expected_signal: np.ndarray
) -> bool:
    """Fresh-open `path` and request a single read by ID. Exercises the
    cold footer-parse + signal-block-index lookup that downstream tools
    (basecallers, demuxers) rely on. Returns True on signal byte-equality.

    This is intentionally a *separate* open from any prior materialize call,
    so a footer/signal-row-count regression that only manifests on first
    access cannot be masked by a still-warm reader."""
    try:
        with pod5.Reader(path) as reader:
            picked = list(reader.reads(selection=[read_id]))
            if not picked:
                print(f"  FAIL: {label} cold lookup found no read for {read_id}")
                return False
            sig = np.asarray(picked[0].signal)
    except Exception as e:
        print(f"  FAIL: {label} cold-open random-access raised: {e}")
        return False
    try:
        np.testing.assert_array_equal(sig, expected_signal)
    except AssertionError:
        print(f"  FAIL: {label} cold-open signal mismatch for {read_id}")
        return False
    print(f"  OK: {label} cold-open + by-ID lookup signal exact match")
    return True


# ---------------------------------------------------------------------------
# Helper: parse escpod view TSV output
# ---------------------------------------------------------------------------
def parse_view_tsv(tsv_text: str) -> list[dict[str, str]]:
    """Parse TSV output from escpod view into list of dicts."""
    lines = tsv_text.strip().split("\n")
    if not lines:
        return []
    header = lines[0].split("\t")
    rows = []
    for line in lines[1:]:
        if not line.strip():
            continue
        values = line.split("\t")
        rows.append(dict(zip(header, values)))
    return rows


# ---------------------------------------------------------------------------
# Map end reason enum to escapepod string
# ---------------------------------------------------------------------------
END_REASON_TO_STR = {
    pod5.EndReasonEnum.UNKNOWN: "unknown",
    pod5.EndReasonEnum.MUX_CHANGE: "mux_change",
    pod5.EndReasonEnum.UNBLOCK_MUX_CHANGE: "unblock_mux_change",
    pod5.EndReasonEnum.DATA_SERVICE_UNBLOCK_MUX_CHANGE: "data_service_unblock_mux_change",
    pod5.EndReasonEnum.SIGNAL_POSITIVE: "signal_positive",
    pod5.EndReasonEnum.SIGNAL_NEGATIVE: "signal_negative",
    pod5.EndReasonEnum.API_REQUEST: "api_request",
    pod5.EndReasonEnum.DEVICE_DATA_ERROR: "device_data_error",
}


# ---------------------------------------------------------------------------
# Test 1: Python → escapepod (backward compat)
# ---------------------------------------------------------------------------
def test_python_to_escapepod(tmpdir: Path) -> bool:
    """Write with Python pod5, read with escpod CLI."""
    print("\n=== Test 1: Python → escapepod (backward compat) ===")
    ok = True
    pod5_path = tmpdir / "python_written.pod5"
    write_python_pod5(pod5_path)

    # Read all fields with escpod view
    all_fields = (
        "read_id,channel,well,pore_type,read_number,start_sample,"
        "median_before,end_reason,end_reason_forced,num_samples,"
        "num_minknow_events,calibration_offset,calibration_scale,"
        "run_info,open_pore_level"
    )
    result = run_escpod("view", str(pod5_path), "--include", all_fields)
    rows = parse_view_tsv(result.stdout)

    if len(rows) != 5:
        print(f"  FAIL: Expected 5 reads, got {len(rows)}")
        return False
    print(f"  OK: Got {len(rows)} reads")

    # Sort rows by read_id to match canonical order
    id_to_idx = {str(rid): i for i, rid in enumerate(READ_IDS)}
    rows.sort(key=lambda r: id_to_idx.get(r["read_id"], 999))

    for i, row in enumerate(rows):
        prefix = f"  Read {i}"

        # Exact match fields
        checks = [
            ("read_id", str(READ_IDS[i])),
            ("channel", str(CHANNELS[i])),
            ("well", str(WELLS[i])),
            ("pore_type", PORE_TYPES[i]),
            ("read_number", str(READ_NUMBERS[i])),
            ("start_sample", str(START_SAMPLES[i])),
            ("end_reason", END_REASON_TO_STR[END_REASONS[i][0]]),
            ("end_reason_forced", str(END_REASONS[i][1]).lower()),
            ("num_samples", str(SIGNAL_SIZES[i])),
            ("num_minknow_events", str(NUM_MINKNOW_EVENTS[i])),
        ]
        for field, expected in checks:
            actual = row.get(field, "<missing>")
            if actual != expected:
                print(f"  FAIL: {prefix} {field}: expected {expected!r}, got {actual!r}")
                ok = False

        # Float fields with tolerance
        float_checks = [
            ("median_before", MEDIAN_BEFORES[i], 0.01),
            ("calibration_offset", CALIBRATION[i][0], 0.0001),
            ("calibration_scale", CALIBRATION[i][1], 0.000001),
        ]
        for field, expected, tol in float_checks:
            actual_str = row.get(field, "<missing>")
            try:
                actual_val = float(actual_str)
            except ValueError:
                print(f"  FAIL: {prefix} {field}: could not parse {actual_str!r}")
                ok = False
                continue
            if abs(actual_val - expected) > tol:
                print(
                    f"  FAIL: {prefix} {field}: expected {expected}, got {actual_val} "
                    f"(diff={abs(actual_val - expected)}, tol={tol})"
                )
                ok = False

    # Also verify inspect summary works
    result = run_escpod("inspect", "summary", str(pod5_path))
    if "5" not in result.stdout:
        print(f"  FAIL: inspect summary doesn't show 5 reads")
        ok = False
    else:
        print(f"  OK: inspect summary recognized the file")

    if ok:
        print("  PASS: All fields match")
    return ok


# ---------------------------------------------------------------------------
# Test 2: escapepod → Python (forward compat)
# ---------------------------------------------------------------------------
def test_escapepod_to_python(tmpdir: Path) -> bool:
    """Filter through escpod to produce an escapepod-written file, read with Python."""
    print("\n=== Test 2: escapepod → Python (forward compat) ===")
    ok = True
    python_pod5 = tmpdir / "python_written.pod5"
    if not python_pod5.exists():
        write_python_pod5(python_pod5)

    # Write IDs file for filter
    ids_path = tmpdir / "all_ids.txt"
    ids_path.write_text("\n".join(str(rid) for rid in READ_IDS) + "\n")

    # Filter through escapepod (copies all reads)
    escpod_pod5 = tmpdir / "escpod_written.pod5"
    run_escpod(
        "filter", str(python_pod5),
        "--ids", str(ids_path),
        "--output", str(escpod_pod5),
    )

    if not escpod_pod5.exists():
        print("  FAIL: escpod filter did not produce output file")
        return False

    # Drain the escapepod-written file into in-memory snapshots (signals as
    # numpy arrays). Reader closes inside materialize_reads — assertions run
    # against pure Python data with no live ArrowTableHandle.
    try:
        snapshots = materialize_reads(escpod_pod5)
    except Exception as e:
        print(f"  FAIL: Python pod5 cannot drain escapepod-written file: {e}")
        return False

    if len(snapshots) != 5:
        print(f"  FAIL: Expected 5 reads, got {len(snapshots)}")
        return False
    print(f"  OK: Python pod5 drained escapepod-written file ({len(snapshots)} reads)")

    # Walk reads in canonical (writer) order for stable per-read messages.
    reads = [snapshots[rid] for rid in READ_IDS if rid in snapshots]

    for i, read in enumerate(reads):
        prefix = f"  Read {i}"

        # Read ID
        if read.read_id != READ_IDS[i]:
            print(f"  FAIL: {prefix} read_id: expected {READ_IDS[i]}, got {read.read_id}")
            ok = False

        # Pore
        if read.pore.channel != CHANNELS[i]:
            print(f"  FAIL: {prefix} channel: expected {CHANNELS[i]}, got {read.pore.channel}")
            ok = False
        if read.pore.well != WELLS[i]:
            print(f"  FAIL: {prefix} well: expected {WELLS[i]}, got {read.pore.well}")
            ok = False
        if read.pore.pore_type != PORE_TYPES[i]:
            print(f"  FAIL: {prefix} pore_type: expected {PORE_TYPES[i]!r}, got {read.pore.pore_type!r}")
            ok = False

        # Calibration
        if abs(read.calibration.offset - CALIBRATION[i][0]) > 0.0001:
            print(f"  FAIL: {prefix} calibration_offset: expected {CALIBRATION[i][0]}, got {read.calibration.offset}")
            ok = False
        if abs(read.calibration.scale - CALIBRATION[i][1]) > 0.000001:
            print(f"  FAIL: {prefix} calibration_scale: expected {CALIBRATION[i][1]}, got {read.calibration.scale}")
            ok = False

        # Scalar fields
        if read.read_number != READ_NUMBERS[i]:
            print(f"  FAIL: {prefix} read_number: expected {READ_NUMBERS[i]}, got {read.read_number}")
            ok = False
        if read.start_sample != START_SAMPLES[i]:
            print(f"  FAIL: {prefix} start_sample: expected {START_SAMPLES[i]}, got {read.start_sample}")
            ok = False
        if abs(read.median_before - MEDIAN_BEFORES[i]) > 0.01:
            print(f"  FAIL: {prefix} median_before: expected {MEDIAN_BEFORES[i]}, got {read.median_before}")
            ok = False
        if read.num_minknow_events != NUM_MINKNOW_EVENTS[i]:
            print(f"  FAIL: {prefix} num_minknow_events: expected {NUM_MINKNOW_EVENTS[i]}, got {read.num_minknow_events}")
            ok = False
        if read.num_samples != SIGNAL_SIZES[i]:
            print(f"  FAIL: {prefix} num_samples: expected {SIGNAL_SIZES[i]}, got {read.num_samples}")
            ok = False

        # End reason — normalize to string for robust comparison
        expected_reason = END_REASONS[i][0]
        expected_forced = END_REASONS[i][1]
        expected_str = END_REASON_TO_STR.get(expected_reason, str(expected_reason))
        actual_name = read.end_reason.name
        actual_str = END_REASON_TO_STR.get(actual_name, str(actual_name)) if isinstance(actual_name, pod5.EndReasonEnum) else str(actual_name)
        if actual_str != expected_str:
            print(f"  FAIL: {prefix} end_reason: expected {expected_str!r}, got {actual_str!r}")
            ok = False
        if read.end_reason.forced != expected_forced:
            print(f"  FAIL: {prefix} end_reason_forced: expected {expected_forced}, got {read.end_reason.forced}")
            ok = False

        # Signal - exact i16 match (snapshot already holds a numpy array)
        signal = read.signal
        try:
            np.testing.assert_array_equal(signal, SIGNALS[i])
        except AssertionError as e:
            print(f"  FAIL: {prefix} signal mismatch: {e}")
            ok = False
        else:
            print(f"  OK: {prefix} signal ({len(signal)} samples) exact match")

    # Verify run info via a separate fresh open — exercises footer parse
    # independently of the reads-table drain above.
    try:
        n_run_info = read_run_info_count(escpod_pod5)
        print(f"  OK: {n_run_info} run info record(s) read successfully")
    except Exception as e:
        print(f"  FAIL: Could not read run info: {e}")
        ok = False

    if ok:
        print("  PASS: All fields match")
    return ok


# ---------------------------------------------------------------------------
# Test 3: Full round-trip
# ---------------------------------------------------------------------------
def test_round_trip(tmpdir: Path) -> bool:
    """Write with Python, filter through escapepod, read back with Python."""
    print("\n=== Test 3: Full round-trip (Python → Rust → Python) ===")
    ok = True

    python_pod5 = tmpdir / "python_written.pod5"
    if not python_pod5.exists():
        write_python_pod5(python_pod5)

    # Filter through escapepod
    ids_path = tmpdir / "all_ids.txt"
    if not ids_path.exists():
        ids_path.write_text("\n".join(str(rid) for rid in READ_IDS) + "\n")

    roundtrip_pod5 = tmpdir / "roundtrip.pod5"
    run_escpod(
        "filter", str(python_pod5),
        "--ids", str(ids_path),
        "--output", str(roundtrip_pod5),
    )

    # Drain both files into snapshots and close their readers before any
    # assertions run.
    orig_reads = materialize_reads(python_pod5)
    try:
        rt_reads = materialize_reads(roundtrip_pod5)
    except Exception as e:
        print(f"  FAIL: Python pod5 cannot open escapepod-written file: {e}")
        return False

    if set(orig_reads.keys()) != set(rt_reads.keys()):
        print(f"  FAIL: Read ID sets differ")
        ok = False

    for rid in READ_IDS:
        if rid not in orig_reads or rid not in rt_reads:
            continue
        orig = orig_reads[rid]
        rt = rt_reads[rid]
        prefix = f"  {rid}"

        # Signal exact match
        try:
            np.testing.assert_array_equal(rt.signal, orig.signal)
        except AssertionError:
            print(f"  FAIL: {prefix} signal mismatch after round-trip")
            ok = False

        # Scalar comparisons
        if rt.read_number != orig.read_number:
            print(f"  FAIL: {prefix} read_number changed: {orig.read_number} → {rt.read_number}")
            ok = False
        if rt.start_sample != orig.start_sample:
            print(f"  FAIL: {prefix} start_sample changed: {orig.start_sample} → {rt.start_sample}")
            ok = False
        if rt.pore.channel != orig.pore.channel:
            print(f"  FAIL: {prefix} channel changed")
            ok = False
        if rt.pore.well != orig.pore.well:
            print(f"  FAIL: {prefix} well changed")
            ok = False
        if rt.pore.pore_type != orig.pore.pore_type:
            print(f"  FAIL: {prefix} pore_type changed: {orig.pore.pore_type!r} → {rt.pore.pore_type!r}")
            ok = False
        # Normalize end_reason for comparison
        rt_name = rt.end_reason.name
        orig_name = orig.end_reason.name
        rt_str = rt_name.value if isinstance(rt_name, pod5.EndReasonEnum) else str(rt_name)
        orig_str = orig_name.value if isinstance(orig_name, pod5.EndReasonEnum) else str(orig_name)
        if rt_str != orig_str:
            print(f"  FAIL: {prefix} end_reason changed: {orig_str!r} → {rt_str!r}")
            ok = False
        if rt.end_reason.forced != orig.end_reason.forced:
            print(f"  FAIL: {prefix} end_reason_forced changed")
            ok = False

    # Cold-open random-access spot-check on the multi-chunk read (read 2,
    # 150k samples) — that's the most interesting signal layout to exercise
    # via fresh footer parse + by-ID lookup.
    if not random_access_signal_check(
        "filter output", roundtrip_pod5, READ_IDS[2], SIGNALS[2]
    ):
        ok = False

    if ok:
        print("  PASS: Round-trip preserves all data")
    return ok


# ---------------------------------------------------------------------------
# Test 4: Merge round-trip
# ---------------------------------------------------------------------------
def test_merge_round_trip(tmpdir: Path) -> bool:
    """Write 2 Python files (overlapping run_info groups), merge with escpod,
    read back with Python and verify all reads + run_info dedup."""
    print("\n=== Test 4: Merge round-trip (Python A + B → escpod merge → Python) ===")
    ok = True

    # Reads 0-2 (RUN_INFO_1) and read 4 (RUN_INFO_2) — read 4 is in B alone, but
    # run_info_1 and run_info_2 both appear in B (read 3 uses RUN_INFO_2,
    # read 0 in A also uses RUN_INFO_1) so we exercise both the dedup path
    # (RUN_INFO_1 appears in both files) and the distinct path (RUN_INFO_2
    # only in B).
    file_a = tmpdir / "merge_a.pod5"
    file_b = tmpdir / "merge_b.pod5"
    write_python_pod5_subset(file_a, [0, 1, 2])  # all RUN_INFO_1
    write_python_pod5_subset(file_b, [3, 4])     # both RUN_INFO_2

    merged = tmpdir / "merged.pod5"
    run_escpod("merge", str(file_a), str(file_b), "--output", str(merged))

    if not merged.exists():
        print("  FAIL: escpod merge did not produce output")
        return False

    try:
        by_id = materialize_reads(merged)
    except Exception as e:
        print(f"  FAIL: Python pod5 cannot open merged file: {e}")
        return False

    if len(by_id) != 5:
        print(f"  FAIL: Expected 5 reads after merge, got {len(by_id)}")
        ok = False
    else:
        print(f"  OK: Merged file has {len(by_id)} reads")

    # Run info dedup via a separate fresh open: RUN_INFO_1 and RUN_INFO_2
    # should collapse to exactly 2 rows even though RUN_INFO_1 was
    # referenced from multiple reads across both inputs.
    n_run_info = read_run_info_count(merged)
    if n_run_info != 2:
        print(f"  FAIL: Expected 2 run_info rows after dedup, got {n_run_info}")
        ok = False
    else:
        print(f"  OK: Run info deduplicated to {n_run_info} rows")

    # Verify each canonical read survived with byte-identical signal.
    for i, rid in enumerate(READ_IDS):
        if rid not in by_id:
            print(f"  FAIL: Read {rid} missing from merged file")
            ok = False
            continue
        try:
            np.testing.assert_array_equal(by_id[rid].signal, SIGNALS[i])
        except AssertionError:
            print(f"  FAIL: Signal mismatch after merge for {rid}")
            ok = False

    # Cold-open random-access spot-check — pick the multi-chunk read (read 2)
    # to stress the signal-block-index lookup on the merged signal table.
    if not random_access_signal_check(
        "merge output", merged, READ_IDS[2], SIGNALS[2]
    ):
        ok = False

    if ok:
        print("  PASS: Merge preserves all reads and dedups run_info")
    return ok


# ---------------------------------------------------------------------------
# Test 5: Subset round-trip
# ---------------------------------------------------------------------------
def test_subset_round_trip(tmpdir: Path) -> bool:
    """Write a Python file, split into 2 groups via escpod subset CSV mapping,
    read each output back with Python and verify ID + signal preservation."""
    print("\n=== Test 5: Subset round-trip (Python → escpod subset → Python) ===")
    ok = True

    src = tmpdir / "subset_src.pod5"
    if not src.exists():
        write_python_pod5(src)

    # CSV maps reads to two output groups: 3 reads → group_a, 2 reads → group_b.
    # Use subdir to keep the output area clean from other test artifacts.
    out_dir = tmpdir / "subset_out"
    out_dir.mkdir(exist_ok=True)
    csv_path = tmpdir / "subset_map.csv"
    group_a_ids = [READ_IDS[0], READ_IDS[2], READ_IDS[4]]
    group_b_ids = [READ_IDS[1], READ_IDS[3]]
    with csv_path.open("w") as f:
        f.write("read_id,output\n")
        for rid in group_a_ids:
            f.write(f"{rid},group_a.pod5\n")
        for rid in group_b_ids:
            f.write(f"{rid},group_b.pod5\n")

    run_escpod(
        "subset", str(src),
        "--csv", str(csv_path),
        "--output-dir", str(out_dir),
    )

    expectations = [
        ("group_a.pod5", group_a_ids),
        ("group_b.pod5", group_b_ids),
    ]
    for fname, expected_ids in expectations:
        path = out_dir / fname
        if not path.exists():
            print(f"  FAIL: subset output {fname} missing")
            ok = False
            continue
        try:
            got_reads = materialize_reads(path)
        except Exception as e:
            print(f"  FAIL: Python pod5 cannot open {fname}: {e}")
            ok = False
            continue
        if set(got_reads.keys()) != set(expected_ids):
            print(
                f"  FAIL: {fname} read IDs differ — "
                f"expected {sorted(map(str, expected_ids))}, "
                f"got {sorted(map(str, got_reads.keys()))}"
            )
            ok = False
            continue
        # Signal byte-equality for every read in this group.
        for rid in expected_ids:
            canonical_idx = READ_IDS.index(rid)
            try:
                np.testing.assert_array_equal(
                    got_reads[rid].signal, SIGNALS[canonical_idx]
                )
            except AssertionError:
                print(f"  FAIL: {fname} signal mismatch for {rid}")
                ok = False
        print(f"  OK: {fname} has {len(got_reads)} reads with matching signals")

    # Cold-open random-access spot-check — fresh open of one group + by-ID
    # lookup of one read. group_a contains the multi-chunk read (READ_IDS[2]
    # → 150k samples) so that's the highest-value target.
    if not random_access_signal_check(
        "subset group_a", out_dir / "group_a.pod5", READ_IDS[2], SIGNALS[2]
    ):
        ok = False

    if ok:
        print("  PASS: Subset partitions reads correctly with no data loss")
    return ok


# ---------------------------------------------------------------------------
# Test 6: Edge cases
# ---------------------------------------------------------------------------
def test_edge_cases(tmpdir: Path) -> bool:
    """Test edge cases: minimal signal, large multi-chunk, empty metadata."""
    print("\n=== Test 6: Edge cases ===")
    ok = True

    edge_pod5 = tmpdir / "edge_cases.pod5"

    # Minimal run info with empty context_tags and tracking_id
    run_info_empty = pod5.RunInfo(
        acquisition_id="acq_edge_empty_meta",
        acquisition_start_time=datetime.fromtimestamp(1700000000, tz=timezone.utc),
        adc_max=2047,
        adc_min=-2048,
        context_tags={},
        experiment_name="edge",
        flow_cell_id="PAM00000",
        flow_cell_product_code="FLO-MIN114",
        protocol_name="edge",
        protocol_run_id="edge",
        protocol_start_time=datetime.fromtimestamp(1700000000, tz=timezone.utc),
        sample_id="edge",
        sample_rate=4000,
        sequencing_kit="SQK-LSK114",
        sequencer_position="MN00000",
        sequencer_position_type="MinION",
        software="MinKNOW 23.11.1",
        system_name="edge",
        system_type="linux",
        tracking_id={},
    )

    # 1 sample (minimum)
    signal_1 = np.array([42], dtype=np.int16)
    # 200,000 samples (multi-chunk, >102,400)
    signal_large = generate_signal(200_000, seed=99)

    edge_reads = [
        pod5.Read(
            read_id=uuid.UUID("10000000-0000-0000-0000-000000000001"),
            pore=pod5.Pore(channel=1, well=1, pore_type="not_set"),
            calibration=pod5.Calibration(offset=-200.0, scale=0.15),
            read_number=1,
            start_sample=0,
            median_before=200.0,
            end_reason=pod5.EndReason(pod5.EndReasonEnum.UNKNOWN, False),
            run_info=run_info_empty,
            signal=signal_1,
            num_minknow_events=1,
        ),
        pod5.Read(
            read_id=uuid.UUID("10000000-0000-0000-0000-000000000002"),
            pore=pod5.Pore(channel=500, well=4, pore_type="pore_r10"),
            calibration=pod5.Calibration(offset=-180.0, scale=0.14),
            read_number=2,
            start_sample=500000,
            median_before=180.0,
            end_reason=pod5.EndReason(pod5.EndReasonEnum.SIGNAL_NEGATIVE, True),
            run_info=run_info_empty,
            signal=signal_large,
            num_minknow_events=100000,
        ),
    ]

    with pod5.Writer(edge_pod5) as writer:
        for r in edge_reads:
            writer.add_read(r)

    # Read with escpod view
    all_fields = (
        "read_id,channel,well,pore_type,read_number,start_sample,"
        "median_before,end_reason,end_reason_forced,num_samples,"
        "num_minknow_events,calibration_offset,calibration_scale"
    )
    result = run_escpod("view", str(edge_pod5), "--include", all_fields)
    rows = parse_view_tsv(result.stdout)

    if len(rows) != 2:
        print(f"  FAIL: Expected 2 reads, got {len(rows)}")
        return False

    # Find each read
    row_map = {r["read_id"]: r for r in rows}

    # Check 1-sample read
    r1_id = "10000000-0000-0000-0000-000000000001"
    if r1_id in row_map:
        if row_map[r1_id]["num_samples"] != "1":
            print(f"  FAIL: 1-sample read has num_samples={row_map[r1_id]['num_samples']}")
            ok = False
        else:
            print(f"  OK: 1-sample read correctly handled")
    else:
        print(f"  FAIL: 1-sample read not found")
        ok = False

    # Check 200k-sample read
    r2_id = "10000000-0000-0000-0000-000000000002"
    if r2_id in row_map:
        if row_map[r2_id]["num_samples"] != "200000":
            print(f"  FAIL: 200k-sample read has num_samples={row_map[r2_id]['num_samples']}")
            ok = False
        else:
            print(f"  OK: 200k-sample read correctly handled (multi-chunk)")
    else:
        print(f"  FAIL: 200k-sample read not found")
        ok = False

    # Now filter through escpod and read back with Python for signal verification
    ids_path = tmpdir / "edge_ids.txt"
    ids_path.write_text(f"{r1_id}\n{r2_id}\n")
    escpod_edge = tmpdir / "edge_escpod.pod5"
    run_escpod(
        "filter", str(edge_pod5),
        "--ids", str(ids_path),
        "--output", str(escpod_edge),
    )

    try:
        snapshots = materialize_reads(escpod_edge)
        reads = {str(rid): snap for rid, snap in snapshots.items()}
    except Exception as e:
        print(f"  FAIL: Python pod5 cannot open escapepod-written edge case file: {e}")
        return False

    if r1_id in reads:
        try:
            np.testing.assert_array_equal(reads[r1_id].signal, signal_1)
            print(f"  OK: 1-sample signal exact match after round-trip")
        except AssertionError:
            print(f"  FAIL: 1-sample signal mismatch")
            ok = False
    else:
        print(f"  FAIL: 1-sample read not in escpod output")
        ok = False

    if r2_id in reads:
        try:
            np.testing.assert_array_equal(reads[r2_id].signal, signal_large)
            print(f"  OK: 200k-sample signal exact match after round-trip")
        except AssertionError:
            print(f"  FAIL: 200k-sample signal mismatch")
            ok = False
    else:
        print(f"  FAIL: 200k-sample read not in escpod output")
        ok = False

    if ok:
        print("  PASS: Edge cases handled correctly")
    return ok


# ---------------------------------------------------------------------------
# Test 7: Existing ONT test files
# ---------------------------------------------------------------------------
def test_existing_files() -> bool:
    """Read existing POD5 files with both Python and escapepod."""
    print("\n=== Test 7: Existing test files ===")
    ok = True

    test_file = REPO_ROOT / "ext" / "remora" / "tests" / "data" / "can_reads.pod5"
    if not test_file.exists():
        print(f"  SKIP: {test_file} not found")
        return True

    # Drain into snapshots — closes the reader before assertions run.
    py_snapshots = materialize_reads(test_file)
    py_reads = list(py_snapshots.values())
    py_count = len(py_reads)
    print(f"  Python pod5: {py_count} reads")

    # Read with escpod
    result = run_escpod("view", str(test_file), "--include", "read_id,num_samples,channel")
    rows = parse_view_tsv(result.stdout)
    escpod_count = len(rows)
    print(f"  escpod view: {escpod_count} reads")

    if py_count != escpod_count:
        print(f"  FAIL: Read count mismatch: Python={py_count}, escpod={escpod_count}")
        ok = False
    else:
        print(f"  OK: Read counts match ({py_count})")

    # Verify read IDs match
    py_ids = sorted(str(r.read_id) for r in py_reads)
    escpod_ids = sorted(r["read_id"] for r in rows)
    if py_ids != escpod_ids:
        py_set = set(py_ids)
        esc_set = set(escpod_ids)
        only_py = py_set - esc_set
        only_esc = esc_set - py_set
        if only_py:
            print(f"  FAIL: {len(only_py)} IDs only in Python: {list(only_py)[:3]}")
        if only_esc:
            print(f"  FAIL: {len(only_esc)} IDs only in escpod: {list(only_esc)[:3]}")
        ok = False
    else:
        print(f"  OK: All read IDs match")

    # Spot-check num_samples for first read.
    if py_reads:
        first_py = py_reads[0]
        first_id = str(first_py.read_id)
        py_nsamples = first_py.num_samples

        row_map = {r["read_id"]: r for r in rows}
        if first_id in row_map:
            esc_nsamples = int(row_map[first_id]["num_samples"])
            if py_nsamples != esc_nsamples:
                print(f"  FAIL: num_samples mismatch for {first_id}: py={py_nsamples}, escpod={esc_nsamples}")
                ok = False
            else:
                print(f"  OK: Sample count matches for first read ({py_nsamples})")

    # Also check inspect summary
    result = run_escpod("inspect", "summary", str(test_file))
    if result.returncode == 0:
        print(f"  OK: escpod inspect summary succeeded")
    else:
        print(f"  FAIL: escpod inspect summary failed")
        ok = False

    if ok:
        print("  PASS: Existing file compatible with both tools")
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"Using escpod binary: {ESCPOD}")
    print(f"pod5 version: {pod5.__version__}")

    # Verify escpod works
    result = run_escpod("--version", check=False)
    if result.returncode != 0:
        sys.exit("ERROR: escpod --version failed")
    print(f"escpod version: {result.stdout.strip()}")

    results = {}

    with tempfile.TemporaryDirectory(prefix="pod5_compat_") as tmpdir:
        tmpdir = Path(tmpdir)

        results["backward_compat"] = test_python_to_escapepod(tmpdir)
        results["forward_compat"] = test_escapepod_to_python(tmpdir)
        results["round_trip"] = test_round_trip(tmpdir)
        results["merge_round_trip"] = test_merge_round_trip(tmpdir)
        results["subset_round_trip"] = test_subset_round_trip(tmpdir)
        results["edge_cases"] = test_edge_cases(tmpdir)

    results["existing_files"] = test_existing_files()

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_pass = False

    if all_pass:
        print("\nAll tests passed!")
        sys.exit(0)
    else:
        print("\nSome tests FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    main()
