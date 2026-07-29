"""Cross-reader interop between the escapepod Python library and Oxford
Nanopore's reference ``pod5`` package.

``test_escapepod.py`` already covers escapepod-writer -> escapepod-reader and
reads committed ONT fixtures. What was missing is a direct, in-process check
that the escapepod *library* (not the ``escpod`` CLI) agrees with the reference
``pod5`` writer/reader on the wire format, in both directions:

  * ``pod5.Writer``      -> ``escapepod.Reader``   (we read what pod5 wrote)
  * ``escapepod.Writer`` -> ``pod5.Reader``        (pod5 reads what we wrote)

The CLI-level version of this matrix lives in ``tests/compat/test_pod5_compat.py``;
this module is the library-binding counterpart. Every scalar field plus the raw
signal is asserted, and the reads include a >102,400-sample read so the VBZ
multi-chunk boundary is crossed by both writers.
"""

import uuid
from datetime import datetime, timezone

import numpy as np
import pod5
import pytest

import escapepod

# ---------------------------------------------------------------------------
# Canonical test data — three reads spanning tiny / medium / multi-chunk signal.
# ---------------------------------------------------------------------------
READ_IDS = [
    uuid.UUID("00000000-0000-0000-0000-0000000000a1"),
    uuid.UUID("00000000-0000-0000-0000-0000000000a2"),
    uuid.UUID("00000000-0000-0000-0000-0000000000a3"),
]
CHANNELS = [10, 200, 400]
WELLS = [1, 2, 3]
PORE_TYPES = ["not_set", "pore_r10", "not_set"]
READ_NUMBERS = [5, 15, 25]
START_SAMPLES = [0, 20_000, 100_000]
MEDIAN_BEFORES = [200.5, 185.25, 210.0]
CALIBRATION = [(-220.5, 0.15), (-180.25, 0.1452), (-200.0, 0.16)]
NUM_MINKNOW_EVENTS = [40, 1200, 8000]
END_REASONS = [
    (pod5.EndReasonEnum.UNKNOWN, False),
    (pod5.EndReasonEnum.MUX_CHANGE, True),
    (pod5.EndReasonEnum.SIGNAL_POSITIVE, True),
]
# 150_000 > 102_400 forces a multi-chunk signal, exercising the VBZ chunk seam.
SIGNAL_SIZES = [100, 5000, 150_000]
SIGNALS = [
    np.random.default_rng(i).integers(-2000, 2000, size=n, dtype=np.int16)
    for i, n in enumerate(SIGNAL_SIZES)
]

ACQUISITION_ID = "interop-acq-001"
SAMPLE_RATE = 4000

# Float tolerances mirror tests/compat/test_pod5_compat.py: calibration and
# median survive a float32 storage round-trip, so compare with a small abs tol.
OFFSET_TOL = 1e-3
SCALE_TOL = 1e-6
MEDIAN_TOL = 1e-2

# pod5's EndReason.name is reason.name.lower(), which is exactly the string
# escapepod stores/accepts (e.g. EndReasonEnum.SIGNAL_POSITIVE -> "signal_positive").
END_REASON_STRS = [enum.name.lower() for enum, _ in END_REASONS]


def _pod5_run_info() -> pod5.RunInfo:
    """A fully-populated pod5 RunInfo (pod5.Writer requires every field)."""
    ts = datetime.fromtimestamp(1_700_000_000, tz=timezone.utc)
    return pod5.RunInfo(
        acquisition_id=ACQUISITION_ID,
        acquisition_start_time=ts,
        adc_max=2047,
        adc_min=-2048,
        context_tags={"experiment_type": "genomic_dna"},
        experiment_name="interop_experiment",
        flow_cell_id="PAM12345",
        flow_cell_product_code="FLO-MIN114",
        protocol_name="sequencing/sequencing_MIN114_DNA",
        protocol_run_id="proto_001",
        protocol_start_time=ts,
        sample_id="interop_sample",
        sample_rate=SAMPLE_RATE,
        sequencing_kit="SQK-LSK114",
        sequencer_position="MN12345",
        sequencer_position_type="MinION",
        software="pod5-interop",
        system_name="host",
        system_type="linux",
        tracking_id={"device_id": "MN12345"},
    )


def _write_with_pod5(path) -> None:
    """Write the canonical reads with the reference ``pod5`` writer."""
    run_info = _pod5_run_info()
    with pod5.Writer(path) as writer:
        for i in range(len(READ_IDS)):
            reason, forced = END_REASONS[i]
            writer.add_read(
                pod5.Read(
                    read_id=READ_IDS[i],
                    pore=pod5.Pore(
                        channel=CHANNELS[i], well=WELLS[i], pore_type=PORE_TYPES[i]
                    ),
                    calibration=pod5.Calibration(
                        offset=CALIBRATION[i][0], scale=CALIBRATION[i][1]
                    ),
                    read_number=READ_NUMBERS[i],
                    start_sample=START_SAMPLES[i],
                    median_before=MEDIAN_BEFORES[i],
                    end_reason=pod5.EndReason(reason, forced),
                    run_info=run_info,
                    signal=SIGNALS[i],
                    num_minknow_events=NUM_MINKNOW_EVENTS[i],
                )
            )


def _write_with_escapepod(path) -> None:
    """Write the canonical reads with the escapepod library writer."""
    run_info = escapepod.create_run_info(
        acquisition_id=ACQUISITION_ID, sample_rate=SAMPLE_RATE
    )
    with escapepod.Writer(str(path)) as writer:
        ri_idx = writer.add_run_info(run_info)
        for i in range(len(READ_IDS)):
            writer.add_read(
                read_id=str(READ_IDS[i]),
                read_number=READ_NUMBERS[i],
                start_sample=START_SAMPLES[i],
                channel=CHANNELS[i],
                well=WELLS[i],
                pore_type=PORE_TYPES[i],
                calibration_offset=CALIBRATION[i][0],
                calibration_scale=CALIBRATION[i][1],
                median_before=MEDIAN_BEFORES[i],
                end_reason=END_REASON_STRS[i],
                end_reason_forced=END_REASONS[i][1],
                run_info_index=ri_idx,
                num_minknow_events=NUM_MINKNOW_EVENTS[i],
                signal=SIGNALS[i],
            )


class TestPod5ToEscapepod:
    """A file written by the reference ``pod5`` package is read by the
    escapepod library, field-for-field and signal-for-signal."""

    def test_read_pod5_written_file(self, tmp_path):
        path = tmp_path / "pod5_written.pod5"
        _write_with_pod5(path)

        with escapepod.Reader(str(path)) as reader:
            assert reader.read_count == len(READ_IDS)
            by_id = {r.read_id: r for r in reader.reads()}
            assert set(by_id) == {str(rid) for rid in READ_IDS}

            for i, rid in enumerate(READ_IDS):
                read = by_id[str(rid)]
                prefix = f"read {i}"
                assert read.channel == CHANNELS[i], prefix
                assert read.well == WELLS[i], prefix
                assert read.pore_type == PORE_TYPES[i], prefix
                assert read.read_number == READ_NUMBERS[i], prefix
                assert read.start_sample == START_SAMPLES[i], prefix
                assert read.num_samples == SIGNAL_SIZES[i], prefix
                assert read.num_minknow_events == NUM_MINKNOW_EVENTS[i], prefix
                assert read.end_reason == END_REASON_STRS[i], prefix
                assert read.end_reason_forced == END_REASONS[i][1], prefix
                assert read.calibration_offset == pytest.approx(
                    CALIBRATION[i][0], abs=OFFSET_TOL
                ), prefix
                assert read.calibration_scale == pytest.approx(
                    CALIBRATION[i][1], abs=SCALE_TOL
                ), prefix
                assert read.median_before == pytest.approx(
                    MEDIAN_BEFORES[i], abs=MEDIAN_TOL
                ), prefix

                signal = reader.get_signal(read)
                assert signal.dtype == np.int16, prefix
                np.testing.assert_array_equal(signal, SIGNALS[i], err_msg=prefix)

            # Run info survives with the acquisition id and sample rate intact.
            run_infos = reader.run_infos
            assert len(run_infos) == 1
            assert run_infos[0].acquisition_id == ACQUISITION_ID
            assert run_infos[0].sample_rate == SAMPLE_RATE


class TestEscapepodToPod5:
    """A file written by the escapepod library is read by the reference
    ``pod5`` package, field-for-field and signal-for-signal."""

    def test_read_escapepod_written_file(self, tmp_path):
        path = tmp_path / "escapepod_written.pod5"
        _write_with_escapepod(path)

        idx_by_id = {str(rid): i for i, rid in enumerate(READ_IDS)}
        seen = set()
        run_info_acqs = set()

        # pod5.Read.signal is lazy, so assert inside the open-reader context.
        with pod5.Reader(path) as reader:
            for r in reader.reads():
                rid = str(r.read_id)
                assert rid in idx_by_id, f"unexpected read {rid}"
                i = idx_by_id[rid]
                seen.add(rid)
                run_info_acqs.add(r.run_info.acquisition_id)

                prefix = f"read {i}"
                assert r.pore.channel == CHANNELS[i], prefix
                assert r.pore.well == WELLS[i], prefix
                assert r.pore.pore_type == PORE_TYPES[i], prefix
                assert r.read_number == READ_NUMBERS[i], prefix
                assert r.start_sample == START_SAMPLES[i], prefix
                assert r.num_samples == SIGNAL_SIZES[i], prefix
                assert r.num_minknow_events == NUM_MINKNOW_EVENTS[i], prefix
                assert r.end_reason.name == END_REASON_STRS[i], prefix
                assert r.end_reason.forced == END_REASONS[i][1], prefix
                assert r.calibration.offset == pytest.approx(
                    CALIBRATION[i][0], abs=OFFSET_TOL
                ), prefix
                assert r.calibration.scale == pytest.approx(
                    CALIBRATION[i][1], abs=SCALE_TOL
                ), prefix
                assert r.median_before == pytest.approx(
                    MEDIAN_BEFORES[i], abs=MEDIAN_TOL
                ), prefix

                signal = np.asarray(r.signal)
                assert signal.dtype == np.int16, prefix
                np.testing.assert_array_equal(signal, SIGNALS[i], err_msg=prefix)

        assert seen == set(idx_by_id)
        assert run_info_acqs == {ACQUISITION_ID}
