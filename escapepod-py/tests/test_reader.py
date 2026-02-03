"""Tests for the escapepod Python bindings."""

import os

import numpy as np
import pytest

import escapepod

# Resolve the test POD5 file relative to the repo root
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEST_POD5 = os.path.join(REPO_ROOT, "data", "drna", "yeast_trna_reads.pod5")


@pytest.fixture
def reader():
    return escapepod.Reader(TEST_POD5)


def test_open():
    r = escapepod.Reader(TEST_POD5)
    assert r.num_reads > 0


def test_open_function():
    r = escapepod.open(TEST_POD5)
    assert r.num_reads > 0


def test_len(reader):
    assert len(reader) == reader.num_reads
    assert len(reader) > 0


def test_repr(reader):
    s = repr(reader)
    assert "Reader" in s
    assert "num_reads=" in s


def test_context_manager():
    with escapepod.Reader(TEST_POD5) as r:
        assert r.num_reads > 0


def test_read_ids(reader):
    ids = reader.read_ids()
    assert len(ids) == reader.num_reads
    # UUIDs should be 36-character strings (8-4-4-4-12)
    for rid in ids:
        assert len(rid) == 36
        assert rid.count("-") == 4


def test_reads_iteration(reader):
    reads = reader.reads()
    assert len(reads) == reader.num_reads
    for read in reads:
        assert len(read.read_id) == 36
        assert read.num_samples > 0
        assert read.channel > 0


def test_read_properties(reader):
    read = reader.reads()[0]
    assert isinstance(read.read_id, str)
    assert isinstance(read.read_number, int)
    assert isinstance(read.start_sample, int)
    assert isinstance(read.channel, int)
    assert isinstance(read.well, int)
    assert isinstance(read.pore_type, str)
    assert isinstance(read.calibration_offset, float)
    assert isinstance(read.calibration_scale, float)
    assert isinstance(read.median_before, float)
    assert isinstance(read.end_reason, str)
    assert isinstance(read.end_reason_forced, bool)
    assert isinstance(read.num_samples, int)
    assert isinstance(read.num_minknow_events, int)
    assert isinstance(read.sample_rate, int)


def test_read_repr(reader):
    read = reader.reads()[0]
    s = repr(read)
    assert "Read" in s
    assert "read_id=" in s


def test_calibration(reader):
    read = reader.reads()[0]
    cal = read.calibration
    assert isinstance(cal, escapepod.Calibration)
    assert isinstance(cal.offset, float)
    assert isinstance(cal.scale, float)
    assert cal.offset == read.calibration_offset
    assert cal.scale == read.calibration_scale
    s = repr(cal)
    assert "Calibration" in s


def test_signal_dtype_and_shape(reader):
    read = reader.reads()[0]
    signal = read.signal
    assert isinstance(signal, np.ndarray)
    assert signal.dtype == np.int16
    assert signal.ndim == 1
    assert signal.shape[0] == read.num_samples


def test_signal_pa(reader):
    read = reader.reads()[0]
    signal = read.signal
    signal_pa = read.signal_pa
    assert signal_pa.dtype == np.float32
    assert signal_pa.shape == signal.shape
    # Verify calibration: signal_pa = (signal + offset) * scale
    expected = (signal.astype(np.float32) + read.calibration_offset) * read.calibration_scale
    np.testing.assert_allclose(signal_pa, expected, rtol=1e-5)


def test_run_info(reader):
    read = reader.reads()[0]
    ri = read.run_info
    assert isinstance(ri, escapepod.RunInfo)
    assert isinstance(ri.acquisition_id, str)
    assert isinstance(ri.flow_cell_id, str)
    assert isinstance(ri.sample_rate, int)
    assert ri.sample_rate > 0
    assert isinstance(ri.context_tags, dict)
    assert isinstance(ri.tracking_id, dict)
    s = repr(ri)
    assert "RunInfo" in s


def test_run_infos_property(reader):
    run_infos = reader.run_infos
    assert len(run_infos) > 0
    for ri in run_infos:
        assert isinstance(ri, escapepod.RunInfo)
        assert isinstance(ri.acquisition_id, str)


def test_selection_filter(reader):
    all_ids = reader.read_ids()
    subset_ids = all_ids[:2]
    subset = reader.reads(selection=subset_ids)
    assert len(subset) == len(subset_ids)
    returned_ids = {r.read_id for r in subset}
    assert returned_ids == set(subset_ids)


def test_invalid_file():
    with pytest.raises((IOError, ValueError)):
        escapepod.Reader("/nonexistent/file.pod5")


def test_file_metadata(reader):
    assert isinstance(reader.file_identifier, str)
    assert isinstance(reader.writing_software, str)
    assert isinstance(reader.pod5_version, str)
    assert len(reader.pod5_version) > 0
