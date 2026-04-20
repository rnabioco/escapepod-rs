"""Tests for the escapepod Python bindings."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

import escapepod

# Locate a small test POD5 file relative to repo root
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TEST_POD5 = REPO_ROOT / "ext" / "dorado" / "tests" / "data" / "pod5" / "single_na24385.pod5"


# ---------------------------------------------------------------------------
# Module-level checks
# ---------------------------------------------------------------------------


class TestModule:
    def test_version(self):
        assert isinstance(escapepod.__version__, str)
        assert len(escapepod.__version__) > 0

    def test_has_reader_class(self):
        assert hasattr(escapepod, "Reader")

    def test_has_writer_class(self):
        assert hasattr(escapepod, "Writer")

    def test_has_create_run_info(self):
        assert callable(escapepod.create_run_info)


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


@pytest.fixture
def reader():
    """Open the test POD5 file."""
    if not TEST_POD5.exists():
        pytest.skip(f"Test POD5 not found: {TEST_POD5}")
    return escapepod.Reader(str(TEST_POD5))


@pytest.fixture
def reader_pathlib():
    """Open via pathlib.Path."""
    if not TEST_POD5.exists():
        pytest.skip(f"Test POD5 not found: {TEST_POD5}")
    return escapepod.Reader(TEST_POD5)


class TestReader:
    def test_open_string(self, reader):
        assert reader is not None

    def test_open_pathlib(self, reader_pathlib):
        assert reader_pathlib is not None

    def test_open_nonexistent(self):
        with pytest.raises(IOError):
            escapepod.Reader("/nonexistent/path.pod5")

    def test_context_manager(self):
        if not TEST_POD5.exists():
            pytest.skip("Test POD5 not found")
        with escapepod.Reader(str(TEST_POD5)) as r:
            assert r.read_count > 0

    def test_repr(self, reader):
        r = repr(reader)
        assert "Reader(" in r
        assert "reads=" in r

    def test_len(self, reader):
        assert len(reader) == reader.read_count
        assert len(reader) > 0

    def test_metadata(self, reader):
        assert isinstance(reader.path, str)
        assert isinstance(reader.file_identifier, str)
        assert isinstance(reader.software, str)
        assert isinstance(reader.pod5_version, str)

    def test_read_count(self, reader):
        assert reader.read_count > 0

    def test_read_batch_count(self, reader):
        assert reader.read_batch_count > 0

    def test_signal_row_count(self, reader):
        assert reader.signal_row_count > 0


class TestReadIds:
    def test_read_ids(self, reader):
        ids = reader.read_ids()
        assert isinstance(ids, list)
        assert len(ids) == reader.read_count
        # Each ID should be a UUID string
        assert all(len(uid) == 36 for uid in ids)


class TestReads:
    def test_all_reads(self, reader):
        reads = reader.reads()
        assert len(reads) == reader.read_count

    def test_reads_with_selection(self, reader):
        ids = reader.read_ids()
        subset = ids[:1]
        reads = reader.reads(selection=subset)
        assert len(reads) == 1
        assert reads[0].read_id == subset[0]

    def test_reads_empty_selection(self, reader):
        reads = reader.reads(selection=[])
        assert len(reads) == 0


class TestGetRead:
    def test_get_read(self, reader):
        read_id = reader.read_ids()[0]
        read = reader.get_read(read_id)
        assert read.read_id == read_id

    def test_get_read_not_found(self, reader):
        with pytest.raises(ValueError, match="Read not found"):
            reader.get_read("00000000-0000-0000-0000-000000000000")

    def test_get_read_invalid_uuid(self, reader):
        with pytest.raises(ValueError, match="Invalid UUID"):
            reader.get_read("not-a-uuid")

    def test_get_reads(self, reader):
        ids = reader.read_ids()[:2]
        reads = reader.get_reads(ids)
        assert len(reads) == len(ids)


class TestReadData:
    def test_properties(self, reader):
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
        assert isinstance(read.run_info_index, int)
        assert isinstance(read.num_minknow_events, int)
        assert isinstance(read.num_samples, int)
        assert isinstance(read.signal_rows, list)

    def test_equality(self, reader):
        reads = reader.reads()
        assert reads[0] == reads[0]
        if len(reads) > 1:
            assert reads[0] != reads[1]

    def test_hashable(self, reader):
        reads = reader.reads()
        # Can be used in sets
        read_set = set(reads)
        assert len(read_set) == len(reads)

    def test_repr(self, reader):
        read = reader.reads()[0]
        assert isinstance(repr(read), str)


class TestRunInfo:
    def test_run_infos(self, reader):
        run_infos = reader.run_infos
        assert len(run_infos) > 0

    def test_properties(self, reader):
        ri = reader.run_infos[0]

        assert isinstance(ri.acquisition_id, str)
        assert isinstance(ri.acquisition_start_time, int)
        assert isinstance(ri.adc_max, int)
        assert isinstance(ri.adc_min, int)
        assert isinstance(ri.sample_rate, int)
        assert isinstance(ri.context_tags, dict)
        assert isinstance(ri.tracking_id, dict)

    def test_repr(self, reader):
        ri = reader.run_infos[0]
        assert isinstance(repr(ri), str)


class TestSignal:
    def test_get_signal(self, reader):
        read = reader.reads()[0]
        signal = reader.get_signal(read)
        assert isinstance(signal, np.ndarray)
        assert signal.dtype == np.int16
        assert len(signal) == read.num_samples

    def test_get_signal_pa(self, reader):
        read = reader.reads()[0]
        signal_pa = reader.get_signal_pa(read)
        assert isinstance(signal_pa, np.ndarray)
        assert signal_pa.dtype == np.float32
        assert len(signal_pa) == read.num_samples

    def test_calibration_math(self, reader):
        read = reader.reads()[0]
        raw = reader.get_signal(read)
        pa = reader.get_signal_pa(read)
        expected = (raw.astype(np.float32) + read.calibration_offset) * read.calibration_scale
        np.testing.assert_allclose(pa, expected, rtol=1e-5)

    def test_get_signals_bulk(self, reader):
        reads = reader.reads()[:2]
        results = reader.get_signals(reads)
        assert len(results) == len(reads)
        for read_id, signal in results:
            assert isinstance(read_id, str)
            assert signal.dtype == np.int16

    def test_get_signals_pa_bulk(self, reader):
        reads = reader.reads()[:2]
        results = reader.get_signals_pa(reads)
        assert len(results) == len(reads)
        for read_id, signal_pa in results:
            assert isinstance(read_id, str)
            assert signal_pa.dtype == np.float32


class TestIterator:
    def test_iter(self, reader):
        count = 0
        for read in reader:
            assert isinstance(read, escapepod.ReadData)
            count += 1
        assert count == reader.read_count

    def test_iter_twice(self, reader):
        """Iterator should work correctly on multiple calls."""
        first = [r.read_id for r in reader]
        second = [r.read_id for r in reader]
        assert first == second


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


class TestWriter:
    def test_create_and_close(self):
        with tempfile.NamedTemporaryFile(suffix=".pod5", delete=False) as f:
            path = f.name
        try:
            writer = escapepod.Writer(path)
            writer.close()
            assert Path(path).stat().st_size > 0
        finally:
            Path(path).unlink(missing_ok=True)

    def test_context_manager(self):
        with tempfile.NamedTemporaryFile(suffix=".pod5", delete=False) as f:
            path = f.name
        try:
            with escapepod.Writer(path):
                pass
            assert Path(path).stat().st_size > 0
        finally:
            Path(path).unlink(missing_ok=True)

    def test_round_trip(self):
        """Write a POD5, then read it back and verify data."""
        with tempfile.NamedTemporaryFile(suffix=".pod5", delete=False) as f:
            path = f.name

        try:
            ri = escapepod.create_run_info(
                acquisition_id="test-acq-001",
                sample_rate=4000,
                experiment_name="pytest_round_trip",
            )

            signal = np.arange(1000, dtype=np.int16)
            read_id = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"

            with escapepod.Writer(path) as writer:
                ri_idx = writer.add_run_info(ri)
                writer.add_read(
                    read_id=read_id,
                    read_number=1,
                    start_sample=0,
                    channel=42,
                    well=1,
                    pore_type="not_set",
                    calibration_offset=0.0,
                    calibration_scale=1.0,
                    median_before=200.0,
                    end_reason="signal_positive",
                    end_reason_forced=False,
                    run_info_index=ri_idx,
                    num_minknow_events=100,
                    signal=signal,
                )

            # Read it back
            with escapepod.Reader(path) as reader:
                assert reader.read_count == 1
                read = reader.reads()[0]
                assert read.read_id == read_id
                assert read.channel == 42
                assert read.num_samples == 1000
                assert read.end_reason == "signal_positive"

                recovered_signal = reader.get_signal(read)
                np.testing.assert_array_equal(recovered_signal, signal)

                # Check run info
                run_infos = reader.run_infos
                assert len(run_infos) == 1
                assert run_infos[0].acquisition_id == "test-acq-001"
                assert run_infos[0].sample_rate == 4000
                assert run_infos[0].experiment_name == "pytest_round_trip"
        finally:
            Path(path).unlink(missing_ok=True)

    def test_round_trip_add_read_data(self):
        """Write using add_read_data from an existing read."""
        if not TEST_POD5.exists():
            pytest.skip("Test POD5 not found")

        with tempfile.NamedTemporaryFile(suffix=".pod5", delete=False) as f:
            path = f.name

        try:
            # Read a read from the test file
            with escapepod.Reader(str(TEST_POD5)) as src:
                original_read = src.reads()[0]
                original_signal = src.get_signal(original_read)
                original_run_info = src.run_infos[0]

            # Write it to a new file
            with escapepod.Writer(path) as writer:
                writer.add_run_info(original_run_info)
                writer.add_read_data(original_read, original_signal)

            # Read back and verify
            with escapepod.Reader(path) as reader:
                assert reader.read_count == 1
                read = reader.reads()[0]
                assert read.read_id == original_read.read_id
                signal = reader.get_signal(read)
                np.testing.assert_array_equal(signal, original_signal)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_pathlib_path(self):
        with tempfile.NamedTemporaryFile(suffix=".pod5", delete=False) as f:
            path = Path(f.name)
        try:
            with escapepod.Writer(path):
                pass
            assert path.stat().st_size > 0
        finally:
            path.unlink(missing_ok=True)


class TestCreateRunInfo:
    def test_minimal(self):
        ri = escapepod.create_run_info("test-acq")
        assert ri.acquisition_id == "test-acq"
        assert ri.sample_rate == 4000  # default

    def test_all_fields(self):
        ri = escapepod.create_run_info(
            acquisition_id="acq-123",
            acquisition_start_time=1000,
            adc_max=4095,
            adc_min=-4096,
            experiment_name="exp",
            flow_cell_id="FC001",
            sample_rate=5000,
            context_tags={"key": "value"},
            tracking_id={"device_id": "dev1"},
        )
        assert ri.acquisition_id == "acq-123"
        assert ri.acquisition_start_time == 1000
        assert ri.adc_max == 4095
        assert ri.experiment_name == "exp"
        assert ri.flow_cell_id == "FC001"
        assert ri.sample_rate == 5000
        assert ri.context_tags == {"key": "value"}
        assert ri.tracking_id == {"device_id": "dev1"}


class TestIndex:
    def test_has_index(self, reader):
        # May or may not have one
        assert isinstance(reader.has_index, bool)

    def test_build_index(self):
        if not TEST_POD5.exists():
            pytest.skip("Test POD5 not found")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Copy the pod5 so we don't pollute the test data directory
            import shutil

            tmp_pod5 = Path(tmpdir) / "test.pod5"
            shutil.copy2(TEST_POD5, tmp_pod5)

            reader = escapepod.Reader(str(tmp_pod5))
            count = reader.build_index()
            assert count == reader.read_count
            assert reader.has_index
