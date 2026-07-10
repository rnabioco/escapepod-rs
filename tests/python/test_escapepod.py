"""Tests for the escapepod Python bindings."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

import escapepod

# Locate a small test POD5 file relative to repo root
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TEST_POD5 = (
    REPO_ROOT / "ext" / "dorado" / "tests" / "data" / "pod5" / "single_na24385.pod5"
)


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

    def test_open_invalid_raises_pod5error(self):
        """A file that exists but isn't POD5 raises the library's Pod5Error."""
        with tempfile.NamedTemporaryFile(suffix=".pod5", delete=False) as f:
            f.write(b"not a pod5 file, just some bytes\n" * 32)
            path = f.name
        try:
            assert issubclass(escapepod.Pod5Error, Exception)
            with pytest.raises(escapepod.Pod5Error):
                escapepod.Reader(path)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_prefetch_signal(self, reader):
        """prefetch_signal is a no-op hint; signal stays readable afterwards."""
        reader.prefetch_signal()
        read = reader.reads()[0]
        assert len(reader.get_signal(read)) == read.num_samples

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
        expected = (
            raw.astype(np.float32) + read.calibration_offset
        ) * read.calibration_scale
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


class TestRunInfoConstructor:
    def test_minimal(self):
        ri = escapepod.RunInfo("test-acq")
        assert ri.acquisition_id == "test-acq"
        assert ri.sample_rate == 4000

    def test_kwargs(self):
        ri = escapepod.RunInfo("acq-1", sample_rate=5000, flow_cell_id="FC")
        assert ri.sample_rate == 5000
        assert ri.flow_cell_id == "FC"

    def test_usable_as_writer_input(self):
        ri = escapepod.RunInfo("acq-1", sample_rate=4000)
        with tempfile.NamedTemporaryFile(suffix=".pod5", delete=False) as f:
            path = f.name
        try:
            with escapepod.Writer(path) as w:
                idx = w.add_run_info(ri)
                assert idx == 0
            with escapepod.Reader(path) as r:
                assert r.run_infos[0].acquisition_id == "acq-1"
        finally:
            Path(path).unlink(missing_ok=True)

    def test_all_fields_round_trip(self):
        """Every RunInfo field should survive a write/read cycle.

        Each field gets a distinct value so a dropped or swapped column fails
        loudly, rather than passing because two fields share a default. Asserts
        are spelled out (not a ``getattr`` loop) so each getter is exercised
        explicitly and a failure names the exact field.
        """
        ri = escapepod.RunInfo(
            "acq-full",
            acquisition_start_time=111,
            adc_max=2000,
            adc_min=-2000,
            experiment_name="exp",
            flow_cell_id="FC-1",
            flow_cell_product_code="FLO-PRO114",
            protocol_name="sequencing/run",
            protocol_run_id="run-abc",
            protocol_start_time=222,
            sample_id="sample-42",
            sample_rate=5000,
            sequencing_kit="SQK-RNA004",
            sequencer_position="X1",
            sequencer_position_type="promethion",
            software="escapepod-test",
            system_name="sys-name",
            system_type="sys-type",
            context_tags={"ctx_key": "ctx_val"},
            tracking_id={"trk_key": "trk_val"},
        )

        with tempfile.NamedTemporaryFile(suffix=".pod5", delete=False) as f:
            path = f.name
        try:
            with escapepod.Writer(path) as w:
                w.add_run_info(ri)
                w.add_read(
                    read_id="11111111-2222-3333-4444-555555555555",
                    read_number=1,
                    start_sample=0,
                    channel=1,
                    well=1,
                    pore_type="not_set",
                    calibration_offset=0.0,
                    calibration_scale=1.0,
                    median_before=100.0,
                    end_reason="signal_positive",
                    end_reason_forced=False,
                    run_info_index=0,
                    num_minknow_events=0,
                    signal=np.arange(50, dtype=np.int16),
                )
            with escapepod.Reader(path) as r:
                out = r.run_infos[0]
                assert out.acquisition_id == "acq-full"
                assert out.acquisition_start_time == 111
                assert out.adc_max == 2000
                assert out.adc_min == -2000
                assert out.experiment_name == "exp"
                assert out.flow_cell_id == "FC-1"
                assert out.flow_cell_product_code == "FLO-PRO114"
                assert out.protocol_name == "sequencing/run"
                assert out.protocol_run_id == "run-abc"
                assert out.protocol_start_time == 222
                assert out.sample_id == "sample-42"
                assert out.sample_rate == 5000
                assert out.sequencing_kit == "SQK-RNA004"
                assert out.sequencer_position == "X1"
                assert out.sequencer_position_type == "promethion"
                assert out.software == "escapepod-test"
                assert out.system_name == "sys-name"
                assert out.system_type == "sys-type"
                assert out.context_tags == {"ctx_key": "ctx_val"}
                assert out.tracking_id == {"trk_key": "trk_val"}
        finally:
            Path(path).unlink(missing_ok=True)


class TestReadDataConstructor:
    def test_minimal(self):
        rd = escapepod.ReadData("a1b2c3d4-e5f6-7890-abcd-ef1234567890")
        assert rd.read_id == "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
        assert rd.read_number == 0
        assert rd.calibration_scale == 1.0
        assert rd.end_reason == "unknown"

    def test_full_kwargs_round_trip(self):
        """All ReadData fields should survive a write/read cycle via the constructor."""
        with tempfile.NamedTemporaryFile(suffix=".pod5", delete=False) as f:
            path = f.name
        try:
            ri = escapepod.RunInfo("acq-1")
            rd = escapepod.ReadData(
                "11111111-2222-3333-4444-555555555555",
                read_number=7,
                channel=42,
                well=2,
                pore_type="not_set",
                calibration_offset=10.0,
                calibration_scale=0.5,
                median_before=199.5,
                end_reason="signal_positive",
                end_reason_forced=True,
                num_minknow_events=88,
                tracked_scaling_scale=1.25,
                tracked_scaling_shift=-3.5,
                predicted_scaling_scale=1.1,
                predicted_scaling_shift=0.25,
                num_reads_since_mux_change=12,
                time_since_mux_change=4.2,
                open_pore_level=180.0,
                expected_open_pore_level=175.0,
                selected_read_level=190.0,
            )
            sig = np.arange(500, dtype=np.int16)
            with escapepod.Writer(path) as w:
                w.add_run_info(ri)
                w.add_read_data(rd, sig)
            with escapepod.Reader(path) as r:
                out = r.reads()[0]
                assert out.read_number == 7
                assert out.channel == 42
                assert out.well == 2
                assert out.end_reason == "signal_positive"
                assert out.end_reason_forced is True
                assert out.num_minknow_events == 88
                assert out.tracked_scaling_scale == pytest.approx(1.25)
                assert out.tracked_scaling_shift == pytest.approx(-3.5)
                assert out.predicted_scaling_scale == pytest.approx(1.1)
                assert out.predicted_scaling_shift == pytest.approx(0.25)
                assert out.num_reads_since_mux_change == 12
                assert out.time_since_mux_change == pytest.approx(4.2)
                assert out.open_pore_level == pytest.approx(180.0)
                # POD5 schema V5 fields.
                assert out.expected_open_pore_level == pytest.approx(175.0)
                assert out.selected_read_level == pytest.approx(190.0)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_invalid_uuid(self):
        with pytest.raises(ValueError, match="Invalid UUID"):
            escapepod.ReadData("not-a-uuid")

    def test_invalid_end_reason(self):
        with pytest.raises(ValueError, match="Invalid end_reason"):
            escapepod.ReadData(
                "11111111-2222-3333-4444-555555555555",
                end_reason="bogus",
            )


class TestWriterDropWarning:
    def test_unclosed_writer_emits_resource_warning(self):
        """Forgetting to close should not corrupt the file; it should warn.

        Writes one read before dropping so the readback path is meaningful
        (a truly empty POD5 has no reads table and can't be reopened).
        """
        import gc

        with tempfile.NamedTemporaryFile(suffix=".pod5", delete=False) as f:
            path = f.name
        try:
            read_id = "11111111-2222-3333-4444-555555555555"
            sig = np.arange(100, dtype=np.int16)

            with pytest.warns(ResourceWarning, match="not explicitly closed"):
                w = escapepod.Writer(path)
                ri_idx = w.add_run_info(escapepod.RunInfo("drop-acq"))
                w.add_read(
                    read_id=read_id,
                    read_number=1,
                    start_sample=0,
                    channel=1,
                    well=1,
                    pore_type="not_set",
                    calibration_offset=0.0,
                    calibration_scale=1.0,
                    median_before=200.0,
                    end_reason="signal_positive",
                    end_reason_forced=False,
                    run_info_index=ri_idx,
                    num_minknow_events=0,
                    signal=sig,
                )
                del w
                gc.collect()

            # Best-effort finalize in Drop should have written a valid file.
            with escapepod.Reader(path) as r:
                assert r.read_count == 1
                assert r.reads()[0].read_id == read_id
        finally:
            Path(path).unlink(missing_ok=True)


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


# ---------------------------------------------------------------------------
# Helpers for tests that need synthetic multi-file datasets
# ---------------------------------------------------------------------------


def _write_pod5(path, read_ids, samples_per_read=500, acq_id="ds-acq"):
    """Write a small POD5 with one read per id and return the id list."""
    ri = escapepod.create_run_info(acquisition_id=acq_id, sample_rate=4000)
    with escapepod.Writer(str(path)) as writer:
        ri_idx = writer.add_run_info(ri)
        for i, rid in enumerate(read_ids):
            signal = np.arange(samples_per_read, dtype=np.int16) + i
            writer.add_read(
                read_id=rid,
                read_number=i,
                start_sample=i * samples_per_read,
                channel=i + 1,
                well=1,
                pore_type="not_set",
                calibration_offset=10.0,
                calibration_scale=0.2,
                median_before=200.0,
                end_reason="signal_positive",
                end_reason_forced=False,
                run_info_index=ri_idx,
                num_minknow_events=0,
                signal=signal,
            )
    return read_ids


# Two files' worth of stable UUIDs
_IDS_A = [f"aaaaaaaa-0000-0000-0000-{i:012d}" for i in range(3)]
_IDS_B = [f"bbbbbbbb-0000-0000-0000-{i:012d}" for i in range(2)]


@pytest.fixture
def single_file():
    """A single synthetic POD5 with the file-A reads (no external fixture)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "single.pod5"
        _write_pod5(path, _IDS_A, acq_id="acq-single")
        yield path


@pytest.fixture
def dataset_dir():
    """A directory holding two POD5 files (3 + 2 reads), with a subdir file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write_pod5(root / "file_a.pod5", _IDS_A, acq_id="acq-a")
        sub = root / "sub"
        sub.mkdir()
        _write_pod5(sub / "file_b.pod5", _IDS_B, acq_id="acq-b")
        yield root


class TestDatasetReader:
    def test_open_directory_recursive(self, dataset_dir):
        ds = escapepod.DatasetReader(dataset_dir)
        assert ds.file_count == 2
        assert ds.read_count == 5
        assert len(ds) == 5

    def test_paths(self, dataset_dir):
        ds = escapepod.DatasetReader(dataset_dir)
        paths = ds.paths
        assert isinstance(paths, list)
        assert len(paths) == ds.file_count
        assert all(str(p).endswith(".pod5") for p in paths)

    def test_non_recursive_skips_subdir(self, dataset_dir):
        ds = escapepod.DatasetReader(dataset_dir, recursive=False)
        assert ds.file_count == 1
        assert ds.read_count == 3

    def test_open_list_of_files(self, dataset_dir):
        files = [dataset_dir / "file_a.pod5", dataset_dir / "sub" / "file_b.pod5"]
        ds = escapepod.DatasetReader(files)
        assert ds.file_count == 2
        assert set(ds.read_ids()) == set(_IDS_A + _IDS_B)

    def test_empty_directory_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="no POD5 files"):
                escapepod.DatasetReader(tmpdir)

    def test_iteration(self, dataset_dir):
        ds = escapepod.DatasetReader(dataset_dir)
        seen = [r.read_id for r in ds]
        assert set(seen) == set(_IDS_A + _IDS_B)
        assert len(seen) == 5

    def test_context_manager(self, dataset_dir):
        with escapepod.DatasetReader(dataset_dir) as ds:
            assert ds.read_count == 5

    def test_run_infos_deduped(self, dataset_dir):
        ds = escapepod.DatasetReader(dataset_dir)
        acqs = {ri.acquisition_id for ri in ds.run_infos}
        assert acqs == {"acq-a", "acq-b"}

    def test_selection_across_files(self, dataset_dir):
        ds = escapepod.DatasetReader(dataset_dir)
        sel = [_IDS_A[0], _IDS_B[1]]
        reads = ds.reads(selection=sel)
        assert {r.read_id for r in reads} == set(sel)

    def test_selection_missing_raises(self, dataset_dir):
        ds = escapepod.DatasetReader(dataset_dir)
        bogus = "cccccccc-0000-0000-0000-000000000000"
        with pytest.raises(KeyError):
            ds.reads(selection=[_IDS_A[0], bogus])

    def test_selection_missing_ok(self, dataset_dir):
        ds = escapepod.DatasetReader(dataset_dir)
        bogus = "cccccccc-0000-0000-0000-000000000000"
        reads = ds.reads(selection=[_IDS_A[0], bogus], missing_ok=True)
        assert {r.read_id for r in reads} == {_IDS_A[0]}

    def test_signal_routing(self, dataset_dir):
        """Signal for a read in file B must come from file B, not A."""
        ds = escapepod.DatasetReader(dataset_dir)
        # Read index 1 in file_b was written with signal arange(500)+1
        read = ds.reads(selection=[_IDS_B[1]])[0]
        sig = ds.get_signal(read)
        np.testing.assert_array_equal(sig, np.arange(500, dtype=np.int16) + 1)

    def test_bulk_signals_order_preserved(self, dataset_dir):
        ds = escapepod.DatasetReader(dataset_dir)
        reads = ds.reads(selection=_IDS_A + _IDS_B)
        results = ds.get_signals(reads)
        by_id = {rid: sig for rid, sig in results}
        assert set(by_id) == set(_IDS_A + _IDS_B)
        # file_a read 0 -> arange+0, file_b read 1 -> arange+1
        np.testing.assert_array_equal(by_id[_IDS_A[0]], np.arange(500, dtype=np.int16))

    def test_signal_pa_calibrated(self, dataset_dir):
        ds = escapepod.DatasetReader(dataset_dir)
        read = ds.reads(selection=[_IDS_A[0]])[0]
        pa = ds.get_signal_pa(read)
        expected = (np.arange(500, dtype=np.float32) + 10.0) * 0.2
        np.testing.assert_allclose(pa, expected, rtol=1e-5)

    def test_byte_count(self, dataset_dir):
        ds = escapepod.DatasetReader(dataset_dir)
        read = ds.reads(selection=[_IDS_B[1]])[0]
        n = ds.byte_count(read)
        # Compressed size is positive and smaller than the raw int16 payload.
        assert 0 < n < read.num_samples * 2


class TestMissingOk:
    def test_reader_selection_missing_raises(self, single_file):
        reader = escapepod.Reader(str(single_file))
        good = _IDS_A[0]
        bogus = "cccccccc-0000-0000-0000-000000000000"
        with pytest.raises(KeyError):
            reader.reads(selection=[good, bogus])

    def test_reader_selection_missing_ok(self, single_file):
        reader = escapepod.Reader(str(single_file))
        good = _IDS_A[0]
        bogus = "cccccccc-0000-0000-0000-000000000000"
        reads = reader.reads(selection=[good, bogus], missing_ok=True)
        assert len(reads) == 1

    def test_get_reads_missing_ok(self, single_file):
        reader = escapepod.Reader(str(single_file))
        good = _IDS_A[0]
        bogus = "cccccccc-0000-0000-0000-000000000000"
        assert len(reader.get_reads([good, bogus], missing_ok=True)) == 1
        with pytest.raises(KeyError):
            reader.get_reads([good, bogus])


class TestDataFrame:
    def test_to_dict_columns(self, dataset_dir):
        ds = escapepod.DatasetReader(dataset_dir)
        d = ds.to_dict()
        assert d["read_id"] and len(d["read_id"]) == 5
        assert "num_samples" in d
        assert "channel" in d
        # every column has the same length
        assert len({len(v) for v in d.values()}) == 1

    def test_reader_to_dict(self, single_file):
        reader = escapepod.Reader(str(single_file))
        d = reader.to_dict()
        assert len(d["read_id"]) == reader.read_count

    def test_to_pandas(self, dataset_dir):
        pytest.importorskip("pandas")
        ds = escapepod.DatasetReader(dataset_dir)
        df = ds.to_pandas()
        assert list(df["read_id"]) and len(df) == 5
        assert "calibration_scale" in df.columns

    def test_to_polars(self, dataset_dir):
        pytest.importorskip("polars")
        ds = escapepod.DatasetReader(dataset_dir)
        df = ds.to_polars()
        assert df.height == 5


class TestCalibrateSignalArray:
    def test_calibrate(self, single_file):
        reader = escapepod.Reader(str(single_file))
        read = reader.reads()[0]
        adc = reader.get_signal(read)
        pa = read.calibrate_signal_array(adc)
        expected = (
            adc.astype(np.float32) + read.calibration_offset
        ) * read.calibration_scale
        np.testing.assert_allclose(pa, expected, rtol=1e-5)


class TestByteCount:
    def test_reader_byte_count(self, single_file):
        reader = escapepod.Reader(str(single_file))
        read = reader.reads()[0]
        n = reader.byte_count(read)
        assert 0 < n < read.num_samples * 2


class TestBatchWrite:
    def test_add_reads_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "batch.pod5"
            ri = escapepod.create_run_info(acquisition_id="batch-acq")
            reads = [
                escapepod.ReadData(read_id=rid, channel=i + 1)
                for i, rid in enumerate(_IDS_A)
            ]
            signals = [np.arange(300, dtype=np.int16) + i for i in range(len(reads))]

            with escapepod.Writer(str(path)) as w:
                w.add_run_info(ri)
                w.add_reads(reads, signals)

            with escapepod.Reader(str(path)) as r:
                assert r.read_count == len(_IDS_A)
                assert set(r.read_ids()) == set(_IDS_A)

    def test_add_reads_length_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.pod5"
            with escapepod.Writer(str(path)) as w:
                reads = [escapepod.ReadData(read_id=_IDS_A[0])]
                with pytest.raises(ValueError, match="same length"):
                    w.add_reads(reads, [])
