"""Type stubs for the escapepod POD5 library."""

from os import PathLike
from typing import Any, Iterator, Optional, Union

import numpy as np
import numpy.typing as npt

__version__: str

class Pod5Error(Exception):
    """Base exception for POD5 file errors."""

    ...

class ReadData:
    """A single read's metadata from a POD5 file."""

    def __init__(
        self,
        read_id: str,
        read_number: int = 0,
        start_sample: int = 0,
        channel: int = 0,
        well: int = 0,
        pore_type: str = "not_set",
        calibration_offset: float = 0.0,
        calibration_scale: float = 1.0,
        median_before: float = 0.0,
        end_reason: str = "unknown",
        end_reason_forced: bool = False,
        run_info_index: int = 0,
        num_minknow_events: int = 0,
        tracked_scaling_scale: float = 1.0,
        tracked_scaling_shift: float = 0.0,
        predicted_scaling_scale: float = 1.0,
        predicted_scaling_shift: float = 0.0,
        num_reads_since_mux_change: int = 0,
        time_since_mux_change: float = 0.0,
        num_samples: int = 0,
        open_pore_level: float = 0.0,
        expected_open_pore_level: float = 0.0,
        selected_read_level: float = 0.0,
        signal_rows: Optional[list[int]] = None,
    ) -> None: ...
    @property
    def read_id(self) -> str: ...
    @property
    def read_number(self) -> int: ...
    @property
    def start_sample(self) -> int: ...
    @property
    def channel(self) -> int: ...
    @property
    def well(self) -> int: ...
    @property
    def pore_type(self) -> str: ...
    @property
    def calibration_offset(self) -> float: ...
    @property
    def calibration_scale(self) -> float: ...
    @property
    def median_before(self) -> float: ...
    @property
    def end_reason(self) -> str: ...
    @property
    def end_reason_forced(self) -> bool: ...
    @property
    def run_info_index(self) -> int: ...
    @property
    def num_minknow_events(self) -> int: ...
    @property
    def tracked_scaling_scale(self) -> float: ...
    @property
    def tracked_scaling_shift(self) -> float: ...
    @property
    def predicted_scaling_scale(self) -> float: ...
    @property
    def predicted_scaling_shift(self) -> float: ...
    @property
    def num_reads_since_mux_change(self) -> int: ...
    @property
    def time_since_mux_change(self) -> float: ...
    @property
    def num_samples(self) -> int: ...
    @property
    def open_pore_level(self) -> float: ...
    @property
    def expected_open_pore_level(self) -> float: ...
    @property
    def selected_read_level(self) -> float: ...
    @property
    def signal_rows(self) -> list[int]: ...
    def calibrate_signal_array(
        self, signal_adc: npt.NDArray[np.int16]
    ) -> npt.NDArray[np.float32]: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    def __repr__(self) -> str: ...

class RunInfo:
    """Run information metadata from a POD5 file."""

    def __init__(
        self,
        acquisition_id: str,
        acquisition_start_time: int = 0,
        adc_max: int = 2047,
        adc_min: int = -2048,
        experiment_name: str = "",
        flow_cell_id: str = "",
        flow_cell_product_code: str = "",
        protocol_name: str = "",
        protocol_run_id: str = "",
        protocol_start_time: int = 0,
        sample_id: str = "",
        sample_rate: int = 4000,
        sequencing_kit: str = "",
        sequencer_position: str = "",
        sequencer_position_type: str = "",
        software: str = "",
        system_name: str = "",
        system_type: str = "",
        context_tags: Optional[dict[str, str]] = None,
        tracking_id: Optional[dict[str, str]] = None,
    ) -> None: ...
    @property
    def acquisition_id(self) -> str: ...
    @property
    def acquisition_start_time(self) -> int: ...
    @property
    def adc_max(self) -> int: ...
    @property
    def adc_min(self) -> int: ...
    @property
    def experiment_name(self) -> str: ...
    @property
    def flow_cell_id(self) -> str: ...
    @property
    def flow_cell_product_code(self) -> str: ...
    @property
    def protocol_name(self) -> str: ...
    @property
    def protocol_run_id(self) -> str: ...
    @property
    def protocol_start_time(self) -> int: ...
    @property
    def sample_id(self) -> str: ...
    @property
    def sample_rate(self) -> int: ...
    @property
    def sequencing_kit(self) -> str: ...
    @property
    def sequencer_position(self) -> str: ...
    @property
    def sequencer_position_type(self) -> str: ...
    @property
    def software(self) -> str: ...
    @property
    def system_name(self) -> str: ...
    @property
    def system_type(self) -> str: ...
    @property
    def context_tags(self) -> dict[str, str]: ...
    @property
    def tracking_id(self) -> dict[str, str]: ...
    def __repr__(self) -> str: ...

class Reader:
    """Reader for POD5 files.

    Can be used as a context manager::

        with Reader("reads.pod5") as reader:
            for read in reader:
                print(read.read_id)
    """

    def __init__(self, path: Union[str, PathLike[str]]) -> None: ...
    @property
    def path(self) -> str: ...
    @property
    def file_identifier(self) -> str: ...
    @property
    def software(self) -> str: ...
    @property
    def pod5_version(self) -> str: ...
    @property
    def read_count(self) -> int: ...
    @property
    def read_batch_count(self) -> int: ...
    @property
    def signal_row_count(self) -> int: ...
    @property
    def run_infos(self) -> list[RunInfo]: ...
    @property
    def has_index(self) -> bool: ...
    def read_ids(self) -> list[str]: ...
    def reads(
        self, selection: Optional[list[str]] = None, missing_ok: bool = False
    ) -> list[ReadData]: ...
    def to_dict(
        self, selection: Optional[list[str]] = None, missing_ok: bool = False
    ) -> dict[str, list]: ...
    def to_pandas(
        self, selection: Optional[list[str]] = None, missing_ok: bool = False
    ) -> Any: ...
    def to_polars(
        self, selection: Optional[list[str]] = None, missing_ok: bool = False
    ) -> Any: ...
    def get_read(self, read_id: str) -> ReadData: ...
    def get_reads(
        self, read_ids: list[str], missing_ok: bool = False
    ) -> list[ReadData]: ...
    def get_signal(self, read: ReadData) -> npt.NDArray[np.int16]: ...
    def get_signal_pa(self, read: ReadData) -> npt.NDArray[np.float32]: ...
    def get_signals(
        self, reads: list[ReadData]
    ) -> list[tuple[str, npt.NDArray[np.int16]]]: ...
    def get_signals_pa(
        self, reads: list[ReadData]
    ) -> list[tuple[str, npt.NDArray[np.float32]]]: ...
    def byte_count(self, read: ReadData) -> int: ...
    def build_index(self) -> int: ...
    def prefetch_signal(self) -> None: ...
    def __enter__(self) -> "Reader": ...
    def __exit__(self, *args: object) -> bool: ...
    def __repr__(self) -> str: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[ReadData]: ...

class DatasetReader:
    """Reader over a collection of POD5 files as one logical dataset.

    Accepts a single file, a directory (scanned for ``*.pod5``), or a list
    mixing files and directories, and presents the reads across every file as
    a single stream — the escapepod analogue of ``pod5.DatasetReader``.

    Can be used as a context manager::

        with DatasetReader("run_dir/") as ds:
            for read in ds:
                signal = ds.get_signal(read)
    """

    def __init__(
        self,
        path: Union[str, PathLike[str], list[Union[str, PathLike[str]]]],
        recursive: bool = True,
        pattern: str = "*.pod5",
    ) -> None: ...
    @property
    def paths(self) -> list[str]: ...
    @property
    def file_count(self) -> int: ...
    @property
    def read_count(self) -> int: ...
    @property
    def run_infos(self) -> list[RunInfo]: ...
    def read_ids(self) -> list[str]: ...
    def reads(
        self,
        selection: Optional[list[str]] = None,
        missing_ok: bool = False,
    ) -> list[ReadData]: ...
    def to_dict(
        self, selection: Optional[list[str]] = None, missing_ok: bool = False
    ) -> dict[str, list]: ...
    def to_pandas(
        self, selection: Optional[list[str]] = None, missing_ok: bool = False
    ) -> Any: ...
    def to_polars(
        self, selection: Optional[list[str]] = None, missing_ok: bool = False
    ) -> Any: ...
    def get_signal(self, read: ReadData) -> npt.NDArray[np.int16]: ...
    def get_signal_pa(self, read: ReadData) -> npt.NDArray[np.float32]: ...
    def byte_count(self, read: ReadData) -> int: ...
    def get_signals(
        self, reads: list[ReadData]
    ) -> list[tuple[str, npt.NDArray[np.int16]]]: ...
    def get_signals_pa(
        self, reads: list[ReadData]
    ) -> list[tuple[str, npt.NDArray[np.float32]]]: ...
    def __enter__(self) -> "DatasetReader": ...
    def __exit__(self, *args: object) -> bool: ...
    def __repr__(self) -> str: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[ReadData]: ...

class Writer:
    """Writer for POD5 files.

    Can be used as a context manager::

        with Writer("output.pod5") as writer:
            ri_idx = writer.add_run_info(run_info)
            writer.add_read(...)
    """

    def __init__(
        self,
        path: Union[str, PathLike[str]],
        max_signal_chunk_size: Optional[int] = None,
        signal_batch_size: Optional[int] = None,
        read_batch_size: Optional[int] = None,
        compress_signal: Optional[bool] = None,
        software: Optional[str] = None,
    ) -> None: ...
    def add_run_info(self, run_info: RunInfo) -> int: ...
    def add_read(
        self,
        read_id: str,
        read_number: int,
        start_sample: int,
        channel: int,
        well: int,
        pore_type: str,
        calibration_offset: float,
        calibration_scale: float,
        median_before: float,
        end_reason: str,
        end_reason_forced: bool,
        run_info_index: int,
        num_minknow_events: int,
        signal: npt.NDArray[np.int16],
        num_samples: Optional[int] = None,
        tracked_scaling_scale: float = 1.0,
        tracked_scaling_shift: float = 0.0,
        predicted_scaling_scale: float = 1.0,
        predicted_scaling_shift: float = 0.0,
        num_reads_since_mux_change: int = 0,
        time_since_mux_change: float = 0.0,
        open_pore_level: float = 0.0,
        expected_open_pore_level: float = 0.0,
        selected_read_level: float = 0.0,
    ) -> None: ...
    def add_read_data(
        self, read: ReadData, signal: npt.NDArray[np.int16]
    ) -> None: ...
    def add_reads(
        self,
        reads: list[ReadData],
        signals: list[npt.NDArray[np.int16]],
    ) -> None: ...
    def close(self) -> None: ...
    def __enter__(self) -> "Writer": ...
    def __exit__(self, *args: object) -> bool: ...

def create_run_info(
    acquisition_id: str,
    acquisition_start_time: int = 0,
    adc_max: int = 2047,
    adc_min: int = -2048,
    experiment_name: str = "",
    flow_cell_id: str = "",
    flow_cell_product_code: str = "",
    protocol_name: str = "",
    protocol_run_id: str = "",
    protocol_start_time: int = 0,
    sample_id: str = "",
    sample_rate: int = 4000,
    sequencing_kit: str = "",
    sequencer_position: str = "",
    sequencer_position_type: str = "",
    software: str = "",
    system_name: str = "",
    system_type: str = "",
    context_tags: Optional[dict[str, str]] = None,
    tracking_id: Optional[dict[str, str]] = None,
) -> RunInfo: ...

# --- Signal processing (escapepod-signal) --------------------------------

class KmerTable:
    """A kmer level table loaded from a ``kmer\\tlevel`` file (gzip supported)."""

    @staticmethod
    def from_file(path: Union[str, PathLike[str]]) -> "KmerTable": ...
    @property
    def k(self) -> int: ...
    def get(self, kmer: str) -> float: ...
    def extract_levels(self, seq: str) -> npt.NDArray[np.float32]: ...

def mad_normalize(signal: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]: ...
def normalize_signal(signal: npt.NDArray[np.int16]) -> npt.NDArray[np.float32]: ...
def refine_signal_map(
    signal: npt.NDArray[np.float32],
    seq_to_signal_map: list[int],
    expected_levels: npt.NDArray[np.float32],
    half_bandwidth: int = 5,
    scale_iters: int = 2,
    dwell_target: float = 4.0,
    dwell_weight: float = 0.5,
) -> tuple[npt.NDArray[np.int64], float, float, float]: ...
