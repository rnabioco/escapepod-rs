# POD5 Format Overview

POD5 is the successor to FAST5 for storing Oxford Nanopore sequencing data. It uses Apache Arrow IPC (Feather V2) for efficient columnar storage with custom signal compression.

## Design Goals

- **Efficient access** - Columnar format enables reading specific fields without loading entire reads
- **Compact storage** - VBZ compression achieves 60-80% reduction in signal data size
- **Batch processing** - Data organized in batches for parallel processing
- **Self-describing** - Schema embedded in file, extensible metadata

## File Structure

A POD5 file contains:

1. **Signature** - Magic bytes identifying the file format
2. **Signal Table** - Compressed raw signal data (Arrow IPC)
3. **Run Info Table** - Sequencing run metadata (Arrow IPC)
4. **Reads Table** - Read records with references to signal (Arrow IPC)
5. **Footer** - FlatBuffer with table locations and metadata

See [Container Structure](container.md) for detailed layout.

escapepod additionally defines one companion artifact: the
[`.p5s` sidecar](sidecar.md), an Arrow file beside the POD5 holding a read
index and per-read annotations (demux barcodes, experimental designs). The
POD5 itself is never modified.

## Data Tables

### Reads Table

One row per read. Columns were added across format versions, so the version a
file declares tells you which of these to expect:

| Field | Type | Added | Description |
|-------|------|-------|-------------|
| read_id | UUID | V0 | Unique identifier |
| signal | List&lt;u64&gt; | V0 | Signal row indices |
| read_number | u32 | V0 | Sequential number |
| start | u64 | V0 | Start sample position |
| median_before | f32 | V0 | Pre-read current |
| num_minknow_events | u64 | V1 | Events MinKNOW counted |
| tracked_scaling_scale / _shift | f32 | V1 | Live scaling estimate |
| predicted_scaling_scale / _shift | f32 | V1 | Predicted scaling |
| num_reads_since_mux_change | u32 | V1 | Reads since the last mux change |
| time_since_mux_change | f32 | V1 | Seconds since the last mux change |
| num_samples | u64 | V2 | Signal length |
| channel | u16 (u32 in V6) | V3 | Channel number |
| well | u8 | V3 | Well (1-4) |
| pore_type | dict&lt;i16, utf8&gt; | V3 | Pore type |
| calibration_offset | f32 | V3 | ADC offset |
| calibration_scale | f32 | V3 | ADC scale |
| end_reason | dict&lt;i16, utf8&gt; | V3 | Why read ended |
| end_reason_forced | bool | V3 | Whether the end was forced |
| run_info | dict&lt;i16, utf8&gt; | V3 | Run info reference |
| open_pore_level | f32 | V4 | Open-pore current |
| expected_open_pore_level | f32 | V5 | Expected open-pore current |
| selected_read_level | f32 | V5 | Selected read level |

### Signal Table

Compressed signal chunks:

| Field | Type | Description |
|-------|------|-------------|
| signal | VBZ binary | Compressed signal data |
| samples | u32 | Number of samples |

### Run Info Table

One row per unique acquisition:

| Field | Type | Description |
|-------|------|-------------|
| acquisition_id | string | Unique run ID |
| acquisition_start_time | i64 | Start timestamp (ms) |
| sample_rate | u16 | Sampling rate (Hz) |
| adc_min | i16 | ADC minimum |
| adc_max | i16 | ADC maximum |
| context_tags | Map | Key-value metadata |
| tracking_id | Map | Tracking metadata |

## Signal Compression

Signal data uses the VBZ codec:

1. **Delta encoding** - Store differences between samples
2. **Zigzag encoding** - Map signed to unsigned integers
3. **SVB16** - Variable-length encoding (1-2 bytes per value)
4. **ZSTD** - Final compression

See [Compression](compression.md) for algorithm details.

## Signal Analysis Algorithms

For barcode demultiplexing and signal analysis, see [Segmentation Algorithms](segmentation.md):

- **LLR boundary detection** - Find adapter regions using log-likelihood ratio
- **T-test segmentation** - Extract fingerprints from signal segments
- **DTW classification** - Dynamic time warping for barcode matching

## Arrow Extension Types

POD5 uses custom Arrow extension types:

| Type | Base Type | Description |
|------|-----------|-------------|
| `minknow.uuid` | FixedSizeBinary(16) | UUID storage |
| `minknow.vbz` | LargeBinary | VBZ compressed data |

## Version History

| Version | Changes | escapepod |
|---------|---------|-----------|
| 0 | Initial format: `read_id`, `signal`, `read_number`, `start`, `median_before` | read |
| 1 | Scaling and mux-change fields, `num_minknow_events` | read |
| 2 | `num_samples` | read |
| 3 | `channel`, `well`, `pore_type`, calibration, `end_reason`, `run_info` | read |
| 4 | `open_pore_level` | read |
| 5 | `expected_open_pore_level`, `selected_read_level` | read + **written** |
| 6 | `channel` retyped `uint16` → `uint32` | read |

escapepod reads every version through 6 and deliberately **writes V5**. V6 is
not an additive change — it retypes an existing column — so a V6 file is not
merely missing fields to an older reader, it fails to parse. The newest `pod5`
on PyPI is still 0.3.44, which predates V6 and hard-rejects a `uint32` channel,
so emitting one today would produce files the rest of the ecosystem cannot
open. A channel number that does not fit `uint16` is an error rather than a
silent version bump.

## Comparison with FAST5

| Aspect | FAST5 | POD5 |
|--------|-------|------|
| Container | HDF5 | Arrow IPC |
| Compression | gzip/VBZ | VBZ only |
| Access pattern | Row-oriented | Columnar |
| Metadata | HDF5 attributes | FlatBuffer |
| File size | Baseline | ~30% smaller |
| Read speed | Slower | 2-10x faster |
