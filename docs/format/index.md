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

## Data Tables

### Reads Table

One row per read containing:

| Field | Type | Description |
|-------|------|-------------|
| read_id | UUID | Unique identifier |
| channel | u16 | Channel number |
| well | u8 | Well (1-4) |
| read_number | u32 | Sequential number |
| start_sample | u64 | Start position |
| num_samples | u64 | Signal length |
| median_before | f32 | Pre-read current |
| calibration_offset | f32 | ADC offset |
| calibration_scale | f32 | ADC scale |
| end_reason | enum | Why read ended |
| signal | List<u64> | Signal row indices |
| run_info | u32 | Run info index |

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

## Arrow Extension Types

POD5 uses custom Arrow extension types:

| Type | Base Type | Description |
|------|-----------|-------------|
| `minknow.uuid` | FixedSizeBinary(16) | UUID storage |
| `minknow.vbz` | LargeBinary | VBZ compressed data |

## Version History

| Version | Changes |
|---------|---------|
| 0 | Initial format |
| 1 | Added open_pore_level field |
| 2 | Modified signal chunking |
| 3 | Added predicted_scaling fields |
| 4 | Current version |

## Comparison with FAST5

| Aspect | FAST5 | POD5 |
|--------|-------|------|
| Container | HDF5 | Arrow IPC |
| Compression | gzip/VBZ | VBZ only |
| Access pattern | Row-oriented | Columnar |
| Metadata | HDF5 attributes | FlatBuffer |
| File size | Baseline | ~30% smaller |
| Read speed | Slower | 2-10x faster |
