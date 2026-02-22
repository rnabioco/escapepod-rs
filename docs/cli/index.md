# CLI Reference

The `escpod` command-line tool provides utilities for working with POD5 files.

## Usage

```bash
escpod <COMMAND> [OPTIONS]
```

## Commands

| Command | Description |
|---------|-------------|
| [summary](summary.md) | Comprehensive file summary with QC metrics |
| [view](view.md) | Display reads as a table |
| [inspect](inspect.md) | Inspect file metadata and contents |
| [merge](merge.md) | Combine multiple POD5 files |
| [filter](filter.md) | Extract reads by ID list |
| [bam-filter](bam-filter.md) | Filter reads based on paired BAM file |
| [repack](repack.md) | Repack POD5 files to optimize storage |
| [subset](subset.md) | Split reads into multiple files based on CSV mapping |
| [demux](demux.md) | Barcode demultiplexing (detect, fingerprint, classify, split, train) |
| [resquiggle](resquiggle.md) | Refine signal-to-base mapping using banded DP |

## Global Options

```
-h, --help     Print help information
-V, --version  Print version information
```

## Examples

### Basic Workflow

```bash
# 1. Inspect what's in your files
escpod inspect summary run1.pod5
escpod inspect summary run2.pod5

# 2. View the reads
escpod view run1.pod5

# 3. Merge files from a run
escpod merge -o combined.pod5 run1.pod5 run2.pod5

# 4. Extract interesting reads
escpod filter -i selected_reads.txt -o subset.pod5 combined.pod5
```

### Working with Multiple Files

Process all POD5 files in a directory:

```bash
# List all files
ls *.pod5

# Merge all files
escpod merge -o all_data.pod5 *.pod5
```

### Extracting Read IDs

To get a list of read IDs from a file:

```bash
escpod inspect reads experiment.pod5 > read_ids.txt
```

Then filter another file:

```bash
escpod filter -i read_ids.txt -o filtered.pod5 other_experiment.pod5
```
