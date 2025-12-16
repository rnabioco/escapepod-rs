# CLI Reference

The `podfive` command-line tool provides utilities for working with POD5 files.

## Usage

```bash
podfive <COMMAND> [OPTIONS]
```

## Commands

| Command | Description |
|---------|-------------|
| [summary](summary.md) | Comprehensive file summary with QC metrics |
| [view](view.md) | Display reads as a table |
| [inspect](inspect.md) | Inspect file metadata and contents |
| [merge](merge.md) | Combine multiple POD5 files |
| [filter](filter.md) | Extract reads by ID |

## Global Options

```
-h, --help     Print help information
-V, --version  Print version information
```

## Examples

### Basic Workflow

```bash
# 1. Inspect what's in your files
podfive inspect summary run1.pod5
podfive inspect summary run2.pod5

# 2. View the reads
podfive view run1.pod5

# 3. Merge files from a run
podfive merge -o combined.pod5 run1.pod5 run2.pod5

# 4. Extract interesting reads
podfive filter -i selected_reads.txt -o subset.pod5 combined.pod5
```

### Working with Multiple Files

Process all POD5 files in a directory:

```bash
# List all files
ls *.pod5

# Merge all files
podfive merge -o all_data.pod5 *.pod5
```

### Extracting Read IDs

To get a list of read IDs from a file:

```bash
podfive inspect reads experiment.pod5 > read_ids.txt
```

Then filter another file:

```bash
podfive filter -i read_ids.txt -o filtered.pod5 other_experiment.pod5
```
