# escpod merge

Merge multiple POD5 files into a single output file.

![escpod merge](../images/merge.gif)

## Usage

```bash
escpod merge -o <OUTPUT> <INPUT>...
```

## Arguments

| Argument | Description |
|----------|-------------|
| `<INPUT>...` | One or more POD5 files to merge |

## Options

| Option | Description |
|--------|-------------|
| `-o, --output <FILE>` | Output file path (required) |
| `-t, --threads <N>` | Number of threads for parallel processing |
| `--profile` | Enable profiling output with timing and throughput |
| `-h, --help` | Print help |

## Description

The merge command combines multiple POD5 files into a single file. This is useful for:

- Combining files from different sequencing runs
- Consolidating files split across multiple output files
- Creating a single file for downstream analysis

### Behavior

- All reads from input files are copied to the output
- Run info entries are deduplicated by acquisition ID
- Signal data is re-compressed in the output file
- The output file uses default compression settings

## Examples

### Merge Two Files

```bash
escpod merge -o combined.pod5 run1.pod5 run2.pod5
```

### Merge All Files in Directory

```bash
escpod merge -o all_data.pod5 *.pod5
```

### Merge with Explicit File List

```bash
escpod merge -o output.pod5 \
    /data/run1/file1.pod5 \
    /data/run1/file2.pod5 \
    /data/run2/file1.pod5
```

## Output

The command prints progress information:

```
Merging 3 files into combined.pod5
  Processing: run1.pod5
    Added 5000 reads
  Processing: run2.pod5
    Added 4500 reads
  Processing: run3.pod5
    Added 5200 reads
Successfully merged 14700 reads into combined.pod5
```

## Notes

- Large files may take significant time to merge
- Ensure sufficient disk space for the output file
- The output file will be approximately the sum of input file sizes
