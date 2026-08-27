# CLI Reference

The `escpod` command-line tool provides utilities for working with POD5 files.

## Usage

```bash
escpod <COMMAND> [OPTIONS]
```

## Commands

These commands are in the default build — no extra Cargo features required.

| Command | Description |
|---------|-------------|
| [summary](summary.md) | Comprehensive file summary with QC metrics |
| [view](view.md) | Display reads as a table (including sidecar annotation columns) |
| [inspect](inspect.md) | Inspect file metadata, contents, and the `.p5s` sidecar |
| [merge](merge.md) | Combine multiple POD5 files |
| [filter](filter.md) | Extract reads by ID list, criteria, or sidecar annotation |
| [bam-filter](bam-filter.md) | Filter reads based on paired BAM file |
| [subset](subset.md) | Split reads into multiple files based on CSV mapping |
| [demux](demux.md) | Barcode demultiplexing — DTW-SVM, GBM, or CTC-CRF, end to end |
| [index](index-command.md) | Build the `.p5s` sidecar caches (read index + signal batch geometry) |
| [signal classify](signal-classify.md) | Read-level classification against a model bundle (tRNA charging) |

Additional commands — `repack`, `resquiggle`, and
[`annotate`](../experimental/annotate.md) — need `--features experimental`;
see the [Experimental](../experimental/index.md) section.

## Global Options

```
-q, --quiet          Errors only (also hides progress bars)
-v, --verbose        Increase log verbosity (-v debug, -vv trace)
    --fsync <MODE>   Output durability: none (default) | file | full
-h, --help           Print help information
-V, --version        Print version information
```

## Output Safety

Every output POD5 is written to a temporary file beside its destination and
renamed into place only once complete — an error, panic, or Ctrl-C leaves
the destination either untouched or absent, never truncated. `escpod` also
traps SIGINT/SIGTERM (what SLURM sends on `scancel` and at walltime) to
remove staging files on the way out.

Renaming alone doesn't make bytes durable against a machine crash: use
`--fsync file` to sync each output before the rename, or `--fsync full` to
also sync the directory. The default `--fsync none` is the right trade on
scratch filesystems where output is cheap to regenerate. If a run is killed
outright (`kill -9`, node failure), leftover staging files are identifiable
by prefix: `find <output-dir> -name '.escpod-tmp-*'`.

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

# 5. Demux into the .p5s sidecar, then materialize groups on demand
escpod demux combined.pod5 --model <bundle> --annotate
escpod filter combined.pod5 --annotation barcode=nbc05 -o nbc05.pod5
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
