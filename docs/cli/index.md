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
| [view](view.md) | Display reads as a table |
| [inspect](inspect.md) | Inspect file metadata and contents |
| [merge](merge.md) | Combine multiple POD5 files |
| [filter](filter.md) | Extract reads by ID list |
| [bam-filter](bam-filter.md) | Filter reads based on paired BAM file |
| [subset](subset.md) | Split reads into multiple files based on CSV mapping |

Additional commands (`repack`, `resquiggle`, `demux`, `index`) live behind
Cargo feature gates — see the [Experimental](../experimental/index.md)
section.

## Global Options

```
-h, --help     Print help information
-V, --version  Print version information
```

## Reading from object storage

Built with the `remote` Cargo feature, `escpod summary`, `escpod view`, and
`escpod inspect` accept a URL wherever they accept a path:

```bash
cargo install escapepod-cli --features remote

escpod inspect summary s3://my-bucket/run1.pod5
escpod view https://example.org/data/run1.pod5
```

Supported schemes are `s3://`, `gs://`, `az://`, and `http(s)://`. Reads are
lazy: opening transfers only the file tail and footer, and the command then
fetches just the reads table. Inspecting a multi-GB object costs a few MB of
range requests rather than a full download — on the bundled 1.7 MB test file,
`inspect summary`, `inspect reads`, and `view` each transfer 98 KB (5.5%) in
four requests, and never touch the signal table.

Credentials come from the standard environment chain:

| Variable | Purpose |
|---|---|
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN` | S3 credentials |
| `AWS_REGION` | S3 region |
| `AWS_ENDPOINT` | S3-compatible endpoint (MinIO, Ceph) |
| `AWS_ALLOW_HTTP=true` | Permit a cleartext S3 endpoint |
| `GOOGLE_SERVICE_ACCOUNT` | GCS service-account key path |
| `AZURE_STORAGE_ACCOUNT_NAME`, `AZURE_STORAGE_ACCOUNT_KEY` | Azure account and credentials |

`AZURE_STORAGE_ACCOUNT_NAME` is required for `az://` — unlike an S3 or GCS
URL, `az://container/path` carries no account name.

!!! warning "Metadata commands only"

    Signal is still fetched a whole table at a time, so signal-heavy commands
    (`demux`, `resquiggle`, `repack`, `merge`, `filter`) would pull essentially
    the entire object. Download the file first for those. Writing to a remote
    destination is not supported — remote access is read-only.

A binary built *without* the `remote` feature rejects a URL with an explanatory
error rather than reporting it as a missing path.

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
