# escpod inspect

Inspect POD5 file metadata and contents.

![escpod inspect](../images/inspect.gif)

## Usage

```bash
escpod inspect <SUBCOMMAND> <INPUT>
```

## Subcommands

| Subcommand | Description |
|------------|-------------|
| `summary` | Show file summary statistics |
| `reads` | List all read IDs |
| `read` | Show details for a specific read |

## escpod inspect summary

Display summary information about a POD5 file.

### Usage

```bash
escpod inspect summary <INPUT>
```

### Example

```bash
escpod inspect summary experiment.pod5
```

Output:
```
File: experiment.pod5
File ID: 3c490d31-0385-4a49-a61e-d717e49c34ac
POD5 version: 0.3.44
Software: MinKNOW 24.x

Reads: 10,543
Read batches: 11

Run info entries: 1
  [0] acquisition_id: abc123def456
      sample_rate: 4000 Hz
      flow_cell_id: FAK12345

Sidecar: experiment.pod5.p5s
  index: 10,543 reads
  annotations: barcode (17 labels, 10,543 reads), condition (16 labels, 9,867 reads)
  design: [barcode] → [condition,replicate], 16 rows
```

The **Sidecar** block reports the [`.p5s` companion file](../format/sidecar.md)
when one exists: the read-index size, each annotation with its label and
assigned-read counts, and the experimental design. Files without a sidecar
print `Sidecar: none`; a sidecar that doesn't match the POD5 (stale, or
copied from another file) is reported inline rather than failing the
command.

`inspect summary` also accepts a directory, printing per-file lines plus
`Total reads:` / `Total batches:`.

## escpod inspect reads

List all read IDs in the file.

### Usage

```bash
escpod inspect reads <INPUT>
```

### Example

```bash
escpod inspect reads experiment.pod5
```

Output:
```
a1b2c3d4-e5f6-7890-abcd-ef1234567890
b2c3d4e5-f6a7-8901-bcde-f12345678901
c3d4e5f6-a7b8-9012-cdef-123456789012
...
```

This is useful for creating filter lists:

```bash
escpod inspect reads experiment.pod5 > all_reads.txt
```

## escpod inspect read

Show detailed information about a specific read.

### Usage

```bash
escpod inspect read <INPUT> <READ_ID>
```

### Arguments

| Argument | Description |
|----------|-------------|
| `<INPUT>` | Path to the POD5 file |
| `<READ_ID>` | UUID of the read to inspect |

### Example

```bash
escpod inspect read experiment.pod5 a1b2c3d4-e5f6-7890-abcd-ef1234567890
```

Output:
```
Read: a1b2c3d4-e5f6-7890-abcd-ef1234567890
  Channel: 1
  Well: 1
  Read number: 42
  Start sample: 1234567
  Num samples: 50000
  Median before: 210.5 pA
  End reason: signal_positive
  Calibration:
    Offset: -240.5
    Scale: 0.145
  Run info: abc123-def456
```
