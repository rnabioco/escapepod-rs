# escapepod view

Display reads from a POD5 file as a formatted table.

![escapepod view](../images/view.gif)

## Usage

```bash
escapepod view [OPTIONS] <INPUT>
```

## Arguments

| Argument | Description |
|----------|-------------|
| `<INPUT>` | Path to the POD5 file |

## Options

| Option | Description |
|--------|-------------|
| `--include <FIELDS>` | Comma-separated list of fields to include |
| `--exclude <FIELDS>` | Comma-separated list of fields to exclude |
| `--ids` | Only show read IDs |
| `--separator <SEP>` | Field separator (default: tab) |
| `--no-header` | Suppress header row output |
| `-o, --output <FILE>` | Write output to file instead of stdout |
| `-h, --help` | Print help |

## Output Fields

The default output includes:

| Field | Description |
|-------|-------------|
| `read_id` | Unique identifier (UUID) |
| `channel` | Channel number |
| `well` | Well number (1-4) |
| `read_number` | Sequential read number |
| `start` | Start sample position |
| `num_samples` | Total signal samples |
| `median_before` | Median current before read |
| `end_reason` | Why the read ended |

## Examples

### Basic Usage

```bash
escapepod view experiment.pod5
```

Output:
```
read_id                               channel  well  read_number  num_samples
a1b2c3d4-e5f6-7890-abcd-ef1234567890  1        1     1            50000
b2c3d4e5-f6a7-8901-bcde-f12345678901  1        1     2            75000
c3d4e5f6-a7b8-9012-cdef-123456789012  2        1     1            62000
...
```

### Show Only Read IDs

```bash
escapepod view --ids experiment.pod5
```

Output:
```
a1b2c3d4-e5f6-7890-abcd-ef1234567890
b2c3d4e5-f6a7-8901-bcde-f12345678901
c3d4e5f6-a7b8-9012-cdef-123456789012
```

### Save to File

```bash
escapepod view experiment.pod5 -o reads.tsv
```

### Select Specific Fields

```bash
escapepod view --include read_id,channel,num_samples experiment.pod5
```

### CSV Output

```bash
# Use comma separator for CSV format
escapepod view --separator ',' experiment.pod5 -o reads.csv

# Without header for piping
escapepod view --ids --no-header experiment.pod5 | head -n 10
```
