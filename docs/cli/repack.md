# escpod repack

Repack POD5 files to optimize storage and apply current compression settings.

![escpod repack](../images/repack.gif)

## Usage

```bash
escpod repack -o <OUTPUT_DIR> [OPTIONS] <FILES>...
```

## Arguments

| Argument | Description |
|----------|-------------|
| `<FILES>...` | Input POD5 files to repack |

## Options

| Option | Description |
|--------|-------------|
| `-o, --output-dir <DIR>` | Output directory (required) |
| `-f, --force` | Overwrite existing files |
| `-h, --help` | Print help |

## Examples

### Repack a Single File

```bash
escpod repack input.pod5 -o repacked/
```

### Repack Multiple Files

```bash
escpod repack *.pod5 -o repacked/
```

### Overwrite Existing Files

```bash
escpod repack *.pod5 -o repacked/ --force
```

### In-Place Repacking

You can safely repack files in place (output to the same directory as input). The command uses temporary files to prevent data corruption:

```bash
escpod repack data/*.pod5 -o data/ --force
```

## Output

The command prints progress and summary:

```
Repacking 5 file(s) to repacked/
Repacking [████████████████████████████████████████] 5/5
Repacked 50000 reads across 5 file(s)
```

## Notes

- Output files retain the same names as input files
- Signal data is decompressed and re-compressed during repacking
- Run info and all metadata is preserved
- Safe for in-place repacking (uses temporary files)
