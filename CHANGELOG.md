# Changelog

## v0.1.0 (2026-03-19)

First stable release of escapepod-rs.

### Features

- **index**: `.p5i` sidecar read index for fast UUID lookup (`escpod index`), with zstd-compressed entry blocks, sorted-vec binary search, and file size checksum validation
- **filter**: Sample count and end reason filters, stdin support for read IDs, fast `reads_by_ids()` path for UUID-only filtering
- **subset**: Accelerated subsetting via indexed batch lookup
- **merge**: Parallel I/O optimization

### Fixes

- Include ZSTD content size in VBZ frames for Dorado/pod5 compatibility
- POD5 forward compatibility with Python pod5 library
- Correct pore count in summary table
