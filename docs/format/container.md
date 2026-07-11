# Container Structure

POD5 files use a custom container format with Arrow IPC tables and a FlatBuffer footer.

## File Layout

```
┌─────────────────────────────────────┐
│ Signature (8 bytes)                 │  "\213POD\r\n\032\n"
├─────────────────────────────────────┤
│ Section Marker (16 bytes)           │  UUID
├─────────────────────────────────────┤
│ Signal Table                        │  Arrow IPC stream
│   - Schema message                  │
│   - RecordBatch messages (×N)       │
├─────────────────────────────────────┤
│ Section Marker (16 bytes)           │
├─────────────────────────────────────┤
│ Run Info Table                      │  Arrow IPC stream
│   - Schema message                  │
│   - RecordBatch messages (×N)       │
├─────────────────────────────────────┤
│ Section Marker (16 bytes)           │
├─────────────────────────────────────┤
│ Reads Table                         │  Arrow IPC stream
│   - Schema message                  │
│   - RecordBatch messages (×N)       │
├─────────────────────────────────────┤
│ Section Marker (16 bytes)           │
├─────────────────────────────────────┤
│ Footer Marker (8 bytes)             │  "FOOTER\0\0"
├─────────────────────────────────────┤
│ Footer (FlatBuffer)                 │  Table locations, metadata
├─────────────────────────────────────┤
│ Footer Length (8 bytes)             │  Little-endian i64
├─────────────────────────────────────┤
│ Section Marker (16 bytes)           │
├─────────────────────────────────────┤
│ Signature (8 bytes)                 │  "\213POD\r\n\032\n"
└─────────────────────────────────────┘
```

## Signature

The file starts and ends with an 8-byte signature:

```
Hex:    8B 50 4F 44 0D 0A 1A 0A
ASCII:  \213 P  O  D  \r \n \032 \n
```

This signature:
- Starts with a non-ASCII byte (0x8B) to detect binary/text confusion
- Contains `POD` for identification
- Uses `\r\n` to detect line-ending corruption
- Ends with `\n` as final newline

## Section Markers

16-byte UUID markers separate major sections. These provide:
- Clear section boundaries
- Corruption detection
- Forward compatibility

## Arrow IPC Tables

Each table is stored as an Arrow IPC stream:

1. **Schema message** - Column names and types
2. **RecordBatch messages** - Data in columnar format

Tables can contain multiple batches for large files.

## Footer Structure

The footer is a FlatBuffer containing:

```flatbuffers
table Footer {
    file_identifier: string;      // "POD5"
    software: string;             // Writer software name
    section_marker: [ubyte];      // 16-byte UUID

    contents: EmbeddedFile;       // File manifest
    reads: EmbeddedFile;          // Reads table location
    run_info: EmbeddedFile;       // Run info location
    signal: EmbeddedFile;         // Signal location
}

table EmbeddedFile {
    offset: long;                 // Byte offset from start
    length: long;                 // Length in bytes
}
```

## Reading Algorithm

1. Read last 8 bytes to verify signature
2. Read footer length (8 bytes before final signature)
3. Parse footer FlatBuffer
4. Use offsets to locate each table
5. Parse Arrow IPC streams on demand

```rust linenums="1"
// Pseudocode
let file_len = file.metadata().len();
let sig_end = &file[file_len - 8..];
assert!(sig_end == SIGNATURE);

let footer_len = read_i64(&file[file_len - 8 - 16 - 8..]);
let footer_start = file_len - 8 - 16 - 8 - footer_len;
let footer = parse_footer(&file[footer_start..]);

// Access tables via footer offsets
let reads_data = &file[footer.reads.offset..][..footer.reads.length];
```

## Memory Mapping

For large files, memory mapping is recommended:

- File is mapped to virtual memory
- Only accessed pages are loaded
- OS handles caching and eviction
- Multiple processes can share mapped pages

## Batch Organization

Data is organized in batches for:

- **Parallel processing** - Each batch processed independently
- **Memory efficiency** - Load one batch at a time
- **Streaming writes** - Flush batches incrementally

Typical batch sizes:
- Reads: 1,000-10,000 reads per batch
- Signal: 100-1,000 chunks per batch

## Integrity Checks

The format provides several integrity checks:

1. **Signature validation** - File starts/ends with magic bytes
2. **Section markers** - UUIDs at section boundaries
3. **Footer checksum** - Optional CRC in footer
4. **Arrow validation** - Schema consistency checks
