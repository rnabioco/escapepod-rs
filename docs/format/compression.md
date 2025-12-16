# Signal Compression (VBZ)

POD5 uses the VBZ codec for signal compression, combining delta encoding, zigzag encoding, StreamVByte (SVB16), and ZSTD.

## Compression Pipeline

```
Raw Signal (i16)
      │
      ▼
┌─────────────┐
│   Delta     │  Δ[i] = signal[i] - signal[i-1]
│  Encoding   │
└─────────────┘
      │
      ▼
┌─────────────┐
│   Zigzag    │  Map signed → unsigned
│  Encoding   │
└─────────────┘
      │
      ▼
┌─────────────┐
│   SVB16     │  Variable-length (1-2 bytes)
│  Encoding   │
└─────────────┘
      │
      ▼
┌─────────────┐
│    ZSTD     │  Final compression
│  Level 1    │
└─────────────┘
      │
      ▼
Compressed Data
```

## Delta Encoding

Consecutive signal samples are highly correlated. Delta encoding stores differences:

```
Original:  [100, 102, 105, 103, 101]
Delta:     [100,   2,   3,  -2,  -2]
```

Benefits:
- Differences are typically small
- Smaller values compress better
- Preserves signal fidelity exactly

## Zigzag Encoding

Maps signed integers to unsigned for efficient variable-length encoding:

```rust
fn zigzag_encode(val: i16) -> u16 {
    ((val << 1) ^ (val >> 15)) as u16
}

fn zigzag_decode(val: u16) -> i16 {
    ((val >> 1) as i16) ^ (-((val & 1) as i16))
}
```

Mapping examples:
```
 0 →  0
-1 →  1
 1 →  2
-2 →  3
 2 →  4
...
```

This keeps small magnitudes (positive or negative) as small unsigned values.

## SVB16 (StreamVByte for 16-bit)

Variable-length encoding where each value uses 1 or 2 bytes:

### Structure

```
┌─────────────────┬─────────────────┐
│   Keys Bitmap   │   Data Section  │
│  (1 bit/value)  │  (1-2 bytes/val)│
└─────────────────┴─────────────────┘
```

### Keys Bitmap

- 1 bit per sample, packed 8 per byte
- Bit = 0: value fits in 1 byte (0-255)
- Bit = 1: value needs 2 bytes (256-65535)

```
Keys length = (sample_count + 7) / 8 bytes
```

### Data Section

Values stored sequentially:
- 1-byte values: stored as-is
- 2-byte values: stored as little-endian u16

### Example

```
Values:    [5, 300, 12, 1000, 8]
Keys:      [0,   1,  0,    1, 0] → 0b01010 → 0x0A (padded)
Data:      [05, 2C 01, 0C, E8 03, 08]
           1b   2b    1b    2b   1b
```

### Encoding Algorithm

```rust
fn svb16_encode(values: &[u16]) -> Vec<u8> {
    let key_len = (values.len() + 7) / 8;
    let mut keys = vec![0u8; key_len];
    let mut data = Vec::new();

    for (i, &val) in values.iter().enumerate() {
        if val > 255 {
            keys[i / 8] |= 1 << (i % 8);
            data.push(val as u8);
            data.push((val >> 8) as u8);
        } else {
            data.push(val as u8);
        }
    }

    let mut result = keys;
    result.extend(data);
    result
}
```

### Decoding Algorithm

```rust
fn svb16_decode(data: &[u8], count: usize) -> Vec<u16> {
    let key_len = (count + 7) / 8;
    let keys = &data[..key_len];
    let mut data_pos = key_len;
    let mut result = Vec::with_capacity(count);

    for i in 0..count {
        let is_two_byte = (keys[i / 8] >> (i % 8)) & 1 == 1;
        if is_two_byte {
            let val = u16::from_le_bytes([data[data_pos], data[data_pos + 1]]);
            result.push(val);
            data_pos += 2;
        } else {
            result.push(data[data_pos] as u16);
            data_pos += 1;
        }
    }

    result
}
```

## ZSTD Compression

Final pass uses ZSTD at level 1:

- Fast compression/decompression
- Good compression ratio
- Streaming support
- Standard library (libzstd)

Level 1 chosen for balance:
- Level 0: No compression
- Level 1: Fast, ~60% ratio
- Level 3: Default, ~65% ratio
- Level 19+: Slow, ~70% ratio

For POD5, level 1 is sufficient as SVB16 already reduces entropy.

## Compression Ratios

Typical compression ratios (compressed/original):

| Stage | Ratio | Cumulative |
|-------|-------|------------|
| Delta | ~1.0x | ~1.0x |
| Zigzag | ~1.0x | ~1.0x |
| SVB16 | ~0.6x | ~0.6x |
| ZSTD | ~0.5x | ~0.3x |

Final result: **30-40% of original size**

## Decompression

Reverse the pipeline:

```rust
pub fn decompress_signal(data: &[u8], sample_count: usize) -> Result<Vec<i16>> {
    // 1. ZSTD decompress
    let svb_data = zstd::decode_all(data)?;

    // 2. SVB16 decode
    let zigzag_values = svb16_decode(&svb_data, sample_count);

    // 3. Zigzag decode
    let deltas: Vec<i16> = zigzag_values.iter()
        .map(|&v| zigzag_decode(v))
        .collect();

    // 4. Delta decode (cumulative sum)
    let mut signal = Vec::with_capacity(sample_count);
    let mut acc: i16 = 0;
    for delta in deltas {
        acc = acc.wrapping_add(delta);
        signal.push(acc);
    }

    Ok(signal)
}
```

## Extension Type

VBZ data uses the Arrow extension type `minknow.vbz`:

```json
{
    "extension_name": "minknow.vbz",
    "storage_type": "LargeBinary",
    "metadata": {
        "codec": "vbz",
        "version": 1
    }
}
```
