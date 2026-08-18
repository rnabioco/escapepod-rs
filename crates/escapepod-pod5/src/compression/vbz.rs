//! VBZ compression: SVB16 + ZSTD pipeline.
//!
//! VBZ is the default compression format for POD5 signal data.
//! It combines SVB16 (delta + zigzag + variable-length encoding)
//! with ZSTD compression at level 1.

use std::cell::RefCell;

use crate::compression::svb16;
use crate::error::{Error, Result};

/// Default ZSTD compression level for VBZ.
pub const ZSTD_LEVEL: i32 = 1;

/// Calculate the maximum compressed size for a given sample count.
/// This is a conservative upper bound.
pub fn max_compressed_size(sample_count: usize) -> usize {
    let svb_max = svb16::max_encoded_size(sample_count);
    // ZSTD can expand data slightly in worst case
    zstd::zstd_safe::compress_bound(svb_max)
}

/// Compress signal samples using VBZ (SVB16 + ZSTD).
///
/// # Arguments
/// * `samples` - The raw signal samples to compress
///
/// # Returns
/// The compressed data
pub fn compress_signal(samples: &[i16]) -> Result<Vec<u8>> {
    if samples.is_empty() {
        return Ok(Vec::new());
    }

    // Stage 1: SVB16 encoding
    let svb_encoded = svb16::encode(samples)?;

    // Stage 2: ZSTD compression
    // Use an encoder with include_contentsize and pledged source size so that
    // the ZSTD frame header contains the decompressed size. The ONT C++ VBZ
    // library (lib_pod5, Dorado) requires this field to be present.
    let mut encoder = zstd::Encoder::new(Vec::new(), ZSTD_LEVEL)
        .map_err(|e| Error::Compression(format!("ZSTD encoder init failed: {}", e)))?;
    encoder
        .include_contentsize(true)
        .map_err(|e| Error::Compression(format!("ZSTD set content size failed: {}", e)))?;
    encoder
        .set_pledged_src_size(Some(svb_encoded.len() as u64))
        .map_err(|e| Error::Compression(format!("ZSTD set pledged src size failed: {}", e)))?;
    std::io::copy(&mut svb_encoded.as_slice(), &mut encoder)
        .map_err(|e| Error::Compression(format!("ZSTD compression failed: {}", e)))?;
    let compressed = encoder
        .finish()
        .map_err(|e| Error::Compression(format!("ZSTD finish failed: {}", e)))?;

    Ok(compressed)
}

/// Largest buffer a thread keeps alive between `decompress_signal` calls.
///
/// Above this the call inflates into a buffer that dies with it, so one
/// pathological read cannot pin memory per rayon worker forever. 1 MiB covers
/// ~850k samples at the ~1.15 B/sample real data encodes to, while bounding a
/// 48-thread pool to ~48 MiB.
const MAX_RETAINED_SCRATCH: usize = 1 << 20;

/// Exact inflated size of a VBZ chunk, from the ZSTD frame header when it is
/// there and believable, else the `svb16` worst case (keys + 2 B/sample).
///
/// Both our writer and ONT's C++ VBZ library record the content size — the
/// latter *requires* it, which `test_zstd_frame_has_content_size` pins — so the
/// exact number is the normal case, and it is worth having: the worst case runs
/// ~1.75x over what real signal encodes to, which inflates every scratch buffer
/// and pushes the prefix gate around.
///
/// The lower-bound check is what makes trusting the header safe. SVB16 emits at
/// least 1 byte per sample, so a header claiming less than that describes
/// something other than this whole chunk — a truncated frame, or the first of
/// several — and we fall back rather than under-allocate.
fn inflated_size(data: &[u8], sample_count: usize) -> usize {
    let worst = svb16::max_encoded_size(sample_count);
    let least = svb16::key_length(sample_count) + sample_count;
    zstd::zstd_safe::get_frame_content_size(data)
        .ok()
        .flatten()
        .and_then(|n| usize::try_from(n).ok())
        .filter(|&n| (least..=worst).contains(&n))
        .unwrap_or(worst)
}

thread_local! {
    /// One ZSTD decompression context + output buffer per thread.
    ///
    /// `zstd::decode_all` builds a fresh `Decoder` per call — a `DCtx`, its
    /// window buffer, and a 32 KB `BufReader` — which at POD5 chunk sizes
    /// (~12 KB inflated for a 9.5k-sample read) is a large fraction of the
    /// decode. Reusing the context is worth 1.13x on tRNA-length reads and
    /// 1.49x on mRNA, for every caller of this function.
    ///
    /// Thread-local rather than pooled because signal decode fans out over
    /// rayon (`Reader::get_signal_bulk`) and `Decompressor` is `Send`, not
    /// `Sync`.
    static ZSTD_SCRATCH: RefCell<Option<(zstd::bulk::Decompressor<'static>, Vec<u8>)>> =
        const { RefCell::new(None) };
}

/// ZSTD-inflate `data` into `out`, reusing a per-thread decompression context.
///
/// `capacity` must be an upper bound on the inflated size; ZSTD's one-shot
/// decompress errors rather than growing the destination.
fn inflate_into(data: &[u8], capacity: usize, out: &mut Vec<u8>) -> Result<()> {
    out.clear();
    out.reserve(capacity);
    ZSTD_SCRATCH.with(|cell| {
        let mut slot = cell.borrow_mut();
        let (dctx, _) = slot.get_or_insert_with(|| {
            (
                zstd::bulk::Decompressor::new().expect("ZSTD decompressor init"),
                Vec::new(),
            )
        });
        dctx.decompress_to_buffer(data, out)
            .map_err(|e| Error::Decompression(format!("ZSTD decompression failed: {}", e)))
    })?;
    Ok(())
}

/// Run `f` over `data` inflated into the thread-local scratch buffer.
fn with_inflated<T>(data: &[u8], capacity: usize, f: impl FnOnce(&[u8]) -> Result<T>) -> Result<T> {
    if capacity > MAX_RETAINED_SCRATCH {
        // Too big to keep around; inflate into a buffer that dies with the call.
        let mut owned = Vec::new();
        inflate_into(data, capacity, &mut owned)?;
        return f(&owned);
    }
    // Move the buffer out of the cell for the duration of the call, so `f` can
    // borrow it with the cell itself un-borrowed. Keeps the scratch reusable
    // without making a nested decode on this thread a `RefCell` panic.
    let mut buf = ZSTD_SCRATCH
        .with(|cell| cell.borrow_mut().as_mut().map(|(_, b)| std::mem::take(b)))
        .unwrap_or_default();
    let result = inflate_into(data, capacity, &mut buf).and_then(|()| f(&buf));
    ZSTD_SCRATCH.with(|cell| {
        if let Some((_, slot)) = cell.borrow_mut().as_mut() {
            *slot = buf;
        }
    });
    result
}

/// Decompress VBZ-compressed signal data.
///
/// # Arguments
/// * `data` - The VBZ-compressed data
/// * `sample_count` - The expected number of samples
///
/// # Returns
/// The decompressed signal samples
pub fn decompress_signal(data: &[u8], sample_count: usize) -> Result<Vec<i16>> {
    if sample_count == 0 {
        return Ok(Vec::new());
    }

    if data.is_empty() {
        return Err(Error::Decompression(
            "VBZ data is empty but sample_count > 0".to_string(),
        ));
    }

    // Stage 1: ZSTD decompression. Stage 2: SVB16 decoding.
    with_inflated(data, inflated_size(data, sample_count), |svb_encoded| {
        svb16::decode(svb_encoded, sample_count)
    })
}

/// A ZSTD frame's largest block, uncompressed. The decoder inflates a whole
/// block into its window before emitting any of it, so this is the granularity
/// at which a prefix read can actually stop early.
const ZSTD_BLOCK_BYTES: usize = 128 * 1024;

/// Extra slack appended to a streamed value section so the SIMD SVB16 decoder's
/// 32-byte load guard stays satisfied to the end of the prefix. The bytes are
/// never read as data — only as load headroom — so their content is irrelevant.
const SIMD_LOAD_SLACK: usize = 32;

/// Decompress only the **first `max_samples`** of a VBZ chunk that holds
/// `total_samples`. Bit-identical to `decompress_signal(...)[..n]` where
/// `n = min(max_samples, total_samples)`.
///
/// The SVB16 layout is `[keys: ceil(total/8)][values]` with a 1-byte vs 2-byte
/// value flag per sample in the keys, so the *SVB16* stage can stop after `n`
/// samples for free. The ZSTD stage cannot: it inflates a whole 128 KiB block
/// into its window before emitting anything, so a chunk that fits in one block
/// — every read up to ~110k samples, which is all of tRNA and most of dRNA —
/// gets nothing from streaming, and pays an extra copy for trying. Measured on
/// 9.5k-sample reads, ZSTD is 79% of the decode and streaming a 10% prefix cost
/// 39.7 ms against 38.1 ms to inflate the lot.
///
/// So: stream only when at least one whole block can be skipped, which needs a
/// genuinely long read (mRNA), and take the one-shot path otherwise. Either way
/// only `n` samples go through SVB16. Both branches are bit-identical to
/// `decompress_signal(..)[..n]`.
pub fn decompress_signal_prefix(
    data: &[u8],
    total_samples: usize,
    max_samples: usize,
) -> Result<Vec<i16>> {
    use std::io::Read;

    let n = max_samples.min(total_samples);
    if n == 0 {
        return Ok(Vec::new());
    }
    if data.is_empty() {
        return Err(Error::Decompression(
            "VBZ data is empty but sample_count > 0".to_string(),
        ));
    }

    let keys_len = svb16::key_length(total_samples);
    let inflated = inflated_size(data, total_samples);
    // An upper bound on what the prefix needs (2 B/sample) against the chunk's
    // inflated size: stream only when a whole block is skippable however the
    // values happen to be sized.
    let streaming_pays = keys_len + 2 * n + ZSTD_BLOCK_BYTES <= inflated;

    if !streaming_pays {
        return with_inflated(data, inflated, |svb| {
            if svb.len() < keys_len {
                return Err(Error::Decompression(format!(
                    "SVB16 data too short: expected at least {} bytes for keys, got {}",
                    keys_len,
                    svb.len()
                )));
            }
            let (keys, values) = svb.split_at(keys_len);
            svb16::decode_split(keys, values, n)
        });
    }

    // `with_buffer` takes the `&[u8]` as its own `BufRead`, skipping the 32 KB
    // `BufReader` that `Decoder::new` would wrap around it.
    let mut decoder = zstd::stream::read::Decoder::with_buffer(data)
        .map_err(|e| Error::Decompression(format!("ZSTD init failed: {}", e)))?;

    // Keys are sized for the chunk's *total* samples, then the value section.
    let mut keys = vec![0u8; keys_len];
    decoder
        .read_exact(&mut keys)
        .map_err(|e| Error::Decompression(format!("ZSTD read (keys) failed: {}", e)))?;

    let values_len = svb16::value_bytes(&keys, n);
    let mut values = vec![0u8; values_len + SIMD_LOAD_SLACK];
    decoder
        .read_exact(&mut values[..values_len])
        .map_err(|e| Error::Decompression(format!("ZSTD read (values) failed: {}", e)))?;

    svb16::decode_split(&keys, &values, n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compress_decompress_empty() {
        let samples: Vec<i16> = vec![];
        let compressed = compress_signal(&samples).unwrap();
        assert!(compressed.is_empty());
        let decompressed = decompress_signal(&compressed, 0).unwrap();
        assert!(decompressed.is_empty());
    }

    #[test]
    fn test_compress_decompress_simple() {
        let samples: Vec<i16> = (0..1000).map(|i| (i % 256) as i16).collect();
        let compressed = compress_signal(&samples).unwrap();
        let decompressed = decompress_signal(&compressed, samples.len()).unwrap();
        assert_eq!(decompressed, samples);
    }

    #[test]
    fn test_decompress_prefix_matches_full() {
        // Deterministic signal with a mix of small (1-byte) and large (2-byte)
        // deltas so the key bits vary across the prefix boundary.
        let mut s: u64 = 0x1234_5678_9abc_def1;
        let samples: Vec<i16> = (0..5000)
            .map(|i| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                // alternate small ramps and big jumps
                if i % 7 == 0 {
                    (s >> 48) as i16 // big jump -> 2-byte delta
                } else {
                    (i as i16 % 5) - 2 // small -> 1-byte delta
                }
            })
            .collect();
        let total = samples.len();
        let compressed = compress_signal(&samples).unwrap();
        let full = decompress_signal(&compressed, total).unwrap();

        for &n in &[
            0usize,
            1,
            2,
            6,
            7,
            8,
            99,
            100,
            4096,
            total - 1,
            total,
            total + 10,
        ] {
            let pref = decompress_signal_prefix(&compressed, total, n).unwrap();
            let want = &full[..n.min(total)];
            assert_eq!(pref.as_slice(), want, "prefix mismatch at n={n}");
        }
    }

    /// The 5000-sample case above only ever takes the one-shot branch — the
    /// gate needs a chunk that spans several ZSTD blocks before streaming can
    /// skip one. Pin the *other* branch on a read long enough to trip it, and
    /// assert the gate really did choose it, so a future retune that quietly
    /// disables streaming altogether shows up as a failing test rather than as
    /// a silent perf regression on long reads.
    #[test]
    fn test_decompress_prefix_streaming_branch() {
        let mut s: u64 = 0xdead_beef_0bad_f00d;
        let total = 400_000;
        let samples: Vec<i16> = (0..total)
            .map(|i| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                if i % 13 == 0 {
                    (s >> 48) as i16
                } else {
                    (i as i16 % 9) - 4
                }
            })
            .collect();
        let compressed = compress_signal(&samples).unwrap();
        let full = decompress_signal(&compressed, total).unwrap();

        let keys_len = svb16::key_length(total);
        let inflated = inflated_size(&compressed, total);
        let streams = |n: usize| keys_len + 2 * n + ZSTD_BLOCK_BYTES <= inflated;

        for &n in &[1usize, 8, 12_345, 20_000, 100_000] {
            assert!(streams(n), "n={n} does not exercise the streaming branch");
            let pref = decompress_signal_prefix(&compressed, total, n).unwrap();
            assert_eq!(pref.as_slice(), &full[..n], "prefix mismatch at n={n}");
        }

        // Just past the gate, the one-shot branch must give the same answer.
        let n = (inflated - keys_len - ZSTD_BLOCK_BYTES) / 2 + 1;
        assert!(!streams(n) && n < total);
        assert_eq!(
            decompress_signal_prefix(&compressed, total, n).unwrap(),
            full[..n]
        );
    }

    /// The frame header is trusted for both the scratch size and the prefix
    /// gate, so its bounds check has to hold: our own frames must report a size
    /// inside the SVB16 envelope, and a header-less frame must fall back rather
    /// than under-allocate.
    #[test]
    fn test_inflated_size_uses_frame_header_within_bounds() {
        let samples: Vec<i16> = (0..10_000).map(|i| ((i * 37) % 251) as i16).collect();
        let total = samples.len();
        let compressed = compress_signal(&samples).unwrap();

        let exact = inflated_size(&compressed, total);
        assert_eq!(exact, svb16::encode(&samples).unwrap().len());
        assert!(
            exact < svb16::max_encoded_size(total),
            "header not consulted"
        );

        // A frame written without a content size falls back to the worst case.
        let headerless = zstd::encode_all(svb16::encode(&samples).unwrap().as_slice(), 1).unwrap();
        assert_eq!(
            inflated_size(&headerless, total),
            svb16::max_encoded_size(total)
        );
        // ...and still decodes, since the fallback is an upper bound.
        assert_eq!(decompress_signal(&headerless, total).unwrap(), samples);
    }

    #[test]
    fn test_zstd_frame_has_content_size() {
        // The ONT VBZ C++ library requires the ZSTD frame header to include
        // the content size. Verify our compressed output has this set.
        let samples: Vec<i16> = (0..100).collect();
        let compressed = compress_signal(&samples).unwrap();
        // ZSTD magic: 28 B5 2F FD
        assert_eq!(&compressed[..4], &[0x28, 0xb5, 0x2f, 0xfd]);
        // Frame_Header_Descriptor byte (byte 4):
        // Bit 5 = Single_Segment_flag — when set, content size is implicit
        // Bits 7-6 = Frame_Content_Size_flag — when non-zero, explicit content size
        let desc = compressed[4];
        let single_segment = (desc >> 5) & 1 == 1;
        let fcs_flag = desc >> 6;
        assert!(
            single_segment || fcs_flag > 0,
            "ZSTD frame must include content size (desc=0x{desc:02x})"
        );
    }

    #[test]
    fn test_compress_decompress_realistic() {
        // Simulate realistic nanopore signal: fluctuating around a baseline
        let mut samples = Vec::with_capacity(10000);
        let mut value: i16 = 500;
        for i in 0..10000 {
            // Add some noise and occasional jumps
            let noise = ((i * 7) % 20) as i16 - 10;
            if i % 500 == 0 {
                value = 400 + ((i / 500) % 3) as i16 * 100;
            }
            samples.push(value + noise);
        }

        let compressed = compress_signal(&samples).unwrap();
        let decompressed = decompress_signal(&compressed, samples.len()).unwrap();
        assert_eq!(decompressed, samples);

        // VBZ should achieve reasonable compression
        let original_size = samples.len() * 2;
        println!(
            "Compression ratio: {:.2}x ({} -> {} bytes)",
            original_size as f64 / compressed.len() as f64,
            original_size,
            compressed.len()
        );
    }

    /// Port of the upstream python `test_signal_tools.test_round_trip_chunked`
    /// (+ its `_empty` sibling): a signal split into arbitrarily sized chunks,
    /// each compressed independently, must decompress and concatenate back to
    /// the original, with the chunk lengths summing to the sample count. This is
    /// exactly the invariant the file writer relies on when it splits a read at
    /// `max_signal_chunk_size` and the reader concatenates the rows.
    #[test]
    fn test_round_trip_chunked() {
        // Full int16-range deterministic signal (xorshift64).
        let mut s: u64 = 0x9E37_79B9_7F4A_7C15;
        let signal: Vec<i16> = (0..12_345)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s >> 48) as i16
            })
            .collect();

        // Chunk sizes spanning tiny, mid, exact-length and over-length; 0-length
        // input is covered by `test_compress_decompress_empty`.
        for &chunk_size in &[1usize, 7, 250, 999, 12_345, 20_000] {
            let mut lengths = Vec::new();
            let mut roundtrip = Vec::new();
            for chunk in signal.chunks(chunk_size) {
                let compressed = compress_signal(chunk).unwrap();
                lengths.push(chunk.len());
                roundtrip.extend(decompress_signal(&compressed, chunk.len()).unwrap());
            }
            assert_eq!(lengths.iter().sum::<usize>(), signal.len());
            assert_eq!(roundtrip, signal, "chunk_size={chunk_size}");
        }
    }

    #[test]
    fn test_compress_decompress_extreme_values() {
        let samples = vec![
            i16::MIN,
            i16::MAX,
            0,
            -1,
            1,
            i16::MIN,
            i16::MAX,
            -32000,
            32000,
        ];
        let compressed = compress_signal(&samples).unwrap();
        let decompressed = decompress_signal(&compressed, samples.len()).unwrap();
        assert_eq!(decompressed, samples);
    }
}
