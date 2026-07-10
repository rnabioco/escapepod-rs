//! VBZ compression: SVB16 + ZSTD pipeline.
//!
//! VBZ is the default compression format for POD5 signal data.
//! It combines SVB16 (delta + zigzag + variable-length encoding)
//! with ZSTD compression at level 1.

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

    // Stage 1: ZSTD decompression
    let svb_encoded = zstd::decode_all(data)
        .map_err(|e| Error::Decompression(format!("ZSTD decompression failed: {}", e)))?;

    // Stage 2: SVB16 decoding
    svb16::decode(&svb_encoded, sample_count)
}

/// Decompress only the **first `max_samples`** of a VBZ chunk that holds
/// `total_samples`. Bit-identical to `decompress_signal(...)[..n]` where
/// `n = min(max_samples, total_samples)`.
///
/// The SVB16 layout is `[keys: ceil(total/8)][values]`, and a 1-byte vs 2-byte
/// value flag lives in the keys. So we ZSTD-*stream* just the keys, sum the
/// value bytes for the first `n` samples, stream that many more bytes, and stop
/// — skipping ZSTD work for the unread tail. For a long read (mRNA) where only
/// the first ~`max_obs_trace` samples feed the adapter detector, this avoids
/// decompressing the entire transcript.
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
    // Streaming a prefix out of ZSTD is slower *per byte* than one-shot
    // `decode_all` — it decodes block-by-block through a `read_exact` loop and
    // always inflates the full keys section (sized for `total_samples`). It only
    // pays off when the skipped tail is large. When the requested prefix is a
    // big fraction of the chunk, one-shot decode + truncate is faster overall:
    // measured on ~10k-sample reads, a ~40%-prefix stream lost to a full decode.
    // Gate at 1/4 — stream only when we can skip ≥75% of the samples — so mRNA
    // (tiny adapter window in a long transcript) still streams while short reads
    // fall back to the fast path. Both branches are bit-identical to
    // `decompress_signal(..)[..n]`.
    if n.saturating_mul(4) >= total_samples {
        let mut full = decompress_signal(data, total_samples)?;
        full.truncate(n);
        return Ok(full);
    }
    if data.is_empty() {
        return Err(Error::Decompression(
            "VBZ data is empty but sample_count > 0".to_string(),
        ));
    }

    let mut decoder = zstd::stream::read::Decoder::new(data)
        .map_err(|e| Error::Decompression(format!("ZSTD init failed: {}", e)))?;

    // Keys are sized for the chunk's *total* samples, then the value section.
    let keys_len = svb16::key_length(total_samples);
    let mut keys = vec![0u8; keys_len];
    decoder
        .read_exact(&mut keys)
        .map_err(|e| Error::Decompression(format!("ZSTD read (keys) failed: {}", e)))?;

    let values_len = svb16::value_bytes(&keys, n);
    let mut values = vec![0u8; values_len];
    decoder
        .read_exact(&mut values)
        .map_err(|e| Error::Decompression(format!("ZSTD read (values) failed: {}", e)))?;

    svb16::decode_prefix(&keys, &values, n)
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
