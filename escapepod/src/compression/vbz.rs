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
