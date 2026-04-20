//! SVB16 (StreamVByte for 16-bit) encoding and decoding.
//!
//! SVB16 is a variable-length encoding for 16-bit integers that uses:
//! - Delta encoding: store differences between consecutive samples
//! - Zigzag encoding: map signed integers to unsigned for better compression
//! - Variable-length encoding: 1 or 2 bytes per value based on magnitude
//!
//! Format:
//! - Keys section: 1 bit per sample (0 = 1-byte value, 1 = 2-byte value)
//! - Data section: variable-length encoded values
//!
//! Decoding runtime-dispatches to an SSSE3 implementation on x86_64 CPUs
//! that advertise the feature; otherwise the scalar implementation below
//! is used.

mod tables;

#[cfg(target_arch = "x86_64")]
mod simd_ssse3;

use crate::error::{Error, Result};

/// Calculate the key length in bytes for a given sample count.
/// Keys use 1 bit per sample, packed into bytes.
#[inline]
pub fn key_length(sample_count: usize) -> usize {
    sample_count.div_ceil(8)
}

/// Calculate the maximum encoded size for a given sample count.
/// Worst case: all 2-byte values.
#[inline]
pub fn max_encoded_size(sample_count: usize) -> usize {
    key_length(sample_count) + sample_count * 2
}

/// Zigzag encode a 16-bit value.
/// Maps signed integers to unsigned: 0 -> 0, -1 -> 1, 1 -> 2, -2 -> 3, etc.
#[inline]
fn zigzag_encode(val: u16) -> u16 {
    (val.wrapping_shl(1)) ^ ((val as i16).wrapping_shr(15) as u16)
}

/// Zigzag decode a 16-bit value.
#[inline]
fn zigzag_decode(val: u16) -> u16 {
    (val >> 1) ^ (val & 1).wrapping_neg()
}

/// Encode samples using SVB16 with delta and zigzag encoding.
///
/// Returns the encoded bytes.
pub fn encode(samples: &[i16]) -> Result<Vec<u8>> {
    if samples.is_empty() {
        return Ok(Vec::new());
    }

    let keys_len = key_length(samples.len());
    let max_size = max_encoded_size(samples.len());
    let mut output = vec![0u8; max_size];

    let (keys, data) = output.split_at_mut(keys_len);

    let mut data_offset = 0;
    let mut prev: u16 = 0;

    for (i, &sample) in samples.iter().enumerate() {
        let key_byte_idx = i / 8;
        let key_bit_idx = i % 8;

        // Delta encode (wrapping arithmetic in unsigned space)
        let current = sample as u16;
        let delta = current.wrapping_sub(prev);
        prev = current;

        // Zigzag encode
        let value = zigzag_encode(delta);

        if value < 256 {
            // 1-byte value, key bit stays 0
            data[data_offset] = value as u8;
            data_offset += 1;
        } else {
            // 2-byte value, set key bit to 1
            keys[key_byte_idx] |= 1 << key_bit_idx;
            data[data_offset] = value as u8;
            data[data_offset + 1] = (value >> 8) as u8;
            data_offset += 2;
        }
    }

    // Truncate to actual size
    output.truncate(keys_len + data_offset);
    Ok(output)
}

/// Decode SVB16-encoded data back to samples.
///
/// Runtime-dispatches to an SSSE3 implementation on capable x86_64 CPUs;
/// otherwise uses the scalar fallback. The two paths are verified equivalent
/// by the tests in `simd_ssse3`.
///
/// # Arguments
/// * `data` - The encoded data (keys + variable-length values)
/// * `sample_count` - The expected number of samples
pub fn decode(data: &[u8], sample_count: usize) -> Result<Vec<i16>> {
    if sample_count == 0 {
        return Ok(Vec::new());
    }

    let keys_len = key_length(sample_count);
    if data.len() < keys_len {
        return Err(Error::Decompression(format!(
            "SVB16 data too short: expected at least {} bytes for keys, got {}",
            keys_len,
            data.len()
        )));
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("ssse3") {
            let (keys, values) = data.split_at(keys_len);
            let mut out: Vec<i16> = Vec::with_capacity(sample_count);
            // SAFETY: is_x86_feature_detected verified at runtime.
            unsafe {
                simd_ssse3::decode(keys, values, sample_count, &mut out)
                    .map_err(|e| Error::Decompression(e.to_string()))?;
            }
            return Ok(out);
        }
    }

    decode_scalar(data, sample_count)
}

/// Scalar SVB16 decode. Always available; serves as the fallback and the
/// reference implementation the SIMD path is tested against.
pub fn decode_scalar(data: &[u8], sample_count: usize) -> Result<Vec<i16>> {
    if sample_count == 0 {
        return Ok(Vec::new());
    }

    let keys_len = key_length(sample_count);
    if data.len() < keys_len {
        return Err(Error::Decompression(format!(
            "SVB16 data too short: expected at least {} bytes for keys, got {}",
            keys_len,
            data.len()
        )));
    }

    let (keys, values) = data.split_at(keys_len);
    let mut samples = Vec::with_capacity(sample_count);
    let mut data_offset = 0;
    let mut prev: u16 = 0;

    for i in 0..sample_count {
        let key_byte_idx = i / 8;
        let key_bit_idx = i % 8;
        let is_two_bytes = (keys[key_byte_idx] >> key_bit_idx) & 1 == 1;

        let value = if is_two_bytes {
            if data_offset + 2 > values.len() {
                return Err(Error::Decompression(
                    "SVB16 data truncated: expected 2-byte value".to_string(),
                ));
            }
            let v = u16::from_le_bytes([values[data_offset], values[data_offset + 1]]);
            data_offset += 2;
            v
        } else {
            if data_offset >= values.len() {
                return Err(Error::Decompression(
                    "SVB16 data truncated: expected 1-byte value".to_string(),
                ));
            }
            let v = values[data_offset] as u16;
            data_offset += 1;
            v
        };

        // Zigzag decode
        let delta = zigzag_decode(value);

        // Delta decode (wrapping arithmetic)
        let current = prev.wrapping_add(delta);
        prev = current;

        samples.push(current as i16);
    }

    Ok(samples)
}

/// Validate SVB16 encoded data without fully decoding.
///
/// Returns true if the data appears to be valid for the given sample count.
pub fn validate(data: &[u8], sample_count: usize) -> bool {
    if sample_count == 0 {
        return data.is_empty();
    }

    let keys_len = key_length(sample_count);
    if data.len() < keys_len {
        return false;
    }

    let keys = &data[..keys_len];
    let values = &data[keys_len..];

    // Calculate expected data length from keys
    let mut expected_data_len: usize = 0;
    for i in 0..sample_count {
        let key_byte_idx = i / 8;
        let key_bit_idx = i % 8;
        let is_two_bytes = (keys[key_byte_idx] >> key_bit_idx) & 1 == 1;
        expected_data_len += if is_two_bytes { 2 } else { 1 };
    }

    expected_data_len == values.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zigzag_roundtrip() {
        for i in i16::MIN..=i16::MAX {
            let encoded = zigzag_encode(i as u16);
            let decoded = zigzag_decode(encoded);
            assert_eq!(decoded as i16, i);
        }
    }

    #[test]
    fn test_encode_decode_empty() {
        let samples: Vec<i16> = vec![];
        let encoded = encode(&samples).unwrap();
        assert!(encoded.is_empty());
        let decoded = decode(&encoded, 0).unwrap();
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_encode_decode_single() {
        let samples = vec![42i16];
        let encoded = encode(&samples).unwrap();
        let decoded = decode(&encoded, 1).unwrap();
        assert_eq!(decoded, samples);
    }

    #[test]
    fn test_encode_decode_sequence() {
        let samples: Vec<i16> = (0..100).map(|i| (i * 10) as i16).collect();
        let encoded = encode(&samples).unwrap();
        let decoded = decode(&encoded, samples.len()).unwrap();
        assert_eq!(decoded, samples);
    }

    #[test]
    fn test_encode_decode_negative() {
        let samples: Vec<i16> = (-50..50).collect();
        let encoded = encode(&samples).unwrap();
        let decoded = decode(&encoded, samples.len()).unwrap();
        assert_eq!(decoded, samples);
    }

    #[test]
    fn test_encode_decode_large_values() {
        let samples = vec![i16::MIN, i16::MAX, 0, -1, 1, 32767, -32768];
        let encoded = encode(&samples).unwrap();
        let decoded = decode(&encoded, samples.len()).unwrap();
        assert_eq!(decoded, samples);
    }

    #[test]
    fn test_validate() {
        let samples: Vec<i16> = (0..100).collect();
        let encoded = encode(&samples).unwrap();
        assert!(validate(&encoded, samples.len()));
        assert!(!validate(&encoded, samples.len() + 1));
        assert!(!validate(&encoded[..encoded.len() - 1], samples.len()));
    }
}
