//! SSSE3 implementation of SVB16 decode.
//!
//! Decodes 8 samples per iteration using a `pshufb` shuffle table keyed
//! on the 8-bit length-key byte, then SIMD zigzag-decode and a 3-stage
//! prefix-sum for delta-decode.
//!
//! Only built on x86_64; gated at the dispatch level by a runtime
//! `is_x86_feature_detected!("ssse3")` check.

#![cfg(target_arch = "x86_64")]

use std::arch::x86_64::*;

use super::tables::{DECODE_SHUFFLE, DECODED_LEN};

/// Decode exactly `sample_count` samples from `(keys, values)` into `out`.
///
/// # Safety
/// Caller must ensure the CPU supports SSSE3 (`is_x86_feature_detected!`).
/// `out` must have capacity for at least `sample_count` elements; this
/// function writes through the raw pointer and sets the length.
#[target_feature(enable = "ssse3")]
pub unsafe fn decode(
    keys: &[u8],
    values: &[u8],
    sample_count: usize,
    out: &mut Vec<i16>,
) -> Result<(), &'static str> {
    debug_assert!(out.capacity() >= sample_count);

    out.clear();
    let out_ptr = out.as_mut_ptr();

    let full_blocks = sample_count / 8;

    let mut data_offset: usize = 0;
    let mut prev_u16: u16 = 0;

    // SIMD fast path: process blocks while we can safely load 16 bytes
    // from `values` at `data_offset`. Intrinsics are safe inside this
    // `unsafe fn` body; the caller's contract is the target_feature check
    // plus `out.capacity() >= sample_count`.
    let ones = _mm_set1_epi16(1);
    let zero = _mm_setzero_si128();

    let mut k = 0usize;
    while k < full_blocks && data_offset + 16 <= values.len() {
        let key = keys[k] as usize;
        let len = DECODED_LEN[key] as usize;

        // SAFETY: `data_offset + 16 <= values.len()` verified by the loop
        // guard above. The shuffle table is a `const`, 16-byte aligned enough
        // for `_mm_loadu_si128` (unaligned load).
        let data = unsafe { _mm_loadu_si128(values.as_ptr().add(data_offset) as *const __m128i) };
        let shuf = unsafe { _mm_loadu_si128(DECODE_SHUFFLE[key].as_ptr() as *const __m128i) };
        // 8 zigzag-encoded u16 values.
        let zz = _mm_shuffle_epi8(data, shuf);

        // Zigzag decode: (v >> 1) ^ -(v & 1)
        let low_bits = _mm_and_si128(zz, ones);
        let neg = _mm_sub_epi16(zero, low_bits);
        let shifted = _mm_srli_epi16::<1>(zz);
        let deltas = _mm_xor_si128(shifted, neg);

        // Prefix-sum delta decode (8 lanes, u16, wrapping add).
        // Three shift+add stages give full-lane inclusive scan.
        let step1 = _mm_add_epi16(deltas, _mm_slli_si128::<2>(deltas));
        let step2 = _mm_add_epi16(step1, _mm_slli_si128::<4>(step1));
        let step3 = _mm_add_epi16(step2, _mm_slli_si128::<8>(step2));

        // Add carry from previous block (broadcast `prev_u16` across all 8 lanes).
        let carry = _mm_set1_epi16(prev_u16 as i16);
        let result = _mm_add_epi16(step3, carry);

        // SAFETY: `k * 8 + 8 <= full_blocks * 8 <= sample_count <= out.capacity()`.
        unsafe {
            _mm_storeu_si128(out_ptr.add(k * 8) as *mut __m128i, result);
        }

        // Extract the last lane as the new running carry.
        prev_u16 = _mm_extract_epi16::<7>(result) as u16;

        data_offset += len;
        k += 1;
    }

    // Scalar tail — either the final partial block, or late blocks where
    // the 16-byte load would have walked off the end of `values`.
    let written_samples = k * 8;
    // SAFETY: we wrote `written_samples` i16s via the store above; capacity
    // is guaranteed by the function contract.
    unsafe {
        out.set_len(written_samples);
    }

    for i in written_samples..sample_count {
        let byte_idx = i / 8;
        let bit_idx = i % 8;
        let is_two = (keys[byte_idx] >> bit_idx) & 1 == 1;

        let value: u16 = if is_two {
            if data_offset + 2 > values.len() {
                return Err("SVB16 data truncated: expected 2-byte value");
            }
            let v = u16::from_le_bytes([values[data_offset], values[data_offset + 1]]);
            data_offset += 2;
            v
        } else {
            if data_offset >= values.len() {
                return Err("SVB16 data truncated: expected 1-byte value");
            }
            let v = values[data_offset] as u16;
            data_offset += 1;
            v
        };

        // Zigzag decode + delta decode.
        let delta = (value >> 1) ^ (value & 1).wrapping_neg();
        let current = prev_u16.wrapping_add(delta);
        prev_u16 = current;
        out.push(current as i16);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::{decode_scalar, encode};
    use super::*;

    fn ssse3() -> bool {
        is_x86_feature_detected!("ssse3")
    }

    #[test]
    fn matches_scalar_short() {
        if !ssse3() {
            return;
        }
        let samples: Vec<i16> = (-50..50).collect();
        let encoded = encode(&samples).unwrap();
        let keys_len = super::super::key_length(samples.len());
        let (keys, values) = encoded.split_at(keys_len);
        let scalar = decode_scalar(&encoded, samples.len()).unwrap();
        let mut simd_out: Vec<i16> = Vec::with_capacity(samples.len());
        unsafe { decode(keys, values, samples.len(), &mut simd_out).unwrap() };
        assert_eq!(simd_out, scalar);
        assert_eq!(simd_out, samples);
    }

    #[test]
    fn matches_scalar_large() {
        if !ssse3() {
            return;
        }
        // Realistic nanopore-ish signal.
        let mut samples = Vec::with_capacity(10_000);
        let mut v: i16 = 500;
        for i in 0..10_000 {
            let noise = ((i * 7) % 20) as i16 - 10;
            if i % 500 == 0 {
                v = 400 + ((i / 500) % 3) as i16 * 100;
            }
            samples.push(v + noise);
        }
        let encoded = encode(&samples).unwrap();
        let keys_len = super::super::key_length(samples.len());
        let (keys, values) = encoded.split_at(keys_len);
        let scalar = decode_scalar(&encoded, samples.len()).unwrap();
        let mut simd_out = Vec::with_capacity(samples.len());
        unsafe { decode(keys, values, samples.len(), &mut simd_out).unwrap() };
        assert_eq!(simd_out, scalar);
        assert_eq!(simd_out, samples);
    }

    #[test]
    fn matches_scalar_extreme_values() {
        if !ssse3() {
            return;
        }
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
            -32767,
            32766,
            0,
            0,
            0,
            0,
            0,
            123,
            -456,
        ];
        let encoded = encode(&samples).unwrap();
        let keys_len = super::super::key_length(samples.len());
        let (keys, values) = encoded.split_at(keys_len);
        let scalar = decode_scalar(&encoded, samples.len()).unwrap();
        let mut simd_out = Vec::with_capacity(samples.len());
        unsafe { decode(keys, values, samples.len(), &mut simd_out).unwrap() };
        assert_eq!(simd_out, scalar);
        assert_eq!(simd_out, samples);
    }

    #[test]
    fn matches_scalar_random_tail_sizes() {
        if !ssse3() {
            return;
        }
        // Verify the scalar tail handles every residue mod 8.
        for n in 1..=33 {
            let samples: Vec<i16> = (0..n as i16).map(|i| (i * 37) ^ 0x2bad).collect();
            let encoded = encode(&samples).unwrap();
            let keys_len = super::super::key_length(samples.len());
            let (keys, values) = encoded.split_at(keys_len);
            let scalar = decode_scalar(&encoded, samples.len()).unwrap();
            let mut simd_out = Vec::with_capacity(samples.len());
            unsafe { decode(keys, values, samples.len(), &mut simd_out).unwrap() };
            assert_eq!(simd_out, scalar, "mismatch at n={}", n);
        }
    }
}
