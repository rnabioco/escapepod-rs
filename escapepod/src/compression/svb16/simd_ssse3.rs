//! SSSE3 implementation of SVB16 decode.
//!
//! Decodes 8 samples per iteration using a `pshufb` shuffle table keyed
//! on the 8-bit length-key byte, then SIMD zigzag-decode and a 3-stage
//! prefix-sum for delta-decode.
//!
//! Only built on x86_64; gated at the dispatch level by a runtime
//! `is_x86_feature_detected!("ssse3")` check.

#![cfg(target_arch = "x86_64")]
// The SIMD loops use the index `k` / `i` to drive multiple arrays and
// pointer-arithmetic offsets; replacing them with iterators is strictly
// worse here.
#![allow(clippy::needless_range_loop)]

use std::arch::x86_64::*;

use super::tables::{DECODE_SHUFFLE, DECODED_LEN, ENCODE_SHUFFLE};

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

/// Encode `samples` into `(keys, data)`. Processes 8 samples per SIMD
/// iteration; hands the tail to the scalar path.
///
/// Returns the number of bytes written to `data`.
///
/// # Safety
/// Caller must ensure the CPU supports SSSE3. `keys` must have length
/// `ceil(samples.len() / 8)`; `data` must have at least
/// `samples.len() * 2 + 16` bytes available (the extra 16 lets the
/// last block use a full 16-byte store without overflow — trailing
/// bytes are garbage, truncated by the caller).
#[target_feature(enable = "ssse3")]
pub unsafe fn encode(samples: &[i16], keys: &mut [u8], data: &mut [u8]) -> usize {
    debug_assert_eq!(keys.len(), samples.len().div_ceil(8));
    debug_assert!(data.len() >= samples.len() * 2 + 16);

    let full_blocks = samples.len() / 8;

    let zero = _mm_setzero_si128();

    let mut prev_u16: u16 = 0;
    let mut out_offset: usize = 0;

    for k in 0..full_blocks {
        // SAFETY: k * 8 + 8 <= full_blocks * 8 <= samples.len(), so the
        // 16-byte load is in bounds. i16 has the same byte layout as u16
        // for the bit patterns we care about (delta via wrapping_sub).
        let input = unsafe { _mm_loadu_si128(samples.as_ptr().add(k * 8) as *const __m128i) };

        // Delta: current - [carry, v0, v1, …, v6]
        // Build `prev_vec` = input shifted right by one u16 lane, with
        // `prev_u16` spliced into lane 0.
        //
        // `_mm_bslli_si128::<2>` shifts the vector left in memory order
        // (i.e. lane 0 gets zero, old lane 0 → lane 1). We then OR in
        // prev_u16 at lane 0.
        let shifted = _mm_bslli_si128::<2>(input);
        let carry_lane = _mm_insert_epi16::<0>(shifted, prev_u16 as i32);
        let deltas = _mm_sub_epi16(input, carry_lane);

        // Zigzag encode: (v << 1) ^ (v >> 15)   [arithmetic shift for sign]
        let shifted_left = _mm_slli_epi16::<1>(deltas);
        let sign = _mm_srai_epi16::<15>(deltas);
        let zz = _mm_xor_si128(shifted_left, sign);

        // Key byte: bit i is set iff zz_i >= 256 (i.e. high byte non-zero).
        // srli 8 bits → each lane holds [0, high_byte]. Compare > 0 (signed
        // OK: high byte always 0..=255, sign bit clear after srli).
        let high_bytes = _mm_srli_epi16::<8>(zz);
        let mask = _mm_cmpgt_epi16(high_bytes, zero);
        // Pack two u16 vectors → saturated i8 lanes. We only care about the
        // lower 8 lanes of the result; duplicate mask into both halves.
        let packed = _mm_packs_epi16(mask, mask);
        let mmask = _mm_movemask_epi8(packed) as u32;
        let key = (mmask & 0xff) as u8;
        keys[k] = key;

        let len = DECODED_LEN[key as usize] as usize;

        // Compact the 16 bytes of zz into the output stream using the
        // precomputed encode shuffle.
        // SAFETY: ENCODE_SHUFFLE is a `const` 16-byte array.
        let shuf =
            unsafe { _mm_loadu_si128(ENCODE_SHUFFLE[key as usize].as_ptr() as *const __m128i) };
        let packed_out = _mm_shuffle_epi8(zz, shuf);

        // SAFETY: data.len() >= samples.len() * 2 + 16 and
        // out_offset + 16 <= samples.len() * 2 + 16 because each block
        // consumes at most 16 output bytes (checked below: out_offset
        // only grows by `len` ≤ 16 per iteration, starting from 0, and
        // total ≤ samples.len() * 2 ≤ data.len() - 16).
        unsafe {
            _mm_storeu_si128(
                data.as_mut_ptr().add(out_offset) as *mut __m128i,
                packed_out,
            );
        }

        // Carry: last sample's u16 value becomes prev for next block.
        // Extract lane 7 of the input (pre-zigzag).
        prev_u16 = _mm_extract_epi16::<7>(input) as u16;

        out_offset += len;
    }

    // Scalar tail for remaining samples (0..7).
    for i in (full_blocks * 8)..samples.len() {
        let current = samples[i] as u16;
        let delta = current.wrapping_sub(prev_u16);
        prev_u16 = current;
        let zz = (delta.wrapping_shl(1)) ^ ((delta as i16).wrapping_shr(15) as u16);

        let byte_idx = i / 8;
        let bit_idx = i % 8;
        if zz < 256 {
            data[out_offset] = zz as u8;
            out_offset += 1;
        } else {
            keys[byte_idx] |= 1 << bit_idx;
            data[out_offset] = zz as u8;
            data[out_offset + 1] = (zz >> 8) as u8;
            out_offset += 2;
        }
    }

    out_offset
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
    fn encode_matches_scalar_short() {
        if !ssse3() {
            return;
        }
        let samples: Vec<i16> = (-50..50).collect();
        let scalar_encoded = super::super::encode_scalar(&samples).unwrap();
        let simd_encoded = super::super::encode(&samples).unwrap();
        assert_eq!(simd_encoded, scalar_encoded);
        // Roundtrip.
        let decoded = decode_scalar(&simd_encoded, samples.len()).unwrap();
        assert_eq!(decoded, samples);
    }

    #[test]
    fn encode_matches_scalar_large() {
        if !ssse3() {
            return;
        }
        let mut samples = Vec::with_capacity(10_000);
        let mut v: i16 = 500;
        for i in 0..10_000 {
            let noise = ((i * 7) % 20) as i16 - 10;
            if i % 500 == 0 {
                v = 400 + ((i / 500) % 3) as i16 * 100;
            }
            samples.push(v + noise);
        }
        let scalar_encoded = super::super::encode_scalar(&samples).unwrap();
        let simd_encoded = super::super::encode(&samples).unwrap();
        assert_eq!(simd_encoded, scalar_encoded);
    }

    #[test]
    fn encode_matches_scalar_extreme_values() {
        if !ssse3() {
            return;
        }
        // Values that force 2-byte encoding and large deltas across block
        // boundaries — stress the carry between blocks.
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
            1000,
            -1000,
            2,
            -2,
            3,
            -3,
        ];
        let scalar_encoded = super::super::encode_scalar(&samples).unwrap();
        let simd_encoded = super::super::encode(&samples).unwrap();
        assert_eq!(simd_encoded, scalar_encoded);
        let decoded = super::super::decode(&simd_encoded, samples.len()).unwrap();
        assert_eq!(decoded, samples);
    }

    #[test]
    fn encode_matches_scalar_every_tail_size() {
        if !ssse3() {
            return;
        }
        for n in 1..=33 {
            let samples: Vec<i16> = (0..n as i16).map(|i| (i * 37) ^ 0x2bad).collect();
            let scalar_encoded = super::super::encode_scalar(&samples).unwrap();
            let simd_encoded = super::super::encode(&samples).unwrap();
            assert_eq!(simd_encoded, scalar_encoded, "mismatch at n={}", n);
        }
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
