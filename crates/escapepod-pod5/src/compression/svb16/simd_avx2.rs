//! AVX2 implementation of SVB16 decode.
//!
//! Processes 16 samples (two key bytes) per iteration using two 128-bit
//! `pshufb` shuffles packed into a single 256-bit vector. The SSSE3 decode
//! serializes the carry between blocks through the scalar prev_u16
//! register; AVX2 pipelines two adjacent blocks per iteration and
//! propagates the cross-block carry via a single `_mm256_extract_epi16`.
//!
//! Only built on x86_64; gated at the dispatch level by a runtime
//! `is_x86_feature_detected!("avx2")` check.

#![cfg(target_arch = "x86_64")]
#![allow(clippy::needless_range_loop)]

use std::arch::x86_64::*;

use super::tables::{DECODE_SHUFFLE, DECODED_LEN};

/// Decode exactly `sample_count` samples from `(keys, values)` into `out`.
///
/// # Safety
/// Caller must ensure the CPU supports AVX2 (`is_x86_feature_detected!`).
/// `out` must have capacity for at least `sample_count` elements; this
/// function writes through the raw pointer and sets the length.
#[target_feature(enable = "avx2")]
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
    let full_pairs = full_blocks / 2;

    let mut data_offset: usize = 0;
    let mut prev_u16: u16 = 0;

    let ones = _mm256_set1_epi16(1);
    let zero = _mm256_setzero_si256();

    let mut pair = 0usize;
    // SIMD fast path: process 2-block pairs while we can safely load two
    // 16-byte windows into a single 256-bit register. The tighter guard
    // is `data_offset + len_a + 16 <= values.len()` — len_a is at most 16,
    // so `+ 32` is a sufficient condition and avoids a table lookup.
    while pair < full_pairs && data_offset + 32 <= values.len() {
        let key_a = keys[pair * 2] as usize;
        let key_b = keys[pair * 2 + 1] as usize;
        let len_a = DECODED_LEN[key_a] as usize;
        let len_b = DECODED_LEN[key_b] as usize;

        // Load each block's 16-byte data window into the low half of a
        // 128-bit register, then glue them into one 256-bit vector.
        // SAFETY: `data_offset + len_a + 16 <= values.len()` because
        // `len_a <= 16` and the loop guard requires `data_offset + 32 <=
        // values.len()`.
        let data_a = unsafe { _mm_loadu_si128(values.as_ptr().add(data_offset) as *const __m128i) };
        let data_b =
            unsafe { _mm_loadu_si128(values.as_ptr().add(data_offset + len_a) as *const __m128i) };
        let shuf_a = unsafe { _mm_loadu_si128(DECODE_SHUFFLE[key_a].as_ptr() as *const __m128i) };
        let shuf_b = unsafe { _mm_loadu_si128(DECODE_SHUFFLE[key_b].as_ptr() as *const __m128i) };

        // `_mm256_shuffle_epi8` is per-128-bit-lane `pshufb`, so we want
        // block A in lane 0 and block B in lane 1. Use the native
        // "zero-extend then insert high" composition for explicit intent.
        let data = _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(data_a), data_b);
        let shuf = _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(shuf_a), shuf_b);

        // 16 zigzag-encoded u16 values: lane A carries block A's 8, lane B block B's 8.
        let zz = _mm256_shuffle_epi8(data, shuf);

        // Zigzag decode: (v >> 1) ^ -(v & 1). Both lanes in one shot.
        let low_bits = _mm256_and_si256(zz, ones);
        let neg = _mm256_sub_epi16(zero, low_bits);
        let shifted = _mm256_srli_epi16::<1>(zz);
        let deltas = _mm256_xor_si256(shifted, neg);

        // Prefix-sum delta decode. `_mm256_slli_si256::<N>` is per-lane
        // byte shift — the two lanes scan independently, which is exactly
        // what we want before we splice in the cross-lane carry.
        let step1 = _mm256_add_epi16(deltas, _mm256_slli_si256::<2>(deltas));
        let step2 = _mm256_add_epi16(step1, _mm256_slli_si256::<4>(step1));
        let step3 = _mm256_add_epi16(step2, _mm256_slli_si256::<8>(step2));

        // Carry for block A is the running prev_u16; block B's carry is
        // block A's final value (prev + cumulative delta through lane A[7]).
        // Compose the 256-bit carry vector: lane 0 = broadcast(prev_u16),
        // lane 1 = zero (we'll add lane B's carry in a second pass once we
        // know the value — we have to extract it after lane A is finalized).
        let carry_a = _mm256_zextsi128_si256(_mm_set1_epi16(prev_u16 as i16));
        let result_a = _mm256_add_epi16(step3, carry_a);

        // `_mm256_extract_epi16::<7>` pulls out the 7th u16 (lane A's last
        // element, bits 112..=127). That's block A's final decoded u16.
        let lane_a_final = _mm256_extract_epi16::<7>(result_a) as u16;

        let carry_b = _mm256_inserti128_si256::<1>(
            _mm256_setzero_si256(),
            _mm_set1_epi16(lane_a_final as i16),
        );
        let result = _mm256_add_epi16(result_a, carry_b);

        // SAFETY: `pair * 16 + 16 <= full_pairs * 16 <= full_blocks * 8
        // <= sample_count <= out.capacity()`.
        unsafe {
            _mm256_storeu_si256(out_ptr.add(pair * 16) as *mut __m256i, result);
        }

        // New running carry is lane B's last u16 (bits 240..=255 of the 256-bit vec).
        prev_u16 = _mm256_extract_epi16::<15>(result) as u16;

        data_offset += len_a + len_b;
        pair += 1;
    }

    let pairs_written = pair * 16;
    // SAFETY: exactly `pairs_written` i16s written above.
    unsafe {
        out.set_len(pairs_written);
    }

    // Tail: at most one leftover full block (8 samples) + residue (0..=7).
    // Keep it scalar — the cost is bounded and we avoid cross-module coupling.
    for i in pairs_written..sample_count {
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

        let delta = (value >> 1) ^ (value & 1).wrapping_neg();
        let current = prev_u16.wrapping_add(delta);
        prev_u16 = current;
        out.push(current as i16);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::{decode_scalar, encode, key_length};
    use super::*;

    fn avx2() -> bool {
        is_x86_feature_detected!("avx2")
    }

    #[test]
    fn matches_scalar_short() {
        if !avx2() {
            return;
        }
        let samples: Vec<i16> = (-50..50).collect();
        let encoded = encode(&samples).unwrap();
        let keys_len = key_length(samples.len());
        let (keys, values) = encoded.split_at(keys_len);
        let scalar = decode_scalar(&encoded, samples.len()).unwrap();
        let mut simd_out: Vec<i16> = Vec::with_capacity(samples.len());
        unsafe { decode(keys, values, samples.len(), &mut simd_out).unwrap() };
        assert_eq!(simd_out, scalar);
        assert_eq!(simd_out, samples);
    }

    #[test]
    fn matches_scalar_large() {
        if !avx2() {
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
        let encoded = encode(&samples).unwrap();
        let keys_len = key_length(samples.len());
        let (keys, values) = encoded.split_at(keys_len);
        let scalar = decode_scalar(&encoded, samples.len()).unwrap();
        let mut simd_out = Vec::with_capacity(samples.len());
        unsafe { decode(keys, values, samples.len(), &mut simd_out).unwrap() };
        assert_eq!(simd_out, scalar);
        assert_eq!(simd_out, samples);
    }

    #[test]
    fn matches_scalar_extreme_values() {
        if !avx2() {
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
        let keys_len = key_length(samples.len());
        let (keys, values) = encoded.split_at(keys_len);
        let scalar = decode_scalar(&encoded, samples.len()).unwrap();
        let mut simd_out = Vec::with_capacity(samples.len());
        unsafe { decode(keys, values, samples.len(), &mut simd_out).unwrap() };
        assert_eq!(simd_out, scalar);
        assert_eq!(simd_out, samples);
    }

    #[test]
    fn matches_scalar_every_tail_size() {
        if !avx2() {
            return;
        }
        // Exercise every residue mod 16 — covers both the leftover full
        // SSSE3-sized block after the AVX2 pair loop and the scalar 0..=7
        // residue.
        for n in 1..=65 {
            let samples: Vec<i16> = (0..n as i16).map(|i| (i * 37) ^ 0x2bad).collect();
            let encoded = encode(&samples).unwrap();
            let keys_len = key_length(samples.len());
            let (keys, values) = encoded.split_at(keys_len);
            let scalar = decode_scalar(&encoded, samples.len()).unwrap();
            let mut simd_out = Vec::with_capacity(samples.len());
            unsafe { decode(keys, values, samples.len(), &mut simd_out).unwrap() };
            assert_eq!(simd_out, scalar, "mismatch at n={}", n);
        }
    }
}
