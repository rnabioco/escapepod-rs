//! Precomputed tables for SVB16 SIMD decoding.
//!
//! For each possible key byte value (0..=255), we need:
//! - `DECODE_SHUFFLE[key]`: a 16-byte shuffle pattern that expands the
//!   variable-length encoded bytes (1 or 2 bytes per sample, 8 samples
//!   per key byte) into a fixed 8-lane u16 layout.
//! - `DECODED_LEN[key]`: the number of encoded bytes consumed by the
//!   key byte (8 samples, each 1 or 2 bytes).
//!
//! For `pshufb`, index byte with the high bit set zeroes the output lane,
//! which is exactly what we want for the "padding" high byte of a 1-byte
//! sample (its u16 value is `v as u16`, high byte must be 0).

/// Number of encoded bytes consumed per key byte.
/// `8 + popcount(key)` — base 8 one-byte samples plus one extra byte for
/// each 2-byte sample.
pub const DECODED_LEN: [u8; 256] = build_decoded_len_table();

const fn build_decoded_len_table() -> [u8; 256] {
    let mut table = [0u8; 256];
    let mut k = 0usize;
    while k < 256 {
        table[k] = 8 + (k as u8).count_ones() as u8;
        k += 1;
    }
    table
}

/// Shuffle patterns for `_mm_shuffle_epi8` (SSSE3 pshufb) used during
/// SVB16 decode. Given a key byte and up to 16 consecutive bytes of
/// encoded data, the shuffle scatters those bytes into 8 u16 lanes in
/// little-endian order.
pub const DECODE_SHUFFLE: [[u8; 16]; 256] = build_decode_shuffle();

const fn build_decode_shuffle() -> [[u8; 16]; 256] {
    // 0x80 (top bit set) tells pshufb to zero the destination lane.
    let mut table = [[0x80u8; 16]; 256];
    let mut k = 0usize;
    while k < 256 {
        let mut input_pos: u8 = 0;
        let mut i = 0usize;
        while i < 8 {
            let is_two = (k >> i) & 1 == 1;
            // Low byte of output u16 lane `i` — always the next input byte.
            table[k][2 * i] = input_pos;
            input_pos += 1;
            if is_two {
                // High byte comes from the next input byte.
                table[k][2 * i + 1] = input_pos;
                input_pos += 1;
            }
            // else: high byte stays 0x80 → zeroed by pshufb.
            i += 1;
        }
        k += 1;
    }
    table
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decoded_len_matches_popcount() {
        for k in 0u32..256 {
            assert_eq!(DECODED_LEN[k as usize], 8 + k.count_ones() as u8);
        }
    }

    #[test]
    fn decode_shuffle_key_zero_is_identity_low_bytes() {
        // key = 0: all 8 samples are 1 byte each (8 bytes total).
        // lane i low byte = input byte i, lane i high byte = zero sentinel.
        let shuf = DECODE_SHUFFLE[0];
        for i in 0..8 {
            assert_eq!(shuf[2 * i], i as u8, "lane {} low byte", i);
            assert_eq!(shuf[2 * i + 1], 0x80, "lane {} high byte", i);
        }
    }

    #[test]
    fn decode_shuffle_key_all_ones_is_u16_passthrough() {
        // key = 0xff: all 8 samples are 2 bytes each (16 bytes total).
        // Straight LE copy.
        let shuf = DECODE_SHUFFLE[0xff];
        for i in 0..16 {
            assert_eq!(shuf[i], i as u8);
        }
    }

    #[test]
    fn decode_shuffle_mixed_key() {
        // key = 0b0000_0001: sample 0 is 2 bytes, others are 1 byte.
        // Input bytes: [v0_lo, v0_hi, v1, v2, v3, v4, v5, v6, v7]  (9 bytes)
        let shuf = DECODE_SHUFFLE[0b0000_0001];
        assert_eq!(shuf[0], 0); // lane 0 lo = input 0
        assert_eq!(shuf[1], 1); // lane 0 hi = input 1
        assert_eq!(shuf[2], 2); // lane 1 lo = input 2
        assert_eq!(shuf[3], 0x80); // lane 1 hi = zero
        assert_eq!(shuf[4], 3); // lane 2 lo = input 3
        assert_eq!(shuf[5], 0x80);
        // ...
        assert_eq!(shuf[14], 8); // lane 7 lo = input 8
        assert_eq!(shuf[15], 0x80);
        assert_eq!(DECODED_LEN[0b0000_0001], 9);
    }
}
