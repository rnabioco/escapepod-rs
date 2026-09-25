// SPDX-License-Identifier: MIT

//! The scoring alphabet.
//!
//! A/C/G/T encode to 0–3, with U folded to T and case ignored. **Every other
//! letter** — `N` and the rest of IUPAC, and anything else a FASTA may carry —
//! encodes to [`WILDCARD`], which pairs with any letter as a match. The tRNA
//! panels this is built for carry an `N` in every reference (the
//! discriminator position a dual-adapter reference cannot fix), and scoring it
//! as a mismatch would penalise every read at the same column for a base the
//! reference does not claim to know.
//!
//! This is the *scoring* rule only. `MD`/`NM` follow `samtools calmd`, which
//! calls a reference `N` a mismatch against anything; see [`crate::sam`].

/// Code for A.
pub const A: u8 = 0;
/// Code for C.
pub const C: u8 = 1;
/// Code for G.
pub const G: u8 = 2;
/// Code for T (and U).
pub const T: u8 = 3;
/// Code for every other letter: matches anything.
pub const WILDCARD: u8 = 4;
/// Number of distinct codes a sequence can carry.
pub const N_CODES: usize = 5;

const fn build_table() -> [u8; 256] {
    let mut t = [WILDCARD; 256];
    t[b'A' as usize] = A;
    t[b'a' as usize] = A;
    t[b'C' as usize] = C;
    t[b'c' as usize] = C;
    t[b'G' as usize] = G;
    t[b'g' as usize] = G;
    t[b'T' as usize] = T;
    t[b't' as usize] = T;
    t[b'U' as usize] = T;
    t[b'u' as usize] = T;
    t
}

static TABLE: [u8; 256] = build_table();

/// Encode one letter.
#[inline]
pub fn encode_base(b: u8) -> u8 {
    TABLE[b as usize]
}

/// Encode a sequence of letters into codes.
pub fn encode(seq: &[u8]) -> Vec<u8> {
    seq.iter().map(|&b| encode_base(b)).collect()
}

/// Reverse complement of an encoded sequence. A wildcard stays a wildcard.
pub fn reverse_complement_codes(codes: &[u8]) -> Vec<u8> {
    codes
        .iter()
        .rev()
        .map(|&c| if c == WILDCARD { WILDCARD } else { 3 - c })
        .collect()
}

/// Reverse complement of a sequence of letters, as SAM writes a reverse-strand
/// `SEQ`: upper case, U complemented to A, every letter outside A/C/G/T/U
/// written as `N`.
pub fn reverse_complement(seq: &[u8]) -> Vec<u8> {
    seq.iter()
        .rev()
        .map(|&b| match b.to_ascii_uppercase() {
            b'A' => b'T',
            b'C' => b'G',
            b'G' => b'C',
            b'T' | b'U' => b'A',
            _ => b'N',
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn folds_case_and_u() {
        assert_eq!(encode(b"ACGTUacgtu"), vec![0, 1, 2, 3, 3, 0, 1, 2, 3, 3]);
    }

    #[test]
    fn everything_else_is_a_wildcard() {
        for b in b"NnRYKMSWBDHV-*.X" {
            assert_eq!(encode_base(*b), WILDCARD, "{}", *b as char);
        }
    }

    #[test]
    fn reverse_complements() {
        assert_eq!(reverse_complement(b"ACGUn"), b"NACGT".to_vec());
        assert_eq!(
            reverse_complement_codes(&encode(b"ACGTN")),
            encode(b"NACGT")
        );
    }
}
