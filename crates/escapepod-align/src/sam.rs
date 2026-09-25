// SPDX-License-Identifier: MIT

//! SAM fields for an alignment: the CIGAR string and the `MD`/`NM` pair.
//!
//! # `MD`/`NM` follow `samtools calmd`, not the scoring
//!
//! The scorer treats a reference `N` as matching anything (see
//! [`crate::alphabet`]). `MD` and `NM` do **not**: they are what
//! `samtools calmd` writes for the same CIGAR, because the pipeline this
//! replaces ran calmd and `escpod classify`'s windowed charging bundle
//! rebuilds each read's reference from `MD`
//! (`escapepod_classify::waveform::reference_from_md`). A reference `N` that
//! vanished from `MD` would put the read's own base in its place, and #306 was
//! exactly a reference-`N` mishap that cost 169/256 chunks their bit-exactness.
//!
//! Measured on samtools 1.23.1 (`tests/fixtures/gen_calmd_golden.py`, pinned
//! by `tests/sam_fields.rs`):
//!
//! * A base is a match only when both letters are A/C/G/T (U counting as T,
//!   case ignored) and equal. A reference `N` is a **mismatch** against every
//!   read base — `N` included — and so is any other IUPAC code; a read `N` is
//!   a mismatch against every reference base. Each counts toward `NM`.
//! * Mismatched and deleted reference letters are written upper-cased but
//!   otherwise verbatim: `n` → `N`, a reference `U` stays `U`.
//! * `NM` = mismatches + inserted bases + deleted bases.
//! * Standard `MD` punctuation: a `0` between adjacent mismatches, between a
//!   deletion and a following mismatch, and at the end when the last aligned
//!   base is a mismatch.

use crate::scalar::CigarOp;

/// The SAM CIGAR string for an aligned region inside a read of `query_len`
/// bases, soft-clipping `query_start` bases before it and
/// `query_len - query_end` after it.
pub fn cigar_string(
    ops: &[(CigarOp, u32)],
    query_len: usize,
    query_start: usize,
    query_end: usize,
) -> String {
    let mut s = String::new();
    if query_start > 0 {
        s.push_str(&format!("{query_start}S"));
    }
    for (op, len) in ops {
        s.push_str(&format!("{len}{}", op.as_char()));
    }
    let tail = query_len - query_end;
    if tail > 0 {
        s.push_str(&format!("{tail}S"));
    }
    s
}

/// Normalise a letter for calmd's equality: upper case, U as T.
#[inline]
fn norm(b: u8) -> u8 {
    match b.to_ascii_uppercase() {
        b'U' => b'T',
        u => u,
    }
}

/// Whether calmd counts this pairing as a match.
#[inline]
pub fn calmd_match(read_base: u8, ref_base: u8) -> bool {
    let (q, r) = (norm(read_base), norm(ref_base));
    q == r && matches!(r, b'A' | b'C' | b'G' | b'T')
}

/// `MD` and `NM` for an alignment, as `samtools calmd` writes them.
///
/// `read` and `reference` are the *letters* (not codes): the read in the
/// orientation it is aligned in, the reference as it appears in the FASTA.
/// `query_start` / `ref_start` are where `ops` begin.
pub fn md_nm(
    read: &[u8],
    reference: &[u8],
    query_start: usize,
    ref_start: usize,
    ops: &[(CigarOp, u32)],
) -> (String, u32) {
    let mut md = String::new();
    let mut nm: u32 = 0;
    let mut run: u32 = 0;
    let (mut i, mut j) = (query_start, ref_start);
    for &(op, len) in ops {
        match op {
            CigarOp::Match => {
                for _ in 0..len {
                    if calmd_match(read[i], reference[j]) {
                        run += 1;
                    } else {
                        md.push_str(&run.to_string());
                        md.push(reference[j].to_ascii_uppercase() as char);
                        run = 0;
                        nm += 1;
                    }
                    i += 1;
                    j += 1;
                }
            }
            CigarOp::Ins => {
                i += len as usize;
                nm += len;
            }
            CigarOp::Del => {
                md.push_str(&run.to_string());
                md.push('^');
                for k in 0..len as usize {
                    md.push(reference[j + k].to_ascii_uppercase() as char);
                }
                run = 0;
                j += len as usize;
                nm += len;
            }
        }
    }
    md.push_str(&run.to_string());
    (md, nm)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn soft_clips() {
        let ops = [(CigarOp::Match, 5), (CigarOp::Ins, 1), (CigarOp::Match, 3)];
        assert_eq!(cigar_string(&ops, 12, 2, 11), "2S5M1I3M1S");
        assert_eq!(cigar_string(&ops, 9, 0, 9), "5M1I3M");
    }

    #[test]
    fn reference_n_is_an_md_mismatch() {
        let ops = [(CigarOp::Match, 10)];
        let (md, nm) = md_nm(b"ACGTAACGTA", b"ACGTANCGTA", 0, 0, &ops);
        assert_eq!((md.as_str(), nm), ("5N4", 1));
    }
}
