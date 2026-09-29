//! Matching a decoded sequence to a barcode reference by edit distance.
//!
//! The last step of basecall-then-match demultiplexing, and the one that turns
//! [`super::lattice`]'s sequences into barcode calls. Distance comes from
//! a bit-parallel Levenshtein (Myers 1999): a 40 nt reference packs into one
//! `u64`, so a comparison is one pass over the decode with no allocation. A
//! decode sits ~4 edits from its own reference and ~10+ from every other one,
//! so the bounded form abandons most of the 96 comparisons well before the end.
//!
//! # What counts as a reference
//!
//! The **last 40 nt of each barcode strand** — the exact targets the model was
//! trained to decode toward (`extract_chunks.py` derives training targets the
//! same way). References are supplied ready to use rather than derived here:
//! turning a pool-oligo table into targets involves the barcode design's own
//! conventions (strand selection, `/5Phos/` stripping, suffix length), which
//! belong with the design, not in a demultiplexer.
//!
//! # Confidence
//!
//! Confidence is the **edit-distance margin to the second-best reference**, not
//! a probability. That is not invented here: it is what the published
//! precision-at-recovery numbers for this model were computed with, so keeping
//! the definition identical is what makes a recovery threshold mean the same
//! thing it meant when the model was evaluated.
//!
//! A margin of 0 means two references tie and the read is genuinely ambiguous.

use std::path::Path;

/// A barcode reference set: names and their target sequences.
#[derive(Debug, Clone, Default)]
pub struct BarcodeRefs {
    names: Vec<String>,
    seqs: Vec<Vec<u8>>,
}

/// Errors from loading or using a reference set.
#[derive(Debug, thiserror::Error)]
pub enum BarcodeError {
    #[error("failed to read barcode references {path}: {source}")]
    Io {
        path: std::path::PathBuf,
        source: std::io::Error,
    },
    #[error("{path}: expected a header with `name` and `sequence` columns, got {got:?}")]
    MissingColumns {
        path: std::path::PathBuf,
        got: Vec<String>,
    },
    #[error("{path} line {line}: expected {want} fields, got {got}")]
    BadRow {
        path: std::path::PathBuf,
        line: usize,
        want: usize,
        got: usize,
    },
    #[error("{path} line {line}: sequence {seq:?} contains a non-ACGT character")]
    BadSequence {
        path: std::path::PathBuf,
        line: usize,
        seq: String,
    },
    #[error("{path}: duplicate barcode name {name:?}")]
    DuplicateName {
        path: std::path::PathBuf,
        name: String,
    },
    #[error("{path}: no references")]
    Empty { path: std::path::PathBuf },
}

/// One read's best and runner-up reference.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BarcodeMatch {
    /// Index into the reference set of the closest reference.
    pub index: usize,
    /// Edit distance to it.
    pub best_dist: u32,
    /// Edit distance to the next-closest reference, if there is more than one.
    pub second_best_dist: Option<u32>,
    /// `second_best_dist - best_dist`; `None` with a single reference.
    pub margin: Option<u32>,
}

impl BarcodeRefs {
    /// Load references from a CSV with `name` and `sequence` columns.
    ///
    /// Extra columns are ignored, so a richer design table can be passed
    /// through unchanged as long as it carries those two.
    pub fn from_csv(path: impl AsRef<Path>) -> Result<Self, BarcodeError> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).map_err(|source| BarcodeError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        let mut lines = text.lines().enumerate();

        let (_, header) = lines.next().ok_or_else(|| BarcodeError::Empty {
            path: path.to_path_buf(),
        })?;
        let cols: Vec<&str> = header.split(',').map(str::trim).collect();
        let find = |want: &str| cols.iter().position(|c| c.eq_ignore_ascii_case(want));
        let (Some(name_at), Some(seq_at)) = (find("name"), find("sequence")) else {
            return Err(BarcodeError::MissingColumns {
                path: path.to_path_buf(),
                got: cols.iter().map(|s| s.to_string()).collect(),
            });
        };
        let want = name_at.max(seq_at) + 1;

        let mut out = Self::default();
        for (i, line) in lines {
            let line = line.trim_end();
            if line.is_empty() {
                continue;
            }
            let fields: Vec<&str> = line.split(',').map(str::trim).collect();
            if fields.len() < want {
                return Err(BarcodeError::BadRow {
                    path: path.to_path_buf(),
                    line: i + 1,
                    want,
                    got: fields.len(),
                });
            }
            let seq = fields[seq_at].to_ascii_uppercase();
            if seq.is_empty() || !seq.bytes().all(|b| b"ACGT".contains(&b)) {
                return Err(BarcodeError::BadSequence {
                    path: path.to_path_buf(),
                    line: i + 1,
                    seq,
                });
            }
            let name = fields[name_at].to_string();
            if out.names.contains(&name) {
                return Err(BarcodeError::DuplicateName {
                    path: path.to_path_buf(),
                    name,
                });
            }
            out.names.push(name);
            out.seqs.push(seq.into_bytes());
        }
        if out.names.is_empty() {
            return Err(BarcodeError::Empty {
                path: path.to_path_buf(),
            });
        }
        Ok(out)
    }

    /// Build a reference set from `(name, sequence)` pairs.
    ///
    /// For references that travel inside the model bundle rather than in a
    /// separate CSV. Applies the same validation as [`Self::from_csv`] —
    /// non-empty, ACGT-only, no duplicate names — because a bundle is no more
    /// trustworthy than a file the user pointed at.
    pub fn from_pairs<N, S>(pairs: impl IntoIterator<Item = (N, S)>) -> Result<Self, BarcodeError>
    where
        N: Into<String>,
        S: AsRef<str>,
    {
        let path = || std::path::PathBuf::from("<bundle metadata.json>");
        let mut out = Self::default();
        for (i, (name, seq)) in pairs.into_iter().enumerate() {
            let seq = seq.as_ref().to_ascii_uppercase();
            if seq.is_empty() || !seq.bytes().all(|b| b"ACGT".contains(&b)) {
                return Err(BarcodeError::BadSequence {
                    path: path(),
                    line: i + 1,
                    seq,
                });
            }
            let name = name.into();
            if out.names.contains(&name) {
                return Err(BarcodeError::DuplicateName { path: path(), name });
            }
            out.names.push(name);
            out.seqs.push(seq.into_bytes());
        }
        if out.names.is_empty() {
            return Err(BarcodeError::Empty { path: path() });
        }
        Ok(out)
    }

    pub fn len(&self) -> usize {
        self.names.len()
    }

    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }

    pub fn name(&self, index: usize) -> &str {
        &self.names[index]
    }

    /// The reference sequence at `index` — the sequence the model emits, as
    /// loaded. Callers scoring against the lattice need these, not the names.
    pub fn sequence(&self, index: usize) -> &[u8] {
        &self.seqs[index]
    }

    /// Every reference sequence, in name order.
    pub fn sequences(&self) -> Vec<&[u8]> {
        self.seqs.iter().map(Vec::as_slice).collect()
    }

    pub fn names(&self) -> &[String] {
        &self.names
    }

    /// Smallest edit distance between any two references.
    ///
    /// The floor on what a call can mean: a read cannot be resolved to better
    /// than half this, and a `best_dist` above it says the decode is closer to
    /// noise than the references are to each other. Worth reporting once at
    /// startup — the L27 design measures 10 across all 96 codes (higher on the
    /// subsets used per flowcell), against a typical `best_dist` of 4.
    ///
    /// `O(n^2)` in the reference count, so this is a startup diagnostic, not
    /// something to call per read.
    pub fn min_pairwise_distance(&self) -> Option<u32> {
        let mut best: Option<u32> = None;
        for (i, a) in self.seqs.iter().enumerate() {
            for b in &self.seqs[i + 1..] {
                let d = edit_distance(a, b);
                best = Some(best.map_or(d, |m: u32| m.min(d)));
            }
        }
        best
    }

    /// Closest reference to `query`, with the runner-up for the margin.
    ///
    /// Ties for best resolve to the lowest reference index and leave a margin of
    /// 0, which is the honest signal that the read is ambiguous rather than a
    /// silent coin flip.
    pub fn match_sequence(&self, query: &[u8]) -> Option<BarcodeMatch> {
        if self.seqs.is_empty() {
            return None;
        }
        let (mut best, mut best_at) = (u32::MAX, 0usize);
        let mut second = u32::MAX;
        for (i, refseq) in self.seqs.iter().enumerate() {
            // Cap each comparison at the current runner-up. Both branches below
            // discard any `d >= second`, so a reference that cannot beat it does
            // not need its exact distance — only the fact that it lost. The
            // bit-vector abandons once the running score cannot get under the
            // cap, so this is what buys the early exit: without a cap every
            // reference runs the whole decode to its true distance.
            //
            // `best <= second` holds throughout (the swap below preserves it),
            // so capping at `second` cannot hide a new best either.
            let Some(d) = edit_distance_within(refseq, query, second) else {
                continue;
            };
            if d < best {
                second = best;
                best = d;
                best_at = i;
            } else if d < second {
                second = d;
            }
        }
        let second_best_dist = (self.seqs.len() > 1).then_some(second);
        Some(BarcodeMatch {
            index: best_at,
            best_dist: best,
            second_best_dist,
            margin: second_best_dist.map(|s| s.saturating_sub(best)),
        })
    }
}

/// Exact Levenshtein distance (substitutions, insertions and deletions cost 1).
fn edit_distance(a: &[u8], b: &[u8]) -> u32 {
    bounded_distance(a, b, u32::MAX).expect("no limit cannot be exceeded")
}

/// Levenshtein distance, but only when it is strictly below `limit`.
///
/// `Some(d)` iff `d < limit`; `None` means "at least `limit`", with no exact
/// value computed, so a reference far from the query stops early instead of
/// running to its true optimum. `limit == u32::MAX` is exact.
fn edit_distance_within(a: &[u8], b: &[u8], limit: u32) -> Option<u32> {
    bounded_distance(a, b, limit)
}

fn bounded_distance(a: &[u8], b: &[u8], limit: u32) -> Option<u32> {
    if limit == 0 {
        return None;
    }
    let (pat, text) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    // Each length difference costs at least one indel.
    if (text.len() - pat.len()) as u64 >= u64::from(limit) {
        return None;
    }
    if pat.is_empty() {
        return Some(text.len() as u32);
    }
    myers64(pat, text, limit).unwrap_or_else(|| {
        let d = levenshtein_dp(pat, text);
        (d < limit).then_some(d)
    })
}

/// 2-bit code of an unambiguous base; anything else is not bit-parallel.
fn base_code(b: u8) -> Option<usize> {
    match b {
        b'A' => Some(0),
        b'C' => Some(1),
        b'G' => Some(2),
        b'T' => Some(3),
        _ => None,
    }
}

/// Myers' bit-vector edit distance, pattern in one machine word.
///
/// The outer `None` means "not applicable" — the pattern is longer than 64 or
/// holds a byte that is not `ACGT` (where a match table over four symbols would
/// silently call two equal `N`s different) — and the caller falls back to the
/// DP. The inner value is [`bounded_distance`]'s.
///
/// `pv`/`mv` are the vertical +1/−1 deltas down the pattern column, so `score`
/// tracks the last row. It can fall by at most one per text column, hence the
/// abandon test: the final distance is at least `score - columns left`.
fn myers64(pat: &[u8], text: &[u8], limit: u32) -> Option<Option<u32>> {
    if pat.len() > 64 {
        return None;
    }
    let mut peq = [0u64; 4];
    for (i, &c) in pat.iter().enumerate() {
        peq[base_code(c)?] |= 1 << i;
    }
    let last = 1u64 << (pat.len() - 1);
    let (mut pv, mut mv) = (!0u64, 0u64);
    let mut score = pat.len() as u32;
    for (j, &c) in text.iter().enumerate() {
        let eq = base_code(c).map_or(0, |k| peq[k]);
        let xv = eq | mv;
        let xh = ((eq & pv).wrapping_add(pv) ^ pv) | eq;
        let ph = mv | !(xh | pv);
        let mh = pv & xh;
        if ph & last != 0 {
            score += 1;
        } else if mh & last != 0 {
            score -= 1;
        }
        // Global alignment: the top row grows by one per column, so a +1 is
        // carried in (a substring search would shift in 0).
        let ph = (ph << 1) | 1;
        let mh = mh << 1;
        pv = mh | !(xv | ph);
        mv = ph & xv;
        if score.saturating_sub((text.len() - 1 - j) as u32) >= limit {
            return Some(None);
        }
    }
    Some((score < limit).then_some(score))
}

/// Textbook two-row DP: the fallback for what the bit-vector cannot take.
fn levenshtein_dp(a: &[u8], b: &[u8]) -> u32 {
    let mut prev: Vec<u32> = (0..=b.len() as u32).collect();
    let mut cur = vec![0u32; b.len() + 1];
    for (i, &x) in a.iter().enumerate() {
        cur[0] = i as u32 + 1;
        for (j, &y) in b.iter().enumerate() {
            let sub = prev[j] + u32::from(x != y);
            cur[j + 1] = sub.min(prev[j + 1] + 1).min(cur[j] + 1);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Textbook DP, kept apart from the production fallback so the oracle the
    /// bit-vector is checked against shares no code with it.
    fn lev_reference(a: &[u8], b: &[u8]) -> u32 {
        let mut prev: Vec<u32> = (0..=b.len() as u32).collect();
        let mut cur = vec![0u32; b.len() + 1];
        for (i, &x) in a.iter().enumerate() {
            cur[0] = i as u32 + 1;
            for (j, &y) in b.iter().enumerate() {
                let sub = prev[j] + u32::from(x != y);
                cur[j + 1] = sub.min(prev[j + 1] + 1).min(cur[j] + 1);
            }
            std::mem::swap(&mut prev, &mut cur);
        }
        prev[b.len()]
    }

    /// `match_sequence` as it read before the runner-up cap: every reference
    /// compared to its true optimum, nothing abandoned.
    fn match_uncapped(refs: &BarcodeRefs, query: &[u8]) -> Option<BarcodeMatch> {
        if refs.seqs.is_empty() {
            return None;
        }
        let (mut best, mut best_at) = (u32::MAX, 0usize);
        let mut second = u32::MAX;
        for (i, refseq) in refs.seqs.iter().enumerate() {
            let d = edit_distance(refseq, query);
            if d < best {
                second = best;
                best = d;
                best_at = i;
            } else if d < second {
                second = d;
            }
        }
        let second_best_dist = (refs.seqs.len() > 1).then_some(second);
        Some(BarcodeMatch {
            index: best_at,
            best_dist: best,
            second_best_dist,
            margin: second_best_dist.map(|s| s.saturating_sub(best)),
        })
    }

    /// Capping each comparison at the running runner-up must not change a single
    /// field of the result — including the tie-to-lowest-index rule and the
    /// runner-up distance the margin is computed from. This is the whole
    /// correctness argument for the cap, so it is checked against the previous
    /// implementation rather than against hand-picked expectations.
    #[test]
    fn capped_matching_is_identical_to_uncapped() {
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for trial in 0..400 {
            // Reference banks from 1 to 12 codes, so the single-reference and
            // ties-everywhere cases are both covered.
            let n_refs = 1 + (trial % 12);
            let len = 20 + (trial % 21);
            let pairs: Vec<(String, String)> = (0..n_refs)
                .map(|k| {
                    let s: String = (0..len)
                        .map(|_| b"ACGT"[(next() % 4) as usize] as char)
                        .collect();
                    (format!("bc{k}"), s)
                })
                .collect();
            let refs = BarcodeRefs::from_pairs(pairs.clone()).unwrap();

            for q in 0..6 {
                // Queries: mutated copies of a reference, plus pure noise.
                let query: Vec<u8> = if q < 4 {
                    let mut s = pairs[(next() as usize) % n_refs].1.clone().into_bytes();
                    for _ in 0..(q * 3) {
                        let p = (next() as usize) % s.len();
                        s[p] = b"ACGT"[(next() % 4) as usize];
                    }
                    s.truncate(s.len() - (next() as usize % 4));
                    s
                } else {
                    (0..len).map(|_| b"ACGT"[(next() % 4) as usize]).collect()
                };

                let got = refs.match_sequence(&query);
                let want = match_uncapped(&refs, &query);
                assert_eq!(got, want, "trial {trial}, query {q}: {query:?}");
            }
        }
    }

    /// The cap must never turn a genuine best hit into a miss, however tight the
    /// runner-up gets: an exact match is distance 0 and has to survive.
    #[test]
    fn exact_match_survives_the_tightest_cap() {
        let refs = BarcodeRefs::from_pairs([
            ("a", "ACGTACGTACGTACGTACGT"),
            ("b", "ACGTACGTACGTACGTACGA"),
            ("c", "TTTTTTTTTTTTTTTTTTTT"),
        ])
        .unwrap();
        // "b" is 1 away from "a", so by the time "c" is reached the cap is 1.
        let m = refs.match_sequence(b"ACGTACGTACGTACGTACGT").unwrap();
        assert_eq!((m.index, m.best_dist), (0, 0));
        assert_eq!(m.second_best_dist, Some(1));
        assert_eq!(m.margin, Some(1));
    }

    #[test]
    fn edit_distance_within_reports_only_below_the_limit() {
        // "abcde" vs "abXde" is distance 1.
        let (a, b) = (b"ACGTA".as_slice(), b"ACTTA".as_slice());
        assert_eq!(edit_distance(a, b), 1);
        assert_eq!(edit_distance_within(a, b, 2), Some(1));
        // limit == the true distance means "cannot beat it" — no value.
        assert_eq!(edit_distance_within(a, b, 1), None);
        assert_eq!(edit_distance_within(a, b, 0), None);
        // An unbounded limit stays exact rather than overflowing the cap.
        assert_eq!(edit_distance_within(a, b, u32::MAX), Some(1));
        // Identical sequences are reachable at any non-zero limit.
        assert_eq!(edit_distance_within(a, a, 1), Some(0));
    }

    fn xorshift(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    /// Bit-vector against the DP over every length the single word can hold,
    /// including empty, prefix/suffix and related pairs, and over the bytes it
    /// hands to the fallback (non-ACGT, > 64 nt).
    #[test]
    fn bit_vector_matches_dp_on_random_pairs() {
        let mut st = 0xD1B5_4A32_D192_ED03u64;
        let alphabet: [&[u8]; 3] = [b"ACGT", b"AC", b"ACGTN"];
        for trial in 0..3000 {
            let alpha = alphabet[trial % 3];
            let la = (xorshift(&mut st) % 80) as usize;
            let a: Vec<u8> = (0..la)
                .map(|_| alpha[(xorshift(&mut st) as usize) % alpha.len()])
                .collect();
            // Half the time b is a mutated copy of a, so small distances (the
            // regime that matters) are well covered.
            let b: Vec<u8> = if trial % 2 == 0 {
                let mut b = a.clone();
                for _ in 0..(xorshift(&mut st) % 6) {
                    match (xorshift(&mut st) % 3, b.len()) {
                        (0, n) if n > 0 => b[(xorshift(&mut st) as usize) % n] = alpha[0],
                        (1, n) if n > 0 => {
                            b.remove((xorshift(&mut st) as usize) % n);
                        }
                        _ => b.insert(
                            (xorshift(&mut st) as usize) % (b.len() + 1),
                            alpha[(xorshift(&mut st) as usize) % alpha.len()],
                        ),
                    }
                }
                b
            } else {
                let lb = (xorshift(&mut st) % 80) as usize;
                (0..lb)
                    .map(|_| alpha[(xorshift(&mut st) as usize) % alpha.len()])
                    .collect()
            };
            let want = lev_reference(&a, &b);
            assert_eq!(edit_distance(&a, &b), want, "a={a:?} b={b:?}");
            assert_eq!(edit_distance(&b, &a), want, "symmetry a={a:?} b={b:?}");
            // `Some(d)` iff `d < limit`, at every limit around the true value.
            for limit in [0, 1, want.saturating_sub(1), want, want + 1, u32::MAX] {
                let expect = (want < limit).then_some(want);
                assert_eq!(
                    edit_distance_within(&a, &b, limit),
                    expect,
                    "limit {limit} want {want} a={a:?} b={b:?}"
                );
            }
        }
    }

    /// Two equal non-ACGT bytes must still match; a four-symbol table would
    /// call them different.
    #[test]
    fn equal_ambiguous_bases_are_not_edits() {
        assert_eq!(edit_distance(b"ACNGT", b"ACNGT"), 0);
        assert_eq!(edit_distance(b"ACNGT", b"ACTGT"), 1);
    }

    /// A pattern longer than one word takes the DP and stays exact.
    #[test]
    fn long_patterns_fall_back_to_the_dp() {
        let a = b"ACGT".repeat(20);
        let mut b = a.clone();
        b[6] = b'T'; // index 6 is a G: a real substitution
        b.remove(40);
        assert_eq!(lev_reference(&a, &b), 2);
        assert_eq!(edit_distance(&a, &b), 2);
        assert_eq!(edit_distance_within(&a, &b, 2), None);
        assert_eq!(edit_distance_within(&a, &b, 3), Some(lev_reference(&a, &b)));
    }

    fn write_csv(body: &str) -> tempfile::NamedTempFile {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(body.as_bytes()).unwrap();
        f.flush().unwrap();
        f
    }

    #[test]
    fn edit_distance_matches_a_reference_dp() {
        // Deterministic pseudo-random sequences over a range of divergences.
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..300 {
            let la = 20 + (next() % 25) as usize;
            let lb = 20 + (next() % 25) as usize;
            let a: Vec<u8> = (0..la).map(|_| b"ACGT"[(next() % 4) as usize]).collect();
            let b: Vec<u8> = (0..lb).map(|_| b"ACGT"[(next() % 4) as usize]).collect();
            assert_eq!(
                edit_distance(&a, &b),
                lev_reference(&a, &b),
                "a={:?} b={:?}",
                String::from_utf8_lossy(&a),
                String::from_utf8_lossy(&b)
            );
        }
    }

    #[test]
    fn identical_sequences_are_distance_zero() {
        assert_eq!(edit_distance(b"ACGTACGT", b"ACGTACGT"), 0);
        assert_eq!(edit_distance(b"", b""), 0);
        assert_eq!(edit_distance(b"ACGT", b""), 4);
    }

    #[test]
    fn matches_the_closest_reference_and_reports_the_margin() {
        let f = write_csv(
            "name,sequence\n\
             bc01,ACGTACGTACGTAAAA\n\
             bc02,TTTTTTTTTTTTTTTT\n\
             bc03,ACGTACGTACGTCCCC\n",
        );
        let refs = BarcodeRefs::from_csv(f.path()).unwrap();
        assert_eq!(refs.len(), 3);

        // One substitution from bc01; bc03 differs from bc01 in 4 places.
        let m = refs.match_sequence(b"ACGTACGTACGTAAAG").unwrap();
        assert_eq!(refs.name(m.index), "bc01");
        assert_eq!(m.best_dist, 1);
        assert_eq!(m.second_best_dist, Some(4));
        assert_eq!(m.margin, Some(3));
    }

    #[test]
    fn a_tie_reports_zero_margin_rather_than_picking_silently() {
        let f = write_csv(
            "name,sequence\n\
             bcA,AAAACCCC\n\
             bcB,TTTTGGGG\n",
        );
        let refs = BarcodeRefs::from_csv(f.path()).unwrap();
        // Equidistant from both.
        let m = refs.match_sequence(b"AAAAGGGG").unwrap();
        assert_eq!(m.best_dist, 4);
        assert_eq!(m.margin, Some(0), "an ambiguous read must show margin 0");
        assert_eq!(
            m.index, 0,
            "ties resolve to the lowest index, deterministically"
        );
    }

    #[test]
    fn single_reference_has_no_margin() {
        let f = write_csv("name,sequence\nonly,ACGTACGT\n");
        let refs = BarcodeRefs::from_csv(f.path()).unwrap();
        let m = refs.match_sequence(b"ACGTACGT").unwrap();
        assert_eq!(m.best_dist, 0);
        assert_eq!(m.second_best_dist, None);
        assert_eq!(m.margin, None);
    }

    #[test]
    fn min_pairwise_distance_reports_the_design_floor() {
        let f = write_csv(
            "name,sequence\n\
             a,AAAAAAAA\n\
             b,AAAAAAAT\n\
             c,TTTTTTTT\n",
        );
        let refs = BarcodeRefs::from_csv(f.path()).unwrap();
        assert_eq!(refs.min_pairwise_distance(), Some(1));
    }

    #[test]
    fn extra_columns_are_ignored_and_case_is_normalised() {
        let f = write_csv(
            "type,name,length,sequence\n\
             barcode_strand,bc01,8,acgtacgt\n",
        );
        let refs = BarcodeRefs::from_csv(f.path()).unwrap();
        assert_eq!(refs.name(0), "bc01");
        assert_eq!(refs.match_sequence(b"ACGTACGT").unwrap().best_dist, 0);
    }

    #[test]
    fn rejects_malformed_reference_files() {
        let f = write_csv("name,seq\nbc01,ACGT\n");
        assert!(matches!(
            BarcodeRefs::from_csv(f.path()),
            Err(BarcodeError::MissingColumns { .. })
        ));

        let f = write_csv("name,sequence\nbc01,ACGTN\n");
        assert!(matches!(
            BarcodeRefs::from_csv(f.path()),
            Err(BarcodeError::BadSequence { .. })
        ));

        let f = write_csv("name,sequence\nbc01,ACGT\nbc01,TGCA\n");
        assert!(matches!(
            BarcodeRefs::from_csv(f.path()),
            Err(BarcodeError::DuplicateName { .. })
        ));

        let f = write_csv("name,sequence\n");
        assert!(matches!(
            BarcodeRefs::from_csv(f.path()),
            Err(BarcodeError::Empty { .. })
        ));
    }
}
