//! Matching a decoded sequence to a barcode reference by edit distance.
//!
//! The last step of basecall-then-match demultiplexing, and the one that turns
//! [`super::lattice`]'s sequences into barcode calls. Alignment comes from
//! `fqxv-align`'s wavefront implementation rather than a local Levenshtein: WFA's
//! work scales with the edit *distance* rather than sequence length, which is
//! the right shape here because a decode sits ~4 edits from its own reference
//! and ~10+ from every other one, so most of the 96 comparisons abandon almost
//! immediately.
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

use fqxv_align::wfa_align;

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

    pub fn len(&self) -> usize {
        self.names.len()
    }

    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }

    pub fn name(&self, index: usize) -> &str {
        &self.names[index]
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
            let d = edit_distance(refseq, query);
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

/// Exact Levenshtein distance via WFA.
///
/// The cap is set to the longest possible optimal score — one full deletion
/// plus one full insertion cannot be beaten by any alignment — so WFA never
/// truncates and this is exact, while the cap still bounds its `O(s²)`
/// traceback storage. `wfa_align`'s `dist` is substitutions plus inserted plus
/// deleted bases, i.e. unit-cost Levenshtein, which is what the reference
/// implementation computes.
fn edit_distance(a: &[u8], b: &[u8]) -> u32 {
    let cap = (a.len() + b.len()) as u32;
    wfa_align(a, b, cap).dist
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Textbook DP, to check the WFA wrapper actually computes Levenshtein and
    /// that the cap never truncates.
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
