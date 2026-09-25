// SPDX-License-Identifier: MIT

//! Winner selection: score the panel, take the tie set, trace the winners.
//!
//! # Semantics
//!
//! * The best score *S* is the maximum over every reference (and, with
//!   [`MapOptions::both_strands`], both orientations of the read).
//! * The **tie set** is every `(reference, strand)` scoring *S*, in reference
//!   order, forward before reverse. Its first member is the primary.
//! * The **suboptimal** score (SAM `XS`) is the best score *outside* the tie
//!   set; absent when every candidate is in it.
//! * A read whose *S* is below [`MapOptions::min_score`], or whose best
//!   alignment aligns no base at all (a local score of zero), is unmapped.
//! * Tracebacks are computed for the primary and for up to
//!   [`MapOptions::max_ties`] further tie-set members — never for losers.

use crate::AlignError;
use crate::alphabet::{self, encode};
use crate::panel::Panel;
use crate::sam;
use crate::scalar::Alignment;
use crate::scoring::{Mode, Scoring};
use crate::simd::{self, Backend, Profile};

/// Per-read policy. The default: minimum score 0, every tie traced, forward
/// strand only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MapOptions {
    /// A read whose best score is below this is unmapped.
    pub min_score: i32,
    /// How many tie-set members *besides the primary* get a traceback (and so
    /// an `XA` entry / secondary record). `None` = all of them.
    pub max_ties: Option<usize>,
    /// Also score the reverse complement of the read.
    pub both_strands: bool,
}

/// One traced alignment of a read to one reference.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Hit {
    /// Panel index of the reference.
    pub reference: usize,
    /// Whether the read's reverse complement is what aligned.
    pub reverse: bool,
    /// Coordinates are on the read *as aligned*: reverse-complemented when
    /// `reverse`, which is also how SAM stores it.
    pub alignment: Alignment,
    /// `samtools calmd`-compatible `MD`.
    pub md: String,
    /// `samtools calmd`-compatible `NM`.
    pub nm: u32,
}

/// Everything one read's alignment produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReadMapping {
    /// Best score over the panel; `None` only for an empty read.
    pub best_score: Option<i32>,
    /// Size of the tie set (≥ 1 whenever `best_score` is set).
    pub n_tied: usize,
    /// Best score outside the tie set (SAM `XS`).
    pub suboptimal: Option<i32>,
    /// Primary first, then up to `max_ties` more tie-set members in order.
    /// Empty when the read is unmapped.
    pub hits: Vec<Hit>,
}

impl ReadMapping {
    pub fn is_mapped(&self) -> bool {
        !self.hits.is_empty()
    }

    /// Mapped to exactly one reference (and strand).
    pub fn is_unique(&self) -> bool {
        self.is_mapped() && self.n_tied == 1
    }

    /// The primary hit, if mapped.
    pub fn primary(&self) -> Option<&Hit> {
        self.hits.first()
    }
}

/// A panel prepared for one scoring, one mode, and one kernel.
///
/// Cheap to share across threads (`&Aligner` is all [`Aligner::map_read`]
/// needs); per-thread DP scratch is kept thread-local inside the kernels.
#[derive(Debug, Clone)]
pub struct Aligner {
    panel: Panel,
    scoring: Scoring,
    mode: Mode,
    backend: Backend,
    profile: Option<Profile>,
}

impl Aligner {
    /// Prepare `panel` with the fastest kernel this machine runs (under any
    /// `ESCAPEPOD_ALIGN_BACKEND` cap).
    pub fn new(panel: Panel, scoring: Scoring, mode: Mode) -> Result<Self, AlignError> {
        Self::with_backend(panel, scoring, mode, Backend::best())
    }

    /// Prepare `panel` for a named kernel. Refuses one the CPU cannot run.
    pub fn with_backend(
        panel: Panel,
        scoring: Scoring,
        mode: Mode,
        backend: Backend,
    ) -> Result<Self, AlignError> {
        scoring.validate()?;
        if !backend.supported() {
            return Err(AlignError::BackendUnavailable(backend.name()));
        }
        let profile = (backend != Backend::Scalar && panel.max_len() < 32_000)
            .then(|| Profile::build(&panel, &scoring, backend.lanes()));
        Ok(Self {
            panel,
            scoring,
            mode,
            backend,
            profile,
        })
    }

    pub fn panel(&self) -> &Panel {
        &self.panel
    }

    pub fn scoring(&self) -> &Scoring {
        &self.scoring
    }

    pub fn mode(&self) -> Mode {
        self.mode
    }

    pub fn backend(&self) -> Backend {
        self.backend
    }

    /// Score `query` (codes) against every reference, one score per panel
    /// index into `out` (resized to the panel).
    pub fn score_all(&self, query: &[u8], out: &mut Vec<i32>) {
        out.clear();
        out.resize(self.panel.len(), 0);
        if query.is_empty() {
            return;
        }
        match &self.profile {
            Some(p) if simd::fits_i16(query.len(), self.panel.max_len(), &self.scoring) => {
                simd::score_profile(self.backend, p, query, &self.scoring, self.mode, out)
            }
            _ => simd::score_scalar(&self.panel, query, &self.scoring, self.mode, out),
        }
    }

    /// Align one read (letters, as they will be written to SAM).
    pub fn map_read(&self, read: &[u8], opts: &MapOptions) -> ReadMapping {
        let unmapped = |best_score, n_tied, suboptimal| ReadMapping {
            best_score,
            n_tied,
            suboptimal,
            hits: Vec::new(),
        };
        let fwd = encode(read);
        if fwd.is_empty() {
            return unmapped(None, 0, None);
        }
        let mut fwd_scores = Vec::new();
        self.score_all(&fwd, &mut fwd_scores);
        let (rev, rev_scores) = if opts.both_strands {
            let rev = alphabet::reverse_complement_codes(&fwd);
            let mut s = Vec::new();
            self.score_all(&rev, &mut s);
            (rev, s)
        } else {
            (Vec::new(), Vec::new())
        };

        // Candidates in tie order: reference order, forward before reverse.
        let candidates = || {
            (0..self.panel.len()).flat_map(|r| {
                let f = std::iter::once((r, false, fwd_scores[r]));
                let b = rev_scores.get(r).map(|&s| (r, true, s));
                f.chain(b)
            })
        };
        let best = candidates()
            .map(|(_, _, s)| s)
            .max()
            .expect("non-empty panel");
        let ties: Vec<(usize, bool)> = candidates()
            .filter(|&(_, _, s)| s == best)
            .map(|(r, rv, _)| (r, rv))
            .collect();
        let suboptimal = candidates()
            .filter(|&(_, _, s)| s != best)
            .map(|(_, _, s)| s)
            .max();
        if best < opts.min_score {
            return unmapped(Some(best), ties.len(), suboptimal);
        }

        let take = opts
            .max_ties
            .map_or(ties.len(), |k| ties.len().min(k.saturating_add(1)));
        let rev_letters = if ties[..take].iter().any(|&(_, rv)| rv) {
            alphabet::reverse_complement(read)
        } else {
            Vec::new()
        };
        // Trace the taken ties, forward and reverse separately (different
        // query codes), each batch through the SIMD traceback kernels.
        let taken = &ties[..take];
        let mut alignments: Vec<Option<Alignment>> = vec![None; take];
        for reverse in [false, true] {
            let slots: Vec<usize> = (0..take).filter(|&k| taken[k].1 == reverse).collect();
            if slots.is_empty() {
                continue;
            }
            let codes: &[u8] = if reverse { &rev } else { &fwd };
            let refs: Vec<&[u8]> = slots
                .iter()
                .map(|&k| &self.panel.get(taken[k].0).codes[..])
                .collect();
            let traced = simd::align_many(self.backend, codes, &refs, &self.scoring, self.mode);
            for (k, a) in slots.into_iter().zip(traced) {
                alignments[k] = Some(a);
            }
        }
        let mut hits = Vec::with_capacity(take);
        for (&(r, reverse), alignment) in taken.iter().zip(alignments) {
            let alignment = alignment.expect("every taken tie was traced");
            debug_assert_eq!(alignment.score, best, "traceback disagrees with the kernel");
            if alignment.is_empty() {
                // Only a local score of zero traces to nothing, and then every
                // tie does: the read aligns nowhere.
                return unmapped(Some(best), ties.len(), suboptimal);
            }
            let reference = self.panel.get(r);
            let letters: &[u8] = if reverse { &rev_letters } else { read };
            let (md, nm) = sam::md_nm(
                letters,
                &reference.seq,
                alignment.query_start,
                alignment.ref_start,
                &alignment.ops,
            );
            hits.push(Hit {
                reference: r,
                reverse,
                alignment,
                md,
                nm,
            });
        }
        ReadMapping {
            best_score: Some(best),
            n_tied: ties.len(),
            suboptimal,
            hits,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn panel() -> Panel {
        Panel::new([
            ("a", b"GGGGACGTACGTACGTTTTT".to_vec()),
            ("b", b"CCCCACGTACGTACGTAAAA".to_vec()),
            ("c", b"TTTTTTTTTTTTTTTTTTTT".to_vec()),
        ])
        .unwrap()
    }

    #[test]
    fn ties_are_reported_in_reference_order() {
        let a = Aligner::new(panel(), Scoring::default(), Mode::Local).unwrap();
        let m = a.map_read(b"ACGTACGTACGT", &MapOptions::default());
        assert_eq!(m.best_score, Some(24));
        assert_eq!(m.n_tied, 2);
        assert_eq!(m.hits.len(), 2);
        assert_eq!(m.hits[0].reference, 0);
        assert_eq!(m.hits[1].reference, 1);
        assert!(m.suboptimal.unwrap() < 24);
        let capped = a.map_read(
            b"ACGTACGTACGT",
            &MapOptions {
                max_ties: Some(0),
                ..MapOptions::default()
            },
        );
        assert_eq!(capped.n_tied, 2);
        assert_eq!(capped.hits.len(), 1);
    }

    #[test]
    fn min_score_unmaps() {
        let a = Aligner::new(panel(), Scoring::default(), Mode::Local).unwrap();
        let m = a.map_read(
            b"ACGTACGTACGT",
            &MapOptions {
                min_score: 1000,
                ..MapOptions::default()
            },
        );
        assert!(!m.is_mapped());
        assert_eq!(m.best_score, Some(24));
    }

    #[test]
    fn reverse_strand_found_when_asked() {
        let a = Aligner::new(panel(), Scoring::default(), Mode::Local).unwrap();
        // Reverse complement of "CCCCACGTACGTACGTAAAA"'s tail, unique to "b".
        let read = alphabet::reverse_complement(b"CGTACGTAAAA");
        let fwd_only = a.map_read(&read, &MapOptions::default());
        let both = a.map_read(
            &read,
            &MapOptions {
                both_strands: true,
                ..MapOptions::default()
            },
        );
        assert!(both.best_score > fwd_only.best_score);
        assert_eq!(both.hits[0].reference, 1);
        assert!(both.hits[0].reverse);
        assert_eq!(both.hits[0].md, "11");
    }
}
