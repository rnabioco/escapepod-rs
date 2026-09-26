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
use crate::simd::{self, Backend, Kernel, Profile};

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

/// Panel scores for a batch of queries, computed ahead of [`Aligner::map_reads_scored`]:
/// row `k` is query `k` against every reference in panel order, or absent
/// when that query was left for the CPU (see `cuda::GpuScorer::score_batch`).
///
/// Plain data, compiled in every build, so a caller can carry one through a
/// pipeline without caring whether a GPU produced it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ScoreMatrix {
    n_refs: usize,
    /// `scores[k * n_refs + r]`; rows of unscored queries are zero.
    scores: Vec<i16>,
    scored: Vec<bool>,
}

impl ScoreMatrix {
    /// Assemble from a row-major `scores` of `scored.len()` rows.
    ///
    /// # Panics
    /// If `scores` is not `scored.len() * n_refs` long.
    pub fn from_parts(n_refs: usize, scores: Vec<i16>, scored: Vec<bool>) -> Self {
        assert_eq!(scores.len(), scored.len() * n_refs, "row-major matrix");
        Self {
            n_refs,
            scores,
            scored,
        }
    }

    /// Number of queries (rows), scored or not.
    pub fn len(&self) -> usize {
        self.scored.len()
    }

    pub fn is_empty(&self) -> bool {
        self.scored.is_empty()
    }

    /// References per row.
    pub fn n_refs(&self) -> usize {
        self.n_refs
    }

    /// How many rows carry scores.
    pub fn n_scored(&self) -> usize {
        self.scored.iter().filter(|&&s| s).count()
    }

    /// Query `k`'s scores, or `None` when it was not scored.
    pub fn row(&self, k: usize) -> Option<&[i16]> {
        self.scored[k].then(|| &self.scores[k * self.n_refs..(k + 1) * self.n_refs])
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
    /// Reads at least this long take [`Kernel::Transposed`]; `None`: none do.
    transposed_min_len: Option<usize>,
}

/// References per unit of [`Aligner::score_group`] when a read is scored by
/// the scalar path (which has no lane groups): the SIMD kernels' widest group.
const SCALAR_GROUP: usize = 32;

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
            transposed_min_len: Some(simd::TRANSPOSED_MIN_READ_LEN),
        })
    }

    /// Score reads of at least `min_len` bases with the reference-major
    /// ([`Kernel::Transposed`]) SIMD kernel, and shorter ones with the
    /// row-major one; `None` keeps every read on the row-major kernel. The
    /// default is [`simd::TRANSPOSED_MIN_READ_LEN`]. Both kernels give the
    /// same scores — this only moves time — so it is the A/B lever, not a
    /// choice a caller needs to make.
    pub fn with_transposed_min_len(mut self, min_len: Option<usize>) -> Self {
        self.transposed_min_len = min_len;
        self
    }

    /// The read length from which the transposed kernel scores, if any.
    pub fn transposed_min_len(&self) -> Option<usize> {
        self.transposed_min_len
    }

    /// The SIMD profile, when a read of `len` bases is scored by a SIMD
    /// kernel at all (not the scalar backend, and inside the i16 bound).
    fn simd_profile(&self, len: usize) -> Option<&Profile> {
        self.profile
            .as_ref()
            .filter(|_| simd::fits_i16(len, self.panel.max_len(), &self.scoring))
    }

    /// Which SIMD loop order scores a read of `len` bases; `None` when the
    /// scalar path does.
    pub fn kernel_for(&self, len: usize) -> Option<Kernel> {
        self.simd_profile(len)?;
        Some(match self.transposed_min_len {
            Some(min) if len >= min => Kernel::Transposed,
            _ => Kernel::RowMajor,
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
        match self.kernel_for(query.len()) {
            Some(kernel) => self.score_simd(query, kernel, out),
            None => simd::score_scalar(&self.panel, query, &self.scoring, self.mode, out),
        }
    }

    /// [`Aligner::score_all`] with the SIMD loop order forced, for tests that
    /// pin one kernel against the other. Falls back to the scalar path where
    /// `score_all` would.
    #[cfg(test)]
    pub(crate) fn score_all_with(&self, query: &[u8], kernel: Kernel, out: &mut Vec<i32>) {
        out.clear();
        out.resize(self.panel.len(), 0);
        if query.is_empty() {
            return;
        }
        match self.kernel_for(query.len()) {
            Some(_) => self.score_simd(query, kernel, out),
            None => simd::score_scalar(&self.panel, query, &self.scoring, self.mode, out),
        }
    }

    /// The SIMD profile, for tests that look inside it.
    #[cfg(test)]
    pub(crate) fn profile_for_tests(&self) -> &Profile {
        self.profile.as_ref().expect("a SIMD backend has a profile")
    }

    fn score_simd(&self, query: &[u8], kernel: Kernel, out: &mut [i32]) {
        let p = self
            .profile
            .as_ref()
            .expect("kernel_for checked the profile");
        simd::score_profile(
            self.backend,
            p,
            query,
            &self.scoring,
            self.mode,
            kernel,
            out,
        )
    }

    /// Into how many independent units [`Aligner::score_group`] splits the
    /// panel for a read of `query_len` bases: the SIMD kernel's lane groups,
    /// or runs of references when the scalar path scores it.
    ///
    /// A caller with threads to spare can score one long read's units on
    /// different threads and assemble the row; this crate itself does not
    /// spawn anything.
    pub fn score_groups(&self, query_len: usize) -> usize {
        match self.simd_profile(query_len) {
            Some(p) => p.groups.len(),
            None => self.panel.len().div_ceil(SCALAR_GROUP),
        }
    }

    /// Score `query` (codes) against the references of unit `group` (of
    /// [`Aligner::score_groups`]), replacing `out` with one `(panel index,
    /// score)` per reference in it. Every unit's scores together equal
    /// [`Aligner::score_all`]'s, reference for reference, whichever kernel
    /// the read's length selects.
    ///
    /// # Panics
    /// If `group` is not below `score_groups(query.len())`.
    pub fn score_group(&self, query: &[u8], group: usize, out: &mut Vec<(usize, i32)>) {
        out.clear();
        match self.kernel_for(query.len()) {
            Some(kernel) => {
                let p = self
                    .profile
                    .as_ref()
                    .expect("kernel_for checked the profile");
                assert!(group < p.groups.len(), "group {group} out of range");
                if query.is_empty() {
                    let refs = p.groups[group].refs.iter().filter(|&&r| r != usize::MAX);
                    out.extend(refs.map(|&r| (r, 0)));
                    return;
                }
                simd::score_profile_group(
                    self.backend,
                    p,
                    group,
                    query,
                    &self.scoring,
                    self.mode,
                    kernel,
                    |r, s| out.push((r, s)),
                );
            }
            None => {
                let start = group * SCALAR_GROUP;
                assert!(start < self.panel.len(), "group {group} out of range");
                let end = (start + SCALAR_GROUP).min(self.panel.len());
                out.extend((start..end).map(|r| {
                    let codes = &self.panel.get(r).codes;
                    let s = if query.is_empty() {
                        0
                    } else {
                        crate::scalar::score(query, codes, &self.scoring, self.mode)
                    };
                    (r, s)
                }));
            }
        }
    }

    /// `row` widened into `out` when the caller has it, else
    /// [`Aligner::score_all`].
    fn score_or_take(&self, query: &[u8], row: Option<&[i16]>, out: &mut Vec<i32>) {
        match row {
            Some(row) => {
                assert_eq!(row.len(), self.panel.len(), "one score per reference");
                out.clear();
                out.extend(row.iter().map(|&s| i32::from(s)));
            }
            None => self.score_all(query, out),
        }
    }

    /// Align one read (letters, as they will be written to SAM).
    pub fn map_read(&self, read: &[u8], opts: &MapOptions) -> ReadMapping {
        self.map_reads(&[read], opts)
            .pop()
            .expect("one mapping per read")
    }

    /// Align several reads, one [`ReadMapping`] each, in order.
    ///
    /// Equivalent to [`Aligner::map_read`] on each, and the way to call it
    /// when there are many: every read's panel is scored on its own, but the
    /// winners of all of them are traced together, so the traceback kernels'
    /// lanes fill with pairs from different reads (see
    /// `simd::align_pairs`). A few dozen reads per call is enough to fill
    /// them.
    pub fn map_reads(&self, reads: &[&[u8]], opts: &MapOptions) -> Vec<ReadMapping> {
        self.map_reads_scored(reads, opts, |_, _| None)
    }

    /// [`Aligner::map_reads`], with some or all of the panel scores already
    /// computed elsewhere (the CUDA score kernel, `cuda::GpuScorer`).
    ///
    /// `prescored(k, reverse)` is read `k`'s score against every reference,
    /// in panel order, for the forward strand or (with
    /// [`MapOptions::both_strands`]) the reverse complement; `None` scores
    /// that one here, with this aligner's kernel. Everything after the scores
    /// — the tie set, the tracebacks, `MD`/`NM` — is the same code as
    /// [`Aligner::map_reads`], so equal scores give an equal result, field
    /// for field. The scores are the caller's contract: they must be exactly
    /// what [`Aligner::score_all`] would compute (`GpuScorer` is pinned to the
    /// scalar oracle by equality).
    pub fn map_reads_scored<'s>(
        &self,
        reads: &[&[u8]],
        opts: &MapOptions,
        prescored: impl Fn(usize, bool) -> Option<&'s [i16]>,
    ) -> Vec<ReadMapping> {
        /// One read after scoring, before tracing.
        struct Scored {
            fwd: Vec<u8>,
            rev: Vec<u8>,
            best: Option<i32>,
            n_tied: usize,
            suboptimal: Option<i32>,
            /// The tie-set members to trace, `(panel index, reverse)`; empty
            /// when the read is unmapped before any traceback.
            take: Vec<(usize, bool)>,
        }

        let mut scores = Vec::new();
        let mut rev_scores = Vec::new();
        let scored: Vec<Scored> = reads
            .iter()
            .enumerate()
            .map(|(k, read)| {
                let fwd = encode(read);
                if fwd.is_empty() {
                    return Scored {
                        fwd,
                        rev: Vec::new(),
                        best: None,
                        n_tied: 0,
                        suboptimal: None,
                        take: Vec::new(),
                    };
                }
                self.score_or_take(&fwd, prescored(k, false), &mut scores);
                let rev = if opts.both_strands {
                    let rev = alphabet::reverse_complement_codes(&fwd);
                    self.score_or_take(&rev, prescored(k, true), &mut rev_scores);
                    rev
                } else {
                    rev_scores.clear();
                    Vec::new()
                };
                // Candidates in tie order: reference order, forward before reverse.
                let candidates = || {
                    (0..self.panel.len()).flat_map(|r| {
                        let f = std::iter::once((r, false, scores[r]));
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
                let n_tied = ties.len();
                let take = if best < opts.min_score {
                    Vec::new()
                } else {
                    let k = opts
                        .max_ties
                        .map_or(n_tied, |k| n_tied.min(k.saturating_add(1)));
                    ties.into_iter().take(k).collect()
                };
                Scored {
                    fwd,
                    rev,
                    best: Some(best),
                    n_tied,
                    suboptimal,
                    take,
                }
            })
            .collect();

        // Trace every taken tie of every read in one batch.
        let pairs: Vec<(&[u8], &[u8])> = scored
            .iter()
            .flat_map(|sc| {
                sc.take.iter().map(move |&(r, reverse)| {
                    let q: &[u8] = if reverse { &sc.rev } else { &sc.fwd };
                    (q, &self.panel.get(r).codes[..])
                })
            })
            .collect();
        let mut traced =
            simd::align_pairs(self.backend, &pairs, &self.scoring, self.mode).into_iter();

        scored
            .iter()
            .zip(reads)
            .map(|(sc, &read)| {
                let alignments: Vec<Alignment> = traced.by_ref().take(sc.take.len()).collect();
                let unmapped = ReadMapping {
                    best_score: sc.best,
                    n_tied: sc.n_tied,
                    suboptimal: sc.suboptimal,
                    hits: Vec::new(),
                };
                if sc.take.is_empty() || alignments[0].is_empty() {
                    // Below --min-score, or a local score of zero: every tie
                    // then traces to nothing, and the read aligns nowhere.
                    return unmapped;
                }
                let rev_letters = if sc.take.iter().any(|&(_, rv)| rv) {
                    alphabet::reverse_complement(read)
                } else {
                    Vec::new()
                };
                let hits = sc
                    .take
                    .iter()
                    .zip(alignments)
                    .map(|(&(r, reverse), alignment)| {
                        debug_assert_eq!(
                            Some(alignment.score),
                            sc.best,
                            "traceback disagrees with the kernel"
                        );
                        let reference = self.panel.get(r);
                        let letters: &[u8] = if reverse { &rev_letters } else { read };
                        let (md, nm) = sam::md_nm(
                            letters,
                            &reference.seq,
                            alignment.query_start,
                            alignment.ref_start,
                            &alignment.ops,
                        );
                        Hit {
                            reference: r,
                            reverse,
                            alignment,
                            md,
                            nm,
                        }
                    })
                    .collect();
                ReadMapping {
                    best_score: sc.best,
                    n_tied: sc.n_tied,
                    suboptimal: sc.suboptimal,
                    hits,
                }
            })
            .collect()
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

    /// Tracing many reads' winners together must not change any of them.
    #[test]
    fn batched_mapping_equals_one_at_a_time() {
        let reads: Vec<&[u8]> = vec![
            b"ACGTACGTACGT",
            b"CCCCACGTACGTACGTAAAA",
            b"TTTTTTTTTTTT",
            b"",
            b"GGGGACGTACGTACGTTTTT",
            b"NNNN",
        ];
        for backend in Backend::available() {
            for mode in [Mode::Local, Mode::SemiGlobal] {
                let a = Aligner::with_backend(panel(), Scoring::default(), mode, backend).unwrap();
                let opts = MapOptions {
                    both_strands: true,
                    ..MapOptions::default()
                };
                let batched = a.map_reads(&reads, &opts);
                for (r, b) in reads.iter().zip(&batched) {
                    assert_eq!(&a.map_read(r, &opts), b, "{}", backend.name());
                }
            }
        }
    }

    /// Scores handed in from outside take exactly the path computed ones do,
    /// and a `None` row is scored here.
    #[test]
    fn prescored_rows_equal_computed_ones() {
        let reads: Vec<&[u8]> = vec![b"ACGTACGTACGT", b"CCCCACGTACGTACGTAAAA", b"", b"NNNN"];
        let a = Aligner::new(panel(), Scoring::default(), Mode::Local).unwrap();
        let opts = MapOptions {
            both_strands: true,
            ..MapOptions::default()
        };
        let n = a.panel().len();
        let (mut scores, mut scored) = (Vec::new(), Vec::new());
        for rev in [false, true] {
            for (k, r) in reads.iter().enumerate() {
                let mut q = encode(r);
                if rev {
                    q = alphabet::reverse_complement_codes(&q);
                }
                let mut row = Vec::new();
                a.score_all(&q, &mut row);
                scores.extend(row.iter().map(|&s| s as i16));
                // Leave one row for the aligner to score itself.
                scored.push(k != 1);
            }
        }
        let m = ScoreMatrix::from_parts(n, scores, scored);
        assert_eq!(m.n_scored(), 6);
        let got = a.map_reads_scored(&reads, &opts, |k, rev| {
            m.row(k + if rev { reads.len() } else { 0 })
        });
        assert_eq!(got, a.map_reads(&reads, &opts));
    }

    /// Every unit of the per-group API, assembled, equals the whole-panel
    /// score: both modes, every backend, short reads (row-major) and a long
    /// one (transposed), on a panel wide enough for several lane groups.
    #[test]
    fn per_group_scores_equal_score_all() {
        let mut x = 0x9e37_79b9_7f4a_7c15u64;
        let mut base = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            b"ACGT"[(x % 4) as usize]
        };
        let refs: Vec<(String, Vec<u8>)> = (0..70)
            .map(|k| (format!("r{k}"), (0..40 + k).map(|_| base()).collect()))
            .collect();
        let panel = Panel::new(refs).unwrap();
        let mut reads: Vec<Vec<u8>> = vec![Vec::new(), b"ACGTACGT".to_vec()];
        for len in [30, 95, 180] {
            reads.push((0..len).map(|_| base()).collect());
        }
        let mut long: Vec<u8> = (0..simd::TRANSPOSED_MIN_READ_LEN + 777)
            .map(|_| base())
            .collect();
        long[500..545].copy_from_slice(&panel.get(5).seq);
        reads.push(long);
        for backend in Backend::available() {
            for mode in [Mode::Local, Mode::SemiGlobal] {
                let a = Aligner::with_backend(panel.clone(), Scoring::default(), mode, backend)
                    .unwrap();
                for read in &reads {
                    let q = encode(read);
                    let mut want = Vec::new();
                    a.score_all(&q, &mut want);
                    let mut got = vec![None; panel.len()];
                    let mut unit = Vec::new();
                    for g in 0..a.score_groups(q.len()) {
                        a.score_group(&q, g, &mut unit);
                        for &(r, s) in &unit {
                            assert!(got[r].replace(s).is_none(), "reference {r} twice");
                        }
                    }
                    let got: Vec<i32> = got
                        .into_iter()
                        .map(|s| s.expect("every reference"))
                        .collect();
                    assert_eq!(
                        got,
                        want,
                        "{} {} len {}",
                        backend.name(),
                        mode.name(),
                        q.len()
                    );
                }
            }
        }
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
