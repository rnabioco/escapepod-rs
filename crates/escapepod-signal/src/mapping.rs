//! Move-table and CIGAR coordinate mapping.
//!
//! [`refine_signal_map`] takes a sequence→signal map as its primary input, but
//! a basecalled BAM record does not hand you one: it hands you a move table
//! (`mv`/`ns`/`ts`) and a CIGAR, and the map is what you get after applying two
//! Oxford Nanopore conventions that are short enough to retype and subtle
//! enough to retype wrongly. This module is those two conventions, written
//! once.
//!
//! Getting either one slightly wrong is invisible downstream: refinement,
//! per-base statistics, and every model fitted on them still produce a number,
//! just for a different set of samples than the caller thinks. The failure mode
//! is a silently shifted answer, never an error, which is why these live here
//! rather than in each consumer.
//!
//! * [`seq_to_signal_from_moves`] — the `mv`/`ns`/`ts` tags to a query→signal
//!   map. Remora's `query_to_signal = np.nonzero(mv)[0] * stride`.
//! * [`ref_to_signal`] — a query→signal map plus a CIGAR to a reference→signal
//!   map, by the Remora knot convention.
//!
//! What a *base* maps to once you have the map is still the caller's business
//! — see [`crate::features`], which takes spans rather than deriving them.
//!
//! [`refine_signal_map`]: crate::resquiggle::refine_signal_map

/// One CIGAR operation kind.
///
/// A local, dependency-free mirror of the SAM operations. Callers holding
/// noodles (or htslib, or pysam) records convert at their own boundary; this
/// crate deliberately does not take an alignment-library dependency for the
/// sake of nine variants.
///
/// Spelled as a named kind rather than the numeric op code it is stored as,
/// because a bare integer pair is exactly the thing a caller transposes
/// (`(len, op)` for `(op, len)`) without the compiler noticing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CigarKind {
    /// `M` — alignment match or mismatch. Consumes query and reference.
    Match,
    /// `=` — sequence match. Consumes query and reference.
    SequenceMatch,
    /// `X` — sequence mismatch. Consumes query and reference.
    SequenceMismatch,
    /// `I` — insertion to the reference. Consumes query only.
    Insertion,
    /// `D` — deletion from the reference. Consumes reference only.
    Deletion,
    /// `N` — skipped region on the reference. Consumes reference only.
    Skip,
    /// `S` — soft clip. Consumes query only (the bases are still in the
    /// record's sequence, and still covered by the move table).
    SoftClip,
    /// `H` — hard clip. Consumes neither.
    HardClip,
    /// `P` — padding. Consumes neither.
    Pad,
}

impl CigarKind {
    /// Whether this op aligns a query base to a reference base (`M`, `=`, `X`).
    ///
    /// This is the predicate the knot convention is built on: only these ops
    /// carry the 1:1 integer correspondence that makes a lookup exact.
    #[must_use]
    pub fn is_match(self) -> bool {
        matches!(
            self,
            Self::Match | Self::SequenceMatch | Self::SequenceMismatch
        )
    }

    /// Whether this op advances the reference position.
    #[must_use]
    pub fn consumes_reference(self) -> bool {
        matches!(
            self,
            Self::Match
                | Self::SequenceMatch
                | Self::SequenceMismatch
                | Self::Deletion
                | Self::Skip
        )
    }

    /// Whether this op advances the query position.
    #[must_use]
    pub fn consumes_query(self) -> bool {
        matches!(
            self,
            Self::Match
                | Self::SequenceMatch
                | Self::SequenceMismatch
                | Self::Insertion
                | Self::SoftClip
        )
    }
}

/// One CIGAR operation: a [`CigarKind`] and its run length.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CigarOp {
    /// The operation.
    pub kind: CigarKind,
    /// How many positions it runs for.
    pub len: u32,
}

impl CigarOp {
    /// Construct an op.
    #[must_use]
    pub fn new(kind: CigarKind, len: u32) -> Self {
        Self { kind, len }
    }
}

/// Build the query→signal map from a basecaller move table.
///
/// The move table (`mv` tag) has one entry per `stride`-sized block of signal;
/// a `1` means "this block starts a new base", a `0` means "the previous base
/// is still going". The map is therefore the positions of the `1`s scaled by
/// `stride`, with one extra entry closing the last base — so a read of `n`
/// bases yields `n + 1` boundaries and base `i` owns samples
/// `map[i]..map[i + 1]`. This is Remora's
/// `query_to_signal = np.nonzero(mv)[0] * stride`.
///
/// # Coordinate frame
///
/// **The returned map is in trimmed-signal coordinates**: sample 0 is the first
/// sample the basecaller saw, i.e. `signal[trim_offset]` of the raw POD5 array.
/// That is the frame the move table itself is in, the frame
/// [`refine_signal_map`] wants (it is handed the trimmed signal), and the frame
/// Remora uses.
///
/// Both `trim_offset` (`ts`) and `num_samples` (`ns`) are passed as they appear
/// on the record, in *raw* coordinates. `trim_offset` therefore shows up in
/// exactly one place — the closing boundary, `num_samples - trim_offset`, which
/// is the length of the trimmed signal — and not in the interior entries, which
/// are already relative to the trim.
///
/// A caller that indexes the **untrimmed** POD5 signal array wants
/// `trim_offset` added back to every entry (the closing boundary then lands on
/// `num_samples`); `escapepod-classify`'s anchoring does exactly that, because
/// it flips spans through `ns`. That shift is one line, but getting the frame
/// wrong silently displaces every base by `ts` samples, so decide which frame
/// you are in before you use the map.
///
/// # Examples
///
/// ```
/// use escapepod_signal::mapping::seq_to_signal_from_moves;
///
/// // Three moves (blocks 0, 2 and 3) at stride 5, in a read of 40 samples
/// // that was basecalled from sample 10 onward.
/// let map = seq_to_signal_from_moves(&[1, 0, 1, 1, 0], 5, 10, 40);
/// // Trimmed frame: base 0 starts at 0 (raw sample 10), and the closing
/// // boundary is the trimmed length, 40 - 10.
/// assert_eq!(map, vec![0, 10, 15, 30]);
///
/// // The untrimmed frame is the same map plus `trim_offset`.
/// let raw: Vec<i64> = map.iter().map(|v| v + 10).collect();
/// assert_eq!(raw, vec![10, 20, 25, 40]);
/// ```
///
/// [`refine_signal_map`]: crate::resquiggle::refine_signal_map
#[must_use]
pub fn seq_to_signal_from_moves(
    moves: &[u8],
    stride: u32,
    trim_offset: i64,
    num_samples: u64,
) -> Vec<i64> {
    let stride = i64::from(stride);
    let mut map: Vec<i64> = moves
        .iter()
        .enumerate()
        .filter_map(|(i, &m)| (m == 1).then_some(i as i64 * stride))
        .collect();
    // The closing boundary is the trimmed signal length, because the interior
    // entries are in trimmed coordinates.
    map.push(num_samples as i64 - trim_offset);
    map
}

/// Map reference positions to signal positions through a query→signal map and a
/// CIGAR, by the Remora knot convention.
///
/// Returns `ref_len + 1` boundaries for the `ref_len` reference positions the
/// alignment covers (after the trailing-op strip below), in the same coordinate
/// frame as `query_to_signal`, so reference position `i` owns samples
/// `out[i]..out[i + 1]`.
///
/// `query_to_signal` is a full-query map — it covers every base in the record's
/// sequence, soft-clipped bases included — because that is what
/// [`seq_to_signal_from_moves`] produces and what the CIGAR's query coordinates
/// are relative to.
///
/// # The conventions, in the order they bite
///
/// 1. **Trailing non-match ops are stripped.** A trailing `S`/`I`/`D`/`H`
///    contributes no aligned base, so the mapping ends at the last `M`/`=`/`X`
///    and `ref_len` is measured after the strip. Leading non-match ops are
///    *not* stripped: they are absorbed by the walk (a leading `S` shifts the
///    query origin; a leading `D`/`N` opens a gap interpolated from the origin
///    knot `(0, 0)`).
/// 2. **Knots sit at the start and `end - 1` of each match block** — not at
///    `end`. The last aligned position of a block is `end - 1`, and using `end`
///    as the right knot stretches every gap by one position.
/// 3. **Inside a match block the lookup is exact 1:1 integer**:
///    `query_to_signal[block_query_start + (r - block_ref_start)]`. No
///    rounding, no interpolation.
/// 4. **Only indel gaps interpolate**, linearly between the bracketing knots.
///
/// # Integer, not a float chain
///
/// The arithmetic is integer throughout except for the one ratio each *gap*
/// position needs. In particular this is **not** the two-step float chain
/// `ref → float query → float signal` that a pair of `np.interp` calls
/// performs. That chain runs every reference position — aligned ones included
/// — through two interpolations, so an aligned position arrives at its sample
/// by arithmetic instead of by lookup, and each interpolation is evaluated as
/// `slope * (x - x0) + y0` with `slope` rounded first. The last step floors,
/// which promotes a **one-ulp** difference in the intermediate query
/// coordinate into a **one-sample** difference in the answer.
///
/// That is not hypothetical: with a query→signal map of `[0, 7, 8]` and a
/// CIGAR of `1M 6D 1M`, the float chain puts reference position 5 at sample 4
/// where the exact ratio puts it at 5 (pinned by
/// `mapping_tests::ref_to_signal_differs_from_the_float_chain`). The two
/// implementations agree on the overwhelming majority of alignments — a
/// 200 000-case random sweep of realistic CIGARs found no difference at all,
/// and the divergences need a long deletion spanned by short dwells — which is
/// exactly what makes the disagreement expensive to find once two consumers
/// have each written their own version.
///
/// # Examples
///
/// ```
/// use escapepod_signal::mapping::{CigarKind, CigarOp, ref_to_signal};
///
/// // 4 query bases at samples 0, 10, 20, 30 (plus the closing boundary),
/// // aligned as 2M1D2M: reference position 2 is deleted from the query.
/// let query_to_signal = vec![0, 10, 20, 30, 40];
/// let cigar = [
///     CigarOp::new(CigarKind::Match, 2),
///     CigarOp::new(CigarKind::Deletion, 1),
///     CigarOp::new(CigarKind::Match, 2),
/// ];
/// let map = ref_to_signal(&query_to_signal, &cigar);
/// // Ref 0, 1 and 3, 4 are exact lookups; ref 2 (the deletion) interpolates
/// // between the knots at ref 1 (sample 10) and ref 3 (sample 20).
/// assert_eq!(map, vec![0, 10, 15, 20, 30, 40]);
/// ```
#[must_use]
pub fn ref_to_signal(query_to_signal: &[i64], cigar: &[CigarOp]) -> Vec<i64> {
    let n_query = query_to_signal.len();
    if n_query == 0 {
        return Vec::new();
    }
    let last_query = (n_query - 1) as i64;

    // Convention 1: strip trailing non-match ops. Slice rather than copy --
    // nothing before the last match block moves.
    let end = cigar
        .iter()
        .rposition(|op| op.kind.is_match())
        .map_or(0, |i| i + 1);
    let cigar = &cigar[..end];
    if cigar.is_empty() {
        return vec![query_to_signal[0]];
    }

    // Walk the CIGAR once, collecting match blocks as
    // `(ref_start, ref_end_exclusive, query_start)` plus the total lengths.
    let mut ref_pos: i64 = 0;
    let mut query_pos: i64 = 0;
    let mut blocks: Vec<(i64, i64, i64)> = Vec::new();
    for op in cigar {
        let len = i64::from(op.len);
        if op.kind.is_match() && len > 0 {
            blocks.push((ref_pos, ref_pos + len, query_pos));
        }
        if op.kind.consumes_reference() {
            ref_pos += len;
        }
        if op.kind.consumes_query() {
            query_pos += len;
        }
    }
    let ref_len = ref_pos;
    let total_query = query_pos;

    if blocks.is_empty() {
        // Only zero-length match ops: nothing anchors the reference.
        return vec![query_to_signal[0]; (ref_len + 1) as usize];
    }

    // Convention 3: exact integer lookup, clamped to the map.
    let sig_at = |q: i64| -> i64 { query_to_signal[q.clamp(0, last_query) as usize] };

    // Convention 4: a gap position's query coordinate is `q_base + num/den`,
    // kept as a ratio so the only floats in the function are this division of
    // two small integers (exact in f64) and the interpolation between the two
    // bracketing map entries.
    let sig_interp = |num: i64, den: i64, q_base: i64| -> i64 {
        let q = q_base as f64 + num as f64 / den as f64;
        if q <= 0.0 {
            return query_to_signal[0];
        }
        if q >= last_query as f64 {
            return query_to_signal[last_query as usize];
        }
        let j = q.floor() as usize;
        let frac = q - j as f64;
        let (lo, hi) = (query_to_signal[j] as f64, query_to_signal[j + 1] as f64);
        (lo + frac * (hi - lo)).floor() as i64
    };

    let mut out = Vec::with_capacity((ref_len + 1) as usize);

    // A gap before the first match block (a leading `D`/`N`) interpolates from
    // the origin knot `(0, 0)`.
    let (first_ref, _, first_query) = blocks[0];
    for r in 0..first_ref {
        out.push(sig_interp(r * first_query, first_ref, 0));
    }

    for (bi, &(blk_ref_start, blk_ref_end, blk_query_start)) in blocks.iter().enumerate() {
        for r in blk_ref_start..blk_ref_end {
            out.push(sig_at(blk_query_start + (r - blk_ref_start)));
        }

        // Convention 2: this block's right knot is its LAST aligned position,
        // `end - 1`, paired with the query base aligned to it. For a
        // single-position block that coincides with the left knot, which is
        // fine -- it is still one aligned pair.
        let knot_ref_a = blk_ref_end - 1;
        let knot_query_a = blk_query_start + (blk_ref_end - 1 - blk_ref_start);
        let (knot_ref_b, knot_query_b) = match blocks.get(bi + 1) {
            // The next block's left knot is its start.
            Some(&(next_ref, _, next_query)) => (next_ref, next_query),
            // Nothing ref-consuming survives the strip after the last block, so
            // this bound only guards a caller passing a hand-built CIGAR.
            None => (ref_len, total_query),
        };
        let ref_span = knot_ref_b - knot_ref_a; // > 0 whenever the loop runs
        let query_span = knot_query_b - knot_query_a;
        for r in blk_ref_end..knot_ref_b {
            out.push(sig_interp(
                (r - knot_ref_a) * query_span,
                ref_span,
                knot_query_a,
            ));
        }
    }

    // Closing boundary: the end of the last aligned base.
    out.push(sig_at(total_query));
    out.truncate((ref_len + 1) as usize);
    out
}

#[cfg(test)]
mod mapping_tests {
    use super::*;

    fn m(len: u32) -> CigarOp {
        CigarOp::new(CigarKind::Match, len)
    }

    #[test]
    fn moves_worked_example() {
        // moves [1,0,1,1,0], stride 5 -> moves at blocks 0, 2, 3 -> 0, 10, 15.
        let map = seq_to_signal_from_moves(&[1, 0, 1, 1, 0], 5, 10, 40);
        assert_eq!(map, vec![0, 10, 15, 30]);
        // n bases -> n + 1 boundaries.
        assert_eq!(map.len(), 3 + 1);
    }

    #[test]
    fn moves_trim_offset_only_moves_the_closing_boundary() {
        // The interior entries are in trimmed coordinates, so they do not
        // depend on `trim_offset` at all; only the closing boundary does.
        let a = seq_to_signal_from_moves(&[1, 1, 1], 4, 0, 12);
        let b = seq_to_signal_from_moves(&[1, 1, 1], 4, 5, 17);
        assert_eq!(a, vec![0, 4, 8, 12]);
        assert_eq!(b, vec![0, 4, 8, 12]);
        // Untrimmed frame is the shift; the closing boundary lands on `ns`.
        let raw: Vec<i64> = b.iter().map(|v| v + 5).collect();
        assert_eq!(raw, vec![5, 9, 13, 17]);
        assert_eq!(*raw.last().unwrap(), 17);
    }

    #[test]
    fn moves_stride_one_and_leading_zero() {
        // A leading `0` means the first block is not a base start; the first
        // base then begins at the first `1`, NOT at 0.
        let map = seq_to_signal_from_moves(&[0, 1, 0, 1], 1, 0, 4);
        assert_eq!(map, vec![1, 3, 4]);
        // Stride > 1 scales those positions.
        let map = seq_to_signal_from_moves(&[0, 1, 0, 1], 6, 0, 24);
        assert_eq!(map, vec![6, 18, 24]);
    }

    #[test]
    fn moves_degenerate_inputs() {
        // No moves at all: the closing boundary is the whole map, i.e. zero
        // bases -- not an empty vector, and not a panic.
        assert_eq!(seq_to_signal_from_moves(&[], 5, 0, 25), vec![25]);
        assert_eq!(seq_to_signal_from_moves(&[0, 0, 0], 5, 0, 25), vec![25]);
        assert_eq!(seq_to_signal_from_moves(&[0, 0, 0], 5, 5, 25), vec![20]);
    }

    #[test]
    fn ref_to_signal_pure_match_is_exact() {
        let q = vec![0, 7, 21, 22, 40];
        let out = ref_to_signal(&q, &[m(4)]);
        // 1:1, so the reference map IS the query map -- every entry exact.
        assert_eq!(out, q);
    }

    #[test]
    fn ref_to_signal_strips_trailing_non_match_ops() {
        let q = vec![0, 10, 20, 30, 40, 50, 60];
        let plain = ref_to_signal(&q, &[m(3)]);
        for trailing in [
            CigarKind::SoftClip,
            CigarKind::Insertion,
            CigarKind::HardClip,
            CigarKind::Deletion,
            CigarKind::Skip,
            CigarKind::Pad,
        ] {
            let out = ref_to_signal(&q, &[m(3), CigarOp::new(trailing, 2)]);
            assert_eq!(out, plain, "trailing {trailing:?} should be stripped");
            // Stripped means the reference length is 3, not 3 + 2.
            assert_eq!(out.len(), 4);
        }
    }

    #[test]
    fn ref_to_signal_leading_soft_clip_shifts_the_query_origin() {
        // 2S3M: the soft-clipped bases are still in the move table, so ref 0
        // is query 2, not query 0. Leading ops are absorbed, not stripped.
        let q = vec![0, 10, 20, 30, 40, 50];
        let out = ref_to_signal(&q, &[CigarOp::new(CigarKind::SoftClip, 2), m(3)]);
        assert_eq!(out, vec![20, 30, 40, 50]);
    }

    #[test]
    fn ref_to_signal_leading_deletion_interpolates_from_the_origin() {
        // 2D2M: reference positions 0 and 1 precede any aligned base, so they
        // interpolate between the origin knot (0, 0) and the first block's
        // left knot (2, 0) -- both of which sit at query 0 here.
        let q = vec![0, 10, 20];
        let out = ref_to_signal(&q, &[CigarOp::new(CigarKind::Deletion, 2), m(2)]);
        assert_eq!(out, vec![0, 0, 0, 10, 20]);
    }

    #[test]
    fn ref_to_signal_deletion_gap_interpolates_linearly() {
        // 2M3D2M: ref 2, 3 and 4 are deleted. The bracketing knots are
        // (ref 1, query 1) -- the `end - 1` of the first block, NOT its `end`
        // -- and (ref 5, query 2), so the gap spans 4 reference positions for
        // 1 query base: ref 2, 3, 4 -> query 1.25, 1.5, 1.75, i.e. quarters of
        // the way from sample 40 to sample 80.
        let q = vec![0, 40, 80, 120, 160];
        let out = ref_to_signal(&q, &[m(2), CigarOp::new(CigarKind::Deletion, 3), m(2)]);
        assert_eq!(out, vec![0, 40, 50, 60, 70, 80, 120, 160]);
        assert_eq!(out.len(), 7 + 1);
    }

    #[test]
    fn ref_to_signal_insertion_gap_skips_query_bases() {
        // 2M2I2M: two query bases align to nothing. The reference is
        // continuous across them, so the knots (ref 1, query 1) and
        // (ref 2, query 4) are adjacent and there is no gap to fill -- the
        // inserted bases' samples simply fall inside ref 1's span.
        let q = vec![0, 10, 20, 30, 40, 50, 60];
        let out = ref_to_signal(&q, &[m(2), CigarOp::new(CigarKind::Insertion, 2), m(2)]);
        assert_eq!(out, vec![0, 10, 40, 50, 60]);
    }

    #[test]
    fn ref_to_signal_adjacent_indels() {
        // 2M1D2I2M: a deletion immediately followed by an insertion. The gap
        // is the single deleted reference position, bracketed by knots
        // (ref 1, query 1) and (ref 3, query 4): ref 2 -> query 2.5.
        let q = vec![0, 10, 20, 30, 40, 50, 60];
        let out = ref_to_signal(
            &q,
            &[
                m(2),
                CigarOp::new(CigarKind::Deletion, 1),
                CigarOp::new(CigarKind::Insertion, 2),
                m(2),
            ],
        );
        assert_eq!(out, vec![0, 10, 25, 40, 50, 60]);
    }

    #[test]
    fn ref_to_signal_single_base_match_block() {
        // 1M1D1M: both blocks are one position, so each block's left knot and
        // `end - 1` right knot coincide. The gap at ref 1 is bracketed by
        // (ref 0, query 0) and (ref 2, query 1).
        let q = vec![0, 10, 20];
        let out = ref_to_signal(&q, &[m(1), CigarOp::new(CigarKind::Deletion, 1), m(1)]);
        assert_eq!(out, vec![0, 5, 10, 20]);
    }

    #[test]
    fn ref_to_signal_sequence_match_and_mismatch_are_matches() {
        let q = vec![0, 10, 20, 30, 40];
        let split = ref_to_signal(
            &q,
            &[
                CigarOp::new(CigarKind::SequenceMatch, 2),
                CigarOp::new(CigarKind::SequenceMismatch, 1),
                CigarOp::new(CigarKind::SequenceMatch, 1),
            ],
        );
        // `=`/`X` are aligned positions, so this is the same as 4M: no gaps,
        // and no knot boundary effects between adjacent blocks.
        assert_eq!(split, ref_to_signal(&q, &[m(4)]));
        assert_eq!(split, q);
    }

    #[test]
    fn ref_to_signal_degenerate_inputs() {
        // Empty map: nothing to map to.
        assert!(ref_to_signal(&[], &[m(3)]).is_empty());
        // Empty CIGAR, or one with no aligned position at all: a single
        // boundary at the map's start, so callers see `len() < 2` and skip.
        assert_eq!(ref_to_signal(&[5, 15, 25], &[]), vec![5]);
        assert_eq!(
            ref_to_signal(&[5, 15, 25], &[CigarOp::new(CigarKind::SoftClip, 3)]),
            vec![5]
        );
        // A zero-length match op anchors nothing, but `D` still consumes
        // reference: every position collapses onto the map's start.
        assert_eq!(
            ref_to_signal(&[5, 15, 25], &[CigarOp::new(CigarKind::Deletion, 2), m(0)]),
            vec![5, 5, 5]
        );
    }

    /// The case the integer path exists for.
    ///
    /// The float chain is `ref_to_query = np.interp(arange(ref_len + 1),
    /// r_knots, q_knots)` followed by `np.floor(np.interp(ref_to_query,
    /// arange(n_query), query_to_signal))`. Both `np.interp` calls evaluate
    /// `slope * (x - x0) + y0` with `slope` rounded to f64 first, and the
    /// floor at the end turns a one-ulp difference in the intermediate query
    /// coordinate into a one-sample difference in the answer.
    ///
    /// This input makes it happen. It was found by sweeping random CIGARs
    /// through both implementations and shrinking the first disagreement;
    /// the float chain returns
    /// `[0, 1, 2, 3, 4, 4, 6, 7, 8]` — reference position 5 lands a sample
    /// early, and position 6 is back in step, so the error is a single dip in
    /// the middle of a monotone map rather than an offset anything downstream
    /// could notice.
    #[test]
    fn ref_to_signal_differs_from_the_float_chain() {
        // 1M 6D 1M over two query bases with dwells 7 and 1: the whole
        // deletion is spanned by the knots (ref 0, query 0) and
        // (ref 7, query 1), so every gap position is `r / 7` of the way
        // through a 7-sample dwell and should land exactly on sample `r`.
        let q = vec![0, 7, 8];
        let out = ref_to_signal(&q, &[m(1), CigarOp::new(CigarKind::Deletion, 6), m(1)]);
        assert_eq!(out, vec![0, 1, 2, 3, 4, 5, 6, 7, 8]);
        // The float chain computes ref 5 as `7.0 * (0.14285714285714285 * 5)`
        // = 4.999999999999999, floors it, and reports 4.
        assert_eq!(out[5], 5);
        // The aligned positions are exact map entries -- looked up, never
        // interpolated -- and the closing boundary is exactly the end of the
        // last aligned base rather than an extrapolation past it.
        assert_eq!(out[7], q[1]);
        assert_eq!(*out.last().unwrap(), q[2]);
    }
}
