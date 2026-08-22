// SPDX-License-Identifier: MIT

//! Per-record anchoring: reference → query → signal coordinates.
//!
//! Port of `escapepod_models.charging.collect_junctions`' per-read mechanics:
//! the CIGAR-based reference→query map (nearest aligned within a 2-base
//! slop), the move-table query→signal map (Remora convention,
//! `move_position * stride + ts`), and the per-run frame-orientation vote.
//!
//! Orientation is detected from the data rather than assumed: RNA004
//! translocates 3'→5', so in time order the junction must precede the tRNA
//! body. On every corpus built so far the move table indexed *reversed*
//! signal — and getting this backwards silently mirrors every window rather
//! than failing, so a corpus that disagrees with itself is an error, not a
//! majority vote to be papered over.

use crate::geometry::RefGeometry;
use anyhow::{Result, bail};
use noodles_sam::alignment::RecordBuf;
use noodles_sam::alignment::record::cigar::op::Kind;
use noodles_sam::alignment::record::data::field::Tag;
use std::collections::HashMap;
use uuid::Uuid;

/// Slop (bases) for the nearest-aligned reference→query lookup.
const REF_TO_QUERY_SLOP: i64 = 2;

/// Whether the move-table map indexes time-ordered or reversed signal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Orientation {
    /// `seq_to_sig` indexes the signal in raw time order.
    Time,
    /// `seq_to_sig` indexes reversed signal; spans flip through `ns`.
    Reversed,
}

/// Per-run orientation tally (junction-before-body in time = `time`).
#[derive(Debug, Default, Clone, Copy)]
pub struct OrientationVotes {
    pub time: usize,
    pub reversed: usize,
}

impl OrientationVotes {
    /// Record one read's evidence, if its body anchor resolved.
    pub fn add(&mut self, read: &AnchoredRead) {
        if let Some(qb) = read.q_body_mid
            && qb < read.nb
        {
            if read.seq_to_sig[read.q_junction] < read.seq_to_sig[qb] {
                self.time += 1;
            } else {
                self.reversed += 1;
            }
        }
    }

    pub fn total(&self) -> usize {
        self.time + self.reversed
    }
}

/// Resolve the per-run orientation from the votes.
///
/// Requires at least `min_informative` informative reads and a ≥95%
/// consensus; anything else is an error (an underpowered or self-
/// contradicting run must not silently pick a frame).
pub fn resolve_orientation(
    votes: &OrientationVotes,
    min_informative: usize,
) -> Result<Orientation> {
    let total = votes.total();
    if total < min_informative {
        bail!(
            "orientation check underpowered: {} informative reads (need {}); \
             pass --orientation to override for small batches",
            total,
            min_informative
        );
    }
    let frac_time = votes.time as f64 / total as f64;
    if frac_time > 0.05 && frac_time < 0.95 {
        bail!(
            "ambiguous move-table orientation: time={}, reversed={}",
            votes.time,
            votes.reversed
        );
    }
    Ok(if frac_time < 0.5 {
        Orientation::Reversed
    } else {
        Orientation::Time
    })
}

/// One read anchored to its reference junction, frame not yet resolved.
///
/// Holds everything the feature stage needs once the run's orientation is
/// known: the query sequence (expected k-mer levels are computed on the
/// basecalled sequence), the move-table map, and the query positions of the
/// junction, the CCA A, each feature offset, the common-arm boundary, and
/// the tRNA-body orientation anchor.
#[derive(Debug, Clone)]
pub struct AnchoredRead {
    pub read_id: Uuid,
    pub reference: String,
    pub mapq: u8,
    /// Total signal samples (`ns` tag).
    pub ns: i64,
    /// Basecalled query sequence (uppercase not applied; the k-mer lookup
    /// uppercases internally).
    pub seq: Vec<u8>,
    /// Move-table map: signal position per base, length `nb + 1` (the last
    /// entry is `ns`). Indexes whichever frame the run's orientation says.
    pub seq_to_sig: Vec<i64>,
    /// Number of basecalled bases (count of 1-moves).
    pub nb: usize,
    pub q_junction: usize,
    pub q_cca_a: usize,
    /// Dorado's `ts` trim offset, carried so callers can record it.
    pub ts: i64,
    /// Query position of the last common-arm base in reference order
    /// (`divergent - 1`) — the arm's *first* base in time, the mask boundary.
    pub q_div_m1: Option<usize>,
    /// Query position of the mid-poly(A) anchor, if it mapped.
    pub q_polya_mid: Option<usize>,
    pub q_body_mid: Option<usize>,
    /// Query position per feature offset; `-1` = unaligned.
    pub qf: Vec<i64>,
    /// Which rule placed `q_junction`. Carried per read so a corpus can be
    /// audited by anchor provenance rather than the change being inferred —
    /// the flank rule fires ~30x more often on charged reads, so its footprint
    /// is itself class-correlated and worth being able to see.
    pub anchor_source: AnchorSource,
}

/// Why a BAM record produced no [`AnchoredRead`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SkipReason {
    /// Unmapped, reverse-strand, secondary, or supplementary.
    Filtered,
    /// Mapping quality below the cutoff.
    LowMapq,
    /// Reference name absent from the junction geometry.
    NoGeometry,
    /// Missing `mv`/`ns` tags (or empty sequence).
    NoTags,
    /// Junction or CCA A did not align within slop.
    Unanchored,
    /// An anchored query position fell outside the move table.
    QueryOutOfRange,
    /// Query name is not a UUID (signal cannot be fetched from POD5).
    BadName,
}

impl SkipReason {
    /// Stable snake_case name, for reports and for keying QC output.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Filtered => "filtered",
            Self::LowMapq => "low_mapq",
            Self::NoGeometry => "no_geometry",
            Self::NoTags => "no_tags",
            Self::Unanchored => "unanchored",
            Self::QueryOutOfRange => "query_out_of_range",
            Self::BadName => "bad_name",
        }
    }
}

/// Outcome of scanning one BAM record.
pub enum ScanOutcome {
    Anchored(Box<AnchoredRead>),
    Skip(SkipReason),
}

/// Extract `(stride, moves, ns, ts)` from a record's `mv`/`ns`/`ts` tags.
fn move_table(record: &RecordBuf) -> Option<(usize, Vec<u8>, i64, i64)> {
    let (stride, moves) = crate::bam_tags::parse_mv_tag(record).ok()?;
    let ns = crate::bam_tags::int_tag(record, Tag::new(b'n', b's'))?;
    let ts = crate::bam_tags::int_tag(record, Tag::new(b't', b's')).unwrap_or(0);
    Some((stride, moves, ns, ts))
}

/// Build the Remora-convention query→signal map from a move table:
/// `seq_to_sig[i] = position_of_ith_move * stride + ts`, with a final
/// boundary entry of `ns`.
fn seq_to_sig_map(moves: &[u8], stride: usize, ts: i64, ns: i64) -> Vec<i64> {
    let mut map: Vec<i64> = moves
        .iter()
        .enumerate()
        .filter_map(|(i, &m)| (m == 1).then_some(i as i64 * stride as i64 + ts))
        .collect();
    map.push(ns);
    map
}

/// Map reference positions → query positions, nearest-aligned within `slop`.
///
/// Port of `charging._ref_to_query`: aligned pairs (CIGAR `M`/`=`/`X` only)
/// within `[min(wanted) - slop, max(wanted) + slop]`, then each wanted
/// position takes the nearest aligned reference position, ties toward `+`.
fn ref_to_query(
    record: &RecordBuf,
    wanted: &[i64],
    slop: i64,
    barrier: Option<i64>,
) -> (HashMap<i64, usize>, HashMap<i64, i64>) {
    let mut out = HashMap::new();
    let mut donor = HashMap::new();
    let Some(start) = record.alignment_start() else {
        return (out, donor);
    };
    let (Some(&lo_w), Some(&hi_w)) = (wanted.iter().min(), wanted.iter().max()) else {
        return (out, donor);
    };
    let (lo, hi) = (lo_w - slop, hi_w + slop);

    let mut by_ref: HashMap<i64, usize> = HashMap::new();
    let mut rpos = start.get() as i64 - 1; // 0-based
    let mut qpos: usize = 0;
    for op in record.cigar().as_ref() {
        let len = op.len();
        match op.kind() {
            Kind::Match | Kind::SequenceMatch | Kind::SequenceMismatch => {
                for k in 0..len {
                    let r = rpos + k as i64;
                    if r >= lo && r <= hi {
                        by_ref.insert(r, qpos + k);
                    }
                }
                rpos += len as i64;
                qpos += len;
            }
            Kind::Insertion | Kind::SoftClip => qpos += len,
            Kind::Deletion | Kind::Skip => rpos += len as i64,
            Kind::HardClip | Kind::Pad => {}
        }
    }

    for &r in wanted {
        // A donor on the far side of the divergent boundary would put the mask
        // boundary, or an arm feature, on a base that encodes the library --
        // the one thing this pipeline must never read. An arm-side position
        // may only be filled from another arm-side position.
        let blocked = |c: i64| matches!(barrier, Some(b) if r < b && c >= b);
        for d in 0..=slop {
            // Only break when a candidate is actually taken: a blocked
            // neighbour must fall through to the other side and to wider d,
            // not abandon the search.
            if let Some(&q) = by_ref.get(&(r + d))
                && !blocked(r + d)
            {
                out.insert(r, q);
                donor.insert(r, r + d);
                break;
            }
            if let Some(&q) = by_ref.get(&(r - d))
                && !blocked(r - d)
            {
                out.insert(r, q);
                donor.insert(r, r - d);
                break;
            }
        }
    }
    (out, donor)
}

/// Which rule placed the junction's query base.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnchorSource {
    /// The aligner put a query base on the junction itself.
    Exact,
    /// Interpolated between undamaged flanks, because it did not.
    FlankInterp,
    /// Neither — the old nearest-neighbour backfill within `slop`.
    Backfill,
}

impl AnchorSource {
    pub fn name(self) -> &'static str {
        match self {
            AnchorSource::Exact => "exact",
            AnchorSource::FlankInterp => "flank_interp",
            AnchorSource::Backfill => "backfill",
        }
    }
}

/// Query base index of the junction, placed from undamaged flanks.
///
/// Port of `charging.flank_anchored_qj`. The modification mis-calls the
/// junction it attaches to, and `ref_to_query` then backfills it from the
/// nearest aligned neighbour, probing `r, r+1, r-1, ...` — upward first, so the
/// origin every counted arm offset is measured from lands on a stand-in base
/// biased toward the adapter. The flanks are not damaged, so interpolate
/// between them instead, in QUERY-BASE space because that is what the counted
/// offsets index.
///
/// Only used when the junction is NOT donor-exact: where the aligner placed it,
/// that is an observation and interpolation could only add error.
fn flank_anchored_qj(
    q: &HashMap<i64, usize>,
    donor: &HashMap<i64, i64>,
    g: &RefGeometry,
    qj_backfill: usize,
) -> (usize, AnchorSource) {
    let j = g.junction as i64;
    if donor.get(&j) == Some(&j) {
        return (qj_backfill, AnchorSource::Exact);
    }
    if let (Some(lo), Some(hi)) = (g.left_anchor, g.right_anchor) {
        let (lo, hi) = (lo as i64, hi as i64);
        // Both anchors must be donor-exact: an anchor that is itself backfilled
        // is a neighbour's base wearing this position's name.
        if hi > lo
            && donor.get(&lo) == Some(&lo)
            && donor.get(&hi) == Some(&hi)
            && let (Some(&qlo), Some(&qhi)) = (q.get(&lo), q.get(&hi))
        {
            let frac = (j - lo) as f64 / (hi - lo) as f64;
            let est = qlo as f64 + frac * (qhi as f64 - qlo as f64);
            if est >= 0.0 {
                return (est.round() as usize, AnchorSource::FlankInterp);
            }
        }
    }
    (qj_backfill, AnchorSource::Backfill)
}

/// Scan one BAM record into an [`AnchoredRead`] (or a skip reason).
///
/// `ref_name` is the record's resolved reference sequence name (the caller
/// looks it up in the BAM header). `offsets` are the bundle's feature
/// offsets relative to the junction.
pub fn scan_record(
    record: &RecordBuf,
    ref_name: &str,
    geometry: &HashMap<String, RefGeometry>,
    offsets: &[i32],
    min_mapq: u8,
) -> ScanOutcome {
    let flags = record.flags();
    if flags.is_unmapped()
        || flags.is_reverse_complemented()
        || flags.is_secondary()
        || flags.is_supplementary()
    {
        return ScanOutcome::Skip(SkipReason::Filtered);
    }
    let mapq = record.mapping_quality().map(u8::from).unwrap_or(255);
    if mapq < min_mapq {
        return ScanOutcome::Skip(SkipReason::LowMapq);
    }
    let Some(g) = geometry.get(ref_name) else {
        return ScanOutcome::Skip(SkipReason::NoGeometry);
    };

    let seq: &[u8] = record.sequence().as_ref();
    let Some((stride, moves, ns, ts)) = move_table(record) else {
        return ScanOutcome::Skip(SkipReason::NoTags);
    };
    if seq.is_empty() {
        return ScanOutcome::Skip(SkipReason::NoTags);
    }

    let mut wanted: Vec<i64> = vec![
        g.cca_a as i64,
        g.junction as i64,
        g.divergent as i64 - 1,
        g.body_mid as i64,
        g.polya_mid as i64,
    ];
    wanted.extend(offsets.iter().map(|&o| g.junction as i64 + o as i64));
    // Flank anchors, for placing a junction the modification has mis-called.
    wanted.extend(
        [g.left_anchor, g.right_anchor]
            .iter()
            .filter_map(|a| a.map(|p| p as i64)),
    );
    let (q, donor) = ref_to_query(record, &wanted, REF_TO_QUERY_SLOP, Some(g.divergent as i64));

    let (Some(&qj_backfill), Some(&qa)) = (q.get(&(g.junction as i64)), q.get(&(g.cca_a as i64)))
    else {
        return ScanOutcome::Skip(SkipReason::Unanchored);
    };
    let (qj, anchor_source) = flank_anchored_qj(&q, &donor, g, qj_backfill);

    let seq_to_sig = seq_to_sig_map(&moves, stride, ts, ns);
    let nb = seq_to_sig.len() - 1;
    // The anchor positions must index inside the move table (the reference
    // implementation checks `> nb`; `== nb` would have crashed it, so `>=`
    // here loses no read that survives the Python).
    if q.values().any(|&qp| qp >= nb) {
        return ScanOutcome::Skip(SkipReason::QueryOutOfRange);
    }

    let Some(read_id) = record
        .name()
        .and_then(|n| std::str::from_utf8(n.as_ref()).ok())
        .and_then(|s| escapepod_signal::parse_uuid_flexible(s).ok())
    else {
        return ScanOutcome::Skip(SkipReason::BadName);
    };

    // Feature bases come from EXACTLY aligned positions only. A backfilled
    // offset resolves to a neighbour's query base, so its "features" are a
    // copy of that neighbour rather than an observation of this base -- when
    // the arm is unaligned, offsets +1 and +2 were byte-identical copies of
    // the junction on 87% of those reads, which is what let a read with no
    // readable arm still look like a confident call. `-1` (and the NaN it
    // produces downstream) is the honest value.
    //
    // The anchors themselves (junction, CCA A, divergent-1, body mid) still
    // accept a backfill: a one-base hole should not cost the whole window.
    let qf: Vec<i64> = offsets
        .iter()
        .map(|&o| {
            let rp = g.junction as i64 + o as i64;
            match (q.get(&rp), donor.get(&rp)) {
                (Some(&qp), Some(&d)) if d == rp => qp as i64,
                _ => -1,
            }
        })
        .collect();

    ScanOutcome::Anchored(Box::new(AnchoredRead {
        read_id,
        reference: ref_name.to_string(),
        mapq,
        ns,
        seq: seq.to_vec(),
        seq_to_sig,
        nb,
        q_junction: qj,
        q_cca_a: qa,
        ts,
        q_div_m1: q.get(&(g.divergent as i64 - 1)).copied(),
        q_body_mid: q.get(&(g.body_mid as i64)).copied(),
        q_polya_mid: q.get(&(g.polya_mid as i64)).copied(),
        qf,
        anchor_source,
    }))
}

/// Base `q`'s samples as a `(start, end)` raw time-order span.
fn sig_span(seq_to_sig: &[i64], ns: i64, q: usize, reversed: bool) -> (i64, i64) {
    let (mut a, mut b) = (seq_to_sig[q], seq_to_sig[q + 1]);
    if reversed {
        (a, b) = (ns - b, ns - a);
    }
    if a <= b { (a, b) } else { (b, a) }
}

/// How the feature-base spans are resolved.
///
/// The training corpus is defined by this choice, so it travels in the bundle
/// rather than being a flag: a caller that picks the other one gets a wrong
/// answer, not an error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpanMode {
    /// Positions come from the aligner's reference→query mapping. What
    /// bundles predating the counting anchor were trained with.
    Aligner,
    /// Arm offsets `1..=arm_bases` are reached by counting that many bases
    /// along the *query* from the junction rather than asking the aligner.
    ///
    /// The aligner dies at the adduct, which made feature availability depend
    /// on charging — the very quantity being measured. Counting gives every
    /// read the same features.
    Counted { arm_bases: u32 },
}

impl SpanMode {
    /// The mode a recipe's `count_arm_bases` implies; `0` is the aligner path.
    pub fn from_arm_bases(arm_bases: u32) -> Self {
        if arm_bases > 0 {
            Self::Counted { arm_bases }
        } else {
            Self::Aligner
        }
    }
}

/// How the mask boundary was established for one read.
///
/// Not cosmetic: `JunctionFallback` means no arm base resolved at all, so the
/// whole left window is masked and the read is **not readable** — callers are
/// expected to abstain rather than score it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaskSource {
    /// The arm's last reference base (`divergent - 1`) aligned.
    Exact,
    /// The counted boundary base resolved (counting mode).
    Counted,
    /// No exact boundary, but some arm base resolved; its earliest start is
    /// used. Over-masks least while never under-masking.
    ArmFallback,
    /// Nothing of the arm resolved. The whole left window is masked.
    JunctionFallback,
}

/// Frame-resolved signal coordinates for one read.
#[derive(Debug, Clone)]
pub struct JunctionCoords {
    /// `(start, end)` raw-sample span per feature offset; `(-1, -1)` if the
    /// base did not align.
    pub feat_spans: Vec<(i64, i64)>,
    /// Earliest common-arm sample we can vouch for — everything earlier in
    /// time is masked (more arm, or divergent adapter).
    pub common_start_sig: i64,
    /// Start of the adapter-G base in raw time coordinates.
    pub junction_sig: i64,
    /// How `common_start_sig` was established.
    pub mask_source: MaskSource,
    /// Start of the CCA A base (where the amino acid attaches).
    pub cca_a_sig: i64,
    /// Samples assigned to the CCA A base; an adduct inflates this.
    pub cca_a_dwell: i64,
    /// Samples assigned to the junction base.
    pub junction_dwell: i64,
    /// Deepest arm offset with a resolved span, under the active span mode.
    /// Feature availability, in one number.
    pub arm_resolved_depth: i32,
    /// Deepest arm offset the ALIGNER exactly placed, whatever the span mode.
    ///
    /// Kept separately because counting gives every read the same features --
    /// which is the point -- but also erases the signal identifying reads no
    /// model can actually read, and inference still has to abstain on those.
    pub aligner_arm_depth: i32,
    /// Start of the mid-poly(A) anchor, or -1.
    pub polya_mid_sig: i64,
    /// Start of the tRNA-body anchor, or -1.
    pub body_mid_sig: i64,
}

/// Query position per feature offset, under a span mode.
///
/// The single source of truth for "which base is offset `o`". It feeds both
/// the signal spans and the k-mer level lookup, exactly as the reference
/// implementation's `_qpos` does -- computing one from the aligner and the
/// other from counting silently mixes two feature spaces, and shows up only
/// in the residual.
pub fn query_positions(read: &AnchoredRead, offsets: &[i32], mode: SpanMode) -> Vec<i64> {
    let qj = read.q_junction as i64;
    read.qf
        .iter()
        .zip(offsets)
        .map(|(&qp, &o)| match mode {
            SpanMode::Counted { arm_bases } if o >= 1 => {
                if o as u32 <= arm_bases {
                    qj + o as i64
                } else {
                    // Past the cap there is no safe position, so the base is
                    // unresolved for EVERY read -- that uniformity is the
                    // point of counting.
                    -1
                }
            }
            _ => qp,
        })
        .collect()
}

/// Resolve a read's spans in the run's detected frame.
///
/// Port of the span/mask logic in `collect_junctions`, for both span modes.
///
/// [`SpanMode::Aligner`]: per-offset spans via the (possibly flipped)
/// move-table map at the aligner's query positions, and the mask boundary
/// from the arm's last reference base (`divergent - 1`) when it aligned, else
/// the smallest resolved arm start (over-masks least, never under-masks),
/// else the junction start (masking the whole arm — lossy, never leaky).
///
/// [`SpanMode::Counted`]: arm offsets `1..=arm_bases` come from counting along
/// the query instead, and the boundary is the counted base itself. Offsets
/// `<= 0` still use the aligner — they run into the CCA and the tRNA body,
/// where the alignment is sound.
pub fn finalize(
    read: &AnchoredRead,
    orientation: Orientation,
    offsets: &[i32],
    mode: SpanMode,
) -> JunctionCoords {
    let reversed = orientation == Orientation::Reversed;
    let s2s = &read.seq_to_sig;
    let qj = read.q_junction as i64;

    let j_span = sig_span(s2s, read.ns, read.q_junction, reversed);

    let span_at = |qp: i64| -> (i64, i64) {
        if qp >= 0 && (qp as usize) < read.nb {
            sig_span(s2s, read.ns, qp as usize, reversed)
        } else {
            (-1, -1)
        }
    };

    let feat_spans: Vec<(i64, i64)> = query_positions(read, offsets, mode)
        .into_iter()
        .map(span_at)
        .collect();

    // Offsets STRICTLY inside the arm. Offset 0 is the junction itself, so
    // including it collapses this to the junction start and makes the arm
    // fallback indistinguishable from the junction fallback — which is how a
    // read with no resolved arm at all passes for a normal one.
    let arm_min = feat_spans
        .iter()
        .zip(offsets)
        .filter_map(|(&(a, _), &o)| (o >= 1 && a >= 0).then_some(a))
        .min();

    let (common_start_sig, mask_source) = match mode {
        SpanMode::Counted { arm_bases } => {
            let qcs = qj + arm_bases as i64;
            if qcs >= 0 && (qcs as usize) < read.nb {
                (
                    sig_span(s2s, read.ns, qcs as usize, reversed).0,
                    MaskSource::Counted,
                )
            } else {
                // The counted base ran off the end of the move table. Both
                // modes share the same fallback chain from here: any resolved
                // arm base over-masks least, and only a read with none at all
                // masks the whole window. Going straight to the junction here
                // over-masked 4 reads in 842 against the reference.
                match arm_min {
                    Some(a) => (a, MaskSource::ArmFallback),
                    None => (j_span.0, MaskSource::JunctionFallback),
                }
            }
        }
        SpanMode::Aligner => match read.q_div_m1 {
            Some(qd) if qd < read.nb => (sig_span(s2s, read.ns, qd, reversed).0, MaskSource::Exact),
            _ => match arm_min {
                Some(a) => (a, MaskSource::ArmFallback),
                None => (j_span.0, MaskSource::JunctionFallback),
            },
        },
    };

    let a_span = sig_span(s2s, read.ns, read.q_cca_a, reversed);
    // `start()` in the reference implementation: -1 when the anchor did not
    // map or maps past the move table. These use the BACKFILLED positions --
    // only the feature bases are exact-only.
    let anchor_start = |q: Option<usize>| -> i64 {
        match q {
            Some(qp) if qp < read.nb => sig_span(s2s, read.ns, qp, reversed).0,
            _ => -1,
        }
    };
    let depth_of = |resolved: &dyn Fn(usize) -> bool| -> i32 {
        offsets
            .iter()
            .enumerate()
            .filter(|&(i, &o)| o >= 1 && resolved(i))
            .map(|(_, &o)| o)
            .max()
            .unwrap_or(0)
    };

    JunctionCoords {
        arm_resolved_depth: depth_of(&|i| feat_spans[i].0 >= 0),
        // Independent of the span mode: read.qf is always the aligner's
        // exact-only answer, so counting cannot mask this.
        aligner_arm_depth: depth_of(&|i| read.qf[i] >= 0 && (read.qf[i] as usize) < read.nb),
        cca_a_sig: a_span.0,
        cca_a_dwell: a_span.1 - a_span.0,
        junction_dwell: j_span.1 - j_span.0,
        polya_mid_sig: anchor_start(read.q_polya_mid),
        body_mid_sig: anchor_start(read.q_body_mid),
        feat_spans,
        common_start_sig,
        junction_sig: j_span.0,
        mask_source,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One base per 10 samples, forward frame, so span(i) == (10i, 10i+10).
    fn read_with(qf: Vec<i64>, q_div_m1: Option<usize>) -> AnchoredRead {
        let nb = 40usize;
        AnchoredRead {
            read_id: Uuid::nil(),
            reference: "ref".into(),
            mapq: 60,
            ns: (nb as i64) * 10,
            seq: vec![b'A'; nb],
            seq_to_sig: (0..=nb as i64).map(|i| i * 10).collect(),
            nb,
            q_junction: 20,
            q_cca_a: 19,
            ts: 0,
            q_div_m1,
            q_body_mid: None,
            q_polya_mid: None,
            qf,
            anchor_source: AnchorSource::Exact,
        }
    }

    const OFFSETS: [i32; 5] = [-1, 0, 1, 2, 3];

    // ---- flank-anchored junction placement --------------------------------
    //
    // The existing goldens cannot separate the two rules: every fixture read
    // has a donor-exact junction, so the flank path never fires and a
    // regression in it would pass silently. These exercise the branch directly.

    fn geom(left: Option<usize>, right: Option<usize>) -> RefGeometry {
        RefGeometry {
            junction: 100,
            cca_a: 99,
            divergent: 117,
            body_mid: 50,
            polya_mid: 160,
            left_anchor: left,
            right_anchor: right,
        }
    }

    #[test]
    fn exact_junction_is_used_directly_not_interpolated() {
        // Where the aligner placed the junction, that is an observation;
        // interpolating there could only add error.
        let g = geom(Some(90), Some(120));
        let q = HashMap::from([(100i64, 500usize), (90, 490), (120, 520)]);
        let donor = HashMap::from([(100i64, 100i64), (90, 90), (120, 120)]);
        assert_eq!(
            flank_anchored_qj(&q, &donor, &g, 500),
            (500, AnchorSource::Exact)
        );
    }

    #[test]
    fn backfilled_junction_is_placed_between_the_flanks() {
        // 3 bases of insertion between the anchors, junction 10/30 along:
        // 490 + 0.333 * 33 = 501, NOT the 999 the backfill would have taken.
        let g = geom(Some(90), Some(120));
        let q = HashMap::from([(100i64, 999usize), (90, 490), (120, 523)]);
        let donor = HashMap::from([(100i64, 101i64), (90, 90), (120, 120)]);
        assert_eq!(
            flank_anchored_qj(&q, &donor, &g, 999),
            (501, AnchorSource::FlankInterp)
        );
    }

    #[test]
    fn a_flank_that_is_itself_backfilled_is_refused() {
        // An anchor is only an anchor if the aligner actually placed it.
        let g = geom(Some(90), Some(120));
        let q = HashMap::from([(100i64, 505usize), (90, 490), (120, 523)]);
        let donor = HashMap::from([(100i64, 101i64), (90, 90), (120, 121)]);
        assert_eq!(
            flank_anchored_qj(&q, &donor, &g, 505),
            (505, AnchorSource::Backfill)
        );
    }

    #[test]
    fn falls_back_when_the_reference_has_no_right_anchor() {
        // A divergent panel disables the right anchor entirely; the old rule
        // must still apply rather than the read being lost.
        let g = geom(Some(90), None);
        let q = HashMap::from([(100i64, 505usize), (90, 490)]);
        let donor = HashMap::from([(100i64, 101i64), (90, 90)]);
        assert_eq!(
            flank_anchored_qj(&q, &donor, &g, 505),
            (505, AnchorSource::Backfill)
        );
    }

    #[test]
    fn counted_mode_counts_along_the_query_and_ignores_the_aligner() {
        // The aligner placed the arm offsets somewhere far away (or nowhere);
        // counting must reach q_junction + o regardless.
        let read = read_with(vec![19, 20, -1, -1, 35], None);
        let c = finalize(
            &read,
            Orientation::Time,
            &OFFSETS,
            SpanMode::Counted { arm_bases: 3 },
        );
        assert_eq!(
            c.feat_spans[0],
            (190, 200),
            "offset -1 still uses the aligner"
        );
        assert_eq!(
            c.feat_spans[1],
            (200, 210),
            "offset 0 still uses the aligner"
        );
        assert_eq!(
            c.feat_spans[2],
            (210, 220),
            "offset +1 counted from junction"
        );
        assert_eq!(
            c.feat_spans[3],
            (220, 230),
            "offset +2 counted, not unaligned"
        );
        assert_eq!(
            c.feat_spans[4],
            (230, 240),
            "offset +3 counted, not the aligner's 35"
        );
        assert_eq!(c.mask_source, MaskSource::Counted);
        assert_eq!(
            c.common_start_sig, 230,
            "boundary is the counted base itself"
        );
    }

    #[test]
    fn counted_mode_caps_at_arm_bases() {
        // Past the cap there is no safe position, so the base is unresolved
        // for EVERY read -- that uniformity is the point of counting.
        let read = read_with(vec![19, 20, 21, 22, 23], None);
        let c = finalize(
            &read,
            Orientation::Time,
            &OFFSETS,
            SpanMode::Counted { arm_bases: 2 },
        );
        assert_eq!(c.feat_spans[2], (210, 220));
        assert_eq!(c.feat_spans[3], (220, 230));
        assert_eq!(
            c.feat_spans[4],
            (-1, -1),
            "offset +3 is past arm_bases=2 even though the aligner resolved it"
        );
    }

    /// When the counted boundary runs off the move table, counting falls back
    /// exactly as the aligner mode does -- arm first, junction only as a last
    /// resort. Going straight to the junction over-masks: it disagreed with
    /// the reference on 4 reads in 842 of a real run, and the golden fixture
    /// missed it because all 19 of its reads resolve as `counted`.
    #[test]
    fn counted_boundary_past_the_read_prefers_a_resolved_arm_base() {
        // Arm offsets still resolve by counting (qj + o), so there IS an arm
        // base to fall back to even though the boundary base is out of range.
        let read = read_with(vec![19, 20, 21, 22, 23], None);
        let c = finalize(
            &read,
            Orientation::Time,
            &OFFSETS,
            SpanMode::Counted { arm_bases: 100 },
        );
        assert_eq!(c.mask_source, MaskSource::ArmFallback);
        assert_eq!(c.common_start_sig, 210, "earliest counted arm base");
    }

    #[test]
    fn counted_boundary_past_the_read_and_no_arm_masks_everything() {
        // arm_bases = 0 would be Aligner, so use a read whose arm offsets are
        // all past the move table: nothing resolves, nothing to fall back to.
        let mut read = read_with(vec![19, 20, -1, -1, -1], None);
        read.q_junction = 39; // the last base; qj + o is out of range for o >= 1
        let c = finalize(
            &read,
            Orientation::Time,
            &OFFSETS,
            SpanMode::Counted { arm_bases: 3 },
        );
        assert_eq!(c.mask_source, MaskSource::JunctionFallback);
        assert_eq!(c.common_start_sig, c.junction_sig);
    }

    #[test]
    fn aligner_mode_prefers_the_exact_boundary() {
        let read = read_with(vec![19, 20, 21, 22, 23], Some(24));
        let c = finalize(&read, Orientation::Time, &OFFSETS, SpanMode::Aligner);
        assert_eq!(c.mask_source, MaskSource::Exact);
        assert_eq!(c.common_start_sig, 240);
    }

    #[test]
    fn aligner_mode_falls_back_to_the_earliest_resolved_arm_base() {
        let read = read_with(vec![19, 20, -1, 22, 23], None);
        let c = finalize(&read, Orientation::Time, &OFFSETS, SpanMode::Aligner);
        assert_eq!(c.mask_source, MaskSource::ArmFallback);
        assert_eq!(c.common_start_sig, 220, "earliest resolved arm base");
    }

    /// Regression: the arm-fallback search must be over offsets STRICTLY
    /// inside the arm.
    ///
    /// Offset 0 is the junction itself, so including it makes the minimum
    /// collapse to the junction start and reports `ArmFallback` for a read
    /// with no resolved arm at all -- which is exactly how such a read once
    /// passed for a normal one. Callers abstain on `JunctionFallback`, so
    /// mislabelling it is not cosmetic.
    #[test]
    fn no_resolved_arm_base_is_a_junction_fallback_not_an_arm_fallback() {
        let read = read_with(vec![19, 20, -1, -1, -1], None);
        let c = finalize(&read, Orientation::Time, &OFFSETS, SpanMode::Aligner);
        assert_eq!(c.mask_source, MaskSource::JunctionFallback);
        assert_eq!(c.common_start_sig, c.junction_sig);
    }

    #[test]
    fn span_mode_comes_from_the_recipe() {
        // Absent / 0 is the aligner path, which is what bundles predating the
        // counting anchor were trained with.
        assert_eq!(SpanMode::from_arm_bases(0), SpanMode::Aligner);
        assert_eq!(
            SpanMode::from_arm_bases(24),
            SpanMode::Counted { arm_bases: 24 }
        );
    }

    #[test]
    fn test_seq_to_sig_map() {
        // moves [1,0,1,1,0], stride 5, ts 10, ns 40 → bases at 10, 20, 25.
        let map = seq_to_sig_map(&[1, 0, 1, 1, 0], 5, 10, 40);
        assert_eq!(map, vec![10, 20, 25, 40]);
    }

    #[test]
    fn test_sig_span_flip() {
        let s2s = vec![10i64, 20, 25, 40];
        assert_eq!(sig_span(&s2s, 40, 0, false), (10, 20));
        // Reversed frame: base 0's samples sit at the END of raw time.
        assert_eq!(sig_span(&s2s, 40, 0, true), (20, 30));
        assert_eq!(sig_span(&s2s, 40, 2, true), (0, 15));
    }

    #[test]
    fn test_resolve_orientation() {
        let v = OrientationVotes {
            time: 0,
            reversed: 60,
        };
        assert_eq!(resolve_orientation(&v, 50).unwrap(), Orientation::Reversed);

        let v = OrientationVotes {
            time: 60,
            reversed: 1,
        };
        assert_eq!(resolve_orientation(&v, 50).unwrap(), Orientation::Time);

        // Underpowered.
        let v = OrientationVotes {
            time: 10,
            reversed: 0,
        };
        assert!(resolve_orientation(&v, 50).is_err());

        // Ambiguous.
        let v = OrientationVotes {
            time: 30,
            reversed: 30,
        };
        assert!(resolve_orientation(&v, 50).is_err());
    }
}
