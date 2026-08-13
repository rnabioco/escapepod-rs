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
    /// Query position of the last common-arm base in reference order
    /// (`divergent - 1`) — the arm's *first* base in time, the mask boundary.
    pub q_div_m1: Option<usize>,
    pub q_body_mid: Option<usize>,
    /// Query position per feature offset; `-1` = unaligned.
    pub qf: Vec<i64>,
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
fn ref_to_query(record: &RecordBuf, wanted: &[i64], slop: i64) -> HashMap<i64, usize> {
    let mut out = HashMap::new();
    let Some(start) = record.alignment_start() else {
        return out;
    };
    let (Some(&lo_w), Some(&hi_w)) = (wanted.iter().min(), wanted.iter().max()) else {
        return out;
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
        for d in 0..=slop {
            if let Some(&q) = by_ref.get(&(r + d)) {
                out.insert(r, q);
                break;
            }
            if let Some(&q) = by_ref.get(&(r - d)) {
                out.insert(r, q);
                break;
            }
        }
    }
    out
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
    ];
    wanted.extend(offsets.iter().map(|&o| g.junction as i64 + o as i64));
    let q = ref_to_query(record, &wanted, REF_TO_QUERY_SLOP);

    let (Some(&qj), Some(&qa)) = (q.get(&(g.junction as i64)), q.get(&(g.cca_a as i64))) else {
        return ScanOutcome::Skip(SkipReason::Unanchored);
    };

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

    let qf: Vec<i64> = offsets
        .iter()
        .map(|&o| {
            q.get(&(g.junction as i64 + o as i64))
                .map(|&qp| qp as i64)
                .unwrap_or(-1)
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
        q_div_m1: q.get(&(g.divergent as i64 - 1)).copied(),
        q_body_mid: q.get(&(g.body_mid as i64)).copied(),
        qf,
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
}

/// Resolve a read's spans in the run's detected frame.
///
/// Port of the span/mask logic in `collect_junctions`: per-offset spans via
/// the (possibly flipped) move-table map, and the mask boundary from the
/// arm's last reference base (`divergent - 1`) when it aligned, else the
/// smallest resolved arm start (over-masks least, never under-masks), else
/// the junction start (masking the whole arm — lossy, never leaky).
pub fn finalize(read: &AnchoredRead, orientation: Orientation, offsets: &[i32]) -> JunctionCoords {
    let reversed = orientation == Orientation::Reversed;
    let s2s = &read.seq_to_sig;

    let j_span = sig_span(s2s, read.ns, read.q_junction, reversed);

    let feat_spans: Vec<(i64, i64)> = read
        .qf
        .iter()
        .map(|&qp| {
            if qp >= 0 && (qp as usize) < read.nb {
                sig_span(s2s, read.ns, qp as usize, reversed)
            } else {
                (-1, -1)
            }
        })
        .collect();

    let arm_min = feat_spans
        .iter()
        .zip(offsets)
        .filter_map(|(&(a, _), &o)| (o >= 0 && a >= 0).then_some(a))
        .min();

    let common_start_sig = match read.q_div_m1 {
        Some(qd) if qd < read.nb => sig_span(s2s, read.ns, qd, reversed).0,
        _ => arm_min.unwrap_or(j_span.0),
    };

    JunctionCoords {
        feat_spans,
        common_start_sig,
        junction_sig: j_span.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
