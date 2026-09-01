// SPDX-License-Identifier: MIT

//! Reference geometry: locating the CCA|adapter junction per reference record.
//!
//! Port of `escapepod_models.charging.junction_positions`. A reference that
//! violates the construct's invariants (exactly one motif, the full common
//! arm after it) would silently corrupt every downstream feature, so both
//! are hard errors, not warnings.

use anyhow::{Context, Result, bail};
use std::collections::HashMap;
use std::path::Path;

/// 5' adapter length of the construct (23 nt adapter + literal N). Only used
/// to place the tRNA-body orientation anchor; matches the training corpus.
const FIVEP_LEN: usize = 24;

/// Junction coordinates for one reference record (0-based).
#[derive(Debug, Clone, Copy)]
pub struct RefGeometry {
    /// First base of the anchor motif itself.
    ///
    /// The junction is where the *common arm* starts, which is the rule that
    /// makes the motif unique — but a model may anchor its window somewhere
    /// else inside the motif (`anchor.motif_offset`), and it needs the motif's
    /// own origin to do that. Keeping both means one scan serves both, rather
    /// than a second scan with a different offset finding a different set of
    /// records.
    pub motif_start: usize,
    /// First 3'-adapter base (the G of CCA|GGC).
    pub junction: usize,
    /// Last tRNA base (the A of CCA); the amino acid attaches here.
    pub cca_a: usize,
    /// First divergent adapter base (`junction + common_arm.len()`).
    pub divergent: usize,
    /// Middle of the tRNA body (orientation anchor).
    pub body_mid: usize,
    /// 4 nt into the trailing poly(A) (orientation + QC).
    ///
    /// Located from the reference's own trailing A-run rather than assumed:
    /// the stretch between the arm and the poly(A) is adapter-family specific
    /// (13 nt in the v2 single-adapter references, 18 in every edx*), so a
    /// fixed `divergent + 13 + 4` lands inside the barcode on an edx
    /// reference instead of in the tail.
    pub polya_mid: usize,
    /// Flank anchors, OUTSIDE the basecall damage the modification causes.
    ///
    /// On aa-tRNA the amino acid mis-calls the junction it is attached to:
    /// 51.9% of charged reads carry a CIGAR indel across `CCAGGC` against 2.4%
    /// of uncharged (23x construct-matched), and the unaligned-base rate is
    /// elevated from offset -6 to +15, peaking at +5. Both classes align
    /// equally well at <= -8 and >= +19, so a junction that did not align can
    /// be placed from there instead of from a nearest-neighbour backfill that
    /// is biased toward the adapter.
    ///
    /// `None` when the reference cannot support the anchor — see
    /// [`junction_positions`], which refuses a right anchor that is not
    /// constant across the panel.
    pub left_anchor: Option<usize>,
    pub right_anchor: Option<usize>,
}

/// Read a FASTA into `name → uppercase sequence` (name = first word).
///
/// Public because a reference-anchored model reads the reference *sequence*,
/// not only the junction's coordinate: the window is cut over
/// `reference[alignment_start..alignment_end]`, and the expected k-mer levels
/// the map is refined against come from those bases. One reader, so the
/// coordinates and the sequence cannot come from differently-parsed files.
pub fn reference_sequences(path: &Path) -> Result<HashMap<String, String>> {
    read_fasta(path)
}

fn read_fasta(path: &Path) -> Result<HashMap<String, String>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("cannot read reference FASTA {}", path.display()))?;
    let mut seqs = HashMap::new();
    let mut name: Option<String> = None;
    let mut parts: Vec<String> = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix('>') {
            if let Some(n) = name.take() {
                seqs.insert(n, parts.join(""));
            }
            name = Some(rest.split_whitespace().next().unwrap_or("").to_string());
            parts.clear();
        } else if !line.is_empty() {
            parts.push(line.to_uppercase());
        }
    }
    if let Some(n) = name {
        seqs.insert(n, parts.join(""));
    }
    Ok(seqs)
}

/// Where the common arm begins, relative to the start of the motif.
///
/// Derived from the two declared strings rather than declared separately,
/// because they describe one contiguous construct and a third number stating
/// how they fit together is a third thing that can disagree. The arm is
/// allowed to *overlap* the motif's tail — `CCAGGC` and `GGCTTCTTCTTGCTCTT`
/// are `CCA` + the arm, not the arm after the motif — so this is the earliest
/// offset at which the motif's suffix is a prefix of the arm. `motif.len()`
/// always qualifies — the empty suffix is a prefix of anything — so there is
/// always an answer, and it is `motif.len()` for a motif and arm that do not
/// overlap at all. That is the right reading of "the arm follows the motif"
/// rather than a failure, which is why this is infallible.
///
/// Deliberately *not* `anchor.motif_offset`: that is where the model's window
/// is anchored, which the design is free to move (the windowed variant anchors
/// at +2, a base earlier than the arm) without moving the arm.
pub fn arm_offset(motif: &str, common_arm: &str) -> usize {
    (0..=motif.len())
        .find(|&p| common_arm.len() >= motif.len() - p && common_arm.starts_with(&motif[p..]))
        .unwrap_or(motif.len())
}

/// Locate the CCA|adapter junction in every reference record.
///
/// `motif` (e.g. `CCAGGC`) must occur exactly once per record; the junction
/// base is `index(motif) + motif_offset`. The full `common_arm` must follow
/// at the junction — a record violating either is a hard error.
pub fn junction_positions(
    fasta_path: &Path,
    motif: &str,
    motif_offset: usize,
    common_arm: &str,
) -> Result<HashMap<String, RefGeometry>> {
    junction_positions_with_anchors(fasta_path, motif, motif_offset, common_arm, None)
}

/// Reference offsets of the flank anchors, relative to the junction.
///
/// Chosen from the measured damage profile: the unaligned-base rate on charged
/// reads is elevated from -6 to +15 and back to the uncharged baseline by -8
/// on the left and +19 on the right.
pub const FLANK_ANCHORS: (i64, i64) = (-10, 20);

/// As [`junction_positions`], with flank anchors installed when usable.
///
/// The RIGHT anchor is adapter-family specific and cannot be assumed: at +20 it
/// lies past a 17 nt common arm, in the `AGGAAGGC` every edx/ndx adapter
/// shares — but on the v2 single-adapter references those offsets are the
/// divergent 13-mer, which IS the library label. Anchoring there would place
/// the window from the one region the design forbids the model to read, so the
/// anchor is installed only when its context is CONSTANT across every record,
/// and disabled everywhere if any record disagrees. A per-record decision would
/// silently anchor some references differently from others.
pub fn junction_positions_with_anchors(
    fasta_path: &Path,
    motif: &str,
    motif_offset: usize,
    common_arm: &str,
    flank_anchors: Option<(i64, i64)>,
) -> Result<HashMap<String, RefGeometry>> {
    let (lo_off, hi_off) = flank_anchors.unwrap_or(FLANK_ANCHORS);
    let arm_off = arm_offset(motif, common_arm);
    let mut out = HashMap::new();
    let seqs = read_fasta(fasta_path)?;
    if seqs.is_empty() {
        bail!("no records in reference FASTA {}", fasta_path.display());
    }
    let mut ref_ctx: Vec<Option<String>> = Vec::new();
    for (name, seq) in seqs {
        // The motif alone does not identify the junction. It is CCA plus the
        // opening bases of the 3' adapter, so a tRNA body can carry it by
        // chance -- ~1.5% of records in both hg38 (4/282) and danRer11
        // (51/3315), which is just the rate a 6-mer collides over ~75 nt of
        // body. Requiring uniqueness here turned that coincidence into a hard
        // failure of the entire run, and dropping the offending records is not
        // neutral: 48 of danRer11's 51 are Glu, so it would remove most of one
        // amino-acid family from a charging analysis.
        //
        // What actually makes the junction unique is the common arm that must
        // follow it; an internal match is in the body and has no arm after it.
        // So filter on the arm FIRST and require uniqueness of what survives.
        let candidates: Vec<usize> = seq
            .match_indices(motif)
            .map(|(i, _)| i)
            .filter(|&i| {
                let a = i + arm_off;
                seq.len() >= a + common_arm.len() && &seq[a..a + common_arm.len()] == common_arm
            })
            .collect();
        let motif_start = match candidates.as_slice() {
            [only] => *only,
            [] => bail!(
                "{}: no {} followed by the common arm {} -- the record does not \
                 end in CCA + adapter",
                name,
                motif,
                common_arm
            ),
            many => bail!(
                "{}: {} occurrences of {} are followed by the common arm; the \
                 junction cannot be placed",
                name,
                many.len(),
                motif
            ),
        };
        // The arm's first base, which is what makes the motif unique, and the
        // model's anchor, which need not be the same base.
        let arm_start = motif_start + arm_off;
        let j = motif_start + motif_offset;
        let anchor_at = |off: i64| -> Option<usize> {
            let p = j as i64 + off;
            if p >= 0 && (p as usize) < seq.len() {
                Some(p as usize)
            } else {
                None
            }
        };
        let right = anchor_at(hi_off);
        // Two bases either side, so a single-base difference between panels
        // still disqualifies the anchor.
        ref_ctx.push(right.and_then(|r| {
            let (a, b) = (r.saturating_sub(2), (r + 3).min(seq.len()));
            seq.get(a..b).map(|s| s.to_string())
        }));
        out.insert(
            name,
            RefGeometry {
                motif_start,
                junction: j,
                cca_a: arm_start - 1,
                divergent: arm_start + common_arm.len(),
                body_mid: (FIVEP_LEN + j) / 2,
                polya_mid: {
                    let n_a = seq.len() - seq.trim_end_matches('A').len();
                    if n_a >= 5 {
                        seq.len() - n_a + 4
                    } else {
                        seq.len()
                    }
                },
                left_anchor: anchor_at(lo_off),
                right_anchor: right,
            },
        );
    }

    // Disable the right anchor everywhere unless every record agrees on its
    // context. See the doc comment: on a divergent panel this is the library
    // label, and anchoring on it would be worse than not anchoring at all.
    let usable =
        !ref_ctx.is_empty() && ref_ctx[0].is_some() && ref_ctx.iter().all(|c| c == &ref_ctx[0]);
    if !usable {
        for g in out.values_mut() {
            g.right_anchor = None;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_fasta(content: &str) -> tempfile::NamedTempFile {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(content.as_bytes()).unwrap();
        f
    }

    const ARM: &str = "GGCTTCTTCTTGCTCTT";

    #[test]
    fn test_junction_positions() {
        let body = "ACGT".repeat(20);
        let fa = format!(">r1 desc\n{body}CCA{ARM}TTTTT\n");
        let f = write_fasta(&fa);
        let geo = junction_positions(f.path(), "CCAGGC", 3, ARM).unwrap();
        let g = geo["r1"];
        assert_eq!(g.junction, 83); // 80 body + "CCA"
        assert_eq!(g.cca_a, 82);
        assert_eq!(g.divergent, 83 + ARM.len());
        assert_eq!(g.body_mid, (24 + 83) / 2);
    }

    #[test]
    fn test_rejects_missing_or_duplicate_motif() {
        let f = write_fasta(">r1\nACGTACGTACGT\n");
        assert!(junction_positions(f.path(), "CCAGGC", 3, ARM).is_err());

        let two = format!(">r1\nCCA{ARM}AACCA{ARM}\n");
        let f = write_fasta(&two);
        assert!(junction_positions(f.path(), "CCAGGC", 3, ARM).is_err());
    }

    /// A tRNA body can carry the junction motif by chance: it is CCA plus the
    /// adapter's opening bases, and a 6-mer collides over ~75 nt of body in
    /// ~1.5% of records (4/282 in hg38, 51/3315 in danRer11). Such a record is
    /// perfectly well formed -- the motif simply is not, on its own, the
    /// junction. The common arm that must follow it is.
    #[test]
    fn test_internal_motif_is_not_the_junction() {
        let body = format!("{}CCAGGCAAAA{}", "ACGT".repeat(10), "ACGT".repeat(5));
        let fa = format!(">r1\n{body}CCA{ARM}TTTTT\n");
        let f = write_fasta(&fa);
        let geo = junction_positions(f.path(), "CCAGGC", 3, ARM).unwrap();
        // The junction is the arm-backed match, not the first one in the body.
        assert_eq!(geo["r1"].junction, body.len() + 3);
    }

    /// The arm overlaps the motif's tail — `CCAGGC` is `CCA` plus the arm's
    /// first three bases — and the offset is derived from the two strings
    /// rather than declared, so a third number cannot disagree with them.
    #[test]
    fn arm_offset_is_derived_from_the_two_strings() {
        assert_eq!(arm_offset("CCAGGC", ARM), 3);
        // No overlap: the arm simply follows the motif.
        assert_eq!(arm_offset("CCAGGC", "TTTTT"), 6);
        // A full overlap: the motif IS the arm's opening.
        assert_eq!(arm_offset("GGCTT", ARM), 0);
        // An arm too short to hold a longer overlap falls back to the
        // no-overlap reading rather than failing; the reference filter then
        // simply finds no record, which is the honest answer.
        assert_eq!(arm_offset("CCAGGC", "GG"), 6);
    }

    /// The motif's own origin is kept beside the junction, because a model may
    /// anchor its window elsewhere inside the motif — the windowed charging
    /// variant anchors at +2, one base before the arm — and a second scan at a
    /// different offset would find a different set of records.
    #[test]
    fn the_motif_start_survives_a_different_anchor_offset() {
        let body = "ACGT".repeat(20);
        let fa = format!(">r1\n{body}CCA{ARM}TTTTT\n");
        let f = write_fasta(&fa);

        let at_arm = junction_positions(f.path(), "CCAGGC", 3, ARM).unwrap()["r1"];
        let at_cca = junction_positions(f.path(), "CCAGGC", 2, ARM).unwrap()["r1"];

        // One motif, located identically either way...
        assert_eq!(at_arm.motif_start, 80);
        assert_eq!(at_cca.motif_start, 80);
        // ...and the anchor moves with the offset the caller asked for.
        assert_eq!(at_arm.junction, 83);
        assert_eq!(at_cca.junction, 82);
        // The arm-derived coordinates do NOT move: they are properties of the
        // construct, not of where a model happens to look.
        assert_eq!(at_arm.cca_a, at_cca.cca_a);
        assert_eq!(at_arm.divergent, at_cca.divergent);
    }

    #[test]
    fn test_rejects_broken_arm() {
        let f = write_fasta(">r1\nAAAACCAGGCAAAAAAAAAAAAAAA\n");
        assert!(junction_positions(f.path(), "CCAGGC", 3, ARM).is_err());
    }
}
