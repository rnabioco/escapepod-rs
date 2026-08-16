// SPDX-License-Identifier: MIT

//! Read-level classification against escapepod-models bundles.
//!
//! Today this is one classifier: the tRNA charging (aminoacylation) model
//! (rnabioco/escapepod-rs#204, rnabioco/escapepod-models#60). Unlike
//! `escpod demux`, which anchors on a signal-derived `adapter_end`, this
//! model anchors on the **CCA–aa junction**, which only exists in reference
//! coordinates — so it takes POD5 *and* an aligned BAM with move tables, the
//! same input pair `remora infer from_pod5_and_bam` takes.
//!
//! The per-read chain (the reference implementation is
//! `escapepod_models.charging`, which built the training corpus — this crate
//! must compute the *same* features, because a caller that computes them
//! differently gets a wrong answer, not an error):
//!
//! 1. locate the anchor motif (`CCAGGC`; the junction is `+3`, the adapter G
//!    after CCA) in every reference record — [`geometry`];
//! 2. reference → query positions via the CIGAR (nearest aligned within a
//!    2-base slop) — [`anchor`];
//! 3. query → signal via the move table, Remora convention
//!    (`move_position * stride + ts`) — [`anchor`];
//! 4. detect per run whether the move table indexes time-ordered or
//!    *reversed* signal: on every corpus built so far it was reversed, and
//!    getting this wrong silently mirrors the window rather than failing, so
//!    it is voted from the data (junction-before-body in time), never
//!    assumed — [`anchor::resolve_orientation`];
//! 5. per-base signal spans → dwell, mean, std over the recipe's offsets,
//!    levels per-**read** median/MAD standardised (a per-window gauge would
//!    subtract the level shift being measured) — [`features`];
//! 6. expected k-mer levels, z-scored per read → the *residual* (observed
//!    minus expected level), the model's decisive feature — [`features`];
//! 7. everything earlier in time than the common arm is never read: the
//!    training libraries' adapters diverge 17 nt after the CCA, so those
//!    samples would classify the adapter, not the chemistry — [`features`]
//!    for the per-base grid, [`window`] for the raw window a signal-level
//!    model takes instead.
//!
//! The recipe — feature order, offsets, stat layout, mask rule, the k-mer
//! table pinned by sha256, and the recommended operating point — comes from
//! the model bundle's `metadata.json` ([`bundle`]), not from flags.
//!
//! The part of it that defines the *feature space* (offsets, span mode,
//! k-mer levels) is [`recipe::FeatureRecipe`], separate from the bundle that
//! parses it: computing these features is also what builds the training
//! corpus, and that caller has no weights to hand over.

pub mod anchor;
pub mod bam_tags;
pub mod bundle;
pub mod features;
pub mod geometry;
pub mod pipeline;
pub mod recipe;
pub mod window;

pub use anchor::{
    AnchoredRead, MaskSource, Orientation, OrientationVotes, ScanOutcome, SpanMode,
    resolve_orientation,
};
pub use anchor::{JunctionCoords, SkipReason, finalize, query_positions, scan_record};
pub use bundle::{ChargingBundle, OperatingPoint};
pub use features::{FEAT_STATS, expected_levels_z, junction_features};
pub use geometry::{RefGeometry, junction_positions};
pub use pipeline::{
    BamScan, ClassifyStats, Pod5Index, ReadCall, classify_reads, feature_grid, feature_grid_at,
    scan_bam, signal_pa,
};
pub use recipe::{FeatureRecipe, KmerLevels};
pub use window::{BaseJustify, signal_window};

/// Encode a probability as the `cl` BAM tag value: `round(p * 255)` as u8.
///
/// This is the encoding rnabioco/escapepod-rs#204 §3 specifies (and the
/// bundle records), replacing the modbase `ML`→`cl` round-trip the Remora
/// pipeline needed.
pub fn cl_from_probability(p: f64) -> u8 {
    (p.clamp(0.0, 1.0) * 255.0).round() as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cl_encoding() {
        assert_eq!(cl_from_probability(0.0), 0);
        assert_eq!(cl_from_probability(1.0), 255);
        assert_eq!(cl_from_probability(0.5), 128); // 127.5 rounds half-up
        assert_eq!(cl_from_probability(0.784), 200); // the legacy threshold
        assert_eq!(cl_from_probability(-0.1), 0);
        assert_eq!(cl_from_probability(1.1), 255);
    }
}
