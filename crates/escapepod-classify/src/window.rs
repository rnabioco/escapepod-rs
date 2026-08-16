// SPDX-License-Identifier: MIT

//! The raw signal window around the junction — the other half of the
//! charging feature contract.
//!
//! [`crate::features`] summarises per-base spans into the GBM's grid. This
//! module produces the other input a read-level model can take: a fixed-width
//! slice of calibrated pA around the junction, padded where the read runs out
//! and masked where we are not allowed to look.
//!
//! Four rules, all of them contract rather than convenience:
//!
//! 1. **Where the window is anchored inside the junction base**
//!    ([`BaseJustify`]). A fixed offset from the base's *start* covers a
//!    different number of bases on a fast read than a slow one, so a
//!    start-justified window encodes translocation rate — which tracks
//!    flowcell, not chemistry.
//! 2. **The slice is `[anchor - left, anchor + right)`**, in raw sample
//!    coordinates, in the run's own frame (the coords are already resolved
//!    there by [`crate::finalize`]).
//! 3. **Padding is `NaN`, never zero or an edge value.** Zero is a legal
//!    picoamp reading; a model cannot tell invented signal from real signal
//!    unless the padding is unrepresentable.
//! 4. **Everything earlier in time than `common_start_sig` is masked.** The
//!    single-ligation training libraries carry adapters that differ in
//!    sequence a short distance after the CCA, so a model that can see into
//!    that region learns to classify the *adapter*, and production adapters
//!    differ again. Under-masking here is not a degraded model, it is a model
//!    measuring the wrong thing.
//!
//! This lived only in the pyo3 binding, where the Rust inference path could
//! not see it and could not reproduce it — the same structural mistake that
//! left this pipeline with two divergent feature definitions once already.
//! It belongs next to the feature grid, in the crate both callers link.

use crate::anchor::JunctionCoords;

/// Where the window is anchored inside the junction base's own signal span.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BaseJustify {
    /// The base's first sample. Cheapest, and rate-dependent: the window
    /// reaches a different number of bases on a fast read than a slow one.
    Start,
    /// The base's midpoint.
    Center,
    /// The base's last sample, i.e. the boundary with the next base.
    End,
}

impl BaseJustify {
    /// Stable lowercase name, the spelling callers pass as a string.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Start => "start",
            Self::Center => "center",
            Self::End => "end",
        }
    }

    /// How far into a junction base of `dwell` samples this justification
    /// puts the anchor.
    fn shift(self, dwell: i64) -> i64 {
        match self {
            Self::Start => 0,
            Self::Center => dwell / 2,
            Self::End => dwell,
        }
    }
}

impl std::str::FromStr for BaseJustify {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "start" => Ok(Self::Start),
            "center" => Ok(Self::Center),
            "end" => Ok(Self::End),
            other => Err(format!(
                "base_justify must be start|center|end, got {other:?}"
            )),
        }
    }
}

/// The `left + right` sample window around this read's junction.
///
/// `sig_pa` is the read's full calibrated signal in the run's frame, and
/// `coords` its resolved junction coordinates ([`crate::finalize`]) — the
/// anchor, the junction base's dwell and the mask boundary all come from
/// there, so the window cannot be built against a different resolution of
/// the read than its features were.
///
/// The result is always `left + right` long: real samples where the read has
/// them, `NaN` where it does not, and `NaN` for every position earlier in
/// time than `coords.common_start_sig`.
///
/// Returns `None` when fewer than `right` real samples fall inside the
/// window at all — that is the reject rule the corpus builder applies, and
/// note it counts samples on *both* sides of the anchor, so a read can be
/// kept with fewer than `right` samples after the anchor as long as the
/// left side makes up the count.
///
/// Requires `left >= 0` and `right > 0`; the caller validates its own inputs
/// (a CLI flag or a binding argument wants its own error message).
pub fn signal_window(
    sig_pa: &[f32],
    coords: &JunctionCoords,
    left: i64,
    right: i64,
    justify: BaseJustify,
) -> Option<Vec<f32>> {
    debug_assert!(left >= 0 && right > 0, "left >= 0 and right > 0 required");

    let anchor = coords.junction_sig + justify.shift(coords.junction_dwell.max(0));
    let (lo, hi) = (anchor - left, anchor + right);
    let (src_lo, src_hi) = (lo.max(0), hi.min(sig_pa.len() as i64));
    if src_hi - src_lo < right {
        return None;
    }

    let w = (left + right) as usize;
    let mut window = vec![f32::NAN; w];
    let dst = (src_lo - lo) as usize;
    window[dst..dst + (src_hi - src_lo) as usize]
        .copy_from_slice(&sig_pa[src_lo as usize..src_hi as usize]);

    // Mask everything earlier than the common arm: poly(A) and the divergent
    // adapter, which differ between libraries.
    let cut = coords.common_start_sig - lo;
    if cut > 0 {
        let end = (cut as usize).min(w);
        window[..end].fill(f32::NAN);
    }
    Some(window)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::anchor::MaskSource;

    /// Coords with only the fields the window rule reads set meaningfully.
    fn coords(junction_sig: i64, junction_dwell: i64, common_start_sig: i64) -> JunctionCoords {
        JunctionCoords {
            feat_spans: Vec::new(),
            common_start_sig,
            junction_sig,
            mask_source: MaskSource::Exact,
            cca_a_sig: -1,
            cca_a_dwell: 0,
            junction_dwell,
            arm_resolved_depth: 0,
            aligner_arm_depth: 0,
            polya_mid_sig: -1,
            body_mid_sig: -1,
        }
    }

    /// `sig[i] == i as f32`, so a window's contents name their own source
    /// index and an off-by-one is visible rather than plausible.
    fn ramp(n: usize) -> Vec<f32> {
        (0..n).map(|i| i as f32).collect()
    }

    #[test]
    fn window_is_the_half_open_slice_around_the_anchor() {
        let sig = ramp(100);
        // Mask boundary at 0 so nothing is masked.
        let w = signal_window(&sig, &coords(50, 0, 0), 4, 6, BaseJustify::Start).unwrap();
        assert_eq!(w, vec![46., 47., 48., 49., 50., 51., 52., 53., 54., 55.]);
    }

    #[test]
    fn each_justification_moves_the_anchor_into_the_base() {
        let sig = ramp(100);
        // Junction base spans samples 50..60, so dwell = 10.
        let c = coords(50, 10, 0);
        for (justify, want_first) in [
            (BaseJustify::Start, 50.0),
            (BaseJustify::Center, 55.0),
            (BaseJustify::End, 60.0),
        ] {
            let w = signal_window(&sig, &c, 0, 4, justify).unwrap();
            assert_eq!(
                w[0],
                want_first,
                "{} should anchor at {want_first}",
                justify.as_str()
            );
        }
    }

    #[test]
    fn a_zero_dwell_base_justifies_nowhere() {
        let sig = ramp(100);
        let c = coords(50, 0, 0);
        for justify in [BaseJustify::Start, BaseJustify::Center, BaseJustify::End] {
            let w = signal_window(&sig, &c, 0, 2, justify).unwrap();
            assert_eq!(w[0], 50.0, "{} shifted a 0-dwell base", justify.as_str());
        }
    }

    #[test]
    fn a_window_running_off_the_start_is_nan_padded_on_the_left() {
        let sig = ramp(100);
        // anchor 3, left 5 → lo = -2, so the first two slots are padding.
        let w = signal_window(&sig, &coords(3, 0, 0), 5, 5, BaseJustify::Start).unwrap();
        assert_eq!(w.len(), 10);
        assert!(w[0].is_nan() && w[1].is_nan(), "left padding must be NaN");
        assert_eq!(w[2..], [0., 1., 2., 3., 4., 5., 6., 7.]);
    }

    #[test]
    fn a_window_running_off_the_end_is_nan_padded_on_the_right() {
        let sig = ramp(100);
        // anchor 97, right 5 → hi = 102, so the last two slots are padding.
        let w = signal_window(&sig, &coords(97, 0, 0), 5, 5, BaseJustify::Start).unwrap();
        assert_eq!(w.len(), 10);
        assert_eq!(w[..8], [92., 93., 94., 95., 96., 97., 98., 99.]);
        assert!(
            w[8..].iter().all(|v| v.is_nan()),
            "right padding must be NaN"
        );
    }

    /// The reject rule counts real samples in the whole window, not samples
    /// after the anchor — pinned because the two differ exactly when a read
    /// ends inside its own window.
    #[test]
    fn too_few_real_samples_rejects_the_read() {
        let sig = ramp(100);
        // 5 samples past the anchor, and no left context to make up the count.
        assert!(signal_window(&sig, &coords(95, 0, 0), 0, 10, BaseJustify::Start).is_none());
        // Same anchor, same `right`, but the left side supplies the samples.
        assert!(signal_window(&sig, &coords(95, 0, 0), 10, 10, BaseJustify::Start).is_some());
        // Exactly `right` real samples is enough.
        assert!(signal_window(&sig, &coords(90, 0, 0), 0, 10, BaseJustify::Start).is_some());
    }

    /// The mask boundary is inclusive of `common_start_sig`: that sample
    /// survives and the one before it does not. Off by one here silently
    /// leaks a sample of the divergent adapter into every window, or throws
    /// away a sample of real arm from every window — neither errors.
    #[test]
    fn the_mask_boundary_lands_exactly_on_common_start() {
        let sig = ramp(100);
        // Window covers 40..60; the arm starts at sample 50.
        let w = signal_window(&sig, &coords(50, 0, 50), 10, 10, BaseJustify::Start).unwrap();
        let first_surviving = w.iter().position(|v| v.is_finite()).unwrap();
        assert_eq!(first_surviving, 10, "index 10 is absolute sample 50");
        assert_eq!(w[first_surviving], 50.0);
        assert!(w[first_surviving - 1].is_nan(), "sample 49 must be masked");
        assert!(w[..10].iter().all(|v| v.is_nan()));
        assert!(w[10..].iter().all(|v| v.is_finite()));
    }

    #[test]
    fn a_mask_boundary_before_the_window_masks_nothing() {
        let sig = ramp(100);
        let w = signal_window(&sig, &coords(50, 0, 40), 10, 10, BaseJustify::Start).unwrap();
        assert!(w.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn a_mask_boundary_past_the_window_masks_everything() {
        let sig = ramp(100);
        let w = signal_window(&sig, &coords(50, 0, 90), 10, 10, BaseJustify::Start).unwrap();
        assert!(w.iter().all(|v| v.is_nan()));
    }

    /// Justification moves the window, so it moves how much of the window
    /// the mask covers — the two rules compose rather than being applied to
    /// independent copies of the anchor.
    #[test]
    fn justification_shifts_the_masked_prefix_too() {
        let sig = ramp(100);
        let c = coords(50, 10, 50);
        let start = signal_window(&sig, &c, 10, 10, BaseJustify::Start).unwrap();
        let end = signal_window(&sig, &c, 10, 10, BaseJustify::End).unwrap();
        assert_eq!(start.iter().filter(|v| v.is_nan()).count(), 10);
        // End-justified: window covers 50..70, all of it at or after the arm.
        assert_eq!(end.iter().filter(|v| v.is_nan()).count(), 0);
    }

    #[test]
    fn base_justify_round_trips_through_its_name() {
        for j in [BaseJustify::Start, BaseJustify::Center, BaseJustify::End] {
            assert_eq!(j.as_str().parse::<BaseJustify>(), Ok(j));
        }
        assert!("sideways".parse::<BaseJustify>().is_err());
    }
}
