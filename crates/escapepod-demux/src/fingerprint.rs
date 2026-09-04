//! Barcode fingerprint types and extraction.
//!
//! The pipeline:
//! 1. `extract_fingerprint_from_signal` — given a read's raw signal and its
//!    detected adapter boundaries, segment the adapter region with
//!    t-test changepoints and take each segment's mean as a fingerprint
//!    feature.
//! 2. `compute_consensus_fingerprint` / `compute_std_dev_fingerprint` —
//!    aggregate per-read fingerprints into a reference barcode
//!    fingerprint at training time.

use escapepod_signal::dtw::{Fingerprint, NormMethod, normalize_fingerprint};
use escapepod_signal::segmentation::{
    clip_outliers, mad_normalize, normalize_dwell_times, segment_signal,
};
use uuid::Uuid;

/// Per-read adapter boundaries — the output of `detect_adapter` augmented
/// with read identity. Flows from the `detect` stage into `fingerprint`.
#[derive(Debug, Clone)]
pub struct ReadBoundaries {
    /// The read identifier
    pub read_id: Uuid,
    /// Total number of samples in the read
    pub num_samples: u64,
    /// Start position of the adapter region
    pub adapter_start: usize,
    /// End position of the adapter region
    pub adapter_end: usize,
}

impl ReadBoundaries {
    /// Check if the adapter region is valid (end > start).
    pub fn has_valid_adapter(&self) -> bool {
        self.adapter_end > self.adapter_start
    }
}

/// A fingerprint extracted from a single read's adapter region.
#[derive(Debug, Clone)]
pub struct ReadFingerprint {
    /// The read identifier
    pub read_id: Uuid,
    /// The fingerprint feature values (segment means, normalized)
    pub values: Vec<f64>,
    /// Per-segment dwell times (in samples), `log1p + z-score` normalized,
    /// aligned 1:1 with `values`. `None` when the caller didn't request
    /// `emit_dwell`; `Some` with `values.len()` entries otherwise.
    pub dwell_times: Option<Vec<f64>>,
}

impl ReadFingerprint {
    /// Create a new read fingerprint (segment means only, no dwell).
    pub fn new(read_id: Uuid, values: Vec<f64>) -> Self {
        Self {
            read_id,
            values,
            dwell_times: None,
        }
    }

    /// Create a read fingerprint with segment means and per-segment dwell
    /// times appended as a parallel feature channel. `dwell_times.len()`
    /// must match `values.len()` — aligned per-segment.
    pub fn with_dwell_times(read_id: Uuid, values: Vec<f64>, dwell_times: Vec<f64>) -> Self {
        debug_assert_eq!(
            values.len(),
            dwell_times.len(),
            "dwell_times must align 1:1 with values",
        );
        Self {
            read_id,
            values,
            dwell_times: Some(dwell_times),
        }
    }
}

/// A reference barcode fingerprint for classification.
#[derive(Debug, Clone)]
pub struct BarcodeFingerprint {
    /// The barcode name (e.g., "BC01")
    pub barcode: String,
    /// The fingerprint feature values
    pub values: Vec<f32>,
}

impl BarcodeFingerprint {
    /// Create a new barcode fingerprint.
    pub fn new(barcode: String, values: Vec<f32>) -> Self {
        Self { barcode, values }
    }
}

/// Context kept on either side of the nominal adapter boundaries under the
/// `keep_last` pipeline, in samples. Public because a caller that decodes only
/// a prefix of the read (the `fingerprint` subcommand) has to decode at least
/// this far past `adapter_end`, or the right-hand context silently vanishes.
pub const BOUNDARY_PADDING_SAMPLES: usize = 100;

/// The most samples, ending at `adapter_end`, that a fingerprint is ever
/// computed from.
///
/// The fingerprint keeps the *last* segments of the window — the barcode sits
/// against `adapter_end` — so only the tail of a wide window ever reaches the
/// model, but every sample of it was paid for: two `f64` prefix sums, a
/// t-score per position, and a sort of every local maximum. A boundary
/// detector with nothing to bound it can hand this function a window of
/// millions of samples (the LLR detector on a read that is one long stall),
/// and one such read then runs for seconds on one thread while the rest of
/// the pool sits idle behind it.
///
/// Measured on a MinKNOW RNA004 run (119,255 reads): windows over 20k samples
/// were 1.0% of reads and 41% of all window samples, with a maximum of 4.6M;
/// the pipeline spent its last 6 of 11 seconds on a single worker inside this
/// function. Real adapter windows on that run sat at 4,070 (median) and 5,120
/// (p90) samples, so 30,000 — seven seconds of signal at 4 kHz — is several
/// times anything an adapter produces, and clamping a wider window to its
/// last 30,000 samples leaves the retained tail untouched. The CNN detector
/// caps `adapter_end` at 6,500 samples and never reaches this.
pub const MAX_FINGERPRINT_WINDOW: usize = 30_000;

/// Extract a fingerprint from an adapter region of a signal.
///
/// Returns `None` if the region is too small or segmentation fails.
///
/// Two pipeline variants are selected by `keep_last`:
/// * `None` — segment the (MAD-normalized) adapter signal directly and
///   normalize the resulting feature vector once.
/// * `Some(n)` — segment the clipped (but not pre-normalized) signal,
///   normalize the full segment-mean population, then keep the last `n`
///   features. Normalizing over the full population before truncation makes
///   the z-score stable against the choice of `n`; truncating first would
///   leave the statistics at the mercy of the small retained tail.
#[allow(clippy::too_many_arguments)]
pub fn extract_fingerprint_from_signal(
    signal: &[i16],
    adapter_start: usize,
    adapter_end: usize,
    num_segments: usize,
    window_width: usize,
    norm_method: NormMethod,
    read_id: Uuid,
    min_separation: Option<usize>,
    keep_last: Option<usize>,
    emit_dwell: bool,
) -> Option<ReadFingerprint> {
    // When the caller selects the "keep last N" pipeline, widen the slice
    // by a small fixed buffer on each side. Adapter-boundary detectors are
    // approximate (LLR / CNN both report positions accurate to within a few
    // tens of samples), and segmentation benefits from a little context on
    // either side of the nominal boundary so changepoints near the edges
    // aren't clamped. The buffer is a fixed sample count rather than a
    // fraction so it remains meaningful regardless of adapter length.
    let (slice_start, slice_end) = if keep_last.is_some() {
        let ss = adapter_start.saturating_sub(BOUNDARY_PADDING_SAMPLES);
        let se = adapter_end
            .saturating_add(BOUNDARY_PADDING_SAMPLES)
            .min(signal.len());
        (ss, se)
    } else {
        (adapter_start, adapter_end.min(signal.len()))
    };
    // Bound the work to the tail of the window, where the barcode is. See
    // [`MAX_FINGERPRINT_WINDOW`]; a no-op for every real adapter.
    let slice_start = slice_start.max(slice_end.saturating_sub(MAX_FINGERPRINT_WINDOW));

    if slice_end <= slice_start || slice_end - slice_start < window_width * 2 {
        return None;
    }

    // Pipeline-variant data prep:
    // * `keep_last == Some(_)`: clip extreme excursions but leave the scale
    //   alone — segmentation operates on (clipped) raw samples and the
    //   normalization step happens later at the feature level.
    // * `keep_last == None`: MAD-normalize up front so the t-test scores
    //   downstream are scale-invariant and directly comparable across reads.
    // The 5×MAD clip cutoff is a conventional "extreme outlier" threshold
    // from robust statistics — wider than 3σ-equivalent (≈3×MAD) so genuine
    // signal extremes survive, narrow enough to suppress spike artefacts.
    let adapter_signal: Vec<f32> = if keep_last.is_some() {
        let raw: Vec<f32> = signal[slice_start..slice_end]
            .iter()
            .map(|&s| s as f32)
            .collect();
        clip_outliers(&raw, 5.0)
    } else {
        let raw: Vec<f32> = signal[slice_start..slice_end]
            .iter()
            .map(|&s| s as f32)
            .collect();
        if raw.len() > 10 {
            mad_normalize(&raw)
        } else {
            raw
        }
    };

    let sep = min_separation.unwrap_or(window_width);

    // Segment the adapter region
    let segments = segment_signal(
        &adapter_signal,
        window_width,
        num_segments.saturating_sub(1),
        sep,
    );

    if segments.is_empty() {
        return None;
    }

    // Extract segment means as fingerprint. Dwell (end - start in samples)
    // comes for free from the same segment tuples — skip the Vec build
    // entirely when emit_dwell is false so the non-dwell path stays at its
    // current cost.
    let mut fingerprint_values: Vec<f32> =
        segments.iter().map(|(_, _, mean)| *mean as f32).collect();
    let mut dwell_values: Vec<f32> = if emit_dwell {
        segments
            .iter()
            .map(|(s, e, _)| e.saturating_sub(*s) as f32)
            .collect()
    } else {
        Vec::new()
    };

    if let Some(n) = keep_last {
        // Normalize the full segment-mean population first, then truncate to
        // the tail of length `n`. Doing it in this order makes the z-score
        // statistics depend on the full adapter feature distribution rather
        // than on the small retained slice; truncate-then-normalize would
        // make the output sensitive to the choice of `n` and to whatever
        // happens to live in the kept tail.
        let mut all_fp = Fingerprint::new(fingerprint_values, read_id);
        normalize_fingerprint(&mut all_fp, norm_method);
        fingerprint_values = all_fp.values;

        // `keep_last` is a fixed feature width, not a maximum: every consumer
        // (SVM/GBM model input, the fingerprint CSV/parquet schema) requires
        // exactly `n` values. A read whose adapter yielded fewer segments than
        // that cannot produce a usable fingerprint, so reject it rather than
        // emitting a short row. Returning a truncated fingerprint here was
        // silently poisoning downstream: the GBM path errors out on the width
        // mismatch, but the DTW/SVM path happily scores the short query and
        // reports a confident — and wrong — barcode.
        if fingerprint_values.len() < n {
            return None;
        }
        fingerprint_values = fingerprint_values[fingerprint_values.len() - n..].to_vec();

        if emit_dwell {
            // Dwell features take the opposite order: truncate first, then
            // normalize. Adapter-onset segments are typically far longer
            // than barcode-region segments, and a log1p+z-score computed
            // over the full population would have its location/scale
            // dominated by those early outliers — leaving the kept-tail
            // values bunched near zero. Normalizing only over the kept tail
            // keeps the dwell channel discriminative for the barcode.
            if dwell_values.len() < n {
                return None;
            }
            dwell_values = dwell_values[dwell_values.len() - n..].to_vec();
            dwell_values = normalize_dwell_times(&dwell_values);
            return Some(ReadFingerprint::with_dwell_times(
                read_id,
                fingerprint_values.iter().map(|&v| v as f64).collect(),
                dwell_values.iter().map(|&v| v as f64).collect(),
            ));
        }

        return Some(ReadFingerprint::new(
            read_id,
            fingerprint_values.iter().map(|&v| v as f64).collect(),
        ));
    }

    let mut fp = Fingerprint::new(fingerprint_values, read_id);
    normalize_fingerprint(&mut fp, norm_method);

    let values_f64: Vec<f64> = fp.values.iter().map(|&v| v as f64).collect();
    if emit_dwell {
        let dwell_f64: Vec<f64> = normalize_dwell_times(&dwell_values)
            .iter()
            .map(|&v| v as f64)
            .collect();
        Some(ReadFingerprint::with_dwell_times(
            read_id, values_f64, dwell_f64,
        ))
    } else {
        Some(ReadFingerprint::new(read_id, values_f64))
    }
}

/// Compute consensus fingerprint as element-wise median.
///
/// Filters fingerprints to only include those with the most common length,
/// so that mixed-width reference sets don't pollute the median.
pub fn compute_consensus_fingerprint(fingerprints: &[Vec<f32>]) -> Vec<f32> {
    if fingerprints.is_empty() {
        return Vec::new();
    }

    let mut length_counts: std::collections::HashMap<usize, usize> =
        std::collections::HashMap::new();
    for fp in fingerprints {
        *length_counts.entry(fp.len()).or_insert(0) += 1;
    }
    let target_length = length_counts
        .into_iter()
        // Most common length, shortest on a tie. The tie used to fall to
        // `HashMap` iteration order, so two equally common lengths gave a
        // different consensus per process — the last non-determinism in
        // `demux train`'s output.
        .max_by_key(|&(len, count)| (count, std::cmp::Reverse(len)))
        .map(|(len, _)| len)
        .unwrap_or(0);

    let filtered: Vec<&Vec<f32>> = fingerprints
        .iter()
        .filter(|fp| fp.len() == target_length)
        .collect();
    if filtered.is_empty() {
        return Vec::new();
    }

    let mut consensus = Vec::with_capacity(target_length);

    for i in 0..target_length {
        let mut values: Vec<f32> = filtered.iter().map(|fp| fp[i]).collect();
        values.sort_unstable_by(|a, b| a.total_cmp(b));

        let median = if values.len().is_multiple_of(2) {
            let mid = values.len() / 2;
            (values[mid - 1] + values[mid]) / 2.0
        } else {
            values[values.len() / 2]
        };

        consensus.push(median);
    }

    consensus
}

/// Compute element-wise standard deviation of a set of fingerprints relative
/// to a consensus. Only uses fingerprints that match the consensus length.
pub fn compute_std_dev_fingerprint(fingerprints: &[Vec<f32>], consensus: &[f32]) -> Vec<f32> {
    if fingerprints.is_empty() || consensus.is_empty() {
        return Vec::new();
    }

    let length = consensus.len();

    let filtered: Vec<&Vec<f32>> = fingerprints
        .iter()
        .filter(|fp| fp.len() == length)
        .collect();
    if filtered.is_empty() {
        return vec![0.0; length];
    }

    let mut std_dev = Vec::with_capacity(length);

    for i in 0..length {
        let mean = consensus[i];
        let variance = filtered
            .iter()
            .map(|fp| {
                let d = fp[i] - mean;
                d * d
            })
            .sum::<f32>()
            / filtered.len() as f32;
        std_dev.push(variance.sqrt());
    }

    std_dev
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_consensus_fingerprint_empty() {
        let result = compute_consensus_fingerprint(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_compute_consensus_fingerprint_single() {
        let fingerprints = vec![vec![1.0, 2.0, 3.0]];
        let result = compute_consensus_fingerprint(&fingerprints);
        assert_eq!(result, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_compute_consensus_fingerprint_multiple_odd() {
        let fingerprints = vec![
            vec![1.0, 2.0, 3.0],
            vec![2.0, 3.0, 4.0],
            vec![3.0, 4.0, 5.0],
        ];
        let result = compute_consensus_fingerprint(&fingerprints);
        assert_eq!(result, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_compute_consensus_fingerprint_multiple_even() {
        let fingerprints = vec![
            vec![1.0, 2.0],
            vec![2.0, 4.0],
            vec![3.0, 6.0],
            vec![4.0, 8.0],
        ];
        let result = compute_consensus_fingerprint(&fingerprints);
        assert_eq!(result, vec![2.5, 5.0]);
    }

    #[test]
    fn test_compute_std_dev_fingerprint_empty() {
        let result = compute_std_dev_fingerprint(&[], &[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_compute_std_dev_fingerprint_single() {
        let fingerprints = vec![vec![1.0, 2.0, 3.0]];
        let consensus = vec![1.0, 2.0, 3.0];
        let result = compute_std_dev_fingerprint(&fingerprints, &consensus);
        assert_eq!(result, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_compute_std_dev_fingerprint_multiple() {
        let fingerprints = vec![vec![1.0, 0.0], vec![3.0, 0.0]];
        let consensus = vec![2.0, 0.0];
        let result = compute_std_dev_fingerprint(&fingerprints, &consensus);
        assert_eq!(result[0], 1.0);
        assert_eq!(result[1], 0.0);
    }

    #[test]
    fn test_extract_fingerprint_from_signal_too_small() {
        let signal: Vec<i16> = vec![100, 200, 300];
        let read_id = Uuid::new_v4();
        let result = extract_fingerprint_from_signal(
            &signal,
            0,
            3,
            10,
            5,
            NormMethod::ZScore,
            read_id,
            None,
            None,
            false,
        );
        assert!(result.is_none());
    }

    #[test]
    fn test_extract_fingerprint_from_signal_valid() {
        let signal: Vec<i16> = (0..1000).map(|i| (i as i16) % 1000).collect();
        let read_id = Uuid::new_v4();
        let result = extract_fingerprint_from_signal(
            &signal,
            0,
            500,
            10,
            5,
            NormMethod::None,
            read_id,
            None,
            None,
            false,
        );
        assert!(result.is_some());
        let fp = result.unwrap();
        assert_eq!(fp.read_id, read_id);
        assert!(!fp.values.is_empty());
        assert!(fp.dwell_times.is_none());
    }

    #[test]
    fn test_extract_fingerprint_emits_dwell_aligned() {
        let signal: Vec<i16> = (0..1000).map(|i| (i as i16) % 1000).collect();
        let read_id = Uuid::new_v4();
        let result = extract_fingerprint_from_signal(
            &signal,
            0,
            500,
            10,
            5,
            NormMethod::None,
            read_id,
            None,
            None,
            true,
        );
        let fp = result.expect("valid signal should produce a fingerprint");
        let dwells = fp.dwell_times.as_ref().expect("emit_dwell=true => Some");
        assert_eq!(
            dwells.len(),
            fp.values.len(),
            "dwell column count must equal means column count",
        );
        // normalize_dwell_times is log1p + z-score: finite values, not all zero
        // unless there's exactly one segment (in which case the normalization
        // collapses to zero — a degenerate but legal case).
        assert!(dwells.iter().all(|v| v.is_finite()));
    }
}

#[cfg(test)]
mod keep_last_width_tests {
    use super::*;
    use escapepod_signal::dtw::NormMethod;

    /// `keep_last` is a fixed width. A read whose adapter yields fewer segments
    /// than requested must be rejected outright, never emitted as a short row —
    /// the DTW/SVM path cannot detect the truncation and would score it as a
    /// confident (wrong) barcode.
    #[test]
    fn short_adapter_is_rejected_rather_than_truncated() {
        let read_id = uuid::Uuid::nil();
        // A very short, nearly flat adapter window cannot yield 25 segments.
        let signal: Vec<i16> = (0..400).map(|i| 500 + (i % 3) as i16).collect();

        let fp = extract_fingerprint_from_signal(
            &signal,
            0,
            signal.len(),
            111,
            12,
            NormMethod::ZScore,
            read_id,
            Some(6),
            Some(25),
            false,
        );

        if let Some(fp) = fp {
            assert_eq!(
                fp.values.len(),
                25,
                "a returned fingerprint must have exactly keep_last values"
            );
        }
    }

    /// Whenever a fingerprint is produced with `keep_last = Some(n)`, it has
    /// exactly `n` values — across a range of adapter lengths.
    #[test]
    fn keep_last_width_is_exact_when_present() {
        let read_id = uuid::Uuid::nil();
        for len in [600usize, 1200, 2400, 4800] {
            let signal: Vec<i16> = (0..len)
                .map(|i| {
                    let x = i as f32 * 0.05;
                    (500.0 + 80.0 * x.sin() + 25.0 * (x * 3.1).cos()) as i16
                })
                .collect();
            let fp = extract_fingerprint_from_signal(
                &signal,
                0,
                len,
                111,
                12,
                NormMethod::ZScore,
                read_id,
                Some(6),
                Some(25),
                false,
            );
            if let Some(fp) = fp {
                assert_eq!(fp.values.len(), 25, "len={len}: ragged fingerprint emitted");
            }
        }
    }

    /// A window wider than [`MAX_FINGERPRINT_WINDOW`] is fingerprinted from its
    /// last `MAX_FINGERPRINT_WINDOW` samples — the same answer as a detector
    /// that had reported the start there — and a window inside the bound is
    /// untouched by the clamp.
    #[test]
    fn a_wide_window_is_clamped_to_its_tail() {
        let read_id = uuid::Uuid::nil();
        let len = MAX_FINGERPRINT_WINDOW + 20_000;
        let signal: Vec<i16> = (0..len)
            .map(|i| {
                let x = i as f32 * 0.05;
                (500.0 + 80.0 * x.sin() + 25.0 * (x * 3.1).cos()) as i16
            })
            .collect();
        let fp = |start: usize, end: usize| {
            extract_fingerprint_from_signal(
                &signal,
                start,
                end,
                111,
                12,
                NormMethod::ZScore,
                read_id,
                Some(6),
                Some(25),
                false,
            )
            .map(|f| f.values)
        };
        let wide = fp(0, len).expect("a long window still fingerprints");
        // The clamp is applied after the boundary padding, so the equivalent
        // explicit start is `len + padding - MAX`.
        let explicit = fp(len + BOUNDARY_PADDING_SAMPLES - MAX_FINGERPRINT_WINDOW, len)
            .expect("the explicit tail fingerprints");
        assert_eq!(wide, explicit, "a wide window must be its own tail");

        // Inside the bound the clamp does nothing: shifting the start by one
        // sample changes the segmentation, which it could not if the clamp had
        // already moved it.
        let narrow_end = MAX_FINGERPRINT_WINDOW - 5_000;
        let a = fp(0, narrow_end).expect("narrow");
        let b = fp(1_000, narrow_end).expect("narrow, later start");
        assert_ne!(
            a, b,
            "a window inside the bound must see its declared start"
        );
    }
}
