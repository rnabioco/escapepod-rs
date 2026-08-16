// SPDX-License-Identifier: MIT

//! Per-base junction features: dwell, mean, std, and the k-mer residual.
//!
//! Port of `escapepod_models.charging._base_features` and
//! `_Refiner.expected_at`, which computed the training corpus — this module
//! must produce the same numbers, so the NumPy arithmetic is mirrored where
//! it is deterministic:
//!
//! - the per-read median/MAD gauge is computed in `f32` exactly as NumPy
//!   does (median of `f32` data, `f32` element-wise deviations, the
//!   even-length midpoint averaged in `f32`), including the `1e-3` MAD
//!   floor and the `1.4826` scale applied in `f64` then rounded to `f32`
//!   for the per-sample division;
//! - expected k-mer levels round-trip through `f32` (leech's Python
//!   `extract_levels` stores `float32`) before the `f64` z-score.
//!
//! One knowing deviation: segment mean/std and the z-score moments
//! accumulate in `f64` sequentially, where NumPy uses pairwise (and
//! SIMD-order-dependent) `f32`/`f64` summation. The difference is bounded
//! by reduction rounding (~1e-6 absolute on `f32`-scale segments), orders
//! below the feature scale; the end-to-end parity test measures what
//! survives into the model's output.

use crate::anchor::JunctionCoords;
use std::collections::HashMap;

/// Canonical per-base statistics, in feature-vector order (stats inner,
/// offsets outer). `resid` — the observed level minus the base's expected
/// k-mer level — is the model's decisive feature.
pub const FEAT_STATS: [&str; 4] = ["dwell", "mean", "std", "resid"];

/// Minimum valid expected levels for a meaningful per-read z-score.
const MIN_VALID_LEVELS: usize = 20;

/// Median of `f32` data, NumPy semantics: order statistic for odd length,
/// `f32` midpoint average for even length. `data` is scratch (reordered).
fn median_f32(data: &mut [f32]) -> f32 {
    let n = data.len();
    debug_assert!(n > 0);
    let mid = n / 2;
    let (_, m, _) = data.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
    let hi = *m;
    if n % 2 == 1 {
        hi
    } else {
        let lo = data[..mid]
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        (lo + hi) / 2.0
    }
}

/// The per-read robust gauge: `(median, scale)` where `scale` is
/// `1.4826 * MAD` floored at `1e-3` MAD (then 1.0), matching
/// `_base_features`. `scale` is returned in `f64` (the Python keeps it as a
/// float) — the per-sample division rounds it to `f32`.
fn read_gauge(signal: &[f32]) -> (f32, f64) {
    let mut scratch = signal.to_vec();
    let med = median_f32(&mut scratch);
    for v in &mut scratch {
        *v = (*v - med).abs();
    }
    let mad = median_f32(&mut scratch);
    let scale = if mad as f64 > 1e-3 {
        1.4826 * mad as f64
    } else {
        1.0
    };
    (med, scale)
}

/// Per-base `(dwell, mean, std, resid)` over the recipe's offsets.
///
/// Returns the full canonical grid, `offsets.len() * 4` values in
/// offsets-outer / [`FEAT_STATS`]-inner order, `NaN` where a base did not
/// align (the GBM handles `NaN` natively). `expected` is the per-offset
/// z-scored expectation from [`expected_levels_z`] (`NaN` where absent).
pub fn junction_features(
    signal_pa: &[f32],
    coords: &JunctionCoords,
    expected: Option<&[f32]>,
) -> Vec<f32> {
    let n_off = coords.feat_spans.len();
    let mut out = vec![f32::NAN; n_off * FEAT_STATS.len()];
    if signal_pa.is_empty() {
        return out;
    }
    let (med, scale) = read_gauge(signal_pa);
    let scale32 = scale as f32;

    for (i, &(a, b)) in coords.feat_spans.iter().enumerate() {
        if a < 0 || b <= a || b as usize > signal_pa.len() {
            continue;
        }
        let (a, b) = (a as usize, b as usize);
        let n = (b - a) as f64;
        let mut sum = 0.0f64;
        for &x in &signal_pa[a..b] {
            sum += ((x - med) / scale32) as f64;
        }
        let mean = sum / n;
        let mut ss = 0.0f64;
        for &x in &signal_pa[a..b] {
            let d = ((x - med) / scale32) as f64 - mean;
            ss += d * d;
        }
        let std = (ss / n).sqrt();

        let j = i * FEAT_STATS.len();
        out[j] = (b - a) as f32;
        out[j + 1] = mean as f32;
        out[j + 2] = std as f32;
        if let Some(exp) = expected
            && exp[i].is_finite()
        {
            // f32 subtraction, as NumPy performs it on the two f32 scalars.
            out[j + 3] = mean as f32 - exp[i];
        }
    }
    out
}

/// Expected k-mer level per feature base, z-scored across this read.
///
/// Port of `_Refiner.expected_at`: expected levels come from the basecalled
/// *query* sequence via [`escapepod_signal::resquiggle::extract_levels`]
/// (levels round-trip through `f32`, matching leech's Python), and the
/// expectation is standardised over the read's valid (finite, non-zero)
/// levels — the table is in its own units while observed levels are
/// per-read MAD-normalised, so only the z-scored difference is meaningful.
///
/// Returns one value per entry of `qf` (`NaN` where unaligned/invalid), or
/// all-`NaN` when fewer than 20 valid levels exist or the spread is zero.
pub fn expected_levels_z(
    seq: &[u8],
    kmer_to_level: &HashMap<String, f64>,
    kmer_len: usize,
    center_idx: usize,
    qf: &[i64],
    nb: usize,
) -> Vec<f32> {
    let mut out = vec![f32::NAN; qf.len()];
    if seq.is_empty() {
        return out;
    }
    let seq_str = match std::str::from_utf8(seq) {
        Ok(s) => s,
        Err(_) => return out,
    };
    let lv64 = escapepod_signal::resquiggle::extract_levels(
        seq_str,
        kmer_to_level,
        kmer_len,
        Some(center_idx),
    );
    // leech's Python extract_levels stores float32; round-trip to match.
    let lv: Vec<f64> = lv64.iter().map(|&v| (v as f32) as f64).collect();

    let valid: Vec<f64> = lv
        .iter()
        .copied()
        .filter(|v| v.is_finite() && *v != 0.0)
        .collect();
    if valid.len() < MIN_VALID_LEVELS {
        return out;
    }
    let n = valid.len() as f64;
    let mu = valid.iter().sum::<f64>() / n;
    let sd = (valid.iter().map(|v| (v - mu) * (v - mu)).sum::<f64>() / n).sqrt();
    if sd <= 0.0 {
        return out;
    }

    let limit = nb.min(lv.len());
    for (o, &qp) in out.iter_mut().zip(qf) {
        if qp >= 0 && (qp as usize) < limit && lv[qp as usize].is_finite() && lv[qp as usize] != 0.0
        {
            *o = ((lv[qp as usize] - mu) / sd) as f32;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_median_f32_numpy_semantics() {
        assert_eq!(median_f32(&mut [3.0, 1.0, 2.0]), 2.0);
        // Even length: f32 midpoint average.
        assert_eq!(median_f32(&mut [4.0, 1.0, 3.0, 2.0]), 2.5);
        assert_eq!(median_f32(&mut [1.0]), 1.0);
    }

    #[test]
    fn test_read_gauge_floor() {
        // Constant signal → MAD 0 → scale floors to 1.0.
        let (med, scale) = read_gauge(&[5.0; 100]);
        assert_eq!(med, 5.0);
        assert_eq!(scale, 1.0);
    }

    #[test]
    fn test_junction_features_basic() {
        // Signal: base 0 sits at level 10, base 1 at level 20, background 0.
        let mut sig = vec![0.0f32; 100];
        for v in &mut sig[10..20] {
            *v = 10.0;
        }
        for v in &mut sig[20..26] {
            *v = 20.0;
        }
        let coords = JunctionCoords {
            feat_spans: vec![(10, 20), (20, 26), (-1, -1), (90, 120)],
            common_start_sig: 0,
            junction_sig: 10,
            mask_source: crate::anchor::MaskSource::Exact,
        };
        let f = junction_features(&sig, &coords, None);
        assert_eq!(f.len(), 16);
        assert_eq!(f[0], 10.0); // dwell of span 0
        assert!(f[1] > 0.0); // mean above the (median 0) background
        assert!(f[2].abs() < 1e-6); // constant segment → std 0
        assert!(f[3].is_nan()); // no expected levels → resid NaN
        assert_eq!(f[4], 6.0); // dwell of span 1
        assert!(f[8].is_nan()); // unaligned base
        assert!(f[12].is_nan()); // span past signal end
    }

    #[test]
    fn test_expected_levels_z() {
        // k=1 table: every base has a level, so a 30-base read has 30 valid
        // levels; z-scores are exactly computable.
        let mut table = HashMap::new();
        table.insert("A".to_string(), 1.0);
        table.insert("C".to_string(), 2.0);
        table.insert("G".to_string(), 3.0);
        table.insert("T".to_string(), 4.0);
        let seq: Vec<u8> = b"ACGT".repeat(8); // 32 bases
        let qf = vec![0i64, 1, 2, 3, -1, 100];
        let z = expected_levels_z(&seq, &table, 1, 0, &qf, 32);
        // mu = 2.5, sd = sqrt(1.25)
        let sd = 1.25f64.sqrt();
        assert!((z[0] as f64 - (1.0 - 2.5) / sd).abs() < 1e-6);
        assert!((z[3] as f64 - (4.0 - 2.5) / sd).abs() < 1e-6);
        assert!(z[4].is_nan()); // unaligned
        assert!(z[5].is_nan()); // out of range
    }

    #[test]
    fn test_expected_levels_z_too_few_valid() {
        let mut table = HashMap::new();
        table.insert("A".to_string(), 1.0);
        let seq = b"AAAAA".to_vec(); // 5 valid levels < 20
        let z = expected_levels_z(&seq, &table, 1, 0, &[0, 1], 5);
        assert!(z.iter().all(|v| v.is_nan()));
    }
}
