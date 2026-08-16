//! Summary statistics over arbitrary spans of a read's signal.
//!
//! Given a read and a set of `[start, end)` signal intervals -- typically one
//! per base of interest, from a move table or a resquiggle -- this reduces each
//! interval to how long the pore held it and what level it sat at. That is the
//! input any per-base signal model wants, whatever the model is about:
//! modification calling, adduct detection, basecall QC.
//!
//! The spans are supplied by the caller rather than derived here, because how a
//! base maps to signal is the caller's business (aligner-derived, counted along
//! the query, refined) and the reduction is the same either way.

use crate::stats::median_and_mad_with_scratch;

/// How to standardise levels before summarising them.
#[derive(Clone, Copy, Debug)]
pub enum Normalization {
    /// Use the signal as given.
    None,
    /// Per-read median/MAD, i.e. `(x - median) / (1.4826 * mad)`.
    ///
    /// `mad_floor` guards a near-flat read: at or below it the scale falls back
    /// to `1.0` rather than exploding. The right value is a property of the
    /// caller's units, not of this function, so there is no default -- callers
    /// that disagree here produce different features from the same signal.
    MedianMad { mad_floor: f32 },
}

/// Output buffers, one entry per span. Separate arrays rather than an
/// interleaved record so callers can lay the numbers out however their model's
/// feature vector is ordered without this module knowing about it.
pub struct SpanStatsOut<'a> {
    /// Number of samples in the span.
    pub dwell: &'a mut [f32],
    /// Mean level over the span, after normalisation.
    pub mean: &'a mut [f32],
    /// Population standard deviation (`ddof = 0`) over the span.
    pub sd: &'a mut [f32],
}

/// Scratch buffers reused across reads so a hot loop allocates at its
/// high-water mark rather than once per read.
#[derive(Default)]
pub struct SpanScratch {
    median: Vec<f32>,
    cumsum: Vec<f64>,
    cumsum_sq: Vec<f64>,
}

/// Reduce each span to `(dwell, mean, sd)`.
///
/// A span is skipped -- leaving `NaN` in all three outputs -- when it is
/// negative, empty, or runs past the end of the signal. `NaN` is deliberate:
/// an unresolved base has no observation, and substituting a neighbour's value
/// would make it indistinguishable from a real one.
///
/// Cost is one pass over the spanned region plus O(1) per span, not one pass
/// per span, so many short spans are as cheap as a few long ones.
pub fn span_stats(
    signal: &[f32],
    spans: &[[i64; 2]],
    norm: Normalization,
    scratch: &mut SpanScratch,
    out: SpanStatsOut<'_>,
) {
    let SpanStatsOut { dwell, mean, sd } = out;
    debug_assert_eq!(dwell.len(), spans.len());
    debug_assert_eq!(mean.len(), spans.len());
    debug_assert_eq!(sd.len(), spans.len());
    dwell.fill(f32::NAN);
    mean.fill(f32::NAN);
    sd.fill(f32::NAN);
    if signal.is_empty() || spans.is_empty() {
        return;
    }

    let (centre, scale) = match norm {
        Normalization::None => (0.0f64, 1.0f64),
        Normalization::MedianMad { mad_floor } => {
            let (med, mad) = median_and_mad_with_scratch(signal, &mut scratch.median);
            let s = if mad > mad_floor { 1.4826 * mad } else { 1.0 };
            (med as f64, s as f64)
        }
    };

    // Prefix sums over only the region the spans actually cover: bases of
    // interest usually sit in a small neighbourhood of a landmark while the
    // read is tens of thousands of samples long.
    let n_sig = signal.len() as i64;
    let mut lo = i64::MAX;
    let mut hi = i64::MIN;
    for s in spans {
        if s[0] >= 0 && s[1] > s[0] && s[1] <= n_sig {
            lo = lo.min(s[0]);
            hi = hi.max(s[1]);
        }
    }
    if lo > hi {
        return;
    }
    let (lo_u, hi_u) = (lo as usize, hi as usize);

    // f64 accumulators: the running totals grow across the region while each
    // span's difference is small next to them.
    scratch.cumsum.clear();
    scratch.cumsum_sq.clear();
    scratch.cumsum.reserve(hi_u - lo_u + 1);
    scratch.cumsum_sq.reserve(hi_u - lo_u + 1);
    scratch.cumsum.push(0.0);
    scratch.cumsum_sq.push(0.0);
    let (mut acc, mut acc_sq) = (0.0f64, 0.0f64);
    for &v in &signal[lo_u..hi_u] {
        let z = (v as f64 - centre) / scale;
        acc += z;
        acc_sq += z * z;
        scratch.cumsum.push(acc);
        scratch.cumsum_sq.push(acc_sq);
    }

    for (i, s) in spans.iter().enumerate() {
        let (a, b) = (s[0], s[1]);
        if a < 0 || b <= a || b > n_sig {
            continue;
        }
        let (ai, bi) = ((a - lo) as usize, (b - lo) as usize);
        let n = (b - a) as f64;
        let m = (scratch.cumsum[bi] - scratch.cumsum[ai]) / n;
        let var = (scratch.cumsum_sq[bi] - scratch.cumsum_sq[ai]) / n - m * m;
        dwell[i] = n as f32;
        mean[i] = m as f32;
        // Clamp the cancellation floor: a constant span can land at -1e-17.
        sd[i] = if var > 0.0 { var.sqrt() as f32 } else { 0.0 };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(
        signal: &[f32],
        spans: &[[i64; 2]],
        norm: Normalization,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let n = spans.len();
        let (mut d, mut m, mut s) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        let mut scratch = SpanScratch::default();
        span_stats(
            signal,
            spans,
            norm,
            &mut scratch,
            SpanStatsOut {
                dwell: &mut d,
                mean: &mut m,
                sd: &mut s,
            },
        );
        (d, m, s)
    }

    /// The obvious per-span loop, as the oracle for the prefix-sum version.
    fn reference(
        signal: &[f32],
        spans: &[[i64; 2]],
        norm: Normalization,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let (centre, scale) = match norm {
            Normalization::None => (0.0f64, 1.0f64),
            Normalization::MedianMad { mad_floor } => {
                let mut buf = Vec::new();
                let (med, mad) = median_and_mad_with_scratch(signal, &mut buf);
                (
                    med as f64,
                    if mad > mad_floor { 1.4826 * mad } else { 1.0 } as f64,
                )
            }
        };
        let n = spans.len();
        let (mut d, mut m, mut s) = (vec![f32::NAN; n], vec![f32::NAN; n], vec![f32::NAN; n]);
        for (i, sp) in spans.iter().enumerate() {
            let (a, b) = (sp[0], sp[1]);
            if a < 0 || b <= a || b > signal.len() as i64 {
                continue;
            }
            let seg: Vec<f64> = signal[a as usize..b as usize]
                .iter()
                .map(|&v| (v as f64 - centre) / scale)
                .collect();
            let cnt = seg.len() as f64;
            let mu = seg.iter().sum::<f64>() / cnt;
            let var = seg.iter().map(|v| (v - mu) * (v - mu)).sum::<f64>() / cnt;
            d[i] = cnt as f32;
            m[i] = mu as f32;
            s[i] = var.sqrt() as f32;
        }
        (d, m, s)
    }

    fn pseudo_signal(n: usize) -> Vec<f32> {
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        (0..n)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                80.0 + 30.0 * ((state >> 11) as f64 / (1u64 << 53) as f64) as f32
            })
            .collect()
    }

    fn mixed_spans() -> Vec<[i64; 2]> {
        let mut spans = Vec::new();
        let mut pos = 500i64;
        for i in 0..25i64 {
            if i % 7 == 3 {
                spans.push([-1, -1]); // unresolved
                continue;
            }
            let d = 5 + (i * 13) % 55;
            spans.push([pos, pos + d]);
            pos += d;
        }
        spans
    }

    #[test]
    fn matches_the_per_span_loop() {
        let sig = pseudo_signal(20_000);
        let spans = mixed_spans();
        let norm = Normalization::MedianMad { mad_floor: 1e-3 };
        let (wd, wm, ws) = reference(&sig, &spans, norm);
        let (gd, gm, gs) = run(&sig, &spans, norm);
        for (name, want, got) in [("dwell", wd, gd), ("mean", wm, gm), ("sd", ws, gs)] {
            for (i, (w, g)) in want.iter().zip(got.iter()).enumerate() {
                assert_eq!(w.is_nan(), g.is_nan(), "{name}[{i}] NaN pattern");
                if !w.is_nan() {
                    assert!(
                        (w - g).abs() <= 1e-5 * w.abs().max(1.0),
                        "{name}[{i}]: {w} vs {g}"
                    );
                }
            }
        }
    }

    #[test]
    fn invalid_spans_are_nan() {
        let sig = vec![1.0f32; 100];
        let spans = [[-1, -1], [50, 40], [90, 200], [10, 20]];
        let (d, m, s) = run(&sig, &spans, Normalization::None);
        for i in 0..3 {
            assert!(d[i].is_nan() && m[i].is_nan() && s[i].is_nan(), "span {i}");
        }
        assert_eq!(d[3], 10.0);
        assert_eq!(m[3], 1.0);
        assert_eq!(s[3], 0.0, "constant span, and not a negative-variance NaN");
    }

    #[test]
    fn scratch_reuse_is_bit_identical() {
        let sig = pseudo_signal(5_000);
        let spans = mixed_spans();
        let norm = Normalization::MedianMad { mad_floor: 1e-3 };
        let mut scratch = SpanScratch::default();
        let n = spans.len();
        let (mut d1, mut m1, mut s1) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        let (mut d2, mut m2, mut s2) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        span_stats(
            &sig,
            &spans,
            norm,
            &mut scratch,
            SpanStatsOut {
                dwell: &mut d1,
                mean: &mut m1,
                sd: &mut s1,
            },
        );
        span_stats(
            &sig,
            &spans,
            norm,
            &mut scratch,
            SpanStatsOut {
                dwell: &mut d2,
                mean: &mut m2,
                sd: &mut s2,
            },
        );
        // to_bits, not ==: NaN never equals itself, and these are mostly NaN.
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&d1), bits(&d2));
        assert_eq!(bits(&m1), bits(&m2));
        assert_eq!(bits(&s1), bits(&s2));
    }

    #[test]
    fn mad_floor_prevents_blowup_on_a_flat_read() {
        let sig = vec![42.0f32; 1000];
        let (_, m, s) = run(
            &sig,
            &[[10, 50]],
            Normalization::MedianMad { mad_floor: 1e-3 },
        );
        assert_eq!(
            m[0], 0.0,
            "flat read centres to zero, scale falls back to 1"
        );
        assert_eq!(s[0], 0.0);
    }

    #[test]
    fn empty_inputs_are_handled() {
        let (d, _, _) = run(&[], &[[0, 10]], Normalization::None);
        assert!(d[0].is_nan());
        let (d, _, _) = run(&pseudo_signal(100), &[], Normalization::None);
        assert!(d.is_empty());
    }
}
