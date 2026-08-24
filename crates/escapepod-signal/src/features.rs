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
//!
//! Everything a consumer might reasonably disagree about -- the sentinel for an
//! unresolved span, what to do with a span that runs off the end, which median
//! -- is a named field on [`SpanConfig`] rather than a hard-coded choice. The
//! alternative is what actually happened downstream: a consumer that needed a
//! different fill, a different out-of-range rule and a median re-implemented
//! the whole reduction, then grew a *second* copy of it that disagreed with the
//! first on exactly one of those rules, so the same read got different features
//! depending on which code path reached it (rnabioco/leech#200). Every default
//! here is this module's original behaviour, so selecting nothing changes
//! nothing.

use std::cmp::Ordering;

use crate::stats::{median_and_mad_with_scratch, median_via_select};

/// How to standardise levels before summarising them.
#[derive(Clone, Copy, Debug, Default)]
pub enum Normalization {
    /// Use the signal as given.
    #[default]
    None,
    /// Per-read median/MAD, i.e. `(x - median) / (1.4826 * mad)`.
    ///
    /// `mad_floor` guards a near-flat read: at or below it the scale falls back
    /// to `1.0` rather than exploding. The right value is a property of the
    /// caller's units, not of this function, so there is no default -- callers
    /// that disagree here produce different features from the same signal.
    MedianMad { mad_floor: f32 },
}

/// What to write for a span that does not resolve.
///
/// [`SpanFill::Nan`] is the default and stays the honest answer: an unresolved
/// base has no observation, and substituting a neighbour's value -- or a zero,
/// which is a perfectly ordinary normalised level -- makes it indistinguishable
/// from a real one.
///
/// The alternatives exist because that argument does not survive contact with a
/// neural network: a single `NaN` poisons a whole forward pass, so a consumer
/// feeding these arrays to a model must either rewrite every output afterwards
/// or carry its own copy of this reduction. Naming the sentinel at the call
/// site is cheaper than either, and it keeps "what did this run write for a
/// missing base" answerable from the config rather than from the caller's
/// post-processing.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum SpanFill {
    /// `f32::NAN`, so an unresolved base cannot be mistaken for a real one.
    #[default]
    Nan,
    /// `0.0`. Identical to `Value(0.0)`, spelled out because it is the common
    /// request -- it is what a model expects after its own imputation.
    Zero,
    /// An arbitrary sentinel.
    Value(f32),
}

impl SpanFill {
    /// The value written into every output array for an unresolved span.
    #[inline]
    fn value(self) -> f32 {
        match self {
            SpanFill::Nan => f32::NAN,
            SpanFill::Zero => 0.0,
            SpanFill::Value(v) => v,
        }
    }
}

/// What to do with a span that falls partly outside the signal.
///
/// [`SpanBounds::Skip`] is the default and the original behaviour: a span that
/// starts before the signal or ends past it does not resolve at all, and takes
/// the fill. It treats an out-of-range coordinate as evidence that the map is
/// wrong.
///
/// [`SpanBounds::Clamp`] instead intersects the span with `[0, signal.len())`
/// and summarises what is left, calling the span unresolved only when the
/// intersection is empty. That is the right answer when the coordinates are
/// *legitimately* allowed to run off the end -- a reference-anchored map whose
/// entries can go negative once the aligned region is cropped, say, where the
/// truncated span still carries real signal and skipping it silently discards
/// data. It is the wrong answer when a negative start means the map is broken,
/// which is why it is not the default.
///
/// Under `Clamp`, `dwell` is the **clamped** length -- the number of samples
/// actually summarised -- not the requested span width. Every other output is
/// computed from exactly those samples, so reporting the requested width would
/// pair a sample count with a mean that was not taken over that many samples,
/// and a model reading both would be reading a contradiction.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SpanBounds {
    /// Skip a span that is negative, empty, or runs past the end.
    #[default]
    Skip,
    /// Clamp into `[0, signal.len())` and summarise the surviving portion.
    Clamp,
}

/// Which median the optional `median` output is.
///
/// Both are offered because the choice is invisible at the call site and the
/// two are not interchangeable, so a consumer that needs one of them has to be
/// able to *name* it rather than discover it from a diff.
///
/// - [`MedianConvention::SelectTotalCmp`] is the crate's own convention, i.e.
///   [`crate::stats::median_via_select`]: `select_nth_unstable` with
///   `f32::total_cmp`. It is what every other median in escapepod-signal
///   already is, so it is the default.
/// - [`MedianConvention::SortPartialCmp`] is a full sort with `partial_cmp`
///   (`NaN` ordered to the high end, as numpy sorts it) plus numpy's own
///   `NaN` check, chosen to reproduce `numpy.median` over a `float32` array
///   exactly -- the parity a consumer that cross-checks against a Python
///   reference implementation needs.
///
/// **Even-length rule, both conventions: the two middle order statistics are
/// averaged** (`(lo + hi) / 2.0`, evaluated in `f32`); an odd length returns
/// the middle one. That is `numpy.median`'s rule as well as this crate's, so
/// neither convention picks one of the two middles and discards the other.
///
/// **Where they actually differ.** Measured rather than assumed (see this
/// module's tests): over a span of finite values the two are *bit-identical*,
/// including on the even-length ulp-separated `f32` spans where a disagreement
/// was expected. They end up averaging the same two order statistics --
/// `total_cmp` and `partial_cmp` induce the same order on non-`NaN` values --
/// and `numpy.median`'s `float32` mean of two elements is bit-for-bit
/// `(a + b) / 2.0` in `f32`, verified against numpy 2.5.1 over 400k random
/// pairs.
///
/// They diverge on a span containing `NaN`: `SelectTotalCmp` sorts `NaN`
/// deterministically to the high end and returns a finite median from the
/// values below it, while `SortPartialCmp` propagates `NaN` the way
/// `numpy.median` does. That case is not exotic -- a caller that pads a window
/// with `NaN` hits it on every padded base -- and the propagating answer is the
/// one consistent with `mean`, which is already `NaN` for such a span.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MedianConvention {
    /// `select_nth_unstable` with `f32::total_cmp`, matching
    /// [`crate::stats::median_via_select`].
    #[default]
    SelectTotalCmp,
    /// Full sort with `partial_cmp`, matching `numpy.median` including its
    /// `NaN` propagation.
    SortPartialCmp,
}

/// Every knob [`span_stats`] has, in one place, so a call site names what it
/// asked for instead of inheriting it.
///
/// [`SpanConfig::default()`] is this module's original behaviour exactly: no
/// normalisation, `NaN` fill, out-of-range spans skipped, and the crate's own
/// median convention.
#[derive(Clone, Copy, Debug, Default)]
pub struct SpanConfig {
    /// How to standardise levels before summarising them.
    pub norm: Normalization,
    /// What to write for a span that does not resolve.
    pub fill: SpanFill,
    /// What to do with a span that falls partly outside the signal.
    pub bounds: SpanBounds,
    /// Which median the optional `median` output is. Ignored when no `median`
    /// buffer is supplied.
    pub median: MedianConvention,
}

impl SpanConfig {
    /// A config with `norm` set and every other knob at its default.
    pub fn new(norm: Normalization) -> Self {
        Self {
            norm,
            ..Self::default()
        }
    }

    /// Set the unresolved-span sentinel.
    pub fn with_fill(mut self, fill: SpanFill) -> Self {
        self.fill = fill;
        self
    }

    /// Set the out-of-range policy.
    pub fn with_bounds(mut self, bounds: SpanBounds) -> Self {
        self.bounds = bounds;
        self
    }

    /// Set the median convention.
    pub fn with_median(mut self, median: MedianConvention) -> Self {
        self.median = median;
        self
    }
}

/// Output buffers, one entry per span. Separate arrays rather than an
/// interleaved record so callers can lay the numbers out however their model's
/// feature vector is ordered without this module knowing about it.
///
/// `median` and `range` are optional because neither can come from the prefix
/// sums: each needs its own pass over the span, and the median needs a select
/// or a sort on top of that. A consumer that only wants `dwell`/`mean`/`sd`
/// should not pay for them, so they are computed only when a buffer is
/// supplied. Build with [`SpanStatsOut::new`] plus
/// [`with_median`](SpanStatsOut::with_median) /
/// [`with_range`](SpanStatsOut::with_range) rather than a struct literal, so a
/// future output does not churn every call site.
pub struct SpanStatsOut<'a> {
    /// Number of samples in the span.
    pub dwell: &'a mut [f32],
    /// Mean level over the span, after normalisation.
    pub mean: &'a mut [f32],
    /// Population standard deviation (`ddof = 0`) over the span.
    pub sd: &'a mut [f32],
    /// Median level over the span, after normalisation. See
    /// [`MedianConvention`] for *which* median it is.
    pub median: Option<&'a mut [f32]>,
    /// `max - min` over the span, after normalisation. `0.0` for a constant or
    /// single-sample span; `NaN` if the span contains one, matching `np.ptp`
    /// and the `mean` for the same span.
    pub range: Option<&'a mut [f32]>,
}

impl<'a> SpanStatsOut<'a> {
    /// The three always-computed outputs.
    pub fn new(dwell: &'a mut [f32], mean: &'a mut [f32], sd: &'a mut [f32]) -> Self {
        Self {
            dwell,
            mean,
            sd,
            median: None,
            range: None,
        }
    }

    /// Also compute the per-span median into `median`.
    pub fn with_median(mut self, median: &'a mut [f32]) -> Self {
        self.median = Some(median);
        self
    }

    /// Also compute the per-span range into `range`.
    pub fn with_range(mut self, range: &'a mut [f32]) -> Self {
        self.range = Some(range);
        self
    }
}

/// Scratch buffers reused across reads so a hot loop allocates at its
/// high-water mark rather than once per read.
#[derive(Default)]
pub struct SpanScratch {
    median: Vec<f32>,
    cumsum: Vec<f64>,
    cumsum_sq: Vec<f64>,
    span: Vec<f32>,
}

/// The `[start, end)` a span actually contributes, or `None` if it does not
/// resolve. The single definition of "resolved", so the pass that sizes the
/// prefix sums and the pass that reads them cannot drift apart.
#[inline]
fn resolve_span(span: &[i64; 2], n_sig: i64, bounds: SpanBounds) -> Option<(i64, i64)> {
    let (a, b) = (span[0], span[1]);
    match bounds {
        SpanBounds::Skip => (a >= 0 && b > a && b <= n_sig).then_some((a, b)),
        SpanBounds::Clamp => {
            let (a, b) = (a.max(0), b.min(n_sig));
            (b > a).then_some((a, b))
        }
    }
}

/// `max - min`, propagating `NaN` like `np.ptp`.
#[inline]
fn span_range(vals: &[f32]) -> f32 {
    let (mut mn, mut mx) = (f32::INFINITY, f32::NEG_INFINITY);
    for &v in vals {
        if v.is_nan() {
            return f32::NAN;
        }
        if v < mn {
            mn = v;
        }
        if v > mx {
            mx = v;
        }
    }
    mx - mn
}

/// `numpy.median` over a `float32` array: sort with `NaN` at the high end,
/// return `NaN` if any is present, otherwise average the two middles (even
/// length) or take the middle one (odd). Reorders `vals`.
fn median_numpy(vals: &mut [f32]) -> f32 {
    let n = vals.len();
    if n == 0 {
        return f32::NAN;
    }
    // partial_cmp with NaN forced to the high end, which is numpy's sort order.
    // A total preorder, so sort_unstable_by is well defined.
    vals.sort_unstable_by(|a, b| {
        a.partial_cmp(b).unwrap_or(match (a.is_nan(), b.is_nan()) {
            (true, true) => Ordering::Equal,
            (true, false) => Ordering::Greater,
            _ => Ordering::Less,
        })
    });
    // numpy's `_median_nancheck`: a NaN anywhere makes the median NaN, and
    // after that sort "anywhere" is the last element.
    if vals[n - 1].is_nan() {
        return f32::NAN;
    }
    let mid = n / 2;
    if n.is_multiple_of(2) {
        (vals[mid - 1] + vals[mid]) / 2.0
    } else {
        vals[mid]
    }
}

/// Reduce each span to `(dwell, mean, sd)`, optionally also `median` and
/// `range`.
///
/// A span that does not resolve -- negative, empty, or past the end under the
/// default [`SpanBounds::Skip`] -- gets `cfg.fill` in *every* requested output.
/// See [`SpanFill`] for why `NaN` is the default and why it is selectable, and
/// [`SpanBounds`] for the out-of-range policy.
///
/// Cost is one pass over the spanned region plus O(1) per span, not one pass
/// per span, so many short spans are as cheap as a few long ones. Requesting
/// `median` or `range` adds one gather per resolved span (and, for the median,
/// a select or a sort over it); neither perturbs `dwell`/`mean`/`sd`, which
/// come from the prefix sums either way.
///
/// ```
/// use escapepod_signal::features::{SpanConfig, SpanScratch, SpanStatsOut, span_stats};
///
/// let signal: Vec<f32> = (0..100).map(|i| i as f32).collect();
/// let spans = [[10, 20], [-1, -1]];
/// let (mut dwell, mut mean, mut sd) = ([0.0; 2], [0.0; 2], [0.0; 2]);
/// let (mut median, mut range) = ([0.0; 2], [0.0; 2]);
/// span_stats(
///     &signal,
///     &spans,
///     SpanConfig::default(),
///     &mut SpanScratch::default(),
///     SpanStatsOut::new(&mut dwell, &mut mean, &mut sd)
///         .with_median(&mut median)
///         .with_range(&mut range),
/// );
/// assert_eq!((dwell[0], mean[0], median[0], range[0]), (10.0, 14.5, 14.5, 9.0));
/// assert!(dwell[1].is_nan(), "an unresolved span abstains rather than guessing");
/// ```
pub fn span_stats(
    signal: &[f32],
    spans: &[[i64; 2]],
    cfg: SpanConfig,
    scratch: &mut SpanScratch,
    out: SpanStatsOut<'_>,
) {
    let SpanStatsOut {
        dwell,
        mean,
        sd,
        mut median,
        mut range,
    } = out;
    debug_assert_eq!(dwell.len(), spans.len());
    debug_assert_eq!(mean.len(), spans.len());
    debug_assert_eq!(sd.len(), spans.len());
    debug_assert!(median.as_deref().is_none_or(|m| m.len() == spans.len()));
    debug_assert!(range.as_deref().is_none_or(|r| r.len() == spans.len()));
    let fill = cfg.fill.value();
    dwell.fill(fill);
    mean.fill(fill);
    sd.fill(fill);
    if let Some(m) = median.as_deref_mut() {
        m.fill(fill);
    }
    if let Some(r) = range.as_deref_mut() {
        r.fill(fill);
    }
    if signal.is_empty() || spans.is_empty() {
        return;
    }

    let (centre, scale) = match cfg.norm {
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
        if let Some((a, b)) = resolve_span(s, n_sig, cfg.bounds) {
            lo = lo.min(a);
            hi = hi.max(b);
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

    let per_span = median.is_some() || range.is_some();
    for (i, s) in spans.iter().enumerate() {
        let Some((a, b)) = resolve_span(s, n_sig, cfg.bounds) else {
            continue;
        };
        let (ai, bi) = ((a - lo) as usize, (b - lo) as usize);
        let n = (b - a) as f64;
        let m = (scratch.cumsum[bi] - scratch.cumsum[ai]) / n;
        let var = (scratch.cumsum_sq[bi] - scratch.cumsum_sq[ai]) / n - m * m;
        dwell[i] = n as f32;
        mean[i] = m as f32;
        // Clamp the cancellation floor: a constant span can land at -1e-17.
        sd[i] = if var > 0.0 { var.sqrt() as f32 } else { 0.0 };

        if !per_span {
            continue;
        }
        // One gather serves both. It holds the normalised levels as `f32`,
        // which is what a numpy-side reference would be holding, so the median
        // convention is the only thing left that can differ from it.
        let buf = &mut scratch.span;
        buf.clear();
        buf.extend(
            signal[a as usize..b as usize]
                .iter()
                .map(|&v| ((v as f64 - centre) / scale) as f32),
        );
        // Range first: it reads the gather in order, the median reorders it.
        if let Some(r) = range.as_deref_mut() {
            r[i] = span_range(buf);
        }
        if let Some(md) = median.as_deref_mut() {
            md[i] = match cfg.median {
                MedianConvention::SelectTotalCmp => median_via_select(buf),
                MedianConvention::SortPartialCmp => median_numpy(buf),
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    /// `(dwell, mean, sd)` only -- the shape every pre-existing caller uses.
    fn run(signal: &[f32], spans: &[[i64; 2]], cfg: SpanConfig) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let n = spans.len();
        let (mut d, mut m, mut s) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        let mut scratch = SpanScratch::default();
        span_stats(
            signal,
            spans,
            cfg,
            &mut scratch,
            SpanStatsOut::new(&mut d, &mut m, &mut s),
        );
        (d, m, s)
    }

    struct Outs {
        dwell: Vec<f32>,
        mean: Vec<f32>,
        sd: Vec<f32>,
        median: Vec<f32>,
        range: Vec<f32>,
    }

    /// Every output, including the two optional ones.
    fn run_full(signal: &[f32], spans: &[[i64; 2]], cfg: SpanConfig) -> Outs {
        let n = spans.len();
        let mut o = Outs {
            dwell: vec![0.0; n],
            mean: vec![0.0; n],
            sd: vec![0.0; n],
            median: vec![0.0; n],
            range: vec![0.0; n],
        };
        let mut scratch = SpanScratch::default();
        span_stats(
            signal,
            spans,
            cfg,
            &mut scratch,
            SpanStatsOut::new(&mut o.dwell, &mut o.mean, &mut o.sd)
                .with_median(&mut o.median)
                .with_range(&mut o.range),
        );
        o
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

    /// `span_stats` exactly as it stood before [`SpanConfig`] existed, kept
    /// verbatim as the oracle for "the default config changed nothing". If this
    /// and the default path ever disagree bit-for-bit, the refactor was a
    /// silent behaviour change.
    fn legacy_span_stats(
        signal: &[f32],
        spans: &[[i64; 2]],
        norm: Normalization,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let n = spans.len();
        let (mut dwell, mut mean, mut sd) = (vec![0.0f32; n], vec![0.0f32; n], vec![0.0f32; n]);
        let (mut sc_median, mut cumsum, mut cumsum_sq) =
            (Vec::<f32>::new(), Vec::<f64>::new(), Vec::<f64>::new());

        dwell.fill(f32::NAN);
        mean.fill(f32::NAN);
        sd.fill(f32::NAN);
        if signal.is_empty() || spans.is_empty() {
            return (dwell, mean, sd);
        }

        let (centre, scale) = match norm {
            Normalization::None => (0.0f64, 1.0f64),
            Normalization::MedianMad { mad_floor } => {
                let (med, mad) = median_and_mad_with_scratch(signal, &mut sc_median);
                let s = if mad > mad_floor { 1.4826 * mad } else { 1.0 };
                (med as f64, s as f64)
            }
        };

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
            return (dwell, mean, sd);
        }
        let (lo_u, hi_u) = (lo as usize, hi as usize);

        cumsum.clear();
        cumsum_sq.clear();
        cumsum.reserve(hi_u - lo_u + 1);
        cumsum_sq.reserve(hi_u - lo_u + 1);
        cumsum.push(0.0);
        cumsum_sq.push(0.0);
        let (mut acc, mut acc_sq) = (0.0f64, 0.0f64);
        for &v in &signal[lo_u..hi_u] {
            let z = (v as f64 - centre) / scale;
            acc += z;
            acc_sq += z * z;
            cumsum.push(acc);
            cumsum_sq.push(acc_sq);
        }

        for (i, s) in spans.iter().enumerate() {
            let (a, b) = (s[0], s[1]);
            if a < 0 || b <= a || b > n_sig {
                continue;
            }
            let (ai, bi) = ((a - lo) as usize, (b - lo) as usize);
            let n = (b - a) as f64;
            let m = (cumsum[bi] - cumsum[ai]) / n;
            let var = (cumsum_sq[bi] - cumsum_sq[ai]) / n - m * m;
            dwell[i] = n as f32;
            mean[i] = m as f32;
            sd[i] = if var > 0.0 { var.sqrt() as f32 } else { 0.0 };
        }
        (dwell, mean, sd)
    }

    fn xorshift(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    fn pseudo_signal(n: usize) -> Vec<f32> {
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        (0..n)
            .map(|_| {
                let r = xorshift(&mut state);
                80.0 + 30.0 * ((r >> 11) as f64 / (1u64 << 53) as f64) as f32
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

    /// Every flavour of out-of-range, plus in-range neighbours either side.
    fn edge_spans() -> Vec<[i64; 2]> {
        vec![
            [-4, 4],
            [-9, -3],
            [0, 1],
            [5, 5],
            [7, 3],
            [12, 20],
            [19, 20],
            [19, 21],
            [20, 30],
            [0, 20],
        ]
    }

    #[test]
    fn matches_the_per_span_loop() {
        let sig = pseudo_signal(20_000);
        let spans = mixed_spans();
        let norm = Normalization::MedianMad { mad_floor: 1e-3 };
        let (wd, wm, ws) = reference(&sig, &spans, norm);
        let (gd, gm, gs) = run(&sig, &spans, SpanConfig::new(norm));
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

    /// The guardrail for the whole change: with `SpanConfig::default()` the new
    /// code must be bit-for-bit the old code, whether or not the optional
    /// outputs are asked for.
    #[test]
    fn the_default_config_is_bit_identical_to_the_pre_config_implementation() {
        let long = pseudo_signal(20_000);
        let short: Vec<f32> = (0..20).map(|i| i as f32 * 0.5 - 3.0).collect();
        let flat = vec![42.0f32; 64];
        let cases: [(&[f32], Vec<[i64; 2]>); 6] = [
            (&long, mixed_spans()),
            (&long, edge_spans()),
            (&short, edge_spans()),
            (&flat, edge_spans()),
            (&long, Vec::new()),
            (&[], mixed_spans()),
        ];
        for norm in [
            Normalization::None,
            Normalization::MedianMad { mad_floor: 1e-3 },
            Normalization::MedianMad { mad_floor: 1e9 },
        ] {
            for (sig, spans) in &cases {
                let (ld, lm, ls) = legacy_span_stats(sig, spans, norm);
                let cfg = SpanConfig::new(norm);

                let (d, m, s) = run(sig, spans, cfg);
                assert_eq!(bits(&ld), bits(&d), "dwell, {norm:?}");
                assert_eq!(bits(&lm), bits(&m), "mean, {norm:?}");
                assert_eq!(bits(&ls), bits(&s), "sd, {norm:?}");

                // Asking for the optional outputs must not perturb the three
                // that come from the prefix sums.
                let full = run_full(sig, spans, cfg);
                assert_eq!(bits(&ld), bits(&full.dwell), "dwell + extras, {norm:?}");
                assert_eq!(bits(&lm), bits(&full.mean), "mean + extras, {norm:?}");
                assert_eq!(bits(&ls), bits(&full.sd), "sd + extras, {norm:?}");
            }
        }
    }

    #[test]
    fn invalid_spans_are_nan() {
        let sig = vec![1.0f32; 100];
        let spans = [[-1, -1], [50, 40], [90, 200], [10, 20]];
        let (d, m, s) = run(&sig, &spans, SpanConfig::default());
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
        let cfg = SpanConfig::new(Normalization::MedianMad { mad_floor: 1e-3 });
        let mut scratch = SpanScratch::default();
        let n = spans.len();
        let mut a = Outs {
            dwell: vec![0.0; n],
            mean: vec![0.0; n],
            sd: vec![0.0; n],
            median: vec![0.0; n],
            range: vec![0.0; n],
        };
        let mut b = Outs {
            dwell: vec![0.0; n],
            mean: vec![0.0; n],
            sd: vec![0.0; n],
            median: vec![0.0; n],
            range: vec![0.0; n],
        };
        for o in [&mut a, &mut b] {
            span_stats(
                &sig,
                &spans,
                cfg,
                &mut scratch,
                SpanStatsOut::new(&mut o.dwell, &mut o.mean, &mut o.sd)
                    .with_median(&mut o.median)
                    .with_range(&mut o.range),
            );
        }
        // to_bits, not ==: NaN never equals itself, and these are mostly NaN.
        assert_eq!(bits(&a.dwell), bits(&b.dwell));
        assert_eq!(bits(&a.mean), bits(&b.mean));
        assert_eq!(bits(&a.sd), bits(&b.sd));
        assert_eq!(bits(&a.median), bits(&b.median));
        assert_eq!(bits(&a.range), bits(&b.range));
    }

    #[test]
    fn mad_floor_prevents_blowup_on_a_flat_read() {
        let sig = vec![42.0f32; 1000];
        let (_, m, s) = run(
            &sig,
            &[[10, 50]],
            SpanConfig::new(Normalization::MedianMad { mad_floor: 1e-3 }),
        );
        assert_eq!(
            m[0], 0.0,
            "flat read centres to zero, scale falls back to 1"
        );
        assert_eq!(s[0], 0.0);
    }

    #[test]
    fn empty_inputs_are_handled() {
        let (d, _, _) = run(&[], &[[0, 10]], SpanConfig::default());
        assert!(d[0].is_nan());
        let (d, _, _) = run(&pseudo_signal(100), &[], SpanConfig::default());
        assert!(d.is_empty());
    }

    // ---- median -----------------------------------------------------------

    // Goldens generated with numpy 2.5.1 (the pixi `python-test` env). The
    // signal is emitted as raw f32 bit patterns so no decimal literal has to
    // round-trip, and the expected values are `np.median` / `np.ptp` of the
    // same array:
    //
    //   sig = np.array(<the values below>, dtype=np.float32)
    //   for a, b in spans:
    //       print(hex(np.float32(np.median(sig[a:b])).view(np.uint32)),
    //             hex(np.float32(np.ptp(sig[a:b])).view(np.uint32)))
    //
    // Spans 0 and 1 are eight ulp-separated values around 1.0 and 100.7,
    // shuffled -- the "near-equal float32" case where the two conventions were
    // expected to disagree. Span 4 contains a NaN, which is where they really
    // do.
    const NUMPY_SIG_BITS: [u32; 59] = [
        0x3f800003, 0x3f800007, 0x3f800000, 0x3f800005, 0x3f800001, 0x3f800006, 0x3f800002,
        0x3f800004, 0x42c9666c, 0x42c96668, 0x42c9666b, 0x42c96666, 0x42c9666d, 0x42c96667,
        0x42c9666a, 0x42c96669, 0x40600000, 0xbfa00000, 0x42b44000, 0x00000000, 0x414c0000,
        0x422a0000, 0x3f800000, 0x7fc00000, 0x40400000, 0x40000000, 0x42bc0419, 0x42bb2a8c,
        0x42c9b26b, 0x42b3f085, 0x42af0260, 0x42c440a6, 0x42cdca9b, 0x42b9e920, 0x42e741d6,
        0x42a7e1e8, 0x42b11175, 0x42b330e2, 0x42c36c62, 0x42cadc9d, 0x42b2f150, 0x42bcf3c0,
        0x42c5eeb2, 0x42cca56b, 0x42b048a5, 0x42b68134, 0x42bdc01e, 0x42be0701, 0x42a28bda,
        0x42c1c51c, 0x42bf2da1, 0x42c94b43, 0x42c02fa8, 0x42bc5760, 0x42bd1a0e, 0x42be1737,
        0x42e47f0a, 0x42be5503, 0x42a963b0,
    ];
    const NUMPY_SPANS: [[i64; 2]; 6] = [[0, 8], [8, 16], [16, 21], [21, 22], [22, 26], [26, 59]];
    /// `np.median(sig[a:b])` per span, as f32 bits.
    const NUMPY_MEDIAN_BITS: [u32; 6] = [
        0x3f800004, // 1.0000004768371582
        0x42c9666a, // 100.70002746582031
        0x40600000, // 3.5
        0x422a0000, // 42.5 (single element)
        0x7fc00000, // nan (the span holds one)
        0x42bdc01e, // 94.87522888183594
    ];
    /// `np.ptp(sig[a:b])` per span, as f32 bits.
    const NUMPY_PTP_BITS: [u32; 6] = [
        0x35600000, // 8.344650268554688e-07 (7 ulps of 1.0)
        0x38600000, // 5.340576171875e-05
        0x42b6c000, // 91.375
        0x00000000, // 0.0 (single element)
        0x7fc00000, // nan
        0x42096bf8, // 34.355438232421875
    ];

    fn numpy_fixture_signal() -> Vec<f32> {
        NUMPY_SIG_BITS.iter().copied().map(f32::from_bits).collect()
    }

    #[test]
    fn sort_partial_cmp_reproduces_numpy_median_and_ptp() {
        let sig = numpy_fixture_signal();
        let o = run_full(
            &sig,
            &NUMPY_SPANS,
            SpanConfig::default().with_median(MedianConvention::SortPartialCmp),
        );
        for (i, (&want_med, &want_ptp)) in NUMPY_MEDIAN_BITS
            .iter()
            .zip(NUMPY_PTP_BITS.iter())
            .enumerate()
        {
            assert_eq!(
                o.median[i].to_bits(),
                want_med,
                "median[{i}] = {} vs numpy {}",
                o.median[i],
                f32::from_bits(want_med)
            );
            assert_eq!(
                o.range[i].to_bits(),
                want_ptp,
                "range[{i}] = {} vs numpy {}",
                o.range[i],
                f32::from_bits(want_ptp)
            );
        }
    }

    /// The issue expected the two conventions to split on even-length spans of
    /// near-equal `f32`. Measured, they do not: `total_cmp` and `partial_cmp`
    /// order non-`NaN` values identically, so both average the same two order
    /// statistics, and `numpy.median`'s `float32` two-element mean is bit-for-bit
    /// `(a + b) / 2.0`. This is the test that says so.
    #[test]
    fn the_median_conventions_agree_on_every_finite_span() {
        let mut state = 0xDEAD_BEEF_1234_5678u64;
        for &base in &[1.0f32, 100.7, 1e-3, -12.5, 65_504.0] {
            for len in 1..=17usize {
                for _ in 0..64 {
                    // Values within a few ulps of `base`, so the two middles of
                    // an even-length span are as close as f32 allows.
                    let sig: Vec<f32> = (0..len)
                        .map(|_| {
                            let step = (xorshift(&mut state) % 5) as u32;
                            f32::from_bits(base.to_bits() + step)
                        })
                        .collect();
                    let spans = [[0i64, len as i64]];
                    let sel = run_full(&sig, &spans, SpanConfig::default());
                    let np = run_full(
                        &sig,
                        &spans,
                        SpanConfig::default().with_median(MedianConvention::SortPartialCmp),
                    );
                    assert_eq!(
                        sel.median[0].to_bits(),
                        np.median[0].to_bits(),
                        "base {base}, len {len}: {sig:?}"
                    );
                }
            }
        }
        // Ordinary magnitudes too, not just the near-equal case.
        for len in 1..=33usize {
            for _ in 0..64 {
                let sig: Vec<f32> = (0..len)
                    .map(|_| {
                        let r = xorshift(&mut state);
                        ((r >> 11) as f64 / (1u64 << 53) as f64) as f32 * 200.0 - 100.0
                    })
                    .collect();
                let spans = [[0i64, len as i64]];
                let sel = run_full(&sig, &spans, SpanConfig::default());
                let np = run_full(
                    &sig,
                    &spans,
                    SpanConfig::default().with_median(MedianConvention::SortPartialCmp),
                );
                assert_eq!(sel.median[0].to_bits(), np.median[0].to_bits(), "len {len}");
            }
        }
    }

    /// ...and this is where they genuinely differ, which is why both are
    /// offered rather than one chosen.
    #[test]
    fn the_median_conventions_disagree_on_a_nan_span() {
        let sig = [1.0f32, f32::NAN, 3.0, 2.0];
        let spans = [[0i64, 4]];
        let sel = run_full(&sig, &spans, SpanConfig::default());
        let np = run_full(
            &sig,
            &spans,
            SpanConfig::default().with_median(MedianConvention::SortPartialCmp),
        );
        assert_eq!(
            sel.median[0], 2.5,
            "total_cmp sorts NaN to the high end, so 2.0 and 3.0 are the middles"
        );
        assert!(np.median[0].is_nan(), "numpy.median propagates NaN");
        assert!(
            sel.mean[0].is_nan(),
            "the mean already propagates it, which is the argument for SortPartialCmp"
        );
    }

    #[test]
    fn median_even_averages_odd_picks_and_a_single_sample_is_itself() {
        let sig = [1.0f32, 2.0, 3.0, 4.0];
        for conv in [
            MedianConvention::SelectTotalCmp,
            MedianConvention::SortPartialCmp,
        ] {
            let o = run_full(
                &sig,
                &[[0, 4], [0, 3], [0, 1], [-1, -1]],
                SpanConfig::default().with_median(conv),
            );
            assert_eq!(o.median[0], 2.5, "{conv:?}: even averages the two middles");
            assert_eq!(o.median[1], 2.0, "{conv:?}: odd takes the middle");
            assert_eq!(o.median[2], 1.0, "{conv:?}: a single sample is itself");
            assert!(o.median[3].is_nan(), "{conv:?}: unresolved takes the fill");
        }
    }

    // ---- range ------------------------------------------------------------

    #[test]
    fn range_is_max_minus_min() {
        let sig = [1.0f32, 5.0, -3.0, 5.0, 7.0, 7.0, 7.0];
        let o = run_full(
            &sig,
            &[[0, 5], [4, 7], [0, 1], [-1, -1]],
            SpanConfig::default(),
        );
        assert_eq!(o.range[0], 10.0, "7 - (-3)");
        assert_eq!(o.range[1], 0.0, "a constant span has zero range");
        assert_eq!(o.range[2], 0.0, "so does a single sample");
        assert!(o.range[3].is_nan(), "unresolved takes the fill");
    }

    #[test]
    fn range_and_median_are_normalised_like_the_mean() {
        let sig = pseudo_signal(2_000);
        let norm = Normalization::MedianMad { mad_floor: 1e-3 };
        let spans = [[100i64, 140]];
        let raw = run_full(&sig, &spans, SpanConfig::default());
        let z = run_full(&sig, &spans, SpanConfig::new(norm));
        let (med, mad) = crate::stats::median_and_mad(&sig);
        let scale = 1.4826 * mad;
        assert!(
            (z.median[0] - (raw.median[0] - med) / scale).abs() < 1e-4,
            "median is taken over the normalised levels"
        );
        assert!(
            (z.range[0] - raw.range[0] / scale).abs() < 1e-4,
            "range scales but does not shift"
        );
    }

    // ---- fill -------------------------------------------------------------

    #[test]
    fn the_fill_lands_in_every_output_including_the_optional_ones() {
        let sig = vec![1.0f32; 20];
        let spans = [[-1, -1], [0, 5]];
        for (fill, want) in [
            (SpanFill::Nan, f32::NAN),
            (SpanFill::Zero, 0.0),
            (SpanFill::Value(-7.5), -7.5),
        ] {
            let o = run_full(&sig, &spans, SpanConfig::default().with_fill(fill));
            for (name, got) in [
                ("dwell", o.dwell[0]),
                ("mean", o.mean[0]),
                ("sd", o.sd[0]),
                ("median", o.median[0]),
                ("range", o.range[0]),
            ] {
                assert_eq!(got.to_bits(), want.to_bits(), "{fill:?}: {name}");
            }
            // A resolved span is untouched by the fill.
            assert_eq!(o.dwell[1], 5.0, "{fill:?}");
            assert_eq!(o.mean[1], 1.0, "{fill:?}");
            assert_eq!(o.median[1], 1.0, "{fill:?}");
            assert_eq!(o.range[1], 0.0, "{fill:?}");
        }
        // Zero is exactly Value(0.0); it is spelled separately for the reader.
        let a = run_full(
            &sig,
            &spans,
            SpanConfig::default().with_fill(SpanFill::Zero),
        );
        let b = run_full(
            &sig,
            &spans,
            SpanConfig::default().with_fill(SpanFill::Value(0.0)),
        );
        assert_eq!(bits(&a.dwell), bits(&b.dwell));
        assert_eq!(bits(&a.median), bits(&b.median));
    }

    // ---- bounds -----------------------------------------------------------

    #[test]
    fn clamp_summarises_a_truncated_span_where_skip_abstains() {
        let sig: Vec<f32> = (0..10).map(|i| i as f32).collect();
        //           truncated left, truncated right, wholly left, wholly
        //           right, inverted, fully in range
        let spans = [[-4, 4], [6, 20], [-9, -3], [20, 30], [7, 3], [2, 5]];
        let skip = run_full(&sig, &spans, SpanConfig::default());
        let clamp = run_full(
            &sig,
            &spans,
            SpanConfig::default().with_bounds(SpanBounds::Clamp),
        );

        for (i, d) in skip.dwell.iter().take(5).enumerate() {
            assert!(d.is_nan(), "Skip abstains on span {i}");
        }

        // [-4, 4) -> [0, 4): samples 0,1,2,3.
        assert_eq!(
            clamp.dwell[0], 4.0,
            "dwell is the clamped length, not the requested 8"
        );
        assert_eq!(clamp.mean[0], 1.5);
        assert_eq!(clamp.median[0], 1.5);
        assert_eq!(clamp.range[0], 3.0);
        assert_eq!(clamp.sd[0], 1.25f32.sqrt());
        // [6, 20) -> [6, 10): samples 6,7,8,9.
        assert_eq!(clamp.dwell[1], 4.0);
        assert_eq!(clamp.mean[1], 7.5);
        assert_eq!(clamp.range[1], 3.0);
        // Nothing survives clamping for these, so they are unresolved either
        // way.
        for (i, d) in clamp.dwell.iter().enumerate().skip(2).take(3) {
            assert!(d.is_nan(), "Clamp abstains on span {i}");
        }
        // A fully in-range span is bit-identical under both policies.
        assert_eq!(skip.dwell[5].to_bits(), clamp.dwell[5].to_bits());
        assert_eq!(skip.mean[5].to_bits(), clamp.mean[5].to_bits());
        assert_eq!(skip.sd[5].to_bits(), clamp.sd[5].to_bits());
    }

    #[test]
    fn clamp_takes_the_fill_for_a_span_that_survives_nothing() {
        let sig = vec![3.0f32; 8];
        let cfg = SpanConfig::default()
            .with_bounds(SpanBounds::Clamp)
            .with_fill(SpanFill::Zero);
        let o = run_full(&sig, &[[-5, 0], [8, 12], [4, 4]], cfg);
        for i in 0..3 {
            assert_eq!(o.dwell[i], 0.0, "span {i}");
            assert_eq!(o.mean[i], 0.0, "span {i}");
            assert_eq!(o.median[i], 0.0, "span {i}");
            assert_eq!(o.range[i], 0.0, "span {i}");
        }
    }

    #[test]
    fn clamp_widens_the_prefix_sum_region_to_the_clamped_spans() {
        // Under Skip nothing resolves, so nothing is summed; under Clamp the
        // region has to cover [0, 10) or the second span reads the wrong
        // prefix.
        let sig: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let spans = [[-3, 2], [8, 14]];
        let o = run_full(
            &sig,
            &spans,
            SpanConfig::default().with_bounds(SpanBounds::Clamp),
        );
        assert_eq!((o.dwell[0], o.mean[0]), (2.0, 0.5));
        assert_eq!((o.dwell[1], o.mean[1]), (2.0, 8.5));
    }
}
