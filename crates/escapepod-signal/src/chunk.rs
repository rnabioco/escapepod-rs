// SPDX-License-Identifier: MIT

//! Windowed model inputs: one read reduced to per-sample channels and
//! per-base rows, then cut into fixed-size chunks around a chosen base.
//!
//! This is the assembly step every signal-level network needs and nothing
//! above it should own: take a read's raw samples and its base-to-signal map,
//! put both in one frame, optionally refine the map against expected k-mer
//! levels, derive the channels the model reads, and cut a window of fixed
//! width around a base the caller nominates. The primitives it is built from
//! already live in this crate — [`crate::mapping`] produces the maps,
//! [`crate::features::span_stats`] reduces the spans,
//! [`crate::seq_encoding`] encodes the sequence, [`crate::resquiggle`]
//! refines the map and looks up the levels — and this module is the one
//! place they are wired together in the order a trained model was fitted on.
//!
//! It exists because that wiring is exactly the kind of rule that fails
//! *silently*. A window placed on the wrong side of a base, a channel list in
//! a different order, a k-mer context split `(4, 4)` instead of `(3, 5)`: each
//! produces a tensor of precisely the right shape, full of plausible numbers,
//! that scores every read confidently and wrongly. The rule had two
//! implementations already (leech's Python dataset and leech-core's Rust
//! pipeline) and a third was about to be written in `escapepod-classify`, so
//! it moved down here instead (rnabioco/escapepod-rs#306).
//!
//! # Nothing here knows what it is measuring
//!
//! The anchor is a base index the caller computes, the window is
//! `(left, right)` samples, and the channels are a *list the caller supplies*
//! — [`FeatureChannel`] and [`SignalChannel`] name signal quantities (dwell,
//! level, k-mer residual), never assay concepts. Two models that read the
//! same twelve rows in a different order are two different `Vec`s here, not
//! two code paths, and a model that wants nine of them pays for nine. That is
//! the whole reason the channel list is data: `leech`'s
//! `merge_feature_channels` says plainly that reordering it "silently feeds
//! `level_mean` into the filter that learned `dwell_log`", and a hard-coded
//! order cannot be checked against what a bundle declares.
//!
//! # Shape of a call
//!
//! ```no_run
//! use escapepod_signal::chunk::{
//!     Anchor, ChunkSpec, FeatureChannel, ProcessConfig, ReadInputs, SignalChannel,
//!     cut_chunk, process_read, read_rows,
//! };
//!
//! # fn demo(raw: &[i16], moves: &[u8], sequence: &[u8]) -> Option<()> {
//! let cfg = ProcessConfig::default();
//! let read = process_read(
//!     ReadInputs { raw, moves, stride: 5, trim: 0, num_samples: raw.len() as u64 },
//!     Anchor::Query { sequence },
//!     &cfg,
//! )?;
//! let spec = ChunkSpec {
//!     signal_channels: vec![SignalChannel::Current],
//!     feature_channels: vec![FeatureChannel::Dwell, FeatureChannel::LevelMean],
//!     ..ChunkSpec::default()
//! };
//! let rows = read_rows(&read, &spec);
//! let chunk = cut_chunk(&read, &rows, &spec, 100)?;
//! assert_eq!(chunk.signal.len(), spec.signal_len);
//! # Some(())
//! # }
//! ```

use std::collections::HashMap;

use crate::features::{
    MedianConvention, Normalization, SpanBounds, SpanConfig, SpanFill, SpanScratch, SpanStatsOut,
    span_stats,
};
use crate::mapping::{CigarOp, ref_to_signal, seq_to_signal_from_moves};
use crate::resquiggle::{RefineSettings, extract_levels, refine_signal_map};
use crate::segmentation::mad_normalize_robust;
use crate::seq_encoding::{
    KmerContext, encode_signal_kmer, sequence_bases_with_context, sequence_to_int,
};

/// Where in a base's signal span the window is centred.
///
/// The three are not interchangeable and the difference is a *shift*, not an
/// error: a window justified to the centre of a base sits roughly half a dwell
/// from one justified to its end, which is tens of samples on a slow base. A
/// model reads whichever one it was trained with, and the other still scores.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BaseJustify {
    /// First sample of the base's span.
    Start,
    /// Midpoint, `(start + end) / 2` (floor).
    #[default]
    Center,
    /// One past the base's last sample — the boundary with the next base.
    End,
}

impl BaseJustify {
    /// Parse the spelling a config file uses (`start`, `center`, `end`).
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "start" => Some(Self::Start),
            "center" | "centre" => Some(Self::Center),
            "end" => Some(Self::End),
            _ => None,
        }
    }

    /// The spelling [`from_name`](Self::from_name) accepts.
    pub fn name(self) -> &'static str {
        match self {
            Self::Start => "start",
            Self::Center => "center",
            Self::End => "end",
        }
    }

    /// The focus sample for a base spanning `[start, end)`.
    #[inline]
    pub fn focus(self, start: i64, end: i64) -> i64 {
        match self {
            Self::Start => start,
            Self::End => end,
            Self::Center => (start + end) / 2,
        }
    }
}

/// A per-sample channel of the signal tensor, one row of `[C, signal_len]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SignalChannel {
    /// The normalised current itself.
    Current,
    /// Current minus the expected level of the base that sample belongs to.
    ///
    /// Zero where the base has no expected level (an unknown k-mer, or a
    /// position with no full k-mer window), which is also how a level of
    /// exactly zero is treated — the two are indistinguishable in the level
    /// table and the reference implementation does not distinguish them.
    KmerResidual,
}

impl SignalChannel {
    /// Parse the name a config file uses.
    ///
    /// `signal`/`signal_kmer_residual` are the spellings a model bundle ships;
    /// `current`/`kmer_residual` are accepted as the names this crate would
    /// have chosen. The canonical form is the bundle's, because that is the
    /// one a declaration is checked against.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "signal" | "current" => Some(Self::Current),
            "signal_kmer_residual" | "kmer_residual" => Some(Self::KmerResidual),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Current => "signal",
            Self::KmerResidual => "signal_kmer_residual",
        }
    }

    /// Does this channel need expected k-mer levels?
    pub fn needs_levels(self) -> bool {
        matches!(self, Self::KmerResidual)
    }
}

/// A per-base row of the feature tensor, one row of `[F, width]`.
///
/// The names are `leech`'s, because they are what a bundle declares and the
/// point of naming them at all is that the declaration can be checked. Adding
/// a variant is adding a row a model may ask for; renaming one silently
/// re-points every bundle that names it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FeatureChannel {
    /// Samples in the base's span, as an exact integer count.
    Dwell,
    /// `ln(dwell + 1e-6)`.
    DwellLog,
    /// Centred rolling mean of `dwell` over [`ChunkSpec::dwell_window`] bases,
    /// edges padded by repeating the first and last dwell.
    DwellMean,
    /// Population standard deviation over the same rolling window.
    DwellStd,
    /// `dwell / (dwell_mean + 1e-6)`.
    DwellRatio,
    /// Mean signal level over the base's span.
    LevelMean,
    /// Median signal level over the base's span (`numpy.median` convention).
    LevelMedian,
    /// Population standard deviation of the level over the span.
    LevelStd,
    /// `max - min` of the level over the span.
    LevelRange,
    /// Expected level of the base's k-mer, from the level table.
    KmerExpected,
    /// `level_mean - kmer_expected`.
    KmerResidual,
    /// `|level_mean - kmer_expected|`.
    KmerResidualAbs,
}

impl FeatureChannel {
    /// Parse the name a config file uses.
    pub fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "dwell" => Self::Dwell,
            "dwell_log" => Self::DwellLog,
            "dwell_mean" => Self::DwellMean,
            "dwell_std" => Self::DwellStd,
            "dwell_ratio" => Self::DwellRatio,
            "level_mean" => Self::LevelMean,
            "level_median" => Self::LevelMedian,
            "level_std" => Self::LevelStd,
            "level_range" => Self::LevelRange,
            "kmer_expected" => Self::KmerExpected,
            "kmer_residual" => Self::KmerResidual,
            "kmer_residual_abs" => Self::KmerResidualAbs,
            _ => return None,
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Dwell => "dwell",
            Self::DwellLog => "dwell_log",
            Self::DwellMean => "dwell_mean",
            Self::DwellStd => "dwell_std",
            Self::DwellRatio => "dwell_ratio",
            Self::LevelMean => "level_mean",
            Self::LevelMedian => "level_median",
            Self::LevelStd => "level_std",
            Self::LevelRange => "level_range",
            Self::KmerExpected => "kmer_expected",
            Self::KmerResidual => "kmer_residual",
            Self::KmerResidualAbs => "kmer_residual_abs",
        }
    }

    /// Does this row need expected k-mer levels?
    pub fn needs_levels(self) -> bool {
        matches!(
            self,
            Self::KmerExpected | Self::KmerResidual | Self::KmerResidualAbs
        )
    }
}

/// How the sequence is presented to the model, if at all.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SeqEncoding {
    /// No sequence tensor.
    #[default]
    None,
    /// One-hot over the `2 * context + 1` bases around the anchor, shape
    /// `[4, 2 * context + 1]`. Bases past either end of the sequence are `N`
    /// and encode as an all-zero column.
    BaseOneHot { context: usize },
    /// The k-mer context of every base, scattered along the *signal* axis:
    /// shape `[4 * ctx.kmer_len(), signal_len]`. See
    /// [`crate::seq_encoding::encode_signal_kmer`].
    SignalKmer { ctx: KmerContext },
}

/// How the raw signal is put on a common scale.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SignalNorm {
    /// Leave the samples as they are.
    None,
    /// Per-read median/MAD with the Gaussian factor 1.4826
    /// ([`crate::segmentation::mad_normalize_robust`]).
    #[default]
    MedianMad,
}

/// The k-mer level table and how to index it.
#[derive(Clone, Copy, Debug)]
pub struct LevelModel<'a> {
    /// k-mer (uppercase, `U` folded to `T`) → expected level.
    pub table: &'a HashMap<String, f64>,
    pub kmer_len: usize,
    /// Which base of the k-mer the level is assigned to.
    pub center_idx: usize,
}

/// Banded-DP refinement of the base-to-signal map, before any feature is taken.
///
/// Refinement moves the boundaries the whole feature set is measured over, so
/// a model trained on refined spans and scored on move-table spans reads a
/// different stretch of signal for every base — the failure
/// `escapepod-classify` used to refuse outright rather than approximate.
#[derive(Clone, Copy, Debug)]
pub struct RefineParams {
    pub half_bandwidth: usize,
    /// Rescaling iterations; `0` is one DP pass with no rescale.
    pub scale_iters: usize,
    /// Seed for the Theil-Sen subsample, so a long read refines reproducibly.
    pub seed: Option<u64>,
}

/// Fixed seed the reference implementations both use.
pub const DEFAULT_REFINE_SEED: u64 = 42;

/// Width of the centred rolling window behind [`FeatureChannel::DwellMean`]
/// and [`FeatureChannel::DwellStd`] in every corpus built so far.
pub const DEFAULT_DWELL_WINDOW: usize = 5;

impl Default for RefineParams {
    fn default() -> Self {
        Self {
            half_bandwidth: 5,
            scale_iters: 2,
            seed: Some(DEFAULT_REFINE_SEED),
        }
    }
}

/// Everything [`process_read`] needs that is a property of the *model*, not of
/// the read.
#[derive(Clone, Copy, Debug, Default)]
pub struct ProcessConfig<'a> {
    /// Reverse the trimmed signal (and flip the map through its length) before
    /// anything else, i.e. put the samples in 5'→3' order.
    pub reverse_signal: bool,
    pub normalization: SignalNorm,
    /// Expected levels, needed by any residual channel and by refinement.
    pub levels: Option<LevelModel<'a>>,
    /// Refine the map before the features are taken. Requires `levels`.
    pub refine: Option<RefineParams>,
}

/// What the base-to-signal map indexes, and therefore what the anchor base
/// index counts in.
#[derive(Clone, Copy, Debug)]
pub enum Anchor<'a> {
    /// Basecall coordinates: the move-table map is used as it is, and
    /// `sequence` is the read's own basecall.
    Query { sequence: &'a [u8] },
    /// Reference coordinates: the move-table map is walked through the CIGAR
    /// ([`ref_to_signal`]), the signal is cropped to the aligned span, and
    /// `sequence` is the reference over exactly that span (i.e.
    /// `reference[alignment_start..alignment_end]`).
    ///
    /// This is what lets a window be placed on a base the basecaller got
    /// wrong: the coordinate comes from the reference and the CIGAR
    /// interpolates across the indel rather than the read having to spell it.
    Reference {
        sequence: &'a [u8],
        cigar: &'a [CigarOp],
    },
}

/// Raw per-read inputs: the signal and the move table that indexes it.
#[derive(Clone, Copy, Debug)]
pub struct ReadInputs<'a> {
    /// Raw ADC samples straight from the POD5 signal table.
    ///
    /// Deliberately *not* calibrated picoamps. Under
    /// [`SignalNorm::MedianMad`] the two differ only by the affine calibration,
    /// which median/MAD divides straight back out, so calibrating first would
    /// buy nothing and cost a rounding difference against the reference
    /// implementation, which reads the ADC values.
    pub raw: &'a [i16],
    /// The `mv` tag's move vector (without its leading stride element).
    pub moves: &'a [u8],
    /// The `mv` tag's stride.
    pub stride: u32,
    /// The `ts` tag: samples trimmed from the front before basecalling.
    pub trim: i64,
    /// The `ns` tag: samples the basecaller saw, counted from sample 0.
    pub num_samples: u64,
}

/// One read, put in the model's frame: normalised samples, a base-to-signal
/// map over them, the sequence that map indexes, and the expected levels.
#[derive(Clone, Debug)]
pub struct ProcessedRead {
    /// Normalised signal in the anchor's frame.
    pub signal: Vec<f32>,
    /// Boundary sample per base, length `n_bases + 1`, indexing [`Self::signal`].
    pub seq_to_sig: Vec<i64>,
    /// The sequence [`Self::seq_to_sig`] indexes.
    pub sequence: Vec<u8>,
    /// Expected k-mer level per base of [`Self::sequence`], when the config
    /// supplied a level model.
    pub levels: Option<Vec<f64>>,
}

impl ProcessedRead {
    /// Bases the map resolves, i.e. `seq_to_sig.len() - 1`.
    pub fn n_bases(&self) -> usize {
        self.seq_to_sig.len().saturating_sub(1)
    }
}

/// The geometry and channel list of the chunks a model reads.
#[derive(Clone, Debug)]
pub struct ChunkSpec {
    /// Samples taken before and after the focus sample; the requested window
    /// is `[focus - left, focus + right)`.
    pub signal_context: (i64, i64),
    /// Width of the emitted signal window. When it differs from
    /// `left + right` the window is centre-cropped (wider) or zero-padded
    /// (narrower) — see [`cut_chunk`].
    pub signal_len: usize,
    pub base_justify: BaseJustify,
    /// Rows of the `[C, signal_len]` signal tensor, in the model's order.
    pub signal_channels: Vec<SignalChannel>,
    pub seq_encoding: SeqEncoding,
    /// Base offsets the feature tensor covers, inclusive both ends.
    pub feature_offsets: (i64, i64),
    /// Rows of the `[F, width]` feature tensor, in the model's order.
    pub feature_channels: Vec<FeatureChannel>,
    /// Width of the centred rolling window behind [`FeatureChannel::DwellMean`]
    /// and [`FeatureChannel::DwellStd`].
    pub dwell_window: usize,
}

impl Default for ChunkSpec {
    fn default() -> Self {
        Self {
            signal_context: (0, 0),
            signal_len: 0,
            base_justify: BaseJustify::default(),
            signal_channels: Vec::new(),
            seq_encoding: SeqEncoding::None,
            feature_offsets: (0, 0),
            feature_channels: Vec::new(),
            dwell_window: DEFAULT_DWELL_WINDOW,
        }
    }
}

impl ChunkSpec {
    /// Bases the feature tensor spans, `feature_offsets` inclusive.
    pub fn feature_width(&self) -> usize {
        (self.feature_offsets.1 - self.feature_offsets.0 + 1).max(0) as usize
    }

    /// Does anything in this spec need expected k-mer levels?
    pub fn needs_levels(&self) -> bool {
        self.signal_channels.iter().any(|c| c.needs_levels())
            || self.feature_channels.iter().any(|c| c.needs_levels())
    }
}

/// The read-wide rows every chunk is cut from, computed once per read.
///
/// Separate from [`cut_chunk`] because a read usually yields several chunks
/// and every row here is read-wide: recomputing them per chunk would repeat
/// the whole per-base reduction for each window.
#[derive(Clone, Debug, Default)]
pub struct ReadRows {
    /// One entry per [`ChunkSpec::signal_channels`], each as long as the
    /// read's signal.
    pub sample_channels: Vec<Vec<f32>>,
    /// One entry per [`ChunkSpec::feature_channels`], each `n_bases` long.
    pub feature_rows: Vec<Vec<f32>>,
}

/// One assembled model input.
#[derive(Clone, Debug)]
pub struct Chunk {
    /// `[signal_channels.len(), signal_len]`, row-major.
    pub signal: Vec<f32>,
    /// The sequence tensor, row-major, `[sequence_rows, sequence_cols]`. Empty
    /// under [`SeqEncoding::None`].
    pub sequence: Vec<f32>,
    pub sequence_rows: usize,
    pub sequence_cols: usize,
    /// `[feature_channels.len(), feature_width]`, row-major.
    pub features: Vec<f32>,
    /// The base this chunk was cut around, in the anchor's coordinates.
    pub base_index: i64,
    /// The sample the window was centred on, in [`ProcessedRead::signal`].
    pub focus_signal_pos: i64,
}

// ---------------------------------------------------------------------------
// Per-read processing
// ---------------------------------------------------------------------------

/// Trim, orient, normalise, anchor and (optionally) refine one read.
///
/// Returns `None` for a read that cannot be put in the model's frame at all:
/// an empty trim window, an alignment that resolves fewer than two reference
/// boundaries, or a map that covers no signal. There is no partial answer to
/// return — every downstream coordinate is relative to this frame.
///
/// The order is load-bearing and matches the reference implementation:
/// trim → reverse → normalise → anchor-crop → refine. Normalising *before*
/// the crop is what makes the levels comparable across reads (the gauge is the
/// whole read, not the window), and refining after it is what puts the DP on
/// the same samples the features are taken from.
pub fn process_read(
    inputs: ReadInputs<'_>,
    anchor: Anchor<'_>,
    cfg: &ProcessConfig<'_>,
) -> Option<ProcessedRead> {
    let trim_start = inputs.trim.max(0) as usize;
    let trim_end = (inputs.num_samples as usize).min(inputs.raw.len());
    if trim_start >= trim_end {
        return None;
    }

    let mut signal: Vec<f32> = inputs.raw[trim_start..trim_end]
        .iter()
        .map(|&x| x as f32)
        .collect();
    let mut query_to_sig =
        seq_to_signal_from_moves(inputs.moves, inputs.stride, inputs.trim, inputs.num_samples);

    if cfg.reverse_signal {
        signal.reverse();
        let n = signal.len() as i64;
        query_to_sig.reverse();
        for v in &mut query_to_sig {
            *v = n - *v;
        }
    }

    let mut signal = match cfg.normalization {
        SignalNorm::None => signal,
        SignalNorm::MedianMad => mad_normalize_robust(&signal),
    };

    let (mut seq_to_sig, sequence) = match anchor {
        Anchor::Query { sequence } => (query_to_sig, sequence.to_vec()),
        Anchor::Reference { sequence, cigar } => {
            let ref_to_sig = ref_to_signal(&query_to_sig, cigar);
            if ref_to_sig.len() < 2 {
                return None;
            }
            let lo = ref_to_sig[0].max(0) as usize;
            let hi = (*ref_to_sig.last().expect("len >= 2")).min(signal.len() as i64);
            if hi <= lo as i64 {
                return None;
            }
            signal = signal[lo..hi as usize].to_vec();
            let shifted = ref_to_sig.iter().map(|&v| v - lo as i64).collect();
            (shifted, sequence.to_vec())
        }
    };

    if seq_to_sig.len() < 2 {
        return None;
    }

    let mut levels = None;
    if let Some(lm) = cfg.levels {
        if let Some(rp) = cfg.refine {
            refine_map(&signal, &mut seq_to_sig, &sequence, &lm, &rp);
        }
        // Extracted whether or not the boundaries moved: the residual channels
        // need them either way.
        levels = Some(extract_levels_bytes(&sequence, &lm));
    }

    if seq_to_sig.len() < 2 {
        return None;
    }

    Some(ProcessedRead {
        signal,
        seq_to_sig,
        sequence,
        levels,
    })
}

/// [`extract_levels`] over a byte sequence.
fn extract_levels_bytes(sequence: &[u8], lm: &LevelModel<'_>) -> Vec<f64> {
    let s = String::from_utf8_lossy(sequence);
    extract_levels(&s, lm.table, lm.kmer_len, Some(lm.center_idx))
}

/// Refine `seq_to_sig` in place against the expected levels, leaving it
/// untouched on any failure.
///
/// The fitted (shift, scale, drift) is deliberately **discarded**: the caller's
/// normalisation is one transform shared by every read, and re-applying a
/// per-read fit made on a window that sits largely in a constant adapter is
/// weakly identified — which destroys exactly the cross-read comparability the
/// residual channels depend on (measured in leech: level-vs-expected
/// correlation r = +0.72 keeping the read-wide gauge, +0.03 applying the fit).
fn refine_map(
    signal: &[f32],
    seq_to_sig: &mut Vec<i64>,
    sequence: &[u8],
    lm: &LevelModel<'_>,
    rp: &RefineParams,
) {
    let levels: Vec<f32> = extract_levels_bytes(sequence, lm)
        .iter()
        .map(|&v| v as f32)
        .collect();
    if levels.is_empty() || seq_to_sig.len() != levels.len() + 1 {
        return;
    }
    if seq_to_sig.iter().any(|&v| v < 0) {
        return;
    }
    let map: Vec<usize> = seq_to_sig.iter().map(|&v| v as usize).collect();
    let last = *map.last().expect("len >= 2 checked by the caller");
    if last > signal.len() || map[0] >= last {
        return;
    }

    let settings =
        RefineSettings::move_table_refinement(rp.half_bandwidth, rp.scale_iters, rp.seed);
    // The signal is already normalised, so start from identity scaling and let
    // the rough rescale derive the level-matching transform.
    let Ok(result) = refine_signal_map(&settings, signal, &map, &levels, 1.0, 0.0) else {
        return;
    };
    if result.seq_to_signal_map.len() != seq_to_sig.len() {
        return;
    }
    *seq_to_sig = result.seq_to_signal_map.iter().map(|&v| v as i64).collect();
}

// ---------------------------------------------------------------------------
// Read-wide rows
// ---------------------------------------------------------------------------

/// Per-base spans of `read`, as [`span_stats`] wants them.
fn base_spans(read: &ProcessedRead) -> Vec<[i64; 2]> {
    (0..read.n_bases())
        .map(|i| [read.seq_to_sig[i], read.seq_to_sig[i + 1]])
        .collect()
}

/// Exact integer dwell per base, straight from the map.
///
/// Taken from the map rather than from [`span_stats`]'s `dwell`, which is the
/// *clamped* sample count: a base whose span starts before the signal must
/// still report the dwell the map gives it, because that is the number the
/// model was trained on.
fn dwells(read: &ProcessedRead) -> Vec<f32> {
    (0..read.n_bases())
        .map(|i| (read.seq_to_sig[i + 1] - read.seq_to_sig[i]) as f32)
        .collect()
}

/// Centred rolling mean and population sd of `dwell` over `window` bases, the
/// edges padded by repeating the first and last value.
fn rolling_dwell(dwell: &[f32], window: usize) -> (Vec<f32>, Vec<f32>) {
    let n = dwell.len();
    if n == 0 || window == 0 {
        return (vec![0.0; n], vec![0.0; n]);
    }
    let pad = window / 2;
    let mut padded = vec![dwell[0]; n + 2 * pad];
    padded[pad..pad + n].copy_from_slice(dwell);
    for v in &mut padded[pad + n..] {
        *v = dwell[n - 1];
    }

    let mut mean = vec![0.0f32; n];
    let mut sd = vec![0.0f32; n];
    for i in 0..n {
        let win = &padded[i..i + window];
        let m = win.iter().sum::<f32>() / window as f32;
        mean[i] = m;
        let var = win.iter().map(|&v| (v - m) * (v - m)).sum::<f32>() / window as f32;
        sd[i] = var.sqrt();
    }
    (mean, sd)
}

/// Small epsilon the log and ratio rows are defined with. Part of the trained
/// definition, not a guard we may tune.
const DWELL_EPS: f32 = 1e-6;

/// Per-sample channels and per-base rows for one read, in the spec's order.
///
/// Only what the spec names is computed; the shared intermediates (dwell, the
/// rolling window, the span statistics) are computed at most once each,
/// whatever order the rows are asked for.
pub fn read_rows(read: &ProcessedRead, spec: &ChunkSpec) -> ReadRows {
    let n_bases = read.n_bases();
    let levels = read.levels.as_deref();

    // --- per-base span statistics, on demand ---------------------------
    let wants = |c: FeatureChannel| spec.feature_channels.contains(&c);
    let need_mean = wants(FeatureChannel::LevelMean)
        || wants(FeatureChannel::KmerResidual)
        || wants(FeatureChannel::KmerResidualAbs);
    let need_spans = need_mean
        || wants(FeatureChannel::LevelMedian)
        || wants(FeatureChannel::LevelStd)
        || wants(FeatureChannel::LevelRange);

    let (mut mean, mut median, mut sd, mut range) = (
        vec![0.0f32; n_bases],
        vec![0.0f32; n_bases],
        vec![0.0f32; n_bases],
        vec![0.0f32; n_bases],
    );
    if need_spans && n_bases > 0 {
        let spans = base_spans(read);
        let mut dwell_scratch = vec![0.0f32; n_bases];
        let mut scratch = SpanScratch::default();
        span_stats(
            &read.signal,
            &spans,
            SpanConfig {
                norm: Normalization::None,
                // These rows feed a network, where one NaN poisons the whole
                // forward pass; an unresolved base reads as an ordinary zero
                // and the mask, if the model has one, is the caller's.
                fill: SpanFill::Zero,
                // A reference-anchored map legitimately starts before the
                // cropped signal, and the truncated span still holds real
                // samples.
                bounds: SpanBounds::Clamp,
                median: MedianConvention::SortPartialCmp,
            },
            &mut scratch,
            SpanStatsOut::new(&mut dwell_scratch, &mut mean, &mut sd)
                .with_median(&mut median)
                .with_range(&mut range),
        );
    }

    // --- dwell rows, on demand -----------------------------------------
    let need_dwell = spec.feature_channels.iter().any(|c| {
        matches!(
            c,
            FeatureChannel::Dwell
                | FeatureChannel::DwellLog
                | FeatureChannel::DwellMean
                | FeatureChannel::DwellStd
                | FeatureChannel::DwellRatio
        )
    });
    let dwell = if need_dwell { dwells(read) } else { Vec::new() };
    let need_rolling = wants(FeatureChannel::DwellMean)
        || wants(FeatureChannel::DwellStd)
        || wants(FeatureChannel::DwellRatio);
    let (dwell_mean, dwell_sd) = if need_rolling {
        rolling_dwell(&dwell, spec.dwell_window)
    } else {
        (Vec::new(), Vec::new())
    };

    // --- k-mer rows -----------------------------------------------------
    //
    // `levels` is indexed by SEQUENCE base and the span statistics by MAPPED
    // base; under a reference anchor whose alignment ends in a non-match op
    // the second is shorter. Fitting the levels onto the mapped grid here is
    // what keeps every row the same width — leech emitted `kmer_expected` at
    // the full sequence length once, and the two derived rows at the shorter
    // one, so a single read produced rows of two different lengths and the
    // chunk cut zeroed some of them and not others.
    let expected = if spec.needs_levels() {
        let mut e = vec![0.0f32; n_bases];
        if let Some(l) = levels {
            for (dst, &src) in e.iter_mut().zip(l.iter()) {
                *dst = src as f32;
            }
        }
        e
    } else {
        Vec::new()
    };

    let feature_rows = spec
        .feature_channels
        .iter()
        .map(|&c| match c {
            FeatureChannel::Dwell => dwell.clone(),
            FeatureChannel::DwellLog => dwell.iter().map(|&d| (d + DWELL_EPS).ln()).collect(),
            FeatureChannel::DwellMean => dwell_mean.clone(),
            FeatureChannel::DwellStd => dwell_sd.clone(),
            FeatureChannel::DwellRatio => dwell
                .iter()
                .zip(&dwell_mean)
                .map(|(&d, &m)| d / (m + DWELL_EPS))
                .collect(),
            FeatureChannel::LevelMean => mean.clone(),
            FeatureChannel::LevelMedian => median.clone(),
            FeatureChannel::LevelStd => sd.clone(),
            FeatureChannel::LevelRange => range.clone(),
            FeatureChannel::KmerExpected => expected.clone(),
            FeatureChannel::KmerResidual => {
                mean.iter().zip(&expected).map(|(&o, &e)| o - e).collect()
            }
            FeatureChannel::KmerResidualAbs => mean
                .iter()
                .zip(&expected)
                .map(|(&o, &e)| (o - e).abs())
                .collect(),
        })
        .collect();

    let sample_channels = spec
        .signal_channels
        .iter()
        .map(|&c| match c {
            SignalChannel::Current => read.signal.clone(),
            SignalChannel::KmerResidual => signal_residual(read, &expected),
        })
        .collect();

    ReadRows {
        sample_channels,
        feature_rows,
    }
}

/// Per-sample residual: each sample minus the expected level of its base.
///
/// A base whose expected level is (near) zero contributes nothing: the level
/// table has no entry for it, and zero is also a legitimate level, so the two
/// are not distinguished — which is the reference implementation's rule and
/// the reason it is written down here rather than inferred.
fn signal_residual(read: &ProcessedRead, expected: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; read.signal.len()];
    for i in 0..read.n_bases() {
        let start = read.seq_to_sig[i].max(0) as usize;
        let end = (read.seq_to_sig[i + 1].max(0) as usize).min(read.signal.len());
        let e = expected.get(i).copied().unwrap_or(0.0);
        if e.abs() > 1e-12 && end > start {
            for (o, &v) in out[start..end].iter_mut().zip(&read.signal[start..end]) {
                *o = v - e;
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Chunk cutting
// ---------------------------------------------------------------------------

/// Copy one per-sample channel's window into `dst`, zero elsewhere.
///
/// `dst` is `spec.signal_len` long. When the requested window is no wider than
/// that it is placed at its own left edge (so a window that underflows the
/// signal keeps its alignment and pads on the left); when it is wider it is
/// centre-cropped down to `signal_len`.
fn place_window(src: &[f32], start: i64, end: i64, len: usize, dst: &mut [f32]) {
    dst.fill(0.0);
    if len == 0 {
        return;
    }
    let requested = (end - start).max(0) as usize;
    if requested <= len {
        let lo = start.max(0);
        let hi = end.min(src.len() as i64);
        if hi <= lo {
            return;
        }
        let off = (lo - start) as usize;
        if off >= len {
            return;
        }
        let n = ((hi - lo) as usize).min(len - off);
        dst[off..off + n].copy_from_slice(&src[lo as usize..lo as usize + n]);
    } else {
        let crop = (requested - len) / 2;
        let lo = (start + crop as i64).max(0) as usize;
        let hi = (lo + len).min(src.len());
        if hi <= lo {
            return;
        }
        dst[..hi - lo].copy_from_slice(&src[lo..hi]);
    }
}

/// Cut one chunk around `base_index`.
///
/// Returns `None` when the base has no signal boundaries (outside the map), or
/// when the spec asks for a [`SeqEncoding::SignalKmer`] tensor and the window
/// covers no base at all. A k-mer window that merely *overhangs* the sequence
/// is not a failure: it is `N`-padded, which the encoder renders as an
/// all-zero column — dropping those reads instead silently withheld a
/// prediction for every read whose alignment stops near the anchor.
pub fn cut_chunk(
    read: &ProcessedRead,
    rows: &ReadRows,
    spec: &ChunkSpec,
    base_index: i64,
) -> Option<Chunk> {
    let n_bases = read.n_bases();
    if base_index < 0 || base_index as usize >= n_bases {
        return None;
    }
    let bi = base_index as usize;

    let focus = spec
        .base_justify
        .focus(read.seq_to_sig[bi], read.seq_to_sig[bi + 1]);
    let (left, right) = spec.signal_context;
    let (sig_start, sig_end) = (focus - left, focus + right);

    // --- signal ---------------------------------------------------------
    let mut signal = vec![0.0f32; spec.signal_channels.len() * spec.signal_len];
    for (ci, channel) in rows.sample_channels.iter().enumerate() {
        let dst = &mut signal[ci * spec.signal_len..(ci + 1) * spec.signal_len];
        place_window(channel, sig_start, sig_end, spec.signal_len, dst);
    }

    // --- sequence -------------------------------------------------------
    let (sequence, sequence_rows, sequence_cols) = match spec.seq_encoding {
        SeqEncoding::None => (Vec::new(), 0, 0),
        SeqEncoding::BaseOneHot { context } => {
            let width = 2 * context + 1;
            let lo = base_index - context as i64;
            let bases: Vec<u8> = (0..width as i64)
                .map(|k| {
                    usize::try_from(lo + k)
                        .ok()
                        .and_then(|u| read.sequence.get(u).copied())
                        .unwrap_or(b'N')
                })
                .collect();
            (base_onehot(&bases), 4, width)
        }
        SeqEncoding::SignalKmer { ctx } => {
            let (map, bases) = signal_kmer_inputs(read, sig_start, sig_end, spec.signal_len, ctx)?;
            let ints = sequence_to_int(&bases);
            (
                encode_signal_kmer(&ints, &map, spec.signal_len, ctx),
                ctx.channels(),
                spec.signal_len,
            )
        }
    };

    // --- features -------------------------------------------------------
    let width = spec.feature_width();
    let mut features = vec![0.0f32; spec.feature_channels.len() * width];
    if width > 0 {
        let fs = base_index + spec.feature_offsets.0;
        let fe = base_index + spec.feature_offsets.1 + 1;
        let safe_start = fs.max(0) as usize;
        let safe_end = fe.clamp(0, n_bases as i64) as usize;
        let left_pad = (-fs).max(0) as usize;
        if safe_start < safe_end && left_pad < width {
            for (fi, row) in rows.feature_rows.iter().enumerate() {
                if safe_end > row.len() {
                    continue;
                }
                let n = (safe_end - safe_start).min(width - left_pad);
                features[fi * width + left_pad..fi * width + left_pad + n]
                    .copy_from_slice(&row[safe_start..safe_start + n]);
            }
        }
    }

    Some(Chunk {
        signal,
        sequence,
        sequence_rows,
        sequence_cols,
        features,
        base_index,
        focus_signal_pos: focus,
    })
}

/// One-hot `[4, seq.len()]`, row-major; an unrecognised base is an all-zero
/// column.
fn base_onehot(seq: &[u8]) -> Vec<f32> {
    let n = seq.len();
    let mut out = vec![0.0f32; 4 * n];
    for (i, &b) in seq.iter().enumerate() {
        let row = match b.to_ascii_uppercase() {
            b'A' => 0,
            b'C' => 1,
            b'G' => 2,
            b'T' | b'U' => 3,
            _ => continue,
        };
        out[row * n + i] = 1.0;
    }
    out
}

/// The chunk-local map and `N`-padded context sequence
/// [`encode_signal_kmer`] takes.
///
/// Keyed off the **signal** window, not the k-mer window: the two select
/// different numbers of bases — the signal window spans however many bases
/// fall inside it, a k-mer window spans exactly `2 * context + 1` — so
/// deriving these from the k-mer window disagrees with the reference on every
/// chunk (leech#186).
fn signal_kmer_inputs(
    read: &ProcessedRead,
    sig_start: i64,
    sig_end: i64,
    chunk_len: usize,
    ctx: KmerContext,
) -> Option<(Vec<i64>, Vec<u8>)> {
    if read.seq_to_sig.len() < 2 {
        return None;
    }
    let n_bases = read.n_bases();

    // The window is clamped into the signal *before* the bases are located,
    // and it is the clamped window the searches use.
    let lo = sig_start.max(0);
    let hi = sig_end.min(read.signal.len() as i64);

    // `searchsorted(map, lo, side="right") - 1`: the base whose span holds it.
    let seq_start = read.seq_to_sig.partition_point(|&v| v <= lo) as i64 - 1;
    // `searchsorted(map, hi, side="left")`: the first boundary at or past it.
    let seq_end = read.seq_to_sig.partition_point(|&v| v < hi) as i64;

    let seq_start = seq_start.max(0) as usize;
    let seq_end = (seq_end.clamp(0, read.sequence.len() as i64) as usize).min(n_bases);
    if seq_start > seq_end {
        return None;
    }

    // Offsets are against the UNCLAMPED window start, so a chunk that
    // underflows the signal still reports positions relative to its own left
    // edge.
    let mut map: Vec<i64> = read.seq_to_sig[seq_start..=seq_end]
        .iter()
        .map(|&v| v - sig_start)
        .collect();
    // The first and last bases only partly overlap the window; snap them to
    // its edges rather than letting them poke outside.
    let last = map.len() - 1;
    map[0] = 0;
    map[last] = chunk_len as i64;

    let bases = sequence_bases_with_context(&read.sequence, seq_start, seq_end - seq_start, ctx);
    Some((map, bases))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec_with(features: Vec<FeatureChannel>) -> ChunkSpec {
        ChunkSpec {
            signal_context: (2, 3),
            signal_len: 5,
            base_justify: BaseJustify::Start,
            signal_channels: vec![SignalChannel::Current],
            feature_offsets: (-1, 1),
            feature_channels: features,
            ..ChunkSpec::default()
        }
    }

    /// Four bases, spans of 2, 3, 1, 4 samples.
    fn read() -> ProcessedRead {
        ProcessedRead {
            signal: (0..10).map(|i| i as f32).collect(),
            seq_to_sig: vec![0, 2, 5, 6, 10],
            sequence: b"ACGT".to_vec(),
            levels: Some(vec![1.0, 2.0, 0.0, 4.0]),
        }
    }

    #[test]
    fn channel_names_round_trip() {
        for c in [
            FeatureChannel::Dwell,
            FeatureChannel::DwellLog,
            FeatureChannel::DwellMean,
            FeatureChannel::DwellStd,
            FeatureChannel::DwellRatio,
            FeatureChannel::LevelMean,
            FeatureChannel::LevelMedian,
            FeatureChannel::LevelStd,
            FeatureChannel::LevelRange,
            FeatureChannel::KmerExpected,
            FeatureChannel::KmerResidual,
            FeatureChannel::KmerResidualAbs,
        ] {
            assert_eq!(FeatureChannel::from_name(c.name()), Some(c));
        }
        for c in [SignalChannel::Current, SignalChannel::KmerResidual] {
            assert_eq!(SignalChannel::from_name(c.name()), Some(c));
        }
        for j in [BaseJustify::Start, BaseJustify::Center, BaseJustify::End] {
            assert_eq!(BaseJustify::from_name(j.name()), Some(j));
        }
        assert_eq!(FeatureChannel::from_name("dwell_median"), None);
    }

    /// The point of the channel list being data: the same read, asked for the
    /// same rows in a different order, is the same numbers transposed — never
    /// a different computation.
    #[test]
    fn row_order_follows_the_spec() {
        let r = read();
        let a = read_rows(
            &r,
            &spec_with(vec![FeatureChannel::Dwell, FeatureChannel::LevelMean]),
        );
        let b = read_rows(
            &r,
            &spec_with(vec![FeatureChannel::LevelMean, FeatureChannel::Dwell]),
        );
        assert_eq!(a.feature_rows[0], b.feature_rows[1]);
        assert_eq!(a.feature_rows[1], b.feature_rows[0]);
        assert_eq!(a.feature_rows[0], vec![2.0, 3.0, 1.0, 4.0]);
    }

    /// A row nobody asked for is not computed, and asking for one twice is
    /// legal (a model may repeat a channel).
    #[test]
    fn only_the_requested_rows_are_emitted() {
        let r = read();
        let rows = read_rows(&r, &spec_with(vec![FeatureChannel::DwellLog]));
        assert_eq!(rows.feature_rows.len(), 1);
        assert!((rows.feature_rows[0][0] - (2.0f32 + 1e-6).ln()).abs() < 1e-6);

        let rows = read_rows(
            &r,
            &spec_with(vec![FeatureChannel::Dwell, FeatureChannel::Dwell]),
        );
        assert_eq!(rows.feature_rows[0], rows.feature_rows[1]);
    }

    #[test]
    fn kmer_rows_are_observed_minus_expected() {
        let r = read();
        let rows = read_rows(
            &r,
            &spec_with(vec![
                FeatureChannel::LevelMean,
                FeatureChannel::KmerExpected,
                FeatureChannel::KmerResidual,
                FeatureChannel::KmerResidualAbs,
            ]),
        );
        let (mean, exp, resid, abs) = (
            &rows.feature_rows[0],
            &rows.feature_rows[1],
            &rows.feature_rows[2],
            &rows.feature_rows[3],
        );
        assert_eq!(exp, &vec![1.0, 2.0, 0.0, 4.0]);
        for i in 0..4 {
            assert!((resid[i] - (mean[i] - exp[i])).abs() < 1e-6);
            assert_eq!(abs[i], resid[i].abs());
        }
    }

    /// The rolling window pads its edges by repeating, so the first base's
    /// mean is not taken over a shorter window than the middle's.
    #[test]
    fn rolling_dwell_pads_by_repetition() {
        let (mean, sd) = rolling_dwell(&[2.0, 4.0, 6.0], 3);
        assert_eq!(
            mean,
            vec![(2.0 + 2.0 + 4.0) / 3.0, 4.0, (4.0 + 6.0 + 6.0) / 3.0]
        );
        assert!(sd[1] > 0.0);
    }

    #[test]
    fn a_window_inside_the_signal_is_copied_verbatim() {
        let r = read();
        let spec = spec_with(vec![FeatureChannel::Dwell]);
        let rows = read_rows(&r, &spec);
        // Base 1 starts at sample 2; the window is [0, 5).
        let c = cut_chunk(&r, &rows, &spec, 1).unwrap();
        assert_eq!(c.focus_signal_pos, 2);
        assert_eq!(c.signal, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    }

    /// A window that runs off either end pads with zeros and keeps its
    /// alignment: the samples that exist stay where the model expects them.
    #[test]
    fn an_overhanging_window_pads_rather_than_slides() {
        let r = read();
        let spec = spec_with(vec![FeatureChannel::Dwell]);
        let rows = read_rows(&r, &spec);
        // Base 0 starts at sample 0; the window is [-2, 3).
        let c = cut_chunk(&r, &rows, &spec, 0).unwrap();
        assert_eq!(c.signal, vec![0.0, 0.0, 0.0, 1.0, 2.0]);
        // Base 3 starts at sample 6; the window is [4, 9) — inside.
        let c = cut_chunk(&r, &rows, &spec, 3).unwrap();
        assert_eq!(c.signal, vec![4.0, 5.0, 6.0, 7.0, 8.0]);
    }

    /// A feature window that runs off the start is padded on the LEFT, so
    /// offset 0 keeps landing on the anchor base rather than sliding onto its
    /// neighbour.
    #[test]
    fn feature_padding_keeps_the_anchor_in_place() {
        let r = read();
        let spec = spec_with(vec![FeatureChannel::Dwell]);
        let rows = read_rows(&r, &spec);
        // Offsets (-1, 1) at base 0: [-1, 0, 1] -> [pad, 2, 3].
        let c = cut_chunk(&r, &rows, &spec, 0).unwrap();
        assert_eq!(c.features, vec![0.0, 2.0, 3.0]);
        // ...and at base 3: [2, 3, off-the-end] -> [1, 4, pad].
        let c = cut_chunk(&r, &rows, &spec, 3).unwrap();
        assert_eq!(c.features, vec![1.0, 4.0, 0.0]);
    }

    #[test]
    fn a_base_outside_the_map_yields_no_chunk() {
        let r = read();
        let spec = spec_with(vec![FeatureChannel::Dwell]);
        let rows = read_rows(&r, &spec);
        assert!(cut_chunk(&r, &rows, &spec, 4).is_none());
        assert!(cut_chunk(&r, &rows, &spec, -1).is_none());
    }

    /// The residual channel is the current minus the base's expected level,
    /// and a base with no level contributes nothing rather than the raw
    /// current.
    #[test]
    fn the_residual_channel_zeroes_a_base_with_no_level() {
        let r = read();
        let spec = ChunkSpec {
            signal_channels: vec![SignalChannel::Current, SignalChannel::KmerResidual],
            ..spec_with(vec![])
        };
        let rows = read_rows(&r, &spec);
        let resid = &rows.sample_channels[1];
        // Base 2 spans sample 5 only and has expected level 0.
        assert_eq!(resid[5], 0.0);
        // Base 0 spans samples 0..2 with expected level 1.
        assert_eq!(&resid[0..2], &[-1.0, 0.0]);
    }

    /// Both tensors are laid out row-major with the spec's channels as rows,
    /// which is the only layout the caller can pin against a graph.
    #[test]
    fn tensors_are_channel_major() {
        let r = read();
        let spec = ChunkSpec {
            signal_channels: vec![SignalChannel::Current, SignalChannel::KmerResidual],
            ..spec_with(vec![FeatureChannel::Dwell, FeatureChannel::LevelMean])
        };
        let rows = read_rows(&r, &spec);
        let c = cut_chunk(&r, &rows, &spec, 1).unwrap();
        assert_eq!(c.signal.len(), 2 * spec.signal_len);
        assert_eq!(c.features.len(), 2 * spec.feature_width());
        assert_eq!(&c.signal[..spec.signal_len], &rows.sample_channels[0][0..5]);
    }

    #[test]
    fn base_onehot_encodes_n_as_a_cold_column() {
        let enc = base_onehot(b"ACN");
        assert_eq!(
            enc,
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        );
    }
}
