// SPDX-License-Identifier: MIT

//! The charging-classifier model bundle: weights + the full feature recipe.
//!
//! `escpod signal classify` reads the recipe from the bundle's `metadata.json`
//! rather than taking flags: a caller that computes the features differently
//! gets a **wrong answer, not an error**, so the definition travels with the
//! weights (rnabioco/escapepod-rs#204). The k-mer table is pinned by sha256
//! for the same reason — the residual is defined relative to it, so a
//! swapped table is a different feature space and an invalid model.
//!
//! Bundle layout (format `escapepod-charging-classifier/1`; emitted by
//! escapepod-models' `build_charging_bundle.py`):
//!
//! ```text
//! <bundle dir>/
//!   metadata.json     — the contract below
//!   <gbm file>        — GbmModel JSON (scripts/export_gbm_model.py), or
//!   <onnx file>       — the per-base-feature network (`feature_model`)
//!   <kmer table>      — tab-separated k-mer levels (optionally .gz)
//! ```
//!
//! `metadata.json` fields consumed here (every other key must still be one
//! this schema *names* — see "Unknown keys are refused" below):
//!
//! ```json
//! {
//!   "format": "escapepod-charging-classifier/1",
//!   "model": {"id": "...", "version": "..."},
//!   "classes": ["uncharged", "charged"],
//!   "gbm": {"file": "model.gbm.json", "sha256": "..."},
//!   "anchor": {"motif": "CCAGGC", "motif_offset": 3,
//!              "common_arm": "GGCTTCTTCTTGCTCTT"},
//!   "features": {"offsets": [-8, ..., 16],
//!                "stats": ["dwell", "mean", "std", "resid"],
//!                "order": ["b-8_dwell", ...]},
//!   "kmer_table": {"file": "levels.txt.gz", "sha256": "...",
//!                  "center_idx": null},
//!   "operating_point": {"probability": 0.784, "cl": 200, "source": "..."}
//! }
//! ```
//!
//! `features.order` is the model's exact input column order (names
//! `b<+/-offset>_<stat>`); the scorer's input width must equal its length.
//! `kmer_table` is required whenever a `resid` column is present.
//! `operating_point` is the recommended call threshold derived from the
//! cross-experiment evaluation — consumers should read it rather than
//! assume the legacy 200.
//!
//! # Two scorers over one feature space
//!
//! The format tag names three models: a raw-signal ONNX CNN, a
//! per-base-feature GBM (`gbm`), and a per-base-feature ONNX network
//! (`feature_model`). The last two read the **same features** — the same
//! `features` block, the same offsets, the same [`ChargingBundle::select_columns`]
//! output — and differ only in what consumes the flat vector, so they load
//! through the same path here and diverge at [`ChargingScorer`]. The
//! raw-signal CNN is a different input space and is not implemented; it is
//! recognised by its top-level `onnx` block and refused by name.
//!
//! Exactly one of `gbm` / `feature_model` must be present. Sharing a format
//! tag across variants is how a mismatch went unnoticed for three days
//! (escapepod-models `build_charging_bundle.py`), so "neither" and "both"
//! are load errors that name what they found rather than a preference for
//! whichever the code happens to check first.
//!
//! # Unknown keys are refused, not ignored
//!
//! Every block that can carry a *rule* is `deny_unknown_fields`. A key in this
//! file is something the model was built with, so accepting one the runtime
//! does not implement is how a read gets a **confident wrong answer** rather
//! than an error — the same failure as a wrong fold or a swapped k-mer table.
//! It is not hypothetical: `abstain` (which reads must not be scored at all)
//! and `features.feature_set` (whether the dwell columns were divided by the
//! read's own median before training) were both being dropped at parse time,
//! and neither is detectable downstream, because a bundle scored against the
//! wrong one produces exactly the output shape it should.
//!
//! So prose is *named* rather than allowed by omission: documenting a rule
//! stays free, introducing one does not. `provenance`, `metrics` and `caveats`
//! are the exception and are free-form — nothing under them can change what
//! the model sees, and new documentation with no natural home belongs there.
//!
//! The cost is real and deliberate: a bundle from a *newer* builder fails to
//! load rather than loading with its new rule quietly ignored. Refusing to
//! answer is the recoverable half of that trade.
//!
//! Two rules are named, refused, and worth calling out because the runtime
//! could so nearly run them: a non-empty `refinement.opts` (a banded DP that
//! re-fits the mapping before the features are taken) and a transforming
//! `features.feature_set`. A third, `abstain`, is named and *carried* rather
//! than refused — the shipped GBM bundle declares it, so refusing would ground
//! the fleet; [`ChargingBundle::abstain`] hands it to the caller instead
//! (rnabioco/escapepod-rs#230).

use crate::features::FEAT_STATS;
use crate::recipe::{FeatureRecipe, KmerLevels};
use anyhow::{Context, Result, anyhow, bail};
use escapepod_demux::{GbmModel, load_gbm_model};
use serde::Deserialize;
use serde::de::IgnoredAny;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// The one format tag this runtime implements.
const FORMAT: &str = "escapepod-charging-classifier/1";

/// A key the runtime does not read: prose for a human, or provenance.
///
/// Named rather than left to fall through, because every block that can carry
/// a *rule* denies what it does not know (see the module docs). Discards the
/// value without allocating.
///
/// A struct holding these carries `allow(dead_code)`: a field that is never
/// read is the design here, not an oversight.
type Doc = IgnoredAny;

/// Which model a bundle carries, established before the strict schema runs.
///
/// The variant has to be known first. A raw-signal CNN bundle is a different
/// input space and must be refused **by name**, not by whichever of its blocks
/// the strict schema trips over — `charging_cnn_rna004@v0.1.0` predates
/// `classes` and spells its k-mer table `path`, so a strict parse would report
/// one of those instead and send the reader looking for a corrupt file, which
/// is the exact failure the named refusal was written for. The strict schema
/// is the schema of the variants this runtime *implements*: it does not know
/// `signal`, `onnx` or `input`, and should not have to.
///
/// Deliberately lenient, therefore — its whole job is to name the variant.
#[derive(Debug, Deserialize)]
struct VariantProbe {
    format: String,
    #[serde(default)]
    gbm: Option<IgnoredAny>,
    #[serde(default)]
    feature_model: Option<IgnoredAny>,
    /// Read for the refusal message only: the raw-signal CNN variant names its
    /// graph here, at the top level, beside a `signal` block describing a
    /// window rather than columns.
    #[serde(default)]
    onnx: Option<String>,
}

/// The scorer a bundle declares — decided once, by [`VariantProbe`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Variant {
    Gbm,
    FeatureNn,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
#[allow(dead_code)] // `Doc` fields: named, never read
struct MetaFile {
    /// Checked on [`VariantProbe`]; named here so the strict schema accepts
    /// the key it was checked from.
    format: Doc,
    model: ModelBlock,
    classes: Vec<String>,
    /// The GBM variant's weights. Optional *only* because the feature-network
    /// variant exists — see the module docs; [`VariantProbe`] has already
    /// established that exactly one of the two is present.
    #[serde(default)]
    gbm: Option<FileRef>,
    /// The per-base-feature ONNX variant's weights and input contract.
    #[serde(default)]
    feature_model: Option<FeatureModelBlock>,
    anchor: AnchorBlock,
    features: FeaturesBlock,
    #[serde(default)]
    kmer_table: Option<KmerTableBlock>,
    #[serde(default)]
    operating_point: Option<OperatingPoint>,
    /// When the model must *not* be asked. Carried, not applied — see
    /// [`AbstainBlock`].
    #[serde(default)]
    abstain: Option<AbstainBlock>,
    /// Mapping refinement the corpus was built with; only the empty case is
    /// reproducible here — see [`RefinementBlock`].
    #[serde(default)]
    refinement: Option<RefinementBlock>,
    #[serde(default)]
    standardisation: Option<StandardisationBlock>,
    // Free-form by design: provenance, not contract. Nothing under these can
    // change what the model sees, so their shape is the builder's business and
    // new documentation with no natural home belongs here.
    #[serde(default)]
    provenance: Doc,
    #[serde(default)]
    metrics: Doc,
    #[serde(default)]
    caveats: Doc,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
#[allow(dead_code)] // `Doc` fields: named, never read
struct ModelBlock {
    id: String,
    #[serde(default)]
    version: Option<String>,
    #[serde(default)]
    chemistry: Doc,
    #[serde(default)]
    task: Doc,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FileRef {
    file: String,
    sha256: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
#[allow(dead_code)] // `Doc` fields: named, never read
pub struct AnchorBlock {
    pub motif: String,
    pub motif_offset: usize,
    pub common_arm: String,
    /// Prose: `method` names the geometry (`reference-junction`), the rest
    /// document what the caller has to supply and which references qualify.
    #[serde(default)]
    method: Doc,
    #[serde(default)]
    description: Doc,
    #[serde(default)]
    requires: Doc,
    #[serde(default)]
    reference_note: Doc,
}

/// When the bundle says the model must not be asked.
///
/// Parsed and carried, **not applied**: this crate scores every read it can
/// anchor, so honouring the rule is still the caller's job
/// (rnabioco/escapepod-rs#230). Naming it is the point — until now the block
/// was dropped at parse time, so the shipped GBM bundle's rule went unapplied
/// with nothing anywhere to say so.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
#[allow(dead_code)] // `Doc` fields: named, never read
pub struct AbstainBlock {
    /// The condition as the bundle states it, e.g. `aligner_arm_depth == 0`.
    pub rule: String,
    #[serde(default)]
    aligner_arm_depth_definition: Doc,
    #[serde(default)]
    emit: Doc,
    #[serde(default)]
    why: Doc,
    #[serde(default)]
    reporting: Doc,
}

/// Mapping refinement the training corpus was built with.
///
/// `opts: {}` means rough-rescale only — the per-read gauge [`crate::features`]
/// reproduces, and no DP. Anything else is a banded DP that re-fits the
/// signal-to-base mapping *before* the features are taken, which this runtime
/// does not implement; a non-empty block is refused rather than scored over
/// differently-resolved spans.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
#[allow(dead_code)] // `Doc` fields: named, never read
struct RefinementBlock {
    /// Kept by key only — the refusal names them, and reprinting the values
    /// would not make it clearer.
    #[serde(default)]
    opts: BTreeMap<String, IgnoredAny>,
    #[serde(default)]
    meaning: Doc,
}

/// Where standardisation happens (per read, and not on the raw signal).
///
/// Documentation of a rule this crate hard-codes and pins by golden vectors.
/// It is still a strict block, so *adding* a constant to it cannot pass unread.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
#[allow(dead_code)] // `Doc` fields: named, never read
struct StandardisationBlock {
    #[serde(default)]
    method: Doc,
    #[serde(default)]
    note: Doc,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
#[allow(dead_code)] // `Doc` fields: named, never read
struct FeaturesBlock {
    offsets: Vec<i32>,
    stats: Vec<String>,
    order: Vec<String>,
    #[serde(default)]
    resolution: ResolutionBlock,
    /// The named selection `order` was cut from. Checked rather than merely
    /// recorded: some names carry a per-read column transform that `order`
    /// cannot express — see [`check_feature_set`].
    #[serde(default)]
    feature_set: Option<String>,
    #[serde(default)]
    layout: Doc,
    #[serde(default)]
    anchor_offset: Doc,
    #[serde(default)]
    per_base: Doc,
    /// Prose for the per-read median/MAD gauge, which this crate hard-codes
    /// and pins by golden vectors rather than reading from here.
    #[serde(default)]
    normalisation: Doc,
    #[serde(default)]
    mask: Doc,
    #[serde(default)]
    missing: Doc,
}

/// How each offset's signal span is found — the half of the feature contract
/// that is not the offsets themselves. `escapepod-models` writes this as
/// `features.resolution`.
///
/// Asking the aligner and walking the query give different spans on precisely
/// the reads that matter, because bwa stops at the adduct. A bundle scored
/// under the wrong one gets a confident wrong answer, not an error.
///
/// Absent means the aligner path: bundles built before the counting anchor
/// existed (`charging_cnn_rna004@v0.1.0`, 2026-08-10) genuinely used it.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
#[allow(dead_code)] // `Doc` fields: named, never read
struct ResolutionBlock {
    #[serde(default)]
    count_arm_bases: u32,
    /// Prose: `arm_offsets` names the rule `count_arm_bases` parameterises
    /// (`count`), and `body_offsets`/`how`/`why` describe it.
    #[serde(default)]
    arm_offsets: Doc,
    #[serde(default)]
    body_offsets: Doc,
    #[serde(default)]
    how: Doc,
    #[serde(default)]
    why: Doc,
}

/// The per-base-feature network variant: weights plus the three rules that
/// turn `features.order`'s flat vector into the graph's input tensor.
///
/// Only the fields consumption needs are deserialised; the block also carries
/// prose (`input.fold`, `standardisation.apply`, `missing`) that documents the
/// same rules for a human reader, and `arch` / `opset` for provenance.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(not(feature = "fnn-onnx"), allow(dead_code))]
#[allow(dead_code)] // `Doc` fields: named, never read
struct FeatureModelBlock {
    file: String,
    sha256: String,
    input: FeatureModelInput,
    standardisation: FeatureModelStd,
    /// Prose and provenance: `arch` is which network was trained (`cnn`,
    /// `lstm`, …) and is deliberately not read — every arm exports the same
    /// `[batch, channel, offset]` graph, with any transpose inside it, so the
    /// architecture cannot change what this runtime feeds it.
    #[serde(default)]
    arch: Doc,
    #[serde(default)]
    opset: Doc,
    /// The output contract, which [`crate::fnn::FeatureNet`] probes the graph
    /// for rather than trusting.
    #[serde(default)]
    output: FeatureModelOutput,
    #[serde(default)]
    missing: Doc,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(not(feature = "fnn-onnx"), allow(dead_code))]
#[allow(dead_code)] // `Doc` fields: named, never read
struct FeatureModelInput {
    /// Tensor channel names: `n_val` value channels (a subset of
    /// [`FEAT_STATS`], in that order) followed by one `<stat>_observed`
    /// indicator each, in the same order.
    channels: Vec<String>,
    /// The tensor's length axis — how many base offsets the model reads.
    n_offsets: usize,
    /// Prose. `layout`/`dtype`/`shape` state in words what
    /// [`crate::fnn::FeatureNet::load`] pins against the graph itself, and
    /// `fold` restates [`check_feature_fold`] for a human reader.
    #[serde(default)]
    name: Doc,
    #[serde(default)]
    shape: Doc,
    #[serde(default)]
    dtype: Doc,
    #[serde(default)]
    layout: Doc,
    #[serde(default)]
    fold: Doc,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(not(feature = "fnn-onnx"), allow(dead_code))]
#[allow(dead_code)] // `Doc` fields: named, never read
struct FeatureModelOutput {
    #[serde(default)]
    name: Doc,
    #[serde(default)]
    shape: Doc,
    #[serde(default)]
    classes: Doc,
    #[serde(default)]
    activation: Doc,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(not(feature = "fnn-onnx"), allow(dead_code))]
#[allow(dead_code)] // `Doc` fields: named, never read
struct FeatureModelStd {
    /// One per **value** channel. The observed-mask channels are indicators
    /// and are never standardised, so this is half as long as `channels`.
    mu: Vec<f64>,
    sd: Vec<f64>,
    #[serde(default)]
    method: Doc,
    #[serde(default)]
    apply: Doc,
    #[serde(default)]
    why: Doc,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
#[allow(dead_code)] // `Doc` fields: named, never read
struct KmerTableBlock {
    file: String,
    sha256: String,
    #[serde(default)]
    center_idx: Option<usize>,
    /// Prose: `k` is read from the table itself, and the rest record where the
    /// copy in the bundle came from.
    #[serde(default)]
    k: Doc,
    #[serde(default)]
    source_path: Doc,
    #[serde(default)]
    note: Doc,
}

/// Recommended call threshold shipped with the model (derived from the
/// cross-experiment evaluation, not the legacy hard-coded 200).
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OperatingPoint {
    /// Probability threshold for calling the positive class.
    pub probability: f64,
    /// The same threshold on the `cl` scale (`round(p * 255)`), if recorded.
    #[serde(default)]
    pub cl: Option<u8>,
    /// Where the threshold came from.
    #[serde(default)]
    pub source: Option<String>,
}

/// What scores the feature vector, once it has been selected.
///
/// Both arms read the identical input — the flat `Vec<f64>` that
/// [`ChargingBundle::select_columns`] produces from the canonical
/// `offsets × FEAT_STATS` grid — so the whole model-specific part of the
/// pipeline is this enum. Which one a bundle carries is a property of the
/// bundle, never a flag.
#[derive(Debug)]
pub enum ChargingScorer {
    /// Gradient-boosted trees over the flat vector, `NaN` routed natively.
    Gbm(GbmModel),
    /// A network over the same features, folded to `[channel, offset]`,
    /// standardised, with missingness carried in explicit mask channels.
    #[cfg(feature = "fnn-onnx")]
    FeatureNn(crate::fnn::FeatureNet),
}

impl ChargingScorer {
    /// Short name for logs and `--info`-style output, so an operator can see
    /// which of the two variants a directory actually holds.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Gbm(_) => "gbm",
            #[cfg(feature = "fnn-onnx")]
            Self::FeatureNn(_) => "feature-nn (onnx)",
        }
    }

    /// The GBM weights, if this is the GBM variant.
    pub fn as_gbm(&self) -> Option<&GbmModel> {
        match self {
            Self::Gbm(g) => Some(g),
            #[cfg(feature = "fnn-onnx")]
            _ => None,
        }
    }
}

/// A loaded, hash-verified charging-classifier bundle.
#[derive(Debug)]
pub struct ChargingBundle {
    pub dir: PathBuf,
    pub model_id: String,
    pub model_version: Option<String>,
    /// Class names in probability order; the `cl` tag encodes
    /// `P(classes[1])`.
    pub classes: [String; 2],
    /// The model itself — see [`ChargingScorer`]. Both variants score the
    /// same [`select_columns`](Self::select_columns) vector.
    pub scorer: ChargingScorer,
    pub anchor: AnchorBlock,
    /// Feature offsets relative to the junction, recipe order.
    pub offsets: Vec<i32>,
    /// How each offset's signal span is found (`features.resolution`).
    pub span_mode: crate::anchor::SpanMode,
    /// Model input columns as `(offset index, stat index)` into the
    /// canonical `offsets × FEAT_STATS` grid, in `features.order` order.
    pub columns: Vec<(usize, usize)>,
    /// Present iff the recipe uses `resid` columns.
    pub kmer: Option<KmerLevels>,
    pub operating_point: Option<OperatingPoint>,
    /// When the bundle says the model must not be asked. Carried so a caller
    /// can apply it — this crate does not (rnabioco/escapepod-rs#230).
    pub abstain: Option<AbstainBlock>,
}

/// sha256 of a file, lowercase hex.
fn sha256_file(path: &Path) -> Result<String> {
    use std::fmt::Write as _;
    let bytes = std::fs::read(path)
        .with_context(|| format!("cannot read {} for hashing", path.display()))?;
    let digest = Sha256::digest(&bytes);
    let mut s = String::with_capacity(digest.len() * 2);
    for b in digest {
        write!(s, "{:02x}", b).expect("writing to a String cannot fail");
    }
    Ok(s)
}

fn verify_sha256(path: &Path, expected: &str, what: &str) -> Result<()> {
    let got = sha256_file(path)?;
    if !got.eq_ignore_ascii_case(expected) {
        bail!(
            "{} checksum mismatch for {}: expected {}, got {} — the bundle is \
             corrupt or its {} was swapped (which would silently change the \
             feature space)",
            what,
            path.display(),
            expected,
            got,
            what
        );
    }
    Ok(())
}

/// Feature-set names whose selection is *only* a selection.
///
/// The runtime reproduces one of these by choosing columns and doing nothing
/// else, which is what makes `features.order` a sufficient description of the
/// model's input.
const PLAIN_FEATURE_SETS: [&str; 6] = [
    "all",
    "no_dwell",
    "dwell_only",
    "level_only",
    "resid_only",
    "level_resid",
];

/// … and the names that also transform the columns per read.
///
/// `escapepod_models.charging.apply_feature_set` divides the dwell columns by
/// the read's own median dwell for these. The transformed columns keep their
/// plain names (`b+3_dwell` either way), so `features.order` cannot express the
/// difference and nothing downstream can detect it — a bundle built on one of
/// these and scored here would be scored on absolute dwell, confidently.
const TRANSFORMED_FEATURE_SETS: [&str; 2] = ["rel_dwell", "all_rel"];

/// Is this an offset rule (`collapse_safe`, `arm_le<N>`) — a cap on *which*
/// offsets are read, which `features.order` already reflects in full?
///
/// Matched by shape rather than against a list: the cap genuinely varies per
/// corpus (8, 12, 16, 24 so far), and a wider one is not a new rule for this
/// runtime to learn.
fn is_offset_rule(name: &str) -> bool {
    name == "collapse_safe"
        || name
            .strip_prefix("arm_le")
            .is_some_and(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()))
}

/// Refuse a `features.feature_set` whose columns are not what
/// `features.order` says they are.
///
/// The name is `<stats>`, `<stats>@<offsets>`, or a bare offset rule. Only the
/// stats half can carry a transform; the offsets half is pure selection.
fn check_feature_set(name: &str) -> Result<()> {
    let stats = name.split_once('@').map_or(name, |(s, _)| s);
    // A bare offset rule means every statistic, offsets capped — a selection
    // and nothing more.
    if stats == name && is_offset_rule(name) {
        return Ok(());
    }
    if TRANSFORMED_FEATURE_SETS.contains(&stats) {
        bail!(
            "features.feature_set is {name:?}, whose dwell columns are divided by the \
             read's own median dwell before the model sees them. This runtime selects \
             columns and applies no per-read transform, so it would score absolute \
             dwell under the very same column names — a wrong answer, not an error"
        );
    }
    if !PLAIN_FEATURE_SETS.contains(&stats) {
        bail!(
            "unknown features.feature_set {name:?}: this runtime cannot tell whether \
             {stats:?} transforms its columns per read (as {TRANSFORMED_FEATURE_SETS:?} \
             do) or only selects them (as {PLAIN_FEATURE_SETS:?} do), and the two are \
             indistinguishable downstream because the columns keep their names either way"
        );
    }
    Ok(())
}

/// Parse a feature-column name (`b<signed offset>_<stat>`) into indices
/// over `offsets` × [`FEAT_STATS`].
fn parse_column(name: &str, offsets: &[i32]) -> Result<(usize, usize)> {
    let rest = name
        .strip_prefix('b')
        .with_context(|| format!("feature name {:?} does not start with 'b'", name))?;
    let (off_str, stat) = rest
        .split_once('_')
        .with_context(|| format!("feature name {:?} has no '_<stat>' suffix", name))?;
    let off: i32 = off_str
        .parse()
        .with_context(|| format!("feature name {:?}: bad offset {:?}", name, off_str))?;
    let oi = offsets
        .iter()
        .position(|&o| o == off)
        .with_context(|| format!("feature {:?}: offset {} not in recipe offsets", name, off))?;
    let si = FEAT_STATS
        .iter()
        .position(|&s| s == stat)
        .with_context(|| format!("feature {:?}: unknown stat {:?}", name, stat))?;
    Ok((oi, si))
}

/// Check that `features.order` folds into the `feature_model`'s declared
/// `[channel, offset]` tensor, and return the value-channel count.
///
/// The declared fold is "the k-th selected column is offset `k / n_val`,
/// value channel `k % n_val`". That is only meaningful if the column names
/// actually lay out that way, so this reads them back:
///
/// * `channels` is `n_val` value channels followed by their `_observed`
///   partners, in the same order (the mask channels are indicators, which is
///   why only the value half is standardised);
/// * each block of `n_val` consecutive columns names one offset, and its
///   stats are exactly the value channels in order;
/// * offsets advance strictly across blocks.
///
/// A feature set that drops a statistic at some offsets but not others makes
/// the reshape ragged and lands here rather than in a transposed tensor.
#[cfg(feature = "fnn-onnx")]
fn check_feature_fold(
    order: &[String],
    columns: &[(usize, usize)],
    channels: &[String],
    n_off: usize,
) -> Result<usize> {
    if channels.is_empty() || !channels.len().is_multiple_of(2) {
        bail!(
            "feature_model.input.channels has {} entries; it must be n value channels \
             followed by n observed-mask channels",
            channels.len()
        );
    }
    let n_val = channels.len() / 2;
    let (values, masks) = channels.split_at(n_val);
    for (v, m) in values.iter().zip(masks) {
        if !FEAT_STATS.contains(&v.as_str()) {
            bail!(
                "feature_model value channel {:?} is not one of {:?}",
                v,
                FEAT_STATS
            );
        }
        if m != &format!("{v}_observed") {
            bail!(
                "feature_model channels are not (values..., observed...): expected \
                 {:?} to pair with {:?}, got {:?}",
                v,
                format!("{v}_observed"),
                m
            );
        }
    }
    if order.len() != n_val * n_off {
        bail!(
            "feature_model declares {} value channels x {} offsets = {} inputs, but \
             the recipe orders {} columns",
            n_val,
            n_off,
            n_val * n_off,
            order.len()
        );
    }
    let mut prev_offset: Option<usize> = None;
    for (k, name) in order.iter().enumerate() {
        let (oi, si) = columns[k];
        let want = &values[k % n_val];
        if FEAT_STATS[si] != want.as_str() {
            bail!(
                "column {k} ({name:?}) is stat {:?}, but the declared fold puts value \
                 channel {:?} there — the tensor would be built from the wrong columns",
                FEAT_STATS[si],
                want
            );
        }
        if k % n_val == 0 {
            if let Some(p) = prev_offset
                && oi <= p
            {
                bail!(
                    "column {k} ({name:?}) does not start a new, later offset block; \
                     `features.order` must be offsets-outer and ascending for the fold \
                     to be a reshape"
                );
            }
            prev_offset = Some(oi);
        } else if Some(oi) != prev_offset {
            bail!(
                "column {k} ({name:?}) belongs to a different offset than the block it \
                 folds into; the feature set does not keep its statistics uniformly \
                 across offsets"
            );
        }
    }
    Ok(n_val)
}

impl ChargingBundle {
    /// Load a bundle from its directory (or a direct `metadata.json` path),
    /// verifying every pinned checksum.
    pub fn load(path: &Path) -> Result<Self> {
        let (dir, meta_path) = if path.is_dir() {
            (path.to_path_buf(), path.join("metadata.json"))
        } else {
            (
                path.parent()
                    .map(Path::to_path_buf)
                    .unwrap_or_else(|| PathBuf::from(".")),
                path.to_path_buf(),
            )
        };
        let text = std::fs::read_to_string(&meta_path)
            .with_context(|| format!("cannot read bundle metadata {}", meta_path.display()))?;

        // Which model, before which schema — see `VariantProbe`.
        let probe: VariantProbe = serde_json::from_str(&text)
            .with_context(|| format!("cannot parse bundle metadata {}", meta_path.display()))?;
        if probe.format != FORMAT {
            bail!(
                "unsupported bundle format {:?} (expected {FORMAT})",
                probe.format
            );
        }
        let variant = match (&probe.gbm, &probe.feature_model) {
            (Some(_), Some(_)) => bail!(
                "bundle declares both `gbm` and `feature_model`; they are different \
                 models over the same features and nothing here can choose between \
                 them — ship one"
            ),
            (None, None) if probe.onnx.is_some() => bail!(
                "this is the raw-signal CNN variant of {FORMAT} (a top-level `onnx` \
                 block, {:?}): it scores a raw signal window, not the per-base feature \
                 columns, and this runtime does not implement it. The per-base-feature \
                 variants are `gbm` and `feature_model`",
                probe.onnx.as_deref().unwrap_or_default()
            ),
            (None, None) => bail!(
                "bundle declares neither `gbm` nor `feature_model`, so nothing here can \
                 score its features — one of the two is required by {FORMAT}"
            ),
            (Some(_), None) => Variant::Gbm,
            (None, Some(_)) => Variant::FeatureNn,
        };

        let meta: MetaFile = serde_json::from_str(&text).map_err(|e| {
            let msg = e.to_string();
            // Why a key nobody reads is nonetheless fatal. Worth saying at the
            // point of failure: the obvious reading of "unknown field" is that
            // the bundle is malformed, and it is not.
            let why = if msg.starts_with("unknown field") {
                ". An unrecognised key is refused rather than ignored: a key in a \
                 charging bundle is a rule the model was built with, and dropping one \
                 silently is how a read gets a confident wrong answer. Either run an \
                 escpod that implements it, or — if it is only documentation — move it \
                 under `provenance`, `metrics` or `caveats`, which are free-form"
            } else {
                ""
            };
            anyhow!(
                "cannot parse bundle metadata {}: {msg}{why}",
                meta_path.display()
            )
        })?;

        let [c0, c1]: [String; 2] = meta
            .classes
            .clone()
            .try_into()
            .map_err(|c: Vec<String>| anyhow::anyhow!("expected 2 classes, got {}", c.len()))?;

        for s in &meta.features.stats {
            if !FEAT_STATS.contains(&s.as_str()) {
                bail!("recipe stat {:?} is not one of {:?}", s, FEAT_STATS);
            }
        }
        let columns = meta
            .features
            .order
            .iter()
            .map(|n| parse_column(n, &meta.features.offsets))
            .collect::<Result<Vec<_>>>()?;

        // Two rules the bundle can state that this runtime cannot reproduce.
        // Both would otherwise be scored, not refused: the columns and the
        // spans come out looking exactly as they should.
        if let Some(fs) = &meta.features.feature_set {
            check_feature_set(fs)?;
        }
        if let Some(r) = &meta.refinement
            && !r.opts.is_empty()
        {
            bail!(
                "the bundle's features were computed after a refinement pass \
                 (refinement.opts: {}), which this runtime does not implement — its \
                 spans come straight from the move table, so every feature would be \
                 taken over a different stretch of signal",
                r.opts.keys().cloned().collect::<Vec<_>>().join(", ")
            );
        }

        // Both variants consume `columns`; the variant itself was settled by
        // `VariantProbe`, which is the only place that decision is made.
        let scorer = match variant {
            Variant::Gbm => {
                let g = meta.gbm.as_ref().expect("the probe saw a `gbm` block");
                let gbm_path = dir.join(&g.file);
                verify_sha256(&gbm_path, &g.sha256, "GBM model")?;
                let gbm = load_gbm_model(&gbm_path)?;
                if gbm.n_classes != 2 {
                    bail!("charging GBM must have 2 classes, has {}", gbm.n_classes);
                }
                if gbm.n_features != meta.features.order.len() {
                    bail!(
                        "GBM expects {} features but the recipe orders {} columns",
                        gbm.n_features,
                        meta.features.order.len()
                    );
                }
                ChargingScorer::Gbm(gbm)
            }
            Variant::FeatureNn => {
                let fm = meta
                    .feature_model
                    .as_ref()
                    .expect("the probe saw a `feature_model` block");
                Self::load_feature_model(&dir, fm, &meta.features.order, &columns)?
            }
        };

        let needs_resid = columns.iter().any(|&(_, si)| FEAT_STATS[si] == "resid");
        let kmer = match (&meta.kmer_table, needs_resid) {
            (Some(kt), _) => {
                let table_path = dir.join(&kt.file);
                verify_sha256(&table_path, &kt.sha256, "k-mer table")?;
                let (map, k) = escapepod_signal::resquiggle::load_kmer_table(&table_path)?;
                let center_idx = kt.center_idx.unwrap_or(k / 2);
                if center_idx >= k {
                    bail!(
                        "kmer_table center_idx {} out of range for k={}",
                        center_idx,
                        k
                    );
                }
                Some(KmerLevels { map, k, center_idx })
            }
            (None, true) => bail!(
                "the recipe has resid columns but the bundle pins no kmer_table — \
                 the residual would be undefined"
            ),
            (None, false) => None,
        };

        Ok(ChargingBundle {
            dir,
            model_id: meta.model.id,
            model_version: meta.model.version,
            classes: [c0, c1],
            scorer,
            anchor: meta.anchor,
            offsets: meta.features.offsets,
            span_mode: crate::anchor::SpanMode::from_arm_bases(
                meta.features.resolution.count_arm_bases,
            ),
            columns,
            kmer,
            operating_point: meta.operating_point,
            abstain: meta.abstain,
        })
    }

    /// Load and contract-check the `feature_model` variant.
    ///
    /// The fold is checked against `features.order` *before* the graph is
    /// opened, because it is the rule a consumer is most likely to get wrong
    /// and the one whose failure is invisible: fold channels-outer instead of
    /// offsets-outer and every read still scores, confidently, on a
    /// transposed input. Here the declared channels must reproduce the
    /// declared column names exactly, so a mismatch is a load error naming
    /// the column.
    #[cfg(feature = "fnn-onnx")]
    fn load_feature_model(
        dir: &Path,
        fm: &FeatureModelBlock,
        order: &[String],
        columns: &[(usize, usize)],
    ) -> Result<ChargingScorer> {
        let n_val = check_feature_fold(order, columns, &fm.input.channels, fm.input.n_offsets)?;
        let path = dir.join(&fm.file);
        verify_sha256(&path, &fm.sha256, "feature model")?;
        let net = crate::fnn::FeatureNet::load(
            &path,
            n_val,
            fm.input.n_offsets,
            &fm.standardisation.mu,
            &fm.standardisation.sd,
        )?;
        Ok(ChargingScorer::FeatureNn(net))
    }

    /// Without `fnn-onnx` there is no ONNX runtime linked, so the bundle is
    /// refused with the rebuild hint rather than silently mis-scored or
    /// reported as malformed.
    #[cfg(not(feature = "fnn-onnx"))]
    fn load_feature_model(
        _dir: &Path,
        _fm: &FeatureModelBlock,
        _order: &[String],
        _columns: &[(usize, usize)],
    ) -> Result<ChargingScorer> {
        bail!(
            "this bundle is the per-base-feature ONNX variant (`feature_model`), but \
             escapepod-classify was built without the `fnn-onnx` feature — rebuild with \
             it enabled (the `escpod` CLI enables it as part of `classify`)"
        )
    }

    /// The bundle's feature recipe, borrowed.
    ///
    /// The three fields that define the model's input space, handed over
    /// without the weights, the checksums or the operating point — see
    /// [`FeatureRecipe`] for why the split exists. Free: it borrows.
    pub fn recipe(&self) -> FeatureRecipe<'_> {
        FeatureRecipe::from(self)
    }

    /// Select the model's input columns (as `f64`, NaN preserved) from the
    /// canonical `offsets × FEAT_STATS` feature grid.
    pub fn select_columns(&self, grid: &[f32]) -> Vec<f64> {
        self.columns
            .iter()
            .map(|&(oi, si)| grid[oi * FEAT_STATS.len() + si] as f64)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The schema, exercised through `serde_json` rather than through
    /// [`ChargingBundle::load`], which would want real weights on disk.
    mod schema {
        use super::*;

        /// A minimal GBM bundle, as JSON, plus whatever `extra` splices in at
        /// the top level.
        fn meta(extra: &str) -> String {
            format!(
                r#"{{
                  "format": "escapepod-charging-classifier/1",
                  "model": {{"id": "m", "version": "0.1.0"}},
                  "classes": ["uncharged", "charged"],
                  "gbm": {{"file": "m.gbm.json", "sha256": "ab"}},
                  "anchor": {{"motif": "CCAGGC", "motif_offset": 3,
                              "common_arm": "GGCTTCTTCTTGCTCTT"}},
                  "features": {{"offsets": [0], "stats": ["mean"],
                                "order": ["b+0_mean"]}}{extra}
                }}"#
            )
        }

        fn parse(json: &str) -> Result<MetaFile, serde_json::Error> {
            serde_json::from_str(json)
        }

        #[test]
        fn the_minimum_parses() {
            assert!(parse(&meta("")).is_ok());
        }

        /// Every documentation key the current builder emits, at every level
        /// it emits one. This is the test that says what `deny_unknown_fields`
        /// costs: each of these had to be *named* to stay legal.
        #[test]
        fn the_builders_prose_is_named_and_accepted() {
            let json = meta(
                r#",
                  "abstain": {"rule": "aligner_arm_depth == 0",
                              "aligner_arm_depth_definition": "…", "emit": "…",
                              "why": "…", "reporting": "…"},
                  "refinement": {"opts": {}, "meaning": "…"},
                  "standardisation": {"method": "per-read", "note": "…"},
                  "provenance": {"seed": 0, "anything": {"nested": true}},
                  "metrics": {"auroc": 0.96, "whatever": [1, 2, 3]},
                  "caveats": ["…"]"#,
            );
            let mut v: serde_json::Value = serde_json::from_str(&json).unwrap();
            v["model"]["chemistry"] = "rna004".into();
            v["model"]["task"] = "…".into();
            v["anchor"]["method"] = "reference-junction".into();
            v["anchor"]["description"] = "…".into();
            v["anchor"]["requires"] = "…".into();
            v["anchor"]["reference_note"] = "…".into();
            for k in [
                "layout",
                "anchor_offset",
                "normalisation",
                "mask",
                "missing",
            ] {
                v["features"][k] = "…".into();
            }
            v["features"]["per_base"] = serde_json::json!({"mean": "…"});
            v["features"]["feature_set"] = "level_resid@arm_le24".into();
            v["features"]["resolution"] = serde_json::json!({
                "arm_offsets": "count", "count_arm_bases": 24,
                "how": "…", "body_offsets": "…", "why": "…"
            });
            v["kmer_table"] = serde_json::json!({
                "file": "l.txt", "sha256": "ab", "center_idx": 4, "k": 9,
                "source_path": "/scratch/l.txt", "note": "…"
            });
            let meta = parse(&v.to_string()).expect("the builder's own prose must parse");
            assert_eq!(meta.abstain.unwrap().rule, "aligner_arm_depth == 0");
        }

        /// The point of the whole change: a key nobody has taught this
        /// runtime is a rule it cannot reproduce, so it is fatal.
        #[test]
        fn an_unknown_top_level_key_is_refused() {
            let err = parse(&meta(r#", "recalibration": {"scale": 1.5}"#))
                .unwrap_err()
                .to_string();
            assert!(err.contains("unknown field `recalibration`"), "{err}");
        }

        /// …including inside a block. A new rule is far likelier to arrive as
        /// a field of `features` than as a new top-level block.
        #[test]
        fn an_unknown_key_inside_a_rule_block_is_refused() {
            let mut v: serde_json::Value = serde_json::from_str(&meta("")).unwrap();
            v["features"]["dwell_transform"] = "median".into();
            let err = parse(&v.to_string()).unwrap_err().to_string();
            assert!(err.contains("unknown field `dwell_transform`"), "{err}");
        }

        /// Provenance is the sanctioned home for shape nobody validates.
        #[test]
        fn provenance_metrics_and_caveats_stay_free_form() {
            let json = meta(
                r#",
                  "provenance": {"a": {"b": [{"c": null}]}},
                  "metrics": {"nested": {"deeply": {"whatever": 1}}},
                  "caveats": ["a", {"b": 2}]"#,
            );
            assert!(parse(&json).is_ok());
        }
    }

    /// `features.feature_set` names a rule that `features.order` cannot
    /// express, so it is checked rather than recorded.
    mod feature_set {
        use super::*;

        #[test]
        fn plain_selections_pass() {
            for name in [
                "all",
                "level_resid",
                "resid_only@arm_le8",
                "level_resid@arm_le24",
                // A bare offset rule: every statistic, offsets capped.
                "arm_le24",
                "collapse_safe",
                // A cap this runtime has never seen is still only a cap.
                "arm_le32",
            ] {
                assert!(check_feature_set(name).is_ok(), "{name} should be plain");
            }
        }

        /// The failure the check exists for: identical column names, different
        /// numbers underneath them.
        #[test]
        fn a_per_read_dwell_transform_is_refused() {
            for name in ["rel_dwell", "all_rel", "rel_dwell@arm_le8"] {
                let err = check_feature_set(name).unwrap_err().to_string();
                assert!(err.contains("median dwell"), "{name}: {err}");
            }
        }

        /// A name nobody here knows might be either kind, and the two are
        /// indistinguishable downstream — so it is refused, not assumed plain.
        #[test]
        fn an_unknown_feature_set_is_refused() {
            let err = check_feature_set("level_std@arm_le24")
                .unwrap_err()
                .to_string();
            assert!(err.contains("unknown features.feature_set"), "{err}");
            // `arm_le` prefix on the STATS half is not an offset rule either.
            assert!(check_feature_set("arm_lex").is_err());
            assert!(check_feature_set("arm_le").is_err());
        }
    }

    #[test]
    fn test_parse_column() {
        let offsets: Vec<i32> = (-8..=16).collect();
        assert_eq!(parse_column("b-8_dwell", &offsets).unwrap(), (0, 0));
        assert_eq!(parse_column("b+0_mean", &offsets).unwrap(), (8, 1));
        assert_eq!(parse_column("b+16_resid", &offsets).unwrap(), (24, 3));
        assert!(parse_column("b+17_mean", &offsets).is_err()); // not in recipe
        assert!(parse_column("b+1_bogus", &offsets).is_err());
        assert!(parse_column("x+1_mean", &offsets).is_err());
    }

    #[cfg(feature = "fnn-onnx")]
    mod fold {
        use super::*;

        /// `order` + its parsed columns for a recipe over `offsets` keeping
        /// `stats` at every offset, offsets-outer — what
        /// `escapepod_models.charging.selected_feature_names` emits.
        fn recipe(offsets: &[i32], stats: &[&str]) -> (Vec<String>, Vec<(usize, usize)>) {
            let order: Vec<String> = offsets
                .iter()
                .flat_map(|o| stats.iter().map(move |s| format!("b{o:+}_{s}")))
                .collect();
            let cols = order
                .iter()
                .map(|n| parse_column(n, offsets).unwrap())
                .collect();
            (order, cols)
        }

        fn channels(stats: &[&str]) -> Vec<String> {
            stats
                .iter()
                .map(|s| s.to_string())
                .chain(stats.iter().map(|s| format!("{s}_observed")))
                .collect()
        }

        #[test]
        fn accepts_the_full_grid() {
            let offsets: Vec<i32> = (-8..=16).collect();
            let stats = ["dwell", "mean", "std", "resid"];
            let (order, cols) = recipe(&offsets, &stats);
            assert_eq!(
                check_feature_fold(&order, &cols, &channels(&stats), 25).unwrap(),
                4
            );
        }

        #[test]
        fn accepts_a_subset_feature_set() {
            // `level_resid@arm_le*`: two statistics, fewer offsets.
            let offsets: Vec<i32> = (-8..=8).collect();
            let stats = ["mean", "resid"];
            let (order, cols) = recipe(&offsets, &stats);
            assert_eq!(
                check_feature_fold(&order, &cols, &channels(&stats), 17).unwrap(),
                2
            );
        }

        /// The failure the fold exists to prevent: channels-outer columns
        /// against an offsets-outer declaration. Nothing about the shapes is
        /// wrong — 4 x 25 either way — so only the names catch it.
        #[test]
        fn rejects_a_channels_outer_order() {
            let offsets: Vec<i32> = (-8..=16).collect();
            let stats = ["dwell", "mean", "std", "resid"];
            let order: Vec<String> = stats
                .iter()
                .flat_map(|s| offsets.iter().map(move |o| format!("b{o:+}_{s}")))
                .collect();
            let cols: Vec<_> = order
                .iter()
                .map(|n| parse_column(n, &offsets).unwrap())
                .collect();
            let err = check_feature_fold(&order, &cols, &channels(&stats), 25)
                .unwrap_err()
                .to_string();
            assert!(err.contains("declared fold"), "{err}");
        }

        #[test]
        fn rejects_a_channel_count_the_columns_cannot_fill() {
            let offsets: Vec<i32> = (-8..=16).collect();
            let stats = ["dwell", "mean", "std", "resid"];
            let (order, cols) = recipe(&offsets, &stats);
            // 4 value channels x 24 offsets != 100 columns.
            let err = check_feature_fold(&order, &cols, &channels(&stats), 24)
                .unwrap_err()
                .to_string();
            assert!(err.contains("orders 100 columns"), "{err}");
        }

        #[test]
        fn rejects_unpaired_mask_channels() {
            let offsets: Vec<i32> = (-1..=1).collect();
            let stats = ["mean", "resid"];
            let (order, cols) = recipe(&offsets, &stats);
            let ch = vec![
                "mean".into(),
                "resid".into(),
                "resid_observed".into(),
                "mean_observed".into(),
            ];
            let err = check_feature_fold(&order, &cols, &ch, 3)
                .unwrap_err()
                .to_string();
            assert!(err.contains("(values..., observed...)"), "{err}");
        }

        #[test]
        fn rejects_a_ragged_selection() {
            // `resid` present at +0 only: the reshape would silently slide.
            let offsets: Vec<i32> = (-1..=1).collect();
            let order: Vec<String> = ["b-1_mean", "b-1_std", "b+0_mean", "b+0_resid"]
                .iter()
                .map(|s| s.to_string())
                .collect();
            let cols: Vec<_> = order
                .iter()
                .map(|n| parse_column(n, &offsets).unwrap())
                .collect();
            let ch = channels(&["mean", "std"]);
            let err = check_feature_fold(&order, &cols, &ch, 2)
                .unwrap_err()
                .to_string();
            assert!(err.contains("declared fold"), "{err}");
        }
    }
}
