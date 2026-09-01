// SPDX-License-Identifier: MIT

//! The charging-classifier model bundle: weights + the full feature recipe.
//!
//! `escpod classify` reads the recipe from the bundle's `metadata.json`
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
    #[serde(default)]
    waveform_model: Option<IgnoredAny>,
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
    Waveform,
}

impl Variant {
    /// The metadata key that names this variant.
    fn key(self) -> &'static str {
        match self {
            Self::Gbm => "gbm",
            Self::FeatureNn => "feature_model",
            Self::Waveform => "waveform_model",
        }
    }
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
    /// The windowed raw-signal variant's weights and input contract.
    #[serde(default)]
    waveform_model: Option<WaveformModelBlock>,
    anchor: AnchorBlock,
    /// The per-base feature space. Absent for the `waveform_model` variant,
    /// which reads a signal window rather than a column vector and declares
    /// its own geometry under `waveform_model.preprocessing`.
    #[serde(default)]
    features: Option<FeaturesBlock>,
    #[serde(default)]
    kmer_table: Option<KmerTableBlock>,
    /// Post-hoc probability calibration. Carried, not applied — see
    /// [`Calibration`].
    #[serde(default)]
    calibration: Option<CalibrationBlock>,
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
    /// Prose: whether the corpus builder also required the reference motif to
    /// map cleanly to the query. It selects which *reads* were kept, never
    /// where a window is cut, so nothing here reproduces it.
    #[serde(default)]
    query_mapping_gate: Doc,
}

/// When the bundle says the model must not be asked.
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

/// A bundle's abstain condition, resolved to something the runtime evaluates.
///
/// The rule arrives as prose (`"aligner_arm_depth == 0"`), and it is matched
/// against the forms this runtime implements rather than parsed as a general
/// expression. An unrecognised rule is a **load error**: a condition saying
/// which reads must not be scored, silently unapplied, is precisely the
/// failure this whole file is built to prevent — and it is the one that
/// actually happened (rnabioco/escapepod-rs#230).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AbstainRule {
    /// `aligner_arm_depth == 0`: the aligner placed no common-arm base at all.
    ///
    /// Not the same as a short window. Under the counting anchor such a read
    /// still *has* arm features — walked along the query — they simply do not
    /// discriminate: the bundle measures balanced accuracy 0.4993 on that
    /// population, with 100% of the uncharged library called charged. So the
    /// read is untrustworthy rather than incomplete, and the answer is no
    /// answer.
    NoAlignedArm,
    /// `no chunk, no call`: the read yields no window at the anchor.
    ///
    /// The windowed variant has no per-base feature availability to key on — a
    /// read either places its anchor inside the map or it does not — so
    /// [`AbstainRule::NoAlignedArm`] has nothing to evaluate here and must not
    /// be inherited. This is not a filter applied on top of scoring; it *is*
    /// what happens when the chunk cut fails, and naming it is what turns that
    /// into a reported no-call instead of a silent drop.
    NoChunk,
}

impl AbstainRule {
    /// Which model variants can evaluate this rule.
    ///
    /// Checked at load, because an abstain rule that *cannot* fire is
    /// indistinguishable downstream from one that never needed to: a bundle
    /// carrying the other variant's rule would score every read it says must
    /// not be scored, and report an abstain rate of zero while doing it.
    fn applies_to(self, variant: Variant) -> bool {
        match self {
            Self::NoAlignedArm => matches!(variant, Variant::Gbm | Variant::FeatureNn),
            Self::NoChunk => matches!(variant, Variant::Waveform),
        }
    }

    /// The rule as a bundle spells it.
    fn declared(self) -> &'static str {
        match self {
            Self::NoAlignedArm => "aligner_arm_depth == 0",
            Self::NoChunk => "no chunk, no call",
        }
    }
}

/// A bundle's abstain rule: what it says, and what it means here.
#[derive(Clone, Debug)]
pub struct Abstain {
    /// Verbatim, for logs and reports.
    pub rule: String,
    /// What the runtime evaluates per read.
    pub kind: AbstainRule,
}

impl Abstain {
    /// Resolve the declared rule, refusing one this runtime cannot evaluate.
    fn parse(block: &AbstainBlock) -> Result<Self> {
        // Whitespace is presentation: `a == 0` and `a==0` are one rule.
        let normalised: String = block.rule.split_whitespace().collect();
        let kind = match normalised.as_str() {
            "aligner_arm_depth==0" => AbstainRule::NoAlignedArm,
            "nochunk,nocall" => AbstainRule::NoChunk,
            _ => bail!(
                "the bundle declares an abstain rule this runtime cannot evaluate: \
                 {:?}. It names the reads the model must not be asked about, so \
                 running without it would score exactly the reads the bundle says \
                 are untrustworthy. Supported: {:?}, {:?}",
                block.rule,
                AbstainRule::NoAlignedArm.declared(),
                AbstainRule::NoChunk.declared()
            ),
        };
        Ok(Self {
            rule: block.rule.clone(),
            kind,
        })
    }
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
    /// Prose: which of the corpus's own recorded checksums this one was
    /// checked against when the bundle was built.
    #[serde(default)]
    verified_against: Doc,
}

/// The windowed raw-signal variant: weights, the three input tensors, the
/// output polarity, and the geometry the corpus was prepared with.
///
/// Everything under `preprocessing` is copied verbatim from the corpus's own
/// `prepare_config.json` rather than restated, for the reason the rest of this
/// file exists: these values *defined* the model's input, and a second
/// hand-written copy of them is a second definition of the rule.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
#[allow(dead_code)] // `Doc` fields: named, never read
struct WaveformModelBlock {
    file: String,
    sha256: String,
    /// The graph's inputs, in call order. Checked against the graph rather
    /// than trusted — see [`crate::waveform::WaveformNet::load`].
    inputs: Vec<WaveformIoBlock>,
    output: WaveformOutputBlock,
    preprocessing: WaveformPreprocessing,
    /// The rows of each tensor, by name, in the model's order.
    channels: WaveformChannels,
    /// Provenance and prose. `architecture`/`framework`/`opset` say what was
    /// trained and with what; the notes restate rules stated elsewhere.
    #[serde(default)]
    architecture: Doc,
    #[serde(default)]
    framework: Doc,
    #[serde(default)]
    opset: Doc,
    #[serde(default)]
    sequence_input_note: Doc,
    #[serde(default)]
    verification: Doc,
}

/// One declared input or output tensor.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
#[allow(dead_code)] // `Doc` fields: named, never read
struct WaveformIoBlock {
    name: String,
    /// Entries are either an integer extent or a symbolic name (`"batch"`).
    shape: Vec<serde_json::Value>,
    #[serde(default)]
    dtype: Doc,
    #[serde(default)]
    role: Doc,
    #[serde(default)]
    produced_by: Doc,
}

impl WaveformIoBlock {
    /// The declared shape with symbolic axes as `None`.
    fn dims(&self) -> Vec<Option<usize>> {
        self.shape
            .iter()
            .map(|v| v.as_u64().map(|n| n as usize))
            .collect()
    }
}

/// The output contract: one logit, and which class it is the logit *of*.
///
/// `positive_class` is the load-bearing field. The graph emits a single BCE
/// logit, and the training corpus assigned its class integers at merge time —
/// so the positive class is whichever one happened to be `1` there, not
/// whichever one `classes` lists second. Read it the obvious way and every
/// call inverts while nothing errors, which is why it is declared and matched
/// against `classes` rather than assumed.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
#[allow(dead_code)] // `Doc` fields: named, never read
struct WaveformOutputBlock {
    name: String,
    shape: Vec<serde_json::Value>,
    /// Must be one of `classes`.
    positive_class: String,
    #[serde(default)]
    convention: Doc,
    #[serde(default)]
    p_charged: Doc,
    #[serde(default)]
    why: Doc,
}

/// The corpus geometry, verbatim from `prepare_config.json`.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
#[allow(dead_code)] // `Doc` fields: named, never read
struct WaveformPreprocessing {
    signal_norm: String,
    reverse_signal: bool,
    /// Samples before and after the focus sample.
    signal_context: [i64; 2],
    signal_len: usize,
    signal_channels: usize,
    feature_start: i64,
    feature_end: i64,
    feature_width: usize,
    n_feature_channels: usize,
    seq_encoding: String,
    signal_kmer_context: [usize; 2],
    base_justify: String,
    refine_signal_map: bool,
    refine_scale_iters: i64,
    refine_half_bandwidth: usize,
    refine_kmer_center_idx: usize,
    /// Where the per-base **sequence** the model is fed comes from: each
    /// read's `MD` tag (`"md"`) or the reference FASTA (`"fasta"`).
    ///
    /// A different question from `motif_reference`, and the two read alike
    /// enough that conflating them is the whole reason this key exists
    /// (rnabioco/escapepod-rs#312). `motif_reference` is about the *anchor* —
    /// the motif is searched in the FASTA, in reference coordinates, which is
    /// correct and is what [`crate::junction_positions`] does. The **bases**
    /// handed to the k-mer lookup are a separate choice, and every corpus built
    /// so far takes them from `MD`, via pysam's `get_reference_sequence()`.
    /// Nothing in the bundle said so, "find the motif in the REFERENCE"
    /// answers only the first question, and reading it as an answer to both is
    /// what this runtime did until #306.
    ///
    /// The two are not interchangeable: every record in the shipped tRNA panel
    /// carries one `N` — an ordered degenerate position, so it cannot be
    /// resolved upstream — and levels are per 9-mer, so one unknown base blanks
    /// **nine** consecutive k-mers and moves the refined map for the whole
    /// read. Both sources score every read and error on none, which is why the
    /// declaration is checked below rather than defaulted over.
    ///
    /// Absent means `md`: optional so bundles published before this key stay
    /// readable, and `md` because that is what all of them were built with.
    #[serde(default)]
    reference_source: Option<String>,
    /// The anchor, restated here from the corpus's own config. Checked against
    /// the `anchor` block rather than merely recorded: they are two statements
    /// of one rule, written by different parts of the builder, and a bundle
    /// whose window was cut at a different base than its geometry names would
    /// score every read three bases off and validate cleanly.
    #[serde(default)]
    motif: Option<String>,
    #[serde(default)]
    motif_offset: Option<usize>,
    /// The k-mer window width the corpus recorded. It parameterises the
    /// `base_onehot` sequence encoding only; under `signal_kmer` the sequence
    /// tensor is sized by `signal_kmer_context` instead and this is unread
    /// provenance.
    #[serde(default)]
    kmer_len: Option<usize>,
    /// Prose: `focus_rule` restates in words what the geometry above says in
    /// numbers, `motif_reference` names where the *motif* was searched — the
    /// anchor only, never the bases, which is what `reference_source` above is
    /// for — and `recover_softclip_signal` was off for every corpus built so
    /// far, a `true` there being refused below rather than ignored.
    #[serde(default)]
    focus_rule: Doc,
    #[serde(default)]
    motif_reference: Doc,
    #[serde(default)]
    recover_softclip_signal: Option<bool>,
    /// The corpus's `prepare_config.json`, carried verbatim. Provenance: every
    /// value this runtime acts on is restated as a field above, and reading it
    /// from here as well would be a second parse of the same rule.
    #[serde(default)]
    prepare_config: Doc,
    #[serde(default)]
    source: Doc,
}

/// The rows of each tensor, by name, in the model's order.
///
/// This is the rule no shape check can catch: permute the feature rows and the
/// tensor still has exactly the dimensions the graph wants, every read still
/// scores, and the answers are wrong. `leech`'s own `merge_feature_channels`
/// says it plainly — reordering "silently feeds `level_mean` into the filter
/// that learned `dwell_log`".
///
/// So the runtime never assumes an order. The bundle ships one (asked of the
/// corpus builder at build time rather than transcribed), and this reads it.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
#[allow(dead_code)] // `Doc` fields: named, never read
struct WaveformChannels {
    signal: WaveformChannelList,
    features: WaveformChannelList,
    /// The sequence tensor's rows are the k-mer encoding's own, fixed by
    /// `signal_kmer_context`, so there is no order to declare — only prose.
    #[serde(default)]
    sequence: Doc,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)] // `Doc` fields: named, never read
struct WaveformChannelList {
    order: Vec<String>,
    /// Free-form: per-channel definitions, layout notes, provenance. Anything
    /// but `order` here is documentation of a row, not a row.
    #[serde(flatten)]
    prose: BTreeMap<String, IgnoredAny>,
}

/// Post-hoc probability calibration the bundle ships.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
#[allow(dead_code)] // `Doc` fields: named, never read
struct CalibrationBlock {
    method: String,
    a: f64,
    b: f64,
    #[serde(default)]
    apply: Doc,
    #[serde(default)]
    expected_calibration_error: Doc,
    #[serde(default)]
    n_val_samples: Doc,
    #[serde(default)]
    fit_on: Doc,
    #[serde(default)]
    note: Doc,
}

/// A Platt scaling of the raw logit: `sigmoid(a * logit + b)`.
///
/// **Carried, not applied.** The bundle's operating point is stated on the
/// *uncalibrated* probability — which is what the graph emits — so applying
/// this would move the probability scale out from under the very threshold
/// shipped beside it, and `cl >= 200` would no longer mean the FPR the bundle
/// measured. A caller who wants calibrated probabilities must re-derive the
/// threshold too, which is a decision, not a default (the same treatment
/// `abstain` got in rnabioco/escapepod-rs#230).
#[derive(Debug, Clone, Copy)]
pub struct Calibration {
    pub a: f64,
    pub b: f64,
}

impl Calibration {
    /// `sigmoid(a * logit + b)` — the calibrated probability of the logit's
    /// positive class.
    pub fn apply(&self, logit: f64) -> f64 {
        1.0 / (1.0 + (-(self.a * logit + self.b)).exp())
    }
}

/// Recommended call threshold shipped with the model (derived from the
/// cross-experiment evaluation, not the legacy hard-coded 200).
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
#[allow(dead_code)] // `Doc` fields: named, never read
pub struct OperatingPoint {
    /// Probability threshold for calling the positive class.
    pub probability: f64,
    /// The same threshold on the `cl` scale (`round(p * 255)`), if recorded.
    #[serde(default)]
    pub cl: Option<u8>,
    /// Where the threshold came from.
    #[serde(default)]
    pub source: Option<String>,
    /// Prose and provenance: `on` restates which probability the threshold is
    /// stated on, `caveat` says what it is not (a recommendation measured on
    /// held-out data rather than a property of the model), and `fpr`/`tpr` are
    /// what it measured there.
    #[serde(default)]
    on: Doc,
    #[serde(default)]
    caveat: Doc,
    #[serde(default)]
    fpr: Doc,
    #[serde(default)]
    tpr: Doc,
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
    /// A network over a *signal window* rather than a column vector: three
    /// tensors assembled by [`escapepod_signal::chunk`], one BCE logit out.
    #[cfg(feature = "waveform-onnx")]
    Waveform(crate::waveform_net::WaveformNet),
}

impl ChargingScorer {
    /// Short name for logs and `--info`-style output, so an operator can see
    /// which variant a directory actually holds.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Gbm(_) => "gbm",
            #[cfg(feature = "fnn-onnx")]
            Self::FeatureNn(_) => "feature-nn (onnx)",
            #[cfg(feature = "waveform-onnx")]
            Self::Waveform(_) => "waveform (onnx)",
        }
    }

    /// The GBM weights, if this is the GBM variant.
    pub fn as_gbm(&self) -> Option<&GbmModel> {
        match self {
            Self::Gbm(g) => Some(g),
            #[allow(unreachable_patterns)]
            _ => None,
        }
    }
}

/// The per-base feature space the column-scoring variants read.
///
/// Absent from a windowed bundle, which reads raw samples and has no columns
/// at all — hence [`ChargingBundle::feature_space`] rather than three fields
/// that would have to be empty and mean "not applicable". An empty offsets
/// vector is a perfectly valid feature space; "there is no feature space" is
/// not the same statement, and only one of the two can be told from a `Vec`.
#[derive(Debug, Clone)]
pub struct FeatureSpace {
    /// Feature offsets relative to the junction, recipe order.
    pub offsets: Vec<i32>,
    /// How each offset's signal span is found (`features.resolution`).
    pub span_mode: crate::anchor::SpanMode,
    /// Model input columns as `(offset index, stat index)` into the
    /// canonical `offsets × FEAT_STATS` grid, in `features.order` order.
    pub columns: Vec<(usize, usize)>,
}

/// The windowed variant's geometry, resolved into what
/// [`escapepod_signal::chunk`] takes.
///
/// Every field here is the bundle's declaration turned into the type that
/// drives the assembly — in particular [`ChunkSpec::feature_channels`], which
/// is a *list*, so a model that reads twelve rows in one order and a model
/// that reads nine in another are two values rather than two code paths.
#[derive(Debug, Clone)]
pub struct WaveformSpec {
    /// Chunk geometry and the channel lists, in the model's order.
    pub chunk: escapepod_signal::chunk::ChunkSpec,
    /// Put the samples in 5'→3' order before the window is cut.
    pub reverse_signal: bool,
    pub normalization: escapepod_signal::chunk::SignalNorm,
    /// Banded-DP refinement of the base-to-signal map, before any feature is
    /// taken. `None` means the corpus used the move table's own boundaries.
    pub refine: Option<escapepod_signal::chunk::RefineParams>,
    /// Index into [`ChargingBundle::classes`] of the class the single logit
    /// scores. `P(classes[1]) = sigmoid(logit)` when this is 1, and
    /// `1 - sigmoid(logit)` when it is 0.
    pub positive_class: usize,
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
    /// The model itself — see [`ChargingScorer`].
    pub scorer: ChargingScorer,
    pub anchor: AnchorBlock,
    /// The column feature space, for the variants that read one.
    pub features: Option<FeatureSpace>,
    /// The window geometry, for the variant that reads one.
    pub waveform: Option<WaveformSpec>,
    /// Present iff the recipe uses `resid` columns, or the window geometry
    /// needs expected levels.
    pub kmer: Option<KmerLevels>,
    pub operating_point: Option<OperatingPoint>,
    /// Shipped calibration of the raw logit. Carried, never applied — see
    /// [`Calibration`].
    pub calibration: Option<Calibration>,
    /// When the model must not be asked. Applied by
    /// [`classify_reads`](crate::classify_reads), which emits no call for a
    /// read the rule excludes.
    pub abstain: Option<Abstain>,
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

// ---------------------------------------------------------------------------
// The windowed variant's geometry
// ---------------------------------------------------------------------------

/// Resolve a declared channel list into the typed rows this runtime computes.
///
/// Every name must be one it can produce, and the list's length must match the
/// count the bundle states beside it. Both are load errors: an unknown name
/// means a row nobody here computes, and a length mismatch means the tensor
/// would be the wrong height — or, worse, the right height with the wrong rows
/// in it.
fn resolve_channels<T, N, P>(
    declared: &[String],
    declared_count: usize,
    what: &str,
    parse: P,
    known: N,
) -> Result<Vec<T>>
where
    P: Fn(&str) -> Option<T>,
    N: Fn() -> String,
{
    let rows = declared
        .iter()
        .map(|n| {
            parse(n).ok_or_else(|| {
                anyhow!(
                    "waveform_model.channels.{what}.order names the row {n:?}, which this \
                     runtime cannot compute. Known: {}",
                    known()
                )
            })
        })
        .collect::<Result<Vec<_>>>()?;
    if rows.len() != declared_count {
        bail!(
            "waveform_model declares {declared_count} {what} channels but \
             channels.{what}.order lists {}",
            rows.len()
        );
    }
    Ok(rows)
}

/// Which of the three assembled tensors a declared graph input is.
///
/// Resolved from the input's *name*, which is part of the graph's own contract
/// rather than something this runtime invents, and refused when it is not one
/// this runtime can produce. The declared `inputs` list is in call order, so
/// this is what says which tensor goes in which slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WaveformTensor {
    /// `[batch, signal_channels, signal_len]`.
    Signal,
    /// `[batch, sequence_rows, sequence_cols]`.
    Sequence,
    /// `[batch, feature_channels, feature_width]`.
    Features,
}

impl WaveformTensor {
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "signal" | "current" => Some(Self::Signal),
            "sequence" | "seq" => Some(Self::Sequence),
            "features" | "feature" => Some(Self::Features),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Signal => "signal",
            Self::Sequence => "sequence",
            Self::Features => "features",
        }
    }
}

use escapepod_signal::chunk as sigchunk;

/// Turn `waveform_model` into the [`WaveformSpec`] the assembly runs on.
///
/// Everything here is a *declaration* being resolved, never a default being
/// chosen: an unknown normalisation, an unknown sequence encoding, a channel
/// list that does not match the count beside it, a positive class that is not
/// one of `classes` — each is a load error, because each of them would
/// otherwise produce a correctly shaped tensor and a confident wrong answer.
fn waveform_spec(
    wm: &WaveformModelBlock,
    anchor: &AnchorBlock,
    classes: &[&str; 2],
) -> Result<WaveformSpec> {
    let p = &wm.preprocessing;

    // Two statements of one rule, from different parts of the builder. The
    // window is cut at `anchor.motif_offset`; if the corpus used another, the
    // model was trained on a different base and every read scores confidently
    // off-anchor.
    if let Some(m) = &p.motif
        && m != &anchor.motif
    {
        bail!(
            "anchor.motif is {:?} but waveform_model.preprocessing.motif is {m:?}; the \
             window would be cut at a motif the geometry never located",
            anchor.motif
        );
    }
    if let Some(o) = p.motif_offset
        && o != anchor.motif_offset
    {
        bail!(
            "anchor.motif_offset is {} but waveform_model.preprocessing.motif_offset is \
             {o}; the window would be cut {} base(s) from where the corpus cut it",
            anchor.motif_offset,
            (o as i64 - anchor.motif_offset as i64).abs()
        );
    }

    // Where the per-base sequence comes from, which the bundle now states
    // rather than leaving to be inferred from `motif_reference` (#312). The
    // FASTA and the `MD` tag disagree wherever the panel carries an ambiguity
    // code, and the shipped panel carries one per record, permanently — the
    // position is ordered degenerate, so 45% of reads would be silently wrong
    // under any single substituted base. Assembling from the other source
    // anyway is the failure this refuses: it scores every read, errors on
    // none, and moves the refined map for the whole read.
    match p.reference_source.as_deref() {
        None | Some("md") => {}
        Some(other) => bail!(
            "waveform_model.preprocessing.reference_source is {other:?}, but this runtime \
             assembles the per-base sequence from each read's `MD` tag (`md`) and \
             implements no other source. The two disagree wherever the reference carries \
             an ambiguity code and neither errors, so a bundle built against {other:?} is \
             refused rather than scored against `md`"
        ),
    }

    let normalization = match p.signal_norm.as_str() {
        "median_mad" => sigchunk::SignalNorm::MedianMad,
        "none" => sigchunk::SignalNorm::None,
        other => bail!(
            "waveform_model.preprocessing.signal_norm is {other:?}; this runtime \
             implements `median_mad` and `none`"
        ),
    };

    let base_justify = sigchunk::BaseJustify::from_name(&p.base_justify).ok_or_else(|| {
        anyhow!(
            "waveform_model.preprocessing.base_justify is {:?}; expected start|center|end. \
             The three differ by a shift of about half a dwell, which is a displaced \
             window rather than an error",
            p.base_justify
        )
    })?;

    let seq_encoding = match p.seq_encoding.as_str() {
        "signal_kmer" => sigchunk::SeqEncoding::SignalKmer {
            ctx: escapepod_signal::seq_encoding::KmerContext::new(
                p.signal_kmer_context[0],
                p.signal_kmer_context[1],
            ),
        },
        "base_onehot" => {
            let k = p.kmer_len.ok_or_else(|| {
                anyhow!(
                    "waveform_model.preprocessing.seq_encoding is `base_onehot` but no \
                     `kmer_len` is declared, so the window width is unknown"
                )
            })?;
            if !k.is_multiple_of(2) {
                sigchunk::SeqEncoding::BaseOneHot { context: k / 2 }
            } else {
                bail!(
                    "waveform_model.preprocessing.kmer_len is {k}, which is not \
                     `2 * context + 1` — a `base_onehot` window has no centre"
                )
            }
        }
        "none" => sigchunk::SeqEncoding::None,
        other => bail!(
            "waveform_model.preprocessing.seq_encoding is {other:?}; this runtime \
             implements `signal_kmer`, `base_onehot` and `none`"
        ),
    };

    let feature_channels = resolve_channels(
        &wm.channels.features.order,
        p.n_feature_channels,
        "features",
        sigchunk::FeatureChannel::from_name,
        || {
            [
                "dwell",
                "dwell_log",
                "dwell_mean",
                "dwell_std",
                "dwell_ratio",
                "level_mean",
                "level_median",
                "level_std",
                "level_range",
                "kmer_expected",
                "kmer_residual",
                "kmer_residual_abs",
            ]
            .join(", ")
        },
    )?;
    let signal_channels = resolve_channels(
        &wm.channels.signal.order,
        p.signal_channels,
        "signal",
        sigchunk::SignalChannel::from_name,
        || "signal, signal_kmer_residual".to_string(),
    )?;

    let chunk = sigchunk::ChunkSpec {
        signal_context: (p.signal_context[0], p.signal_context[1]),
        signal_len: p.signal_len,
        base_justify,
        signal_channels,
        seq_encoding,
        feature_offsets: (p.feature_start, p.feature_end),
        feature_channels,
        dwell_window: sigchunk::DEFAULT_DWELL_WINDOW,
    };
    if chunk.feature_width() != p.feature_width {
        bail!(
            "waveform_model.preprocessing declares feature_width {} but \
             feature_start..feature_end ({}..={}) spans {}",
            p.feature_width,
            p.feature_start,
            p.feature_end,
            chunk.feature_width()
        );
    }

    // leech extracts the expected levels only on the refinement path, so a
    // corpus built with `refine_signal_map: false` has all-zero residual rows.
    // Reproducing that silently is worse than refusing it: the model would be
    // fed a constant channel that looks like a real one.
    if !p.refine_signal_map && chunk.needs_levels() {
        bail!(
            "waveform_model.preprocessing has refine_signal_map = false, but the channel \
             order includes rows that are defined against expected k-mer levels; the \
             corpus builder does not compute levels off that path, so those rows would \
             be a constant channel indistinguishable from a real one"
        );
    }
    let refine =
        (p.refine_signal_map && p.refine_scale_iters >= 0).then_some(sigchunk::RefineParams {
            half_bandwidth: p.refine_half_bandwidth,
            scale_iters: p.refine_scale_iters as usize,
            seed: Some(sigchunk::DEFAULT_REFINE_SEED),
        });

    let positive_class = classes
        .iter()
        .position(|c| *c == wm.output.positive_class)
        .with_context(|| {
            format!(
                "waveform_model.output.positive_class is {:?}, which is not one of the \
                 bundle's classes {classes:?} — the single logit would be read as the \
                 probability of the wrong class, and every call would invert",
                wm.output.positive_class
            )
        })?;

    if p.recover_softclip_signal == Some(true) {
        bail!(
            "waveform_model.preprocessing has recover_softclip_signal = true, which \
             extends the window into signal the alignment does not cover. This runtime \
             cuts its window inside the aligned span only, so it would read a different \
             stretch of samples for every read whose anchor sits near a clip"
        );
    }

    let spec = WaveformSpec {
        chunk,
        reverse_signal: p.reverse_signal,
        normalization,
        refine,
        positive_class,
    };
    check_declared_io(wm, &spec)?;
    Ok(spec)
}

/// Check the bundle's own declared tensor shapes against the geometry it
/// declares beside them.
///
/// Two statements of one contract: `inputs[].shape` is what the graph takes,
/// and `preprocessing` is what the corpus produced. They come from different
/// places in the builder, so a mismatch between them is a real signal — and it
/// is caught here, before the graph is opened, so the message names the field
/// rather than a tensor rank.
fn check_declared_io(wm: &WaveformModelBlock, spec: &WaveformSpec) -> Result<()> {
    let mut seen = Vec::new();
    for input in &wm.inputs {
        let role = WaveformTensor::from_name(&input.name).ok_or_else(|| {
            anyhow!(
                "waveform_model.inputs names a tensor {:?} this runtime cannot produce; \
                 it assembles `signal`, `sequence` and `features`",
                input.name
            )
        })?;
        if seen.contains(&role) {
            bail!(
                "waveform_model.inputs names the {} tensor twice",
                role.name()
            );
        }
        seen.push(role);
        let want = spec.tensor_shape(role);
        let got = input.dims();
        // The batch axis is symbolic in the declaration and pinned by the
        // runtime, so only the trailing axes are compared.
        let got_tail: Vec<Option<usize>> = got.iter().skip(1).copied().collect();
        if got.len() != 3 || got_tail != want.iter().map(|&d| Some(d)).collect::<Vec<_>>() {
            bail!(
                "waveform_model.inputs[{}] declares shape {:?}, but the declared geometry \
                 makes it [batch, {}, {}]",
                input.name,
                input.shape,
                want[0],
                want[1]
            );
        }
    }
    let missing: Vec<&str> = [
        WaveformTensor::Signal,
        WaveformTensor::Sequence,
        WaveformTensor::Features,
    ]
    .iter()
    .filter(|r| !seen.contains(r) && spec.tensor_shape(**r)[0] > 0)
    .map(|r| r.name())
    .collect();
    if !missing.is_empty() {
        bail!(
            "the declared geometry produces a {} tensor that waveform_model.inputs does \
             not list, so nothing says where it goes",
            missing.join(" and a ")
        );
    }
    Ok(())
}

impl WaveformSpec {
    /// The `[rows, cols]` of one assembled tensor, batch axis excluded.
    pub fn tensor_shape(&self, role: WaveformTensor) -> [usize; 2] {
        match role {
            WaveformTensor::Signal => [self.chunk.signal_channels.len(), self.chunk.signal_len],
            WaveformTensor::Sequence => match self.chunk.seq_encoding {
                sigchunk::SeqEncoding::None => [0, 0],
                sigchunk::SeqEncoding::BaseOneHot { context } => [4, 2 * context + 1],
                sigchunk::SeqEncoding::SignalKmer { ctx } => {
                    [ctx.channels(), self.chunk.signal_len]
                }
            },
            WaveformTensor::Features => [
                self.chunk.feature_channels.len(),
                self.chunk.feature_width(),
            ],
        }
    }
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
        let declared: Vec<Variant> = [
            probe.gbm.is_some().then_some(Variant::Gbm),
            probe.feature_model.is_some().then_some(Variant::FeatureNn),
            probe.waveform_model.is_some().then_some(Variant::Waveform),
        ]
        .into_iter()
        .flatten()
        .collect();
        let variant = match declared.as_slice() {
            [one] => *one,
            [] if probe.onnx.is_some() => bail!(
                "this is the raw-signal CNN variant of {FORMAT} (a top-level `onnx` \
                 block, {:?}): it scores a raw signal window through a graph this \
                 runtime does not implement. The variants it does are `gbm`, \
                 `feature_model` and `waveform_model`",
                probe.onnx.as_deref().unwrap_or_default()
            ),
            [] => bail!(
                "bundle declares none of `gbm`, `feature_model` or `waveform_model`, so \
                 nothing here can score it — exactly one is required by {FORMAT}"
            ),
            many => bail!(
                "bundle declares {}{}; they are different models and nothing here can \
                 choose between them — ship one",
                if many.len() == 2 { "both " } else { "" },
                many.iter()
                    .map(|v| format!("`{}`", v.key()))
                    .collect::<Vec<_>>()
                    .join(" and ")
            ),
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

        // The column feature space, for the two variants that read one. A
        // windowed bundle has no `features` block at all, and inventing an
        // empty one for it would make "no feature space" and "a feature space
        // with nothing in it" the same value.
        let feature_space = match (&meta.features, variant) {
            (Some(f), _) => {
                for s in &f.stats {
                    if !FEAT_STATS.contains(&s.as_str()) {
                        bail!("recipe stat {:?} is not one of {:?}", s, FEAT_STATS);
                    }
                }
                let columns = f
                    .order
                    .iter()
                    .map(|n| parse_column(n, &f.offsets))
                    .collect::<Result<Vec<_>>>()?;
                // A rule the bundle can state that this runtime cannot
                // reproduce, and that would otherwise be scored rather than
                // refused: the columns come out looking exactly as they should.
                if let Some(fs) = &f.feature_set {
                    check_feature_set(fs)?;
                }
                Some(FeatureSpace {
                    offsets: f.offsets.clone(),
                    span_mode: crate::anchor::SpanMode::from_arm_bases(
                        f.resolution.count_arm_bases,
                    ),
                    columns,
                })
            }
            (None, Variant::Waveform) => None,
            (None, v) => bail!(
                "bundle declares `{}` but no `features` block: that variant scores a \
                 column vector and nothing here can say which columns",
                v.key()
            ),
        };

        // Refinement re-fits the base-to-signal map *before* the features are
        // taken, so a bundle built with it and scored without reads a different
        // stretch of signal for every base. The windowed variant reproduces it
        // (`waveform_model.preprocessing.refine_*`); the column variants take
        // their spans straight from the move table and cannot.
        if let Some(r) = &meta.refinement
            && !r.opts.is_empty()
            && variant != Variant::Waveform
        {
            bail!(
                "the bundle's features were computed after a refinement pass \
                 (refinement.opts: {}), which the `{}` variant does not implement — its \
                 spans come straight from the move table, so every feature would be \
                 taken over a different stretch of signal",
                r.opts.keys().cloned().collect::<Vec<_>>().join(", "),
                variant.key()
            );
        }

        let kmer_table = |needed: &str| -> Result<Option<KmerLevels>> {
            match &meta.kmer_table {
                Some(kt) => {
                    let table_path = dir.join(&kt.file);
                    verify_sha256(&table_path, &kt.sha256, "k-mer table")?;
                    let (map, k) = escapepod_signal::resquiggle::load_kmer_table(&table_path)?;
                    let center_idx = kt.center_idx.unwrap_or(k / 2);
                    if center_idx >= k {
                        bail!("kmer_table center_idx {center_idx} out of range for k={k}");
                    }
                    Ok(Some(KmerLevels { map, k, center_idx }))
                }
                None if needed.is_empty() => Ok(None),
                None => bail!(
                    "{needed}, but the bundle pins no kmer_table — the expected levels \
                     would be undefined"
                ),
            }
        };

        let (scorer, waveform, kmer) = match variant {
            Variant::Gbm | Variant::FeatureNn => {
                let f = meta
                    .features
                    .as_ref()
                    .expect("checked above for the column variants");
                let space = feature_space
                    .as_ref()
                    .expect("checked above for the column variants");
                let scorer = match variant {
                    Variant::Gbm => {
                        let g = meta.gbm.as_ref().expect("the probe saw a `gbm` block");
                        let gbm_path = dir.join(&g.file);
                        verify_sha256(&gbm_path, &g.sha256, "GBM model")?;
                        let gbm = load_gbm_model(&gbm_path)?;
                        if gbm.n_classes != 2 {
                            bail!("charging GBM must have 2 classes, has {}", gbm.n_classes);
                        }
                        if gbm.n_features != f.order.len() {
                            bail!(
                                "GBM expects {} features but the recipe orders {} columns",
                                gbm.n_features,
                                f.order.len()
                            );
                        }
                        ChargingScorer::Gbm(gbm)
                    }
                    _ => {
                        let fm = meta
                            .feature_model
                            .as_ref()
                            .expect("the probe saw a `feature_model` block");
                        Self::load_feature_model(&dir, fm, &f.order, &space.columns)?
                    }
                };
                let needs_resid = space
                    .columns
                    .iter()
                    .any(|&(_, si)| FEAT_STATS[si] == "resid");
                let kmer = kmer_table(if needs_resid {
                    "the recipe has resid columns"
                } else {
                    ""
                })?;
                (scorer, None, kmer)
            }
            Variant::Waveform => {
                let wm = meta
                    .waveform_model
                    .as_ref()
                    .expect("the probe saw a `waveform_model` block");
                let spec = waveform_spec(wm, &meta.anchor, &[&c0, &c1])?;
                let kmer = kmer_table(if spec.refine.is_some() || spec.chunk.needs_levels() {
                    "the window geometry needs expected k-mer levels"
                } else {
                    ""
                })?;
                // Two declarations of one rule: `kmer_table.center_idx` sizes
                // the levels the residual is defined against, and
                // `refine_kmer_center_idx` sizes the ones the DP is run
                // against. They index the same table, so disagreeing means the
                // corpus and the bundle describe different feature spaces.
                if let Some(k) = &kmer
                    && k.center_idx != wm.preprocessing.refine_kmer_center_idx
                {
                    bail!(
                        "kmer_table.center_idx is {} but \
                         waveform_model.preprocessing.refine_kmer_center_idx is {}; they \
                         index the same table, so one of them is not the table the \
                         corpus was built with",
                        k.center_idx,
                        wm.preprocessing.refine_kmer_center_idx
                    );
                }
                let scorer = Self::load_waveform_model(&dir, wm, &spec)?;
                (scorer, Some(spec), kmer)
            }
        };

        let abstain = meta.abstain.as_ref().map(Abstain::parse).transpose()?;
        if let Some(a) = &abstain
            && !a.kind.applies_to(variant)
        {
            bail!(
                "the bundle declares the abstain rule {:?}, which the `{}` variant has \
                 nothing to evaluate it against — it belongs to the other input space, \
                 so carrying it would report an abstain rate of zero while scoring every \
                 read the bundle says must not be scored. The rule for this variant is \
                 {:?}",
                a.rule,
                variant.key(),
                if variant == Variant::Waveform {
                    AbstainRule::NoChunk.declared()
                } else {
                    AbstainRule::NoAlignedArm.declared()
                }
            );
        }

        let calibration = match &meta.calibration {
            None => None,
            Some(c) if c.method == "platt" => Some(Calibration { a: c.a, b: c.b }),
            Some(c) => bail!(
                "the bundle ships a {:?} calibration, which this runtime cannot \
                 reproduce. Supported: `platt`",
                c.method
            ),
        };

        Ok(ChargingBundle {
            dir,
            model_id: meta.model.id,
            model_version: meta.model.version,
            classes: [c0, c1],
            scorer,
            anchor: meta.anchor,
            features: feature_space,
            waveform,
            kmer,
            operating_point: meta.operating_point,
            calibration,
            abstain,
        })
    }

    /// The column feature space, or an error naming the variant that has none.
    pub fn feature_space(&self) -> Result<&FeatureSpace> {
        self.features.as_ref().ok_or_else(|| {
            anyhow!(
                "bundle {} is the {} variant: it scores a signal window, not per-base \
                 feature columns, so it has no feature space to select from",
                self.model_id,
                self.scorer.kind()
            )
        })
    }

    /// The window geometry, or an error naming the variant that has none.
    pub fn waveform_spec(&self) -> Result<&WaveformSpec> {
        self.waveform.as_ref().ok_or_else(|| {
            anyhow!(
                "bundle {} is the {} variant: it scores per-base feature columns, not a \
                 signal window",
                self.model_id,
                self.scorer.kind()
            )
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

    /// The loaded windowed graph, or an error naming why there is none.
    ///
    /// Gated: without the feature there is no such type to return, and a
    /// bundle that would need one has already been refused at load.
    #[cfg(feature = "waveform-onnx")]
    pub fn waveform_net(&self) -> Result<&crate::waveform_net::WaveformNet> {
        if let ChargingScorer::Waveform(net) = &self.scorer {
            return Ok(net);
        }
        bail!(
            "bundle {} carries no windowed graph to run ({})",
            self.model_id,
            self.scorer.kind()
        )
    }

    /// Load and contract-check the `waveform_model` variant.
    #[cfg(feature = "waveform-onnx")]
    fn load_waveform_model(
        dir: &Path,
        wm: &WaveformModelBlock,
        spec: &WaveformSpec,
    ) -> Result<ChargingScorer> {
        let path = dir.join(&wm.file);
        verify_sha256(&path, &wm.sha256, "waveform model")?;
        Ok(ChargingScorer::Waveform(
            crate::waveform_net::WaveformNet::load(&path, spec)?,
        ))
    }

    /// Without `waveform-onnx` there is no ONNX runtime linked, so the bundle
    /// is refused with the rebuild hint rather than silently mis-scored.
    ///
    /// `escapepod-cli`'s default build enables it, so this path is reachable
    /// only from a `default-features = false` library consumer that asked for
    /// the column variants and not this one.
    #[cfg(not(feature = "waveform-onnx"))]
    fn load_waveform_model(
        _dir: &Path,
        _wm: &WaveformModelBlock,
        _spec: &WaveformSpec,
    ) -> Result<ChargingScorer> {
        bail!(
            "this bundle is the windowed raw-signal variant (`waveform_model`), but \
             escapepod-classify was built without the `waveform-onnx` feature — rebuild \
             with it enabled"
        )
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
    pub fn recipe(&self) -> Result<FeatureRecipe<'_>> {
        let space = self.feature_space()?;
        Ok(FeatureRecipe::new(
            &space.offsets,
            space.span_mode,
            self.kmer.as_ref(),
        ))
    }

    /// Select the model's input columns (as `f64`, NaN preserved) from the
    /// canonical `offsets × FEAT_STATS` feature grid.
    pub fn select_columns(&self, grid: &[f32]) -> Result<Vec<f64>> {
        Ok(self
            .feature_space()?
            .columns
            .iter()
            .map(|&(oi, si)| grid[oi * FEAT_STATS.len() + si] as f64)
            .collect())
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

    /// The windowed variant's geometry, resolved from a declaration.
    ///
    /// These run against `waveform_spec` rather than `ChargingBundle::load`,
    /// which would want a real graph and a real k-mer table on disk. What they
    /// pin is the part that has no weights in it: every rule that would
    /// otherwise be *guessed*, and whose wrong guess produces a correctly
    /// shaped tensor.
    mod waveform {
        use super::*;

        const FEATURES: &str = r#"["dwell","dwell_log","dwell_mean","dwell_std","dwell_ratio",
            "level_mean","level_median","level_std","level_range",
            "kmer_expected","kmer_residual","kmer_residual_abs"]"#;

        fn block(patch: impl Fn(&mut serde_json::Value)) -> WaveformModelBlock {
            let mut v: serde_json::Value = serde_json::from_str(&format!(
                r#"{{
                  "file": "m.onnx", "sha256": "ab", "framework": "leech",
                  "inputs": [
                    {{"name": "signal",   "shape": ["batch", 2, 390]}},
                    {{"name": "sequence", "shape": ["batch", 36, 390]}},
                    {{"name": "features", "shape": ["batch", 12, 21]}}
                  ],
                  "output": {{"name": "logits", "shape": ["batch", 1],
                              "positive_class": "uncharged"}},
                  "channels": {{
                    "signal": {{"order": ["signal", "signal_kmer_residual"]}},
                    "features": {{"order": {FEATURES}}}
                  }},
                  "preprocessing": {{
                    "signal_norm": "median_mad", "reverse_signal": true,
                    "signal_context": [90, 300], "signal_len": 390,
                    "signal_channels": 2,
                    "feature_start": 0, "feature_end": 20, "feature_width": 21,
                    "n_feature_channels": 12,
                    "seq_encoding": "signal_kmer", "signal_kmer_context": [4, 4],
                    "base_justify": "end",
                    "refine_signal_map": true, "refine_scale_iters": 2,
                    "refine_half_bandwidth": 5, "refine_kmer_center_idx": 4
                  }}
                }}"#
            ))
            .unwrap();
            patch(&mut v);
            serde_json::from_value(v).expect("the fixture block must parse")
        }

        fn anchor_block() -> AnchorBlock {
            serde_json::from_str(
                r#"{"motif": "CCAGGC", "motif_offset": 2,
                    "common_arm": "GGCTTCTTCTTGCTCTT"}"#,
            )
            .unwrap()
        }

        fn spec_of(wm: &WaveformModelBlock) -> Result<WaveformSpec> {
            waveform_spec(wm, &anchor_block(), &["uncharged", "charged"])
        }

        #[test]
        fn the_shipped_geometry_resolves() {
            let spec = spec_of(&block(|_| {})).expect("the shipped shape must load");
            assert_eq!(spec.chunk.signal_context, (90, 300));
            assert_eq!(spec.chunk.signal_len, 390);
            assert_eq!(spec.chunk.feature_width(), 21);
            assert_eq!(
                spec.chunk.base_justify,
                escapepod_signal::chunk::BaseJustify::End
            );
            assert!(spec.reverse_signal);
            // The logit's positive class is `uncharged`, i.e. classes[0], so
            // `P(classes[1])` is its complement.
            assert_eq!(spec.positive_class, 0);
            assert_eq!(
                spec.refine.map(|r| (r.half_bandwidth, r.scale_iters)),
                Some((5, 2))
            );
            assert_eq!(spec.tensor_shape(WaveformTensor::Signal), [2, 390]);
            assert_eq!(spec.tensor_shape(WaveformTensor::Sequence), [36, 390]);
            assert_eq!(spec.tensor_shape(WaveformTensor::Features), [12, 21]);
        }

        /// The channel list is *data*, and it is the one rule no shape check
        /// can catch — so it comes from the bundle and a permutation of it is
        /// a different (equally well-shaped) tensor.
        #[test]
        fn the_channel_order_comes_from_the_bundle() {
            use escapepod_signal::chunk::FeatureChannel as F;
            let spec = spec_of(&block(|_| {})).unwrap();
            assert_eq!(spec.chunk.feature_channels[0], F::Dwell);
            assert_eq!(spec.chunk.feature_channels[11], F::KmerResidualAbs);

            let swapped = spec_of(&block(|v| {
                let o = v["channels"]["features"]["order"].as_array_mut().unwrap();
                o.swap(0, 5);
            }))
            .unwrap();
            assert_eq!(swapped.chunk.feature_channels[0], F::LevelMean);
            assert_eq!(swapped.chunk.feature_channels[5], F::Dwell);
        }

        #[test]
        fn a_row_this_runtime_cannot_compute_is_refused() {
            let err = spec_of(&block(|v| {
                v["channels"]["features"]["order"][3] = "dwell_kurtosis".into();
            }))
            .unwrap_err()
            .to_string();
            assert!(err.contains("dwell_kurtosis"), "{err}");
        }

        /// Two statements of the tensor's height, from different parts of the
        /// builder. A tensor of the right height with the wrong rows in it is
        /// the failure this catches.
        #[test]
        fn a_channel_count_that_disagrees_with_the_order_is_refused() {
            let err = spec_of(&block(|v| {
                v["channels"]["features"]["order"]
                    .as_array_mut()
                    .unwrap()
                    .pop();
            }))
            .unwrap_err()
            .to_string();
            assert!(err.contains("12 features channels"), "{err}");
        }

        /// `preprocessing` restates the anchor the corpus used. Inheriting the
        /// other variant's `+3` here would place every window one base off and
        /// validate cleanly, which is exactly what the check is for.
        #[test]
        fn an_anchor_that_disagrees_with_the_geometry_is_refused() {
            let err = spec_of(&block(|v| {
                v["preprocessing"]["motif_offset"] = 3.into();
            }))
            .unwrap_err()
            .to_string();
            assert!(err.contains("motif_offset"), "{err}");

            let err = spec_of(&block(|v| {
                v["preprocessing"]["motif"] = "CCATGGC".into();
            }))
            .unwrap_err()
            .to_string();
            assert!(err.contains("anchor.motif"), "{err}");
        }

        /// The declared tensor shapes and the declared geometry are two
        /// statements of one contract; a mismatch is caught before the graph
        /// is opened, so the message names the field.
        #[test]
        fn a_declared_shape_that_disagrees_with_the_geometry_is_refused() {
            let err = spec_of(&block(|v| {
                v["inputs"][1]["shape"] = serde_json::json!(["batch", 44, 390]);
            }))
            .unwrap_err()
            .to_string();
            assert!(err.contains("sequence"), "{err}");
        }

        #[test]
        fn a_positive_class_outside_the_bundles_classes_is_refused() {
            let err = spec_of(&block(|v| {
                v["output"]["positive_class"] = "acylated".into();
            }))
            .unwrap_err()
            .to_string();
            assert!(err.contains("positive_class"), "{err}");
        }

        /// A window justified to the centre of a base sits about half a dwell
        /// from one justified to its end. Both score.
        #[test]
        fn an_unknown_justification_is_refused() {
            let err = spec_of(&block(|v| {
                v["preprocessing"]["base_justify"] = "left".into();
            }))
            .unwrap_err()
            .to_string();
            assert!(err.contains("base_justify"), "{err}");
        }

        /// leech computes the expected levels only on the refinement path, so
        /// a corpus built without it has all-zero residual rows. Reproducing
        /// that silently would feed the model a constant channel that looks
        /// like a real one.
        #[test]
        fn residual_rows_without_refinement_are_refused() {
            let err = spec_of(&block(|v| {
                v["preprocessing"]["refine_signal_map"] = false.into();
            }))
            .unwrap_err()
            .to_string();
            assert!(err.contains("constant channel"), "{err}");
        }

        #[test]
        fn a_softclip_recovering_corpus_is_refused() {
            let err = spec_of(&block(|v| {
                v["preprocessing"]["recover_softclip_signal"] = true.into();
            }))
            .unwrap_err()
            .to_string();
            assert!(err.contains("recover_softclip_signal"), "{err}");
        }

        /// The per-base sequence has two plausible sources and the bundle now
        /// says which one it was built with (#312).
        ///
        /// `fasta` is not a hypothetical value: it is what `motif_reference`
        /// says, about the *motif*, and reading that as an answer for the
        /// bases too is the mistake that cost 169 of 256 chunks their
        /// bit-exactness. A runtime that assembled from `md` anyway would score
        /// such a bundle to five decimal places and never say it had ignored
        /// the declaration — so it is refused, naming both.
        #[test]
        fn a_reference_source_this_runtime_does_not_assemble_is_refused() {
            for asked in ["fasta", "reference_fasta", "query"] {
                let err = spec_of(&block(|v| {
                    v["preprocessing"]["reference_source"] = asked.into();
                }))
                .unwrap_err()
                .to_string();
                assert!(err.contains("reference_source"), "{asked}: {err}");
                // Both sides named: what the bundle asked for, and what this
                // runtime does instead.
                assert!(err.contains(asked), "{asked}: {err}");
                assert!(err.contains("`MD`"), "{asked}: {err}");
            }
        }

        /// Absent means `md`, because that is what every bundle published
        /// before the key existed was actually built with — so the key stays
        /// optional and those bundles keep loading.
        #[test]
        fn an_absent_reference_source_means_md() {
            assert!(spec_of(&block(|_| {})).is_ok());
            assert!(
                spec_of(&block(|v| {
                    v["preprocessing"]["reference_source"] = "md".into();
                }))
                .is_ok()
            );
        }

        /// A negative iteration count means "no DP at all" on the reference
        /// side; clamping it to zero would refine here while the corpus left
        /// its boundaries alone.
        #[test]
        fn negative_scale_iters_means_no_refinement() {
            let spec = spec_of(&block(|v| {
                v["preprocessing"]["refine_scale_iters"] = (-1).into();
            }))
            .unwrap();
            assert!(spec.refine.is_none());
        }
    }

    /// An abstain rule the loaded variant cannot evaluate is worse than none:
    /// it would report a rate of zero while scoring every read the bundle says
    /// must not be scored.
    #[test]
    fn an_abstain_rule_belongs_to_one_variant() {
        assert!(AbstainRule::NoAlignedArm.applies_to(Variant::Gbm));
        assert!(AbstainRule::NoAlignedArm.applies_to(Variant::FeatureNn));
        assert!(!AbstainRule::NoAlignedArm.applies_to(Variant::Waveform));
        assert!(AbstainRule::NoChunk.applies_to(Variant::Waveform));
        assert!(!AbstainRule::NoChunk.applies_to(Variant::Gbm));
    }

    #[test]
    fn the_windowed_abstain_rule_parses() {
        let block: AbstainBlock = serde_json::from_str(r#"{"rule": "no chunk, no call"}"#).unwrap();
        assert_eq!(Abstain::parse(&block).unwrap().kind, AbstainRule::NoChunk);
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
