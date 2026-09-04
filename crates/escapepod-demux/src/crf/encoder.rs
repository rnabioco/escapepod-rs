//! The CTC-CRF encoder: ONNX inference through tract, plus the signal
//! preprocessing the trained weights expect.
//!
//! Available with the `crf-decode` feature. The caller supplies the ONNX file
//! and its `metadata.json` sidecar at runtime (produced by escapepod-models'
//! `scripts/nbc/export_crf_to_onnx.py`); this crate ships only the code that
//! consumes them.
//!
//! # Why there is a sidecar
//!
//! The standardisation constants are not recoverable from the weights or from
//! bonito's `config.toml`. That file carries SeqTagger's `mean = 80.876 /
//! stdev = 17.270`, but the training script ignores it and standardises with
//! the corpus's own `mean = 62.405 / stdev = 10.232`. Reading the config is the
//! obvious implementation, and it is wrong by ~1.8 sigma of shift and 1.7x of
//! scale — the model still decodes, just worse, with nothing to indicate it.
//! So the constants travel in the sidecar and this loader refuses to guess at
//! them.
//!
//! # Shape contract
//!
//! ```text
//! input   [batch, 1, chunk]            f32, batch-major, standardised raw pA
//! output  [chunk / stride, batch, n_score]  f32, TIME-major
//! ```
//!
//! The time-major output is the trap worth naming: [`crate::adapter_cnn`] takes
//! batch-major `[B, 2, L]` and hard-rejects anything else, so the two loaders
//! cannot share a probe. [`CrfEncoder::load`] runs its own dummy forward pass
//! at load time for exactly the reason the boundary CNN does — a wrong-shaped
//! model that is only caught per-read tends to produce silently empty output
//! rather than an error.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::Deserialize;
use thiserror::Error;
use tract_onnx::prelude::*;
use tract_onnx::tract_core::framework::Framework;
use tract_onnx::tract_core::model::TypedRunnableModel;

use super::lattice::{Backend, CrfLayout, CrfScratch, decode_with, decode_with_refs};
use super::refchain::{RefChains, ScoredDecode};

/// Errors from loading or running the CRF encoder.
#[derive(Debug, Error)]
pub enum CrfError {
    #[error("failed to load ONNX model: {0}")]
    Load(String),
    #[error("failed to run inference: {0}")]
    Run(String),
    #[error("failed to read metadata sidecar {path}: {source}")]
    Metadata {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("malformed metadata sidecar {path}: {source}")]
    MetadataParse {
        path: PathBuf,
        source: serde_json::Error,
    },
    #[error(
        "unexpected output shape: expected time-major [{t}, {batch}, {n_score}], got {got:?}. \
         The boundary-CNN export is batch-major and will fail here — check this is the CRF encoder."
    )]
    BadShape {
        got: Vec<usize>,
        t: usize,
        batch: usize,
        n_score: usize,
    },
    #[error(
        "signal.window says {window:?} but signal.anchor says {anchor:?}. These describe the \
         same rule and one of them is wrong; fix the export rather than guessing which."
    )]
    AnchorDisagreesWithWindow {
        window: String,
        anchor: &'static str,
    },
    #[error(
        "this bundle anchors its window on the read end but declares boundary.method \
         {method:?}. A read-end model consumes no boundary detector, and a bundle that pins \
         one will be run against a detector's adapter_end -- the far end of the molecule. \
         Record the detector under `built_beside` if it is provenance."
    )]
    BoundaryPinnedOnReadEnd { method: String },
    #[error("lattice geometry {n_base}^{state_len} is not representable")]
    BadGeometry { n_base: usize, state_len: usize },
    #[error("alphabet has {got} symbols, expected {expected} (blank plus one per base)")]
    BadAlphabet { got: usize, expected: usize },
    #[error("decode failed: {0}")]
    Decode(String),
    #[error("reference panel is not scorable against this model: {0}")]
    RefChain(#[from] super::refchain::RefChainError),
}

/// The `metadata.json` sidecar that travels with a CRF encoder export.
///
/// Unknown fields are ignored **at this level** on purpose: the sidecar also
/// carries provenance and human-facing notes that a consumer has no business
/// depending on, and `built_beside` (a detector recorded for history, not
/// pinned) is exactly that.
///
/// Blocks that carry *rules* are closed instead — see [`SignalSpec`]. The
/// distinction is which way the failure goes: an ignored provenance key costs
/// nothing, while an ignored rule produces a confident wrong answer.
#[derive(Debug, Clone, Deserialize)]
pub struct CrfMetadata {
    /// Filename of the ONNX graph, relative to the sidecar.
    #[serde(default = "default_onnx_name")]
    pub onnx: String,
    pub standardisation: Standardisation,
    pub signal: SignalSpec,
    pub crf: CrfSpec,
    /// References the decoded sequence is matched against, if the bundle
    /// carries them.
    ///
    /// A CRF emits sequence, not a class index, so it is useless without a
    /// reference set — but unlike the fingerprint heads it has nowhere natural
    /// to keep one, so callers were passing it separately every time. Shipping
    /// it here also fixes the trimming at export rather than at each call site:
    /// the model emits `target[state_len:]`, so anyone writing their own CSV
    /// can silently supply full-length targets, which inflates every distance
    /// and compresses the confidence margin (escapepod-models#36). A
    /// caller-supplied list still overrides this.
    #[serde(default)]
    pub barcodes: Option<Vec<BarcodeEntry>>,
    /// The boundary detector this model is calibrated against, if the bundle
    /// pins one.
    ///
    /// The training window is defined relative to that detector's
    /// `adapter_end`, so pairing the model with a different detector silently
    /// degrades it. That coupling is a property of the model and belongs with
    /// it, not in the user's shell history.
    #[serde(default)]
    pub boundary: Option<BoundarySpec>,
    /// Registry identity, for `--info` and for logging what actually ran.
    #[serde(default)]
    pub model: Option<ModelIdent>,
    /// Published performance, carried verbatim from the model's provenance.
    ///
    /// Deliberately untyped: metric names differ per model kind and per
    /// evaluation, and a schema here would either lag the provenance or force
    /// every producer through this crate. `--info` pretty-prints whatever is
    /// present rather than interpreting it.
    #[serde(default)]
    pub metrics: Option<serde_json::Value>,
    /// Per-run override of the bundle's declared `boundary.margin`, set by
    /// `demux --boundary-margin`. Never read from the sidecar — the bundle
    /// states the contract, this is the operator overruling it.
    #[serde(skip)]
    margin_override: Option<usize>,
    /// Per-run override of the bundle's declared `boundary.clamp_max_shift`,
    /// set by `demux --clamp-max-shift`. Never read from the sidecar.
    #[serde(skip)]
    clamp_override: Option<usize>,
}

/// Who this model is, for provenance in logs and `--info`.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelIdent {
    pub id: String,
    #[serde(default)]
    pub version: Option<String>,
    #[serde(default)]
    pub chemistry: Option<String>,
    /// What the model does and anything a user needs to know before trusting
    /// its output.
    #[serde(default)]
    pub notes: Option<String>,
    /// Caveats worth surfacing every time, e.g. a confounded pilot.
    #[serde(default)]
    pub caveats: Vec<String>,
}

/// One reference a decoded sequence can be matched to.
#[derive(Debug, Clone, Deserialize)]
pub struct BarcodeEntry {
    pub name: String,
    /// The sequence the model EMITS, not the training target.
    pub sequence: String,
}

/// The boundary detector a CRF bundle is calibrated against.
#[derive(Debug, Clone, Deserialize)]
pub struct BoundarySpec {
    /// Detection method the model expects (`cnn` or `llr`).
    pub method: String,
    /// ONNX graph for `cnn`, relative to the sidecar. Absent means the bundle
    /// names a method but does not ship the weights.
    #[serde(default)]
    pub onnx: Option<String>,
    /// Registry id of that model, for provenance in logs.
    #[serde(default)]
    pub model_id: Option<String>,
    /// Lowercase-hex sha256 of the pinned ONNX, as built into the bundle.
    /// The pinned copy is the one bundle file the registry manifest does not
    /// hash, so this declaration is what makes it verifiable; the runtime
    /// checks it before trusting the pin.
    #[serde(default)]
    pub sha256: Option<String>,
    /// The input tensor the pinned model consumes. Absent in bundles built
    /// before escapepod-models shipped the contract; the runtime then assumes
    /// the legacy rna004 geometry, which is what those models trained with.
    ///
    /// Lives in [`crate::crf`] rather than here so the pin plumbing can name
    /// it in builds without `crf-decode`.
    #[serde(default)]
    pub input: Option<crate::crf::BoundaryInputSpec>,
    /// Samples of `adapter_end` beyond `signal.chunk` a read needs before it is
    /// decoded. Absent uses [`BOUNDARY_MARGIN`].
    ///
    /// This belongs to the bundle for the same reason `input` does: it is a
    /// property of how the corpus was framed, not of the caller. The exporter
    /// knows the filter its `extract_chunks` applied (rna004 nbc16 used
    /// `adapter_end > chunk + 200`) and is the only party that can state it
    /// without guessing. A bundle that declares 0 is asserting its model was
    /// trained to tolerate a window reaching the read's opening samples.
    #[serde(default)]
    pub margin: Option<usize>,
    /// Largest `chunk - adapter_end` for which a read whose adapter ends before
    /// `chunk` is decoded from `[0, chunk]` rather than refused. Absent or 0
    /// disables clamping.
    ///
    /// Belongs to the bundle for the same reason `margin` does: how far the
    /// window can slide before the decode stops meaning anything is a property
    /// of the trained model, measurable once at export against held-out reads,
    /// and not something an operator can derive per run.
    #[serde(default)]
    pub clamp_max_shift: Option<usize>,
}

fn default_onnx_name() -> String {
    "crf_encoder.onnx".to_string()
}

#[derive(Debug, Clone, Copy, Deserialize)]
pub struct Standardisation {
    pub mean: f32,
    pub stdev: f32,
}

/// What the model's `chunk`-sample window is measured back from.
///
/// Not a cosmetic distinction: the two anchors sit at opposite ends of the
/// molecule. RNA004 translocates 3'->5', so a 3'-adapter barcode is found by
/// windowing back from the boundary detector's `adapter_end`, while a 5' index
/// goes through the pore *last* and its adapter is simply where the signal
/// stops. Reading a `read_end` model at `adapter_end` decodes the far end of
/// the molecule and returns exactly the output shape it should.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Anchor {
    /// `[adapter_end - chunk, adapter_end]`, off the boundary detector.
    AdapterEnd,
    /// `[len - chunk, len]`, off the read's last sample. Consumes no detector.
    ReadEnd,
}

impl Anchor {
    /// The token this anchor is spelled with in a `signal.window` string.
    const fn token(self) -> &'static str {
        match self {
            Anchor::AdapterEnd => "adapter_end",
            Anchor::ReadEnd => "read_end",
        }
    }
}

/// Where the model's input window sits and how wide it is.
///
/// `deny_unknown_fields`, unlike [`CrfMetadata`] itself. Every key in this
/// block is a *rule the model was built with*, so accepting one this runtime
/// does not implement is how a read gets a confident wrong answer rather than
/// an error — the doctrine already written down for the charging bundle
/// (`escapepod-classify::bundle`). It is not hypothetical: every shipped
/// bundle declares `window`, `barcode_crf_fdx4_rna004@v0.1.1` declares
/// `anchor`, and both were being dropped here — which windowed a 5'-index
/// model onto the 3' adapter.
///
/// The cost is deliberate: a bundle from a newer builder fails to load rather
/// than loading with its new rule silently ignored. Provenance keys have no
/// business in this block; they belong at the top level, which stays open.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SignalSpec {
    /// Samples fed to the encoder per read.
    pub chunk: usize,
    /// Samples per output timestep.
    pub stride: usize,
    /// What the window is measured back from, as the bundle spells it.
    ///
    /// Read it through [`Self::anchor`], never directly: absent means
    /// [`Anchor::AdapterEnd`] (every bundle predating this key was built that
    /// way and means exactly that), and a caller that matches on the `Option`
    /// itself is one `None` arm away from reinstating the bug this key exists
    /// to fix. Hence the name — there is no `.anchor` field to misread.
    #[serde(default, rename = "anchor")]
    pub declared_anchor: Option<Anchor>,
    /// The window in the exporter's own words, e.g.
    /// `"[read_end - chunk, read_end]"`. Documentation, not the contract —
    /// [`Self::anchor`] is — but it is cross-checked against the anchor at
    /// load, because the one time these disagreed the prose was right and the
    /// machine-readable half was wrong.
    #[serde(default)]
    pub window: Option<String>,
}

impl SignalSpec {
    /// The anchor in effect: what the bundle declares, else
    /// [`Anchor::AdapterEnd`].
    pub fn anchor(&self) -> Anchor {
        self.declared_anchor.unwrap_or(Anchor::AdapterEnd)
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct CrfSpec {
    pub state_len: usize,
    pub n_base: usize,
    /// Blank symbol first, then one per base — e.g. `["N", "A", "C", "G", "T"]`.
    pub alphabet: Vec<String>,
}

impl CrfMetadata {
    /// Read a sidecar from disk.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, CrfError> {
        let path = path.as_ref();
        let raw = std::fs::read_to_string(path).map_err(|source| CrfError::Metadata {
            path: path.to_path_buf(),
            source,
        })?;
        let meta: Self = serde_json::from_str(&raw).map_err(|source| CrfError::MetadataParse {
            path: path.to_path_buf(),
            source,
        })?;
        meta.validate()?;
        Ok(meta)
    }

    /// Cross-checks between blocks that parse fine on their own.
    ///
    /// `deny_unknown_fields` on [`SignalSpec`] catches a rule this runtime does
    /// not implement; this catches two the runtime *does* implement and that
    /// contradict each other. Both are the same defect seen from different
    /// sides — a bundle asserting a window it does not have — and both were
    /// live in `barcode_crf_fdx4_rna004@v0.1.0`.
    pub fn validate(&self) -> Result<(), CrfError> {
        let anchor = self.signal.anchor();
        if let Some(window) = &self.signal.window {
            let other = match anchor {
                Anchor::AdapterEnd => Anchor::ReadEnd,
                Anchor::ReadEnd => Anchor::AdapterEnd,
            };
            // Neither token is a substring of the other, so "names mine and not
            // the other one" is decidable by two `contains`.
            if !window.contains(anchor.token()) || window.contains(other.token()) {
                return Err(CrfError::AnchorDisagreesWithWindow {
                    window: window.clone(),
                    anchor: anchor.token(),
                });
            }
        }
        if anchor == Anchor::ReadEnd
            && let Some(b) = &self.boundary
        {
            return Err(CrfError::BoundaryPinnedOnReadEnd {
                method: b.method.clone(),
            });
        }
        Ok(())
    }

    /// Whether this model's window is placed by a boundary detector.
    ///
    /// A run whose every head answers `false` needs no detector at all, and
    /// must not be given one: a detector's `signal_decode_bound` truncates each
    /// read to its own leading window, which moves where "the read end" is.
    pub fn needs_boundary(&self) -> bool {
        self.signal.anchor() == Anchor::AdapterEnd
    }

    /// Whether this model needs the read decoded to its last sample.
    pub fn needs_full_read(&self) -> bool {
        self.signal.anchor() == Anchor::ReadEnd
    }

    /// Blank-first alphabet as single bytes, e.g. `b"NACGT"`.
    pub(super) fn alphabet_bytes(&self) -> Vec<u8> {
        self.crf
            .alphabet
            .iter()
            .map(|s| s.as_bytes().first().copied().unwrap_or(b'N'))
            .collect()
    }

    /// Lattice geometry implied by the sidecar, validated against the alphabet.
    pub(super) fn layout(&self) -> Result<CrfLayout, CrfError> {
        let layout =
            CrfLayout::new(self.crf.n_base, self.crf.state_len).ok_or(CrfError::BadGeometry {
                n_base: self.crf.n_base,
                state_len: self.crf.state_len,
            })?;
        if self.crf.alphabet.len() != layout.n_edges {
            return Err(CrfError::BadAlphabet {
                got: self.crf.alphabet.len(),
                expected: layout.n_edges,
            });
        }
        Ok(layout)
    }

    /// Number of encoder timesteps per read.
    pub fn t_len(&self) -> usize {
        self.signal.chunk / self.signal.stride
    }

    /// Standardise a `chunk`-sample window of raw pA in place, as the training
    /// corpus was standardised.
    fn standardise(&self, window: &[f32], out: &mut Vec<f32>) {
        let (mean, stdev) = (self.standardisation.mean, self.standardisation.stdev);
        out.clear();
        out.extend(window.iter().map(|&v| (v - mean) / stdev));
    }

    /// Slice and standardise the model input for one read.
    ///
    /// The window is [`Self::window`] of **raw calibrated pA** — not ADC
    /// counts, not MAD-normalised signal — matching `extract_chunks.py`. Under
    /// [`Anchor::ReadEnd`] `adapter_end` is ignored and `signal_pa` must run to
    /// the read's last sample. Returns `None` when the read cannot supply a full
    /// window, which is also how `adapter_end == 0` (the boundary detector's
    /// overloaded "no adapter" / "too short" / "inference failed" sentinel)
    /// ends up unclassified rather than guessed at.
    pub fn prep(&self, signal_pa: &[f32], adapter_end: usize) -> Option<Vec<f32>> {
        let (lo, hi) = self.window(adapter_end, signal_pa.len())?;
        let mut out = Vec::with_capacity(self.signal.chunk);
        self.standardise(&signal_pa[lo..hi], &mut out);
        Some(out)
    }

    /// The `chunk`-sample window to decode, or `None` if the read cannot supply
    /// one.
    ///
    /// Which end it is measured from is the bundle's [`Anchor`]. Under
    /// [`Anchor::ReadEnd`] this is `[len - chunk, len]` and `adapter_end` is
    /// ignored entirely — there is no detector, so there is no margin to clear
    /// and no shift to clamp, and the only way to fail is a read shorter than
    /// the window. `len` must therefore be the read's *full* length; see
    /// [`Self::needs_full_read`].
    fn window(&self, adapter_end: usize, len: usize) -> Option<(usize, usize)> {
        match self.signal.anchor() {
            Anchor::AdapterEnd => self.window_from_adapter_end(adapter_end, len),
            // `checked_sub` rather than a `len >= chunk` guard with
            // `then_some`, which evaluates the subtraction eagerly and
            // underflows on exactly the short read the guard exists to reject.
            Anchor::ReadEnd => len.checked_sub(self.signal.chunk).map(|lo| (lo, len)),
        }
    }

    /// [`Self::window`] under [`Anchor::AdapterEnd`].
    ///
    /// `[adapter_end - chunk, adapter_end]`. When `adapter_end < chunk`
    /// no such window exists — the read starts mid-adapter, so the signal simply
    /// runs out — and the read is refused, unless clamping is enabled. Clamping
    /// substitutes `[0, chunk]`: the same width, anchored at the read start,
    /// sliding `chunk - adapter_end` samples of downstream signal into the tail.
    ///
    /// The CRF tolerates that slide better than it looks: measured on RNA004
    /// nbc16 with known-good reads deliberately slid forward, the same barcode
    /// is still called for 98.6% at shift 0 and 93.5% at shift 500. Quality does
    /// decay with the shift, which is why the allowance is a bound and not a
    /// bool — see [`CrfMetadata::clamp_max_shift`].
    fn window_from_adapter_end(&self, adapter_end: usize, len: usize) -> Option<(usize, usize)> {
        let chunk = self.signal.chunk;
        if adapter_end > len {
            return None;
        }
        if adapter_end >= self.min_adapter_end() {
            return Some((adapter_end - chunk, adapter_end));
        }
        // `adapter_end` in `[chunk, chunk + margin)` has a real window and is
        // refused by the margin, not by geometry; clamping is not that lever.
        if adapter_end >= chunk {
            return None;
        }
        // 0 is the detector's overloaded "no adapter" / "too short" / "inference
        // failed" sentinel. Clamping it would decode a window with no adapter in
        // it at all and hand back a confident-looking call.
        if adapter_end == 0 || len < chunk {
            return None;
        }
        let shift = chunk - adapter_end;
        (shift <= self.clamp_max_shift()).then_some((0, chunk))
    }

    /// [`Self::prep`], but straight from ADC counts and converting only the
    /// window the model actually sees.
    ///
    /// The model reads `chunk` samples of calibrated pA ending at
    /// `adapter_end`, but callers hold ADC counts for the whole *decoded
    /// prefix* — `max_obs_trace` samples under the CNN detector (16 000 by
    /// default, 8× the window) and the entire read under LLR, which has no
    /// decode bound at all and routinely runs to millions of samples.
    /// Calibrating all of that to read the last 2000 samples is the waste this
    /// avoids: calibration and standardisation fuse into one pass over exactly
    /// the window.
    ///
    /// Calibration is the **fused** `adc.mul_add(scale, offset * scale)` that
    /// `escapepod_python::adc_to_pa` and `demux basecall` use, then
    /// `(pa - mean) / stdev` — the same three operations in the same order, so
    /// this is bit-identical to those two paths rather than merely equivalent.
    ///
    /// The fused pipeline previously used an unfused `(adc + offset) * scale`
    /// here, which its own doc comment claimed matched the reference but did
    /// not: two roundings instead of one, differing by ~1 ulp. Both CRF entry
    /// points now agree with the reference.
    ///
    /// Writes into `out` so a per-worker buffer survives across reads. Returns
    /// `false`, leaving `out` empty, exactly where [`Self::prep`] returns `None`.
    pub fn prep_adc_into(
        &self,
        adc: &[i16],
        adapter_end: usize,
        offset: f32,
        scale: f32,
        out: &mut Vec<f32>,
    ) -> bool {
        out.clear();
        let Some((lo, hi)) = self.window(adapter_end, adc.len()) else {
            return false;
        };
        let (mean, stdev) = (self.standardisation.mean, self.standardisation.stdev);
        let bias = offset * scale;
        out.extend(
            adc[lo..hi]
                .iter()
                .map(|&v| (f32::from(v).mul_add(scale, bias) - mean) / stdev),
        );
        true
    }

    /// Smallest `adapter_end` that yields a usable window.
    ///
    /// `extract_chunks.py` required `adapter_end > chunk + 200` when building
    /// the training corpus, so reads below that were never represented; the
    /// margin is kept here rather than accepting any read with `chunk` samples
    /// of history.
    pub fn min_adapter_end(&self) -> usize {
        self.signal.chunk + self.boundary_margin()
    }

    /// The margin in effect: the operator's override, else the bundle's
    /// declared `boundary.margin`, else [`BOUNDARY_MARGIN`].
    pub fn boundary_margin(&self) -> usize {
        self.margin_override
            .or_else(|| self.boundary.as_ref().and_then(|b| b.margin))
            .unwrap_or(BOUNDARY_MARGIN)
    }

    /// Largest `chunk - adapter_end` for which the window is clamped to
    /// `[0, chunk]` instead of the read being refused. 0 (the default) disables
    /// clamping entirely.
    ///
    /// A bound rather than a bool because recovery quality decays with the
    /// shift, so the useful question is how far to go, not whether. On the
    /// RNA004 nbc16 run, decoding the whole `adapter_end` 2,500-2,999 band this
    /// way returned 42,404 reads at median edit distance 0 — but agreement fell
    /// from 97.4% within 2 edits at shift 0-99 to 92.9% at 400-499, and the
    /// fraction that still align to a tRNA fell from 78.5% to 50.0%, because a
    /// larger shift means more of the adapter was truncated in the first place.
    /// Pick the bound from how much of that tail is worth having.
    pub fn clamp_max_shift(&self) -> usize {
        self.clamp_override
            .or_else(|| self.boundary.as_ref().and_then(|b| b.clamp_max_shift))
            .unwrap_or(0)
    }

    /// Overrule the bundle's declared `boundary.clamp_max_shift` for this run
    /// (`demux --clamp-max-shift`).
    pub fn set_clamp_max_shift(&mut self, shift: usize) {
        self.clamp_override = Some(shift);
    }

    /// Overrule the bundle's declared margin for this run
    /// (`demux --boundary-margin`).
    ///
    /// The margin describes how the training corpus was *filtered*, not what
    /// the encoder needs: any read with `adapter_end >= chunk` has a full
    /// window. Lowering it trades a window that reaches into the read's opening
    /// samples for reads that would otherwise be dropped undecoded. Prefer
    /// fixing the bundle's declaration; this exists for evaluating a change
    /// before it is baked into an export.
    pub fn set_boundary_margin(&mut self, margin: usize) {
        self.margin_override = Some(margin);
    }
}

/// Extra samples beyond `chunk` that `extract_chunks.py` demanded before a read
/// entered the training corpus.
const BOUNDARY_MARGIN: usize = 200;

// tract 0.23 renamed `SimplePlan<F, O, M>` to `RunnableModel<F, O>`.
type Plan = TypedRunnableModel;

/// CPU CTC-CRF basecaller: tract for the encoder, [`super::lattice`] for the
/// decode.
///
/// Build once and share across rayon workers — the plan is immutable, so
/// `&CrfEncoder` is `Sync` and each worker only needs its own [`CrfScratch`].
///
/// Inference runs one read at a time. That is deliberate and matches the
/// boundary-CNN path: tract has no efficient batched convolution, so batching
/// buys nothing on CPU and parallelism comes from running reads concurrently.
/// The GPU path batches instead.
pub struct CrfEncoder {
    plan: Arc<Plan>,
    meta: CrfMetadata,
    layout: CrfLayout,
    alphabet: Vec<u8>,
    /// Fastest decode kernels for this layout, resolved once at load rather
    /// than re-probed on every read.
    backend: Backend,
}

impl CrfEncoder {
    /// Load an export directory containing `metadata.json` and the ONNX graph
    /// it names.
    pub fn load_bundle(dir: impl AsRef<Path>) -> Result<Self, CrfError> {
        let dir = dir.as_ref();
        let meta = CrfMetadata::load(dir.join("metadata.json"))?;
        let onnx = dir.join(&meta.onnx);
        Self::load(onnx, meta)
    }

    /// Load an ONNX graph with an already-parsed sidecar.
    pub fn load(onnx: impl AsRef<Path>, meta: CrfMetadata) -> Result<Self, CrfError> {
        // Re-run for a caller that built the sidecar by hand rather than
        // through `CrfMetadata::load`; it is a handful of string compares once
        // per process, and skipping it is how the check gets bypassed.
        meta.validate()?;
        let layout = meta.layout()?;
        let alphabet = meta.alphabet_bytes();

        // The export leaves batch dynamic; tract needs it concrete to optimize,
        // and CPU inference is per-read, so pin batch = 1 here. The padding
        // hoist first: this graph's convolutions are zero-padded like the
        // boundary CNN's, and `padded_valid_x_loop` was 6% of the CPU encoder
        // arm in the 2026-09 profile. See `onnx_rewrite::hoist_conv_padding`.
        let framework = tract_onnx::onnx();
        let mut proto = framework
            .proto_model_for_path(onnx)
            .map_err(|e| CrfError::Load(e.to_string()))?;
        crate::onnx_rewrite::hoist_conv_padding(&mut proto, 1);
        let plan = framework
            .model_for_proto_model(&proto)
            .map_err(|e| CrfError::Load(e.to_string()))?
            .with_input_fact(0, f32::fact([1, 1, meta.signal.chunk]).into())
            .map_err(|e| CrfError::Load(e.to_string()))?
            .into_optimized()
            .map_err(|e| CrfError::Load(e.to_string()))?
            .into_runnable()
            .map_err(|e| CrfError::Load(e.to_string()))?;

        let encoder = Self {
            plan,
            meta,
            layout,
            alphabet,
            backend: Backend::best_for(&layout),
        };
        encoder.probe_output_contract()?;
        Ok(encoder)
    }

    /// Decode backend chosen for this layout.
    pub fn backend(&self) -> Backend {
        self.backend
    }

    /// One dummy forward pass asserting the time-major `[T, 1, n_score]`
    /// contract, so a mismatched export fails at load with a clear message
    /// instead of decoding noise for every read.
    fn probe_output_contract(&self) -> Result<(), CrfError> {
        let dummy = vec![0f32; self.meta.signal.chunk];
        self.encode(&dummy).map(|_| ())
    }

    /// Metadata sidecar in effect.
    pub fn metadata(&self) -> &CrfMetadata {
        &self.meta
    }

    /// Override the decode's boundary margin. See
    /// [`CrfMetadata::set_boundary_margin`].
    pub fn set_boundary_margin(&mut self, margin: usize) {
        self.meta.set_boundary_margin(margin);
    }

    /// Override how far the window may be clamped. See
    /// [`CrfMetadata::clamp_max_shift`].
    pub fn set_clamp_max_shift(&mut self, shift: usize) {
        self.meta.set_clamp_max_shift(shift);
    }

    /// Lattice geometry implied by the sidecar.
    pub fn layout(&self) -> &CrfLayout {
        &self.layout
    }

    /// Decode encoder scores for one read to a sequence.
    ///
    /// Split out from [`Self::basecall_prepped`] so the GPU path — which
    /// produces scores through onnxruntime instead of tract — can share the
    /// exact same decode, and so the two halves can be benchmarked separately.
    pub fn decode_scores(
        &self,
        scores: &[f32],
        scratch: &mut CrfScratch,
        backend: Backend,
    ) -> Result<String, CrfError> {
        decode_with(
            &self.layout,
            &self.alphabet,
            scores,
            self.meta.t_len(),
            scratch,
            backend,
        )
        .map_err(|e| CrfError::Decode(e.to_string()))
    }

    /// One forward pass over a standardised `chunk`-sample window.
    ///
    /// Kept separate from the score extraction so callers that only want to read
    /// the output can borrow it rather than own it — the outputs must stay alive
    /// for as long as the borrow, which the return type makes explicit.
    fn run_encoder(&self, prepped: &[f32]) -> Result<TVec<TValue>, CrfError> {
        let chunk = self.meta.signal.chunk;
        let input = Tensor::from_shape(&[1, 1, chunk], prepped)
            .map_err(|e| CrfError::Run(e.to_string()))?;
        self.plan
            .run(tvec!(input.into()))
            .map_err(|e| CrfError::Run(e.to_string()))
    }

    /// Borrow one forward pass's scores as a flat `t_len * n_score` slice,
    /// checking the time-major `[T, 1, n_score]` contract on the way.
    ///
    /// Batch is 1, so time-major already lays this read out contiguously and no
    /// de-interleave is needed — the GPU path's `split_time_major` exists
    /// precisely because that stops being true above batch 1.
    fn scores_of<'a>(&self, outputs: &'a TVec<TValue>) -> Result<&'a [f32], CrfError> {
        let view = outputs[0]
            .to_plain_array_view::<f32>()
            .map_err(|e| CrfError::Run(e.to_string()))?;
        let t_len = self.meta.t_len();
        if view.shape() != [t_len, 1, self.layout.n_score] {
            return Err(CrfError::BadShape {
                got: view.shape().to_vec(),
                t: t_len,
                batch: 1,
                n_score: self.layout.n_score,
            });
        }
        view.to_slice()
            .ok_or_else(|| CrfError::Run("encoder output is not contiguous".into()))
    }

    /// Run the encoder on one standardised `chunk`-sample window, returning
    /// `t_len * n_score` scores in the decoder's expected `[t][dest][edge]`
    /// order.
    ///
    /// This allocates and copies the whole score buffer — 1 MB for the RNA004
    /// geometry. [`Self::basecall_prepped`] decodes out of tract's own output
    /// instead and does not pay it; prefer that unless you genuinely need to own
    /// the scores.
    pub fn encode(&self, prepped: &[f32]) -> Result<Vec<f32>, CrfError> {
        let outputs = self.run_encoder(prepped)?;
        Ok(self.scores_of(&outputs)?.to_vec())
    }

    /// Build the constrained lattices for a reference panel, once per run.
    ///
    /// `seqs` are the sequences the model **emits** — what a bundle's
    /// `barcodes[].sequence` holds, already trimmed to `target[state_len..]`.
    /// Passing full-length training targets here scores a longer string than
    /// the model can emit and drives every reference's probability to zero,
    /// the same trap that inflates every edit distance on the matching path.
    pub fn ref_chains(&self, seqs: &[&[u8]]) -> Result<RefChains, CrfError> {
        Ok(RefChains::build(&self.layout, &self.alphabet, seqs)?)
    }

    /// [`Self::decode_scores`], additionally scoring every reference in
    /// `chains`: `out` receives `log P(reference | signal)` per reference.
    ///
    /// See [`super::refchain`] for what that is and why the decode is where it
    /// has to be computed.
    pub fn decode_scores_with_refs(
        &self,
        scores: &[f32],
        scratch: &mut CrfScratch,
        backend: Backend,
        chains: &RefChains,
        out: &mut Vec<f32>,
    ) -> Result<String, CrfError> {
        decode_with_refs(
            &self.layout,
            &self.alphabet,
            scores,
            self.meta.t_len(),
            scratch,
            backend,
            chains,
            out,
        )
        .map_err(|e| CrfError::Decode(e.to_string()))
    }

    /// [`Self::basecall_prepped`], additionally scoring every reference in
    /// `chains`.
    pub fn basecall_prepped_with_refs(
        &self,
        prepped: &[f32],
        scratch: &mut CrfScratch,
        chains: &RefChains,
    ) -> Result<ScoredDecode, CrfError> {
        let outputs = self.run_encoder(prepped)?;
        let scores = self.scores_of(&outputs)?;
        let mut ref_logp = Vec::with_capacity(chains.len());
        let sequence =
            self.decode_scores_with_refs(scores, scratch, self.backend, chains, &mut ref_logp)?;
        Ok(ScoredDecode {
            sequence,
            ref_logp,
            mean_logpost: scratch.path_score() / self.meta.t_len().max(1) as f32,
        })
    }

    /// Basecall one already-prepped read.
    ///
    /// The scores are decoded straight out of tract's output tensor. The decode
    /// immediately transposes them into `scratch`, so materialising an owned
    /// copy first would be a 1 MB allocation and memcpy per read that nothing
    /// ever reads twice.
    pub fn basecall_prepped(
        &self,
        prepped: &[f32],
        scratch: &mut CrfScratch,
    ) -> Result<String, CrfError> {
        let outputs = self.run_encoder(prepped)?;
        let scores = self.scores_of(&outputs)?;
        self.decode_scores(scores, scratch, self.backend)
    }

    /// Basecall one read from raw calibrated pA and a detected adapter end.
    ///
    /// Returns `Ok(None)` when the read has no usable window — see
    /// [`CrfMetadata::prep`].
    pub fn basecall(
        &self,
        signal_pa: &[f32],
        adapter_end: usize,
        scratch: &mut CrfScratch,
    ) -> Result<Option<String>, CrfError> {
        match self.meta.prep(signal_pa, adapter_end) {
            Some(prepped) => self.basecall_prepped(&prepped, scratch).map(Some),
            None => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta() -> CrfMetadata {
        serde_json::from_str(
            r#"{
              "onnx": "crf_encoder.onnx",
              "standardisation": {"mean": 62.404976, "stdev": 10.232168},
              "signal": {"chunk": 2000, "stride": 10},
              "crf": {"state_len": 4, "n_base": 4,
                      "alphabet": ["N", "A", "C", "G", "T"]},
              "provenance": {"ignored": true}
            }"#,
        )
        .unwrap()
    }

    #[test]
    fn sidecar_parses_and_ignores_extra_fields() {
        let m = meta();
        assert_eq!(m.t_len(), 200);
        assert_eq!(m.layout().unwrap().n_score, 1280);
        assert_eq!(m.min_adapter_end(), 2200);
    }

    /// Precedence: operator override > bundle's declared `boundary.margin` >
    /// [`BOUNDARY_MARGIN`]. A bundle stating 0 asserts its model tolerates a
    /// window reaching the read's opening samples.
    #[test]
    fn boundary_margin_precedence() {
        // No boundary block at all: the legacy default.
        let mut m = meta();
        assert_eq!(m.boundary_margin(), BOUNDARY_MARGIN);
        assert_eq!(m.min_adapter_end(), m.signal.chunk + BOUNDARY_MARGIN);

        m.set_boundary_margin(0);
        assert_eq!(m.min_adapter_end(), m.signal.chunk, "override wins");

        // Declared by the bundle, no override.
        let declared: CrfMetadata = serde_json::from_str(
            r#"{
              "standardisation": {"mean": 1.0, "stdev": 1.0},
              "signal": {"chunk": 2000, "stride": 10},
              "crf": {"state_len": 4, "n_base": 4,
                      "alphabet": ["N", "A", "C", "G", "T"]},
              "boundary": {"method": "cnn", "margin": 0}
            }"#,
        )
        .unwrap();
        assert_eq!(declared.boundary_margin(), 0);
        assert_eq!(declared.min_adapter_end(), 2000);

        // A boundary block that omits it still falls back.
        let silent: CrfMetadata = serde_json::from_str(
            r#"{
              "standardisation": {"mean": 1.0, "stdev": 1.0},
              "signal": {"chunk": 2000, "stride": 10},
              "crf": {"state_len": 4, "n_base": 4,
                      "alphabet": ["N", "A", "C", "G", "T"]},
              "boundary": {"method": "cnn"}
            }"#,
        )
        .unwrap();
        assert_eq!(silent.min_adapter_end(), 2000 + BOUNDARY_MARGIN);

        // Override beats a declaration too.
        let mut both = declared;
        both.set_boundary_margin(300);
        assert_eq!(both.min_adapter_end(), 2300);
    }

    /// Clamping substitutes `[0, chunk]` for reads whose adapter ends before
    /// `chunk`, bounded by how far the window may slide. Off by default.
    #[test]
    fn window_clamps_only_within_the_declared_shift() {
        let chunk = 2000;
        let mut m = meta();
        m.set_boundary_margin(0);
        assert_eq!(m.clamp_max_shift(), 0, "off unless asked for");

        // Disabled: anything short of a full window is refused.
        assert_eq!(m.window(chunk, 9_000), Some((0, chunk)), "exactly a window");
        assert_eq!(m.window(1_900, 9_000), None);

        m.set_clamp_max_shift(500);
        // Inside the bound: same width, anchored at the read start.
        assert_eq!(m.window(1_900, 9_000), Some((0, chunk)), "shift 100");
        assert_eq!(
            m.window(1_500, 9_000),
            Some((0, chunk)),
            "shift 500, the edge"
        );
        // Past it.
        assert_eq!(m.window(1_499, 9_000), None, "shift 501");
        // A normal read is untouched by clamping.
        assert_eq!(m.window(3_000, 9_000), Some((1_000, 3_000)));

        // The detector's "no adapter" sentinel must never be clamped, however
        // generous the bound — the window would hold no adapter at all.
        m.set_clamp_max_shift(chunk + 1_000);
        assert_eq!(m.window(0, 9_000), None, "adapter_end == 0 stays refused");
        // Nor may a clamp invent signal the read does not have.
        assert_eq!(m.window(1_900, chunk - 1), None, "read shorter than chunk");

        // Clamping does not rescue a read the MARGIN refuses: that window
        // exists, so it is a different decision.
        let mut margin_gated = meta();
        margin_gated.set_clamp_max_shift(500);
        assert_eq!(margin_gated.min_adapter_end(), chunk + BOUNDARY_MARGIN);
        assert_eq!(margin_gated.window(chunk + 50, 9_000), None);
    }

    /// The bundle can declare the clamp, and an override beats it.
    #[test]
    fn clamp_max_shift_precedence() {
        let declared: CrfMetadata = serde_json::from_str(
            r#"{
              "standardisation": {"mean": 1.0, "stdev": 1.0},
              "signal": {"chunk": 2000, "stride": 10},
              "crf": {"state_len": 4, "n_base": 4,
                      "alphabet": ["N", "A", "C", "G", "T"]},
              "boundary": {"method": "cnn", "margin": 0, "clamp_max_shift": 400}
            }"#,
        )
        .unwrap();
        assert_eq!(declared.clamp_max_shift(), 400);
        assert_eq!(declared.window(1_600, 9_000), Some((0, 2_000)));
        assert_eq!(declared.window(1_599, 9_000), None);

        let mut overridden = declared;
        overridden.set_clamp_max_shift(0);
        assert_eq!(overridden.window(1_600, 9_000), None, "override disables");
    }

    /// The boundary block's `input` contract and `sha256` are optional
    /// (bundles from before escapepod-models shipped them), and parse when
    /// present (#187).
    #[test]
    fn boundary_input_contract_is_optional_and_parses() {
        let bare: BoundarySpec = serde_json::from_str(
            r#"{"method": "cnn", "onnx": "adapter.onnx", "model_id": "adapter_rna004@v1.1.0"}"#,
        )
        .unwrap();
        assert!(bare.input.is_none());
        assert!(bare.sha256.is_none());

        let declared: BoundarySpec = serde_json::from_str(
            r#"{
              "method": "cnn",
              "onnx": "adapter.onnx",
              "model_id": "adapter_rna004@v1.1.0",
              "sha256": "b59f8667187ef9fa7e940cd37b108f8d5f3c6d6213ca841cda6eced0e33d26b5",
              "input": {
                "min_obs_adapter": 1000,
                "max_obs_trace": 16000,
                "downscale_factor": 10,
                "input_len": 1500,
                "pad_value": -5.0,
                "source": "escapepod_models.config.DataConfig"
              }
            }"#,
        )
        .unwrap();
        assert_eq!(
            declared.sha256.as_deref(),
            Some("b59f8667187ef9fa7e940cd37b108f8d5f3c6d6213ca841cda6eced0e33d26b5")
        );
        let input = declared.input.expect("input block present");
        assert_eq!(input.min_obs_adapter, 1000);
        assert_eq!(input.max_obs_trace, 16000);
        assert_eq!(input.downscale_factor, 10);
        assert_eq!(input.input_len, 1500);
        assert_eq!(input.pad_value, -5.0);
    }

    #[test]
    fn prep_takes_the_window_ending_at_the_adapter() {
        let m = meta();
        let signal: Vec<f32> = (0..5000).map(|i| i as f32).collect();
        let got = m.prep(&signal, 4000).unwrap();
        assert_eq!(got.len(), 2000);
        // First sample of the window is signal[4000 - 2000], standardised.
        let want = (2000.0 - m.standardisation.mean) / m.standardisation.stdev;
        assert!((got[0] - want).abs() < 1e-3, "got {} want {want}", got[0]);
        let want_last = (3999.0 - m.standardisation.mean) / m.standardisation.stdev;
        assert!((got[1999] - want_last).abs() < 1e-3);
    }

    /// `adapter_end == 0` is the boundary detector's overloaded sentinel and
    /// must not be treated as a window.
    #[test]
    fn prep_rejects_reads_without_a_full_window() {
        let m = meta();
        let signal: Vec<f32> = vec![1.0; 5000];
        assert!(m.prep(&signal, 0).is_none());
        assert!(m.prep(&signal, 2199).is_none(), "below the training margin");
        assert!(m.prep(&signal, 2200).is_some(), "exactly at the margin");
        assert!(m.prep(&signal, 6000).is_none(), "past the end of the read");
    }

    /// `prep_adc_into` must be bit-identical to calibrating the whole prefix
    /// and calling `prep` on it — that equality is the only reason it is safe to
    /// convert just the window. Checked against the *fused* reference
    /// (`escapepod_python::adc_to_pa`), which is what `demux basecall` used.
    #[test]
    fn prep_adc_into_matches_calibrate_then_prep() {
        let m = meta();
        let (offset, scale) = (7.5f32, 0.1875f32);
        let adc: Vec<i16> = (0..5000).map(|i| ((i * 37) % 4096 - 2048) as i16).collect();

        let bias = offset * scale;
        let pa: Vec<f32> = adc
            .iter()
            .map(|&v| f32::from(v).mul_add(scale, bias))
            .collect();

        for end in [2200usize, 3000, 4000, 5000] {
            let want = m.prep(&pa, end).expect("usable window");
            let mut got = Vec::new();
            assert!(m.prep_adc_into(&adc, end, offset, scale, &mut got));
            assert_eq!(got, want, "adapter_end {end}");
        }
    }

    /// The rejection cases must line up exactly with `prep`, including the
    /// `adapter_end == 0` sentinel, and must leave the buffer empty so a reused
    /// per-worker `Vec` cannot leak the previous read's window.
    #[test]
    fn prep_adc_into_rejects_where_prep_does() {
        let m = meta();
        let adc = vec![100i16; 5000];
        let mut buf = vec![1.0f32; 8];
        for end in [0usize, 2199, 6000] {
            assert!(!m.prep_adc_into(&adc, end, 0.0, 1.0, &mut buf), "end {end}");
            assert!(buf.is_empty(), "end {end}: buffer not cleared");
        }
        assert!(m.prep_adc_into(&adc, 2200, 0.0, 1.0, &mut buf));
        assert_eq!(buf.len(), 2000);
    }

    #[test]
    fn rejects_an_alphabet_that_does_not_match_the_geometry() {
        let mut m = meta();
        m.crf.alphabet.pop();
        assert!(matches!(m.layout(), Err(CrfError::BadAlphabet { .. })));
    }

    /// A `signal` block with the shape `barcode_crf_fdx4_rna004@v0.1.1` ships.
    fn read_end_meta() -> CrfMetadata {
        serde_json::from_str(
            r#"{
              "standardisation": {"mean": 69.535726, "stdev": 16.106312},
              "signal": {"chunk": 3500, "stride": 10,
                         "window": "[read_end - chunk, read_end]",
                         "anchor": "read_end"},
              "crf": {"state_len": 4, "n_base": 4,
                      "alphabet": ["N", "A", "C", "G", "T"]},
              "built_beside": {"model_id": "adapter_rna004@v1.1.0", "role": "none"}
            }"#,
        )
        .unwrap()
    }

    /// The window is `[len - chunk, len]` and `adapter_end` is ignored — which
    /// is the whole point, since a read-end model runs with no detector and
    /// therefore only ever sees the `(0, 0)` sentinel.
    #[test]
    fn read_end_windows_back_from_the_last_sample() {
        let m = read_end_meta();
        assert_eq!(m.signal.anchor(), Anchor::ReadEnd);
        assert!(!m.needs_boundary());
        assert!(m.needs_full_read());

        assert_eq!(m.window(0, 9_000), Some((5_500, 9_000)), "sentinel ignored");
        assert_eq!(
            m.window(4_242, 9_000),
            Some((5_500, 9_000)),
            "any adapter_end gives the same window"
        );
        assert_eq!(m.window(0, 3_500), Some((0, 3_500)), "exactly a window");
        assert_eq!(m.window(0, 3_499), None, "read shorter than chunk");

        // Neither knob applies, so neither can rescue or refuse a read: the
        // margin would otherwise gate everything below chunk + 200.
        let mut m = m;
        m.set_boundary_margin(5_000);
        m.set_clamp_max_shift(5_000);
        assert_eq!(m.window(0, 3_600), Some((100, 3_600)), "margin is inert");
    }

    /// `prep` reads the tail, so it must be handed the read's *full* signal —
    /// truncating it does not shorten the window, it relocates it.
    #[test]
    fn read_end_prep_takes_the_tail() {
        let m = read_end_meta();
        let signal: Vec<f32> = (0..5_000).map(|i| i as f32).collect();
        let got = m.prep(&signal, 0).unwrap();
        assert_eq!(got.len(), 3_500);
        let want = (1_500.0 - m.standardisation.mean) / m.standardisation.stdev;
        assert!((got[0] - want).abs() < 1e-3, "got {} want {want}", got[0]);
        let want_last = (4_999.0 - m.standardisation.mean) / m.standardisation.stdev;
        assert!((got[3_499] - want_last).abs() < 1e-3);
    }

    /// The default is `adapter_end`, because every bundle predating the key was
    /// built that way. Silence must not become a second meaning.
    #[test]
    fn absent_anchor_means_adapter_end() {
        let m = meta();
        assert_eq!(m.signal.anchor(), Anchor::AdapterEnd);
        assert!(m.needs_boundary());
        assert!(!m.needs_full_read());
        assert_eq!(m.window(4_000, 9_000), Some((2_000, 4_000)));
    }

    /// `deny_unknown_fields` on `signal` only. A rule this runtime does not
    /// implement must fail the load rather than be dropped — but the top level
    /// still carries provenance nobody should depend on, and that stays open.
    #[test]
    fn unknown_keys_are_refused_in_signal_and_ignored_above_it() {
        let with_rule = serde_json::from_str::<CrfMetadata>(
            r#"{
              "standardisation": {"mean": 1.0, "stdev": 1.0},
              "signal": {"chunk": 2000, "stride": 10, "detrend": "median"},
              "crf": {"state_len": 4, "n_base": 4,
                      "alphabet": ["N", "A", "C", "G", "T"]}
            }"#,
        );
        assert!(
            with_rule.is_err(),
            "an unimplemented signal rule must refuse"
        );

        // A spelling this runtime does not know is a value, not a key, and
        // serde refuses it for the same reason.
        let bad_anchor = serde_json::from_str::<CrfMetadata>(
            r#"{
              "standardisation": {"mean": 1.0, "stdev": 1.0},
              "signal": {"chunk": 2000, "stride": 10, "anchor": "poly_a_start"},
              "crf": {"state_len": 4, "n_base": 4,
                      "alphabet": ["N", "A", "C", "G", "T"]}
            }"#,
        );
        assert!(bad_anchor.is_err(), "an unknown anchor must refuse");

        // Provenance above `signal` is still free-form.
        assert!(meta().signal.window.is_none());

        // Closing this block is only safe because it is closed over the key set
        // the fleet actually ships. As of this change every published CRF
        // bundle declares `chunk`/`stride`/`window`, and fdx4@v0.1.1 adds
        // `anchor`; both shapes must load, or `deny_unknown_fields` grounds
        // every model rather than the one bad export it is aimed at.
        for signal in [
            r#"{"chunk": 3000, "stride": 10,
                "window": "[adapter_end - chunk, adapter_end]"}"#,
            r#"{"chunk": 3500, "stride": 10, "anchor": "read_end",
                "window": "[read_end - chunk, read_end]"}"#,
        ] {
            let json = format!(
                r#"{{"standardisation": {{"mean": 1.0, "stdev": 1.0}},
                     "signal": {signal},
                     "crf": {{"state_len": 4, "n_base": 4,
                              "alphabet": ["N", "A", "C", "G", "T"]}}}}"#
            );
            let m: CrfMetadata = serde_json::from_str(&json).expect("shipped key set parses");
            assert!(m.validate().is_ok(), "shipped key set validates");
        }
    }

    /// The prose and the machine-readable key describe one rule. When they
    /// disagree, refuse — do not pick a side. This is
    /// `barcode_crf_fdx4_rna004@v0.1.0`, whose sidecar claimed the adapter-end
    /// window for a read-end model.
    #[test]
    fn a_window_string_that_contradicts_the_anchor_is_refused() {
        let m: CrfMetadata = serde_json::from_str(
            r#"{
              "standardisation": {"mean": 1.0, "stdev": 1.0},
              "signal": {"chunk": 3500, "stride": 10,
                         "window": "[adapter_end - chunk, adapter_end]",
                         "anchor": "read_end"},
              "crf": {"state_len": 4, "n_base": 4,
                      "alphabet": ["N", "A", "C", "G", "T"]}
            }"#,
        )
        .unwrap();
        assert!(matches!(
            m.validate(),
            Err(CrfError::AnchorDisagreesWithWindow { .. })
        ));
        // The agreeing pair, both ways round, passes.
        assert!(read_end_meta().validate().is_ok());
        let adapter: CrfMetadata = serde_json::from_str(
            r#"{
              "standardisation": {"mean": 1.0, "stdev": 1.0},
              "signal": {"chunk": 3000, "stride": 10,
                         "window": "[adapter_end - chunk, adapter_end]"},
              "crf": {"state_len": 4, "n_base": 4,
                      "alphabet": ["N", "A", "C", "G", "T"]}
            }"#,
        )
        .unwrap();
        assert!(adapter.validate().is_ok());
    }

    /// The other half of the same defect: a read-end model that pins a
    /// detector. escpod honours a `boundary` block at runtime (it refuses
    /// `--method llr` against a `cnn` pin), so believing this one would window
    /// the 3' adapter and feed the model the far end of the molecule.
    #[test]
    fn a_read_end_bundle_may_not_pin_a_boundary_detector() {
        let m: CrfMetadata = serde_json::from_str(
            r#"{
              "standardisation": {"mean": 1.0, "stdev": 1.0},
              "signal": {"chunk": 3500, "stride": 10, "anchor": "read_end"},
              "crf": {"state_len": 4, "n_base": 4,
                      "alphabet": ["N", "A", "C", "G", "T"]},
              "boundary": {"method": "cnn", "onnx": "adapter_rna004.onnx"}
            }"#,
        )
        .unwrap();
        assert!(matches!(
            m.validate(),
            Err(CrfError::BoundaryPinnedOnReadEnd { .. })
        ));
    }
}
