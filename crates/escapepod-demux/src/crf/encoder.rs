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
use tract_onnx::tract_core::model::TypedRunnableModel;

use super::lattice::{Backend, CrfLayout, CrfScratch, decode_with};

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
    #[error("lattice geometry {n_base}^{state_len} is not representable")]
    BadGeometry { n_base: usize, state_len: usize },
    #[error("alphabet has {got} symbols, expected {expected} (blank plus one per base)")]
    BadAlphabet { got: usize, expected: usize },
    #[error("decode failed: {0}")]
    Decode(String),
}

/// The `metadata.json` sidecar that travels with a CRF encoder export.
///
/// Unknown fields are ignored on purpose: the sidecar also carries provenance
/// and human-facing notes that a consumer has no business depending on.
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

#[derive(Debug, Clone, Copy, Deserialize)]
pub struct SignalSpec {
    /// Samples fed to the encoder per read.
    pub chunk: usize,
    /// Samples per output timestep.
    pub stride: usize,
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
        serde_json::from_str(&raw).map_err(|source| CrfError::MetadataParse {
            path: path.to_path_buf(),
            source,
        })
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
    /// The window is `[adapter_end - chunk, adapter_end]` of **raw calibrated
    /// pA** — not ADC counts, not MAD-normalised signal — matching
    /// `extract_chunks.py`. Returns `None` when the read cannot supply a full
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
    /// Normally `[adapter_end - chunk, adapter_end]`. When `adapter_end < chunk`
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
    fn window(&self, adapter_end: usize, len: usize) -> Option<(usize, usize)> {
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
        let layout = meta.layout()?;
        let alphabet = meta.alphabet_bytes();

        // The export leaves batch dynamic; tract needs it concrete to optimize,
        // and CPU inference is per-read, so pin batch = 1 here.
        let plan = tract_onnx::onnx()
            .model_for_path(onnx)
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
        assert_eq!(m.window(1_500, 9_000), Some((0, chunk)), "shift 500, the edge");
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
}
