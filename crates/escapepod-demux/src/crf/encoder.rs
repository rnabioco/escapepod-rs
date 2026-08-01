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
        let chunk = self.signal.chunk;
        if adapter_end < self.min_adapter_end() || adapter_end > signal_pa.len() {
            return None;
        }
        let mut out = Vec::with_capacity(chunk);
        self.standardise(&signal_pa[adapter_end - chunk..adapter_end], &mut out);
        Some(out)
    }

    /// Smallest `adapter_end` that yields a usable window.
    ///
    /// `extract_chunks.py` required `adapter_end > chunk + 200` when building
    /// the training corpus, so reads below that were never represented; the
    /// margin is kept here rather than accepting any read with `chunk` samples
    /// of history.
    pub fn min_adapter_end(&self) -> usize {
        self.signal.chunk + BOUNDARY_MARGIN
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
        };
        encoder.probe_output_contract()?;
        Ok(encoder)
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

    /// Run the encoder on one standardised `chunk`-sample window, returning
    /// `t_len * n_score` scores in the decoder's expected `[t][dest][edge]`
    /// order.
    pub fn encode(&self, prepped: &[f32]) -> Result<Vec<f32>, CrfError> {
        let chunk = self.meta.signal.chunk;
        let input = Tensor::from_shape(&[1, 1, chunk], prepped)
            .map_err(|e| CrfError::Run(e.to_string()))?;
        let outputs = self
            .plan
            .run(tvec!(input.into()))
            .map_err(|e| CrfError::Run(e.to_string()))?;
        let view = outputs[0]
            .to_plain_array_view::<f32>()
            .map_err(|e| CrfError::Run(e.to_string()))?;

        let t_len = self.meta.t_len();
        let want = [t_len, 1, self.layout.n_score];
        if view.shape() != want {
            return Err(CrfError::BadShape {
                got: view.shape().to_vec(),
                t: t_len,
                batch: 1,
                n_score: self.layout.n_score,
            });
        }
        // Batch is 1, so time-major already lays this read out contiguously.
        Ok(view.iter().copied().collect())
    }

    /// Basecall one already-prepped read.
    pub fn basecall_prepped(
        &self,
        prepped: &[f32],
        scratch: &mut CrfScratch,
    ) -> Result<String, CrfError> {
        let scores = self.encode(prepped)?;
        self.decode_scores(&scores, scratch, Backend::best_for(&self.layout))
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

    #[test]
    fn rejects_an_alphabet_that_does_not_match_the_geometry() {
        let mut m = meta();
        m.crf.alphabet.pop();
        assert!(matches!(m.layout(), Err(CrfError::BadAlphabet { .. })));
    }
}
