// SPDX-License-Identifier: MIT

//! The windowed variant's ONNX graph, through tract.
//!
//! # Why the export version is load-bearing
//!
//! This runs through tract, statically linked, like every other ONNX graph
//! `escpod` loads — but only because the model was re-exported. The first
//! shipped export could not go through tract at all, and the reason is worth
//! keeping precisely: **tract runs these ops fine; its shape inference could
//! not close that export.** Measured on `charging_tcn_rna004@v0.1.0`, five
//! ways, all of which parse the graph and then fail during analysis:
//!
//! ```text
//! inputs pinned to batch 1        node_conv1d       Sym(batch) vs Val(1)
//! value_info cleared              node_index        rank 8 vs rank 6
//! value_info cleared + pinned     node_index        rank 8 vs rank 6
//! symbolic dims rewritten to 1    node_GatherND_329 Val(64) vs Val(1)
//! nothing pinned at all           node_GatherND_329 Sym(batch) vs Val(1)
//! ```
//!
//! That is **two independent causes**, which is why the first row fails
//! somewhere else than the last two, and why fixing either alone leaves the
//! graph unloadable:
//!
//! 1. dynamo writes a `value_info` entry for all 667 intermediates with the
//!    batch axis as the *symbol* `batch`. A consumer that pins the batch — as
//!    this loader and [`crate::fnn`] both do — cannot unify that, and tract
//!    dies at the **first convolution**, nowhere near anything interesting.
//!    Every other graph `escpod` loads carries zero `value_info`, because the
//!    legacy TorchScript exporter never wrote any.
//! 2. `adaptive_avg_pool1d(390 -> 11)`, which dynamo open-codes into a rank-8
//!    `GatherND` because the output size does not divide the input.
//!
//! **Not** `nn.MultiheadAttention`, which earlier revisions of this module and
//! of rnabioco/escapepod-models#96 both named. Reading the graph settles it:
//! the offending `GatherND` consumes `relu_17`, the last block of `signal_tcn`;
//! its `(11, 37)` bool mask is a bin mask and its `(11,)` divisor is
//! `[36, 36, 37, 36, 37, …]`, the bin widths of that pool. `cross_attn` exports
//! as plain `Mul`/`MatMul`/`Softmax`/`MatMul`/`Gemm`, with no mask and no
//! gather. rnabioco/escapepod-rs#306's original suspect was right and its
//! retraction was not.
//!
//! Neither standard rewrite helped, so it could not be papered over at load
//! time the way [`crate::fnn`]'s `hoist_conv_padding` papers over padded
//! convolutions: onnx-simplifier folds away every `Shape` node (479 -> 428
//! nodes) and tract fails at the same `GatherND`; onnxruntime's own optimiser
//! keeps it and adds hardware-specific fusions. The fix had to be, and was,
//! the export — the third time in this model family that tract shape inference
//! turned out to be an export bug, after the retracted `Resize` gotcha.
//!
//! `charging_tcn_rna004@v0.1.1` is that re-export (leech 0.10.0,
//! rnabioco/leech#233): the pool written as one `MatMul` against a constant
//! segment-mean matrix, and `value_info` stripped. Same weights, no retrain,
//! evaluation bit-identical — 479 -> 319 ONNX nodes, `GatherND` 2 -> 0.
//! Re-measured here with `escapepod-demux/examples/tract_dynamo_probe.rs`,
//! which is kept precisely so this claim can be re-run (its counts are tract's
//! own, after parsing, so they are larger than the ONNX node counts above):
//!
//! ```text
//! v0.1.0   669 nodes   analysis fails at node_GatherND_329 / node_index
//! v0.1.1   471 nodes   optimized to 655, runs, output [1, 1]
//! ```
//!
//! **The lesson worth carrying**, and the reason escapepod-models now gates
//! `ship` on it: onnxruntime loaded the broken graph perfectly, so the export's
//! own torch round-trip was green throughout. "It exports and agrees with
//! torch" is a weaker claim than "a runtime can load it". So an unloadable
//! bundle is now a *bundle* problem with a build-time gate on it
//! (escapepod-models#97), and tract's own analysis error is the most
//! informative thing this loader could say about one that slips through anyway.
//!
//! # What this buys, and why there is no feature flag
//!
//! The alternative was `ort` (onnxruntime), which is how this module was first
//! written. It works, but it is built `load-dynamic`: onnxruntime is dlopened
//! at run time from `ORT_DYLIB_PATH`, and every `escpod` release artifact is
//! **static musl**, which cannot dlopen anything. A `waveform_model` bundle
//! was therefore unreachable from a released binary by construction, and the
//! variant needed an opt-in feature to keep that runtime requirement out of
//! the default build.
//!
//! On tract all of that goes away — the variant is in the default build, works
//! from a stock release, and needs nothing on the path. Two smaller things go
//! with it. There is no session pool: `ort`'s `Session::run` takes `&mut
//! self`, so scoring under rayon needed one session per worker to avoid
//! serialising every inference behind a mutex, whereas a tract plan is
//! immutable and one instance serves every worker. And the `ort` dependency
//! edge is gone from this crate, which is one fewer place for
//! `download-binaries` to drag OpenSSL into a build that never downloads
//! anything.
//!
//! It is slower, and by enough to say so: **6.27 ms/chunk against
//! onnxruntime's 4.4**, single-threaded, on the same 256 chunks through
//! `examples/verify_waveform_model`. That buys reachability from a release
//! binary — which the `ort` path did not have at any speed — and this pipeline
//! scores reads under rayon, so the per-chunk figure is not the wall clock.
//! Graph parity is unaffected: max |dlogit| 3.3e-6 over the corpus's own
//! tensors, against an export whose own residual vs torch is 1.3e-5.
//!
//! # What is checked, and what is trusted
//!
//! The bundle declares three input tensors and one output; this resolves all
//! four against the graph at load and refuses a mismatch. That matters more
//! here than for a single-input graph: the three tensors are *different
//! shapes*, and feeding them in the wrong order is not a shape error on two of
//! the three, so the names are resolved from the graph rather than assumed
//! positional.

use anyhow::{Result, anyhow, bail};
use std::path::Path;
use std::sync::Arc;

use tract_onnx::prelude::*;
use tract_onnx::tract_core::model::TypedRunnableModel;

use escapepod_signal::chunk::Chunk;

use crate::bundle::{WaveformSpec, WaveformTensor};

/// A loaded windowed-variant graph, ready to score chunks.
///
/// Batch is pinned to 1 at load, for the reason [`crate::fnn::FeatureNet`]
/// pins it: classification fans out across reads with rayon, so a batch axis
/// would buy nothing and cost a re-optimisation per batch size. The plan is
/// immutable, so the handle is `Sync` and one instance serves every worker.
///
/// That is a choice, not a limit — escapepod-models ran the re-export through
/// tract at batch 1 *and* batch 32 (max |dlogit| vs torch 5.72e-06 over 256
/// real chunks, 0 decision disagreements), so a batched path is open if the
/// per-chunk cost ever justifies one.
pub struct WaveformNet {
    plan: Arc<TypedRunnableModel>,
    /// Which assembled tensor feeds each graph input, in the graph's own input
    /// order — resolved by name at load, never assumed positional.
    inputs: Vec<WaveformTensor>,
}

impl std::fmt::Debug for WaveformNet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The plan is a whole optimised graph; printing it in a bundle dump is
        // noise, and `ChargingBundle` derives Debug.
        f.debug_struct("WaveformNet")
            .field("inputs", &self.inputs)
            .finish_non_exhaustive()
    }
}

impl WaveformNet {
    /// Open the graph and pin its contract against `spec`.
    pub fn load(path: &Path, spec: &WaveformSpec) -> Result<Self> {
        let onnx = tract_onnx::onnx();
        let proto = onnx
            .proto_model_for_path(path)
            .map_err(|e| anyhow!("cannot read the waveform model {}: {e}", path.display()))?;
        let model = onnx
            .model_for_proto_model(&proto)
            .map_err(|e| anyhow!("cannot parse the waveform model {}: {e}", path.display()))?;

        // Resolve each graph input to the tensor this runtime assembles for
        // it, by name. Two of the six orderings would be caught by a shape
        // check and the rest would not, so the name is the only thing that
        // makes this safe.
        let inputs: Vec<WaveformTensor> = {
            let outlets = model
                .input_outlets()
                .map_err(|e| anyhow!("the waveform model declares no usable inputs: {e}"))?
                .to_vec();
            outlets
                .iter()
                .map(|outlet| {
                    let name = model.node(outlet.node).name.as_str();
                    WaveformTensor::from_name(name).ok_or_else(|| {
                        anyhow!(
                            "the waveform model takes an input named {name:?}, which this \
                             runtime does not assemble; it produces `signal`, `sequence` \
                             and `features`"
                        )
                    })
                })
                .collect::<Result<_>>()?
        };

        // Every tensor the geometry declares must be an input, and no input
        // may be one it does not declare. A `[0, _]` shape is how
        // `tensor_shape` spells "this variant has no such tensor".
        for role in [
            WaveformTensor::Signal,
            WaveformTensor::Sequence,
            WaveformTensor::Features,
        ] {
            let wanted = spec.tensor_shape(role);
            let present = inputs.contains(&role);
            if present == (wanted[0] == 0) {
                bail!(
                    "the declared geometry {} a {} tensor, but the graph {} one",
                    if wanted[0] == 0 {
                        "produces no"
                    } else {
                        "produces"
                    },
                    role.name(),
                    if present { "takes" } else { "does not take" }
                );
            }
        }

        let n_outputs = model
            .output_outlets()
            .map_err(|e| anyhow!("the waveform model declares no usable outputs: {e}"))?
            .len();
        if n_outputs != 1 {
            bail!(
                "the waveform model has {n_outputs} outputs; the contract is exactly one, \
                 a [batch, 1] logit"
            );
        }

        // Pin the batch, in the graph's input order.
        let mut model = model;
        for (i, role) in inputs.iter().enumerate() {
            let [rows, cols] = spec.tensor_shape(*role);
            model = model
                .with_input_fact(i, f32::fact([1, rows, cols]).into())
                .map_err(|e| {
                    anyhow!(
                        "the waveform model {} does not accept the declared {} input \
                         [1, {rows}, {cols}]: {e}",
                        path.display(),
                        role.name()
                    )
                })?;
        }
        let plan = model
            .into_optimized()
            .map_err(|e| {
                anyhow!(
                    "cannot optimize the waveform model {}: {e}. tract parses a graph and \
                     then analyses it, so a failure here is typically shape inference \
                     rather than a missing op — see `escapepod_classify::waveform_net` \
                     and rnabioco/escapepod-models#96",
                    path.display()
                )
            })?
            .into_runnable()
            .map_err(|e| anyhow!("cannot plan the waveform model {}: {e}", path.display()))?;

        let net = Self { plan, inputs };
        net.probe(spec)?;
        Ok(net)
    }

    /// Run one zeroed chunk and insist the output is a single `[1, 1]` logit.
    ///
    /// The same discipline the other ONNX loaders here use, for the same
    /// reason: a graph with a two-class softmax head, or a per-timestep
    /// output, has to fail at load with the file named rather than downstream,
    /// where a wrong shape becomes a wrong probability on every read.
    fn probe(&self, spec: &WaveformSpec) -> Result<()> {
        let zero = Chunk {
            signal: vec![0.0; prod(spec.tensor_shape(WaveformTensor::Signal))],
            sequence: vec![0.0; prod(spec.tensor_shape(WaveformTensor::Sequence))],
            sequence_rows: spec.tensor_shape(WaveformTensor::Sequence)[0],
            sequence_cols: spec.tensor_shape(WaveformTensor::Sequence)[1],
            features: vec![0.0; prod(spec.tensor_shape(WaveformTensor::Features))],
            base_index: 0,
            focus_signal_pos: 0,
        };
        self.logit(&zero, spec)
            .map_err(|e| anyhow!("the waveform model failed on a zeroed probe chunk: {e}"))?;
        Ok(())
    }

    /// Score one chunk, returning the graph's raw logit.
    ///
    /// Raw on purpose: the *polarity* (which class the logit is of) and the
    /// shipped Platt calibration are both bundle-level decisions, and applying
    /// either here would put them out of reach of the caller that has to
    /// report which was applied.
    pub fn logit(&self, chunk: &Chunk, spec: &WaveformSpec) -> Result<f64> {
        let mut values: TVec<TValue> = tvec!();
        for role in &self.inputs {
            let [rows, cols] = spec.tensor_shape(*role);
            let data: &[f32] = match role {
                WaveformTensor::Signal => &chunk.signal,
                WaveformTensor::Sequence => &chunk.sequence,
                WaveformTensor::Features => &chunk.features,
            };
            if data.len() != rows * cols {
                bail!(
                    "the assembled {} tensor is {} values, but the geometry says \
                     {rows} x {cols}",
                    role.name(),
                    data.len()
                );
            }
            let t = Tensor::from_shape(&[1, rows, cols], data)
                .map_err(|e| anyhow!("cannot build the {} tensor: {e}", role.name()))?;
            values.push(t.into());
        }

        let out = self
            .plan
            .run(values)
            .map_err(|e| anyhow!("waveform model inference failed: {e}"))?;
        let view = out[0]
            .to_plain_array_view::<f32>()
            .map_err(|e| anyhow!("the waveform model output is not f32: {e}"))?;
        let data: Vec<f32> = view.iter().copied().collect();
        if data.len() != 1 {
            bail!(
                "the waveform model emitted {} values (shape {:?}); the contract is one \
                 BCE logit per read, so this is a differently-headed graph",
                data.len(),
                out[0].shape()
            );
        }
        Ok(data[0] as f64)
    }
}

fn prod(shape: [usize; 2]) -> usize {
    shape[0] * shape[1]
}
