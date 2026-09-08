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
//! of rnabioco/escapepod-models#96 both named while explicitly ruling the pool
//! out. Two constants identify it, and they are worth reading closely.
//! `charging_tcn_rna004` pools 390 down to 11. PyTorch's bin rule is
//! `[floor(j*L/K), ceil((j+1)*L/K))`, which for 390 -> 11 gives eleven bins of
//! width 36 or 37 — so a gather that evaluates every bin at once needs an
//! `(11, 37)` index grid and an `(11, 37)` mask marking the slot the 36-wide
//! bins do not use. That is exactly the pair the offending subgraph carries,
//! alongside a `(11,)` divisor of `[36, 36, 37, 36, 37, …]`. An attention mask
//! is shaped by sequence length and head count and would never be `11 x 37`.
//! `cross_attn` in fact exports as plain
//! `Mul`/`MatMul`/`Softmax`/`MatMul`/`Gemm`, with no mask and no gather.
//!
//! The trap is that a non-dividing adaptive pool does not lower to *any* ONNX
//! pooling op, so grepping the graph for one finds nothing and the ragged-bin
//! gather looks like it must have come from somewhere else. The layers named in
//! the model config are real and they are what tract died on — they just do not
//! appear under a pooling name. rnabioco/escapepod-rs#306's original suspect was
//! right and its retraction was not.
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
//! Neither cause is visible to a round-trip check against onnxruntime, which
//! loads the old graph happily — which is why this surfaced at integration
//! rather than at build time.
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

use escapepod_demux::onnx_rewrite::hoist_conv_padding;
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
        // Taken as a proto so the padding *can* be hoisted, but it is not
        // hoisted by default — and that is a measurement, not an oversight.
        //
        // 27 of this graph's 29 convolutions carry a causal left pad, which
        // looks like exactly the shape [`hoist_conv_padding`] exists for:
        // tract's im2col abandons its block-copy path whenever `pads != 0`,
        // and on the shipped FNN CNN hoisting was worth 6.1x (305 -> 50 us,
        // bit-identical). It does not transfer. Paired, in one job on one
        // node, `benches/charging.rs`'s `waveform` group on
        // `charging_tcn_rna004@v0.1.2`:
        //
        //     hoist on    6.4240 ms/read
        //     hoist off   5.9613 ms/read
        //
        // The rewrite splices a zero block in front of the convolution, and
        // here that block is ~(390+64) x 64 x 4 B per padded conv — about
        // 2.5 MB of extra copy per read. On the FNN CNN the activations were
        // 33 wide and the copy was free; at 390 it is not, and it is doing
        // more work than the im2col path it avoids. The convolutions are also
        // 64->64 over 390 samples, so the per-element fallback it removes is a
        // small share of a large GEMM rather than the whole cost.
        //
        // Read that pair narrowly. Run-to-run variance across *separate* jobs
        // was larger than the effect (the same "on" arm measured 5.86 ms in
        // one job and 6.42 ms in another), so what is established is that the
        // hoist is not a win here, not that it is precisely a 7% loss. The
        // default is therefore what this loader has always done.
        //
        // It is the family, not one export. `charging_tcn_rna004@v0.1.1` and
        // `@v0.1.2` are byte-identical ONNX (md5 b16810fc...), so the pair
        // above was measured on what ships; and a second, independently built
        // bundle (`charging_tcn_sup6_rna004@v0.1.0`) carries the same 29
        // convolutions, the same 27 padded, and the same causal dilation
        // ladder 1..32. So a future `waveform_model` will have this shape too,
        // and this decision travels with it rather than needing re-litigating
        // per bundle.
        //
        // What IS settled is parity: both arms return the same logit to the
        // bit (the rewrite splices in the zeros the padding already was), so
        // `ESCAPEPOD_WAVEFORM_HOIST=1` is safe to turn on for a re-measurement
        // on another machine or a future export whose shapes differ. The batch
        // argument must be the batch pinned below, because the zero block is a
        // concrete constant.
        let mut proto = onnx
            .proto_model_for_path(path)
            .map_err(|e| anyhow!("cannot read the waveform model {}: {e}", path.display()))?;
        let hoisted = if std::env::var_os("ESCAPEPOD_WAVEFORM_HOIST").is_some() {
            hoist_conv_padding(&mut proto, 1)
        } else {
            0
        };
        if hoisted > 0 {
            tracing::debug!(
                "waveform model {}: hoisted the padding out of {hoisted} convolution(s) \
                 (ESCAPEPOD_WAVEFORM_HOIST); this measured *slower* than leaving it alone",
                path.display()
            );
        }
        let model = onnx
            .model_for_proto_model(&proto)
            .map_err(|e| anyhow!("cannot parse the waveform model {}: {e}", path.display()))?;

        // Resolve each graph input to the tensor this runtime assembles for
        // it, by name — see `resolve_inputs`, shared verbatim with the GPU
        // loader so the "resolve by name, not position" safety property below
        // cannot silently drift between the two.
        let outlets = model
            .input_outlets()
            .map_err(|e| anyhow!("the waveform model declares no usable inputs: {e}"))?
            .to_vec();
        let input_names: Vec<&str> = outlets
            .iter()
            .map(|outlet| model.node(outlet.node).name.as_str())
            .collect();
        let n_outputs = model
            .output_outlets()
            .map_err(|e| anyhow!("the waveform model declares no usable outputs: {e}"))?
            .len();
        let inputs = resolve_inputs(&input_names, n_outputs, spec)?;

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

/// Build one row-major `[batch, rows, cols]` buffer for `role`, one row per
/// chunk, zero-filling the trailing `batch - chunks.len()` rows.
///
/// Shared by [`crate::waveform_net_gpu::WaveformNetGpu`]: a tract-cuda plan's
/// batch is fixed at compile time, so a shorter group of chunks is padded
/// rather than rebuilding the plan. Padding with zero rows is safe only
/// because the GPU loader unconditionally applies
/// [`escapepod_demux::onnx_rewrite::expand_instance_norm`] first — with it, a
/// zero row's own normalisation cannot perturb any real row's, since each row
/// only reduces over its own axis. Pure and tract-independent so it is
/// testable without the `cuda` feature.
#[cfg(any(feature = "cuda", test))]
pub(crate) fn pack_batch(
    role: WaveformTensor,
    chunks: &[&Chunk],
    spec: &WaveformSpec,
    batch: usize,
) -> Result<Vec<f32>> {
    let [rows, cols] = spec.tensor_shape(role);
    let cell = rows * cols;
    let mut buf = vec![0.0f32; batch * cell];
    for (i, chunk) in chunks.iter().enumerate() {
        let data: &[f32] = match role {
            WaveformTensor::Signal => &chunk.signal,
            WaveformTensor::Sequence => &chunk.sequence,
            WaveformTensor::Features => &chunk.features,
        };
        if data.len() != cell {
            bail!(
                "the assembled {} tensor is {} values, but the geometry says {rows} x {cols}",
                role.name(),
                data.len()
            );
        }
        buf[i * cell..(i + 1) * cell].copy_from_slice(data);
    }
    Ok(buf)
}

/// The first `n_valid` values of a `[batch, 1]` output, as f64 logits —
/// padding rows are sliced off here, before anything downstream sees them.
/// Takes a plain iterator rather than a tract array-view type, so, like
/// [`pack_batch`], it needs nothing beyond `waveform-onnx` to test.
#[cfg(any(feature = "cuda", test))]
pub(crate) fn unpack_logits(values: impl Iterator<Item = f32>, n_valid: usize) -> Result<Vec<f64>> {
    let data: Vec<f32> = values.collect();
    if data.len() < n_valid {
        bail!(
            "the waveform model emitted {} values, fewer than the {n_valid} requested",
            data.len()
        );
    }
    Ok(data[..n_valid].iter().map(|&v| v as f64).collect())
}

/// Resolve a graph's input names (in the graph's own input order) to the
/// tensor each one is, and cross-check against `spec`: every tensor the
/// geometry declares must be an input, and no input may be one it does not
/// declare (a `[0, _]` shape is how `tensor_shape` spells "this variant has
/// no such tensor"). Also checks there is exactly one output — the contract
/// is a single `[batch, 1]` logit.
///
/// Takes bare names rather than a tract model so it works identically
/// whether the caller's model is still an `InferenceModel` (the CPU loader,
/// and the GPU loader before `into_typed()`) or already a `TypedModel` —
/// and so it needs no tract types at all, which is what makes it testable
/// without any ONNX runtime feature linked in beyond `waveform-onnx`.
///
/// Two of the six input orderings would be caught by a shape check and the
/// rest would not, so the name is the only thing that makes this safe —
/// shared between both loaders so that property cannot silently drift.
pub(crate) fn resolve_inputs(
    input_names: &[&str],
    n_outputs: usize,
    spec: &WaveformSpec,
) -> Result<Vec<WaveformTensor>> {
    let inputs: Vec<WaveformTensor> = input_names
        .iter()
        .map(|name| {
            WaveformTensor::from_name(name).ok_or_else(|| {
                anyhow!(
                    "the waveform model takes an input named {name:?}, which this \
                     runtime does not assemble; it produces `signal`, `sequence` \
                     and `features`"
                )
            })
        })
        .collect::<Result<_>>()?;

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

    if n_outputs != 1 {
        bail!(
            "the waveform model has {n_outputs} outputs; the contract is exactly one, \
             a [batch, 1] logit"
        );
    }

    Ok(inputs)
}

#[cfg(test)]
mod gpu_support_tests {
    use super::{pack_batch, resolve_inputs, unpack_logits};
    use crate::bundle::WaveformTensor;
    use escapepod_signal::chunk::Chunk;

    /// A minimal `WaveformSpec` whose declared tensors are exactly `present`
    /// — enough to drive `tensor_shape`'s `[0, _]`-means-absent convention,
    /// nothing else about the geometry matters to `resolve_inputs`.
    fn spec_with(present: &[WaveformTensor]) -> crate::bundle::WaveformSpec {
        use escapepod_signal::chunk::{
            BaseJustify, ChunkSpec, FeatureChannel, SeqEncoding, SignalChannel, SignalNorm,
        };
        let has = |t: WaveformTensor| present.contains(&t);
        crate::bundle::WaveformSpec {
            chunk: ChunkSpec {
                signal_context: (0, 0),
                signal_len: 4,
                base_justify: BaseJustify::Center,
                signal_channels: if has(WaveformTensor::Signal) {
                    vec![SignalChannel::Current]
                } else {
                    vec![]
                },
                seq_encoding: if has(WaveformTensor::Sequence) {
                    SeqEncoding::BaseOneHot { context: 1 }
                } else {
                    SeqEncoding::None
                },
                feature_offsets: (0, 2),
                feature_channels: if has(WaveformTensor::Features) {
                    vec![FeatureChannel::Dwell]
                } else {
                    vec![]
                },
                dwell_window: 3,
            },
            reverse_signal: false,
            normalization: SignalNorm::MedianMad,
            refine: None,
            positive_class: 1,
        }
    }

    #[test]
    fn resolves_known_names_in_graph_order() {
        let spec = spec_with(&[
            WaveformTensor::Signal,
            WaveformTensor::Sequence,
            WaveformTensor::Features,
        ]);
        let got = resolve_inputs(&["sequence", "signal", "features"], 1, &spec).unwrap();
        assert_eq!(
            got,
            vec![
                WaveformTensor::Sequence,
                WaveformTensor::Signal,
                WaveformTensor::Features,
            ]
        );
    }

    #[test]
    fn refuses_an_unknown_input_name() {
        let spec = spec_with(&[WaveformTensor::Signal]);
        let err = resolve_inputs(&["signal", "mystery"], 1, &spec)
            .unwrap_err()
            .to_string();
        assert!(err.contains("mystery"), "{err}");
    }

    #[test]
    fn refuses_a_tensor_the_geometry_does_not_declare() {
        let spec = spec_with(&[WaveformTensor::Signal]);
        let err = resolve_inputs(&["signal", "sequence"], 1, &spec)
            .unwrap_err()
            .to_string();
        assert!(err.contains("sequence"), "{err}");
    }

    #[test]
    fn refuses_a_missing_declared_tensor() {
        let spec = spec_with(&[WaveformTensor::Signal, WaveformTensor::Sequence]);
        let err = resolve_inputs(&["signal"], 1, &spec)
            .unwrap_err()
            .to_string();
        assert!(err.contains("sequence"), "{err}");
    }

    #[test]
    fn refuses_more_than_one_output() {
        let spec = spec_with(&[WaveformTensor::Signal]);
        let err = resolve_inputs(&["signal"], 2, &spec)
            .unwrap_err()
            .to_string();
        assert!(err.contains("2 outputs"), "{err}");
    }

    fn filled_chunk(value: f32, len: usize) -> Chunk {
        Chunk {
            signal: vec![value; len],
            sequence: vec![],
            sequence_rows: 0,
            sequence_cols: 0,
            features: vec![],
            base_index: 0,
            focus_signal_pos: 0,
        }
    }

    #[test]
    fn pack_batch_zero_pads_the_trailing_rows() {
        let spec = spec_with(&[WaveformTensor::Signal]);
        let [rows, cols] = spec.tensor_shape(WaveformTensor::Signal);
        let cell = rows * cols;
        let chunk = filled_chunk(1.0, cell);
        let refs = [&chunk, &chunk];
        let buf = pack_batch(WaveformTensor::Signal, &refs, &spec, 4).unwrap();
        assert_eq!(buf.len(), 4 * cell);
        assert!(
            buf[..2 * cell].iter().all(|&v| v == 1.0),
            "real rows: {buf:?}"
        );
        assert!(
            buf[2 * cell..].iter().all(|&v| v == 0.0),
            "padding rows: {buf:?}"
        );
    }

    #[test]
    fn pack_batch_refuses_a_mismatched_tensor_length() {
        let spec = spec_with(&[WaveformTensor::Signal]);
        let chunk = filled_chunk(1.0, 1);
        let refs = [&chunk];
        let err = pack_batch(WaveformTensor::Signal, &refs, &spec, 2)
            .unwrap_err()
            .to_string();
        assert!(err.contains("geometry says"), "{err}");
    }

    #[test]
    fn unpack_logits_takes_only_the_valid_rows_in_order() {
        let got = unpack_logits([1.0f32, 2.0, 3.0, 4.0].into_iter(), 2).unwrap();
        assert_eq!(got, vec![1.0, 2.0]);
    }

    #[test]
    fn unpack_logits_refuses_fewer_values_than_requested() {
        let err = unpack_logits([1.0f32].into_iter(), 2)
            .unwrap_err()
            .to_string();
        assert!(err.contains("fewer than"), "{err}");
    }
}
