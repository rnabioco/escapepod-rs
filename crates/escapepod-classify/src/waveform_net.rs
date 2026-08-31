// SPDX-License-Identifier: MIT

//! The windowed variant's ONNX graph, through onnxruntime.
//!
//! # Why not tract — and why this is a bridge, not a destination
//!
//! Every other ONNX graph `escpod` runs goes through tract, which is statically
//! linked and needs nothing at run time. This one cannot, and the reason is
//! worth stating precisely: **tract runs these ops fine; its shape inference
//! cannot close this export.** Measured on `charging_tcn_rna004@v0.1.0`, five
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
//! The offending subgraph is `Unsqueeze` -> `Transpose` -> `GatherND` ->
//! `Transpose` -> `Where`, and it **is** `adaptive_avg_pool1d`, as the
//! **dynamo** exporter lowers an output size that does not divide the input.
//! Every input to it is a constant initializer (a `(1,1,11,37,2)` index
//! tensor, an `(11,37)` bool mask), so the pattern is entirely static and
//! tract still cannot close it.
//!
//! Those two constants are what identify it, and they are worth reading
//! closely, because this file previously blamed `nn.MultiheadAttention`'s mask
//! handling and explicitly ruled the pool out. `charging_tcn_rna004` pools 390
//! down to 11. PyTorch's bin rule is `[floor(j*L/K), ceil((j+1)*L/K))`, which
//! for 390 -> 11 gives eleven bins of width 36 or 37 -- so a gather that
//! evaluates every bin at once needs an `(11, 37)` index grid and an `(11, 37)`
//! mask marking the slot the 36-wide bins do not use. That is exactly the pair
//! above. An attention mask is shaped by sequence length and head count and
//! would never be `11 x 37`; 11 and 37 are the pool's output width and its
//! widest bin. `nn.MultiheadAttention` in fact exports as plain
//! `MatMul`/`Softmax`/`MatMul`.
//!
//! The trap is that a non-dividing adaptive pool does not lower to *any* ONNX
//! pooling op, so grepping the graph for one finds nothing and the ragged-bin
//! gather looks like it must have come from somewhere else. The layers named
//! in the model config are real and they are what tract dies on -- they just
//! do not appear under a pooling name. rnabioco/escapepod-rs#306 had it right.
//!
//! Diagnosed and fixed upstream in rnabioco/leech#233. There were **two**
//! independent causes, and the table above shows both: the pool, and the
//! `value_info` dynamo writes for every intermediate carrying the batch axis
//! as the symbol `batch` (which is the `node_conv1d` row -- a consumer that
//! pins the batch cannot unify against a symbol, so it fails at the *first*
//! convolution before ever reaching the gather). leech 0.10.0 emits the pool
//! as a single matmul against a constant segment-mean matrix and strips
//! `value_info` on every export. Neither is visible to a round-trip check
//! against onnxruntime, which loads the old graph happily -- which is why this
//! surfaced here, at integration, rather than at build time.
//!
//! Neither standard rewrite helps, so this cannot be papered over at load time
//! the way [`crate::fnn`]'s `hoist_conv_padding` papers over padded
//! convolutions: onnx-simplifier folds away every `Shape` node (479 -> 428
//! nodes) and tract fails at the same `GatherND`; onnxruntime's own optimiser
//! keeps it and adds hardware-specific fusions.
//!
//! So this path uses `ort` (onnxruntime), exactly as `escapepod-demux`'s CRF
//! encoder does, and carries the same runtime cost: `ort` is built
//! `load-dynamic`, so onnxruntime is **dlopened at run time** and
//! `ORT_DYLIB_PATH` must point at a `libonnxruntime.so`. Nothing is needed at
//! build time.
//!
//! That cost is why it is a separate feature (`waveform-onnx`) rather than part
//! of `classify`: a build without it refuses such a bundle *by name*, with the
//! rebuild hint, instead of shipping a binary whose charging command fails at
//! the first read with a dlopen error. It also means a *released* `escpod`
//! cannot run such a bundle at all -- the shipped artifacts are static musl,
//! which cannot dlopen anything.
//!
//! Which is why this module is a bridge. The fix belongs in the export, and is
//! the fix this model family already needed once: `escapepod-models` retracted
//! "tract cannot run `Resize`" on 2026-07-27 after finding the failure was
//! shape inference over a runtime-computed shape, that our export was the
//! cause, and that one line fixed it with no retrain (and a 6x speedup in
//! tract). This is the same shape of mistake and the same shape of fix: a
//! re-export from leech >= 0.10.0, same weights and no retrain, would let this
//! file and the `ort` dependency go away -- rnabioco/escapepod-models#96.
//!
//! Measured there on the shipped `TCNDwellResidualLN` weights: 479 -> 319
//! nodes, `GatherND` 2 -> 0, `Gather` 76 -> 0, and tract loads, optimizes and
//! runs the result at batch 1 and 32, within 5.72e-06 of torch over 256 real
//! chunks with no decision disagreements. So the bridge can be removed once
//! `charging_tcn_rna004` is re-exported and shipped; what is *not* yet
//! verified is a released `escpod` running such a bundle end to end, since
//! 0.18.1 has no waveform bundle variant.
//!
//! # What is checked, and what is trusted
//!
//! The bundle declares three input tensors and one output; this probes the
//! session for all four and refuses a mismatch at load. That matters more here
//! than for a single-input graph: the three tensors are *different shapes* and
//! feeding them in the wrong order is not a shape error on two of the three, so
//! the names are resolved from the session rather than assumed positional.

use anyhow::{Result, anyhow, bail};
use std::path::Path;

use escapepod_signal::chunk::Chunk;

use crate::bundle::{WaveformSpec, WaveformTensor};

/// A loaded windowed-variant graph, ready to score chunks.
///
/// Holds a **pool** of sessions rather than one. `Session::run` takes `&mut
/// self` — onnxruntime's own guidance is one session per thread or a batched
/// call — and this pipeline scores reads under `rayon`, so a single guarded
/// session would serialise every inference behind one mutex while the assembly
/// around it stayed parallel. That is not a slow path; it is a pipeline that
/// stops scaling with cores, and it looks exactly like a correct one.
pub struct WaveformNet {
    sessions: Vec<std::sync::Mutex<ort::session::Session>>,
    /// The session's input names, in the order this runtime feeds them, paired
    /// with which assembled tensor goes in each.
    inputs: Vec<(String, WaveformTensor)>,
    output: String,
}

impl std::fmt::Debug for WaveformNet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WaveformNet")
            .field("sessions", &self.sessions.len())
            .field("inputs", &self.inputs)
            .field("output", &self.output)
            .finish()
    }
}

/// How many sessions the pool holds.
///
/// One per worker, capped: each carries its own copy of the weights and its own
/// arena, and past the point where the CPU is saturated another session is
/// memory without throughput. `ESCAPEPOD_WAVEFORM_SESSIONS` overrides it for
/// measurement.
fn pool_size() -> usize {
    if let Ok(v) = std::env::var("ESCAPEPOD_WAVEFORM_SESSIONS")
        && let Ok(n) = v.parse::<usize>()
        && n > 0
    {
        return n;
    }
    rayon::current_num_threads().clamp(1, 32)
}

impl WaveformNet {
    /// Open the graph and pin its contract against `spec`.
    pub fn load(path: &Path, spec: &WaveformSpec) -> Result<Self> {
        // One intra-op thread per session: parallelism here is across reads,
        // and letting each session spawn its own pool would oversubscribe the
        // machine by the pool size squared.
        let open = || -> Result<ort::session::Session> {
            ort::session::Session::builder()
                .map_err(|e| anyhow!("cannot create an onnxruntime session builder: {e}"))?
                .with_intra_threads(1)
                .map_err(|e| anyhow!("cannot configure the onnxruntime session: {e}"))?
                .commit_from_file(path)
                .map_err(|e| {
                    anyhow!(
                        "cannot load the waveform model {}: {e}. `ort` is built \
                         `load-dynamic`, so onnxruntime is opened at run time — set \
                         ORT_DYLIB_PATH to a libonnxruntime.so if this is a dlopen failure",
                        path.display()
                    )
                })
        };
        let session = open()?;

        // Resolve each session input to the tensor this runtime assembles for
        // it, by name. The three differ in shape, so two of the six orderings
        // would be caught by onnxruntime and the rest would not.
        let mut inputs = Vec::with_capacity(session.inputs().len());
        for input in session.inputs() {
            let role = WaveformTensor::from_name(input.name()).ok_or_else(|| {
                anyhow!(
                    "the waveform model takes an input named {:?}, which this runtime \
                     does not assemble; it produces `signal`, `sequence` and `features`",
                    input.name()
                )
            })?;
            inputs.push((input.name().to_string(), role));
        }
        for role in [
            WaveformTensor::Signal,
            WaveformTensor::Sequence,
            WaveformTensor::Features,
        ] {
            let wanted = spec.tensor_shape(role);
            let present = inputs.iter().any(|(_, r)| *r == role);
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
        if session.outputs().len() != 1 {
            bail!(
                "the waveform model has {} outputs; the contract is exactly one, a \
                 [batch, 1] logit",
                session.outputs().len()
            );
        }
        let output = session.outputs()[0].name().to_string();

        let n = pool_size();
        let mut sessions = Vec::with_capacity(n);
        sessions.push(std::sync::Mutex::new(session));
        for _ in 1..n {
            sessions.push(std::sync::Mutex::new(open()?));
        }
        tracing::debug!(
            "waveform model {}: {n} session(s) for {} rayon worker(s)",
            path.display(),
            rayon::current_num_threads()
        );

        let net = Self {
            inputs,
            output,
            sessions,
        };
        net.probe(spec)?;
        Ok(net)
    }

    /// Run one zeroed chunk and insist the output is a single `[1, 1]` logit.
    ///
    /// The same discipline the other ONNX loaders here use, for the same
    /// reason: a graph with a two-class softmax head, or a per-timestep output,
    /// has to fail at load with the file named rather than downstream, where a
    /// wrong shape becomes a wrong probability on every read.
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
        use ort::value::Tensor;

        let mut values: Vec<(&str, ort::value::DynValue)> = Vec::with_capacity(self.inputs.len());
        for (name, role) in &self.inputs {
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
            let t = Tensor::from_array(([1usize, rows, cols], data.to_vec()))
                .map_err(|e| anyhow!("cannot build the {} tensor: {e}", role.name()))?;
            values.push((name.as_str(), t.into_dyn()));
        }

        // One session per rayon worker, so parallel reads do not queue behind
        // each other. Outside a rayon pool every caller uses session 0, which
        // is the single-threaded case and correct.
        let slot = rayon::current_thread_index().unwrap_or(0) % self.sessions.len();
        let mut session = self.sessions[slot]
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let outputs = session
            .run(values)
            .map_err(|e| anyhow!("waveform model inference failed: {e}"))?;
        let out = outputs
            .get(self.output.as_str())
            .ok_or_else(|| anyhow!("the waveform model produced no {:?} output", self.output))?;
        let (shape, data) = out
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow!("the waveform model output is not f32: {e}"))?;
        if data.len() != 1 {
            bail!(
                "the waveform model emitted {} values (shape {shape:?}); the contract is \
                 one BCE logit per read, so this is a differently-headed graph",
                data.len()
            );
        }
        Ok(data[0] as f64)
    }
}

fn prod(shape: [usize; 2]) -> usize {
    shape[0] * shape[1]
}
