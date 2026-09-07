// SPDX-License-Identifier: MIT

//! The windowed variant's ONNX graph, batched through `tract-cuda`.
//!
//! This is [`crate::waveform_net::WaveformNet`]'s sibling, not a replacement:
//! that loader pins batch 1 and serves every `rayon` worker from one
//! immutable plan, which is the right shape for a CPU where batching this
//! graph buys nothing (`examples/tcn_cuda_probe.rs`). Here the opposite is
//! true — the GPU is barely faster than a CPU core at batch 1 (5.01 vs
//! 4.95 ms/read) and the entire win is the batch (11.1x at 128) — so this
//! loader pins a fixed batch instead, and pays for it with a padding and a
//! remainder problem `WaveformNet` never has: see
//! [`crate::waveform::classify_reads_gpu`] for how a ragged tail of reads is
//! kept off this path rather than padded through it.
//!
//! Two facts from the probe are load-bearing here and are not optional
//! levers:
//!
//! 1. **`expand_instance_norm` is mandatory, not a performance choice.**
//!    tract's ONNX lowering reduces `InstanceNormalization` over the batch
//!    axis as well as the spatial one; at batch > 1 that makes every read's
//!    logit depend on which reads it shared a batch with (the probe's read-0
//!    logit moved +1.47 -> +0.26 -> -0.28 across batch sizes before the
//!    rewrite). It is applied unconditionally below, with no env gate — unlike
//!    the CPU loader's hoist lever, there is no batch-1 case here where it
//!    would be a no-op, and the ~5% cost the CPU module doc measures for it
//!    was measured on the CPU path only; this loader never builds one.
//! 2. **`hoist_conv_padding` moves cuDNN onto a different, less accurate
//!    algorithm here**, unlike on the CPU where the same rewrite is
//!    bit-identical. The probe measured max |dlogit| jump from 3.1e-5 to
//!    1.7e-2 — five hundred times worse — for a further 4.1x. That is why it
//!    is gated by its own env var, `ESCAPEPOD_WAVEFORM_GPU_HOIST`, rather than
//!    the CPU path's `ESCAPEPOD_WAVEFORM_HOIST`: sharing one name would let a
//!    routine CPU re-benchmark silently change a GPU run's answers too.
//!
//! # Batch 1 is refused outright — measured wrong, not merely pointless
//!
//! The probe already said batch 1 buys nothing (0.99x — a GPU call is not
//! even faster than a CPU core for one read). What real hardware then showed,
//! against `charging_tcn_sup6_rna004@v0.1.0` (2026-09-07, A30,
//! `examples/tcn_cuda_probe.rs --batches 1,2,4,32,128 --fix-norm`), is worse
//! than pointless — it is wrong:
//!
//! ```text
//!   batch   max |dlogit| vs CPU
//!       1        4.454e-1     <-- LOOK
//!       2        2.480e-5
//!       4        2.480e-5
//!      32        2.551e-5
//!     128        2.861e-5
//! ```
//!
//! The CPU reference itself is stable across the sweep (read 0's logit is
//! batch-invariant to 3.6e-7, as it must be), so this is not noise on either
//! side: batch 1 lands 4 orders of magnitude off a reference every other
//! batch size agrees with to ~2.5e-5. The cause is not confirmed — this
//! export's decluttered graph shows tract's optimizer fusing the
//! `expand_instance_norm` rewrite into an `RmsNorm` op, 30 of them, where
//! `charging_tcn_rna004@v0.1.2`'s graph (the one the rest of this module doc's
//! numbers come from) does not, so this may be a kernel- or fusion-level bug
//! specific to that op at `N = 1`. But the fix does not need the cause: a
//! batch size with no speed upside and a demonstrated correctness failure has
//! nothing to recommend it, so [`validate_batch`] refuses anything below 2
//! rather than trusting a future bundle to avoid the same trap.
//!
//! # Batch 2 and up is *also* not yet trustworthy — on real chunks
//!
//! The sweep above uses the probe's synthetic per-role tensors: uniform
//! `[-1, 1]` random draws, independent per batch element. Scoring **real**
//! reads from `charging_tcn_sup6_rna004@v0.1.0` through the actual
//! [`crate::waveform::classify_reads_gpu`] path at batch 2 disagrees with the
//! CPU scorer on more than half the reads it scored — not a tolerance gap, a
//! flipped call (0.999 CPU vs 0.06-0.11 GPU on several reads). NaN and
//! near-zero-variance feature channels are both ruled out; the leading
//! unconfirmed hypothesis is that real feature values (variance up to 1e5 in
//! the raw-level channels, against the probe's ~0.33) trigger a
//! numerically-unstable reduction inside whichever kernel evaluates the
//! decluttered `RmsNorm` fusion — a regime the probe's synthetic data never
//! exercises. Full writeup, evidence and next steps:
//! rnabioco/escapepod-rs#343.
//!
//! **This is why `commands/classify.rs` calls `place_ruled_out` rather than
//! `place_and_report` for this stage** — `--device gpu` is refused outright
//! rather than routed here, regardless of what batch size is requested. Do
//! not wire that call site back up to `place_and_report` until #343 is
//! resolved and re-verified on real, not synthetic, chunks.

use anyhow::{Result, anyhow, bail};
use std::path::Path;
use std::sync::Arc;

use tract_core::transform::ModelTransform;
use tract_onnx::prelude::*;
use tract_onnx::tract_core::model::TypedRunnableModel;

use escapepod_demux::onnx_rewrite::{expand_instance_norm, hoist_conv_padding};
use escapepod_signal::chunk::Chunk;

use crate::bundle::{WaveformSpec, WaveformTensor};
use crate::waveform_net::{pack_batch, resolve_inputs, unpack_logits};

/// A loaded windowed-variant graph, pinned to `batch` reads per call.
///
/// Unlike [`crate::waveform_net::WaveformNet`], batch is fixed at load time —
/// a tract-cuda plan's batch dimension is baked into the compiled plan, so a
/// different batch needs a different `WaveformNetGpu`, not a different call.
pub struct WaveformNetGpu {
    plan: Arc<TypedRunnableModel>,
    /// Which assembled tensor feeds each graph input, in the graph's own
    /// input order — resolved by name at load, exactly as `WaveformNet` does
    /// (`resolve_inputs`, shared with it verbatim).
    inputs: Vec<WaveformTensor>,
    batch: usize,
}

impl std::fmt::Debug for WaveformNetGpu {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WaveformNetGpu")
            .field("inputs", &self.inputs)
            .field("batch", &self.batch)
            .finish_non_exhaustive()
    }
}

impl WaveformNetGpu {
    /// Open the graph, move it onto CUDA, and pin its contract against `spec`
    /// at exactly `batch` reads per call.
    pub fn load(path: &Path, spec: &WaveformSpec, batch: usize) -> Result<Self> {
        validate_batch(batch)?;

        let onnx = tract_onnx::onnx();
        let mut proto = onnx
            .proto_model_for_path(path)
            .map_err(|e| anyhow!("cannot read the waveform model {}: {e}", path.display()))?;

        // Mandatory — see the module doc's point 1. Not conditioned on
        // `batch`: even a `WaveformNetGpu` loaded at batch 1 goes through
        // this same code path, so there is no size at which skipping it would
        // be safe *and* save anything (a batch-1 GPU plan is not what this
        // loader is for, but nothing stops one from being built).
        expand_instance_norm(&mut proto, 3);

        // Off by default — see the module doc's point 2. Deliberately a
        // different env var from the CPU loader's `ESCAPEPOD_WAVEFORM_HOIST`.
        let hoisted = if std::env::var_os("ESCAPEPOD_WAVEFORM_GPU_HOIST").is_some() {
            hoist_conv_padding(&mut proto, batch)
        } else {
            0
        };
        if hoisted > 0 {
            tracing::warn!(
                "waveform model {}: hoisted the padding out of {hoisted} convolution(s) \
                 (ESCAPEPOD_WAVEFORM_GPU_HOIST) — on the GPU this changes the logit, not \
                 just the speed (measured max |dlogit| ~1.7e-2 at batch 128); it is not the \
                 same lever as ESCAPEPOD_WAVEFORM_HOIST",
                path.display()
            );
        }

        let model = onnx
            .model_for_proto_model(&proto)
            .map_err(|e| anyhow!("cannot parse the waveform model {}: {e}", path.display()))?;

        // Resolve inputs/outputs by name against `spec`, exactly as the CPU
        // loader does — see `resolve_inputs`.
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
                .with_input_fact(i, f32::fact([batch, rows, cols]).into())
                .map_err(|e| {
                    anyhow!(
                        "the waveform model {} does not accept the declared {} input \
                         [{batch}, {rows}, {cols}]: {e}",
                        path.display(),
                        role.name()
                    )
                })?;
        }

        let mut typed = model
            .into_typed()
            .map_err(|e| anyhow!("cannot type the waveform model {}: {e}", path.display()))?
            .into_decluttered()
            .map_err(|e| {
                anyhow!(
                    "cannot declutter the waveform model {}: {e}",
                    path.display()
                )
            })?;

        tract_cuda::CudaTransform
            .transform(&mut typed)
            .map_err(|e| {
                anyhow!(
                    "cannot move the waveform model {} onto CUDA: {e}",
                    path.display()
                )
            })?;

        let plan = typed
            .into_optimized()
            .map_err(|e| {
                anyhow!(
                    "cannot optimize the waveform model {} for CUDA: {e}",
                    path.display()
                )
            })?
            .into_runnable()
            .map_err(|e| {
                anyhow!(
                    "cannot plan the waveform model {} for CUDA: {e}",
                    path.display()
                )
            })?;

        let net = Self {
            plan,
            inputs,
            batch,
        };
        net.probe(spec)?;
        Ok(net)
    }

    /// The fixed batch size this scorer was compiled for.
    pub fn batch_size(&self) -> usize {
        self.batch
    }

    /// Run one zeroed chunk through a full batch and insist on exactly one
    /// logit back, the same load-time discipline `WaveformNet::probe` uses —
    /// and, on top of that, a warm-up: NVRTC JIT-compiles tract-cuda's
    /// kernels on first use, so this keeps that latency out of the first real
    /// batch of a run.
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
        let logits = self
            .logits(&[&zero], spec)
            .map_err(|e| anyhow!("the waveform model failed on a zeroed probe batch: {e}"))?;
        if logits.len() != 1 {
            bail!(
                "a probe batch of one chunk returned {} logits back",
                logits.len()
            );
        }
        Ok(())
    }

    /// Score up to `batch_size()` chunks in one GPU call.
    ///
    /// `chunks.len()` may be less than `batch_size()` — the trailing rows are
    /// zero-padded (see [`pack_batch`]) and sliced back off on the way out —
    /// but never more; that is a caller bug; see
    /// [`crate::waveform::classify_reads_gpu`] for why a shorter group is
    /// routed to the CPU scorer instead of padded through here.
    pub fn logits(&self, chunks: &[&Chunk], spec: &WaveformSpec) -> Result<Vec<f64>> {
        if chunks.is_empty() {
            bail!("logits() called with zero chunks");
        }
        if chunks.len() > self.batch {
            bail!(
                "{} chunks given but this GPU scorer is compiled for batch {}",
                chunks.len(),
                self.batch
            );
        }

        let mut values: TVec<TValue> = tvec!();
        for role in &self.inputs {
            let [rows, cols] = spec.tensor_shape(*role);
            let buf = pack_batch(*role, chunks, spec, self.batch)?;
            let t = Tensor::from_shape(&[self.batch, rows, cols], &buf)
                .map_err(|e| anyhow!("cannot build the {} tensor: {e}", role.name()))?;
            values.push(t.into());
        }

        let out = tract_cuda::with_cuda_stream(|_| self.plan.run(values))
            .map_err(|e| anyhow!("waveform model GPU inference failed: {e}"))?;
        let view = out[0]
            .to_plain_array_view::<f32>()
            .map_err(|e| anyhow!("the waveform model output is not f32: {e}"))?;
        unpack_logits(view.iter().copied(), chunks.len())
    }
}

fn prod(shape: [usize; 2]) -> usize {
    shape[0] * shape[1]
}

/// Refuse a batch size with nothing to recommend it — see the module doc's
/// "Batch 1 is refused outright" for the measurement this enforces.
fn validate_batch(batch: usize) -> Result<()> {
    if batch < 2 {
        bail!(
            "a GPU batch size of {batch} is refused: batch 1 has no speed benefit \
             (measured 0.99x a single CPU core) and is measurably wrong for at least \
             one shipped bundle (max |dlogit| 4.45e-1 against a CPU reference that is \
             itself stable to 3.6e-7 — see `waveform_net_gpu`'s module doc). Every \
             batch size from 2 to 128 measured agrees with the CPU scorer to ~2.5e-5. \
             Use a batch of at least 2 (the default is 128)."
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::validate_batch;

    #[test]
    fn refuses_batch_sizes_below_two() {
        assert!(validate_batch(0).is_err());
        assert!(validate_batch(1).is_err());
    }

    #[test]
    fn accepts_batch_sizes_from_two_up() {
        assert!(validate_batch(2).is_ok());
        assert!(validate_batch(128).is_ok());
    }
}
