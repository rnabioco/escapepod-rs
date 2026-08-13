//! One place that knows how `ort` spells "use the CUDA execution provider".
//!
//! `ort` is still pre-release and moves this between release candidates —
//! rc.12's `ort::execution_providers::CUDAExecutionProvider` became rc.13's
//! `ort::ep::CUDA`. That is normal for an RC, but it broke the build in two
//! files at once, so the construction lives here and the next rename is a
//! one-line change rather than a hunt.
//!
//! The exact-version pin in Cargo.toml is deliberate and not the fragility:
//! without `=`, Cargo's pre-release matching lets `^2.0.0-rc.12` resolve to
//! rc.13 on any lockfile refresh, which would break the build unprompted.
//! Bumping is a decision, not an accident.

use ort::ep::{ArenaExtendStrategy, CUDA};
use ort::execution_providers::ExecutionProviderDispatch;

/// Execution providers for a CUDA session on a given device ordinal, in
/// preference order.
///
/// Session construction still falls back to CPU if the provider cannot be
/// registered (no device, no CUDA-enabled `libonnxruntime` on
/// `ORT_DYLIB_PATH`), so this is a request rather than a requirement.
///
/// Ordinals are indices into the *visible* devices, so they already respect
/// `CUDA_VISIBLE_DEVICES` — under SLURM `--gres=gpu:1` the one allocated GPU is
/// ordinal 0 whichever physical card it is. This is the same numbering cudarc
/// uses, which is what lets an encoder session and its lattice-decode context be
/// placed on the same device by passing the same ordinal to both.
///
/// # Do not "clean up" the defaults
///
/// `CUDA::default()` leaves onnxruntime's `use_tf32` **on**, and on Ampere that
/// is load-bearing: measured on an A30, turning it off costs **1.35x** on the
/// CRF encoder. TF32 keeps an fp32 exponent and rounds the mantissa to 10 bits
/// for the tensor-core matmuls, so unlike fp16 it is not a precision decision
/// anyone has to defend — it is free speed that is already switched on. Adding
/// `.with_tf32(false)` would look like a conservatism improvement and would in
/// fact be a silent 35% regression.
///
/// There is no fp16 option here to reach for, and that is correct: ORT's CUDA EP
/// cannot convert an fp32 graph at runtime, and an offline-converted fp16
/// encoder was measured and **rejected** — see `crf::encoder_gpu` for why.
///
/// # The arena strategy is the one default we do override
///
/// Read alongside the section above, this looks inconsistent. It is not: TF32 is
/// a default that is right and merely looks wrong, and this is a default that is
/// wrong for a streaming workload and looks fine.
///
/// onnxruntime defaults to `kNextPowerOfTwo`, which **doubles** the device arena
/// every time it has to extend. That suits a fixed-shape server: the arena
/// reaches its working size in a few steps and stays there. It does not suit a
/// demux run, where each `Session::run` sees whatever batch the stream handed it
/// — full `batch_rows` mid-file, a short tail at every POD5 boundary, and
/// whatever the halve-and-retry produces after an OOM. Each new shape asks for a
/// block no existing bin can serve, the arena doubles rather than taking what
/// was asked, and because an arena never returns memory the high-water mark only
/// climbs. It ends holding the whole device in bins too small for the next
/// request, and every subsequent allocation fails while `nvidia-smi` reports
/// 100% memory at 0% utilisation.
///
/// That is a function of how LONG the stream is, not how big the batch is, which
/// is why it hides in testing: measured on RNA004 nbc16, runs of 1.0M and 1.8M
/// reads finished clean on a 24 GB A30, and a 4.88M-read run wedged 61% of the
/// way in after 409 failed `Reshape` allocations of ~780 MB.
///
/// `kSameAsRequested` extends by exactly the amount asked for. Fragmentation
/// stops compounding, and the arena tracks real demand instead of the largest
/// power of two above it. It costs a little allocator time on shapes never seen
/// before and changes no numerics whatsoever — the graph, the inputs and the
/// arithmetic are untouched.
///
/// This is prevention, which `crf::encoder_gpu::DEVICE_ROW_BUDGET` already
/// argues is the only thing that works here: the halve-and-retry backstop
/// recognised all 409 of those failures and still could not save the run,
/// because by then the context was wedged.
pub(crate) fn cuda_providers_on(device_id: i32) -> [ExecutionProviderDispatch; 1] {
    [CUDA::default()
        .with_device_id(device_id)
        .with_arena_extend_strategy(ArenaExtendStrategy::SameAsRequested)
        .build()]
}
