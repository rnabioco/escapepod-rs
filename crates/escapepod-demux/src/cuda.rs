//! Is there a CUDA device here at all?
//!
//! One cheap, dependency-light probe, separate from every path that actually
//! *uses* a device. `--device auto` has to answer "should this stage go to the
//! GPU?" before it has loaded a model, built an ORT session or compiled an
//! NVRTC kernel — and it has to answer it without the several-second CUDA/cuDNN
//! initialisation those paths pay, because under `auto` the answer is very often
//! "no" and the run should not have stalled to find that out.
//!
//! Compiled whenever cudarc is in the graph — that is `cnn-gpu`, `crf-gpu` or
//! `gpu`, since all three pull it in. A build with none of them has no device
//! code at all, so the caller already knows the answer without asking.

use std::sync::OnceLock;

use cudarc::driver::CudaContext;

/// How many CUDA devices this process can see, or `None` if the driver cannot be
/// reached at all (no driver, no device node, a container without `/dev/nvidia*`).
///
/// `None` and `Some(0)` are different facts and both mean "no GPU here", so
/// callers that only want a yes/no should use [`device_visible`].
///
/// This is the *visible* count, so it already honours `CUDA_VISIBLE_DEVICES`:
/// under SLURM `--gres=gpu:1` it is 1 no matter how many cards the node holds.
/// Ordinals `0..count` index the same devices onnxruntime's `device_id` does,
/// which is what lets an encoder session and its lattice-decode context be
/// placed together by passing the same ordinal to both.
///
/// # Why this checks `is_culib_present` first
///
/// `CudaContext::device_count()` does not return an `Err` when no CUDA
/// driver library (`libcuda.so`) is discoverable at all — cudarc's own
/// dlopen wrapper `panic!`s in that case
/// (`cudarc-0.19.9/src/driver/sys/mod.rs`'s `culib()`), which this
/// function's own doc above used to claim was handled gracefully. That is
/// the same failure *shape* as rnabioco/escapepod-rs#347 (there, the CUDA
/// *runtime*, `libcudart`, one level further along the same stage) — see
/// `escapepod_classify::waveform_net_gpu::cudart_runtime_probe`'s doc for
/// the full explanation. It matters here specifically because this
/// function backs `--device auto`, whose entire contract is "never fails" —
/// a `-gpu`-featured binary run on a machine with no NVIDIA driver at all
/// (a laptop, a login node, this workspace's own CPU-only CI/build hosts)
/// is exactly the case `auto` exists to fall back from, not crash on.
pub fn visible_device_count() -> Option<usize> {
    // Safety: `is_culib_present` only attempts to `dlopen` a shared library
    // by name and reports whether that succeeded — no pointers from it are
    // ever dereferenced.
    if !unsafe { cudarc::driver::sys::is_culib_present() } {
        return None;
    }
    CudaContext::device_count().ok().map(|n| n.max(0) as usize)
}

/// Whether at least one CUDA device is visible, cached for the process.
///
/// Cached because `--device auto` asks per stage and a fused run has several:
/// `cuDeviceGetCount` implies `cuInit`, which is tens of milliseconds the first
/// time and pointless to repeat. The answer cannot change mid-process — SLURM
/// does not hand a job a GPU it did not allocate — so a `OnceLock` is the whole
/// cache-invalidation story.
///
/// This is a *necessary* condition for a GPU run, not a sufficient one. The
/// onnxruntime paths additionally need a CUDA-enabled `libonnxruntime` on
/// `ORT_DYLIB_PATH`, and a driver that answers `cuInit` says nothing about that
/// — see [`crate::require_cuda_ep`] for the check that does.
pub fn device_visible() -> bool {
    static VISIBLE: OnceLock<bool> = OnceLock::new();
    *VISIBLE.get_or_init(|| visible_device_count().unwrap_or(0) > 0)
}
