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

use ort::ep::CUDA;
use ort::execution_providers::ExecutionProviderDispatch;

/// Execution providers for a CUDA session, in preference order.
///
/// Session construction still falls back to CPU if the provider cannot be
/// registered (no device, no CUDA-enabled `libonnxruntime` on
/// `ORT_DYLIB_PATH`), so this is a request rather than a requirement.
pub(crate) fn cuda_providers() -> [ExecutionProviderDispatch; 1] {
    [CUDA::default().build()]
}
