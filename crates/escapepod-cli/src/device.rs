//! `--device auto|cpu|gpu` — where each GPU-capable stage runs, and why.
//!
//! This module exists because "is the GPU on?" used to be answerable only by
//! reading a log line that was never printed. The old `--gpu` boolean failed in
//! both directions at once:
//!
//! * **Silent CPU.** GPU was opt-in, and the flag only *existed* when the
//!   matching Cargo feature was compiled in. A user who forgot it — or who ran
//!   a release binary with no GPU code at all — got the CPU path and no
//!   indication that anything was slower than it had to be. Boundary-CNN
//!   detection is ~7x slower on the CPU; a downstream consumer measured 37
//!   minutes on one flowcell before working out why (#270).
//! * **Silent GPU fallback.** `--gpu` was a *request*. onnxruntime registers the
//!   CUDA execution provider on a best-effort basis and commits to the CPU
//!   provider when it cannot, so a broken runtime — a CUDA 13 `libonnxruntime`
//!   against a CUDA 12 library set, say (#278) — produced a correct, slow run
//!   that looked accelerated.
//!
//! The fix is one flag with three values and, more importantly, a **stage** at
//! the other end of it: `auto` is not "GPU if there is one", it is "GPU for the
//! stages where the GPU actually wins". See [`Stage::auto_prefers_gpu`] — the
//! DTW carve-out there is deliberate and measured, not an oversight.
//!
//! # This module compiles in every build
//!
//! Nothing here is behind `#[cfg(feature = "…")]`, including the warning path.
//! That is load-bearing: the musl release artifacts contain no GPU code
//! whatsoever, and they are exactly the builds where a user is most likely to
//! wonder why detection is taking half an hour. `--device gpu` on such a binary
//! must produce a sentence explaining that the feature is not compiled in — not
//! `error: unexpected argument '--device'`. Feature detection is done with
//! `cfg!(…)`, which evaluates in all builds, rather than `#[cfg(…)]`, which
//! deletes the code that would have explained itself.

use std::fmt;

/// Where the user wants GPU-capable work to run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, clap::ValueEnum)]
pub enum Device {
    /// GPU for the stages where it wins, when one is actually usable; CPU
    /// otherwise. Placing a stage never fails.
    #[default]
    Auto,
    /// CPU everywhere, even with a device visible.
    Cpu,
    /// GPU or fail. Not a preference — a requirement.
    Gpu,
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Auto => "auto",
            Self::Cpu => "cpu",
            Self::Gpu => "gpu",
        })
    }
}

/// The `--device` flag plus its two aliases, flattened into every command that
/// has a GPU-capable stage.
///
/// `--gpu` is the old spelling and stays hidden but functional; `--cpu` is the
/// symmetric convenience. Both conflict with `--device`, so there is never a
/// question of which one won.
#[derive(Debug, Default, clap::Args)]
pub struct DeviceArgs {
    /// Where GPU-capable stages run: `auto` (default), `cpu`, or `gpu`.
    ///
    /// `auto` uses the GPU only for the stages that are measurably faster on it
    /// — CNN/TCN adapter detection (~7x) and the CTC-CRF encoder (~4x) — and
    /// only when the corresponding Cargo feature is compiled in and a CUDA
    /// device is visible. DTW classification stays on the CPU under `auto`
    /// because the CPU is faster there (113 s on 64 cores vs 132 s on an A30).
    ///
    /// `gpu` is a requirement, not a request: if the feature is missing, no
    /// device is visible, or onnxruntime cannot register its CUDA execution
    /// provider, the run fails instead of silently continuing on the CPU. It
    /// also opts DTW into the GPU.
    ///
    /// `cpu` forces everything onto the CPU even with a device present.
    #[arg(
        long,
        value_enum,
        value_name = "auto|cpu|gpu",
        conflicts_with_all = ["gpu", "cpu"],
        help_heading = "Advanced Options"
    )]
    pub device: Option<Device>,

    /// Deprecated alias for `--device gpu`. Kept so existing scripts keep
    /// working; emits a warning when used.
    #[arg(long, hide = true, conflicts_with = "cpu")]
    pub gpu: bool,

    /// Alias for `--device cpu`.
    #[arg(long, help_heading = "Advanced Options")]
    pub cpu: bool,
}

impl DeviceArgs {
    /// The device this run asked for, folding in the two aliases.
    ///
    /// Call once per command, early — [`Stage`] placement and the CPU-cost
    /// warnings both hang off the result, and the point of the warnings is that
    /// they arrive at second one rather than after the run.
    pub fn resolve(&self) -> Device {
        if self.gpu {
            tracing::warn!(
                "--gpu is deprecated; use `--device gpu`. Note the change in meaning: \
                 `--device gpu` fails if the GPU is unusable instead of quietly \
                 running on the CPU. Continuing as `--device gpu`."
            );
            return Device::Gpu;
        }
        if self.cpu {
            return Device::Cpu;
        }
        self.device.unwrap_or_default()
    }
}

/// A pipeline stage that has both a CPU and a GPU implementation.
///
/// Stages with no GPU implementation at all (GBM classification, LLR adapter
/// detection) are deliberately absent: they are not a *choice*, so they never
/// go through [`place`]. Use [`note_cpu_only`] to tell a user who asked for
/// `--device gpu` that one of those is in the path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    /// Boundary CNN/TCN adapter detection (`--method cnn`), onnxruntime CUDA.
    CnnDetect,
    /// CTC-CRF encoder inference, onnxruntime CUDA. The lattice decode's own
    /// CPU/GPU split is internal to the encoder and not modelled here.
    CrfEncoder,
    /// Batched DTW distance for classify / train-svm, via the CUDA DTW kernel.
    Dtw,
}

impl Stage {
    /// Human-readable name, used verbatim in every message about this stage.
    pub const fn label(self) -> &'static str {
        match self {
            Self::CnnDetect => "boundary CNN adapter detection",
            Self::CrfEncoder => "CTC-CRF encoder inference",
            Self::Dtw => "DTW distance",
        }
    }

    /// The Cargo feature that compiles this stage's GPU path in.
    ///
    /// One feature covers all three, for the same reason [`Self::compiled_in`]
    /// tests only one: `gpu` is atomic. This stays a per-stage accessor so the
    /// messages below have a single source of truth for the name, and so a
    /// future stage whose feature *does* differ has somewhere to say so.
    pub const fn feature(self) -> &'static str {
        match self {
            Self::CnnDetect | Self::CrfEncoder | Self::Dtw => "gpu",
        }
    }

    /// Whether this binary contains this stage's GPU path.
    ///
    /// `cfg!` and not `#[cfg]` on purpose — see the module docs. A build without
    /// the feature has to be able to *say* it lacks the feature.
    pub const fn compiled_in(self) -> bool {
        match self {
            // One feature now covers all three: `gpu` is atomic, so a
            // build either has every device path or none of them.
            Self::CnnDetect | Self::CrfEncoder | Self::Dtw => cfg!(feature = "gpu"),
        }
    }

    /// What running this stage on the CPU costs, when the GPU is the better
    /// place for it. `None` means the CPU is not a downgrade.
    ///
    /// These are end-to-end numbers from `benchmarks/README.md`, not isolated
    /// kernel times: the isolated CNN inference speedup is ~99x against tract,
    /// which would be a wildly misleading thing to print next to a wall clock.
    pub const fn cpu_cost(self) -> Option<&'static str> {
        match self {
            Self::CnnDetect => Some("~7x slower than GPU end-to-end"),
            Self::CrfEncoder => Some("~4x slower than GPU end-to-end"),
            // Not a typo and not an omission: see `auto_prefers_gpu`.
            Self::Dtw => None,
        }
    }

    /// Whether `--device auto` should send this stage to the GPU.
    ///
    /// # The DTW carve-out
    ///
    /// This is the counterintuitive one, so it is written down where the next
    /// reader will look for it: **`auto` leaves DTW on the CPU even on a machine
    /// with an idle A30.** The GPU is slower for it. Measured on 1.22 M reads,
    /// same model, same input: 113 s across 64 CPU cores against 132 s on an
    /// A30, and the GPU arm additionally holds ~2.2 GB more RSS. DTW's inner
    /// loop is a narrow sequential recurrence with almost no arithmetic per
    /// byte moved, and the CPU's cache hierarchy suits that better than the
    /// device's does; batching across reads recovers some of it but not the
    /// transfer and the launch overhead.
    ///
    /// It may still win when cores are scarce — a 4-core allocation is not the
    /// 64-core node this was measured on — which is why `--device gpu` still
    /// opts into it. `auto` picks the default that is right on the hardware
    /// people actually run this on.
    ///
    /// GBM classification is not here at all because it has no GPU path, and
    /// adding one is not planned: a 32-core CPU pool beats a single GPU stream
    /// on the tree walk by roughly 20x.
    pub const fn auto_prefers_gpu(self) -> bool {
        match self {
            Self::CnnDetect | Self::CrfEncoder => true,
            Self::Dtw => false,
        }
    }
}

/// Why a GPU-capable stage ended up on the CPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuReason {
    /// `--device cpu`.
    Requested,
    /// This binary has no GPU path for the stage.
    NotCompiledIn,
    /// GPU path is compiled in, but no CUDA device is visible.
    NoDevice,
    /// A device is there and usable; the CPU is simply faster for this stage.
    FasterOnCpu,
    /// The device is fine; something else in this run rules the GPU path out.
    /// Carries the explanation, which is printed verbatim.
    Incompatible(&'static str),
}

/// Where a stage will run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Placement {
    Gpu,
    Cpu(CpuReason),
}

impl Placement {
    /// Convenience for the `if gpu { … } else { … }` branches at the call sites.
    pub fn is_gpu(self) -> bool {
        matches!(self, Self::Gpu)
    }
}

/// Whether at least one CUDA device is visible to this process.
///
/// Always `false` in a build with no GPU features — there is no cudarc to ask,
/// and the answer could not be acted on anyway. Callers reach this only after
/// [`Stage::compiled_in`] has already said yes, so the two never disagree in a
/// way anyone sees.
pub fn cuda_available() -> bool {
    #[cfg(feature = "gpu")]
    {
        escapepod_demux::cuda::device_visible()
    }
    #[cfg(not(feature = "gpu"))]
    {
        false
    }
}

/// Under `--device gpu`, make onnxruntime's CUDA EP registration fatal.
///
/// The device probe above proves the CUDA *driver* works. It says nothing about
/// whether `ORT_DYLIB_PATH` points at a CUDA-enabled `libonnxruntime`, and that
/// is the failure that actually bit us (#278): ort registers the CUDA provider
/// best-effort and falls through to the CPU provider, so the run is correct and
/// silently unaccelerated. `escapepod_demux::require_cuda_ep` flips that to an
/// error for every session built afterwards.
///
/// # What this does not catch, and where that is caught instead
///
/// Registering the provider only appends a factory to the session options.
/// Measured on an A30 with `LD_LIBRARY_PATH` stripped: ort logs `Successfully
/// registered CUDAExecutionProvider`, the session builds, and then *every* node
/// fails with `NOT_IMPLEMENTED : cuDNN is unavailable` — the kernels dlopen their
/// libraries later, inside the first `Conv`, long after this switch has had its
/// say. Nothing here can pre-check that, so `demux detect --method cnn` catches
/// it downstream: an all-reads-failed GPU run is an error rather than a
/// boundaries CSV of `adapter_end=0`.
fn arm_strict_ep() {
    #[cfg(feature = "gpu")]
    escapepod_demux::require_cuda_ep();
}

/// Decide where `stage` runs under `device`.
///
/// Errors only under `--device gpu`, and only for the two causes it can settle
/// up front: feature missing, or no device. onnxruntime EP registration failure
/// cannot be detected here — it happens when the session is built — so this arms
/// the strict-registration switch instead and the error arrives from the loader.
/// See [`arm_strict_ep`] for the third failure, which neither can reach.
pub fn place(device: Device, stage: Stage) -> anyhow::Result<Placement> {
    match device {
        Device::Cpu => Ok(Placement::Cpu(CpuReason::Requested)),
        Device::Gpu => {
            if !stage.compiled_in() {
                anyhow::bail!(
                    "--device gpu cannot run {}: this binary was built without the \
                     `{}` Cargo feature, so it contains no GPU code for that stage. \
                     Rebuild with `--features {}` — one flag covers every GPU path — \
                     or use `--device auto` to run on the CPU instead.",
                    stage.label(),
                    stage.feature(),
                    stage.feature(),
                );
            }
            if !cuda_available() {
                anyhow::bail!(
                    "--device gpu cannot run {}: no CUDA device is visible. Check \
                     `nvidia-smi` and `CUDA_VISIBLE_DEVICES`; under SLURM the job needs \
                     a GPU allocation (`--gres=gpu:1`). Use `--device auto` to fall \
                     back to the CPU automatically.",
                    stage.label(),
                );
            }
            arm_strict_ep();
            Ok(Placement::Gpu)
        }
        Device::Auto => {
            if !stage.auto_prefers_gpu() {
                Ok(Placement::Cpu(CpuReason::FasterOnCpu))
            } else if !stage.compiled_in() {
                Ok(Placement::Cpu(CpuReason::NotCompiledIn))
            } else if !cuda_available() {
                Ok(Placement::Cpu(CpuReason::NoDevice))
            } else {
                Ok(Placement::Gpu)
            }
        }
    }
}

/// [`place`], plus the log line that is the whole point of this change.
///
/// Every command calls this instead of `place` unless it has a reason not to.
/// A GPU-capable stage that lands on the CPU says so, in one line, with the
/// cause distinguished — "not compiled in" and "no device" need different
/// fixes, and conflating them is how you get someone rebuilding a binary that
/// was already correct.
pub fn place_and_report(device: Device, stage: Stage) -> anyhow::Result<Placement> {
    let placement = place(device, stage)?;
    match placement {
        Placement::Gpu => {
            tracing::info!(
                "{} {} on GPU",
                crate::style::label("Device:"),
                stage.label()
            );
            // `--device gpu` is the only way DTW reaches the device, and the
            // measurement that keeps it out of `auto` applies just as much when
            // it is asked for explicitly. Say so once, here, so `demux classify`
            // and the fused pipeline cannot drift on the wording.
            if stage == Stage::Dtw {
                tracing::warn!(
                    "GPU DTW is experimental and usually slower than a full CPU node \
                     (113 s on 64 cores vs 132 s on an A30 for 1.22M reads, plus \
                     ~2.2 GB more RSS). It helps mainly when CPU cores are scarce."
                );
            }
        }
        Placement::Cpu(reason) => report_cpu(stage, reason),
    }
    Ok(placement)
}

/// The CPU-cost warning. Split out so the wording lives in one place.
fn report_cpu(stage: Stage, reason: CpuReason) {
    let Some(cost) = stage.cpu_cost() else {
        // The CPU is the right answer for this stage; nothing to warn about.
        tracing::debug!("{} runs on the CPU", stage.label());
        return;
    };
    match reason {
        // Asked for explicitly. Still worth a line — a wrapper script may be
        // passing `--device cpu` the user has forgotten about — but not a
        // warning, because nothing is wrong.
        CpuReason::Requested => tracing::info!(
            "{} {} on CPU ({}) — requested with `--device cpu`",
            crate::style::label("Device:"),
            stage.label(),
            cost,
        ),
        CpuReason::NotCompiledIn => tracing::warn!(
            "{} is running on the CPU ({}); this build has no `{}` feature. \
             Rebuild with `--features {}` for GPU inference, or pass `--device cpu` \
             to silence this.",
            stage.label(),
            cost,
            stage.feature(),
            stage.feature(),
        ),
        CpuReason::NoDevice => tracing::warn!(
            "{} is running on the CPU ({}); no CUDA device is visible. Allocate a GPU \
             (SLURM: `--gres=gpu:1`) and check `nvidia-smi`, or pass `--device cpu` to \
             silence this.",
            stage.label(),
            cost,
        ),
        // Not the device's fault and not a build problem, so not a warning — but
        // it still has to be said, or the run looks like it lost its GPU.
        CpuReason::Incompatible(why) => tracing::info!(
            "{} {} on CPU ({}) — {}",
            crate::style::label("Device:"),
            stage.label(),
            cost,
            why,
        ),
        // Unreachable for a stage with a `cpu_cost`, but not worth an `unwrap`.
        CpuReason::FasterOnCpu => tracing::debug!("{} runs on the CPU", stage.label()),
    }
}

/// [`place_and_report`] for a stage whose GPU path is ruled out by something
/// other than the hardware — another flag in this run that the device path
/// cannot serve.
///
/// Under `--device auto` this is a reason to stay on the CPU, not a reason to
/// fail: the alternative is a run that dies on a GPU node for asking a
/// diagnostic question it could have answered. Under an explicit `--device gpu`
/// it is a genuine conflict between two things the user asked for, and errors.
pub fn place_ruled_out(
    device: Device,
    stage: Stage,
    why: &'static str,
) -> anyhow::Result<Placement> {
    if device == Device::Gpu {
        anyhow::bail!("--device gpu cannot run {}: {}", stage.label(), why);
    }
    let reason = if device == Device::Cpu {
        CpuReason::Requested
    } else {
        CpuReason::Incompatible(why)
    };
    report_cpu(stage, reason);
    Ok(Placement::Cpu(reason))
}

/// Tell a user who asked for `--device gpu` that a stage in their path has no
/// GPU implementation at all.
///
/// Not an error, and deliberately so. `--device gpu` is a requirement about
/// *stages that could run on a GPU*: it means "do not quietly fall back", not
/// "refuse to run anything that is inherently CPU work". GBM classification and
/// LLR adapter detection have no device path and never will, so failing the run
/// over them would make `--device gpu` unusable with exactly the model choices
/// it is otherwise fine with (GPU CNN detection feeding a GBM classifier, for
/// instance).
pub fn note_cpu_only(device: Device, what: &str, detail: &str) {
    if device == Device::Gpu {
        tracing::warn!("`--device gpu` does not apply to {what}: {detail}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_forces_cpu_for_every_stage() {
        for stage in [Stage::CnnDetect, Stage::CrfEncoder, Stage::Dtw] {
            assert_eq!(
                place(Device::Cpu, stage).unwrap(),
                Placement::Cpu(CpuReason::Requested)
            );
        }
    }

    /// The carve-out this whole module is about: a GPU that exists is still not
    /// used for DTW under `auto`.
    #[test]
    fn auto_never_puts_dtw_on_the_gpu() {
        assert!(!Stage::Dtw.auto_prefers_gpu());
        assert_eq!(
            place(Device::Auto, Stage::Dtw).unwrap(),
            Placement::Cpu(CpuReason::FasterOnCpu)
        );
    }

    /// `auto` never errors, whatever the build or the node.
    #[test]
    fn auto_is_infallible() {
        for stage in [Stage::CnnDetect, Stage::CrfEncoder, Stage::Dtw] {
            assert!(place(Device::Auto, stage).is_ok());
        }
    }

    /// A build without the feature must *explain* itself rather than be missing
    /// the flag — this is the musl-release case from the module docs.
    #[test]
    fn gpu_on_an_uncompiled_stage_names_the_feature() {
        for stage in [Stage::CnnDetect, Stage::CrfEncoder, Stage::Dtw] {
            if stage.compiled_in() {
                continue;
            }
            let err = place(Device::Gpu, stage).unwrap_err().to_string();
            assert!(err.contains(stage.feature()), "{err}");
            assert!(err.contains("--device gpu"), "{err}");
        }
    }

    #[test]
    fn aliases_map_onto_device() {
        let gpu = DeviceArgs {
            device: None,
            gpu: true,
            cpu: false,
        };
        assert_eq!(gpu.resolve(), Device::Gpu);
        let cpu = DeviceArgs {
            device: None,
            gpu: false,
            cpu: true,
        };
        assert_eq!(cpu.resolve(), Device::Cpu);
        assert_eq!(DeviceArgs::default().resolve(), Device::Auto);
        let explicit = DeviceArgs {
            device: Some(Device::Cpu),
            gpu: false,
            cpu: false,
        };
        assert_eq!(explicit.resolve(), Device::Cpu);
    }

    /// Every stage that is worth accelerating must be able to say what CPU
    /// costs, and the one that is not must not claim a cost it does not have.
    #[test]
    fn cpu_cost_tracks_auto_preference() {
        for stage in [Stage::CnnDetect, Stage::CrfEncoder, Stage::Dtw] {
            assert_eq!(stage.auto_prefers_gpu(), stage.cpu_cost().is_some());
        }
    }
}
