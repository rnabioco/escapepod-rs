//! Periodic GPU-vs-CPU parity guard for the CRF encoder (#448).
//!
//! The GPU encoder is the same ONNX graph as the CPU one, so a library pairing
//! that computes it wrongly (#416: a cuBLAS/cuBLASLt pair from two CUDA
//! releases) gives barcode calls that look entirely plausible. Nothing else in
//! the pipeline would notice. This is the CRF counterpart of
//! `escapepod_classify::waveform_net_gpu::check_parity`: every N-th device
//! batch, a few reads are scored again on the CPU encoder and the two answers
//! compared.
//!
//! What is compared is the *decoded sequence* (the contract the native-vs-tract
//! check holds the CPU encoders to) and, when the caller scored a reference
//! panel, the `ref_logp` / `mean_logpost` the decode produced. The raw
//! encoder scores are not compared: with the lattice decode and zero-copy on,
//! they never leave the device.
//!
//! The CPU re-score runs as a rayon task so the thread that submits to the
//! device never waits for it. A disagreement is logged at `error!` from that
//! task; under `--device gpu` it is also latched, and the next device batch
//! (or [`ParityGuard::finish`]) returns it as an error.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use super::encoder::CrfError;
use super::refchain::{RefChains, ScoredDecode};

/// Env knob: check every N-th device batch. `0` turns the guard off.
pub const ENV_EVERY: &str = "ESCAPEPOD_CRF_GPU_PARITY_EVERY";
/// Default period, matching the waveform guard.
pub const DEFAULT_EVERY: usize = 64;
/// Reads re-scored per checked batch.
pub const MAX_SAMPLE: usize = 8;
/// Largest `|gpu - cpu|` allowed on a reference log-probability or the mean
/// path score. `tests/crf_gpu_parity.rs` prints the healthy-device maximum
/// this was set against.
pub const SCORE_TOLERANCE: f32 = 0.5;

/// Resolve the period from the environment. `0` means off.
pub fn resolve_every() -> usize {
    escapepod_signal::pod5::env::usize_allow_zero(ENV_EVERY).unwrap_or(DEFAULT_EVERY)
}

/// The CPU side of the comparison. A trait so the guard's logic is testable
/// without a model (and without a device).
pub trait CpuReference: Send + Sync {
    fn score(&self, prepped: &[f32], chains: Option<&RefChains>) -> Result<ScoredDecode, CrfError>;
}

/// Outcome of comparing one sample.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Verdict {
    pub n: usize,
    pub n_seq_mismatch: usize,
    /// Largest score difference over reads whose sequences agree; `INFINITY`
    /// when either side produced a NaN.
    pub max_abs_diff: f32,
}

impl Verdict {
    /// A broken pairing miscalls nearly everything (#416 peaked at 0.91 in
    /// P(charged)); a healthy device flips the odd borderline read through TF32
    /// rounding. So one or two flips in a sample are noise and more than a
    /// quarter of it is not. Score differences have no such allowance.
    pub fn tripped(&self) -> bool {
        self.n_seq_mismatch * 4 > self.n || self.max_abs_diff > SCORE_TOLERANCE
    }
}

/// Compare GPU answers against the CPU's on the same reads. `scored` says
/// whether `ref_logp` / `mean_logpost` are meaningful.
pub fn compare(cpu: &[ScoredDecode], gpu: &[ScoredDecode], scored: bool) -> Verdict {
    let mut v = Verdict {
        n: gpu.len().max(cpu.len()),
        n_seq_mismatch: gpu.len().abs_diff(cpu.len()),
        max_abs_diff: 0.0,
    };
    for (c, g) in cpu.iter().zip(gpu) {
        if c.sequence != g.sequence {
            v.n_seq_mismatch += 1;
            continue;
        }
        if !scored {
            continue;
        }
        let mut d = (c.mean_logpost - g.mean_logpost).abs();
        if c.ref_logp.len() != g.ref_logp.len() {
            d = f32::INFINITY;
        }
        for (a, b) in c.ref_logp.iter().zip(&g.ref_logp) {
            let x = (a - b).abs();
            // NaN is a disagreement, not a pass.
            d = if x.is_nan() { f32::INFINITY } else { d.max(x) };
        }
        v.max_abs_diff = if d.is_nan() {
            f32::INFINITY
        } else {
            v.max_abs_diff.max(d)
        };
    }
    v
}

/// Pick up to [`MAX_SAMPLE`] evenly spaced indices from `n`.
fn sample_indices(n: usize) -> Vec<usize> {
    if n <= MAX_SAMPLE {
        (0..n).collect()
    } else {
        (0..MAX_SAMPLE).map(|i| i * n / MAX_SAMPLE).collect()
    }
}

/// Owns the cadence, the pending CPU checks and the latched failure.
pub struct ParityGuard {
    every: usize,
    reference: Arc<dyn CpuReference>,
    batches: AtomicUsize,
    pending: Arc<AtomicUsize>,
    failure: Arc<Mutex<Option<String>>>,
    strict: AtomicBool,
    label: String,
}

impl ParityGuard {
    /// `None` when `every == 0`.
    pub fn new(
        every: usize,
        reference: Arc<dyn CpuReference>,
        label: impl Into<String>,
    ) -> Option<Self> {
        (every > 0).then(|| Self {
            every,
            reference,
            batches: AtomicUsize::new(0),
            pending: Arc::new(AtomicUsize::new(0)),
            failure: Arc::new(Mutex::new(None)),
            strict: AtomicBool::new(false),
            label: label.into(),
        })
    }

    /// Under strict (`--device gpu`) a disagreement aborts the run.
    pub fn set_strict(&self, strict: bool) {
        self.strict.store(strict, Ordering::Relaxed);
    }

    /// Err if a strict guard has latched a failure.
    pub fn check(&self) -> Result<(), CrfError> {
        if !self.strict.load(Ordering::Relaxed) {
            return Ok(());
        }
        match self
            .failure
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .as_ref()
        {
            Some(msg) => Err(CrfError::Run(msg.clone())),
            None => Ok(()),
        }
    }

    /// Wait for in-flight CPU checks, then [`check`](Self::check). Call once a
    /// run's last batch is in, so a failure on the final check is not lost.
    pub fn finish(&self) -> Result<(), CrfError> {
        self.drain();
        self.check()
    }

    /// Block until no CPU check is in flight.
    pub fn drain(&self) {
        while self.pending.load(Ordering::Acquire) > 0 {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }

    /// Account for one device batch. `rows[i]` produced `gpu[i]`. Every
    /// `every`-th batch (the first included, so a broken install shows at
    /// once) a sample is handed to a rayon task and this returns immediately.
    pub fn observe(&self, rows: &[&[f32]], gpu: &[ScoredDecode], chains: Option<&RefChains>) {
        let batch = self.batches.fetch_add(1, Ordering::Relaxed);
        if !batch.is_multiple_of(self.every) || rows.is_empty() || rows.len() != gpu.len() {
            return;
        }
        let idx = sample_indices(rows.len());
        let rows: Vec<Vec<f32>> = idx.iter().map(|&i| rows[i].to_vec()).collect();
        let gpu: Vec<ScoredDecode> = idx.iter().map(|&i| gpu[i].clone()).collect();
        let chains = chains.cloned();
        let reference = Arc::clone(&self.reference);
        let pending = Arc::clone(&self.pending);
        let failure = Arc::clone(&self.failure);
        let strict = self.strict.load(Ordering::Relaxed);
        let label = self.label.clone();
        pending.fetch_add(1, Ordering::AcqRel);
        rayon::spawn(move || {
            use rayon::prelude::*;
            let cpu: Result<Vec<ScoredDecode>, CrfError> = rows
                .par_iter()
                .map(|r| reference.score(r, chains.as_ref()))
                .collect();
            let msg = match cpu {
                Err(e) => Some(format!(
                    "GPU parity check, batch {batch} ({label}): the CPU reference failed: {e}"
                )),
                Ok(cpu) => {
                    let v = compare(&cpu, &gpu, chains.is_some());
                    v.tripped().then(|| {
                        format!(
                            "GPU parity check, batch {batch} ({label}): {} of {} reads decode to a \
                             different sequence than on the CPU, max abs score diff {:.4} \
                             (tolerance {SCORE_TOLERANCE}). The GPU's barcode calls for this run \
                             cannot be trusted; rerun with `--device cpu`, and see \
                             rnabioco/escapepod-rs#416 for the known cause (a cuBLAS/cuBLASLt \
                             pair from two different CUDA releases)",
                            v.n_seq_mismatch, v.n, v.max_abs_diff
                        )
                    })
                }
            };
            if let Some(msg) = msg {
                tracing::error!("{msg}");
                if strict {
                    failure
                        .lock()
                        .unwrap_or_else(|p| p.into_inner())
                        .get_or_insert(msg);
                }
            }
            pending.fetch_sub(1, Ordering::AcqRel);
        });
    }
}

/// The real CPU reference: a [`CrfEncoder`](super::encoder::CrfEncoder) loaded
/// on first use, so a guard that never fires costs nothing and the load (tract
/// plan, native self-check) is off the device thread.
pub struct LazyCpuEncoder {
    onnx: std::path::PathBuf,
    meta: super::encoder::CrfMetadata,
    enc: std::sync::OnceLock<Result<super::encoder::CrfEncoder, String>>,
}

impl LazyCpuEncoder {
    pub fn new(onnx: impl Into<std::path::PathBuf>, meta: super::encoder::CrfMetadata) -> Self {
        Self {
            onnx: onnx.into(),
            meta,
            enc: std::sync::OnceLock::new(),
        }
    }
}

impl CpuReference for LazyCpuEncoder {
    fn score(&self, prepped: &[f32], chains: Option<&RefChains>) -> Result<ScoredDecode, CrfError> {
        let enc = self
            .enc
            .get_or_init(|| {
                super::encoder::CrfEncoder::load(&self.onnx, self.meta.clone())
                    .map_err(|e| e.to_string())
            })
            .as_ref()
            .map_err(|e| CrfError::Load(e.clone()))?;
        let mut scratch = super::lattice::CrfScratch::new();
        match chains {
            Some(c) => enc.basecall_prepped_with_refs(prepped, &mut scratch, c),
            None => Ok(ScoredDecode {
                sequence: enc.basecall_prepped(prepped, &mut scratch)?,
                ref_logp: Vec::new(),
                mean_logpost: f32::NAN,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sd(seq: &str, lp: &[f32], m: f32) -> ScoredDecode {
        ScoredDecode {
            sequence: seq.into(),
            ref_logp: lp.to_vec(),
            mean_logpost: m,
        }
    }

    /// Stub CPU reference: the "truth" for a read is a function of its first
    /// sample.
    struct Stub;
    impl CpuReference for Stub {
        fn score(&self, p: &[f32], _: Option<&RefChains>) -> Result<ScoredDecode, CrfError> {
            Ok(sd(&format!("ACGT{}", p[0] as usize), &[-1.0, -9.0], -0.1))
        }
    }

    fn rows12() -> Vec<Vec<f32>> {
        (0..12).map(|i| vec![i as f32; 4]).collect()
    }

    fn truth(rows: &[Vec<f32>]) -> Vec<ScoredDecode> {
        rows.iter()
            .map(|r| sd(&format!("ACGT{}", r[0] as usize), &[], f32::NAN))
            .collect()
    }

    fn run(gpu: &[ScoredDecode], strict: bool) -> Result<(), CrfError> {
        let g = ParityGuard::new(1, Arc::new(Stub), "test").unwrap();
        g.set_strict(strict);
        let rows = rows12();
        let refs: Vec<&[f32]> = rows.iter().map(Vec::as_slice).collect();
        g.observe(&refs, gpu, None);
        g.finish()
    }

    #[test]
    fn identical_output_passes() {
        assert!(run(&truth(&rows12()), true).is_ok());
    }

    #[test]
    fn perturbed_gpu_output_trips_a_strict_guard() {
        let mut gpu = truth(&rows12());
        for g in &mut gpu {
            g.sequence.push('A'); // a miscalling device
        }
        let msg = run(&gpu, true)
            .expect_err("a perturbed GPU must trip")
            .to_string();
        assert!(msg.contains("batch 0"), "{msg}");
    }

    #[test]
    fn perturbed_gpu_output_only_warns_when_not_strict() {
        let mut gpu = truth(&rows12());
        for g in &mut gpu {
            g.sequence.push('A');
        }
        assert!(run(&gpu, false).is_ok());
    }

    #[test]
    fn compare_tolerates_a_lone_flip_but_not_a_pattern() {
        let cpu: Vec<_> = (0..8).map(|i| sd(&format!("A{i}"), &[], 0.0)).collect();
        let mut gpu = cpu.clone();
        gpu[3].sequence = "X".into();
        assert!(!compare(&cpu, &gpu, false).tripped());
        gpu[4].sequence = "X".into();
        assert!(!compare(&cpu, &gpu, false).tripped());
        gpu[5].sequence = "X".into();
        assert!(compare(&cpu, &gpu, false).tripped());
    }

    #[test]
    fn compare_catches_score_drift_and_nan() {
        let cpu = vec![sd("A", &[-1.0, -5.0], -0.1)];
        let ok = vec![sd("A", &[-1.01, -5.02], -0.1)];
        assert!(!compare(&cpu, &ok, true).tripped());
        let drift = vec![sd("A", &[-1.0, -7.0], -0.1)];
        assert!(compare(&cpu, &drift, true).tripped());
        let nan = vec![sd("A", &[f32::NAN, -5.0], -0.1)];
        assert!(compare(&cpu, &nan, true).tripped());
        // Scores are ignored when the caller scored no panel.
        assert!(!compare(&cpu, &drift, false).tripped());
    }

    #[test]
    fn cadence_is_every_nth_batch_and_zero_is_off() {
        assert!(ParityGuard::new(0, Arc::new(Stub), "t").is_none());
        let g = ParityGuard::new(3, Arc::new(Stub), "t").unwrap();
        g.set_strict(true);
        let rows = rows12();
        let refs: Vec<&[f32]> = rows.iter().map(Vec::as_slice).collect();
        let good = truth(&rows);
        let mut bad = good.clone();
        for b in &mut bad {
            b.sequence.push('A');
        }
        g.observe(&refs, &good, None); // 0: checked, clean
        g.observe(&refs, &bad, None); // 1: skipped
        g.observe(&refs, &bad, None); // 2: skipped
        assert!(g.finish().is_ok());
        g.observe(&refs, &bad, None); // 3: checked, trips
        assert!(g.finish().is_err());
    }

    #[test]
    fn sample_is_bounded_and_spread() {
        assert_eq!(sample_indices(3), vec![0, 1, 2]);
        assert_eq!(
            sample_indices(512),
            vec![0, 64, 128, 192, 256, 320, 384, 448]
        );
    }
}
