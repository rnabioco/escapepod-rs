//! The CRF parity guard (#448) must not fire on a healthy device.
//!
//! Scores the same windows on the GPU encoder and on the CPU encoder and runs
//! the guard's own comparison over them, printing the worst score difference
//! so `parity::SCORE_TOLERANCE` can be read against a measurement. Also runs
//! the guard live at period 1 (every batch checked) through `basecall_batch*`
//! and asserts no failure latches.
//!
//! Needs `--features gpu`, a visible CUDA device, and `ESCAPEPOD_CRF_BUNDLE`;
//! skips with a message otherwise. The windows are synthetic (a fixed formula);
//! the dual-axis 20k set is exercised end to end by
//! `benchmarks/benchmark_demux_crf.sh` with `ESCAPEPOD_CRF_GPU_PARITY_EVERY=1`.

#![cfg(feature = "gpu")]

use escapepod_demux::crf::parity::{MAX_SAMPLE, compare};
use escapepod_demux::crf::{CrfEncoder, CrfEncoderGpu, CrfScratch};

#[test]
#[allow(clippy::disallowed_methods)] // bundle path + the knob under test, as in crf_encoder_native_parity
fn healthy_gpu_does_not_trip_the_guard() {
    let Ok(bundle) = std::env::var("ESCAPEPOD_CRF_BUNDLE") else {
        eprintln!("skipping crf_gpu_parity: set ESCAPEPOD_CRF_BUNDLE to a CRF bundle directory");
        return;
    };
    // The guard under test, at the tightest cadence. Set before load: the
    // guard reads it then.
    // SAFETY: this test binary has one test and no other thread reads the env.
    unsafe { std::env::set_var("ESCAPEPOD_CRF_GPU_PARITY_EVERY", "1") };
    let gpu = match std::panic::catch_unwind(|| CrfEncoderGpu::load_bundle(&bundle)) {
        Ok(Ok(e)) => e,
        Ok(Err(e)) => {
            eprintln!("skipping crf_gpu_parity: no usable GPU encoder: {e}");
            return;
        }
        Err(_) => {
            eprintln!("skipping crf_gpu_parity: GPU init panicked (no device / libraries)");
            return;
        }
    };
    gpu.set_parity_strict(true);
    let cpu = CrfEncoder::load_bundle(&bundle).expect("CPU encoder");

    let chunk = gpu.metadata().signal.chunk;
    let windows: Vec<Option<Vec<f32>>> = (0..64)
        .map(|b| {
            Some(
                (0..chunk)
                    .map(|i| (((i + b * 7) as f32) * 0.01).sin())
                    .collect(),
            )
        })
        .collect();

    // Sequences only: the guard's live path.
    let seqs = gpu.basecall_batch(&windows).expect("gpu basecall");
    assert_eq!(seqs.len(), windows.len());

    // With a reference panel, if the bundle ships one: compare directly so the
    // measured score drift is visible.
    if let Some(entries) = gpu.metadata().barcodes.clone() {
        let refs: Vec<&[u8]> = entries.iter().map(|b| b.sequence.as_bytes()).collect();
        let chains = gpu.ref_chains(&refs).expect("chains");
        let got = gpu
            .basecall_batch_with_refs(&windows, &chains)
            .expect("gpu scored");
        let gpu_sd: Vec<_> = got
            .into_iter()
            .take(MAX_SAMPLE)
            .map(Option::unwrap)
            .collect();
        let mut scratch = CrfScratch::new();
        let cpu_sd: Vec<_> = windows
            .iter()
            .take(MAX_SAMPLE)
            .map(|w| {
                cpu.basecall_prepped_with_refs(w.as_ref().unwrap(), &mut scratch, &chains)
                    .expect("cpu scored")
            })
            .collect();
        let v = compare(&cpu_sd, &gpu_sd, true);
        eprintln!("crf_gpu_parity: {v:?}");
        assert!(!v.tripped(), "healthy GPU tripped the guard: {v:?}");
    }

    gpu.finish_parity().expect("a healthy GPU must not trip");
}
