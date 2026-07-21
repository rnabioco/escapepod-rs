//! GPU/CPU parity for the batched t-test fingerprint extractor.
//!
//! Gated on `--features gpu`; skips transparently without a CUDA device.
//! Compares `GpuDtwContext::fingerprint_batch` against the CPU reference
//! `extract_fingerprint_from_signal` on the demux path
//! (`keep_last = Some(25)`, z-score, no dwell).

#![cfg(feature = "gpu")]

use escapepod_demux::extract_fingerprint_from_signal;
use escapepod_signal::dtw::{GpuDtwContext, GpuDtwError, NormMethod};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use uuid::Uuid;

// Demux pipeline parameters (see FpParams in the CLI run pipeline).
const WINDOW_WIDTH: usize = 12;
const NUM_SEGMENTS: usize = 111;
const MIN_SEP: usize = 6;
const KEEP_LAST: usize = 25;

fn try_context() -> Option<GpuDtwContext> {
    match std::panic::catch_unwind(GpuDtwContext::new) {
        Ok(Ok(ctx)) => Some(ctx),
        Ok(Err(GpuDtwError::Driver(e))) => {
            eprintln!("[gpu_fingerprint] skipping — CUDA driver not usable: {e}");
            None
        }
        Ok(Err(e)) => {
            eprintln!("[gpu_fingerprint] skipping — GPU context init failed: {e}");
            None
        }
        Err(_) => {
            eprintln!("[gpu_fingerprint] skipping — GPU context init panicked (no CUDA libs?)");
            None
        }
    }
}

/// Piecewise-constant signal: distinct, well-separated random levels with mild
/// noise. Clean changepoints minimise exact-tie ambiguity, so this isolates the
/// core segmentation/normalisation logic.
fn stepped_signal(seed: u64, n_segments: usize) -> Vec<i16> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut sig = Vec::new();
    for _ in 0..n_segments {
        let len = 50 + (rng.random::<u32>() % 150) as usize;
        // Distinct, spread-out levels so adjacent segments differ clearly.
        let level = (rng.random::<i16>() / 4) as i32;
        for _ in 0..len {
            let noise = (rng.random::<i16>() % 8) as i32; // small jitter
            sig.push((level + noise).clamp(i16::MIN as i32, i16::MAX as i32) as i16);
        }
    }
    sig
}

fn cpu_fp(sig: &[i16]) -> Option<Vec<f64>> {
    extract_fingerprint_from_signal(
        sig,
        0,
        sig.len(),
        NUM_SEGMENTS,
        WINDOW_WIDTH,
        NormMethod::ZScore,
        Uuid::nil(),
        Some(MIN_SEP),
        Some(KEEP_LAST),
        false,
    )
    .map(|f| f.values)
}

#[test]
fn gpu_fingerprint_matches_cpu() {
    let Some(ctx) = try_context() else { return };

    let n = 256usize;
    let signals: Vec<Vec<i16>> = (0..n)
        .map(|i| stepped_signal(1000 + i as u64, 40 + (i % 20)))
        .collect();

    let cpu: Vec<Option<Vec<f64>>> = signals.iter().map(|s| cpu_fp(s)).collect();

    let reads: Vec<(&[i16], usize, usize)> = signals
        .iter()
        .map(|s| (s.as_slice(), 0usize, s.len()))
        .collect();
    let gpu = ctx
        .fingerprint_batch(&reads, WINDOW_WIDTH, NUM_SEGMENTS, MIN_SEP, KEEP_LAST)
        .expect("gpu fingerprint");

    assert_eq!(gpu.len(), cpu.len());

    const TOL: f64 = 2e-3;
    let mut close = 0usize;
    let mut none_agree = 0usize;
    let mut diverged: Vec<usize> = Vec::new();
    for (i, (g, c)) in gpu.iter().zip(&cpu).enumerate() {
        match (g, c) {
            (None, None) => {
                none_agree += 1;
                close += 1;
            }
            (Some(gv), Some(cv)) if gv.len() == cv.len() => {
                let maxd = gv
                    .iter()
                    .zip(cv)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0_f64, f64::max);
                if maxd < TOL {
                    close += 1;
                } else {
                    diverged.push(i);
                }
            }
            _ => diverged.push(i),
        }
    }

    let rate = close as f64 / n as f64;
    eprintln!(
        "[gpu_fingerprint] {close}/{n} within tol ({:.1}%); {none_agree} both-None; {} diverged",
        rate * 100.0,
        diverged.len()
    );
    if !diverged.is_empty() {
        let show = &diverged[..diverged.len().min(8)];
        for &i in show {
            eprintln!(
                "  read {i}: gpu_len={:?} cpu_len={:?}",
                gpu[i].as_ref().map(|v| v.len()),
                cpu[i].as_ref().map(|v| v.len()),
            );
        }
    }

    // Clean stepped signal should agree almost everywhere; allow a tiny tail
    // for the documented exact-tie / pdqsort tie-break divergence.
    assert!(
        rate >= 0.97,
        "GPU/CPU fingerprint agreement {:.1}% below 97% threshold",
        rate * 100.0
    );
}
