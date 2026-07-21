//! GPU/CPU parity for the batched LLR adapter detector.
//!
//! Gated on `--features gpu`; skips transparently without a CUDA device.
//! Compares `GpuDtwContext::detect_adapter_batch` against the CPU demux path
//! (`normalize_signal` → optional `downscale` → `detect_adapter` → scale back).

#![cfg(feature = "gpu")]

use escapepod_signal::dtw::{GpuDtwContext, GpuDtwError};
use escapepod_signal::segmentation::{detect_adapter, downscale, normalize_signal};
use rand::{RngExt, SeedableRng, rngs::StdRng};

const MIN_ADAPTER: usize = 200;
const BORDER_TRIM: usize = 50;

fn try_context() -> Option<GpuDtwContext> {
    match std::panic::catch_unwind(GpuDtwContext::new) {
        Ok(Ok(ctx)) => Some(ctx),
        Ok(Err(GpuDtwError::Driver(e))) => {
            eprintln!("[gpu_llr] skipping — CUDA driver not usable: {e}");
            None
        }
        Ok(Err(e)) => {
            eprintln!("[gpu_llr] skipping — GPU context init failed: {e}");
            None
        }
        Err(_) => {
            eprintln!("[gpu_llr] skipping — GPU context init panicked (no CUDA libs?)");
            None
        }
    }
}

fn cpu_detect(sig: &[i16], ds: usize) -> (usize, usize) {
    let normalized = normalize_signal(sig);
    let (processed, scale) = if ds > 1 {
        let trunc = (normalized.len() / ds) * ds;
        (downscale(&normalized[..trunc], ds), ds)
    } else {
        (normalized, 1)
    };
    let (s, e) = detect_adapter(
        &processed,
        (MIN_ADAPTER / scale).max(1),
        (BORDER_TRIM / scale).max(1),
    );
    (s * scale, e * scale)
}

/// Adapter-like read: open-pore (high) → adapter (low) → RNA (mid), each with
/// noise and randomised lengths/levels, so the LLR detector has real structure
/// to find. Returns a signal long enough to clear the min-adapter constraint.
fn adapter_read(seed: u64) -> Vec<i16> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut sig = Vec::new();
    let pore_len = 300 + (rng.random::<u32>() % 400) as usize;
    let adapter_len = 400 + (rng.random::<u32>() % 600) as usize;
    let rna_len = 800 + (rng.random::<u32>() % 1500) as usize;
    let pore = 400 + (rng.random::<u32>() % 120) as i32;
    let adapter = 80 + (rng.random::<u32>() % 80) as i32;
    let rna = 220 + (rng.random::<u32>() % 120) as i32;
    let mut push = |level: i32, len: usize, rng: &mut StdRng| {
        for _ in 0..len {
            let noise = (rng.random::<i16>() % 24) as i32;
            sig.push((level + noise).clamp(0, i16::MAX as i32) as i16);
        }
    };
    push(pore, pore_len, &mut rng);
    push(adapter, adapter_len, &mut rng);
    push(rna, rna_len, &mut rng);
    sig
}

fn run_parity(ds: usize, label: &str) {
    let Some(ctx) = try_context() else { return };

    let n = 300usize;
    let signals: Vec<Vec<i16>> = (0..n).map(|i| adapter_read(7000 + i as u64)).collect();
    let cpu: Vec<(usize, usize)> = signals.iter().map(|s| cpu_detect(s, ds)).collect();

    let refs: Vec<&[i16]> = signals.iter().map(|s| s.as_slice()).collect();
    let gpu = ctx
        .detect_adapter_batch(&refs, MIN_ADAPTER, BORDER_TRIM, ds)
        .expect("gpu detect");
    let gpu_blk = ctx
        .detect_adapter_batch_block(&refs, MIN_ADAPTER, BORDER_TRIM, ds)
        .expect("gpu block detect");

    // Block-per-read uses z-score (affine) instead of MAD normalize, so it can
    // diverge from CPU only on rare argmax ties; require the same high bar.
    let blk_exact = gpu_blk.iter().zip(&cpu).filter(|(a, b)| a == b).count();
    let blk_rate = blk_exact as f64 / n as f64;
    eprintln!(
        "[gpu_llr {label} block] {blk_exact}/{n} exact ({:.1}%)",
        blk_rate * 100.0
    );
    assert!(
        blk_rate >= 0.95,
        "GPU(block)/CPU LLR detect agreement {:.1}% below 95%",
        blk_rate * 100.0
    );

    assert_eq!(gpu.len(), cpu.len());
    let mut exact = 0usize;
    let mut detected = 0usize;
    let mut diverged: Vec<usize> = Vec::new();
    for (i, (g, c)) in gpu.iter().zip(&cpu).enumerate() {
        if *c != (0, 0) {
            detected += 1;
        }
        if g == c {
            exact += 1;
        } else {
            diverged.push(i);
        }
    }
    let rate = exact as f64 / n as f64;
    eprintln!(
        "[gpu_llr {label}] {exact}/{n} exact ({:.1}%); {detected} CPU-detected adapters; {} diverged",
        rate * 100.0,
        diverged.len()
    );
    for &i in diverged.iter().take(8) {
        eprintln!("  read {i}: gpu={:?} cpu={:?}", gpu[i], cpu[i]);
    }

    assert!(
        detected as f64 / n as f64 > 0.5,
        "test signals should mostly detect an adapter"
    );
    assert!(
        rate >= 0.95,
        "GPU/CPU LLR detect agreement {:.1}% below 95% threshold",
        rate * 100.0
    );
}

#[test]
fn gpu_llr_detect_matches_cpu_no_downscale() {
    run_parity(1, "ds=1");
}

#[test]
fn gpu_llr_detect_matches_cpu_downscale10() {
    run_parity(10, "ds=10");
}
