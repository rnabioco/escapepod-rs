//! GPU/CPU parity tests for the banded DTW distance matrix.
//!
//! Compiled only with `--features gpu`. At runtime the test transparently
//! skips (printing a message) if no CUDA device is available, so a GPU-less
//! host can still run `cargo test --features gpu` without a hard failure.

#![cfg(feature = "gpu")]

use escapepod_signal::dtw::{GpuDtwContext, GpuDtwError, dtw_distance_matrix};
use rand::{RngExt, SeedableRng, rngs::StdRng};

const TOL_FACTOR: f32 = 1e-4;

fn random_matrix(seed: u64, n: usize, len: usize) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| (0..len).map(|_| rng.random::<f32>()).collect())
        .collect()
}

/// Try to build a GPU context; if no device or NVRTC lib is present, print a
/// diagnostic and return `None` so the caller can skip gracefully.
///
/// cudarc dynamically loads `libnvrtc.so` and `libcuda.so` and *panics* on
/// the first call if either is missing, so a plain `Result` match isn't
/// enough — we catch unwinds here too.
fn try_context() -> Option<GpuDtwContext> {
    let result = std::panic::catch_unwind(GpuDtwContext::new);
    match result {
        Ok(Ok(ctx)) => Some(ctx),
        Ok(Err(GpuDtwError::Driver(e))) => {
            eprintln!("[gpu_dtw] skipping — CUDA driver not usable: {e}");
            None
        }
        Ok(Err(e)) => {
            eprintln!("[gpu_dtw] skipping — GPU context init failed: {e}");
            None
        }
        Err(panic) => {
            let msg = panic
                .downcast_ref::<&'static str>()
                .map(|s| s.to_string())
                .or_else(|| panic.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "<non-string panic payload>".into());
            eprintln!(
                "[gpu_dtw] skipping — GPU init panicked (likely missing libnvrtc / libcuda): {msg}"
            );
            None
        }
    }
}

fn assert_close(cpu: &ndarray::Array2<f32>, gpu: &ndarray::Array2<f32>) {
    assert_eq!(cpu.shape(), gpu.shape(), "shape mismatch");
    for ((i, j), &c) in cpu.indexed_iter() {
        let g = gpu[[i, j]];
        // Both sides return +inf for empty/impossible pairs; (inf - inf) is
        // NaN, so compare those explicitly before the absolute-diff path.
        if c.is_infinite() && g.is_infinite() && c.is_sign_positive() == g.is_sign_positive() {
            continue;
        }
        let tol = TOL_FACTOR * c.abs().max(1.0);
        let diff = (c - g).abs();
        assert!(
            diff <= tol,
            "mismatch at ({i},{j}): cpu={c} gpu={g} diff={diff} tol={tol}"
        );
    }
}

#[test]
fn parity_unwindowed() {
    let Some(ctx) = try_context() else { return };
    let queries = random_matrix(1, 32, 128);
    let refs = random_matrix(2, 16, 128);
    let cpu = dtw_distance_matrix(&queries, &refs, None);
    let gpu = ctx
        .distance_matrix(&queries, &refs, None)
        .expect("gpu matrix");
    assert_close(&cpu, &gpu);
}

#[test]
fn parity_banded() {
    let Some(ctx) = try_context() else { return };
    let queries = random_matrix(3, 24, 110);
    let refs = random_matrix(4, 24, 110);
    let cpu = dtw_distance_matrix(&queries, &refs, Some(12));
    let gpu = ctx
        .distance_matrix(&queries, &refs, Some(12))
        .expect("gpu matrix");
    assert_close(&cpu, &gpu);
}

#[test]
fn parity_uneven_lengths() {
    let Some(ctx) = try_context() else { return };
    let mut queries = random_matrix(5, 8, 100);
    queries.push(vec![0.0; 50]);
    queries.push(vec![1.0; 200]);
    let mut refs = random_matrix(6, 8, 110);
    refs.push(vec![0.5; 80]);

    let cpu = dtw_distance_matrix(&queries, &refs, Some(20));
    let gpu = ctx
        .distance_matrix(&queries, &refs, Some(20))
        .expect("gpu matrix");
    assert_close(&cpu, &gpu);
}

#[test]
fn empty_inputs() {
    let Some(ctx) = try_context() else { return };
    let empty: Vec<Vec<f32>> = Vec::new();
    let refs = random_matrix(7, 4, 50);
    let gpu = ctx.distance_matrix(&empty, &refs, None).expect("empty ok");
    assert_eq!(gpu.shape(), &[0, 4]);
    let gpu2 = ctx.distance_matrix(&refs, &empty, None).expect("empty ok");
    assert_eq!(gpu2.shape(), &[4, 0]);
}
