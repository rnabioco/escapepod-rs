//! GPU/CPU parity for the batched BoundariesCNN. Gated on `gpu`+`cnn-detect`;
//! skips when no CUDA device or the gitignored weights blob is absent.
#![cfg(all(feature = "gpu", feature = "cnn-detect"))]

use std::path::PathBuf;

use escapepod_demux::{CnnCompute, GpuCnn};
use rand::{RngExt, SeedableRng, rngs::StdRng};

fn weights_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../benchmarks")
        .join("adapter_cnn_rna004.weights")
}

fn synth_reads(n: usize) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(7);
    (0..n)
        .map(|_| {
            let len = 3000 + (rng.random::<u32>() % 4000) as usize;
            let step = 1200 + (rng.random::<u32>() % 3000) as usize;
            (0..len)
                .map(|i| {
                    let base = if i < step { 95.0 } else { 75.0 };
                    base + 8.0 * (rng.random::<f32>() - 0.5)
                })
                .collect()
        })
        .collect()
}

fn try_gpu(path: &PathBuf) -> Option<GpuCnn> {
    match std::panic::catch_unwind(|| GpuCnn::load(path)) {
        Ok(Ok(g)) => Some(g),
        Ok(Err(e)) => {
            eprintln!("[gpu_cnn] skipping — GPU init failed: {e}");
            None
        }
        Err(_) => {
            eprintln!("[gpu_cnn] skipping — GPU init panicked (no libnvrtc/libcuda?)");
            None
        }
    }
}

#[test]
fn gpu_cnn_matches_cpu() {
    let weights = weights_path();
    if !weights.exists() {
        eprintln!("[gpu_cnn] skipping — missing {weights:?}");
        return;
    }
    let Some(gpu) = try_gpu(&weights) else { return };
    let cpu = CnnCompute::load(&weights).expect("cpu load");

    let reads = synth_reads(96);
    let refs: Vec<&[f32]> = reads.iter().map(|v| v.as_slice()).collect();

    let gpu_out = gpu.detect_adapter_end_batch(&refs).expect("gpu batch");
    let cpu_out: Vec<usize> = reads
        .iter()
        .map(|s| cpu.detect_adapter_end(s).expect("cpu"))
        .collect();

    assert_eq!(gpu_out.len(), cpu_out.len());
    let mut exact = 0usize;
    let mut near = 0usize;
    let ds = 10usize; // downscale_factor: one argmax slot == ds samples
    for (i, (&g, &c)) in gpu_out.iter().zip(cpu_out.iter()).enumerate() {
        if g == c {
            exact += 1;
        } else if g.abs_diff(c) <= ds {
            near += 1; // f32 summation-order near-tie: adjacent argmax slot
            eprintln!("read {i}: gpu={g} cpu={c} (adjacent slot)");
        } else {
            panic!("read {i}: gpu={g} cpu={c} differ by > {ds} samples");
        }
    }
    eprintln!(
        "gpu/cpu CNN: {exact} exact, {near} adjacent-slot of {}",
        reads.len()
    );
    // The vast majority must match exactly; only rare f32 near-ties may slip a slot.
    assert!(
        exact >= reads.len() - 3,
        "too many non-exact matches: {exact}/{}",
        reads.len()
    );
}
