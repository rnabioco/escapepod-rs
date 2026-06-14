//! Parity of the self-contained `CnnCompute` (our conv stack) against the
//! `tract-onnx` reference `AdapterCnn`, on the same weights. Skips when the
//! ONNX / weights blobs aren't present (they are gitignored — CC BY-NC).
#![cfg(feature = "cnn-detect")]

use std::path::PathBuf;

use escapepod_demux::{AdapterCnn, CnnCompute};
use rand::{RngExt, SeedableRng, rngs::StdRng};

fn artifact(name: &str) -> PathBuf {
    // CARGO_MANIFEST_DIR = crates/escapepod-demux; the blobs live at the
    // workspace-root benchmarks/ dir.
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../benchmarks")
        .join(name)
}

/// A handful of synthetic calibrated-pA reads with an adapter-like step.
fn synth_reads(n: usize) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(2024);
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

#[test]
fn cnn_compute_matches_tract() {
    let onnx = artifact("adapter_cnn_rna004.onnx");
    let weights = artifact("adapter_cnn_rna004.weights");
    if !onnx.exists() || !weights.exists() {
        eprintln!("[cnn_compute] skipping — missing {onnx:?} / {weights:?}");
        return;
    }

    let tract = AdapterCnn::load(&onnx).expect("load onnx");
    let mine = CnnCompute::load(&weights).expect("load weights");

    let reads = synth_reads(64);
    let mut mismatches = 0;
    for (i, sig) in reads.iter().enumerate() {
        let a = tract.detect_adapter_end(sig).expect("tract");
        let b = mine.detect_adapter_end(sig).expect("compute");
        if a != b {
            mismatches += 1;
            eprintln!("read {i}: tract={a} compute={b}");
        }
    }
    assert_eq!(
        mismatches,
        0,
        "{mismatches}/{} reads disagreed between tract and CnnCompute",
        reads.len()
    );
}
