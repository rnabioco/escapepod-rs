//! Throughput comparison for BoundariesCNN adapter detection:
//!   tract-onnx (per-read CPU)  vs  CnnCompute (per-read CPU)  vs  GpuCnn (batched).
//!
//! Run on a GPU node:
//!   pixi run -e gpu cargo run --release --example cnn_gpu_bench \
//!       -p escapepod-demux --features "gpu cnn-detect" -- [N_READS]
//!
//! Needs benchmarks/adapter_cnn_rna004.{onnx,weights} (gitignored; generated
//! by scripts/export_adapter_cnn_to_onnx.py + dump_adapter_cnn_weights.py).
#![cfg(all(feature = "gpu", feature = "cnn-detect"))]

use std::path::PathBuf;
use std::time::Instant;

use escapepod_demux::{AdapterCnn, CnnCompute, GpuCnn};

fn artifact(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../benchmarks")
        .join(name)
}

// Deterministic LCG — avoids a dev-dep on rand for an example.
fn synth(n: usize) -> Vec<Vec<f32>> {
    let mut s: u64 = 0x1234_5678;
    let mut next = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 33) as u32) as f32 / u32::MAX as f32
    };
    (0..n)
        .map(|_| {
            let len = 3000 + (next() * 4000.0) as usize;
            let step = 1200 + (next() * 3000.0) as usize;
            (0..len)
                .map(|i| (if i < step { 95.0 } else { 75.0 }) + 8.0 * (next() - 0.5))
                .collect()
        })
        .collect()
}

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(2000);
    let onnx = artifact("adapter_cnn_rna004.onnx");
    let weights = artifact("adapter_cnn_rna004.weights");
    if !onnx.exists() || !weights.exists() {
        eprintln!("missing {onnx:?} / {weights:?}");
        return;
    }

    let reads = synth(n);
    let refs: Vec<&[f32]> = reads.iter().map(|v| v.as_slice()).collect();
    println!("BoundariesCNN adapter detection — {n} reads\n");

    // tract-onnx, per read.
    let tract = AdapterCnn::load(&onnx).expect("onnx");
    let t = Instant::now();
    let mut acc = 0usize;
    for s in &reads {
        acc ^= tract.detect_adapter_end(s).unwrap();
    }
    let tract_s = t.elapsed().as_secs_f64();
    println!(
        "  tract  (CPU, per-read):  {tract_s:7.3} s   {:8.0} reads/s",
        n as f64 / tract_s
    );

    // CnnCompute, per read.
    let cpu = CnnCompute::load(&weights).expect("weights");
    let t = Instant::now();
    for s in &reads {
        acc ^= cpu.detect_adapter_end(s).unwrap();
    }
    let cpu_s = t.elapsed().as_secs_f64();
    println!(
        "  compute(CPU, per-read):  {cpu_s:7.3} s   {:8.0} reads/s",
        n as f64 / cpu_s
    );

    // GpuCnn, batched (one warm-up to amortize NVRTC compile, then timed).
    let gpu = GpuCnn::load(&weights).expect("gpu");
    let _ = gpu
        .detect_adapter_end_batch(&refs[..refs.len().min(8)])
        .expect("warmup");
    let t = Instant::now();
    let out = gpu.detect_adapter_end_batch(&refs).expect("gpu batch");
    let gpu_s = t.elapsed().as_secs_f64();
    acc ^= out.iter().copied().fold(0, |a, b| a ^ b);
    println!(
        "  gpu    (batched):        {gpu_s:7.3} s   {:8.0} reads/s",
        n as f64 / gpu_s
    );

    println!("\n  speedup gpu vs tract:  {:5.1}x", tract_s / gpu_s);
    println!("  speedup gpu vs compute:{:5.1}x", cpu_s / gpu_s);
    std::hint::black_box(acc); // keep the accumulator (and thus the work) live
}
