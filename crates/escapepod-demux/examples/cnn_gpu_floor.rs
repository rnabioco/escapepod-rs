//! Is adapter detection's device cost real compute, or per-call overhead?
//!
//! Tracing the fused pipeline over 1 M reads put detection at **401 s of device
//! time in a 425 s wall** — 94% of the run, and ~0.40 ms/read. That is hard to
//! believe for a small 1-D CNN over ~800 samples when the LSTM encoder next to
//! it costs 0.105 ms/read, so this prices the same call in isolation across
//! batch sizes.
//!
//! **It answered ~0.009 ms/read at a steady shape — 44× below the pipeline's
//! rate, and flat.** So the 0.40 ms was per-call overhead, not compute, and the
//! cause was the ~680 distinct input shapes prep was emitting; see #187 and
//! `prep_adapter_signal`. Keep the example: it is the measurement that
//! distinguishes "the model is slow" from "we are calling it badly", and that
//! question recurs.
//!
//! Read it as a shape, not a number:
//!
//! * **ms/read falls steeply with batch size, then plateaus** → fixed per-call
//!   cost dominates at small batches (cuDNN plan setup, launch, sync). The fix is
//!   fewer, larger calls, or removing whatever re-runs per call.
//! * **ms/read is flat across batch sizes** → it is genuine compute, and the only
//!   levers are a cheaper model, fp16, or a shorter input window.
//!
//! Note `detect_prepped` splits internally at `ESCAPEPOD_CNN_GPU_BATCH_ELEMS`
//! (default ≈ VRAM/5500 elements, so ~2800 rows at today's 1500-sample prep). Batches
//! above that become several device calls, which is itself part of the answer —
//! set it explicitly to hold the split fixed while sweeping.
//!
//! ```text
//! cargo run --release --features cnn-gpu --example cnn_gpu_floor -- <adapter.onnx> [iters]
//! ```
//!
//! Needs a CUDA-enabled `libonnxruntime` on `ORT_DYLIB_PATH` and a visible GPU.

use std::time::Instant;

use escapepod_demux::{AdapterCnnConfig, AdapterCnnGpu};

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let onnx = args
        .next()
        .ok_or_else(|| anyhow::anyhow!("usage: cnn_gpu_floor <adapter.onnx> [iters]"))?;
    let iters: usize = args.next().map_or(Ok(10), |s| s.parse())?;

    let cfg = AdapterCnnConfig::default();
    let detector =
        AdapterCnnGpu::load(&onnx).map_err(|e| anyhow::anyhow!("loading {onnx}: {e}"))?;

    // Derive the prepped length from the real preprocessing rather than
    // hardcoding it — if the config's window or downscale ever changes, this
    // follows, and the benchmark keeps measuring what the pipeline runs.
    let raw: Vec<f32> = (0..cfg.max_obs_trace)
        .map(|i| 90.0 + ((i as f32) * 0.01).sin() * 8.0)
        .collect();
    let one = cfg
        .prep(&raw)
        .ok_or_else(|| anyhow::anyhow!("prep rejected a {}-sample signal", raw.len()))?;
    let len = one.len();
    println!(
        "model={onnx}\nprepped length={len} samples/read   iters={iters} per batch size\n\
         in-pipeline reference: detection measured ~0.40 ms/read over 1M reads\n"
    );

    println!(
        "{:>7}  {:>10}  {:>12}  {:>11}  {:>9}",
        "batch", "ms/call", "reads/s", "ms/read", "vs 0.40"
    );
    for &batch in &[32usize, 128, 512, 1024, 2048, 4096, 8192] {
        let prepped: Vec<Option<escapepod_demux::PreppedWindow>> = (0..batch)
            .map(|b| {
                // Vary per row so nothing can be cached across the batch axis.
                Some(escapepod_demux::PreppedWindow {
                    data: one.data.iter().map(|v| v + (b as f32) * 1e-4).collect(),
                    valid_len: one.valid_len,
                })
            })
            .collect();

        // Warm up: first call pays session/plan setup for this shape.
        let warm = detector.detect_prepped(&prepped);
        if warm.iter().any(|r| r.is_err()) {
            println!("{batch:>7}  (failed — see error below)");
            if let Some(Err(e)) = warm.into_iter().find(|r| r.is_err()) {
                println!("         {e}");
            }
            continue;
        }

        let t = Instant::now();
        for _ in 0..iters {
            let out = detector.detect_prepped(&prepped);
            std::hint::black_box(&out);
        }
        let secs = t.elapsed().as_secs_f64();
        let per_call = secs / iters as f64;
        let per_read = per_call * 1e3 / batch as f64;
        println!(
            "{batch:>7}  {:>10.2}  {:>12.0}  {:>11.4}  {:>8.2}x",
            per_call * 1e3,
            (batch * iters) as f64 / secs,
            per_read,
            0.40 / per_read
        );
    }

    println!(
        "\nIf ms/read falls with batch and then plateaus, the cost is per-call overhead\n\
         and the pipeline's 0.40 ms/read means its calls are too small. If it is flat,\n\
         the cost is compute and only a cheaper model / fp16 / shorter window will move it."
    );
    Ok(())
}
