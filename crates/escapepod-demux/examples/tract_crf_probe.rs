//! Probe whether tract can load, optimize, and run the CTC-CRF encoder export.
//!
//! This answers the first blocker on rnabioco/escapepod-models#27 — "verify tract
//! can load and optimize the CRF export" — which decides the feature design: if
//! tract handles it, `crf-decode` can be tract-based like `cnn-detect`; if not,
//! the path is ort-only and cannot live in the default build.
//!
//! Recurrent graphs are where tract historically struggles, and this encoder is
//! 5 stacked LSTMs, so the question is real rather than a formality.
//!
//! Run:
//!   cargo run --release --features cnn-detect \
//!     --example tract_crf_probe -- <crf_encoder.onnx> [out.f32]
//!
//! Writes the raw little-endian f32 output when given a second path, so numeric
//! parity against the torch/onnxruntime reference can be checked outside Rust.

use std::io::Write;
use tract_onnx::prelude::*;

fn main() -> TractResult<()> {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .expect("usage: tract_crf_probe <model.onnx> [out.f32]");
    let dump = args.next();

    const BATCH: usize = 2;
    const CHUNK: usize = 2000;

    let t0 = std::time::Instant::now();
    let raw = tract_onnx::onnx().model_for_path(&path)?;
    println!(
        "parsed        {:?}  ({} nodes)",
        t0.elapsed(),
        raw.nodes().len()
    );

    // The export leaves batch dynamic; tract wants it concrete to optimize.
    let t1 = std::time::Instant::now();
    let typed = raw
        .with_input_fact(0, f32::fact([BATCH, 1, CHUNK]).into())?
        .into_optimized()?;
    println!(
        "optimized     {:?}  ({} nodes)",
        t1.elapsed(),
        typed.nodes().len()
    );

    let plan = typed.into_runnable()?;

    // Deterministic input so the dumped output is reproducible across runs.
    let input: Vec<f32> = (0..BATCH * CHUNK)
        .map(|i| ((i % 251) as f32 - 125.0) / 125.0)
        .collect();
    let tensor = Tensor::from_shape(&[BATCH, 1, CHUNK], &input)?;

    let t2 = std::time::Instant::now();
    let out = plan.run(tvec!(tensor.into()))?;
    println!("ran           {:?}", t2.elapsed());

    let view = out[0].to_plain_array_view::<f32>()?;
    println!("output shape  {:?}", view.shape());

    let expected = [CHUNK / 10, BATCH, 1280];
    if view.shape() == expected {
        println!(
            "CONTRACT OK   [T={}, B={}, n_score=1280] time-major",
            expected[0], BATCH
        );
    } else {
        println!("CONTRACT MISMATCH expected {expected:?}");
    }

    if let Some(dump) = dump {
        let mut f = std::io::BufWriter::new(std::fs::File::create(&dump)?);
        for v in view.iter() {
            f.write_all(&v.to_le_bytes())?;
        }
        f.flush()?;
        println!("wrote         {dump} ({} f32)", view.len());
    }
    Ok(())
}
