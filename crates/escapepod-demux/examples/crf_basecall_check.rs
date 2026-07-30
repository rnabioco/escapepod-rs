//! End-to-end check of the CRF basecaller against the torch/bonito reference.
//!
//! Runs the exported encoder through tract and the lattice decode in Rust over
//! the *same* input the export's `reference_io.npz` recorded, so the printed
//! sequences can be compared against what bonito's `decode_batch` produced for
//! those scores. Together with `tests/crf_golden.rs` — which pins the decode
//! alone — this closes the loop from ONNX file to sequence.
//!
//! Kept as an example rather than a test because it needs a model file that is
//! not committed (weights ship via Releases, not git).
//!
//! ```text
//! # dump the reference input once:
//! #   python -c "import numpy as np; \
//! #     np.load('reference_io.npz')['signal'].astype('float32').tofile('reference_signal.f32')"
//! cargo run --release --features crf-decode --example crf_basecall_check -- \
//!     <bundle_dir> <reference_signal.f32>
//! ```
//!
//! Expected for `st_model_clean.pt`'s export:
//! ```text
//! read 0  AAATGATGATAGCCGAAGGTAGTAGGTTC
//! read 1  ATAGGCGAATAGTCCCGGTAAGGTAGTAGGTTC
//! ```

use std::time::Instant;

use escapepod_demux::crf::{Backend, CrfEncoder, CrfScratch};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let bundle = args
        .next()
        .expect("usage: crf_basecall_check <bundle_dir> <signal.f32>");
    let signal_path = args
        .next()
        .expect("usage: crf_basecall_check <bundle_dir> <signal.f32>");

    let t0 = Instant::now();
    let encoder = CrfEncoder::load_bundle(&bundle)?;
    println!("loaded        {:?}", t0.elapsed());

    let meta = encoder.metadata();
    let chunk = meta.signal.chunk;
    println!(
        "contract      chunk={chunk} stride={} t_len={} n_score={} backend={:?}",
        meta.signal.stride,
        meta.t_len(),
        encoder.layout().n_score,
        Backend::best_for(encoder.layout()),
    );

    // The reference input is already the model's input domain (standardised),
    // so it bypasses `prep` and goes straight to the encoder.
    let raw = std::fs::read(&signal_path)?;
    let signal: Vec<f32> = raw
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    assert!(
        signal.len().is_multiple_of(chunk),
        "{} floats is not a whole number of {chunk}-sample chunks",
        signal.len()
    );

    let mut scratch = CrfScratch::new();
    for (i, window) in signal.chunks_exact(chunk).enumerate() {
        let t = Instant::now();
        let seq = encoder.basecall_prepped(window, &mut scratch)?;
        println!("read {i}  {seq}");
        println!("         {:?}  ({} bases)", t.elapsed(), seq.len());
    }

    // Split encoder from decode. Which dominates decides whether moving the
    // encoder to the GPU is worth anything: if the decode is the larger half,
    // a GPU encoder just relocates the bottleneck.
    const REPS: usize = 20;
    let window = &signal[..chunk];
    let scores = encoder.encode(window)?;

    let t = Instant::now();
    for _ in 0..REPS {
        std::hint::black_box(encoder.encode(window)?);
    }
    let enc = t.elapsed() / REPS as u32;

    for backend in [Backend::Scalar, Backend::best_for(encoder.layout())] {
        let t = Instant::now();
        for _ in 0..REPS {
            std::hint::black_box(encoder.decode_scores(&scores, &mut scratch, backend)?);
        }
        let dec = t.elapsed() / REPS as u32;
        println!(
            "split         encoder {enc:?}  decode[{backend:?}] {dec:?}  \
             ({:.0}% of total is decode)",
            100.0 * dec.as_secs_f64() / (enc + dec).as_secs_f64()
        );
    }
    Ok(())
}
