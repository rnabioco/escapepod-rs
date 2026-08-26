//! What is the *floor* for the GPU CRF head, with no pipeline around it?
//!
//! The fused pipeline reaches ~2400 reads/s with the device ~25% busy, and six
//! attempts to explain the idle time as a scheduling problem all came back
//! negative. This isolates the two device stages and times them on synthetic
//! windows, so the pipeline's number can be compared against what the graph and
//! kernels can actually do back to back:
//!
//! * **encode** — one `Session::run` per batch, nothing else.
//! * **encode + decode** — the same, plus the batched lattice decode, which is
//!   what `basecall_batch` does and what the pipeline's worker calls.
//!
//! If encode alone is already close to the pipeline's per-read cost, there is no
//! starvation to find: the encoder graph simply does not saturate the device,
//! and only a faster graph or a bigger batch would help.
//!
//! ```text
//! cargo run --release --features gpu --example crf_gpu_floor -- <bundle> [batch] [iters]
//! ```
//!
//! Needs a CUDA-enabled `libonnxruntime` on `ORT_DYLIB_PATH` and a visible GPU.

use std::time::Instant;

use escapepod_demux::crf::CrfEncoderGpu;

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let bundle = args
        .next()
        .ok_or_else(|| anyhow::anyhow!("usage: crf_gpu_floor <bundle-dir> [batch] [iters]"))?;
    let batch: usize = args.next().map_or(Ok(512), |s| s.parse())?;
    let iters: usize = args.next().map_or(Ok(20), |s| s.parse())?;

    let encoder = CrfEncoderGpu::load_bundle(&bundle, Some(16))?;
    let meta = encoder.metadata();
    let chunk = meta.signal.chunk;
    let t_len = meta.t_len();
    println!(
        "bundle={bundle}  chunk={chunk}  t_len={t_len}  n_score={}  batch={batch}  iters={iters}",
        encoder.layout().n_score
    );
    println!(
        "decode on GPU: {}   scores in place: {}",
        encoder.gpu_decode_active(),
        encoder.zero_copy_active()
    );

    // Deterministic, standardised-looking input. Content does not affect timing
    // — the graph is dense — but a fixed formula keeps runs comparable.
    let windows: Vec<Option<Vec<f32>>> = (0..batch)
        .map(|b| {
            Some(
                (0..chunk)
                    .map(|i| (((i + b * 7) as f32) * 0.01).sin())
                    .collect(),
            )
        })
        .collect();
    let raw: Vec<Vec<f32>> = windows.iter().map(|w| w.clone().unwrap()).collect();

    // Warm up: first call pays graph init, cuDNN plan selection and NVRTC.
    let _ = encoder.encode_batch(&raw[..1.min(batch)])?;
    let _ = encoder.basecall_batch(&windows[..1.min(batch)])?;

    let report = |label: &str, secs: f64| {
        let per_call = secs / iters as f64;
        let reads = (batch * iters) as f64;
        println!(
            "{label:<22} {:>7.1} ms/call   {:>8.0} reads/s   {:>6.3} ms/read",
            per_call * 1e3,
            reads / secs,
            per_call * 1e3 / batch as f64
        );
    };

    // NOT a clean encode measurement, and reported as what it is: `encode_batch`
    // is the legacy accessor, which still de-interleaves on the host into one
    // 1.5 MB allocation per read. It comes out *slower* than encode+decode
    // below, which is the whole reason the fused path stopped using it — keep it
    // here as the standing evidence for that.
    let t = Instant::now();
    for _ in 0..iters {
        let _ = encoder.encode_batch(&raw)?;
    }
    report("encode + host transpose", t.elapsed().as_secs_f64());

    // encode + lattice decode: what the pipeline's worker actually calls.
    let t = Instant::now();
    for _ in 0..iters {
        let out = encoder.basecall_batch(&windows)?;
        std::hint::black_box(&out);
    }
    report("encode + decode", t.elapsed().as_secs_f64());

    println!(
        "\nfused pipeline for comparison: ~2400 reads/s end to end (40k reads, 16 cores, one A30),\n\
         which includes POD5 read + VBZ decode, adapter-CNN detect, prep, barcode match and 17 writers."
    );
    Ok(())
}
