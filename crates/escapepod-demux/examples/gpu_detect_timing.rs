//! CPU vs GPU timing for LLR adapter detect — the dominant prep stage (~85%).
//!
//! Decodes all reads from a POD5, then times single-thread CPU detect against
//! the batched GPU detector on identical inputs. Reports throughput + speedup.
//!
//! Run: `cargo run --release --features gpu --example gpu_detect_timing -- <file.pod5> [max_reads] [downscale]`

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("build with --features gpu");
}

#[cfg(feature = "gpu")]
fn main() {
    use std::time::Instant;

    use escapepod_signal::Reader;
    use escapepod_signal::dtw::GpuDtwContext;
    use escapepod_signal::segmentation::{detect_adapter, downscale, normalize_signal};

    const MIN_ADAPTER: usize = 200;
    const BORDER_TRIM: usize = 50;

    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .expect("usage: gpu_detect_timing <file.pod5> [max_reads] [downscale]");
    let max_reads: usize = args
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or(usize::MAX);
    let ds: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(1);

    let cpu_detect = |sig: &[i16]| -> (usize, usize) {
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
    };

    let reader = Reader::open(&path).expect("open");
    let extractor = reader.signal_extractor().expect("extractor");
    let mut signals: Vec<Vec<i16>> = Vec::new();
    for read in reader.reads().expect("reads") {
        let read = read.expect("read");
        if read.signal_rows.is_empty() {
            continue;
        }
        if let Ok(s) = extractor.get_signal(&read.signal_rows) {
            signals.push(s);
        }
        if signals.len() >= max_reads {
            break;
        }
    }
    let total_samples: usize = signals.iter().map(|s| s.len()).sum();
    println!(
        "reads: {}  total samples: {}  ({:.0} avg)  downscale: {ds}",
        signals.len(),
        total_samples,
        total_samples as f64 / signals.len().max(1) as f64
    );

    // CPU single-thread.
    let t0 = Instant::now();
    let cpu: Vec<(usize, usize)> = signals.iter().map(|s| cpu_detect(s)).collect();
    let cpu_dt = t0.elapsed();

    // GPU batched (build context once; time the classify call only).
    let ctx = GpuDtwContext::new().expect("gpu ctx");
    let refs: Vec<&[i16]> = signals.iter().map(|s| s.as_slice()).collect();
    // warm-up (NVRTC already compiled in new(); this primes allocations).
    let _ = ctx.detect_adapter_batch(&refs[..refs.len().min(64)], MIN_ADAPTER, BORDER_TRIM, ds);
    let t1 = Instant::now();
    let gpu = ctx
        .detect_adapter_batch(&refs, MIN_ADAPTER, BORDER_TRIM, ds)
        .expect("gpu detect");
    let gpu_dt = t1.elapsed();

    // GPU block-per-read.
    let _ =
        ctx.detect_adapter_batch_block(&refs[..refs.len().min(64)], MIN_ADAPTER, BORDER_TRIM, ds);
    let t2 = Instant::now();
    let gpu_blk = ctx
        .detect_adapter_batch_block(&refs, MIN_ADAPTER, BORDER_TRIM, ds)
        .expect("gpu block detect");
    let gpu_blk_dt = t2.elapsed();

    let exact = cpu.iter().zip(&gpu).filter(|(a, b)| a == b).count();
    let exact_blk = cpu.iter().zip(&gpu_blk).filter(|(a, b)| a == b).count();
    let nr = signals.len() as f64;
    println!("--- LLR detect timing ---");
    println!(
        "CPU (1 thread):     {:>8.1} ms   {:>7.1} µs/read",
        cpu_dt.as_secs_f64() * 1e3,
        cpu_dt.as_secs_f64() * 1e6 / nr
    );
    println!(
        "GPU (1 thr/read):   {:>8.1} ms   {:>7.1} µs/read   {:.2}x vs 1 CPU   parity {exact}/{}",
        gpu_dt.as_secs_f64() * 1e3,
        gpu_dt.as_secs_f64() * 1e6 / nr,
        cpu_dt.as_secs_f64() / gpu_dt.as_secs_f64(),
        signals.len(),
    );
    println!(
        "GPU (blk/read):     {:>8.1} ms   {:>7.1} µs/read   {:.2}x vs 1 CPU   parity {exact_blk}/{}",
        gpu_blk_dt.as_secs_f64() * 1e3,
        gpu_blk_dt.as_secs_f64() * 1e6 / nr,
        cpu_dt.as_secs_f64() / gpu_blk_dt.as_secs_f64(),
        signals.len(),
    );

    // ---- fingerprint stage (use CPU detect boundaries for both) ----
    use escapepod_demux::extract_fingerprint_from_signal;
    use escapepod_signal::dtw::NormMethod;
    use uuid::Uuid;
    let tf0 = Instant::now();
    let _cpu_fp: Vec<_> = signals
        .iter()
        .zip(&cpu)
        .map(|(s, &(a_s, a_e))| {
            extract_fingerprint_from_signal(
                s,
                a_s,
                a_e,
                111,
                12,
                NormMethod::ZScore,
                Uuid::nil(),
                Some(6),
                Some(25),
                false,
            )
        })
        .collect();
    let cpu_fp_dt = tf0.elapsed();

    let fp_reads: Vec<(&[i16], usize, usize)> = signals
        .iter()
        .zip(&cpu)
        .map(|(s, &(a_s, a_e))| (s.as_slice(), a_s, a_e))
        .collect();
    let _ = ctx.fingerprint_batch(&fp_reads[..fp_reads.len().min(64)], 12, 111, 6, 25);
    let tf1 = Instant::now();
    let _gpu_fp = ctx
        .fingerprint_batch(&fp_reads, 12, 111, 6, 25)
        .expect("gpu fp");
    let gpu_fp_dt = tf1.elapsed();
    println!("--- fingerprint timing ---");
    println!(
        "CPU (1 thread): {:>8.1} ms   {:>7.1} µs/read",
        cpu_fp_dt.as_secs_f64() * 1e3,
        cpu_fp_dt.as_secs_f64() * 1e6 / signals.len() as f64
    );
    println!(
        "GPU (batched):  {:>8.1} ms   {:>7.1} µs/read",
        gpu_fp_dt.as_secs_f64() * 1e3,
        gpu_fp_dt.as_secs_f64() * 1e6 / signals.len() as f64
    );
    println!(
        "speedup vs 1 CPU thread: {:.1}x",
        cpu_fp_dt.as_secs_f64() / gpu_fp_dt.as_secs_f64()
    );
}
