//! Measure the CPU prep breakdown for the demux pipeline: decode (signal
//! extraction) vs adapter detect (LLR) vs fingerprint extraction.
//!
//! This tells us which prep stage dominates wall-clock — the decision gate for
//! how much GPU-offload effort each stage is worth. The classify stage is
//! excluded (already GPU-accelerated and measured elsewhere).
//!
//! Run: `cargo run --release --example prep_breakdown -- <file.pod5> [max_reads]`

use std::time::Instant;

use escapepod_demux::extract_fingerprint_from_signal;
use escapepod_signal::Reader;
use escapepod_signal::dtw::NormMethod;
use escapepod_signal::segmentation::{detect_adapter, downscale, normalize_signal};
use uuid::Uuid;

// Demux defaults (FpParams + LLR detect args).
const MIN_ADAPTER: usize = 200;
const BORDER_TRIM: usize = 50;
const DOWNSCALE: usize = 1;
const NUM_SEGMENTS: usize = 111;
const WINDOW_WIDTH: usize = 12;
const MIN_SEP: usize = 6;
const KEEP_LAST: usize = 25;

fn llr_detect(signal: &[i16]) -> (usize, usize) {
    let normalized = normalize_signal(signal);
    let (processed, scale) = if DOWNSCALE > 1 {
        let trunc = (normalized.len() / DOWNSCALE) * DOWNSCALE;
        (downscale(&normalized[..trunc], DOWNSCALE), DOWNSCALE)
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

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .expect("usage: prep_breakdown <file.pod5> [max_reads]");
    let max_reads: usize = args
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or(usize::MAX);

    let reader = Reader::open(&path).expect("open pod5");
    let extractor = reader.signal_extractor().expect("signal extractor");

    let mut n = 0usize;
    let mut total_samples = 0u64;
    let mut t_decode = 0u128;
    let mut t_detect = 0u128;
    let mut t_fp = 0u128;

    // Iterate reads; for each, time the three prep stages separately.
    for read in reader.reads().expect("reads") {
        let read = read.expect("read");
        if read.signal_rows.is_empty() {
            continue;
        }

        let t0 = Instant::now();
        let signal = match extractor.get_signal(&read.signal_rows) {
            Ok(s) => s,
            Err(_) => continue,
        };
        t_decode += t0.elapsed().as_nanos();
        total_samples += signal.len() as u64;

        let t1 = Instant::now();
        let (s, e) = llr_detect(&signal);
        t_detect += t1.elapsed().as_nanos();

        let t2 = Instant::now();
        let _fp = extract_fingerprint_from_signal(
            &signal,
            s,
            e,
            NUM_SEGMENTS,
            WINDOW_WIDTH,
            NormMethod::ZScore,
            Uuid::nil(),
            Some(MIN_SEP),
            Some(KEEP_LAST),
            false,
        );
        t_fp += t2.elapsed().as_nanos();

        n += 1;
        if n >= max_reads {
            break;
        }
    }

    let total = (t_decode + t_detect + t_fp).max(1) as f64;
    let ms = |x: u128| x as f64 / 1e6;
    let pct = |x: u128| x as f64 / total * 100.0;
    println!("reads:          {n}");
    println!(
        "total samples:  {total_samples} ({:.0} avg/read)",
        total_samples as f64 / n.max(1) as f64
    );
    println!("--- prep stage breakdown (single-thread, wall) ---");
    println!(
        "decode (vbz):   {:>9.1} ms  {:>5.1}%   {:.1} µs/read",
        ms(t_decode),
        pct(t_decode),
        ms(t_decode) * 1000.0 / n as f64
    );
    println!(
        "detect (LLR):   {:>9.1} ms  {:>5.1}%   {:.1} µs/read",
        ms(t_detect),
        pct(t_detect),
        ms(t_detect) * 1000.0 / n as f64
    );
    println!(
        "fingerprint:    {:>9.1} ms  {:>5.1}%   {:.1} µs/read",
        ms(t_fp),
        pct(t_fp),
        ms(t_fp) * 1000.0 / n as f64
    );
    println!("total prep:     {:>9.1} ms", ms(t_decode + t_detect + t_fp));
}
