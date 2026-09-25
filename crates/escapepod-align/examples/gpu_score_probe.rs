// SPDX-License-Identifier: MIT

//! Time the CUDA score kernel alone, on real reads, without BAM I/O or
//! tracebacks: the measurement to re-run when the kernel changes.
//!
//! ```text
//! samtools view reads.bam | cut -f10 > reads.seq      # one sequence per line
//! cargo run --release --features gpu -p escapepod-align --example gpu_score_probe -- \
//!     panel.fa reads.seq [batch_size] [local|semiglobal]
//! ```
//!
//! Prints the kernel's geometry and registers, then the total seconds in
//! `score_batch` (host packing, transfers and the kernel) and cells per
//! second (`read length × panel bases`, GPU-scored reads only), after one
//! untimed warm-up batch.

use std::io::BufRead;
use std::time::Instant;

use escapepod_align::alphabet::encode;
use escapepod_align::cuda::GpuScorer;
use escapepod_align::{Mode, Panel, Scoring};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    assert!(
        args.len() >= 3,
        "usage: gpu_score_probe PANEL.fa READS.seq [BATCH] [MODE]"
    );
    let batch: usize = args.get(3).map_or(50_000, |s| s.parse().unwrap());
    let mode: Mode = args.get(4).map_or(Mode::Local, |s| s.parse().unwrap());

    let mut refs: Vec<(String, Vec<u8>)> = Vec::new();
    for line in std::fs::read_to_string(&args[1]).unwrap().lines() {
        if let Some(h) = line.strip_prefix('>') {
            refs.push((h.split_whitespace().next().unwrap().to_string(), Vec::new()));
        } else {
            refs.last_mut().unwrap().1.extend(line.trim().bytes());
        }
    }
    let panel = Panel::new(refs).unwrap();
    let reads: Vec<Vec<u8>> = std::io::BufReader::new(std::fs::File::open(&args[2]).unwrap())
        .lines()
        .map(|l| encode(l.unwrap().trim().as_bytes()))
        .collect();

    let t0 = Instant::now();
    let mut gpu = GpuScorer::new(&panel, Scoring::default(), mode).unwrap();
    let (lanes, groups, blocks) = gpu.geometry();
    println!(
        "{}: {groups} group(s) x {blocks} blocks x {lanes} threads, {:?} registers, ready in {:.2} s",
        gpu.device_name(),
        gpu.registers(),
        t0.elapsed().as_secs_f64()
    );
    let chunks: Vec<Vec<&[u8]>> = reads
        .chunks(batch)
        .map(|c| c.iter().map(Vec::as_slice).collect())
        .collect();
    gpu.score_batch(&chunks[0]).unwrap(); // warm-up: JIT, first allocations

    let (mut secs, mut cells, mut scored) = (0.0, 0f64, 0usize);
    let panel_bases = panel.total_len() as f64;
    for c in &chunks {
        let t = Instant::now();
        let m = gpu.score_batch(c).unwrap();
        secs += t.elapsed().as_secs_f64();
        for (k, q) in c.iter().enumerate() {
            if m.row(k).is_some() {
                cells += q.len() as f64 * panel_bases;
                scored += 1;
            }
        }
    }
    println!(
        "{} reads ({scored} on the GPU) in {} batches: {secs:.2} s, {:.3e} cells/s ({})",
        reads.len(),
        chunks.len(),
        cells / secs,
        mode.name()
    );
}
