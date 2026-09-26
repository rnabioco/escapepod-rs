// SPDX-License-Identifier: MIT

//! Both SIMD score-kernel loop orders on long reads, one thread, so the
//! read length at which `Aligner` switches to the reference-major
//! (`Kernel::Transposed`) kernel is a measurement rather than a guess
//! (rnabioco/escapepod-rs#415). Keep it re-runnable.
//!
//! ```text
//! cargo run --release -p escapepod-align --example long_read_probe
//! cargo run --release -p escapepod-align --example long_read_probe -- \
//!     --reference sacCer3-mature-tRNAs-dual-adapt-v2.fa --reads long_read.fa --split
//! ```
//!
//! Without `--reference` the panel is synthetic: 164 random references of
//! 138–200 nt, the shape of the sacCer3 dual-adapter panel. Without
//! `--reads` the reads are random, at `--lengths` (default 1 k, 2 k, 4 k,
//! 16 k, 64 k, 256 k and 400 k nt) — the DP's cost does not depend on the
//! bases. With `--reads FASTA` each record is timed instead. `--split` also
//! times each read's lane groups on one thread per group (`std::thread`),
//! the way `escpod align` spreads a long read over its pool.
//!
//! Output is TSV on stdout: backend, kernel, read length, seconds per read,
//! cells/s (read length × panel bases, the useful cells), and for `--split`
//! the wall time with one thread per group.

use std::time::{Duration, Instant};

use escapepod_align::alphabet::encode;
use escapepod_align::simd::Kernel;
use escapepod_align::{Aligner, Backend, Mode, Panel, Scoring};

fn xorshift(x: &mut u64) -> u64 {
    *x ^= *x << 13;
    *x ^= *x >> 7;
    *x ^= *x << 17;
    *x
}

fn random_seq(x: &mut u64, len: usize) -> Vec<u8> {
    (0..len)
        .map(|_| b"ACGT"[(xorshift(x) % 4) as usize])
        .collect()
}

/// Time `f` until at least `budget` has passed (at least once); seconds per call.
fn time(budget: Duration, mut f: impl FnMut()) -> f64 {
    let t0 = Instant::now();
    let mut n = 0u32;
    while n == 0 || t0.elapsed() < budget {
        f();
        n += 1;
    }
    t0.elapsed().as_secs_f64() / n as f64
}

fn main() {
    let mut args = std::env::args().skip(1);
    let (mut reference, mut reads_path, mut split) = (None, None, false);
    let mut lengths = vec![1_000, 2_000, 4_000, 16_000, 64_000, 256_000, 400_000];
    let mut mode = Mode::Local;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--reference" => reference = args.next(),
            "--reads" => reads_path = args.next(),
            "--split" => split = true,
            "--mode" => mode = args.next().expect("--mode MODE").parse().expect("mode"),
            "--lengths" => {
                lengths = args
                    .next()
                    .expect("--lengths N,N,...")
                    .split(',')
                    .map(|s| s.parse().expect("length"))
                    .collect()
            }
            other => panic!("unknown argument {other}"),
        }
    }
    let read_fasta = |path: &str| {
        let f = std::fs::File::open(path).unwrap_or_else(|e| panic!("{path}: {e}"));
        escapepod_align::fasta::read_fasta(std::io::BufReader::new(f)).expect("FASTA")
    };
    let mut x = 0x0123_4567_89ab_cdefu64;
    let panel = match &reference {
        Some(path) => Panel::new(read_fasta(path)).unwrap(),
        None => Panel::new((0..164).map(|k| {
            let len = 138 + (xorshift(&mut x) % 63) as usize;
            (format!("r{k}"), random_seq(&mut x, len))
        }))
        .unwrap(),
    };
    let reads: Vec<(String, Vec<u8>)> = match &reads_path {
        Some(path) => read_fasta(path),
        None => lengths
            .iter()
            .map(|&n| (format!("random_{n}"), random_seq(&mut x, n)))
            .collect(),
    };
    let scoring = Scoring::default();
    eprintln!(
        "# panel: {} references, {} bases (longest {}); {} mode, scoring {scoring}",
        panel.len(),
        panel.total_len(),
        panel.max_len(),
        mode.name()
    );
    println!("backend\tkernel\tread_len\tsecs_per_read\tcells_per_s\tgroups\tsplit_wall_s");
    for backend in Backend::available() {
        if backend == Backend::Scalar {
            continue;
        }
        let base = Aligner::with_backend(panel.clone(), scoring, mode, backend).unwrap();
        for (name, seq) in &reads {
            let q = encode(seq);
            let cells = q.len() as f64 * panel.total_len() as f64;
            // Budget: long enough to average short reads, one call for long ones.
            let budget = Duration::from_millis(300);
            let mut row0: Option<Vec<i32>> = None;
            for kernel in [Kernel::RowMajor, Kernel::Transposed] {
                let a = base.clone().with_transposed_min_len(match kernel {
                    Kernel::RowMajor => None,
                    Kernel::Transposed => Some(0),
                });
                let mut out = Vec::new();
                let secs = time(budget, || a.score_all(&q, &mut out));
                match &row0 {
                    None => row0 = Some(out.clone()),
                    Some(r) => assert_eq!(r, &out, "{name}: kernels disagree"),
                }
                let groups = a.score_groups(q.len());
                let split_wall = if split {
                    let t = time(budget, || {
                        std::thread::scope(|s| {
                            for g in 0..groups {
                                let (a, q) = (&a, &q);
                                s.spawn(move || {
                                    let mut unit = Vec::new();
                                    a.score_group(q, g, &mut unit);
                                    unit
                                });
                            }
                        })
                    });
                    format!("{t:.4}")
                } else {
                    "-".into()
                };
                println!(
                    "{}\t{}\t{}\t{:.6}\t{:.3e}\t{}\t{}",
                    backend.name(),
                    kernel.name(),
                    q.len(),
                    secs,
                    cells / secs,
                    groups,
                    split_wall
                );
            }
        }
    }
}
