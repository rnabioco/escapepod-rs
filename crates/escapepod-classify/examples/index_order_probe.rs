// SPDX-License-Identifier: MIT

//! What the two halves of escapepod-rs#334 are each worth, measured.
//!
//! The issue reports a `classify` run reading 20,000 of 12,129,287 reads
//! (0.165%) from a 173.7 GB POD5 at 17.6 reads/s — 236 major faults/s at 0.00
//! cores. Two causes are named, and the fix for each is separable, so the
//! measurement has to separate them too:
//!
//! 1. **The lookup.** `Pod5Index::build` used to decode every reads batch and
//!    keep the rows a BAM had anchored. It now goes through the reader's read
//!    index — the `.p5s` sidecar when there is one, an in-memory build when
//!    there is not.
//! 2. **The order.** Reads arrived in BAM order, which is `SO:coordinate` and
//!    so unrelated to POD5 layout. They are now sorted on
//!    `Pod5Index::storage_key` before any signal is touched.
//!
//! Three arms, therefore, each a complete lookup-plus-fetch of its own read
//! sample:
//!
//! * `scan+shuffled`  — the 0.19.0 path the issue measured.
//! * `scan+storage`   — ordering alone (the 0.21.0 column-variant path).
//! * `index+storage`  — this branch. Run the probe a second time after
//!   `escpod index` to get the sidecar-backed version of the same arm.
//!
//! **Every arm gets its own disjoint read sample**, because a shared one would
//! hand each later arm the previous arm's page cache — the confound the issue
//! calls out by name. Arms are interleaved across reps (forward, then reverse)
//! for the same reason `benchmarks/benchmark_charging.sh` interleaves. Samples
//! are small fractions of the file by construction, so what one arm warms is
//! mostly not what the next one reads.
//!
//! Arms therefore read *different bytes*, so wall clock alone would not compare
//! them: the report gives decoded MB and MB/s beside it. It also gives major
//! faults, which is the quantity the issue actually diagnosed on.
//!
//! Each arm prints a checksum over the decoded samples of its own reads, and
//! the same sample is re-fetched through the other path at the end
//! (`--verify`), which is how "order changes nothing but the clock" is checked
//! on real data rather than on a fixture.
//!
//! ```text
//! cargo run --release -p escapepod-classify --example index_order_probe -- \
//!     /path/to/pod5dir --ids 5000 --reps 2 --threads 16 --verify
//! ```
//!
//! Give it a directory of symlinks rather than the run itself if you do not
//! want `escpod index` writing sidecars into primary data.
//!
//! # What it measured (2026-09-07, rna, BeeGFS)
//!
//! **`--threads` is not a detail, it is the experiment.** On one worker the
//! ordering arm is worth *nothing* — 2,000 of 3,057,492 reads (0.065%), two
//! interleaved reps, sidecars present:
//!
//! ```text
//!   scan+shuffled   fetch 56.56s   1.7 MB/s
//!   scan+storage    fetch 57.56s   1.6 MB/s
//!   index+storage   fetch 60.04s   1.7 MB/s
//! ```
//!
//! There is nothing for readahead to exploit: one walker, and consecutive
//! selected reads ~120 MB apart. The same set at `--threads 16`:
//!
//! ```text
//!   scan+shuffled   fetch  7.29s    8.1 MB/s   ~344 reads/s
//!   scan+storage    fetch  4.47s   11.9 MB/s   ~499 reads/s   <- 1.63x
//!   index+storage   fetch  5.83s   11.9 MB/s   ~426 reads/s
//! ```
//!
//! With T workers there are T sweeps in flight, and whether they are T forward
//! sweeps or T random walks is the entire difference. A single-threaded probe
//! would have reported the ordering fix as worthless.
//!
//! **The lookup is not where the time is**, which the issue's framing implies
//! it is. Scan or index, sidecar or not, resolving 2,000 ids cost ~2 s. What
//! the sidecar actually buys is the *enumerate*, a fixed cost paid once per
//! process and scaling with the file set rather than with how many reads are
//! wanted: 3 files / 3.06 M reads went 5.7 s → 0.34 s, and 70 files / 27.45 M
//! reads (574 GB) cost 54.5 s with no sidecar at all.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use escapepod_signal::{Reader, ReadsBatchView, SignalExtractor, Uuid};

thread_local! {
    /// Worker count for the fetch phase, set once from `--threads`.
    static THREADS: usize = std::env::var("ESCAPEPOD_PROBE_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
}

/// Order-independent per-read digest: a sum over reads cannot notice which
/// read came first, so two arms that read the same sample in different orders
/// must agree.
fn hash_signal(signal: &[i16]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for s in signal {
        h = (h ^ (*s as u16 as u64)).wrapping_mul(0x100000001b3);
    }
    h
}

/// One read's signal location, as either lookup path resolves it.
#[derive(Clone)]
struct Located {
    reader_idx: usize,
    signal_rows: Vec<u64>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cfg = Config::from_args()?;

    let paths = collect_pod5(&cfg.inputs)?;
    if paths.is_empty() {
        return Err("no POD5 files found".into());
    }
    let total_bytes: u64 = paths
        .iter()
        .filter_map(|p| std::fs::metadata(p).ok())
        .map(|m| m.len())
        .sum();

    // --- Setup, untimed: what ids exist, and where they live. -------------
    //
    // Deliberately outside every arm: an arm must not be charged for the
    // sampling, and each arm opens its own fresh readers so that no arm
    // inherits another's cached index.
    let setup = Instant::now();
    let mut all_ids: Vec<Uuid> = Vec::new();
    let mut any_sidecar = false;
    for path in &paths {
        let reader = Reader::open(path)?;
        any_sidecar |= reader.has_sidecar_index();
        all_ids.extend(reader.read_index()?.uuids());
    }
    println!(
        "{} file(s), {:.1} GB, {} reads, sidecars {}, {} fetch thread(s) ({:.1?} to enumerate)",
        paths.len(),
        total_bytes as f64 / 1e9,
        all_ids.len(),
        if any_sidecar { "present" } else { "ABSENT" },
        cfg.threads,
        setup.elapsed(),
    );
    if cfg.ids * cfg.reps * 3 > all_ids.len() {
        return Err(format!(
            "need {} distinct ids for {} reps x 3 arms, file set has {}",
            cfg.ids * cfg.reps * 3,
            cfg.reps,
            all_ids.len()
        )
        .into());
    }

    // A fixed permutation, then disjoint windows off it: every arm of every
    // rep reads a different part of the file set.
    let mut rng = Xorshift::new(cfg.seed);
    shuffle(&mut all_ids, &mut rng);

    let arms = ["scan+shuffled", "scan+storage", "index+storage"];
    let mut taken = 0usize;
    let mut results: Vec<(usize, &str, Measure)> = Vec::new();

    for rep in 1..=cfg.reps {
        // Forward on odd reps, reversed on even: no arm keeps first place, so
        // a cold-start advantage cannot accrue to one of them.
        let order: Vec<usize> = if rep % 2 == 1 {
            (0..arms.len()).collect()
        } else {
            (0..arms.len()).rev().collect()
        };
        for &arm in &order {
            let sample: Vec<Uuid> = all_ids[taken..taken + cfg.ids].to_vec();
            taken += cfg.ids;
            let m = run_arm(arm, &paths, &sample, &mut rng)?;
            println!(
                "rep {rep}  {:<14}  lookup {:>8.2?}  fetch {:>8.2?}  total {:>8.2?}  \
                 {:>7.1} MB  {:>6.1} MB/s  {:>9} majflt  {:>8.1} reads/s  sum {:016x}",
                arms[arm],
                m.lookup,
                m.fetch,
                m.lookup + m.fetch,
                m.bytes as f64 / 1e6,
                m.bytes as f64 / 1e6 / (m.lookup + m.fetch).as_secs_f64(),
                m.majflt,
                cfg.ids as f64 / (m.lookup + m.fetch).as_secs_f64(),
                m.checksum,
            );
            results.push((rep, arms[arm], m));
        }
    }

    // --- Same reads, both orders: the equality the speedup must not cost. --
    if cfg.verify {
        let sample: Vec<Uuid> = all_ids[taken..taken + cfg.ids.min(2000)].to_vec();
        let a = run_arm(0, &paths, &sample, &mut rng)?;
        let b = run_arm(2, &paths, &sample, &mut rng)?;
        println!(
            "\nverify: same {} reads through scan+shuffled and index+storage — \
             checksum {:016x} vs {:016x}  bytes {} vs {}  =>  {}",
            sample.len(),
            a.checksum,
            b.checksum,
            a.bytes,
            b.bytes,
            if a.checksum == b.checksum && a.bytes == b.bytes {
                "IDENTICAL"
            } else {
                "*** DIVERGED ***"
            }
        );
    }

    println!("\nper-arm totals across {} rep(s):", cfg.reps);
    for arm in arms {
        let rows: Vec<&Measure> = results
            .iter()
            .filter(|(_, a, _)| *a == arm)
            .map(|(_, _, m)| m)
            .collect();
        let lookup: Duration = rows.iter().map(|m| m.lookup).sum();
        let fetch: Duration = rows.iter().map(|m| m.fetch).sum();
        let bytes: u64 = rows.iter().map(|m| m.bytes).sum();
        let majflt: u64 = rows.iter().map(|m| m.majflt).sum();
        println!(
            "  {arm:<14}  lookup {:>8.2?}  fetch {:>8.2?}  {:>6.1} MB/s  {:>9} majflt",
            lookup,
            fetch,
            bytes as f64 / 1e6 / (lookup + fetch).as_secs_f64(),
            majflt,
        );
    }
    Ok(())
}

struct Measure {
    lookup: Duration,
    fetch: Duration,
    bytes: u64,
    majflt: u64,
    checksum: u64,
}

/// One arm end to end, on readers opened for this arm alone.
fn run_arm(
    arm: usize,
    paths: &[PathBuf],
    sample: &[Uuid],
    rng: &mut Xorshift,
) -> Result<Measure, Box<dyn std::error::Error>> {
    let readers: Vec<Reader> = paths
        .iter()
        .map(Reader::open)
        .collect::<escapepod_signal::Result<_>>()?;
    let wanted: HashSet<Uuid> = sample.iter().copied().collect();

    let majflt0 = major_faults();
    let t0 = Instant::now();
    let located = match arm {
        // The 0.19.0 lookup, kept here because it no longer exists in the
        // crate: decode every reads batch, keep the rows that were wanted.
        0 | 1 => locate_by_scan(&readers, &wanted)?,
        _ => locate_by_index(&readers, &wanted)?,
    };
    let lookup = t0.elapsed();

    let mut order: Vec<Uuid> = located.keys().copied().collect();
    if arm == 0 {
        // Stand-in for BAM order. A uniform shuffle is the *conservative*
        // version of it: a coordinate-sorted BAM over a tRNA reference groups
        // reads by identity, which is structured against storage order rather
        // than merely uncorrelated with it.
        shuffle(&mut order, rng);
    } else {
        order.sort_by_key(|id| {
            let l = &located[id];
            (l.reader_idx, l.signal_rows.first().copied().unwrap_or(0))
        });
    }

    let extractors: Vec<SignalExtractor<'_>> = readers
        .iter()
        .map(|r| r.signal_extractor())
        .collect::<escapepod_signal::Result<_>>()?;

    // Threading is the whole question, not a detail. `classify_reads` hands
    // rayon `par_chunks` of the ordered reads, so with T workers there are T
    // sweeps in flight; whether they are T *forward* sweeps or T random walks
    // is exactly what the ordering fix decides. A single-threaded probe cannot
    // see that difference and will report that ordering does nothing — which
    // is what the first run of this probe did.
    //
    // `--threads 1` keeps the old shape for reference.
    let t1 = Instant::now();
    let (bytes, checksum) = if THREADS.with(|t| *t) <= 1 {
        let mut bytes = 0u64;
        let mut checksum = 0u64;
        for id in &order {
            let l = &located[id];
            let signal = extractors[l.reader_idx].get_signal(&l.signal_rows)?;
            bytes += (signal.len() * 2) as u64;
            checksum = checksum.wrapping_add(hash_signal(&signal));
        }
        (bytes, checksum)
    } else {
        use rayon::prelude::*;
        // The chunk size `classify_reads` uses for the native BiLSTM group.
        let per: Vec<(u64, u64)> = order
            .par_chunks(24)
            .map(|chunk| -> Result<(u64, u64), String> {
                let mut b = 0u64;
                let mut c = 0u64;
                for id in chunk {
                    let l = &located[id];
                    let signal = extractors[l.reader_idx]
                        .get_signal(&l.signal_rows)
                        .map_err(|e| e.to_string())?;
                    b += (signal.len() * 2) as u64;
                    c = c.wrapping_add(hash_signal(&signal));
                }
                Ok((b, c))
            })
            .collect::<Result<Vec<_>, String>>()?;
        per.into_iter().fold((0u64, 0u64), |(b, c), (nb, nc)| {
            (b + nb, c.wrapping_add(nc))
        })
    };
    let fetch = t1.elapsed();

    Ok(Measure {
        lookup,
        fetch,
        bytes,
        majflt: major_faults().saturating_sub(majflt0),
        checksum,
    })
}

/// The lookup this branch replaced: every reads batch decoded, wanted rows kept.
fn locate_by_scan(
    readers: &[Reader],
    wanted: &HashSet<Uuid>,
) -> Result<HashMap<Uuid, Located>, Box<dyn std::error::Error>> {
    let mut out = HashMap::new();
    for (reader_idx, reader) in readers.iter().enumerate() {
        for batch in reader.read_batches()? {
            let batch = batch?;
            let view = ReadsBatchView::new(&batch, false)?;
            for row in 0..view.num_rows() {
                let read = view.read(row)?;
                if wanted.contains(&read.read_id) {
                    out.insert(
                        read.read_id,
                        Located {
                            reader_idx,
                            signal_rows: read.signal_rows,
                        },
                    );
                }
            }
        }
    }
    Ok(out)
}

/// The lookup this branch installed.
fn locate_by_index(
    readers: &[Reader],
    wanted: &HashSet<Uuid>,
) -> Result<HashMap<Uuid, Located>, Box<dyn std::error::Error>> {
    let mut out = HashMap::new();
    for (reader_idx, reader) in readers.iter().enumerate() {
        for found in reader.find_signal_rows_with_calibration_by_ids(wanted)? {
            out.insert(
                found.read_id,
                Located {
                    reader_idx,
                    signal_rows: found.signal_rows,
                },
            );
        }
    }
    Ok(out)
}

/// Major faults this process has taken (field 12 of `/proc/self/stat`).
///
/// The quantity the issue diagnosed on: a POD5 is an mmap, so its I/O never
/// appears in `read_bytes`, and a run stalled on faults shows no CPU either.
fn major_faults() -> u64 {
    let Ok(stat) = std::fs::read_to_string("/proc/self/stat") else {
        return 0;
    };
    // Field 2 (comm) may contain spaces inside parentheses; split after it.
    let Some((_, rest)) = stat.rsplit_once(") ") else {
        return 0;
    };
    // `rest` starts at field 3 (state), so majflt (field 12) is the tenth.
    rest.split_whitespace()
        .nth(9)
        .and_then(|v| v.parse().ok())
        .unwrap_or(0)
}

fn collect_pod5(inputs: &[PathBuf]) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
    let mut out = Vec::new();
    for input in inputs {
        if input.is_dir() {
            for entry in std::fs::read_dir(input)? {
                let p = entry?.path();
                if p.extension().is_some_and(|e| e == "pod5") {
                    out.push(p);
                }
            }
        } else {
            out.push(input.clone());
        }
    }
    out.sort();
    Ok(out)
}

/// Deterministic, dependency-free shuffle so a rerun of the probe reads the
/// same reads in the same order.
struct Xorshift(u64);

impl Xorshift {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

fn shuffle<T>(v: &mut [T], rng: &mut Xorshift) {
    for i in (1..v.len()).rev() {
        v.swap(i, (rng.next() % (i as u64 + 1)) as usize);
    }
}

struct Config {
    inputs: Vec<PathBuf>,
    threads: usize,
    ids: usize,
    reps: usize,
    seed: u64,
    verify: bool,
}

impl Config {
    fn from_args() -> Result<Self, Box<dyn std::error::Error>> {
        let mut cfg = Config {
            inputs: Vec::new(),
            threads: 1,
            ids: 5000,
            reps: 2,
            seed: 0x5eed,
            verify: false,
        };
        let mut args = std::env::args().skip(1);
        while let Some(a) = args.next() {
            match a.as_str() {
                "--ids" => cfg.ids = args.next().ok_or("--ids needs a value")?.parse()?,
                "--reps" => cfg.reps = args.next().ok_or("--reps needs a value")?.parse()?,
                "--seed" => cfg.seed = args.next().ok_or("--seed needs a value")?.parse()?,
                "--verify" => cfg.verify = true,
                "--threads" => {
                    let n: usize = args.next().ok_or("--threads needs a value")?.parse()?;
                    // Read back by every rayon worker in the fetch phase.
                    unsafe { std::env::set_var("ESCAPEPOD_PROBE_THREADS", n.to_string()) };
                    rayon::ThreadPoolBuilder::new()
                        .num_threads(n)
                        .build_global()?;
                    cfg.threads = n;
                }
                other if other.starts_with("--") => {
                    return Err(format!("unknown flag {other}").into());
                }
                other => cfg.inputs.push(PathBuf::from(other)),
            }
        }
        if cfg.inputs.is_empty() {
            return Err("usage: index_order_probe <pod5...> [--ids N] [--reps N] [--seed N] [--threads N] [--verify]".into());
        }
        Ok(cfg)
    }
}
