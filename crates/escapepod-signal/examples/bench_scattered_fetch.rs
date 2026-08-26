//! Benchmark: where does the time go in a scattered N-read fetch?
//!
//! `bench_reads_by_ids` answers "sidecar or built index?". This one answers the
//! next question: once the index is warm, a fetch of ~10^5 reads scattered
//! across a large POD5 is still slow — *which phase* is it?
//!
//! It splits the fetch into the three costs that are actually distinct, and
//! that a `.p5s` change would affect differently:
//!
//! 1. **Signal footer parse.** `ArrowIpcFooter::parse` reads every record
//!    batch's message header (`parse_batch_row_count`) — one scattered mmap
//!    touch per batch, thousands on a large file. Probed here through
//!    `signal_batch_row_counts()`, which forces exactly that parse and nothing
//!    else. This is the cost that a constant-stride derivation would remove.
//! 2. **Reads-table decode.** `find_signal_rows_by_ids` resolves UUIDs to
//!    signal rows, decoding each reads batch that holds a target — 10k rows
//!    per batch even when a handful are wanted. This is the cost that storing
//!    the signal extent in the sidecar would remove outright.
//! 3. **Signal fetch.** `get_signal_bulk`: page-faulting the compressed bytes
//!    plus VBZ decode. This is irreducible bytes; only *ordering* it better
//!    (one monotonic sweep) can help, not eliminating work.
//!
//! The verdict this is meant to settle: if (3) dominates, a sidecar format
//! change buys only the offset-sorted sweep and is not worth a version bump on
//! its own. If (1)+(2) are a large share, they are.
//!
//! Two driver shapes are timed, because they are not the same:
//!
//! * **one-shot** — all ids in a single `find` + `get_signal_bulk` pair.
//! * **chunked** — ids in batches of `--chunk` (what `rnabioco/leech` does).
//!   Anything re-derived per call is paid `n_ids / chunk` times here and once
//!   above, so the gap between the two is the per-call fixed cost.
//!
//! Usage:
//! ```text
//! cargo run --release -p escapepod-signal --example bench_scattered_fetch -- \
//!     <pod5_file> [--ids N] [--chunk N] [--order spread|random|contiguous] [--seed N]
//! ```
//!
//! `--order` controls *which* reads are asked for, not the order they are asked
//! in (the request is always shuffled, as a `HashSet<Uuid>` iteration is):
//! `spread` (default) samples evenly across the file, `random` picks uniformly
//! at random, `contiguous` takes a single run — the best case, for reference.
//!
//! Page cache matters more than anything else here. Vary `--seed` between runs
//! on a file too large to cache, and treat a file that fits in RAM as a warm
//! measurement only.

use escapepod_signal::Reader;
use escapepod_signal::pod5::Uuid;
use std::collections::HashSet;
use std::time::{Duration, Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cfg = match Config::from_args() {
        Ok(c) => c,
        Err(msg) => {
            eprintln!("{msg}");
            std::process::exit(2);
        }
    };

    let file_bytes = std::fs::metadata(&cfg.pod5_path)?.len();
    let has_sidecar = std::path::Path::new(&format!("{}.p5s", cfg.pod5_path)).exists();

    println!("file       {}", cfg.pod5_path);
    println!(
        "size       {:.2} GB    sidecar: {}",
        file_bytes as f64 / 1e9,
        if has_sidecar {
            "yes"
        } else {
            "no (index will be built)"
        }
    );
    println!(
        "request    {} ids, order={}, chunk={}, seed={}\n",
        cfg.n_ids, cfg.order, cfg.chunk, cfg.seed
    );

    // ---- Phase 1: signal footer parse -----------------------------------
    // A fresh reader, so the `OnceLock` is cold and this is the real parse.
    // `signal_batch_row_counts()` is the narrowest public call that forces it.
    //
    // Measured twice, against two *different* readers. The first pays the cold
    // page faults for every batch header; the second finds those pages in the
    // page cache and so measures only the CPU of the same walk. The gap is the
    // scattered-I/O cost, which is the whole point — it is what scales with
    // batch count and what a constant-stride derivation would delete.
    let reader = Reader::open(&cfg.pod5_path)?;
    let t = Instant::now();
    let batch_rows = reader.signal_batch_row_counts();
    let t_footer_cold = t.elapsed();

    let probe = Reader::open(&cfg.pod5_path)?;
    let t = Instant::now();
    let _ = probe.signal_batch_row_counts();
    let t_footer_warm = t.elapsed();
    drop(probe);

    let n_sig_batches = batch_rows.len();
    let stride = batch_rows.first().copied().unwrap_or(0);
    let uniform = reader.nonuniform_signal_batch().is_none();
    println!("--- signal footer ---");
    println!(
        "  {n_sig_batches} batches x {stride} rows  (uniform stride: {})",
        if uniform { "yes" } else { "NO" }
    );
    println!("  parse, cold cache        {t_footer_cold:>12.2?}");
    println!("  parse, warm cache        {t_footer_warm:>12.2?}   (same walk, no faults)");
    if uniform && n_sig_batches > 2 {
        println!(
            "  ^ {n_sig_batches} scattered header reads to recover a stride that is\n    \
             constant — derivable from batch 0 + the total, at no I/O."
        );
    }

    // ---- Phase 2: read index --------------------------------------------
    let total_reads = reader.read_count()?;
    let t = Instant::now();
    let index = reader.read_index()?;
    let t_index = t.elapsed();
    println!("\n--- read index ---");
    println!(
        "  {} reads, {} indexed  ({})",
        total_reads,
        index.len(),
        if has_sidecar {
            "loaded from .p5s"
        } else {
            "built by projected scan"
        }
    );
    println!("  load/build               {t_index:>12.2?}");

    // ---- Select targets --------------------------------------------------
    //
    // Split into two DISJOINT halves, one per driver shape. Running both arms
    // over the same ids would let whichever ran first fault in every byte the
    // second one needs, and the second would then measure the page cache
    // rather than the driver — a ~170x effect on a network filesystem, which
    // is far larger than anything being compared. Disjoint sets keep both arms
    // equally cold. The order they run in also alternates with the seed, so a
    // residual first-mover advantage (readahead, BeeGFS client warmup) does
    // not always land on the same arm.
    let all_ids: Vec<Uuid> = index.uuids().collect();
    let targets = select_targets(&all_ids, &cfg);
    // Sorted before splitting: iterating the `HashSet` puts a different set of
    // ids in each half on every run, which makes two runs of the same seed
    // incomparable — and silently so, since only the reported sample counts
    // move. Sorting by UUID keeps the request unordered with respect to the
    // *file* (a UUID sort is unrelated to file order), which is the property
    // being modelled, while making the split reproducible.
    let mut ids: Vec<Uuid> = targets.iter().copied().collect();
    ids.sort_unstable_by_key(|u| *u.as_bytes());
    let (set_a, set_b) = ids.split_at(ids.len() / 2);
    let set_a: HashSet<Uuid> = set_a.iter().copied().collect();
    let n = targets.len();
    println!(
        "\n--- fetching {n} reads ({:.3}% of file), split into two disjoint halves ---",
        100.0 * n as f64 / total_reads.max(1) as f64
    );

    let mut one_shot: Option<Phase> = None;
    let mut chunked: Option<Chunked> = None;
    let chunked_first = cfg.seed % 2 == 1;

    for run_chunked in [chunked_first, !chunked_first] {
        if run_chunked {
            chunked = Some(run_chunked_arm(&reader, set_b, cfg.chunk)?);
        } else {
            one_shot = Some(run_one_shot_arm(&reader, &set_a)?);
        }
    }
    let one_shot = one_shot.expect("one-shot arm ran");
    let chunked = chunked.expect("chunked arm ran");

    println!("\n  == one-shot: {} ids in 1 call ==", one_shot.n_reads);
    println!(
        "  find_signal_rows_by_ids  {:>12.2?}   ({} signal rows)",
        one_shot.t_find, one_shot.n_signal_rows
    );
    println!(
        "  get_signal_bulk          {:>12.2?}   ({:.1} M samples, {:.0} MB/s decoded)",
        one_shot.t_signal,
        one_shot.samples as f64 / 1e6,
        (one_shot.samples * 2) as f64 / 1e6 / one_shot.t_signal.as_secs_f64().max(1e-9),
    );

    // Both halves together, so a run can be checked against another
    // configuration of the same seed — notably with and without a `.p5s`,
    // which must not change a single sample.
    println!(
        "\n  checksum: {} reads, {} signal rows, {} samples total",
        one_shot.n_reads + chunked.phase.n_reads,
        one_shot.n_signal_rows + chunked.phase.n_signal_rows,
        one_shot.samples + chunked.phase.samples,
    );

    println!(
        "\n  == chunked: {} ids in {} calls of {} ==",
        chunked.phase.n_reads, chunked.calls, cfg.chunk
    );
    println!("  find_signal_rows_by_ids  {:>12.2?}", chunked.phase.t_find);
    println!(
        "  get_signal_bulk          {:>12.2?}",
        chunked.phase.t_signal
    );
    println!(
        "  total                    {:>12.2?}",
        chunked.phase.t_find + chunked.phase.t_signal
    );

    // Per-read, so the two halves compare directly even if they differ by one.
    let per_read =
        |p: &Phase| (p.t_find + p.t_signal).as_secs_f64() / p.n_reads.max(1) as f64 * 1e6;
    println!(
        "\n  per read: one-shot {:.1} us   chunked {:.1} us   ratio {:.2}x",
        per_read(&one_shot),
        per_read(&chunked.phase),
        per_read(&chunked.phase) / per_read(&one_shot).max(1e-9),
    );
    println!(
        "  ({} ran first — and on a network filesystem the arm that runs first\n   \
         is systematically slower by roughly 2x, whichever one it is. Do NOT read\n   \
         this ratio from a single run: measure both seed parities and compare only\n   \
         same-position arms, or the number you get is client warmup.)",
        if chunked_first { "chunked" } else { "one-shot" }
    );

    // ---- Attribution -----------------------------------------------------
    //
    // Scaled to the whole request so the two halves sum to one comparable
    // fetch, and stated against the cold footer parse — the number a process
    // that opens the file once and fetches once actually pays.
    let t_find = one_shot.t_find + chunked.phase.t_find;
    let t_signal = one_shot.t_signal + chunked.phase.t_signal;
    let total = t_footer_cold + t_index + t_find + t_signal;
    let pct = |d: Duration| 100.0 * d.as_secs_f64() / total.as_secs_f64().max(1e-9);

    println!("\n--- attribution (both halves, cold) ---");
    println!(
        "  signal footer parse      {t_footer_cold:>12.2?}  {:>5.1}%   fix in-crate: constant stride",
        pct(t_footer_cold)
    );
    println!(
        "  read index load/build    {t_index:>12.2?}  {:>5.1}%   once per file, not per call",
        pct(t_index)
    );
    println!(
        "  reads-table decode       {t_find:>12.2?}  {:>5.1}%   needs .p5s signal extents",
        pct(t_find)
    );
    println!(
        "  signal fault + VBZ       {t_signal:>12.2?}  {:>5.1}%   irreducible bytes; reorderable only",
        pct(t_signal)
    );
    println!("  {:-<62}", "");
    println!("  total                    {total:>12.2?}");

    // The two levers are NOT interchangeable, so they are judged separately.
    // The footer parse is fixable inside the crate with no format change; only
    // the reads-table decode needs a sidecar that carries signal extents.
    println!("\n--- verdict ---");
    println!(
        "  footer parse:      {:>5.1}% of cold total  -> {}",
        pct(t_footer_cold),
        if pct(t_footer_cold) >= 10.0 {
            "WORTH FIXING (no format change needed)"
        } else {
            "not worth chasing"
        }
    );
    println!(
        "  reads-table decode:{:>5.1}% of cold total  -> {}",
        pct(t_find),
        if pct(t_find) >= 15.0 {
            "a .p5s carrying signal extents would pay for itself"
        } else {
            "too small to justify a .p5s version bump on its own"
        }
    );

    Ok(())
}

/// Timings for one driver shape over one disjoint half of the request.
struct Phase {
    t_find: Duration,
    t_signal: Duration,
    n_reads: usize,
    n_signal_rows: usize,
    samples: usize,
}

struct Chunked {
    phase: Phase,
    calls: usize,
}

fn run_one_shot_arm(
    reader: &Reader,
    targets: &HashSet<Uuid>,
) -> Result<Phase, Box<dyn std::error::Error>> {
    let t = Instant::now();
    let rows = reader.find_signal_rows_by_ids(targets)?;
    let t_find = t.elapsed();

    let n_signal_rows: usize = rows.iter().map(|(_, r)| r.len()).sum();
    let t = Instant::now();
    let signal = reader.get_signal_bulk(&rows)?;
    let t_signal = t.elapsed();

    Ok(Phase {
        t_find,
        t_signal,
        n_reads: rows.len(),
        n_signal_rows,
        samples: signal.iter().map(|(_, s)| s.len()).sum(),
    })
}

/// The shape a downstream consumer actually uses (`rnabioco/leech` fetches in
/// batches). Anything re-derived per call is paid `ceil(n / chunk)` times here
/// against once in the one-shot arm, so the per-read gap between the two is
/// that fixed cost.
fn run_chunked_arm(
    reader: &Reader,
    ids: &[Uuid],
    chunk: usize,
) -> Result<Chunked, Box<dyn std::error::Error>> {
    let mut phase = Phase {
        t_find: Duration::ZERO,
        t_signal: Duration::ZERO,
        n_reads: 0,
        n_signal_rows: 0,
        samples: 0,
    };
    let mut calls = 0usize;
    for slice in ids.chunks(chunk) {
        let sub: HashSet<Uuid> = slice.iter().copied().collect();
        let t = Instant::now();
        let rows = reader.find_signal_rows_by_ids(&sub)?;
        phase.t_find += t.elapsed();
        let t = Instant::now();
        let got = reader.get_signal_bulk(&rows)?;
        phase.t_signal += t.elapsed();
        phase.n_reads += rows.len();
        phase.n_signal_rows += rows.iter().map(|(_, r)| r.len()).sum::<usize>();
        phase.samples += got.iter().map(|(_, s)| s.len()).sum::<usize>();
        calls += 1;
    }
    Ok(Chunked { phase, calls })
}

// ---------------------------------------------------------------------------

struct Config {
    pod5_path: String,
    n_ids: usize,
    chunk: usize,
    order: String,
    seed: u64,
}

impl Config {
    fn from_args() -> Result<Self, String> {
        let args: Vec<String> = std::env::args().skip(1).collect();
        let mut cfg = Config {
            pod5_path: String::new(),
            n_ids: 100_000,
            chunk: 1_000,
            order: "spread".to_string(),
            seed: 1,
        };
        let mut i = 0;
        while i < args.len() {
            let a = &args[i];
            let val = |name: &str| -> Result<String, String> {
                args.get(i + 1)
                    .cloned()
                    .ok_or_else(|| format!("{name} needs a value"))
            };
            match a.as_str() {
                "--ids" => {
                    cfg.n_ids = val("--ids")?.parse().map_err(|e| format!("--ids: {e}"))?;
                    i += 2;
                }
                "--chunk" => {
                    cfg.chunk = val("--chunk")?
                        .parse()
                        .map_err(|e| format!("--chunk: {e}"))?;
                    i += 2;
                }
                "--order" => {
                    cfg.order = val("--order")?;
                    i += 2;
                }
                "--seed" => {
                    cfg.seed = val("--seed")?.parse().map_err(|e| format!("--seed: {e}"))?;
                    i += 2;
                }
                other if other.starts_with("--") => {
                    return Err(format!("unknown flag {other}"));
                }
                other => {
                    cfg.pod5_path = other.to_string();
                    i += 1;
                }
            }
        }
        if cfg.pod5_path.is_empty() {
            return Err("usage: bench_scattered_fetch <pod5> [--ids N] [--chunk N] \
                 [--order spread|random|contiguous] [--seed N]"
                .to_string());
        }
        if cfg.chunk == 0 {
            return Err("--chunk must be > 0".to_string());
        }
        Ok(cfg)
    }
}

/// Pick which reads to ask for.
///
/// `index.uuids()` is UUID-sorted, i.e. already unrelated to file order, so
/// "spread" here means spread across the *index*, which is a uniform sample of
/// the file. That is the point: a realistic id set has no locality.
fn select_targets(all: &[Uuid], cfg: &Config) -> HashSet<Uuid> {
    let n = cfg.n_ids.min(all.len());
    match cfg.order.as_str() {
        "contiguous" => {
            let start = (splitmix(cfg.seed) as usize) % (all.len() - n).max(1);
            all[start..start + n].iter().copied().collect()
        }
        "random" => {
            let mut state = cfg.seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut out = HashSet::with_capacity(n);
            let mut guard = 0usize;
            while out.len() < n && guard < n * 8 {
                state = splitmix(state);
                out.insert(all[(state as usize) % all.len()]);
                guard += 1;
            }
            out
        }
        _ => {
            let step = (all.len() / n).max(1);
            let offset = (splitmix(cfg.seed) as usize) % step;
            all.iter()
                .skip(offset)
                .step_by(step)
                .take(n)
                .copied()
                .collect()
        }
    }
}

/// splitmix64 — a deterministic seed mixer, so runs are reproducible without
/// pulling `rand` into an example.
fn splitmix(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}
