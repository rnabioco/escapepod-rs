// SPDX-License-Identifier: MIT

//! End-to-end (not isolated-kernel) measurement for `waveform::classify_reads_gpu`
//! (#351): real bundle, real POD5, real aligned BAM, real reference — the whole
//! scan -> index -> GPU-batched-score path `escpod classify --device gpu` runs,
//! timed, with a `nvidia-smi dmon` capture running alongside so the GPU's duty
//! cycle over the run is a printed number rather than an eyeballed graph.
//!
//! `examples/tcn_cuda_probe.rs` answered whether the GPU kernel itself is
//! worth it, in isolation, with synthetic input. This is the complementary
//! question: with real prep in front of it, does the *pipeline* keep the GPU
//! fed? A superbatch-serial pipeline (prepare 16,384 reads on CPU, only then
//! call the GPU, repeat) can have a fast kernel and a starved device at the
//! same time — the kernel probe cannot see that, only an end-to-end run can.
//!
//! ```text
//! srun -p gpu -A gpu_rbi -c 16 --gres=gpu:1 -- \
//!   pixi run -e gpu ./target/release/examples/waveform_gpu_pipeline_probe \
//!     <bundle dir> <pod5 dir-or-file> --bam <aligned.bam> --reference <ref.fa> \
//!     [--min-mapq N] [--threads N] [--dmon-log PATH]
//! ```
//!
//! `--threads` defaults to 4 — the production shape this issue was filed
//! against (`--threads 4`, one A30, a CPU-constrained Slurm node) — not
//! `nproc`, so a laptop run reproduces the starved-node case rather than
//! masking it under spare cores.
//!
//! The `nvidia-smi dmon` capture spans exactly the `classify_reads_gpu` call
//! (spawned just before, killed just after) and reports the fraction of
//! 1-second samples with `sm > 0` — a superbatch-serial run shows this well
//! under 100% with a visible run of consecutive zero samples; a pipelined run
//! should not have long zero runs even though full saturation is not
//! expected on a 4-core node (CPU prep is the bottleneck; see the module doc
//! on `waveform::classify_reads_gpu` for why overlap still helps).
//!
//! Bit-identical output is not this probe's job — see
//! `tests/charging_waveform_chunks.rs`-style parity coverage for that; this
//! exists to produce a wall-clock number and a duty cycle, on demand, against
//! whatever bundle and dataset are pointed at it.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use escapepod_classify::{ChargingBundle, Pod5Index, junction_positions};

fn resolve_pod5_inputs(path: &Path) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
    if path.is_file() {
        return Ok(vec![path.to_path_buf()]);
    }
    let mut files: Vec<PathBuf> = std::fs::read_dir(path)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_file() && p.extension().is_some_and(|e| e == "pod5"))
        .collect();
    if files.is_empty() {
        return Err(format!("no *.pod5 files in {}", path.display()).into());
    }
    files.sort();
    Ok(files)
}

/// GPU utilization batch size, matching `commands/classify.rs`'s
/// `waveform_gpu_batch()` convention (same env var, same default).
fn waveform_gpu_batch() -> usize {
    std::env::var("ESCAPEPOD_WAVEFORM_GPU_BATCH")
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|&n: &usize| n > 0)
        .unwrap_or(128)
}

/// Spawn `nvidia-smi dmon` writing SM-utilization samples to `log_path`,
/// running until [`stop_dmon`] kills it. `-s u` limits the columns to
/// utilization metrics (sm, mem, enc, dec, jpg, ofa); we only read `sm`.
fn spawn_dmon(log_path: &Path) -> std::io::Result<Child> {
    let log = std::fs::File::create(log_path)?;
    Command::new("nvidia-smi")
        .args(["dmon", "-s", "u", "-d", "1"])
        .stdout(Stdio::from(log))
        .stderr(Stdio::null())
        .spawn()
}

fn stop_dmon(mut child: Child) {
    let _ = child.kill();
    let _ = child.wait();
}

/// `(samples, samples with sm > 0, mean sm%)` from a dmon log — data rows
/// only (lines starting with `#` are the two header rows).
fn duty_cycle(log_path: &Path) -> (usize, usize, f64) {
    let Ok(text) = std::fs::read_to_string(log_path) else {
        return (0, 0, 0.0);
    };
    let mut n = 0usize;
    let mut nonzero = 0usize;
    let mut sum = 0u64;
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut cols = line.split_whitespace();
        let _gpu = cols.next();
        let Some(sm) = cols.next().and_then(|s| s.parse::<u64>().ok()) else {
            continue;
        };
        n += 1;
        sum += sm;
        if sm > 0 {
            nonzero += 1;
        }
    }
    let mean = if n > 0 { sum as f64 / n as f64 } else { 0.0 };
    (n, nonzero, mean)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let (Some(bundle_dir), Some(pod5_path)) = (args.next(), args.next()) else {
        eprintln!(
            "usage: waveform_gpu_pipeline_probe <bundle dir> <pod5 dir-or-file> \
             --bam <aligned.bam> --reference <ref.fa> [--min-mapq N] [--threads N] \
             [--dmon-log PATH]"
        );
        std::process::exit(2);
    };
    let mut bam: Option<PathBuf> = None;
    let mut reference: Option<PathBuf> = None;
    let mut min_mapq: u8 = 0;
    let mut threads: usize = 4;
    let mut dmon_log = std::env::temp_dir().join("waveform_gpu_pipeline_probe_dmon.log");
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--bam" => bam = Some(PathBuf::from(args.next().expect("--bam needs a value"))),
            "--reference" => {
                reference = Some(PathBuf::from(
                    args.next().expect("--reference needs a value"),
                ))
            }
            "--min-mapq" => {
                min_mapq = args
                    .next()
                    .expect("--min-mapq needs a value")
                    .parse()
                    .expect("--min-mapq must be a u8")
            }
            "--threads" => {
                threads = args
                    .next()
                    .expect("--threads needs a value")
                    .parse()
                    .expect("--threads must be a number")
            }
            "--dmon-log" => {
                dmon_log = PathBuf::from(args.next().expect("--dmon-log needs a value"))
            }
            other => panic!("unknown flag {other}"),
        }
    }
    let bam = bam.expect("--bam is required");
    let reference = reference.expect("--reference is required");

    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .expect("rayon global pool builds exactly once");
    println!("rayon pool: {threads} thread(s) (production shape: --threads 4)");

    let bundle = ChargingBundle::load(Path::new(&bundle_dir))?;
    println!(
        "bundle: {} [{}], GPU batch {}",
        bundle.model_id,
        bundle.scorer.kind(),
        waveform_gpu_batch()
    );

    let geometry = junction_positions(
        &reference,
        &bundle.anchor.motif,
        bundle.anchor.motif_offset,
        &bundle.anchor.common_arm,
    )?;
    println!("junction located in {} reference record(s)", geometry.len());

    let scanning = Instant::now();
    let scan = escapepod_classify::waveform::scan_bam(
        &bam,
        &geometry,
        bundle.anchor.motif_offset,
        min_mapq,
    )?;
    println!(
        "scan: {} BAM record(s), {} anchored read(s) ({:.1}s)",
        scan.records_scanned,
        scan.anchored.len(),
        scanning.elapsed().as_secs_f64()
    );
    if scan.anchored.is_empty() {
        return Err("no reads anchored; nothing to classify".into());
    }

    let pod5_files = resolve_pod5_inputs(Path::new(&pod5_path))?;
    let wanted: HashSet<uuid::Uuid> = scan.anchored.keys().copied().collect();
    let indexing = Instant::now();
    let pod5 = Pod5Index::build(&pod5_files, &wanted)?;
    println!(
        "index: {} of {} anchored read(s) have signal in {} POD5 file(s) ({:.1}s)",
        pod5.reads().len(),
        scan.anchored.len(),
        pod5.n_files(),
        indexing.elapsed().as_secs_f64()
    );

    let loading = Instant::now();
    let gpu = bundle.waveform_net_gpu(waveform_gpu_batch())?;
    println!(
        "GPU scorer loaded and warmed ({:.1}s)",
        loading.elapsed().as_secs_f64()
    );

    // Capture spans exactly the timed call below, not model load or the scan
    // — those are one-time costs this issue does not touch.
    let dmon = spawn_dmon(&dmon_log).ok();
    if dmon.is_none() {
        eprintln!("(nvidia-smi dmon not available; skipping duty-cycle capture)");
    }

    let scoring = Instant::now();
    let (calls, stats) =
        escapepod_classify::waveform::classify_reads_gpu(&bundle, &scan.anchored, &pod5, &gpu)?;
    let elapsed = scoring.elapsed();

    if let Some(child) = dmon {
        // dmon samples once a second; give the last sample a moment to land.
        std::thread::sleep(Duration::from_millis(1200));
        stop_dmon(child);
        let (n, nonzero, mean) = duty_cycle(&dmon_log);
        if n > 0 {
            println!(
                "GPU duty cycle over the scoring call: {}/{} 1s samples with sm>0 ({:.0}%), mean sm {:.1}%",
                nonzero,
                n,
                100.0 * nonzero as f64 / n as f64,
                mean
            );
        }
    }

    let n_calls = calls.len();
    println!(
        "scored: {n_calls} call(s), {} no-call(s) ({} no-signal, {} ns-mismatch, {} no-chunk, {} abstained) in {:.2}s ({:.0} reads/s)",
        stats.no_calls.len(),
        stats.no_signal,
        stats.ns_mismatch,
        stats.no_chunk,
        stats.abstained,
        elapsed.as_secs_f64(),
        n_calls as f64 / elapsed.as_secs_f64().max(1e-9),
    );

    Ok(())
}
