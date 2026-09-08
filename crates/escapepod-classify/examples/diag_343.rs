// SPDX-License-Identifier: MIT

//! Standalone repro/diagnostic for rnabioco/escapepod-rs#343: does
//! `classify_reads_gpu` really disagree with `classify_reads`, and by how
//! much per read?
//!
//! Calls the library functions directly, exactly as `run_waveform` does
//! internally, skipping the CLI's argument parsing and progress reporting.
//! Nothing here is wired into `escpod`; it exists as a quick CPU-vs-GPU
//! regression check — the one #343 needed and any future report like it
//! will too.
//!
//! Defaults to the shipped 19-read fixture (too small to fill a batch of
//! 128 — every read falls through to the CPU remainder path, so it can only
//! exercise small batches). `--pod5`/`--bam`/`--fasta` point at a real,
//! larger dataset to test the batch size production actually uses.
//!
//! ```text
//! srun -p gpu -A gpu_rbi -c 8 --gres=gpu:1 -- \
//!   pixi run -e gpu ./target/release/examples/diag_343 <bundle dir> --batch 128 \
//!     --pod5 <dir or file> --bam <aligned.bam> --fasta <reference.fa>
//! ```

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use escapepod_classify::{ChargingBundle, Pod5Index, junction_positions, waveform};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let Some(bundle_dir) = args.next() else {
        eprintln!(
            "usage: diag_343 <bundle dir> [--batch N] [--pod5 PATH] [--bam PATH] [--fasta PATH]"
        );
        std::process::exit(2);
    };
    let mut batch = 2usize;
    let mut pod5_arg: Option<PathBuf> = None;
    let mut bam_arg: Option<PathBuf> = None;
    let mut fasta_arg: Option<PathBuf> = None;
    let mut min_mapq = 1u8;
    let mut limit: Option<usize> = None;
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--batch" => batch = args.next().expect("--batch needs a value").parse().unwrap(),
            "--pod5" => pod5_arg = Some(PathBuf::from(args.next().expect("--pod5 needs a value"))),
            "--bam" => bam_arg = Some(PathBuf::from(args.next().expect("--bam needs a value"))),
            "--fasta" => {
                fasta_arg = Some(PathBuf::from(args.next().expect("--fasta needs a value")))
            }
            "--min-mapq" => {
                min_mapq = args
                    .next()
                    .expect("--min-mapq needs a value")
                    .parse()
                    .unwrap()
            }
            "--limit" => limit = Some(args.next().expect("--limit needs a value").parse().unwrap()),
            other => panic!("unknown flag {other}"),
        }
    }

    let fixtures = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let pod5_path = pod5_arg.unwrap_or_else(|| fixtures.join("trna_reads.pod5"));
    let bam_path = bam_arg.unwrap_or_else(|| fixtures.join("trna_mappings_padded.bam"));
    let fasta_path = fasta_arg.unwrap_or_else(|| fixtures.join("trna_reference.fa"));

    let bundle = ChargingBundle::load(&PathBuf::from(&bundle_dir))?;
    println!("bundle: {bundle_dir} ({})", bundle.model_id);
    println!("pod5:   {}", pod5_path.display());
    println!("bam:    {}", bam_path.display());
    println!("fasta:  {}", fasta_path.display());

    let geometry = junction_positions(
        &fasta_path,
        &bundle.anchor.motif,
        bundle.anchor.motif_offset,
        &bundle.anchor.common_arm,
    )?;

    let mut scan = waveform::scan_bam(&bam_path, &geometry, bundle.anchor.motif_offset, min_mapq)?;
    println!(
        "scan: {} records, {} anchored reads",
        scan.records_scanned,
        scan.anchored.len()
    );

    if let Some(n) = limit {
        // Deterministic truncation (sorted by id) rather than arbitrary
        // HashMap order, so a re-run with the same --limit compares the same
        // reads. A diagnostic has no business processing an entire
        // production run when a few thousand reads exercises every batch
        // shape `--batch` asks for just as well.
        let mut ids: Vec<uuid::Uuid> = scan.anchored.keys().copied().collect();
        ids.sort();
        ids.truncate(n);
        let keep: HashSet<uuid::Uuid> = ids.into_iter().collect();
        scan.anchored.retain(|id, _| keep.contains(id));
        println!("limit: truncated to {} anchored reads", scan.anchored.len());
    }

    let pod5_files = resolve_pod5(&pod5_path)?;
    println!("pod5 files: {}", pod5_files.len());
    let wanted: HashSet<uuid::Uuid> = scan.anchored.keys().copied().collect();
    let pod5 = Pod5Index::build(&pod5_files, &wanted)?;
    println!(
        "pod5: {} of {} anchored reads have signal",
        pod5.reads().len(),
        scan.anchored.len()
    );

    let (cpu_calls, cpu_stats) = waveform::classify_reads(&bundle, &scan.anchored, &pod5)?;
    println!(
        "cpu:  {} calls, {} no-calls",
        cpu_calls.len(),
        cpu_stats.no_calls.len()
    );

    let gpu = bundle.waveform_net_gpu(batch)?;
    let (gpu_calls, gpu_stats) =
        waveform::classify_reads_gpu(&bundle, &scan.anchored, &pod5, &gpu)?;
    println!(
        "gpu:  {} calls, {} no-calls (batch {batch})",
        gpu_calls.len(),
        gpu_stats.no_calls.len()
    );

    let op_point = bundle
        .operating_point
        .as_ref()
        .map(|o| o.probability)
        .unwrap_or(0.5);

    let mut cpu_by_id: std::collections::HashMap<uuid::Uuid, f64> =
        cpu_calls.iter().map(|c| (c.read_id, c.p)).collect();
    let mut max_delta = 0.0f64;
    let mut n_flipped = 0usize;
    let mut n_shown = 0usize;
    for g in &gpu_calls {
        let Some(cpu_p) = cpu_by_id.remove(&g.read_id) else {
            println!("{}  GPU CALLED, NO CPU CALL", g.read_id);
            continue;
        };
        let delta = (cpu_p - g.p).abs();
        max_delta = max_delta.max(delta);
        let cpu_charged = cpu_p >= op_point;
        let gpu_charged = g.p >= op_point;
        if cpu_charged != gpu_charged {
            n_flipped += 1;
            println!(
                "{}  {:>10.6}  {:>10.6}  {:>10.3e}  <-- FLIPPED CALL",
                g.read_id, cpu_p, g.p, delta
            );
        } else if delta > 1e-3 && n_shown < 50 {
            n_shown += 1;
            println!(
                "{}  {:>10.6}  {:>10.6}  {:>10.3e}  <-- LOOK",
                g.read_id, cpu_p, g.p, delta
            );
        }
    }
    for (id, cpu_p) in cpu_by_id {
        println!("{id}  CPU CALLED, NO GPU CALL (cpu_p={cpu_p:.6})");
    }
    println!(
        "\nsummary: {} reads compared, max |delta p| = {max_delta:.3e}, {n_flipped} flipped \
         call(s) at operating point {op_point:.3}",
        gpu_calls.len(),
    );
    Ok(())
}

/// A single `.pod5`, or every `.pod5` directly inside a directory
/// (non-recursive — matches how a run's own `pod5/` folder is laid out).
fn resolve_pod5(path: &Path) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
    if path.is_dir() {
        let mut files: Vec<PathBuf> = std::fs::read_dir(path)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "pod5"))
            .collect();
        files.sort();
        if files.is_empty() {
            return Err(format!("no .pod5 files in {}", path.display()).into());
        }
        Ok(files)
    } else {
        Ok(vec![path.to_path_buf()])
    }
}
