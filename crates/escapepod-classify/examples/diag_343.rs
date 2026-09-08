// SPDX-License-Identifier: MIT

//! Standalone repro/diagnostic for rnabioco/escapepod-rs#343: does
//! `classify_reads_gpu` really disagree with `classify_reads` on the shipped
//! fixture, and by how much per read?
//!
//! `commands/classify.rs` refuses `--device gpu` for a `waveform_model`
//! bundle outright (`place_ruled_out`), so the CLI itself cannot reproduce
//! this any more — this example calls the library functions directly,
//! exactly as `run_waveform` does internally, without going through that
//! gate. Nothing here is wired into `escpod`; it exists to keep #343's own
//! repro re-runnable now that the CLI path is closed.
//!
//! ```text
//! srun -p gpu -A gpu_rbi -c 8 --gres=gpu:1 -- \
//!   pixi run -e gpu ./target/release/examples/diag_343 <bundle dir> --batch 2
//! ```

use std::collections::HashSet;
use std::path::PathBuf;

use escapepod_classify::{ChargingBundle, Pod5Index, junction_positions, waveform};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let Some(bundle_dir) = args.next() else {
        eprintln!("usage: diag_343 <bundle dir> [--batch N]");
        std::process::exit(2);
    };
    let mut batch = 2usize;
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--batch" => batch = args.next().expect("--batch needs a value").parse().unwrap(),
            other => panic!("unknown flag {other}"),
        }
    }

    let fixtures = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let pod5_path = fixtures.join("trna_reads.pod5");
    let bam_path = fixtures.join("trna_mappings_padded.bam");
    let fasta_path = fixtures.join("trna_reference.fa");

    let bundle = ChargingBundle::load(&PathBuf::from(&bundle_dir))?;
    println!("bundle: {bundle_dir} ({})", bundle.model_id);

    let geometry = junction_positions(
        &fasta_path,
        &bundle.anchor.motif,
        bundle.anchor.motif_offset,
        &bundle.anchor.common_arm,
    )?;

    let scan = waveform::scan_bam(&bam_path, &geometry, bundle.anchor.motif_offset, 1)?;
    println!(
        "scan: {} records, {} anchored reads",
        scan.records_scanned,
        scan.anchored.len()
    );

    let wanted: HashSet<uuid::Uuid> = scan.anchored.keys().copied().collect();
    let pod5 = Pod5Index::build(&[pod5_path], &wanted)?;
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
    println!(
        "\n{:<36}  {:>10}  {:>10}  {:>10}",
        "read_id", "cpu_p", "gpu_p", "|delta|"
    );
    let mut max_delta = 0.0f64;
    let mut n_flipped = 0usize;
    for g in &gpu_calls {
        let Some(cpu_p) = cpu_by_id.remove(&g.read_id) else {
            println!("{}  GPU CALLED, NO CPU CALL", g.read_id);
            continue;
        };
        let delta = (cpu_p - g.p).abs();
        max_delta = max_delta.max(delta);
        let cpu_charged = cpu_p >= op_point;
        let gpu_charged = g.p >= op_point;
        let flag = if cpu_charged != gpu_charged {
            n_flipped += 1;
            "  <-- FLIPPED CALL"
        } else if delta > 1e-3 {
            "  <-- LOOK"
        } else {
            ""
        };
        println!(
            "{}  {:>10.6}  {:>10.6}  {:>10.3e}{flag}",
            g.read_id, cpu_p, g.p, delta
        );
    }
    for (id, cpu_p) in cpu_by_id {
        println!("{id}  CPU CALLED, NO GPU CALL (cpu_p={cpu_p:.6})");
    }
    println!(
        "\nsummary: max |delta p| = {max_delta:.3e}, {n_flipped} flipped call(s) at operating point {op_point:.3}",
    );
    Ok(())
}
