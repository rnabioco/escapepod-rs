//! Where `scan_bam`'s time goes: BGZF+parse, versus the per-record anchoring.
//!
//! Run against a real BAM before optimising either half.
//!
//! ```text
//! cargo run --release --example scan_profile -- <bam> <ref.fa>
//! ```

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use escapepod_classify::{ScanOutcome, junction_positions, scan_bam};
use noodles_bam as bam;
use noodles_bgzf as bgzf;
use noodles_sam::alignment::RecordBuf;

const MOTIF: &str = "CCAGGC";
const COMMON_ARM: &str = "GGCTTCTTCTTGCTCTT";
const OFFSETS: [i32; 25] = [
    -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
];

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let bam_path = args.next().expect("usage: scan_profile <bam> <ref.fa>");
    let ref_fa = args.next().expect("usage: scan_profile <bam> <ref.fa>");
    let geometry = junction_positions(Path::new(&ref_fa), MOTIF, 3, COMMON_ARM)?;

    // (a) decode only: BGZF + RecordBuf materialisation, nothing else.
    let t = Instant::now();
    let mut n = 0u64;
    {
        let file = std::fs::File::open(&bam_path)?;
        let mut reader = bam::io::Reader::from(bgzf::io::MultithreadedReader::new(file));
        let header = reader.read_header()?;
        let mut rec = RecordBuf::default();
        while reader.read_record_buf(&header, &mut rec)? != 0 {
            n += 1;
        }
    }
    let t_decode = t.elapsed();

    // (b) decode + the reference-name lookup + scan_record.
    let t = Instant::now();
    let mut anchored = 0u64;
    {
        let file = std::fs::File::open(&bam_path)?;
        let mut reader = bam::io::Reader::from(bgzf::io::MultithreadedReader::new(file));
        let header = reader.read_header()?;
        let names: Vec<String> = header
            .reference_sequences()
            .keys()
            .map(|k| k.to_string())
            .collect();
        let mut rec = RecordBuf::default();
        while reader.read_record_buf(&header, &mut rec)? != 0 {
            let Some(name) = rec.reference_sequence_id().and_then(|i| names.get(i)) else {
                continue;
            };
            if let ScanOutcome::Anchored(_) =
                escapepod_classify::scan_record(&rec, name, &geometry, &OFFSETS, 1)
            {
                anchored += 1;
            }
        }
    }
    let t_scan = t.elapsed();

    // (c) the real thing, including dedup and the vote.
    let t = Instant::now();
    let scan = scan_bam(Path::new(&bam_path), &geometry, &OFFSETS, 1)?;
    let t_full = t.elapsed();

    let per = |d: std::time::Duration| d.as_secs_f64() * 1e6 / n as f64;
    println!("records                {n}");
    println!(
        "  (a) decode only      {:7.1}s  {:6.2} us/rec",
        t_decode.as_secs_f64(),
        per(t_decode)
    );
    println!(
        "  (b) + scan_record    {:7.1}s  {:6.2} us/rec   (+{:.2} us anchoring, {anchored} anchored)",
        t_scan.as_secs_f64(),
        per(t_scan),
        per(t_scan) - per(t_decode)
    );
    println!(
        "  (c) full scan_bam    {:7.1}s  {:6.2} us/rec   (+{:.2} us dedup/vote, {} kept)",
        t_full.as_secs_f64(),
        per(t_full),
        per(t_full) - per(t_scan),
        scan.anchored.len()
    );
    let _: &HashMap<_, _> = &geometry;
    Ok(())
}
