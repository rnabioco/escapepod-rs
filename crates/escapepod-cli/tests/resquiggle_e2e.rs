// SPDX-License-Identifier: MIT

//! `escpod resquiggle`'s `@PG` `CL` field must record the real argv (as
//! `align.rs` does), not a hand-reconstructed subset of flags that omitted
//! both `--kmer-table` and `--kmer-model` (rnabioco/escapepod-rs#409).

#![cfg(feature = "experimental")]

use noodles_bam as bam;
use noodles_sam::header::record::value::map::program::tag as pg_tag;
use std::path::{Path, PathBuf};
use std::process::Command;

fn classify_fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("escapepod-classify/tests/fixtures")
}

fn kmer_table() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("data/kmer_models/rna004_9mer_levels_v1.txt.gz")
}

#[test]
fn resquiggle_pg_cl_records_the_kmer_table_flag() {
    let kmer_table = kmer_table();
    if !kmer_table.exists() {
        eprintln!("skipping test: {:?} not found", kmer_table);
        return;
    }

    let dir = tempfile::tempdir().unwrap();
    let out_bam = dir.path().join("out.bam");

    let o = Command::new(env!("CARGO_BIN_EXE_escpod"))
        .arg("resquiggle")
        .arg(classify_fixtures().join("trna_reads.pod5"))
        .arg("--bam")
        .arg(classify_fixtures().join("trna_mappings_padded.bam"))
        .arg("--kmer-table")
        .arg(&kmer_table)
        .arg("--output")
        .arg(&out_bam)
        .output()
        .expect("escpod resquiggle should launch");
    let stderr = String::from_utf8_lossy(&o.stderr).into_owned();
    assert!(o.status.success(), "escpod resquiggle failed:\n{stderr}");

    let mut reader = bam::io::Reader::new(std::fs::File::open(&out_bam).unwrap());
    let header = reader.read_header().unwrap();
    let (_, pg) = header
        .programs()
        .as_ref()
        .iter()
        .find(|(id, _)| String::from_utf8_lossy(id.as_ref()) == "escpod-resquiggle")
        .expect("escpod-resquiggle @PG record present");
    let cl = pg
        .other_fields()
        .get(&pg_tag::COMMAND_LINE)
        .expect("@PG CL field present");
    let cl = String::from_utf8_lossy(cl.as_ref()).into_owned();

    assert!(
        cl.contains("--kmer-table"),
        "CL should record the --kmer-table flag that was passed: {cl}"
    );
    assert!(
        cl.contains(kmer_table.to_str().unwrap()),
        "CL should record the kmer-table path that was passed: {cl}"
    );
}
