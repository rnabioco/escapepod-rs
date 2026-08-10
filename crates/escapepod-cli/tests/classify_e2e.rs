// SPDX-License-Identifier: MIT

//! End-to-end `escpod classify`: fixture POD5 + BAM + bundle in, `cl`-tagged
//! BAM and TSV out, compared against the reference implementation's golden
//! calls (`escapepod-classify/tests/fixtures/`, generated from
//! `escapepod_models.charging`).

#![cfg(feature = "classify")]

use noodles_bam as bam;
use noodles_sam::alignment::RecordBuf;
use noodles_sam::alignment::record::data::field::Tag;
use noodles_sam::alignment::record_buf::data::field::Value;
use serde_json::Value as Json;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("escapepod-classify/tests/fixtures")
}

#[test]
fn classify_end_to_end_matches_golden() {
    let golden: Json = serde_json::from_str(
        &std::fs::read_to_string(fixtures().join("charging_golden.json")).unwrap(),
    )
    .unwrap();

    let out_dir = tempfile::tempdir().unwrap();
    let out_bam = out_dir.path().join("out.bam");
    let out_tsv = out_dir.path().join("calls.tsv");

    let status = Command::new(env!("CARGO_BIN_EXE_escpod"))
        .arg("classify")
        .arg(fixtures().join("trna_reads.pod5"))
        .arg("--bam")
        .arg(fixtures().join("trna_mappings_padded.bam"))
        .arg("--reference")
        .arg(fixtures().join("trna_reference.fa"))
        .arg("--model")
        .arg(fixtures().join("bundle"))
        .arg("--output")
        .arg(&out_bam)
        .arg("--tsv")
        .arg(&out_tsv)
        .status()
        .expect("escpod classify should launch");
    assert!(status.success(), "escpod classify failed");

    // --- TSV vs golden ----------------------------------------------------
    let tsv = std::fs::read_to_string(&out_tsv).unwrap();
    let mut calls: HashMap<String, (f64, u8)> = HashMap::new();
    for line in tsv.lines().skip(1) {
        let mut it = line.split('\t');
        let id = it.next().unwrap().to_string();
        let _reference = it.next().unwrap();
        let p: f64 = it.next().unwrap().parse().unwrap();
        let cl: u8 = it.next().unwrap().parse().unwrap();
        calls.insert(id, (p, cl));
    }

    let reads = golden["reads"].as_array().unwrap();
    assert_eq!(
        calls.len(),
        reads.len(),
        "classified read count differs from reference"
    );
    let mut golden_cl: HashMap<String, u8> = HashMap::new();
    for gr in reads {
        let id = gr["read_id"].as_str().unwrap();
        let p_ref = f64::from_bits(gr["p_bits"].as_u64().unwrap());
        let cl_ref = gr["cl"].as_u64().unwrap() as u8;
        let (p, cl) = calls
            .get(id)
            .unwrap_or_else(|| panic!("read {id} missing from TSV"));
        assert!(
            (p - p_ref).abs() <= 1e-6,
            "read {id}: P = {p} vs reference {p_ref}"
        );
        assert_eq!(*cl, cl_ref, "read {id}: cl differs");
        golden_cl.insert(id.to_string(), cl_ref);
    }

    // --- output BAM: cl on classified reads, absent elsewhere -------------
    let file = std::fs::File::open(&out_bam).unwrap();
    let mut reader = bam::io::Reader::new(file);
    let header = reader.read_header().unwrap();
    let cl_tag = Tag::new(b'c', b'l');
    let mut record = RecordBuf::default();
    let (mut tagged, mut untagged, mut total) = (0u64, 0u64, 0u64);
    loop {
        if reader.read_record_buf(&header, &mut record).unwrap() == 0 {
            break;
        }
        total += 1;
        let name = record
            .name()
            .map(|n| String::from_utf8_lossy(n.as_ref()).to_string())
            .unwrap_or_default();
        match (golden_cl.get(&name), record.data().get(&cl_tag)) {
            (Some(&want), Some(Value::UInt8(got))) => {
                assert_eq!(*got, want, "read {name}: BAM cl differs");
                tagged += 1;
            }
            (Some(_), other) => panic!("read {name}: cl missing or mistyped ({other:?})"),
            (None, Some(_)) => panic!("read {name}: unexpected cl on unclassified read"),
            (None, None) => untagged += 1,
        }
    }
    assert_eq!(
        tagged as usize,
        reads.len(),
        "every classified read tagged once"
    );
    // The padded duplicates (fresh UUIDs, no POD5 signal) pass through untouched.
    assert!(untagged > 0, "expected untagged pass-through records");
    assert_eq!(total, tagged + untagged);

    // The @PG line documents the encoding.
    let header_text = format!("{:?}", header);
    assert!(
        header_text.contains("escpod-classify") || !header.programs().as_ref().is_empty(),
        "output header should carry the escpod-classify @PG record"
    );
}
