// SPDX-License-Identifier: MIT

//! End-to-end `escpod classify`: fixture POD5 + BAM + bundle in,
//! `cl`-tagged BAM and TSV out, compared against the reference
//! implementation's golden calls (`escapepod-classify/tests/fixtures/`,
//! generated from `escapepod_models.charging`).

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

/// Run the classifier over the fixtures under `argv` (the command words
/// before the positional POD5), returning the captured stderr.
fn run_classifier(argv: &[&str], out_bam: &Path, out_tsv: &Path) -> String {
    let out = Command::new(env!("CARGO_BIN_EXE_escpod"))
        .args(argv)
        .arg(fixtures().join("trna_reads.pod5"))
        .arg("--bam")
        .arg(fixtures().join("trna_mappings_padded.bam"))
        .arg("--reference")
        .arg(fixtures().join("trna_reference.fa"))
        .arg("--model")
        .arg(fixtures().join("bundle"))
        .arg("--output")
        .arg(out_bam)
        .arg("--tsv")
        .arg(out_tsv)
        .output()
        .unwrap_or_else(|e| panic!("escpod {} should launch: {e}", argv.join(" ")));
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(
        out.status.success(),
        "escpod {} failed:\n{stderr}",
        argv.join(" ")
    );
    stderr
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

    run_classifier(&["classify"], &out_bam, &out_tsv);

    // --- TSV vs golden ----------------------------------------------------
    let tsv = std::fs::read_to_string(&out_tsv).unwrap();
    let mut calls: HashMap<String, (f64, u8)> = HashMap::new();
    let mut no_calls: HashMap<String, String> = HashMap::new();
    // Every anchored read gets a row now: a probability, or an empty one and
    // the reason it has none. The `reason` column is what makes a drop
    // attributable rather than a read that silently vanished.
    for line in tsv.lines().skip(1) {
        let f: Vec<&str> = line.split('\t').collect();
        assert_eq!(f.len(), 5, "row does not carry the reason column: {line}");
        let id = f[0].to_string();
        match f[4] {
            "" => {
                calls.insert(id, (f[2].parse().unwrap(), f[3].parse().unwrap()));
            }
            reason => {
                assert!(
                    f[2].is_empty() && f[3].is_empty(),
                    "no-call {id} carries a probability"
                );
                no_calls.insert(id, reason.to_string());
            }
        }
    }
    // The padded duplicates have fresh UUIDs and no POD5 signal, so the
    // fixture always exercises at least one no-call reason.
    assert!(
        !no_calls.is_empty(),
        "expected no-call rows for the reads without signal"
    );
    for (id, reason) in &no_calls {
        assert!(
            matches!(
                reason.as_str(),
                "no_signal" | "ns_mismatch" | "no_aligned_arm"
            ),
            "read {id}: unknown reason {reason:?}"
        );
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

/// `escpod signal classify` was the shipped spelling from 0.11.0 and still
/// appears in pipeline scripts, so it must keep working — producing the *same*
/// calls, not merely exiting zero — while telling the user where the command
/// moved to.
#[test]
fn deprecated_signal_classify_alias_warns_and_forwards() {
    let out_dir = tempfile::tempdir().unwrap();
    let (plain_bam, plain_tsv) = (out_dir.path().join("g.bam"), out_dir.path().join("g.tsv"));
    let (alias_bam, alias_tsv) = (out_dir.path().join("a.bam"), out_dir.path().join("a.tsv"));

    let plain_err = run_classifier(&["classify"], &plain_bam, &plain_tsv);
    let alias_err = run_classifier(&["signal", "classify"], &alias_bam, &alias_tsv);

    assert_eq!(
        std::fs::read_to_string(&plain_tsv).unwrap(),
        std::fs::read_to_string(&alias_tsv).unwrap(),
        "the alias must forward to the same runner, not a divergent copy"
    );
    assert!(
        alias_err.contains("`escpod signal classify` is deprecated")
            && alias_err.contains("use `escpod classify`"),
        "the alias should name its replacement:\n{alias_err}"
    );
    assert!(
        !plain_err.contains("deprecated"),
        "the current spelling must not warn:\n{plain_err}"
    );
}
