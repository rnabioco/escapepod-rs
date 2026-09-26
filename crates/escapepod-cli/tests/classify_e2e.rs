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
use noodles_sam::header::record::value::map::program::tag as pg_tag;
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

/// The `@PG` `DS` field carries the model identity that would otherwise only
/// ever reach stdout (rnabioco/escapepod-rs#370) — checked against the
/// fixture bundle's own `metadata.json` rather than merely asserted
/// non-empty, since a field that round-trips the wrong bundle is exactly the
/// failure this exists to catch. Also pins `device` (rnabioco/escapepod-rs#425)
/// and that `@PG` `CL` is the real invoked argv rather than the old
/// hand-templated string (rnabioco/escapepod-rs#409's precedent).
#[test]
fn classify_bam_header_records_full_model_provenance() {
    let out_dir = tempfile::tempdir().unwrap();
    let out_bam = out_dir.path().join("out.bam");
    let out_tsv = out_dir.path().join("calls.tsv");
    run_classifier(&["classify"], &out_bam, &out_tsv);

    let meta: Json = serde_json::from_str(
        &std::fs::read_to_string(fixtures().join("bundle/metadata.json")).unwrap(),
    )
    .unwrap();

    let file = std::fs::File::open(&out_bam).unwrap();
    let mut reader = bam::io::Reader::new(file);
    let header = reader.read_header().unwrap();

    let (_, pg) = header
        .programs()
        .as_ref()
        .iter()
        .find(|(id, _)| String::from_utf8_lossy(id.as_ref()) == "escpod-classify")
        .expect("escpod-classify @PG record present");
    let cl = pg
        .other_fields()
        .get(&pg_tag::COMMAND_LINE)
        .expect("@PG CL field present");
    let cl = String::from_utf8_lossy(cl.as_ref());
    assert!(
        cl.contains("classify") && cl.contains(&fixtures().join("bundle").display().to_string()),
        "@PG CL should be the real invoked argv, not a hand-templated string: {cl}"
    );
    let ds = pg
        .other_fields()
        .get(&pg_tag::DESCRIPTION)
        .expect("@PG DS field present");
    let ds: Json = serde_json::from_slice(ds.as_ref()).expect("DS field is valid JSON");

    // model_id / model_version / scorer_sha256 round-trip the bundle's own
    // metadata.json rather than some other value that merely parses as JSON.
    assert_eq!(ds["model_id"], meta["model"]["id"]);
    assert_eq!(ds["model_version"], meta["model"]["version"]);
    assert_eq!(ds["scorer_sha256"], meta["gbm"]["sha256"]);
    assert_eq!(
        ds["operating_point"]["probability"],
        meta["operating_point"]["probability"]
    );
    assert_eq!(ds["operating_point"]["cl"], meta["operating_point"]["cl"]);
    // The class `cl`/`operating_point.probability` are stated on — dropped
    // silently by an earlier draft of #425 along with the old hand-templated
    // `@PG CL` string that used to be the only place it was recorded.
    assert_eq!(ds["positive_class"], meta["classes"][1]);
    // The fixture bundle ships neither a calibration nor a basecaller nor an
    // abstain block — `calibration` must still report `false` (it is not
    // Option on the bundle), but the other two must be omitted rather than
    // fabricated.
    assert_eq!(ds["calibration"], Json::Bool(false));
    assert!(
        ds.get("basecaller").is_none(),
        "fixture bundle carries no basecaller; DS must not invent one: {ds}"
    );
    assert!(
        ds.get("abstain_rule").is_none(),
        "fixture bundle carries no abstain rule; DS must not invent one: {ds}"
    );
    // The fixture bundle is `gbm` — no GPU path at all (`note_cpu_only`), so
    // `device.requested` reports the device that actually ran the classifier
    // (always CPU here) regardless of what `--device` this test happened to
    // pass, and none of the GPU-only fields are fabricated to fill a slot.
    assert_eq!(ds["device"]["requested"], "cpu");
    for key in [
        "cublas_path",
        "cublas_version",
        "cublaslt_path",
        "cublaslt_version",
        "cublas_repaired",
        "gpu_batches_scored",
        "parity_checked_batches",
        "parity_worst_abs_dp",
    ] {
        assert!(
            ds["device"].get(key).is_none(),
            "device.{key} must be omitted under CPU, not present: {ds}"
        );
    }
}

/// `--device cpu` explicitly, rather than relying on the default: the exact
/// case rnabioco/escapepod-rs#425's acceptance criteria names.
#[test]
fn classify_device_cpu_flag_records_requested_cpu() {
    let out_dir = tempfile::tempdir().unwrap();
    let out_bam = out_dir.path().join("out.bam");
    let out_tsv = out_dir.path().join("calls.tsv");
    run_classifier(&["classify", "--device", "cpu"], &out_bam, &out_tsv);

    let file = std::fs::File::open(&out_bam).unwrap();
    let mut reader = bam::io::Reader::new(file);
    let header = reader.read_header().unwrap();
    let (_, pg) = header
        .programs()
        .as_ref()
        .iter()
        .find(|(id, _)| String::from_utf8_lossy(id.as_ref()) == "escpod-classify")
        .expect("escpod-classify @PG record present");
    let ds = pg
        .other_fields()
        .get(&pg_tag::DESCRIPTION)
        .expect("@PG DS field present");
    let ds: Json = serde_json::from_slice(ds.as_ref()).expect("DS field is valid JSON");
    assert_eq!(ds["device"]["requested"], "cpu");
    assert!(ds["device"].get("cublas_version").is_none());
    assert!(ds["device"].get("parity_checked_batches").is_none());
}

/// `escpod classify` now shares `escpod align`'s reference FASTA reader
/// (rnabioco/escapepod-rs#410), so it accepts a gzipped reference exactly as
/// `align` does — previously `geometry::read_fasta` called `read_to_string`
/// directly and failed outright on a `.gz` file. The gzipped run must score
/// identically to the plain-FASTA run.
#[test]
fn classify_accepts_a_gzipped_reference() {
    use std::io::Write as _;

    let dir = tempfile::tempdir().unwrap();

    let plain_bam = dir.path().join("plain.bam");
    let plain_tsv = dir.path().join("plain.tsv");
    run_classifier(&["classify"], &plain_bam, &plain_tsv);

    let reference_gz = dir.path().join("trna_reference.fa.gz");
    let plain_fasta = std::fs::read(fixtures().join("trna_reference.fa")).unwrap();
    let mut enc = flate2::write::GzEncoder::new(
        std::fs::File::create(&reference_gz).unwrap(),
        flate2::Compression::default(),
    );
    enc.write_all(&plain_fasta).unwrap();
    enc.finish().unwrap();

    let gz_bam = dir.path().join("gz.bam");
    let gz_tsv = dir.path().join("gz.tsv");
    let out = Command::new(env!("CARGO_BIN_EXE_escpod"))
        .arg("classify")
        .arg(fixtures().join("trna_reads.pod5"))
        .arg("--bam")
        .arg(fixtures().join("trna_mappings_padded.bam"))
        .arg("--reference")
        .arg(&reference_gz)
        .arg("--model")
        .arg(fixtures().join("bundle"))
        .arg("--output")
        .arg(&gz_bam)
        .arg("--tsv")
        .arg(&gz_tsv)
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(
        out.status.success(),
        "escpod classify --reference *.fa.gz failed:\n{stderr}"
    );

    let plain_calls = std::fs::read_to_string(&plain_tsv).unwrap();
    let gz_calls = std::fs::read_to_string(&gz_tsv).unwrap();
    assert_eq!(
        plain_calls, gz_calls,
        "a gzipped reference must score identically to the plain one"
    );
}

/// A duplicate reference name is refused, exactly as `escpod align` refuses
/// one — the same shared reader (rnabioco/escapepod-rs#410). Previously
/// `geometry::read_fasta` built a `HashMap` directly and let the second
/// record silently win.
#[test]
fn classify_refuses_a_duplicate_reference_name() {
    let dir = tempfile::tempdir().unwrap();

    let mut fasta = std::fs::read_to_string(fixtures().join("trna_reference.fa")).unwrap();
    let first_record: String = fasta.lines().take(2).collect::<Vec<_>>().join("\n");
    let first_name = first_record
        .lines()
        .next()
        .unwrap()
        .trim_start_matches('>')
        .to_string();
    fasta.push('\n');
    fasta.push_str(&first_record);
    fasta.push('\n');
    let dup_reference = dir.path().join("dup_reference.fa");
    std::fs::write(&dup_reference, &fasta).unwrap();

    let out_bam = dir.path().join("out.bam");
    let out_tsv = dir.path().join("calls.tsv");
    let out = Command::new(env!("CARGO_BIN_EXE_escpod"))
        .arg("classify")
        .arg(fixtures().join("trna_reads.pod5"))
        .arg("--bam")
        .arg(fixtures().join("trna_mappings_padded.bam"))
        .arg("--reference")
        .arg(&dup_reference)
        .arg("--model")
        .arg(fixtures().join("bundle"))
        .arg("--output")
        .arg(&out_bam)
        .arg("--tsv")
        .arg(&out_tsv)
        .output()
        .unwrap();
    assert!(
        !out.status.success(),
        "escpod classify should refuse a duplicate reference name"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("duplicate") && stderr.contains(&first_name),
        "error should name the duplicate record ({first_name:?}): {stderr}"
    );
}
