// SPDX-License-Identifier: MIT

//! End-to-end `escpod align` over escapepod-classify's fixtures: 60 reads
//! (bwa-aligned, tags carried through — treated here as input, the way a
//! uBAM would be) against the 47-reference tRNA panel they were aligned to.

use noodles_bam as bam;
use noodles_sam as sam;
use sam::alignment::RecordBuf;
use sam::alignment::record::data::field::Tag;
use sam::alignment::record_buf::data::field::Value;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

const OWNED: [[u8; 2]; 5] = [*b"NM", *b"MD", *b"AS", *b"XS", *b"XA"];

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("escapepod-classify/tests/fixtures")
}

fn input_bam() -> PathBuf {
    fixtures().join("trna_mappings_padded.bam")
}

/// Run `escpod align <reads> -r <reference> -o <out> <extra…>`, returning stderr.
fn align(reads: &Path, reference: &Path, out: &Path, extra: &[&str]) -> String {
    let o = Command::new(env!("CARGO_BIN_EXE_escpod"))
        .arg("align")
        .arg(reads)
        .arg("--reference")
        .arg(reference)
        .arg("--output")
        .arg(out)
        .args(extra)
        .output()
        .expect("escpod align should launch");
    let stderr = String::from_utf8_lossy(&o.stderr).into_owned();
    assert!(o.status.success(), "escpod align failed:\n{stderr}");
    stderr
}

fn read_bam(path: &Path) -> (sam::Header, Vec<RecordBuf>) {
    let mut r = bam::io::Reader::new(std::fs::File::open(path).unwrap());
    let header = r.read_header().unwrap();
    let recs = r
        .record_bufs(&header)
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    (header, recs)
}

fn name(r: &RecordBuf) -> String {
    r.name().unwrap().to_string()
}

fn ref_name(h: &sam::Header, r: &RecordBuf) -> Option<String> {
    r.reference_sequence_id()
        .map(|i| h.reference_sequences().get_index(i).unwrap().0.to_string())
}

fn cigar_string(r: &RecordBuf) -> String {
    use sam::alignment::record::cigar::op::Kind;
    r.cigar()
        .as_ref()
        .iter()
        .map(|op| {
            let c = match op.kind() {
                Kind::Match => 'M',
                Kind::Insertion => 'I',
                Kind::Deletion => 'D',
                Kind::SoftClip => 'S',
                Kind::HardClip => 'H',
                Kind::Skip => 'N',
                Kind::Pad => 'P',
                Kind::SequenceMatch => '=',
                Kind::SequenceMismatch => 'X',
            };
            format!("{}{c}", op.len())
        })
        .collect()
}

fn string_tag(r: &RecordBuf, tag: [u8; 2]) -> Option<String> {
    match r.data().get(&Tag::from(tag)) {
        Some(Value::String(s)) => Some(s.to_string()),
        _ => None,
    }
}

fn int_tag(r: &RecordBuf, tag: [u8; 2]) -> Option<i64> {
    r.data().get(&Tag::from(tag)).and_then(|v| v.as_int())
}

#[test]
fn align_fixture_reproduces_reference_assignment() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("out.bam");
    let stderr = align(
        &input_bam(),
        &fixtures().join("trna_reference.fa"),
        &out,
        &[],
    );
    assert!(
        stderr.contains("60 reads"),
        "summary line missing:\n{stderr}"
    );

    let (in_h, input) = read_bam(&input_bam());
    let (out_h, output) = read_bam(&out);
    assert_eq!(output.len(), input.len(), "one record per input read");
    // @SQ in reference-file order, and our @PG chained to the input's last.
    assert_eq!(out_h.reference_sequences().len(), 47);
    assert_eq!(
        out_h
            .reference_sequences()
            .get_index(0)
            .unwrap()
            .0
            .to_string(),
        "tRNA-Ala-GGC-1-1"
    );
    assert!(out_h.programs().as_ref().contains_key(&b"escpod"[..]));
    for (i, o) in input.iter().zip(&output) {
        // Input order is preserved record for record.
        assert_eq!(name(i), name(o));
        assert!(!o.flags().is_unmapped(), "{} unmapped", name(o));
        assert_eq!(
            ref_name(&out_h, o),
            ref_name(&in_h, i),
            "{}: primary reference differs from the fixture's",
            name(o)
        );
        assert!(string_tag(o, *b"MD").is_some() && int_tag(o, *b"NM").is_some());
        let mapq = o.mapping_quality().map_or(255, |q| q.get());
        let tied = string_tag(o, *b"XA").is_some();
        assert_eq!(mapq, if tied { 0 } else { 60 }, "{}", name(o));
    }
}

#[test]
fn ubam_tags_survive_alignment() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("out.bam");
    align(
        &input_bam(),
        &fixtures().join("trna_reference.fa"),
        &out,
        &[],
    );
    let (_, input) = read_bam(&input_bam());
    let (_, output) = read_bam(&out);
    let mut checked = 0;
    for (i, o) in input.iter().zip(&output) {
        for (tag, value) in i.data().iter() {
            if OWNED.contains(tag.as_ref()) {
                continue;
            }
            // Value equality includes the BAM type (B:c stays B:c, f stays f).
            assert_eq!(
                o.data().get(&tag),
                Some(value),
                "{}: tag {:?} not carried through",
                name(i),
                tag
            );
            checked += 1;
        }
        for t in [*b"mv", *b"ns", *b"ts", *b"RG", *b"MM", *b"ML"] {
            assert!(i.data().get(&Tag::from(t)).is_some(), "fixture lacks {t:?}");
        }
        assert_eq!(o.sequence(), i.sequence());
        assert_eq!(o.quality_scores(), i.quality_scores());
    }
    assert!(checked > 60 * 10, "too few tags compared ({checked})");
}

#[test]
fn ties_are_reported_in_xa_and_optional_secondaries() {
    // Duplicate the reads' reference under a second name: every read then ties.
    let dir = tempfile::tempdir().unwrap();
    let fa = dir.path().join("dup.fa");
    let text = std::fs::read_to_string(fixtures().join("trna_reference.fa")).unwrap();
    let ala = text
        .split('>')
        .find(|r| r.starts_with("tRNA-Ala-GGC-1-1\n"))
        .unwrap()
        .to_string();
    std::fs::write(&fa, format!("{text}>dup-{ala}")).unwrap();

    let out = dir.path().join("out.bam");
    align(&input_bam(), &fa, &out, &[]);
    let (h, recs) = read_bam(&out);
    assert_eq!(recs.len(), 60);
    let mut xa_entries = 0;
    for r in &recs {
        // The original is first in the panel, so it stays primary; its copy
        // (and any isodecoder that ties on its own) is listed in XA.
        assert_eq!(ref_name(&h, r).as_deref(), Some("tRNA-Ala-GGC-1-1"));
        assert_eq!(r.mapping_quality().map(|q| q.get()), Some(0));
        let xa = string_tag(r, *b"XA").unwrap_or_else(|| panic!("{} has no XA", name(r)));
        let pos = r.alignment_start().unwrap().get();
        let dup = format!(
            "dup-tRNA-Ala-GGC-1-1,+{pos},{},{};",
            cigar_string(r),
            int_tag(r, *b"NM").unwrap()
        );
        assert!(xa.ends_with(&dup), "{}: XA {xa} lacks {dup}", name(r));
        let entries: Vec<&str> = xa.trim_end_matches(';').split(';').collect();
        for e in &entries {
            let f: Vec<&str> = e.split(',').collect();
            assert_eq!(f.len(), 4, "{e}");
            assert!(f[1].starts_with('+'), "{e}");
        }
        xa_entries += entries.len();
    }

    let out2 = dir.path().join("sec.bam");
    align(&input_bam(), &fa, &out2, &["--secondary"]);
    let (h2, recs2) = read_bam(&out2);
    assert_eq!(recs2.len(), 60 + xa_entries, "one secondary per XA entry");
    let mut i = 0;
    while i < recs2.len() {
        let p = &recs2[i];
        assert!(!p.flags().is_secondary(), "{} out of order", name(p));
        let xa = string_tag(p, *b"XA").unwrap();
        let mut j = i + 1;
        while j < recs2.len() && recs2[j].flags().is_secondary() {
            let s = &recs2[j];
            assert_eq!(name(p), name(s));
            let entry = format!(
                "{},+{},{},{};",
                ref_name(&h2, s).unwrap(),
                s.alignment_start().unwrap().get(),
                cigar_string(s),
                int_tag(s, *b"NM").unwrap()
            );
            assert!(xa.contains(&entry), "secondary {entry} not in XA {xa}");
            assert_eq!(int_tag(p, *b"AS"), int_tag(s, *b"AS"));
            assert_eq!(s.mapping_quality().map(|q| q.get()), Some(0));
            assert!(s.sequence().is_empty(), "secondaries carry SEQ as *");
            assert_eq!(
                s.data().get(&Tag::READ_GROUP),
                p.data().get(&Tag::READ_GROUP)
            );
            j += 1;
        }
        assert_eq!(j - i - 1, xa.matches(';').count(), "{}", name(p));
        i = j;
    }

    // --max-ties 0 keeps MAPQ 0 but lists nobody.
    let out3 = dir.path().join("capped.bam");
    align(
        &input_bam(),
        &fa,
        &out3,
        &["--max-ties", "0", "--secondary"],
    );
    let (_, recs3) = read_bam(&out3);
    assert_eq!(recs3.len(), 60);
    assert!(recs3.iter().all(
        |r| string_tag(r, *b"XA").is_none() && r.mapping_quality().map(|q| q.get()) == Some(0)
    ));
}

#[test]
fn unmapped_reads_are_written_not_dropped() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("out.bam");
    align(
        &input_bam(),
        &fixtures().join("trna_reference.fa"),
        &out,
        &["--min-score", "100000"],
    );
    let (_, input) = read_bam(&input_bam());
    let (h, output) = read_bam(&out);
    assert_eq!(output.len(), input.len());
    for (i, o) in input.iter().zip(&output) {
        assert!(o.flags().is_unmapped(), "{}", name(o));
        assert_eq!(ref_name(&h, o), None);
        assert!(o.cigar().as_ref().is_empty());
        for (tag, value) in i.data().iter() {
            if OWNED.contains(tag.as_ref()) {
                assert!(o.data().get(&tag).is_none(), "stale {tag:?} on unmapped");
            } else {
                assert_eq!(o.data().get(&tag), Some(value));
            }
        }
    }
}

#[test]
fn fastq_input_matches_ubam_input() {
    let dir = tempfile::tempdir().unwrap();
    let (_, input) = read_bam(&input_bam());
    let fq = dir.path().join("reads.fq");
    let mut text = String::new();
    for r in &input {
        let seq = String::from_utf8(r.sequence().as_ref().to_vec()).unwrap();
        let qual: String = r
            .quality_scores()
            .as_ref()
            .iter()
            .map(|q| (q + 33) as char)
            .collect();
        text.push_str(&format!("@{} comment\n{seq}\n+\n{qual}\n", name(r)));
    }
    std::fs::write(&fq, text).unwrap();

    let reference = fixtures().join("trna_reference.fa");
    let (a, b) = (dir.path().join("bam.bam"), dir.path().join("fq.bam"));
    align(&input_bam(), &reference, &a, &[]);
    align(&fq, &reference, &b, &[]);
    let (_, from_bam) = read_bam(&a);
    let (_, from_fq) = read_bam(&b);
    assert_eq!(from_bam.len(), from_fq.len());
    for (x, y) in from_bam.iter().zip(&from_fq) {
        assert_eq!(name(x), name(y));
        assert_eq!(x.flags(), y.flags());
        assert_eq!(x.reference_sequence_id(), y.reference_sequence_id());
        assert_eq!(x.alignment_start(), y.alignment_start());
        assert_eq!(x.mapping_quality(), y.mapping_quality());
        assert_eq!(cigar_string(x), cigar_string(y));
        assert_eq!(x.sequence(), y.sequence());
        assert_eq!(x.quality_scores(), y.quality_scores());
        for t in OWNED {
            assert_eq!(
                x.data().get(&Tag::from(t)),
                y.data().get(&Tag::from(t)),
                "{t:?}"
            );
        }
    }
}

/// `escpod classify` over the aligner's output must reproduce the charging
/// golden for every read whose alignment is the fixture's own. Scored with
/// bwa `-x ont2d`'s penalties in this convention (the fixture BAM's @PG), so
/// the CIGARs are comparable; the reads whose CIGAR differs are counted and
/// printed rather than asserted on — a different optimal path is a different
/// feature window, not an aligner error.
#[cfg(feature = "classify")]
#[test]
fn align_then_classify_reproduces_golden_calls() {
    let dir = tempfile::tempdir().unwrap();
    let aligned = dir.path().join("aligned.bam");
    let reference = fixtures().join("trna_reference.fa");
    align(
        &input_bam(),
        &reference,
        &aligned,
        &["--scoring", "1,-1,-2,-1"],
    );

    let (in_h, input) = read_bam(&input_bam());
    let (out_h, output) = read_bam(&aligned);
    let fixture_aln: HashMap<String, (Option<String>, usize, String)> = input
        .iter()
        .map(|r| {
            (
                name(r),
                (
                    ref_name(&in_h, r),
                    r.alignment_start().unwrap().get(),
                    cigar_string(r),
                ),
            )
        })
        .collect();
    let mut differing = Vec::new();
    for r in &output {
        let ours = (
            ref_name(&out_h, r),
            r.alignment_start().map_or(0, |p| p.get()),
            cigar_string(r),
        );
        if fixture_aln[&name(r)] != ours {
            differing.push(name(r));
        }
    }
    println!(
        "align_then_classify: {} of {} reads have a CIGAR (or position) different from the \
         fixture BAM's bwa alignment",
        differing.len(),
        output.len()
    );

    let calls_tsv = dir.path().join("calls.tsv");
    let classified = dir.path().join("classified.bam");
    let o = Command::new(env!("CARGO_BIN_EXE_escpod"))
        .arg("classify")
        .arg(fixtures().join("trna_reads.pod5"))
        .arg("--bam")
        .arg(&aligned)
        .arg("--reference")
        .arg(&reference)
        .arg("--model")
        .arg(fixtures().join("bundle"))
        .arg("--output")
        .arg(&classified)
        .arg("--tsv")
        .arg(&calls_tsv)
        .output()
        .unwrap();
    assert!(
        o.status.success(),
        "classify failed:\n{}",
        String::from_utf8_lossy(&o.stderr)
    );

    let golden: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(fixtures().join("charging_golden.json")).unwrap(),
    )
    .unwrap();
    let calls: HashMap<String, (f64, u8)> = std::fs::read_to_string(&calls_tsv)
        .unwrap()
        .lines()
        .skip(1)
        .filter_map(|l| {
            let f: Vec<&str> = l.split('\t').collect();
            (f[4].is_empty()).then(|| {
                (
                    f[0].to_string(),
                    (f[2].parse().unwrap(), f[3].parse().unwrap()),
                )
            })
        })
        .collect();
    let mut compared = 0;
    for g in golden["reads"].as_array().unwrap() {
        let id = g["read_id"].as_str().unwrap();
        if differing.iter().any(|d| d == id) {
            continue;
        }
        let p_ref = f64::from_bits(g["p_bits"].as_u64().unwrap());
        let (p, cl) = calls
            .get(id)
            .unwrap_or_else(|| panic!("read {id}: identical CIGAR but no call"));
        assert!(
            (p - p_ref).abs() <= 1e-6,
            "read {id}: P {p} vs golden {p_ref}"
        );
        assert_eq!(*cl as u64, g["cl"].as_u64().unwrap(), "read {id}: cl");
        compared += 1;
    }
    println!(
        "align_then_classify: {compared} of {} golden calls compared (the rest are reads whose \
         alignment differs), all reproduced exactly",
        golden["reads"].as_array().unwrap().len()
    );
    assert!(compared > 0, "no read kept the fixture's alignment");
}
