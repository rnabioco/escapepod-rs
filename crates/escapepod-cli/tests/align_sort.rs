// SPDX-License-Identifier: MIT

//! `escpod align --sort coordinate` (#417): the sorted output must be the
//! records of the unsorted output in exactly `samtools sort`'s order — `tid`,
//! then `pos`, then the reverse flag, then input order, unmapped last — with
//! the header unchanged apart from `SO` and the `@PG CL`.
//!
//! Every comparison is made twice: against an in-test model of that order
//! (always), and against `samtools sort` itself when a `samtools` is on `PATH`
//! or named by `$SAMTOOLS` (it is in the pixi environment; the CI runners
//! have none, and the samtools half then says it skipped).

use noodles_bam as bam;
use noodles_sam as sam;
use sam::alignment::RecordBuf;
use sam::alignment::record::data::field::Tag;
use std::path::{Path, PathBuf};
use std::process::Command;

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("escapepod-classify/tests/fixtures")
}

fn input_bam() -> PathBuf {
    fixtures().join("trna_mappings_padded.bam")
}

fn reference() -> PathBuf {
    fixtures().join("trna_reference.fa")
}

/// Run `escpod align`, returning stderr.
fn align(reads: &Path, reference: &Path, out: &Path, extra: &[&str]) -> String {
    let o = Command::new(env!("CARGO_BIN_EXE_escpod"))
        .arg("align")
        .arg(reads)
        .arg("-r")
        .arg(reference)
        .arg("-o")
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
    r.name().map(|n| n.to_string()).unwrap_or_default()
}

/// The header as SAM text, with the `@HD SO` and our `@PG`'s `CL` removed —
/// the only two things `--sort coordinate` may change.
fn header_text(h: &sam::Header) -> Vec<String> {
    let mut w = sam::io::Writer::new(Vec::new());
    w.write_header(h).unwrap();
    String::from_utf8(w.into_inner())
        .unwrap()
        .lines()
        .map(|l| {
            l.split('\t')
                .filter(|f| {
                    !f.starts_with("SO:")
                        && !(l.starts_with("@PG\tID:escpod-align") && f.starts_with("CL:"))
                })
                .collect::<Vec<_>>()
                .join("\t")
        })
        .collect()
}

fn sort_order(h: &sam::Header) -> String {
    let hd = h.header().expect("@HD");
    let so = hd
        .other_fields()
        .get(&sam::header::record::value::map::header::tag::SORT_ORDER)
        .expect("SO");
    so.to_string()
}

/// `samtools sort`'s coordinate order, stated independently of the code
/// under test: a stable sort (so input order breaks ties) on `(tid, pos,
/// reverse)` with no reference (`tid -1`) last.
fn model_sort(recs: &[RecordBuf]) -> Vec<RecordBuf> {
    let mut v = recs.to_vec();
    v.sort_by_key(|r| {
        (
            r.reference_sequence_id().unwrap_or(usize::MAX),
            r.alignment_start().map_or(0, |p| p.get()),
            r.flags().is_reverse_complemented(),
        )
    });
    v
}

// SAMTOOLS and PATH are third-party variables, not ESCAPEPOD_* knobs.
#[allow(clippy::disallowed_methods)]
fn samtools() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("SAMTOOLS") {
        return Some(PathBuf::from(p));
    }
    std::env::split_paths(&std::env::var_os("PATH")?)
        .map(|d| d.join("samtools"))
        .find(|p| p.is_file())
}

/// `samtools sort --no-PG` of `unsorted`, or `None` (said) without samtools.
fn samtools_sort(unsorted: &Path, dir: &Path, test: &str) -> Option<PathBuf> {
    let Some(st) = samtools() else {
        eprintln!("[align_sort] {test}: no samtools on PATH or $SAMTOOLS; model order only");
        return None;
    };
    let out = dir.join(format!(
        "{}.samtools.bam",
        unsorted.file_stem().unwrap().to_string_lossy()
    ));
    let o = Command::new(&st)
        .args(["sort", "--no-PG", "-o"])
        .arg(&out)
        .arg(unsorted)
        .output()
        .expect("samtools should launch");
    assert!(
        o.status.success(),
        "samtools sort failed:\n{}",
        String::from_utf8_lossy(&o.stderr)
    );
    Some(out)
}

/// Align `reads` twice, unsorted and sorted, and hold the sorted output to
/// both orders and the header rule. Returns the sorted records.
fn check_sorted(
    dir: &Path,
    tag: &str,
    reads: &Path,
    reference: &Path,
    extra: &[&str],
) -> Vec<RecordBuf> {
    let unsorted = dir.join(format!("{tag}.unsorted.bam"));
    let sorted = dir.join(format!("{tag}.sorted.bam"));
    align(reads, reference, &unsorted, extra);
    let mut sorted_args = extra.to_vec();
    sorted_args.extend(["--sort", "coordinate"]);
    align(reads, reference, &sorted, &sorted_args);

    let (uh, urecs) = read_bam(&unsorted);
    let (sh, srecs) = read_bam(&sorted);
    assert_eq!(sort_order(&uh), "unsorted");
    assert_eq!(sort_order(&sh), "coordinate");
    assert_eq!(header_text(&uh), header_text(&sh), "{tag}: header changed");
    assert_eq!(srecs.len(), urecs.len(), "{tag}: record count");
    let want = model_sort(&urecs);
    for (i, (a, b)) in srecs.iter().zip(&want).enumerate() {
        assert_eq!(a, b, "{tag}: record {i} ({}) out of model order", name(a));
    }

    if let Some(st) = samtools_sort(&unsorted, dir, tag) {
        let (th, trecs) = read_bam(&st);
        assert_eq!(sort_order(&th), "coordinate");
        assert_eq!(
            header_text(&th),
            header_text(&sh),
            "{tag}: header vs samtools"
        );
        assert_eq!(trecs.len(), srecs.len());
        for (i, (a, b)) in srecs.iter().zip(&trecs).enumerate() {
            assert_eq!(
                a,
                b,
                "{tag}: record {i} ({}) differs from samtools sort",
                name(a)
            );
        }
    }
    srecs
}

#[test]
fn sort_coordinate_matches_samtools_sort() {
    let dir = tempfile::tempdir().unwrap();
    let recs = check_sorted(dir.path(), "default", &input_bam(), &reference(), &[]);
    assert_eq!(recs.len(), 60);
    // Small batches over several threads: chunks finish out of order, and
    // the sort's input-order tie-break must still be the input's.
    check_sorted(
        dir.path(),
        "chunked",
        &input_bam(),
        &reference(),
        &["--scoring", "1,-1,-2,-1", "--batch-size", "7", "-t", "4"],
    );

    // `-o -`: output starts once input ends, and is the same file.
    let o = Command::new(env!("CARGO_BIN_EXE_escpod"))
        .arg("align")
        .arg(input_bam())
        .arg("-r")
        .arg(reference())
        .args(["-o", "-", "--sort", "coordinate"])
        .output()
        .unwrap();
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let mut r = bam::io::Reader::new(std::io::Cursor::new(o.stdout));
    let h = r.read_header().unwrap();
    let piped: Vec<RecordBuf> = r.record_bufs(&h).collect::<Result<_, _>>().unwrap();
    assert_eq!(piped, recs, "-o - differs from -o FILE");
}

#[test]
fn sort_coordinate_puts_unmapped_last() {
    let dir = tempfile::tempdir().unwrap();
    // Pick a --min-score at the median AS, so about half the reads fall below.
    let probe = dir.path().join("probe.bam");
    align(&input_bam(), &reference(), &probe, &[]);
    let (_, recs) = read_bam(&probe);
    let mut scores: Vec<i64> = recs
        .iter()
        .filter_map(|r| r.data().get(&Tag::ALIGNMENT_SCORE).and_then(|v| v.as_int()))
        .collect();
    scores.sort_unstable();
    let min_score = (scores[scores.len() / 2] + 1).to_string();

    let sorted = check_sorted(
        dir.path(),
        "min_score",
        &input_bam(),
        &reference(),
        &["--min-score", &min_score],
    );
    let n_unmapped = sorted.iter().filter(|r| r.flags().is_unmapped()).count();
    assert!(
        n_unmapped > 0 && n_unmapped < sorted.len(),
        "--min-score {min_score} left {n_unmapped} of {} unmapped",
        sorted.len()
    );
    let first_unmapped = sorted.iter().position(|r| r.flags().is_unmapped()).unwrap();
    assert!(
        sorted[first_unmapped..]
            .iter()
            .all(|r| r.flags().is_unmapped()),
        "a mapped record after an unmapped one"
    );
    // ...and the unmapped ones keep input order.
    let (_, input) = read_bam(&input_bam());
    let unmapped: Vec<String> = sorted[first_unmapped..].iter().map(name).collect();
    let in_order: Vec<String> = input
        .iter()
        .map(name)
        .filter(|n| unmapped.contains(n))
        .collect();
    assert_eq!(unmapped, in_order);
}

fn revcomp(s: &[u8]) -> Vec<u8> {
    s.iter()
        .rev()
        .map(|b| match b {
            b'A' => b'T',
            b'C' => b'G',
            b'G' => b'C',
            b'T' => b'A',
            _ => b'N',
        })
        .collect()
}

#[test]
fn sort_coordinate_with_secondary_and_both_strands() {
    let dir = tempfile::tempdir().unwrap();
    // Every read twice, its reverse complement FIRST: the two land on the
    // same reference and position with opposite strands, so the reverse
    // flag — not input order — must put the forward copy first.
    let (_, input) = read_bam(&input_bam());
    let reads = dir.path().join("both.fa");
    let mut text = String::new();
    for r in &input {
        let seq = r.sequence().as_ref();
        let rc = String::from_utf8(revcomp(seq)).unwrap();
        let fwd = String::from_utf8(seq.to_vec()).unwrap();
        text.push_str(&format!(">{}_rc\n{rc}\n>{}_fwd\n{fwd}\n", name(r), name(r)));
    }
    std::fs::write(&reads, text).unwrap();
    // A duplicated reference makes every Ala read tie: secondaries.
    let fa = dir.path().join("dup.fa");
    let refs = std::fs::read_to_string(reference()).unwrap();
    let ala = refs
        .split('>')
        .find(|r| r.starts_with("tRNA-Ala-GGC-1-1\n"))
        .unwrap()
        .to_string();
    std::fs::write(&fa, format!("{refs}>dup-{ala}")).unwrap();

    let sorted = check_sorted(
        dir.path(),
        "sec_both",
        &reads,
        &fa,
        &["--secondary", "--strand", "both"],
    );
    let secondaries = sorted.iter().filter(|r| r.flags().is_secondary()).count();
    assert!(secondaries > 0, "no secondary records to sort");
    // The reverse-flag tie-break was exercised: some forward record sorts
    // directly before a reverse one at the same (tid, pos) that preceded it
    // in the input.
    let flipped = sorted.windows(2).any(|w| {
        w[0].reference_sequence_id() == w[1].reference_sequence_id()
            && w[0].alignment_start() == w[1].alignment_start()
            && !w[0].flags().is_reverse_complemented()
            && w[1].flags().is_reverse_complemented()
            && name(&w[0]).ends_with("_fwd")
            && name(&w[1]).ends_with("_rc")
    });
    assert!(flipped, "no forward/reverse tie at one position");
}

#[test]
fn sort_spill_matches_in_memory() {
    let dir = tempfile::tempdir().unwrap();
    let tmp = dir.path().join("spill");
    std::fs::create_dir(&tmp).unwrap();
    let (mem, spill) = (dir.path().join("mem.bam"), dir.path().join("spill.bam"));
    let extra = ["--sort", "coordinate", "--secondary", "--min-score", "150"];
    let e = align(&input_bam(), &reference(), &mem, &extra);
    assert!(!e.contains("spilled"), "the default budget spilled:\n{e}");
    let mut small = extra.to_vec();
    let tmp_arg = tmp.to_str().unwrap();
    small.extend(["--sort-memory", "16K", "--tmp-dir", tmp_arg]);
    let e = align(&input_bam(), &reference(), &spill, &small);
    assert!(e.contains("spilled"), "16K did not spill:\n{e}");

    let (mh, mrecs) = read_bam(&mem);
    let (sh, srecs) = read_bam(&spill);
    assert_eq!(header_text(&mh), header_text(&sh));
    assert_eq!(mrecs, srecs, "spilled output differs from in-memory");
    assert_eq!(
        std::fs::read_dir(&tmp).unwrap().count(),
        0,
        "temporary files left in --tmp-dir"
    );
}

#[test]
fn sort_output_is_indexable() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("sorted.bam");
    align(
        &input_bam(),
        &reference(),
        &out,
        &["--sort", "coordinate", "--min-score", "150", "--secondary"],
    );
    let index = bam::fs::index(&out).expect("noodles indexes the sorted output");
    drop(index);
    match samtools() {
        Some(st) => {
            let o = Command::new(st).arg("index").arg(&out).output().unwrap();
            assert!(
                o.status.success(),
                "samtools index refused the output:\n{}",
                String::from_utf8_lossy(&o.stderr)
            );
            assert!(dir.path().join("sorted.bam.bai").is_file());
        }
        None => eprintln!("[align_sort] sort_output_is_indexable: no samtools; noodles only"),
    }
    // And the unsorted output is refused by the same check, so it is a check.
    let unsorted = dir.path().join("unsorted.bam");
    align(&input_bam(), &reference(), &unsorted, &[]);
    assert!(bam::fs::index(&unsorted).is_err());
}

#[test]
fn sort_options_need_sort_coordinate() {
    let dir = tempfile::tempdir().unwrap();
    let e = align(
        &input_bam(),
        &reference(),
        &dir.path().join("o.bam"),
        &["--sort-memory", "1G"],
    );
    assert!(e.contains("only apply to --sort coordinate"), "{e}");
    // An unusable --tmp-dir fails before any read is aligned.
    let o = Command::new(env!("CARGO_BIN_EXE_escpod"))
        .arg("align")
        .arg(input_bam())
        .arg("-r")
        .arg(reference())
        .arg("-o")
        .arg(dir.path().join("o2.bam"))
        .args(["--sort", "coordinate", "--tmp-dir"])
        .arg(dir.path().join("does-not-exist"))
        .output()
        .unwrap();
    assert!(!o.status.success());
    assert!(String::from_utf8_lossy(&o.stderr).contains("--tmp-dir"));
}
