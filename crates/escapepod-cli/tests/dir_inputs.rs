// SPDX-License-Identifier: MIT

//! Every command that takes POD5 input accepts a directory, and refuses a path
//! that does not exist (#293).
//!
//! Both halves have to run the real binary: input resolution happens in the
//! clap dispatch, and what #293 was actually about is the *exit status* of the
//! process — `demux fingerprint` and `demux basecall` used to skip an
//! unreadable input, write a header-only CSV and exit 0, which no caller can
//! tell from a run where no read passed. A unit test on the resolver would
//! have been green throughout.

#![cfg(feature = "cli")]

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn fixture_pod5() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("escapepod-classify/tests/fixtures/trna_reads.pod5")
}

fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_escpod"))
        .args(args)
        .env_remove("RUST_LOG")
        .output()
        .expect("failed to run escpod")
}

fn assert_ok(args: &[&str]) -> Output {
    let out = run(args);
    assert!(
        out.status.success(),
        "`escpod {}` failed:\n{}",
        args.join(" "),
        String::from_utf8_lossy(&out.stderr)
    );
    out
}

/// A CSV is more than its header — the failure this file exists for wrote a
/// header and stopped.
fn assert_has_rows(path: &Path) {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("{} unreadable: {e}", path.display()));
    let rows = text.lines().skip(1).filter(|l| !l.is_empty()).count();
    assert!(rows > 0, "{} has no data rows:\n{text}", path.display());
}

/// A directory holding one copy of the fixture, plus somewhere to write.
struct Workspace {
    _tmp: tempfile::TempDir,
    dir: PathBuf,
    out: PathBuf,
}

impl Workspace {
    fn new() -> Self {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("pod5");
        let out = tmp.path().join("out");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::create_dir_all(&out).unwrap();
        std::fs::copy(fixture_pod5(), dir.join("reads.pod5")).unwrap();
        Self {
            _tmp: tmp,
            dir,
            out,
        }
    }

    fn dir(&self) -> &str {
        self.dir.to_str().unwrap()
    }

    fn out(&self, name: &str) -> PathBuf {
        self.out.join(name)
    }

    /// Boundaries for the fixture, detected from the directory — which makes
    /// this both a fixture builder and the `detect` half of the assertion.
    fn boundaries(&self) -> PathBuf {
        let csv = self.out("boundaries.csv");
        assert_ok(&[
            "demux",
            "detect",
            "--method",
            "llr",
            self.dir(),
            "-o",
            csv.to_str().unwrap(),
        ]);
        assert_has_rows(&csv);
        csv
    }
}

/// `read_id,<column>` over every read in a boundaries CSV, for the commands
/// that take an assignment table.
fn mapping_from(boundaries: &Path, column: &str, value: &str) -> String {
    let text = std::fs::read_to_string(boundaries).unwrap();
    let mut csv = format!("read_id,{column}\n");
    for line in text.lines().skip(1).filter(|l| !l.is_empty()) {
        let id = line.split(',').next().unwrap();
        csv.push_str(&format!("{id},{value}\n"));
    }
    csv
}

#[test]
fn demux_detect_accepts_a_directory() {
    // The whole assertion is inside `boundaries()`: run against the directory,
    // exit 0, rows written.
    Workspace::new().boundaries();
}

#[test]
fn demux_fingerprint_accepts_a_directory() {
    let ws = Workspace::new();
    let boundaries = ws.boundaries();
    let fp = ws.out("fingerprints.csv");
    assert_ok(&[
        "demux",
        "fingerprint",
        ws.dir(),
        "--boundaries",
        boundaries.to_str().unwrap(),
        "-o",
        fp.to_str().unwrap(),
    ]);
    assert_has_rows(&fp);
}

#[test]
fn demux_split_accepts_a_directory() {
    let ws = Workspace::new();
    let boundaries = ws.boundaries();
    let cls = ws.out("classifications.csv");
    std::fs::write(&cls, mapping_from(&boundaries, "barcode", "bc01")).unwrap();
    let split_dir = ws.out("split");
    assert_ok(&[
        "demux",
        "split",
        ws.dir(),
        "--classifications",
        cls.to_str().unwrap(),
        "-d",
        split_dir.to_str().unwrap(),
    ]);
    assert!(
        std::fs::read_dir(&split_dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .any(|e| e.path().extension().is_some_and(|x| x == "pod5")),
        "no POD5 written to {}",
        split_dir.display()
    );
}

#[test]
fn subset_accepts_a_directory() {
    let ws = Workspace::new();
    let boundaries = ws.boundaries();
    let map = ws.out("mapping.csv");
    std::fs::write(&map, mapping_from(&boundaries, "output", "g1.pod5")).unwrap();
    let subset_dir = ws.out("subset");
    assert_ok(&[
        "subset",
        ws.dir(),
        "--csv",
        map.to_str().unwrap(),
        "-o",
        subset_dir.to_str().unwrap(),
    ]);
    assert!(
        subset_dir.join("g1.pod5").exists(),
        "no g1.pod5 in {}",
        subset_dir.display()
    );
}

/// Assert `args` fails and says so about the missing path, rather than exiting
/// 0 with an empty table.
fn assert_missing_input_is_fatal(args: &[&str]) {
    let out = run(args);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !out.status.success(),
        "`escpod {}` succeeded on a nonexistent input:\n{stderr}",
        args.join(" ")
    );
    assert!(
        stderr.contains("Path does not exist"),
        "`escpod {}` failed without naming the missing path:\n{stderr}",
        args.join(" ")
    );
}

#[test]
fn a_nonexistent_input_is_fatal() {
    let ws = Workspace::new();
    let boundaries = ws.boundaries();
    let boundaries = boundaries.to_str().unwrap();
    let missing = ws.out("nope.pod5");
    let missing = missing.to_str().unwrap();
    let out = ws.out("out.csv");
    let out = out.to_str().unwrap();

    assert_missing_input_is_fatal(&["demux", "detect", "--method", "llr", missing, "-o", out]);
    assert_missing_input_is_fatal(&[
        "demux",
        "fingerprint",
        missing,
        "--boundaries",
        boundaries,
        "-o",
        out,
    ]);
    assert_missing_input_is_fatal(&[
        "demux",
        "split",
        missing,
        "--classifications",
        boundaries,
        "-d",
        out,
    ]);
    assert_missing_input_is_fatal(&["subset", missing, "--csv", boundaries, "-o", out]);

    // Resolution runs before the model is opened, so this needs no bundle: the
    // point is that the missing POD5 is what the run dies of.
    #[cfg(feature = "crf-decode")]
    assert_missing_input_is_fatal(&[
        "demux",
        "basecall",
        missing,
        "--boundaries",
        boundaries,
        "--model",
        "/nonexistent-bundle",
        "-o",
        out,
    ]);
}

#[test]
fn an_empty_directory_is_fatal() {
    let ws = Workspace::new();
    let empty = ws.out("empty");
    std::fs::create_dir_all(&empty).unwrap();
    let out = run(&[
        "demux",
        "detect",
        "--method",
        "llr",
        empty.to_str().unwrap(),
        "-o",
        ws.out("b.csv").to_str().unwrap(),
    ]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "empty directory accepted:\n{stderr}");
    assert!(
        stderr.contains("No POD5 files found"),
        "unexpected error for an empty directory:\n{stderr}"
    );
}

/// A directory gets **one** sidecar, and every consumer still finds a read's
/// labels through the POD5 it asks about.
///
/// This is the whole point of the collection shape: the answer a run produced
/// is one result, so it lands in one file rather than being copied into every
/// member. `view --include` going through `operations::read_columns` is what
/// says the copies are not needed — nothing downstream has to learn that a
/// collection exists.
#[cfg(feature = "experimental")]
#[test]
fn annotate_on_a_directory_writes_one_collection_sidecar() {
    let ws = Workspace::new();
    let boundaries = ws.boundaries();
    let csv = ws.out("assign.csv");
    std::fs::write(&csv, mapping_from(&boundaries, "barcode", "BC01")).unwrap();

    assert_ok(&["annotate", ws.dir(), "-a", csv.to_str().unwrap()]);

    let collection = ws.dir.with_extension("p5s");
    assert!(
        collection.exists(),
        "no collection sidecar at {}",
        collection.display()
    );
    let strays: Vec<PathBuf> = std::fs::read_dir(&ws.dir)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "p5s"))
        .collect();
    assert!(
        strays.is_empty(),
        "a directory input must not also write per-file sidecars: {strays:?}"
    );

    let out = assert_ok(&[
        "view",
        ws.dir.join("reads.pod5").to_str().unwrap(),
        "--include",
        "barcode",
    ]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("BC01"),
        "view did not resolve the collection:\n{stdout}"
    );

    // `--list` has to account for the members, not disown them. A file whose
    // labels live in the collection is *covered*; printing "no sidecar" under
    // the collection that holds them says the opposite of what is true.
    let out = assert_ok(&["annotate", "--list", ws.dir()]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("reads indexed across 1 files"),
        "--list did not report the collection:\n{stdout}"
    );
    assert!(
        stdout.contains("covered by") && !stdout.contains("no sidecar"),
        "--list must call a member covered, not unannotated:\n{stdout}"
    );
}
