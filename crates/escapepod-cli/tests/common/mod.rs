// SPDX-License-Identifier: MIT

//! Helpers shared by the integration tests that run the real `escpod` binary
//! against a copy of the tracked 20-read fixture.
//!
//! Each test crate uses a different subset, so unused items are expected.

#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

pub fn fixture_pod5() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("escapepod-classify/tests/fixtures/trna_reads.pod5")
}

pub fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_escpod"))
        .args(args)
        .env_remove("RUST_LOG")
        .output()
        .expect("failed to run escpod")
}

pub fn assert_ok(args: &[&str]) -> Output {
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
pub fn assert_has_rows(path: &Path) {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("{} unreadable: {e}", path.display()));
    let rows = text.lines().skip(1).filter(|l| !l.is_empty()).count();
    assert!(rows > 0, "{} has no data rows:\n{text}", path.display());
}

/// A directory holding one copy of the fixture, plus somewhere to write.
pub struct Workspace {
    _tmp: tempfile::TempDir,
    pub dir: PathBuf,
    pub out: PathBuf,
}

impl Workspace {
    pub fn new() -> Self {
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

    pub fn dir(&self) -> &str {
        self.dir.to_str().unwrap()
    }

    pub fn out(&self, name: &str) -> PathBuf {
        self.out.join(name)
    }

    /// Boundaries for the fixture, detected from the directory — which makes
    /// this both a fixture builder and the `detect` half of the assertion.
    pub fn boundaries(&self) -> PathBuf {
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
