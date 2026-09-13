// SPDX-License-Identifier: MIT

//! `repack` has no data product — everything it prints is status — so it must
//! flow through `tracing` to stderr, leaving stdout empty for `escpod repack
//! *.pod5 -o out/ > log.txt` to redirect independently of the files it wrote.

#![cfg(feature = "experimental")]

use std::path::{Path, PathBuf};
use std::process::Command;

fn fixture_pod5() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("escapepod-classify/tests/fixtures/trna_reads.pod5")
}

#[test]
fn repack_writes_nothing_to_stdout() {
    let out_dir = tempfile::tempdir().unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_escpod"))
        .args([
            "repack",
            fixture_pod5().to_str().unwrap(),
            "-o",
            out_dir.path().to_str().unwrap(),
        ])
        .env_remove("RUST_LOG")
        .output()
        .expect("failed to run escpod");
    assert!(
        out.status.success(),
        "escpod repack failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        out.stdout.is_empty(),
        "repack has no data product but wrote to stdout:\n{}",
        String::from_utf8_lossy(&out.stdout)
    );
}
