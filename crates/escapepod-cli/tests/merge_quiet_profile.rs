// SPDX-License-Identifier: MIT

//! `merge --profile`'s multi-line report is status output, not the command's
//! data — like `subset`'s own summary block, it must be gated on the verbosity
//! level so `-q merge --profile` stays quiet rather than printing anyway.

#![cfg(feature = "cli")]

use std::path::{Path, PathBuf};
use std::process::Command;

fn fixture_pod5() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("escapepod-classify/tests/fixtures/trna_reads.pod5")
}

#[test]
fn quiet_merge_profile_prints_no_report() {
    let out_file = tempfile::NamedTempFile::new().unwrap();
    let out_path = out_file.path();
    let out = Command::new(env!("CARGO_BIN_EXE_escpod"))
        .args([
            "-q",
            "merge",
            fixture_pod5().to_str().unwrap(),
            "-o",
            out_path.to_str().unwrap(),
            "--force",
            "--profile",
        ])
        .env_remove("RUST_LOG")
        .output()
        .expect("failed to run escpod");
    assert!(
        out.status.success(),
        "escpod -q merge --profile failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !stderr.contains("Profile"),
        "-q suppressed status but --profile's report still printed:\n{stderr}"
    );
}
