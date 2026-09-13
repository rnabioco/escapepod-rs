// SPDX-License-Identifier: MIT

//! `inspect read`'s own help text promises a dash-less 32-hex read ID works
//! (`src/main.rs`'s `after_help` for the `Read` subcommand), but it used to
//! reject that form outright: `read_id.parse::<Uuid>()` only accepts the
//! canonical dashed form, where `escapepod_signal::parse_uuid_flexible` (used
//! everywhere else the CLI takes a read ID) accepts both.

#![cfg(feature = "cli")]

use std::path::{Path, PathBuf};
use std::process::Command;

fn fixture_pod5() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("escapepod-classify/tests/fixtures/trna_reads.pod5")
}

fn run(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_escpod"))
        .args(args)
        .env_remove("RUST_LOG")
        .output()
        .expect("failed to run escpod")
}

#[test]
fn inspect_read_accepts_a_dash_less_id() {
    let pod5 = fixture_pod5();
    let pod5 = pod5.to_str().unwrap();

    let reads = run(&["inspect", "reads", pod5]);
    assert!(reads.status.success(), "`escpod inspect reads` failed");
    let stdout = String::from_utf8_lossy(&reads.stdout);
    let dashed_id = stdout
        .lines()
        .nth(2)
        .and_then(|line| line.split_whitespace().next())
        .expect("no read row in `inspect reads` output")
        .to_string();
    let dashless_id: String = dashed_id.chars().filter(|c| *c != '-').collect();
    assert_eq!(dashless_id.len(), 32, "not a 32-hex UUID: {dashed_id}");

    let out = run(&["inspect", "read", pod5, &dashless_id]);
    assert!(
        out.status.success(),
        "`escpod inspect read` rejected the dash-less form:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains(&dashed_id),
        "output did not resolve to read {dashed_id}:\n{stdout}"
    );
}
