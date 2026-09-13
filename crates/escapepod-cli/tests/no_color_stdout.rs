// SPDX-License-Identifier: MIT

//! `summary` and `inspect` print their data to stdout, so their coloring must
//! be gated on stdout's own terminal status — not stderr's, and not emitted
//! unconditionally. `Command::output()` captures both streams as pipes, which
//! is exactly the "redirected" case (`escpod summary x.pod5 > out.txt`,
//! `escpod inspect ... | less`) that must come out with no ANSI at all.

#![cfg(feature = "cli")]

use std::path::{Path, PathBuf};
use std::process::Command;

fn fixture_pod5() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("escapepod-classify/tests/fixtures/trna_reads.pod5")
}

fn assert_no_ansi(args: &[&str]) {
    let out = Command::new(env!("CARGO_BIN_EXE_escpod"))
        .args(args)
        .env_remove("RUST_LOG")
        .env_remove("NO_COLOR")
        .env_remove("CLICOLOR")
        .env_remove("CLICOLOR_FORCE")
        .output()
        .expect("failed to run escpod");
    assert!(
        out.status.success(),
        "`escpod {}` failed:\n{}",
        args.join(" "),
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        !stdout.contains('\x1b'),
        "`escpod {}` wrote raw ANSI to redirected stdout:\n{stdout}",
        args.join(" ")
    );
}

#[test]
fn summary_table_has_no_ansi_when_stdout_is_redirected() {
    assert_no_ansi(&["summary", fixture_pod5().to_str().unwrap()]);
}

#[test]
fn inspect_summary_has_no_ansi_when_stdout_is_redirected() {
    assert_no_ansi(&["inspect", "summary", fixture_pod5().to_str().unwrap()]);
}
