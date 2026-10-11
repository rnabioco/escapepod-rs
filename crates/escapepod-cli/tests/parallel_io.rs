// SPDX-License-Identifier: MIT

//! Commands that overlap per-file I/O must print what the serial loop printed
//! (#465).
//!
//! `summary` and `inspect summary` analyse their inputs on the rayon pool; the
//! only thing that makes that safe is folding the results back in input order.
//! Nothing but the output shows a broken fold, so these run the binary at
//! different pool widths and demand the same bytes.

#![cfg(feature = "cli")]

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("escapepod-classify/tests/fixtures")
}

fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_escpod"))
        .args(args)
        .env_remove("RUST_LOG")
        .env_remove("RAYON_NUM_THREADS")
        .env("NO_COLOR", "1")
        .output()
        .expect("failed to run escpod")
}

fn stdout_of(args: &[&str]) -> String {
    let out = run(args);
    assert!(
        out.status.success(),
        "`escpod {}` failed:\n{}",
        args.join(" "),
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).unwrap()
}

/// A scratch directory of POD5 copies, written in an order that is neither
/// sorted nor reverse-sorted, plus one file that is not a POD5 at all.
struct Dir {
    path: PathBuf,
}

impl Dir {
    fn new(name: &str, with_corrupt: bool) -> Self {
        let path = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("parallel-io-{name}"));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).unwrap();
        for n in ["d", "b", "f", "a", "e", "c"] {
            std::fs::copy(
                fixtures().join("trna_reads.pod5"),
                path.join(format!("{n}.pod5")),
            )
            .unwrap();
        }
        if with_corrupt {
            // Sorts between `b` and `c`: a failure in the middle of the list.
            std::fs::write(path.join("bz.pod5"), b"not a pod5 file").unwrap();
        }
        Dir { path }
    }

    fn arg(&self) -> &str {
        self.path.to_str().unwrap()
    }
}

impl Drop for Dir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

#[test]
fn summary_dir_output_order_stable() {
    let dir = Dir::new("summary", true);

    let text_1 = stdout_of(&["summary", dir.arg(), "-t", "1"]);
    let text_8 = stdout_of(&["summary", dir.arg(), "-t", "8"]);
    assert_eq!(text_1, text_8, "table output depends on pool width");

    let json_1: serde_json::Value =
        serde_json::from_str(&stdout_of(&["summary", dir.arg(), "--json", "-t", "1"])).unwrap();
    let json_8: serde_json::Value =
        serde_json::from_str(&stdout_of(&["summary", dir.arg(), "--json", "-t", "8"])).unwrap();
    assert_eq!(json_1, json_8, "JSON output depends on pool width");

    // Files appear in the order the directory resolves to (sorted), the
    // unreadable one is absent from the list and named in the warnings.
    let names: Vec<String> = json_8["files"]
        .as_array()
        .unwrap()
        .iter()
        .map(|f| {
            Path::new(f["path"].as_str().unwrap())
                .file_name()
                .unwrap()
                .to_string_lossy()
                .into_owned()
        })
        .collect();
    assert_eq!(
        names,
        ["a.pod5", "b.pod5", "c.pod5", "d.pod5", "e.pod5", "f.pod5"]
    );
    let warnings = json_8["warnings"].as_array().unwrap();
    assert!(
        warnings[0].as_str().unwrap().contains("1 file(s)"),
        "{warnings:?}"
    );
    assert!(
        warnings[1].as_str().unwrap().contains("bz.pod5"),
        "{warnings:?}"
    );
}

#[test]
fn inspect_summary_dir_output_order_stable() {
    let dir = Dir::new("inspect", true);

    let one = run(&["inspect", "summary", dir.arg(), "-t", "1"]);
    let many = run(&["inspect", "summary", dir.arg(), "-t", "8"]);
    assert!(one.status.success() && many.status.success());
    assert_eq!(one.stdout, many.stdout);

    let text = String::from_utf8(many.stdout).unwrap();
    let positions: Vec<usize> = ["a.pod5", "b.pod5", "c.pod5", "d.pod5", "e.pod5", "f.pod5"]
        .iter()
        .map(|n| {
            text.find(n)
                .unwrap_or_else(|| panic!("{n} missing:\n{text}"))
        })
        .collect();
    assert!(positions.windows(2).all(|w| w[0] < w[1]), "{text}");
    assert!(
        String::from_utf8_lossy(&many.stderr).contains("skipping"),
        "unreadable file not reported"
    );
}

#[test]
fn bam_filter_threads_flag_respected() {
    let out_dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("parallel-io-bamfilter");
    let _ = std::fs::remove_dir_all(&out_dir);
    std::fs::create_dir_all(&out_dir).unwrap();

    let pod5 = fixtures().join("trna_reads.pod5");
    let bam = fixtures().join("trna_mappings_padded.bam");

    let mut views = Vec::new();
    for t in ["1", "4"] {
        let out = out_dir.join(format!("t{t}.pod5"));
        // `-v` makes `threads::init` report the pool it built.
        let r = Command::new(env!("CARGO_BIN_EXE_escpod"))
            .arg("-v")
            .arg("bam-filter")
            .arg(&pod5)
            .arg("-b")
            .arg(&bam)
            .arg("-o")
            .arg(&out)
            .args(["--mapped", "-t", t])
            .env_remove("RUST_LOG")
            .env_remove("RAYON_NUM_THREADS")
            .output()
            .expect("failed to run escpod");
        let stderr = String::from_utf8_lossy(&r.stderr);
        assert!(r.status.success(), "bam-filter -t {t} failed:\n{stderr}");
        assert!(
            stderr.contains(&format!("rayon pool: {t} thread(s)")),
            "-t {t} did not size the pool:\n{stderr}"
        );
        // Row order is the BAM-id `HashSet`'s iteration order, which is
        // per-process random on main too; the *set* of reads is the contract.
        let mut rows: Vec<String> = stdout_of(&["view", out.to_str().unwrap()])
            .lines()
            .map(str::to_owned)
            .collect();
        rows.sort();
        views.push(rows);
    }
    // The BGZF inflate pool is sized from the rayon pool; the scan's result
    // must not depend on how many workers inflate.
    assert!(views[0].len() > 1, "filter selected nothing");
    assert_eq!(views[0], views[1]);

    let _ = std::fs::remove_dir_all(&out_dir);
}
