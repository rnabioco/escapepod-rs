//! `--threads` actually bounds the rayon pool (#155).
//!
//! The unit tests in `src/threads.rs` cover the resolution policy and the ones
//! in `main.rs` cover the flag reaching `main`. What neither can check is the
//! process-global part: rayon's pool is built once and irreversibly, so the
//! only honest way to observe its width is to run the binary. `threads::init`
//! logs the *observed* `rayon::current_num_threads()` at debug level, which
//! makes that a real assertion target.

#![cfg(feature = "cli")]

use std::process::Command;

/// Run `escpod -v <args>` and return stderr.
///
/// The commands below are all expected to *fail* (missing inputs). That is the
/// point: the pool is sized before dispatch, so the width is reported no matter
/// how the command itself fares — which is itself the ordering property #155
/// was about.
fn stderr_of(args: &[&str]) -> String {
    let out = Command::new(env!("CARGO_BIN_EXE_escpod"))
        .arg("-v")
        .args(args)
        // Must not leak in from the caller's environment: it would override the
        // default and break `default_is_capped_at_available`.
        .env_remove("RAYON_NUM_THREADS")
        .env_remove("RUST_LOG")
        .output()
        .expect("failed to run escpod");
    String::from_utf8_lossy(&out.stderr).into_owned()
}

fn assert_pool_width(args: &[&str], want: usize) {
    let stderr = stderr_of(args);
    let want = format!("rayon pool: {want} thread(s)");
    assert!(
        stderr.contains(&want),
        "expected {want:?} for `escpod {}`, got:\n{stderr}",
        args.join(" ")
    );
}

#[test]
fn explicit_threads_bound_the_pool() {
    assert_pool_width(
        &["merge", "no-such-file.pod5", "-o", "out.pod5", "-t", "3"],
        3,
    );
}

#[test]
fn default_is_capped_at_available_parallelism() {
    let available: usize = std::thread::available_parallelism().map_or(1, Into::into);
    assert_pool_width(
        &["merge", "no-such-file.pod5", "-o", "out.pod5"],
        available.min(16),
    );
}

/// The #155 regression itself.
///
/// `demux detect --method cnn` used to run a full-width pool for every value of
/// `-j`, because its pool-sizing call sat *after* a `par_iter()` that had
/// already built the global pool at `available_parallelism()`. Both methods must
/// now report the requested width. Neither command gets as far as reading a
/// read — that is fine, the pool is sized before dispatch either way.
#[cfg(feature = "demux")]
#[test]
fn demux_detect_honours_threads_for_every_method() {
    let base = ["demux", "detect", "no-such-file.pod5", "-o", "out.csv"];

    let argv = [&base[..], &["--method", "llr", "-j", "3"]].concat();
    assert_pool_width(&argv, 3);

    #[cfg(feature = "cnn-detect")]
    {
        let argv = [
            &base[..],
            &[
                "--method",
                "cnn",
                "--cnn-model",
                "no-such-model.onnx",
                "-j",
                "3",
            ],
        ]
        .concat();
        assert_pool_width(&argv, 3);
    }
}

/// `bam-filter` sizes the pool like every other fan-out command.
#[test]
fn bam_filter_threads_bound_the_pool() {
    assert_pool_width(
        &[
            "bam-filter",
            "no-such-file.pod5",
            "-b",
            "no-such-file.bam",
            "-o",
            "out.pod5",
            "-t",
            "3",
        ],
        3,
    );
}

/// `-t 0` would mean "rayon picks for me", i.e. another silently-ignored flag.
#[test]
fn zero_threads_is_rejected() {
    let stderr = stderr_of(&["merge", "no-such-file.pod5", "-o", "out.pod5", "-t", "0"]);
    assert!(
        stderr.contains("--threads must be at least 1"),
        "expected a rejection, got:\n{stderr}"
    );
}

/// Nothing outside `threads.rs` may build the global rayon pool.
///
/// This is the guard that keeps #155 from coming back. The bug was not the
/// arithmetic — it was that a command sized the pool itself, from a spot where
/// something else had already built it, and swallowed the resulting `Err`. Any
/// new `build_global()` call site reopens that hazard, so fail the build here
/// rather than discover it in a benchmark a release later.
#[test]
fn build_global_is_confined_to_the_threads_module() {
    fn visit(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
        for entry in std::fs::read_dir(dir).expect("read_dir") {
            let path = entry.expect("dir entry").path();
            if path.is_dir() {
                visit(&path, out);
            } else if path.extension().is_some_and(|e| e == "rs") {
                out.push(path);
            }
        }
    }

    let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    visit(&src, &mut files);
    assert!(
        !files.is_empty(),
        "found no sources under {}",
        src.display()
    );

    let offenders: Vec<_> = files
        .iter()
        .filter(|p| p.file_name().is_some_and(|n| n != "threads.rs"))
        .filter(|p| std::fs::read_to_string(p).is_ok_and(|s| s.contains("build_global")))
        .map(|p| p.strip_prefix(&src).unwrap_or(p).display().to_string())
        .collect();

    assert!(
        offenders.is_empty(),
        "the global rayon pool must be built only in threads.rs, from main, \
         before dispatch (#155); found build_global in: {offenders:?}"
    );
}
