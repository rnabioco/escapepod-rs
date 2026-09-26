//! `ESCAPEPOD_AUTOINDEX_MAX=0` disables speculative index warm-up
//! (rnabioco/escapepod-rs#421).
//!
//! `0` is the documented way to turn warm-up off: every file has more reads
//! than it, so every `reads > autoindex_max()` gate skips. #412 moved the
//! parse onto `env::positive_usize`, which rejects `0` and falls back to the
//! 5,000,000 default — so `=0` silently warmed everything again, and the only
//! test setting it (the Python one) checked read ids, which are the same
//! either way.
//!
//! Its own test binary on purpose: it sets a process-global environment
//! variable, and the warm-up assertions in `test_reader_cache.rs` would race
//! it under a threaded `cargo test`.

mod common;

use std::collections::HashSet;

use escapepod_pod5::{DatasetCache, ReaderCache, Uuid, autoindex_max};
use tempfile::TempDir;

use common::write_fixture;

const VAR: &str = "ESCAPEPOD_AUTOINDEX_MAX";

#[test]
fn autoindex_max_zero_disables_warmup() {
    let tmp = TempDir::new().unwrap();
    let control_dir = tmp.path().join("control");
    let zero_dir = tmp.path().join("zero");
    std::fs::create_dir_all(&control_dir).unwrap();
    std::fs::create_dir_all(&zero_dir).unwrap();
    let control = control_dir.join("reads.pod5");
    let zero = zero_dir.join("reads.pod5");
    write_fixture(&control, "autoindex_control", 20, 400);
    write_fixture(&zero, "autoindex_zero", 20, 400);

    // Control: unset means the default, and a 20-read file is warmed — so the
    // `is_none` below is the gate at work, not a cache that never warms.
    unsafe { std::env::remove_var(VAR) };
    assert_eq!(autoindex_max(), 5_000_000);
    let warm = ReaderCache::new().get(&control).unwrap();
    assert!(
        warm.read_index_if_built().is_some(),
        "control: with {VAR} unset a small file must be warmed"
    );

    unsafe { std::env::set_var(VAR, "0") };
    assert_eq!(autoindex_max(), 0, "{VAR}=0 must be a threshold of 0");

    // ReaderCache::get's warm-up is skipped.
    let cold = ReaderCache::new().get(&zero).unwrap();
    assert!(
        cold.read_index_if_built().is_none(),
        "{VAR}=0 must skip ReaderCache warm-up"
    );

    // Dataset::open routes every file through the global reader cache, so its
    // per-file readers must come back cold as well; and DatasetCache's own
    // routing-index warm-up (gated identically) must not build them either.
    let dataset = DatasetCache::new()
        .get(&[&zero_dir], true, ".pod5")
        .unwrap();
    assert!(
        dataset
            .reader_at(0)
            .unwrap()
            .read_index_if_built()
            .is_none(),
        "{VAR}=0 must skip dataset warm-up"
    );

    // Skipping defers the build, it never refuses it: a lookup still works.
    let ids = cold.read_ids().unwrap();
    let wanted: HashSet<Uuid> = ids[..3].iter().copied().collect();
    assert_eq!(cold.reads_by_ids(&wanted).unwrap().len(), 3);
    assert!(cold.read_index_if_built().is_some());

    unsafe { std::env::remove_var(VAR) };
}
