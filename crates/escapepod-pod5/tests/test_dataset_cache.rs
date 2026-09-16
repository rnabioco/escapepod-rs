//! Integration tests for [`Dataset`] / [`DatasetCache`] / [`cached_dataset`].
//!
//! `Dataset` lifts the directory-scan + read-id-routing logic that used to
//! exist only in `escapepod-python`'s `PyDatasetReader` into plain Rust
//! (escapepod-rs#384), so a Rust consumer of `escapepod-pod5` (or
//! `escapepod-signal`) can treat a MinKNOW run directory as one logical,
//! random-access-by-read-id source without reimplementing the scan. Its value
//! is in three contracts, mirrored here as in `test_reader_cache.rs`: the
//! directory scan resolves to the right file set, a read's signal comes back
//! identical whichever path reaches it, and every file it opens is shared
//! with the process-global [`ReaderCache`] rather than a second, parallel
//! cache.
//!
//! Every fixture is written by the test itself, so all of this runs in CI
//! (`ext/` is an empty submodule there).

mod common;

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;

use escapepod_pod5::{
    Dataset, DatasetCache, Reader, cached_dataset, cached_reader, global_dataset_cache,
};
use tempfile::TempDir;

use common::{WrittenFile, write_fixture};

const N_READS: usize = 12;
const SAMPLES_PER_READ: usize = 400;

/// Write a fixture POD5 at `tmp/name` and return its path plus the ids it
/// holds, for tests that need to name a specific read.
fn fixture(tmp: &TempDir, name: &str, acq_id: &str, n_reads: usize) -> (PathBuf, WrittenFile) {
    let path = tmp.path().join(name);
    let written = write_fixture(&path, acq_id, n_reads, SAMPLES_PER_READ);
    (path, written)
}

#[test]
fn directory_scan_is_recursive_by_default_and_can_be_shallow() {
    let tmp = TempDir::new().unwrap();
    let (top, _) = fixture(&tmp, "top.pod5", "top_acq", N_READS);
    std::fs::create_dir(tmp.path().join("sub")).unwrap();
    fixture(&tmp, "sub/nested.pod5", "nested_acq", N_READS);

    let shallow = Dataset::open_with(&[tmp.path()], false, ".pod5").unwrap();
    assert_eq!(
        shallow.file_count(),
        1,
        "a non-recursive scan must not descend into subdirectories"
    );
    assert_eq!(shallow.paths(), vec![top.as_path()]);

    let deep = Dataset::open_with(&[tmp.path()], true, ".pod5").unwrap();
    assert_eq!(
        deep.file_count(),
        2,
        "a recursive scan must find files in subdirectories too"
    );
}

#[test]
fn a_mixed_file_and_directory_input_dedupes_overlapping_files() {
    let tmp = TempDir::new().unwrap();
    let (top, _) = fixture(&tmp, "top.pod5", "top_acq", N_READS);

    // The file is named explicitly *and* reachable via the directory scan.
    let dataset = Dataset::open_with(&[top, tmp.path().to_path_buf()], true, ".pod5").unwrap();
    assert_eq!(
        dataset.file_count(),
        1,
        "a file reachable both directly and via a directory scan must appear once"
    );
}

#[test]
fn reads_by_ids_and_bulk_decode_match_a_direct_reader_read() {
    let tmp = TempDir::new().unwrap();
    fixture(&tmp, "a.pod5", "acq_a", N_READS);
    std::fs::create_dir(tmp.path().join("sub")).unwrap();
    let (b_path, b) = fixture(&tmp, "sub/b.pod5", "acq_b", N_READS);

    let dataset = Dataset::open(tmp.path()).unwrap();
    assert_eq!(dataset.file_count(), 2);

    let target_id = b.read_ids[3];
    let targets = HashSet::from([target_id]);

    let via_dataset = dataset.reads_by_ids(&targets).unwrap();
    assert_eq!(via_dataset.len(), 1);
    assert_eq!(via_dataset[0].read_id, target_id);

    let decoded = dataset.decode_bulk(&via_dataset, usize::MAX).unwrap();
    assert_eq!(decoded.len(), 1);
    let (decoded_id, decoded_signal) = &decoded[0];
    assert_eq!(*decoded_id, target_id);

    // The reference: fetch the same read's signal directly off its owning
    // file, with no Dataset involved at all.
    let direct_reader = Reader::open(&b_path).unwrap();
    let expected_signal = direct_reader
        .get_signal_prefix(&via_dataset[0].signal_rows, usize::MAX)
        .unwrap();
    assert_eq!(
        *decoded_signal, expected_signal,
        "Dataset::decode_bulk must return byte-identical signal to a direct Reader read"
    );
}

#[test]
fn run_infos_is_deduplicated_by_acquisition_id_across_files() {
    let tmp = TempDir::new().unwrap();
    fixture(&tmp, "a.pod5", "shared_acq", N_READS);
    fixture(&tmp, "b.pod5", "shared_acq", N_READS);
    fixture(&tmp, "c.pod5", "distinct_acq", N_READS);

    let dataset = Dataset::open(tmp.path()).unwrap();
    assert_eq!(dataset.file_count(), 3);

    let infos = dataset.run_infos();
    let ids: HashSet<_> = infos.iter().map(|ri| ri.acquisition_id.clone()).collect();
    assert_eq!(
        ids.len(),
        2,
        "two distinct acquisition ids across three files"
    );
    assert_eq!(
        infos.len(),
        2,
        "run_infos must hold one entry per acquisition id, not one per file"
    );
}

#[test]
fn opening_a_directory_with_no_matching_files_is_a_clear_error() {
    let tmp = TempDir::new().unwrap();
    let empty = tmp.path().join("empty");
    std::fs::create_dir(&empty).unwrap();

    // `Dataset` holds `Arc<Reader>`, which does not implement `Debug`, so the
    // `Ok` arm can't go through `unwrap_err()`; match instead.
    let message = match Dataset::open(&empty) {
        Err(e) => e.to_string(),
        Ok(_) => panic!("an empty directory must not open as a dataset"),
    };
    assert!(
        message.contains("no POD5 files found"),
        "expected a clear 'no files found' error, got: {message}"
    );
}

#[test]
fn cached_dataset_shares_one_dataset_per_root_whatever_the_spelling() {
    let tmp = TempDir::new().unwrap();
    fixture(&tmp, "shared.pod5", "shared_acq", N_READS);

    let first = cached_dataset(tmp.path()).unwrap();
    let second = cached_dataset(tmp.path().join(".")).unwrap();

    assert!(
        Arc::ptr_eq(&first, &second),
        "the same directory, spelled differently, must return the same Arc<Dataset>"
    );
    assert!(!global_dataset_cache().is_empty());
}

#[test]
fn cached_dataset_routes_file_opens_through_the_existing_reader_cache() {
    let tmp = TempDir::new().unwrap();
    let (path, written) = fixture(&tmp, "routed.pod5", "routed_acq", N_READS);

    let dataset = cached_dataset(tmp.path()).unwrap();
    let target_id = written.read_ids[0];

    let via_dataset = dataset
        .owning_reader(&target_id)
        .expect("the read must be routable to its owning file");
    let via_direct = cached_reader(&path).unwrap();

    assert!(
        Arc::ptr_eq(via_dataset, &via_direct),
        "a file inside a cached dataset and that same file opened directly through \
         cached_reader must share one Arc<Reader> — proof Dataset routes through the \
         existing per-file cache rather than a second one"
    );
}

#[test]
fn a_local_dataset_cache_behaves_like_reader_cache_one_level_up() {
    let tmp = TempDir::new().unwrap();
    fixture(&tmp, "a.pod5", "acq_a", N_READS);
    fixture(&tmp, "b.pod5", "acq_b", N_READS + 1);

    let cache = DatasetCache::new();
    assert!(cache.is_empty());

    let d1 = cache.get(&[tmp.path()], true, ".pod5").unwrap();
    let d2 = cache.get(&[tmp.path()], true, ".pod5").unwrap();
    assert!(Arc::ptr_eq(&d1, &d2));
    assert_eq!(cache.len(), 1);
    assert!(!cache.is_empty());

    // clear() is the escape hatch: the cache lets go, the caller's Arc stays
    // alive, and the next get re-opens.
    cache.clear();
    assert!(cache.is_empty());
    assert_eq!(d1.file_count(), 2, "still usable after clear");
    let d3 = cache.get(&[tmp.path()], true, ".pod5").unwrap();
    assert!(!Arc::ptr_eq(&d1, &d3), "clear must force a re-open");
}
