//! Integration tests for [`ReaderCache`] / [`cached_reader`].
//!
//! The cache exists so that the read-id index is built once per *file* instead
//! of once per *reader* (escapepod-rs#258), and its value is in the ordering
//! and the failure semantics rather than in the map. So these pin: identity
//! (one `Arc` per file, whatever the spelling), warmth (the index is built
//! *before* the entry is published), survivability (a bad path is an error but
//! not a poisoned cache; an un-indexable file is still a usable reader), and
//! the concurrency the whole design is for — exercised with threads, not
//! asserted by construction.
//!
//! Every fixture is written by the test itself, so all of this runs in CI
//! (`ext/` is an empty submodule there).

mod common;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use escapepod_pod5::sidecar::sidecar_path;
use escapepod_pod5::{Reader, ReaderCache, cached_reader, global_reader_cache};
use tempfile::TempDir;

use common::write_fixture;

const N_READS: usize = 20;

/// A temp dir holding `name` with `n_reads` reads.
fn fixture(tmp: &TempDir, name: &str, n_reads: usize) -> std::path::PathBuf {
    let path = tmp.path().join(name);
    write_fixture(&path, "cache_acq", n_reads, 400);
    path
}

/// Run `body` on a scratch thread and fail (rather than hang forever) if it has
/// not finished within `secs`. A deadlock in the cache is the failure mode the
/// design is built to avoid, so it has to be a test *failure*, not a stuck run.
fn with_deadline<F: FnOnce() + Send + 'static>(secs: u64, body: F) {
    let (tx, rx) = std::sync::mpsc::channel();
    let handle = std::thread::spawn(move || {
        body();
        let _ = tx.send(());
    });
    use std::sync::mpsc::RecvTimeoutError;
    match rx.recv_timeout(Duration::from_secs(secs)) {
        // Disconnected means the worker panicked before sending: re-raise its
        // panic rather than blaming a timeout that did not happen.
        Ok(()) | Err(RecvTimeoutError::Disconnected) => {
            handle.join().expect("worker thread panicked")
        }
        Err(RecvTimeoutError::Timeout) => {
            panic!("timed out after {secs}s — the cache deadlocked")
        }
    }
}

#[test]
fn same_path_is_one_reader_and_different_paths_are_not() {
    let tmp = TempDir::new().unwrap();
    let a = fixture(&tmp, "a.pod5", N_READS);
    let b = fixture(&tmp, "b.pod5", N_READS + 1);

    let cache = ReaderCache::new();
    assert!(cache.is_empty());

    let a1 = cache.get(&a).unwrap();
    let a2 = cache.get(&a).unwrap();
    let b1 = cache.get(&b).unwrap();

    assert!(Arc::ptr_eq(&a1, &a2), "same file must be the same reader");
    assert!(
        !Arc::ptr_eq(&a1, &b1),
        "different files must be different readers"
    );
    assert_eq!(cache.len(), 2);
    assert!(!cache.is_empty());

    // `clear` is the escape hatch for the no-eviction policy: the cache lets go,
    // the caller's Arc stays alive, and the next get re-opens.
    cache.clear();
    assert!(cache.is_empty());
    assert_eq!(
        a1.read_count().unwrap(),
        N_READS,
        "still usable after clear"
    );
    let a3 = cache.get(&a).unwrap();
    assert!(!Arc::ptr_eq(&a1, &a3), "clear must force a re-open");
}

#[test]
fn spellings_of_one_file_collapse_to_one_entry() {
    let tmp = TempDir::new().unwrap();
    let path = fixture(&tmp, "reads.pod5", N_READS);
    let cache = ReaderCache::new();

    let direct = cache.get(&path).unwrap();

    // `./reads.pod5` and `sub/../reads.pod5` — same file, different spelling.
    let dotted = tmp.path().join("./reads.pod5");
    assert!(Arc::ptr_eq(&direct, &cache.get(&dotted).unwrap()));

    std::fs::create_dir(tmp.path().join("sub")).unwrap();
    let via_parent = tmp.path().join("sub/../reads.pod5");
    assert!(Arc::ptr_eq(&direct, &cache.get(&via_parent).unwrap()));

    // …and a symlink to it.
    #[cfg(unix)]
    {
        let link = tmp.path().join("link.pod5");
        std::os::unix::fs::symlink(&path, &link).unwrap();
        assert!(Arc::ptr_eq(&direct, &cache.get(&link).unwrap()));
    }

    assert_eq!(cache.len(), 1, "one file, one entry, one index");
}

#[test]
fn index_is_warm_before_the_reader_is_handed_out() {
    let tmp = TempDir::new().unwrap();
    let path = fixture(&tmp, "reads.pod5", N_READS);

    // Control: a plain reader is cold, so `read_index_if_built` is not simply
    // answering "yes" to everything.
    let plain = Reader::open(&path).unwrap();
    assert!(
        plain.read_index_if_built().is_none(),
        "a freshly opened reader must be cold"
    );

    let cache = ReaderCache::new();
    let reader = cache.get(&path).unwrap();
    let index = reader
        .read_index_if_built()
        .expect("the cache must warm the index before publishing the entry");
    assert_eq!(index.len(), N_READS);
}

#[test]
fn concurrent_gets_never_deadlock_and_yield_one_entry_per_path() {
    const THREADS: usize = 24;
    const ROUNDS: usize = 40;

    with_deadline(120, || {
        let tmp = TempDir::new().unwrap();
        let paths: Vec<_> = (0..4)
            .map(|i| fixture(&tmp, &format!("f{i}.pod5"), N_READS + i))
            .collect();

        let cache = ReaderCache::new();
        // One canonical Arc per path, claimed by whichever thread wins the
        // race; every other thread must observe exactly that pointer.
        let winners: Vec<_> = paths
            .iter()
            .map(|_| std::sync::Mutex::new(None::<Arc<Reader>>))
            .collect();
        let observed = AtomicUsize::new(0);

        std::thread::scope(|scope| {
            for t in 0..THREADS {
                let cache = &cache;
                let winners = &winners;
                let observed = &observed;
                let paths = &paths;
                scope.spawn(move || {
                    for round in 0..ROUNDS {
                        // Different threads start on different files so opens
                        // genuinely overlap across paths, and every thread
                        // still hammers every path.
                        let i = (t + round) % paths.len();
                        let reader = cache.get(&paths[i]).unwrap();
                        assert_eq!(reader.read_count().unwrap(), N_READS + i);
                        assert!(
                            reader.read_index_if_built().is_some(),
                            "a reader handed out by the cache is always warm"
                        );
                        let mut slot = winners[i].lock().unwrap();
                        match slot.as_ref() {
                            Some(first) => assert!(
                                Arc::ptr_eq(first, &reader),
                                "path {i} produced two live readers"
                            ),
                            None => *slot = Some(reader),
                        }
                        observed.fetch_add(1, Ordering::Relaxed);
                    }
                });
            }
        });

        assert_eq!(observed.load(Ordering::Relaxed), THREADS * ROUNDS);
        assert_eq!(
            cache.len(),
            paths.len(),
            "one entry per path, no duplicates"
        );
    });
}

#[test]
fn a_bad_path_errors_without_poisoning_the_cache() {
    let tmp = TempDir::new().unwrap();
    let good = fixture(&tmp, "good.pod5", N_READS);

    let missing = tmp.path().join("nope.pod5");
    let garbage = tmp.path().join("garbage.pod5");
    std::fs::write(&garbage, vec![b'x'; 512]).unwrap();

    let cache = ReaderCache::new();
    assert!(cache.get(&missing).is_err(), "missing file must error");
    assert!(cache.get(&garbage).is_err(), "non-POD5 file must error");
    assert!(cache.is_empty(), "a failed open must not leave an entry");

    // The failures are not sticky, in either direction.
    let reader = cache.get(&good).unwrap();
    assert_eq!(reader.read_count().unwrap(), N_READS);
    assert!(cache.get(&missing).is_err());
    assert!(Arc::ptr_eq(&reader, &cache.get(&good).unwrap()));
    assert_eq!(cache.len(), 1);
}

#[test]
fn an_unindexable_file_still_yields_a_usable_reader() {
    let tmp = TempDir::new().unwrap();
    let donor = fixture(&tmp, "donor.pod5", N_READS + 5);
    let target = fixture(&tmp, "target.pod5", N_READS);

    // A sidecar bound to *another* POD5 makes `read_index()` fail loudly rather
    // than fall back to a scan (that is the escapepod-rs#251 rule). It is the
    // cheapest un-indexable file there is.
    Reader::open(&donor)
        .unwrap()
        .build_and_write_index(sidecar_path(&target))
        .unwrap();
    assert!(
        Reader::open(&target).unwrap().read_index().is_err(),
        "fixture must actually be un-indexable"
    );

    let cache = ReaderCache::new();
    let reader = cache
        .get(&target)
        .expect("a failed index warm-up must not fail the open");

    assert!(
        reader.read_index_if_built().is_none(),
        "the warm-up failed, so nothing should be cached"
    );
    assert_eq!(reader.read_count().unwrap(), N_READS);
    assert_eq!(
        reader.reads().unwrap().filter_map(|r| r.ok()).count(),
        N_READS,
        "iteration does not need the index"
    );
    // The caller that actually demands a lookup still sees the error.
    assert!(reader.read_index().is_err());
}

#[test]
fn cached_reader_shares_one_reader_per_process() {
    let tmp = TempDir::new().unwrap();
    let path = fixture(&tmp, "global.pod5", N_READS);

    let first = cached_reader(&path).unwrap();
    let second = cached_reader(tmp.path().join("./global.pod5")).unwrap();

    assert!(Arc::ptr_eq(&first, &second));
    assert!(first.read_index_if_built().is_some());
    assert!(!global_reader_cache().is_empty());
    assert!(cached_reader(tmp.path().join("missing.pod5")).is_err());
}
