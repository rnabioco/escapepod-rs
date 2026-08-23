//! A cache of open, indexed readers, keyed by file path.
//!
//! [`Reader`] caches its read-id index in a `OnceLock` on the *instance*, so
//! the index is per-reader, not per-file. A consumer that opens a reader per
//! batch therefore throws the index away and rebuilds it on the next batch,
//! and that is not a small constant: on a 145 GB POD5 on a network filesystem
//! it was minutes of uninterruptible sleep in `folio_wait_bit_common` per
//! batch at ~0.6% of one core (the 10–80x data-preparation regression in
//! rnabioco/leech#176). escapepod-rs#251 fixed the other half of that — the
//! scan variants are gone and lookups route through [`Reader::read_index`]
//! unconditionally — but "one reader per file per process" was left to every
//! consumer, and each one that did not write it silently got the slow path.
//! leech wrote it in Rust and then again, independently, in Python.
//!
//! This module is that missing half. The interesting part is not the `static`
//! behind [`cached_reader`]; it is the *ordering* (the index is warmed before
//! the entry is published, so N worker threads do not pile up inside one lazy
//! init on their first batch) and the *failure semantics* (a failed warm-up is
//! not a failed batch). Both are spelled out on [`ReaderCache::get`].

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

use super::file_reader::{Reader, autoindex_max};
use crate::error::Result;

/// A cache of open, indexed readers, keyed by file path.
///
/// Hand this out once per process (see [`cached_reader`] for a ready-made
/// process-global one) or own one per pipeline stage. Every
/// [`get`](ReaderCache::get) for the same file returns the same `Arc<Reader>`,
/// so the read-id index is built once per *file* rather than once per
/// *reader*.
///
/// # Resident cost
///
/// What stays resident is the index, not the file: ~24 bytes/read (16-byte
/// UUID + batch + row), so even a multi-million-read POD5 costs a few tens of
/// MB. The POD5 itself is memory-mapped — its pages are the page cache's
/// business and the kernel can reclaim them.
///
/// Entries are **never evicted**. That is deliberate for the workload this
/// exists for (a fixed set of POD5s traversed batch by batch), and
/// [`clear`](ReaderCache::clear) is the escape hatch for a process that walks
/// an unbounded set of files. Clearing drops the cache's handles; readers a
/// caller is still holding stay alive until that caller drops them.
///
/// # Example
///
/// ```no_run
/// use escapepod_pod5::ReaderCache;
///
/// let cache = ReaderCache::new();
/// let reader = cache.get("reads.pod5")?;
/// // Same file, different spelling — same reader, index already warm.
/// let again = cache.get("./reads.pod5")?;
/// assert!(std::sync::Arc::ptr_eq(&reader, &again));
/// # Ok::<(), escapepod_pod5::Error>(())
/// ```
#[derive(Default)]
pub struct ReaderCache {
    /// Path → open reader.
    ///
    /// The mutex guards **only the map**. No `Reader::open` and no index build
    /// ever happens while it is held (see [`ReaderCache::get`]), so a slow open
    /// on one path cannot block a lookup on another and the lock is never held
    /// across I/O.
    entries: Mutex<HashMap<PathBuf, Arc<Reader>>>,
}

impl ReaderCache {
    /// An empty cache.
    pub fn new() -> Self {
        Self::default()
    }

    /// The shared reader for `path`, opening and indexing it if this is the
    /// first time this cache has seen it.
    ///
    /// Three properties, in the order they matter:
    ///
    /// 1. **The file is opened outside the lock.** A slow open on one path
    ///    never blocks a lookup on another, and the lock is never held across
    ///    I/O, so this cannot deadlock. Two threads racing on the same path
    ///    cost one redundant open — the loser's reader is dropped and both get
    ///    the winner's `Arc`. That trade is intentional: a redundant open is
    ///    cheap and bounded, whereas serializing opens behind one held lock (or
    ///    one per-cache `OnceLock`) makes every path wait on the slowest.
    ///
    /// 2. **The index is warmed before the entry is published.** By the time
    ///    another thread can observe the entry, [`Reader::read_index`] has
    ///    already run, so N workers hitting their first batch together find a
    ///    built index instead of piling up inside one lazy init.
    ///
    ///    The warm-up respects [`autoindex_max`]: above that read count it is
    ///    skipped, because warming is a *guess* that random access is coming
    ///    and a huge file that is only iterated should not pay for an index
    ///    nobody asked for. Skipping only defers the build to the first lookup
    ///    that genuinely demands one (which still indexes, loudly, rather than
    ///    scanning — escapepod-rs#251), and because the reader is shared that
    ///    build still happens once per file, not once per batch. The cache
    ///    keeps its whole value above the threshold; it just stops guessing.
    ///
    /// 3. **Warm-up failure is not propagated.** `Reader::open` failing *is* an
    ///    error — there is no reader to hand back. A failed *index* build is
    ///    logged and otherwise ignored: the reader is still perfectly usable
    ///    for iteration, metadata, and signal access, and failing an open for a
    ///    caller that may never do a lookup is worse than the slowdown. A
    ///    caller that does demand a lookup gets the same error then, from the
    ///    call that needs it.
    ///
    /// Keys are **canonicalized** ([`std::fs::canonicalize`]), so `reads.pod5`,
    /// `./reads.pod5`, and a symlink to it are one entry rather than three
    /// readers with three indexes. If canonicalization fails the path is used
    /// as given, which at worst costs a duplicate entry. The reader is opened
    /// on the canonical path too, so `.p5s` sidecar resolution does not depend
    /// on which spelling happened to arrive first.
    pub fn get<P: AsRef<Path>>(&self, path: P) -> Result<Arc<Reader>> {
        let key = cache_key(path.as_ref());

        if let Some(hit) = self.lock().get(&key).cloned() {
            return Ok(hit);
        }

        // Outside the lock, deliberately — see (1) above.
        let reader = Reader::open(&key)?;
        warm_index(&reader, &key);

        // Re-lock only to publish. A concurrent winner keeps its entry and our
        // reader is dropped here — `entry`, not `insert`, so a race can never
        // leave two live readers for one file.
        Ok(self
            .lock()
            .entry(key)
            .or_insert_with(|| Arc::new(reader))
            .clone())
    }

    /// Number of files currently held open.
    pub fn len(&self) -> usize {
        self.lock().len()
    }

    /// Whether the cache holds no readers.
    pub fn is_empty(&self) -> bool {
        self.lock().is_empty()
    }

    /// Drop every cached reader.
    ///
    /// The escape hatch for the no-eviction policy. Readers other code is still
    /// holding stay alive until those `Arc`s drop; only the cache's own handles
    /// go, so the next [`get`](ReaderCache::get) re-opens and re-indexes.
    pub fn clear(&self) {
        self.lock().clear();
    }

    /// Lock the map, recovering from poisoning.
    ///
    /// The critical sections are a `HashMap` lookup, insert, and clear, none of
    /// which can panic partway and leave a torn map. A poisoned lock therefore
    /// means some *unrelated* panic happened elsewhere in the process while a
    /// guard was alive, and refusing every future lookup over it would turn one
    /// panic into a permanently disabled cache.
    fn lock(&self) -> MutexGuard<'_, HashMap<PathBuf, Arc<Reader>>> {
        self.entries.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// The key a path is filed under: its canonical form, or the path as given if
/// it cannot be canonicalized (it does not exist, a component is not readable,
/// …). Falling back rather than failing keeps the error where it belongs — in
/// `Reader::open`, which produces a far better message than `canonicalize`
/// would.
fn cache_key(path: &Path) -> PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

/// Build the read-id index now, before the reader is shared, so concurrent
/// first-batch lookups find it built.
///
/// Best-effort by design: see property (3) on [`ReaderCache::get`].
fn warm_index(reader: &Reader, path: &Path) {
    let reads = reader.read_count().unwrap_or(usize::MAX);
    if reads > autoindex_max() {
        tracing::debug!(
            file = ?path,
            reads,
            "not warming the read index for a large file; the first lookup that \
             needs it will build it (`escpod index` persists it across processes)"
        );
        return;
    }
    if let Err(e) = reader.read_index() {
        // Not propagated: the reader is still usable for everything that does
        // not need the index, and a caller that demands a lookup will see this
        // same error from the call that demands it.
        tracing::warn!(
            file = ?path,
            error = %e,
            "could not build the read-id index; the reader is cached anyway and \
             lookups by read id will report this again"
        );
    }
}

/// The process-global [`ReaderCache`].
///
/// The `OnceLock` guards only the construction of an *empty* map — never an
/// open — so it cannot serialize anything.
static GLOBAL_CACHE: OnceLock<ReaderCache> = OnceLock::new();

/// The process-global [`ReaderCache`], for reaching
/// [`clear`](ReaderCache::clear) / [`len`](ReaderCache::len) on the cache
/// [`cached_reader`] uses.
pub fn global_reader_cache() -> &'static ReaderCache {
    GLOBAL_CACHE.get_or_init(ReaderCache::new)
}

/// Shared reader for `path`, opened and indexed once per process.
///
/// The convenience shape of [`ReaderCache::get`] — same ordering, same failure
/// semantics, same canonicalized keys, against a process-global cache. It
/// exists because the alternative is what every consumer of batched
/// [`Reader::reads_by_ids`] currently writes for itself, and the ones that do
/// not write it get an index rebuild per batch instead.
///
/// Own a [`ReaderCache`] instead when a library needs its lifetime bounded, or
/// when one part of a process must not share readers with another.
///
/// # Example
///
/// ```no_run
/// use escapepod_pod5::cached_reader;
///
/// // In a per-batch worker: the open and the index build happen on the first
/// // batch only; every later batch gets the warm reader.
/// let reader = cached_reader("reads.pod5")?;
/// println!("{} reads", reader.read_count()?);
/// # Ok::<(), escapepod_pod5::Error>(())
/// ```
pub fn cached_reader<P: AsRef<Path>>(path: P) -> Result<Arc<Reader>> {
    global_reader_cache().get(path)
}
