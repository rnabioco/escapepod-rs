//! A collection of POD5 files presented as one logical dataset.
//!
//! [`ReaderCache`](super::ReaderCache) solved "one indexed reader per file per
//! process"; this module solves the same problem one level up, for a
//! directory of files. A MinKNOW run is never one POD5 — every Rust consumer
//! that wants to treat a run directory as a single random-access-by-read-id
//! source had to write the directory walk and the read-id → owning-file
//! routing map itself, and until now the only place that logic existed was
//! `escapepod-python`'s `PyDatasetReader`, which is pyo3-specific and
//! unreachable from a plain Rust dependency on this crate (escapepod-rs#384).
//!
//! [`Dataset`] is that missing piece: open a file, a directory, or a mix of
//! both as one collection, and get bulk-decode and read-id routing across
//! every file in it. It never opens a file itself — every entry is fetched
//! through [`cached_reader`], the *same* process-global [`ReaderCache`] a
//! direct caller would use, so a file reachable both directly and via a
//! directory scan shares one `Arc<Reader>` and one warmed index rather than
//! two independent copies.
//!
//! [`DatasetCache`] and [`cached_dataset`] are the dataset-level analogue of
//! `ReaderCache`/`cached_reader`: same ordering guarantees, same failure
//! semantics, same canonicalized-key story — see [`DatasetCache::get`].

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

use super::cache::cache_key;
use super::cached_reader;
use super::file_reader::Reader;
use crate::error::{Error, Result};
use crate::types::{ReadData, RunInfoData, Uuid};

/// Suffix used to match files inside a scanned directory when the caller
/// does not name one — `*.pod5` with the glob stripped to a plain suffix.
const DEFAULT_SUFFIX: &str = ".pod5";

/// Recursively collect POD5 files under `path` into `out`.
///
/// A path that is an explicit file is included regardless of `suffix` (the
/// caller named it directly); directories are scanned for entries whose file
/// name ends with `suffix`, descending into subdirectories only when
/// `recursive` is set.
fn collect_pod5(
    path: &Path,
    recursive: bool,
    suffix: &str,
    out: &mut Vec<PathBuf>,
) -> std::io::Result<()> {
    if path.is_dir() {
        for entry in fs::read_dir(path)? {
            let p = entry?.path();
            if p.is_dir() {
                if recursive {
                    collect_pod5(&p, recursive, suffix, out)?;
                }
            } else if p
                .file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.ends_with(suffix))
            {
                out.push(p);
            }
        }
    } else {
        // Explicit file path — include it even if it doesn't match the suffix.
        out.push(path.to_path_buf());
    }
    Ok(())
}

/// A collection of POD5 files presented as one logical dataset.
///
/// Accepts a single file, a directory (scanned for a suffix, `.pod5` by
/// default), or a mixed list of files and directories, and presents the reads
/// across every file as a single, random-access-by-read-id source — the
/// escapepod analogue of `pod5.DatasetReader`.
///
/// Every file is opened through [`cached_reader`], never through a second,
/// dataset-private cache, so a `Dataset` and a direct [`cached_reader`] call
/// on one of its files always share the same `Arc<Reader>`.
///
/// # Example
///
/// ```no_run
/// use escapepod_pod5::Dataset;
///
/// let dataset = Dataset::open("run_dir/")?;
/// println!("{} files, {} reads", dataset.file_count(), dataset.read_count()?);
/// # Ok::<(), escapepod_pod5::Error>(())
/// ```
pub struct Dataset {
    /// File path → its shared reader, in sorted, deduplicated file order.
    files: Vec<(PathBuf, Arc<Reader>)>,
    /// Lazily built map from read UUID to the index of its owning file in
    /// [`Self::files`], used to route bulk signal decode and single-read
    /// lookups to the file that holds a read.
    id_index: OnceLock<HashMap<Uuid, usize>>,
}

impl Dataset {
    /// Open `root` — a single file or a directory — as one dataset.
    ///
    /// A directory is scanned recursively for files whose name ends with
    /// `.pod5`. Use [`Dataset::open_with`] for a mixed file/directory list, a
    /// non-recursive scan, or a different suffix.
    pub fn open<P: AsRef<Path>>(root: P) -> Result<Self> {
        Self::open_with(std::slice::from_ref(&root), true, DEFAULT_SUFFIX)
    }

    /// Open `roots` — files and/or directories — as one dataset.
    ///
    /// Each directory in `roots` is scanned for entries whose file name ends
    /// with `suffix`, descending into subdirectories only when `recursive` is
    /// set; an explicit file is always included regardless of `suffix`. The
    /// resolved file list is sorted and deduplicated, so a file reachable
    /// both directly (in `roots`) and via a directory scan appears once.
    ///
    /// Errors if `roots` resolves to zero files — a directory of the wrong
    /// shape is a configuration mistake, not an empty-but-successful dataset.
    pub fn open_with<P: AsRef<Path>>(roots: &[P], recursive: bool, suffix: &str) -> Result<Self> {
        let mut files = Vec::new();
        for root in roots {
            collect_pod5(root.as_ref(), recursive, suffix, &mut files).map_err(|e| {
                Error::Io(std::io::Error::new(
                    e.kind(),
                    format!("scanning {}: {e}", root.as_ref().display()),
                ))
            })?;
        }
        files.sort();
        files.dedup();

        if files.is_empty() {
            let roots: Vec<String> = roots
                .iter()
                .map(|p| p.as_ref().display().to_string())
                .collect();
            return Err(Error::InvalidState(format!(
                "no POD5 files found matching '{suffix}' in: {roots:?}"
            )));
        }

        let mut entries = Vec::with_capacity(files.len());
        for file in files {
            let reader = cached_reader(&file)?;
            entries.push((file, reader));
        }

        Ok(Self {
            files: entries,
            id_index: OnceLock::new(),
        })
    }

    /// Paths of the POD5 files in this dataset, in sorted order.
    pub fn paths(&self) -> Vec<&Path> {
        self.files.iter().map(|(p, _)| p.as_path()).collect()
    }

    /// Number of POD5 files in the dataset.
    pub fn file_count(&self) -> usize {
        self.files.len()
    }

    /// The shared reader for the file at `index`, in the same order as
    /// [`Self::paths`] — for a caller that wants to walk the dataset file by
    /// file (e.g. a streaming iterator over every read) rather than route by
    /// read id.
    pub fn reader_at(&self, index: usize) -> Option<&Arc<Reader>> {
        self.files.get(index).map(|(_, r)| r)
    }

    /// Total number of reads across every file.
    pub fn read_count(&self) -> Result<usize> {
        let mut total = 0;
        for (_, reader) in &self.files {
            total += reader.read_count()?;
        }
        Ok(total)
    }

    /// All run info records across the dataset, deduplicated by acquisition id.
    pub fn run_infos(&self) -> Vec<RunInfoData> {
        let mut seen = HashSet::new();
        let mut out = Vec::new();
        for (_, reader) in &self.files {
            for ri in reader.run_infos() {
                if seen.insert(ri.acquisition_id.clone()) {
                    out.push(ri.clone());
                }
            }
        }
        out
    }

    /// All read IDs across every file.
    pub fn read_ids(&self) -> Result<Vec<Uuid>> {
        let mut out = Vec::new();
        for (_, reader) in &self.files {
            out.extend(reader.read_ids()?);
        }
        Ok(out)
    }

    /// Reads across the dataset whose id is in `target_ids`.
    ///
    /// Looks each file's matches up via [`Reader::reads_by_ids`]; ids absent
    /// from every file are simply not present in the result — the same
    /// contract a single [`Reader::reads_by_ids`] call has.
    pub fn reads_by_ids(&self, target_ids: &HashSet<Uuid>) -> Result<Vec<ReadData>> {
        let mut out = Vec::new();
        for (_, reader) in &self.files {
            out.extend(reader.reads_by_ids(target_ids)?);
        }
        Ok(out)
    }

    /// The shared reader that owns `id`, or `None` if `id` is not part of
    /// this dataset (or its routing index could not be built).
    ///
    /// Builds (and caches) the dataset's read-id → owning-file index on first
    /// call; a [`Dataset`] obtained via [`cached_dataset`] already has it warm.
    pub fn owning_reader(&self, id: &Uuid) -> Option<&Arc<Reader>> {
        let idx = *self.id_index().ok()?.get(id)?;
        Some(&self.files[idx].1)
    }

    /// Decode signal for many reads, one bulk call per owning file.
    ///
    /// Buckets `reads` by the file that owns each one, calls
    /// [`Reader::get_signal_bulk_prefix`] once per file, and restores the
    /// caller's input order — the shape a batched consumer wants instead of
    /// one [`Reader::get_signal_prefix`] call per read. `max_samples` decodes
    /// at most that many leading samples of each read, as on
    /// [`Reader::get_signal_prefix`].
    ///
    /// Errors with [`Error::ReadNotFound`] if any read's id is not part of
    /// this dataset.
    pub fn decode_bulk(
        &self,
        reads: &[ReadData],
        max_samples: usize,
    ) -> Result<Vec<(Uuid, Vec<i16>)>> {
        let index = self.id_index()?;

        // Bucket (original position, id, signal_rows) by owning file.
        let mut buckets: HashMap<usize, Vec<(usize, Uuid, Vec<u64>)>> = HashMap::new();
        for (pos, r) in reads.iter().enumerate() {
            let idx = *index
                .get(&r.read_id)
                .ok_or(Error::ReadNotFound(r.read_id))?;
            buckets
                .entry(idx)
                .or_default()
                .push((pos, r.read_id, r.signal_rows.clone()));
        }

        let mut ordered: Vec<Option<(Uuid, Vec<i16>)>> = (0..reads.len()).map(|_| None).collect();
        for (file_idx, items) in &buckets {
            let inputs: Vec<(usize, Vec<u64>)> = items
                .iter()
                .map(|(pos, _, rows)| (*pos, rows.clone()))
                .collect();
            let results = self.files[*file_idx]
                .1
                .get_signal_bulk_prefix(&inputs, max_samples)?;
            // get_signal_bulk_prefix preserves input order, so zip back to (pos, id).
            for ((pos, id, _), (_, signal)) in items.iter().zip(results) {
                ordered[*pos] = Some((*id, signal));
            }
        }

        Ok(ordered
            .into_iter()
            .map(|o| o.expect("every position was filled by its owning file's bucket"))
            .collect())
    }

    /// The read-id → owning-file-index routing map, building it on first call.
    fn id_index(&self) -> Result<&HashMap<Uuid, usize>> {
        if let Some(map) = self.id_index.get() {
            return Ok(map);
        }
        let mut map = HashMap::new();
        for (i, (_, reader)) in self.files.iter().enumerate() {
            for id in reader.read_ids()? {
                map.insert(id, i);
            }
        }
        // Ignore the error case: another thread won the race and set it
        // first, which is fine — either map is equivalent.
        let _ = self.id_index.set(map);
        Ok(self.id_index.get().unwrap())
    }
}

/// A cache of open datasets, keyed by the canonicalized root path(s) each was
/// opened from.
///
/// The dataset-level mirror of [`ReaderCache`](super::ReaderCache): every
/// [`get`](DatasetCache::get) for the same roots (and the same `recursive`,
/// `suffix`) returns the same `Arc<Dataset>`, and every file inside it is
/// opened through the process-global [`ReaderCache`] via [`cached_reader`] —
/// never a second, dataset-private cache.
///
/// # Example
///
/// ```no_run
/// use escapepod_pod5::DatasetCache;
///
/// let cache = DatasetCache::new();
/// let dataset = cache.get(&["run_dir/"], true, ".pod5")?;
/// // Same roots, different spelling — same dataset, index already warm.
/// let again = cache.get(&["./run_dir/"], true, ".pod5")?;
/// assert!(std::sync::Arc::ptr_eq(&dataset, &again));
/// # Ok::<(), escapepod_pod5::Error>(())
/// ```
#[derive(Default)]
pub struct DatasetCache {
    /// Key (canonicalized roots + scan settings) → open dataset.
    ///
    /// The mutex guards **only the map**, mirroring
    /// [`ReaderCache`](super::ReaderCache): no directory scan, no file open,
    /// and no index build ever happens while it is held (see
    /// [`DatasetCache::get`]).
    entries: Mutex<HashMap<DatasetKey, Arc<Dataset>>>,
}

/// The key a `(roots, recursive, suffix)` combination is filed under: each
/// root canonicalized (or used as given if that fails), sorted and
/// deduplicated so root order and spelling don't change identity.
#[derive(Clone, PartialEq, Eq, Hash)]
struct DatasetKey {
    roots: Vec<PathBuf>,
    recursive: bool,
    suffix: String,
}

impl DatasetKey {
    fn new<P: AsRef<Path>>(roots: &[P], recursive: bool, suffix: &str) -> Self {
        let mut roots: Vec<PathBuf> = roots.iter().map(|p| cache_key(p.as_ref())).collect();
        roots.sort();
        roots.dedup();
        Self {
            roots,
            recursive,
            suffix: suffix.to_string(),
        }
    }
}

impl DatasetCache {
    /// An empty cache.
    pub fn new() -> Self {
        Self::default()
    }

    /// The shared dataset for `roots`, opening it (and warming its read-id
    /// routing index) if this is the first time this cache has seen this
    /// combination of roots, `recursive`, and `suffix`.
    ///
    /// Mirrors [`ReaderCache::get`](super::ReaderCache::get)'s three
    /// properties, one level up:
    ///
    /// 1. **The dataset is built outside the lock.** [`Dataset::open_with`]
    ///    — including every per-file open, which itself never blocks on
    ///    another path (see [`ReaderCache::get`](super::ReaderCache::get)) —
    ///    runs with no lock held, so a slow scan or open on one set of roots
    ///    never blocks a lookup on another.
    /// 2. **The routing index is warmed before the entry is published.** By
    ///    the time another thread can observe the entry,
    ///    [`Dataset::owning_reader`]'s backing map has already been built, so
    ///    concurrent first lookups find it warm instead of piling up inside
    ///    one lazy init.
    /// 3. **Warm-up failure is not propagated.** A failed [`Dataset::open_with`]
    ///    *is* an error — there is no dataset to hand back. A failed routing-index
    ///    build is logged and otherwise ignored: the dataset is still usable for
    ///    everything that does not need it (iteration, per-file metadata), and a
    ///    caller that does need routing sees the same error from the call that
    ///    demands it.
    ///
    /// Keys are the **canonicalized** roots ([`std::fs::canonicalize`]), sorted
    /// and deduplicated, so the same roots in a different order or spelling
    /// collapse to one entry.
    pub fn get<P: AsRef<Path>>(
        &self,
        roots: &[P],
        recursive: bool,
        suffix: &str,
    ) -> Result<Arc<Dataset>> {
        let key = DatasetKey::new(roots, recursive, suffix);

        if let Some(hit) = self.lock().get(&key).cloned() {
            return Ok(hit);
        }

        // Outside the lock, deliberately — see (1) above.
        let dataset = Dataset::open_with(roots, recursive, suffix)?;
        warm_id_index(&dataset);

        // Re-lock only to publish. A concurrent winner keeps its entry and
        // our dataset is dropped here.
        Ok(self
            .lock()
            .entry(key)
            .or_insert_with(|| Arc::new(dataset))
            .clone())
    }

    /// Number of datasets currently held open.
    pub fn len(&self) -> usize {
        self.lock().len()
    }

    /// Whether the cache holds no datasets.
    pub fn is_empty(&self) -> bool {
        self.lock().is_empty()
    }

    /// Drop every cached dataset.
    ///
    /// Datasets other code is still holding stay alive until those `Arc`s
    /// drop; the underlying `Arc<Reader>`s they share with [`ReaderCache`](super::ReaderCache)
    /// are unaffected either way — this only forgets the dataset-level grouping.
    pub fn clear(&self) {
        self.lock().clear();
    }

    /// Lock the map, recovering from poisoning — mirrors
    /// [`ReaderCache`](super::ReaderCache)'s own lock helper: the critical
    /// sections here can't panic partway and leave a torn map, so a poisoned
    /// lock means an unrelated panic happened elsewhere while a guard was
    /// alive, and refusing every future lookup over it would turn one panic
    /// into a permanently disabled cache.
    fn lock(&self) -> MutexGuard<'_, HashMap<DatasetKey, Arc<Dataset>>> {
        self.entries.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// Build the dataset's read-id routing index now, before the dataset is
/// shared, so concurrent first lookups find it built.
///
/// Best-effort by design: see property (3) on [`DatasetCache::get`].
fn warm_id_index(dataset: &Dataset) {
    if let Err(e) = dataset.id_index() {
        tracing::warn!(
            error = %e,
            "could not build the dataset's read-id routing index; the dataset is \
             cached anyway and lookups by read id will report this again"
        );
    }
}

/// The process-global [`DatasetCache`].
///
/// The `OnceLock` guards only the construction of an *empty* map — never a
/// scan or an open — so it cannot serialize anything.
static GLOBAL_DATASET_CACHE: OnceLock<DatasetCache> = OnceLock::new();

/// The process-global [`DatasetCache`], for reaching
/// [`clear`](DatasetCache::clear) / [`len`](DatasetCache::len) on the cache
/// [`cached_dataset`] uses.
pub fn global_dataset_cache() -> &'static DatasetCache {
    GLOBAL_DATASET_CACHE.get_or_init(DatasetCache::new)
}

/// Shared dataset for `path` — a single file or directory, scanned
/// recursively for `.pod5` files — opened and indexed once per process.
///
/// The convenience shape of [`DatasetCache::get`] against a process-global
/// cache, for the common case of one root with the default scan settings.
/// Call [`global_dataset_cache`] directly for a mixed roots list, a
/// non-recursive scan, or a different suffix.
///
/// # Example
///
/// ```no_run
/// use escapepod_pod5::cached_dataset;
///
/// // In a per-batch worker: the scan, the opens, and the routing-index build
/// // happen on the first batch only; every later batch gets the warm dataset.
/// let dataset = cached_dataset("run_dir/")?;
/// println!("{} reads", dataset.read_count()?);
/// # Ok::<(), escapepod_pod5::Error>(())
/// ```
pub fn cached_dataset<P: AsRef<Path>>(path: P) -> Result<Arc<Dataset>> {
    global_dataset_cache().get(std::slice::from_ref(&path), true, DEFAULT_SUFFIX)
}
