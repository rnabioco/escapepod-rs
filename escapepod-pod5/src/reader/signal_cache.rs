//! LRU cache for signal record batches.

use arrow::record_batch::RecordBatch;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

/// Metadata about signal table batches for efficient lookup.
///
/// Stores cumulative row counts so signal-row → (batch_idx, local_row) lookups
/// work correctly even when batches have non-uniform sizes (e.g. files produced
/// by `merge_files`, which concatenates each source file's signal batches
/// verbatim and so inherits whatever row counts those sources used).
pub(super) struct SignalBatchMetadata {
    /// Cumulative row counts. Length is `num_batches + 1`; `cumulative_rows[i]`
    /// is the first signal row belonging to batch `i`, and `cumulative_rows[n]`
    /// is the total row count.
    pub cumulative_rows: Vec<u64>,
}

impl SignalBatchMetadata {
    /// Number of batches described by this metadata.
    pub fn num_batches(&self) -> usize {
        self.cumulative_rows.len().saturating_sub(1)
    }

    /// Locate the `(batch_idx, local_row)` for an absolute signal row.
    ///
    /// Uses binary search over `cumulative_rows`. Returns `None` when `row`
    /// is past the end of the table.
    pub fn locate(&self, row: u64) -> Option<(usize, usize)> {
        if self.cumulative_rows.len() < 2 {
            return None;
        }
        let total = *self.cumulative_rows.last().unwrap();
        if row >= total {
            return None;
        }
        let batch_idx = match self.cumulative_rows.binary_search(&row) {
            Ok(i) => i,
            Err(i) => i - 1,
        };
        let local = row - self.cumulative_rows[batch_idx];
        Some((batch_idx, local as usize))
    }
}

/// A cached signal batch with access tracking for LRU eviction.
struct CachedSignalBatch {
    batch: RecordBatch,
    // AtomicU64 so cache hits can touch access time under a shared read lock.
    last_access: AtomicU64,
}

/// LRU cache for signal batches.
pub(super) struct SignalBatchCache {
    /// Cached batches indexed by batch number.
    batches: HashMap<usize, CachedSignalBatch>,
    /// Maximum number of batches to cache.
    max_size: usize,
    /// Access counter for LRU tracking.
    access_counter: AtomicU64,
}

impl SignalBatchCache {
    /// Create a new signal batch cache with the given maximum size.
    pub fn new(max_size: usize) -> Self {
        Self {
            batches: HashMap::with_capacity(max_size),
            max_size,
            access_counter: AtomicU64::new(0),
        }
    }

    /// Get a batch from the cache, updating access time.
    ///
    /// Takes `&self` so cache hits can proceed under a shared read lock —
    /// `AtomicU64` makes the access-time update safe without exclusion.
    pub fn get(&self, batch_idx: usize) -> Option<&RecordBatch> {
        let cached = self.batches.get(&batch_idx)?;
        let tick = self.access_counter.fetch_add(1, Ordering::Relaxed) + 1;
        cached.last_access.store(tick, Ordering::Relaxed);
        Some(&cached.batch)
    }

    /// Insert a batch into the cache, evicting old entries if necessary.
    pub fn insert(&mut self, batch_idx: usize, batch: RecordBatch) {
        // Evict if at capacity
        if self.batches.len() >= self.max_size && !self.batches.contains_key(&batch_idx) {
            self.evict_oldest();
        }

        let tick = self.access_counter.fetch_add(1, Ordering::Relaxed) + 1;
        self.batches.insert(
            batch_idx,
            CachedSignalBatch {
                batch,
                last_access: AtomicU64::new(tick),
            },
        );
    }

    /// Evict approximately 20% of the oldest entries (like C++ implementation).
    fn evict_oldest(&mut self) {
        if self.batches.is_empty() {
            return;
        }

        let to_evict = std::cmp::max(1, self.batches.len() / 5);

        // Collect entries sorted by access time
        let mut entries: Vec<_> = self
            .batches
            .iter()
            .map(|(&idx, cached)| (idx, cached.last_access.load(Ordering::Relaxed)))
            .collect();
        entries.sort_by_key(|&(_, access)| access);

        // Remove oldest entries
        for (idx, _) in entries.into_iter().take(to_evict) {
            self.batches.remove(&idx);
        }
    }
}
