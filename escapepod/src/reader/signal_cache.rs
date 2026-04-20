//! LRU cache for signal record batches.

use arrow::record_batch::RecordBatch;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

/// Metadata about signal table batches for efficient lookup.
pub(super) struct SignalBatchMetadata {
    /// Number of rows per batch (assumed uniform, determined from batch 0).
    pub batch_size: usize,
    /// Total number of signal batches.
    pub num_batches: usize,
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
