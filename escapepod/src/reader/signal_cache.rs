//! LRU cache for signal record batches.

use arrow::record_batch::RecordBatch;
use std::collections::HashMap;

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
    last_access: u64,
}

/// LRU cache for signal batches.
pub(super) struct SignalBatchCache {
    /// Cached batches indexed by batch number.
    batches: HashMap<usize, CachedSignalBatch>,
    /// Maximum number of batches to cache.
    max_size: usize,
    /// Access counter for LRU tracking.
    access_counter: u64,
}

impl SignalBatchCache {
    /// Create a new signal batch cache with the given maximum size.
    pub fn new(max_size: usize) -> Self {
        Self {
            batches: HashMap::with_capacity(max_size),
            max_size,
            access_counter: 0,
        }
    }

    /// Get a batch from the cache, updating access time.
    pub fn get(&mut self, batch_idx: usize) -> Option<&RecordBatch> {
        if let Some(cached) = self.batches.get_mut(&batch_idx) {
            self.access_counter += 1;
            cached.last_access = self.access_counter;
            Some(&cached.batch)
        } else {
            None
        }
    }

    /// Insert a batch into the cache, evicting old entries if necessary.
    pub fn insert(&mut self, batch_idx: usize, batch: RecordBatch) {
        // Evict if at capacity
        if self.batches.len() >= self.max_size && !self.batches.contains_key(&batch_idx) {
            self.evict_oldest();
        }

        self.access_counter += 1;
        self.batches.insert(
            batch_idx,
            CachedSignalBatch {
                batch,
                last_access: self.access_counter,
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
            .map(|(&idx, cached)| (idx, cached.last_access))
            .collect();
        entries.sort_by_key(|&(_, access)| access);

        // Remove oldest entries
        for (idx, _) in entries.into_iter().take(to_evict) {
            self.batches.remove(&idx);
        }
    }
}
