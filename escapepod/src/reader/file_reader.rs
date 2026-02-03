//! Main POD5 file reader.

use crate::arrow_helpers::BatchFieldExtractor;
use crate::compression;
use crate::error::{Error, Result};
use crate::footer::{self, Footer};
use crate::types::{ReadData, RunInfoData, Uuid, POD5_SIGNATURE};
use crate::CompressedSignalChunk;
use arrow::ipc::reader::FileReader as ArrowFileReader;
use arrow::record_batch::RecordBatch;
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::io::Cursor;
use std::path::Path;
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::RwLock;

/// Default maximum number of signal batches to cache.
const DEFAULT_MAX_CACHED_BATCHES: usize = 10;

/// Metadata about signal table batches for efficient lookup.
#[derive(Debug, Clone)]
struct SignalBatchMetadata {
    /// Number of rows per batch (assumed uniform, determined from batch 0).
    batch_size: usize,
    /// Total number of signal batches.
    num_batches: usize,
}

/// A cached signal batch with access tracking for LRU eviction.
struct CachedSignalBatch {
    batch: RecordBatch,
    last_access: u64,
}

/// LRU cache for signal batches.
struct SignalBatchCache {
    /// Cached batches indexed by batch number.
    batches: HashMap<usize, CachedSignalBatch>,
    /// Maximum number of batches to cache.
    max_size: usize,
    /// Access counter for LRU tracking.
    access_counter: u64,
}

impl SignalBatchCache {
    /// Create a new signal batch cache with the given maximum size.
    fn new(max_size: usize) -> Self {
        Self {
            batches: HashMap::with_capacity(max_size),
            max_size,
            access_counter: 0,
        }
    }

    /// Get a batch from the cache, updating access time.
    fn get(&mut self, batch_idx: usize) -> Option<&RecordBatch> {
        if let Some(cached) = self.batches.get_mut(&batch_idx) {
            self.access_counter += 1;
            cached.last_access = self.access_counter;
            Some(&cached.batch)
        } else {
            None
        }
    }

    /// Insert a batch into the cache, evicting old entries if necessary.
    fn insert(&mut self, batch_idx: usize, batch: RecordBatch) {
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

/// A reader for POD5 files.
pub struct Reader {
    /// Memory-mapped file data.
    mmap: Mmap,
    /// Parsed file footer.
    footer: Footer,
    /// Cached run info data.
    run_info_cache: Vec<RunInfoData>,
    /// Signal batch metadata for O(1) batch lookup (lazy — computed on first use).
    signal_metadata: OnceLock<Option<SignalBatchMetadata>>,
    /// LRU cache for signal batches (thread-safe for parallel operations).
    signal_cache: RwLock<SignalBatchCache>,
}

impl Reader {
    /// Open a POD5 file for reading.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::open_with_cache_size(path, DEFAULT_MAX_CACHED_BATCHES)
    }

    /// Open a POD5 file with a custom signal batch cache size.
    pub fn open_with_cache_size<P: AsRef<Path>>(path: P, cache_size: usize) -> Result<Self> {
        let file = File::open(path.as_ref())?;
        let mmap = unsafe { Mmap::map(&file)? };

        // Verify signature at start
        if mmap.len() < 8 || mmap[..8] != POD5_SIGNATURE {
            return Err(Error::InvalidSignature);
        }

        // Parse footer
        let footer = footer::parse_footer(&mmap)?;

        // Load run info eagerly (it's usually small)
        let run_info_cache = Self::load_run_info(&mmap, &footer)?;

        Ok(Self {
            mmap,
            footer,
            run_info_cache,
            signal_metadata: OnceLock::new(),
            signal_cache: RwLock::new(SignalBatchCache::new(cache_size)),
        })
    }

    /// Load signal batch metadata for O(1) batch lookup.
    ///
    /// Uses the Arrow IPC footer (a few KB at the end of the signal table)
    /// to extract batch count and row counts, avoiding deserialization of
    /// the first signal batch (which can be 50-100MB on large files).
    fn load_signal_metadata(mmap: &Mmap, footer: &Footer) -> Result<Option<SignalBatchMetadata>> {
        use crate::arrow_ipc::ArrowIpcFooter;

        let embedded = match footer.signal_table() {
            Some(e) => e,
            None => return Ok(None),
        };

        let start = embedded.offset as usize;
        let end = start + embedded.length as usize;

        if end > mmap.len() {
            return Err(Error::InvalidFooter(
                "Signal table extends beyond file".to_string(),
            ));
        }

        let slice = &mmap[start..end];
        let ipc_footer = ArrowIpcFooter::parse(slice)?;

        let num_batches = ipc_footer.record_batches.len();
        if num_batches == 0 {
            return Ok(None);
        }

        // Use the row count from batch 0 as the uniform batch size
        let batch_size = ipc_footer.record_batches[0].row_count as usize;
        if batch_size == 0 {
            return Ok(None);
        }

        Ok(Some(SignalBatchMetadata {
            batch_size,
            num_batches,
        }))
    }

    /// Get the file identifier (UUID).
    pub fn file_identifier(&self) -> &str {
        &self.footer.file_identifier
    }

    /// Get the software that wrote this file.
    pub fn software(&self) -> &str {
        &self.footer.software
    }

    /// Get the POD5 version.
    pub fn pod5_version(&self) -> &str {
        &self.footer.pod5_version
    }

    /// Get the number of run info entries.
    pub fn run_info_count(&self) -> usize {
        self.run_info_cache.len()
    }

    /// Get run info by index.
    pub fn get_run_info(&self, index: usize) -> Option<&RunInfoData> {
        self.run_info_cache.get(index)
    }

    /// Get all run info entries.
    pub fn run_infos(&self) -> &[RunInfoData] {
        &self.run_info_cache
    }

    /// Get the number of read batches.
    pub fn read_batch_count(&self) -> Result<usize> {
        let embedded = self
            .footer
            .reads_table()
            .ok_or_else(|| Error::MissingField("reads table".to_string()))?;

        let reader = self.create_arrow_reader(embedded)?;
        Ok(reader.num_batches())
    }

    /// Get a specific read batch.
    pub fn read_batch(&self, index: usize) -> Result<RecordBatch> {
        let embedded = self
            .footer
            .reads_table()
            .ok_or_else(|| Error::MissingField("reads table".to_string()))?;

        let mut reader = self.create_arrow_reader(embedded)?;

        if index >= reader.num_batches() {
            return Err(Error::BatchIndexOutOfBounds {
                index,
                max: reader.num_batches(),
            });
        }

        // Skip to the desired batch
        for _ in 0..index {
            reader.next();
        }

        reader
            .next()
            .ok_or_else(|| Error::BatchIndexOutOfBounds {
                index,
                max: reader.num_batches(),
            })?
            .map_err(Error::from)
    }

    /// Iterate over all reads in the file.
    pub fn reads(&self) -> Result<ReadIterator<'_>> {
        let embedded = self
            .footer
            .reads_table()
            .ok_or_else(|| Error::MissingField("reads table".to_string()))?;

        let reader = self.create_arrow_reader(embedded)?;

        Ok(ReadIterator {
            pod5_reader: self,
            arrow_reader: reader,
            current_batch: None,
            batch_row: 0,
        })
    }

    /// Get the total number of reads (requires scanning all batches).
    pub fn read_count(&self) -> Result<usize> {
        let embedded = self
            .footer
            .reads_table()
            .ok_or_else(|| Error::MissingField("reads table".to_string()))?;

        let reader = self.create_arrow_reader(embedded)?;

        let mut count = 0;
        for batch_result in reader {
            let batch = batch_result?;
            count += batch.num_rows();
        }

        Ok(count)
    }

    /// Get signal data for a read.
    ///
    /// The `signal_rows` parameter should be the signal row indices from the read record.
    /// Uses O(1) batch lookup and LRU caching for efficient repeated access.
    pub fn get_signal(&self, signal_rows: &[u64]) -> Result<Vec<i16>> {
        // Lazily compute signal metadata on first use
        let metadata = self
            .signal_metadata
            .get_or_init(|| Self::load_signal_metadata(&self.mmap, &self.footer).unwrap_or(None));

        if let Some(ref metadata) = metadata {
            return self.get_signal_optimized(signal_rows, metadata);
        }

        // Fallback to original implementation for files without signal table
        self.get_signal_fallback(signal_rows)
    }

    /// Optimized signal retrieval using O(1) batch lookup and LRU cache.
    fn get_signal_optimized(
        &self,
        signal_rows: &[u64],
        metadata: &SignalBatchMetadata,
    ) -> Result<Vec<i16>> {
        let embedded = self
            .footer
            .signal_table()
            .ok_or_else(|| Error::MissingField("signal table".to_string()))?;

        let mut all_samples = Vec::new();

        for &row_idx in signal_rows {
            // O(1) batch lookup: batch_idx = row / batch_size
            let batch_idx = (row_idx as usize) / metadata.batch_size;
            let local_row = (row_idx as usize) % metadata.batch_size;

            if batch_idx >= metadata.num_batches {
                return Err(Error::BatchIndexOutOfBounds {
                    index: batch_idx,
                    max: metadata.num_batches,
                });
            }

            // Try to get from cache first
            let samples = {
                let mut cache = self.signal_cache.write().unwrap();
                if let Some(batch) = cache.get(batch_idx) {
                    // Cache hit - extract signal directly
                    self.extract_signal_from_batch(batch, local_row)?
                } else {
                    // Cache miss - need to load the batch
                    drop(cache); // Release lock before loading

                    let batch = self.load_signal_batch(embedded, batch_idx)?;
                    let samples = self.extract_signal_from_batch(&batch, local_row)?;

                    // Insert into cache
                    self.signal_cache.write().unwrap().insert(batch_idx, batch);

                    samples
                }
            };

            all_samples.extend(samples);
        }

        Ok(all_samples)
    }

    /// Load a specific signal batch by index.
    fn load_signal_batch(
        &self,
        embedded: &crate::footer::EmbeddedFile,
        batch_idx: usize,
    ) -> Result<RecordBatch> {
        let mut reader = self.create_arrow_reader(embedded)?;

        // Skip to the desired batch
        for _ in 0..batch_idx {
            reader.next();
        }

        reader
            .next()
            .ok_or_else(|| Error::BatchIndexOutOfBounds {
                index: batch_idx,
                max: reader.num_batches(),
            })?
            .map_err(Error::from)
    }

    /// Fallback signal retrieval for edge cases (no signal metadata).
    fn get_signal_fallback(&self, signal_rows: &[u64]) -> Result<Vec<i16>> {
        let embedded = self
            .footer
            .signal_table()
            .ok_or_else(|| Error::MissingField("signal table".to_string()))?;

        let reader = self.create_arrow_reader(embedded)?;
        let mut all_samples = Vec::new();

        // Load all batches (original behavior)
        let mut signal_batches: Vec<RecordBatch> = Vec::new();
        for batch_result in reader {
            signal_batches.push(batch_result?);
        }

        for &row_idx in signal_rows {
            // Find which batch contains this row
            let mut cumulative_rows = 0u64;
            for batch in &signal_batches {
                let batch_rows = batch.num_rows() as u64;
                if row_idx < cumulative_rows + batch_rows {
                    let local_row = (row_idx - cumulative_rows) as usize;
                    let samples = self.extract_signal_from_batch(batch, local_row)?;
                    all_samples.extend(samples);
                    break;
                }
                cumulative_rows += batch_rows;
            }
        }

        Ok(all_samples)
    }

    /// Get all compressed signal chunks without decompressing.
    /// This is efficient for block-level copying during merge/filter operations.
    pub fn get_all_signal_compressed(&self) -> Result<Vec<CompressedSignalChunk>> {
        let embedded = self
            .footer
            .signal_table()
            .ok_or_else(|| Error::MissingField("signal table".to_string()))?;

        let reader = self.create_arrow_reader(embedded)?;
        let mut all_chunks = Vec::new();

        for batch_result in reader {
            let batch = batch_result?;
            self.extract_compressed_signal_from_batch(&batch, &mut all_chunks)?;
        }

        Ok(all_chunks)
    }

    /// Get signal batches as Arrow RecordBatches for direct batch-level copying.
    /// This is the fastest method for merge operations - copies batches without unpacking.
    pub fn signal_batches(&self) -> Result<Vec<RecordBatch>> {
        let embedded = self
            .footer
            .signal_table()
            .ok_or_else(|| Error::MissingField("signal table".to_string()))?;

        let reader = self.create_arrow_reader(embedded)?;
        let mut batches = Vec::new();

        for batch_result in reader {
            batches.push(batch_result?);
        }

        Ok(batches)
    }

    /// Get raw bytes of the signal table for direct byte-level copying.
    /// This returns a slice into the memory-mapped file containing the complete
    /// Arrow IPC stream for the signal table.
    pub fn signal_table_bytes(&self) -> Result<&[u8]> {
        let embedded = self
            .footer
            .signal_table()
            .ok_or_else(|| Error::MissingField("signal table".to_string()))?;

        let start = embedded.offset as usize;
        let end = start + embedded.length as usize;

        if end > self.mmap.len() {
            return Err(Error::InvalidFooter(format!(
                "Signal table extends beyond file: {}..{} > {}",
                start,
                end,
                self.mmap.len()
            )));
        }

        Ok(&self.mmap[start..end])
    }

    /// Bulk extract decompressed signal for multiple reads.
    ///
    /// Takes a slice of `(key, signal_rows)` pairs and returns a Vec of
    /// `(key, Vec<i16>)` with the decompressed signal for each. Uses the fast
    /// raw byte extraction path (batch-grouped, no Arrow deserialization),
    /// which is much faster than calling `get_signal` per read.
    pub fn get_signal_bulk<K: Clone + Send>(
        &self,
        reads: &[(K, Vec<u64>)],
    ) -> Result<Vec<(K, Vec<i16>)>> {
        use crate::arrow_ipc::ArrowIpcFooter;
        use crate::compression::vbz::decompress_signal;
        use rayon::prelude::*;

        let signal_bytes = self.signal_table_bytes()?;
        let signal_footer = ArrowIpcFooter::parse(signal_bytes)?;

        // Collect all signal rows with back-references to which read they belong to
        // (read_index, chunk_index_within_read, signal_row)
        let mut all_rows: Vec<(usize, usize, u64)> = Vec::new();
        for (read_idx, (_key, rows)) in reads.iter().enumerate() {
            for (chunk_idx, &row) in rows.iter().enumerate() {
                all_rows.push((read_idx, chunk_idx, row));
            }
        }

        // Extract all signal rows at once (batch-grouped, sequential I/O)
        let row_indices: Vec<u64> = all_rows.iter().map(|&(_, _, row)| row).collect();
        let raw_chunks = signal_footer.extract_signal_rows(&row_indices, signal_bytes)?;

        // Decompress in parallel (VBZ decompression is CPU-bound)
        let decompressed: Vec<Result<Vec<i16>>> = raw_chunks
            .par_iter()
            .map(|chunk| decompress_signal(chunk.signal, chunk.samples as usize))
            .collect();

        // Assemble per-read
        let mut result_chunks: Vec<Vec<(usize, Vec<i16>)>> = vec![Vec::new(); reads.len()];
        for (i, decompressed_result) in decompressed.into_iter().enumerate() {
            let (read_idx, chunk_idx, _) = all_rows[i];
            result_chunks[read_idx].push((chunk_idx, decompressed_result?));
        }

        // Sort chunks within each read and concatenate
        let mut results = Vec::with_capacity(reads.len());
        for (read_idx, (key, _)) in reads.iter().enumerate() {
            let chunks = &mut result_chunks[read_idx];
            chunks.sort_by_key(|(idx, _)| *idx);
            let signal: Vec<i16> = chunks.iter().flat_map(|(_, s)| s.iter().copied()).collect();
            results.push((key.clone(), signal));
        }

        Ok(results)
    }

    /// Create a thread-safe `SignalExtractor` for parallel per-read signal extraction.
    ///
    /// The returned extractor borrows the memory-mapped signal table and can be
    /// shared across rayon threads (`Send + Sync`). Each thread can call
    /// `extractor.get_signal(&signal_rows)` independently without contention.
    pub fn signal_extractor(&self) -> Result<SignalExtractor<'_>> {
        use crate::arrow_ipc::ArrowIpcFooter;

        let signal_bytes = self.signal_table_bytes()?;
        let footer = ArrowIpcFooter::parse(signal_bytes)?;

        Ok(SignalExtractor {
            signal_bytes,
            footer,
        })
    }

    /// Prefetch signal table pages using madvise (if supported).
    /// This hints to the OS to read pages ahead, improving sequential read performance.
    pub fn prefetch_signal(&self) {
        if let Some(embedded) = self.footer.signal_table() {
            let start = embedded.offset as usize;
            let end = (start + embedded.length as usize).min(self.mmap.len());
            // Use madvise to hint sequential access
            #[cfg(unix)]
            {
                let _ = self.mmap.advise_range(
                    memmap2::Advice::WillNeed,
                    start,
                    end.saturating_sub(start),
                );
            }
            // Fallback for non-unix: touch pages manually
            #[cfg(not(unix))]
            {
                let signal_bytes = &self.mmap[start..end];
                let _ = signal_bytes
                    .iter()
                    .step_by(4096)
                    .fold(0u8, |acc, &b| acc.wrapping_add(b));
            }
        }
    }

    /// Get the total number of signal rows across all batches.
    pub fn signal_row_count(&self) -> Result<u64> {
        let embedded = match self.footer.signal_table() {
            Some(e) => e,
            None => return Ok(0),
        };

        let reader = self.create_arrow_reader(embedded)?;
        let mut count = 0u64;

        for batch_result in reader {
            count += batch_result?.num_rows() as u64;
        }

        Ok(count)
    }

    /// Get compressed signal chunks for specific row indices only.
    /// This is more efficient than get_all_signal_compressed() when only a subset
    /// of reads are needed (e.g., for filter operations).
    /// Uses O(1) batch lookup and LRU caching for repeated access.
    pub fn get_compressed_signal_for_rows(
        &self,
        signal_rows: &[u64],
    ) -> Result<Vec<CompressedSignalChunk>> {
        // Lazily compute signal metadata on first use
        let metadata = self
            .signal_metadata
            .get_or_init(|| Self::load_signal_metadata(&self.mmap, &self.footer).unwrap_or(None));

        if let Some(ref metadata) = metadata {
            return self.get_compressed_signal_optimized(signal_rows, metadata);
        }

        // Fallback: load all and filter (less efficient)
        let all_signal = self.get_all_signal_compressed()?;
        let mut result = Vec::with_capacity(signal_rows.len());
        for &idx in signal_rows {
            if let Some(chunk) = all_signal.get(idx as usize) {
                result.push(chunk.clone());
            }
        }
        Ok(result)
    }

    /// Optimized compressed signal retrieval using O(1) batch lookup and LRU cache.
    fn get_compressed_signal_optimized(
        &self,
        signal_rows: &[u64],
        metadata: &SignalBatchMetadata,
    ) -> Result<Vec<CompressedSignalChunk>> {
        let embedded = self
            .footer
            .signal_table()
            .ok_or_else(|| Error::MissingField("signal table".to_string()))?;

        let mut result = Vec::with_capacity(signal_rows.len());

        for &row_idx in signal_rows {
            // O(1) batch lookup
            let batch_idx = (row_idx as usize) / metadata.batch_size;
            let local_row = (row_idx as usize) % metadata.batch_size;

            if batch_idx >= metadata.num_batches {
                return Err(Error::BatchIndexOutOfBounds {
                    index: batch_idx,
                    max: metadata.num_batches,
                });
            }

            // Try to get from cache first
            let chunk = {
                let mut cache = self.signal_cache.write().unwrap();
                if let Some(batch) = cache.get(batch_idx) {
                    self.extract_single_compressed_chunk(batch, local_row)?
                } else {
                    drop(cache);
                    let batch = self.load_signal_batch(embedded, batch_idx)?;
                    let chunk = self.extract_single_compressed_chunk(&batch, local_row)?;
                    self.signal_cache.write().unwrap().insert(batch_idx, batch);
                    chunk
                }
            };

            result.push(chunk);
        }

        Ok(result)
    }

    /// Extract a single compressed signal chunk from a batch row.
    fn extract_single_compressed_chunk(
        &self,
        batch: &RecordBatch,
        row: usize,
    ) -> Result<CompressedSignalChunk> {
        use arrow::array::{Array, FixedSizeBinaryArray, LargeBinaryArray, UInt32Array};

        let read_id_col = batch
            .column_by_name("read_id")
            .ok_or_else(|| Error::MissingField("read_id column".to_string()))?;
        let signal_col = batch
            .column_by_name("signal")
            .ok_or_else(|| Error::MissingField("signal column".to_string()))?;
        let samples_col = batch
            .column_by_name("samples")
            .ok_or_else(|| Error::MissingField("samples column".to_string()))?;

        let read_id_array = read_id_col
            .as_any()
            .downcast_ref::<FixedSizeBinaryArray>()
            .ok_or_else(|| Error::InvalidField {
                field: "read_id".to_string(),
                message: "Expected FixedSizeBinaryArray".to_string(),
            })?;

        let signal_array = signal_col
            .as_any()
            .downcast_ref::<LargeBinaryArray>()
            .ok_or_else(|| Error::InvalidField {
                field: "signal".to_string(),
                message: "Expected LargeBinaryArray".to_string(),
            })?;

        let samples_array = samples_col
            .as_any()
            .downcast_ref::<UInt32Array>()
            .ok_or_else(|| Error::InvalidField {
                field: "samples".to_string(),
                message: "Expected UInt32Array".to_string(),
            })?;

        let read_id_bytes = read_id_array.value(row);
        let read_id =
            Uuid::from_slice(read_id_bytes).map_err(|e| Error::InvalidUuid(e.to_string()))?;
        let compressed_data = signal_array.value(row);
        let samples = samples_array.value(row);

        Ok(CompressedSignalChunk {
            read_id,
            samples,
            data: Arc::from(compressed_data),
        })
    }

    /// Extract compressed signal chunks from a batch.
    fn extract_compressed_signal_from_batch(
        &self,
        batch: &RecordBatch,
        chunks: &mut Vec<CompressedSignalChunk>,
    ) -> Result<()> {
        use arrow::array::{Array, FixedSizeBinaryArray, LargeBinaryArray, UInt32Array};

        let read_id_col = batch
            .column_by_name("read_id")
            .ok_or_else(|| Error::MissingField("read_id column".to_string()))?;
        let signal_col = batch
            .column_by_name("signal")
            .ok_or_else(|| Error::MissingField("signal column".to_string()))?;
        let samples_col = batch
            .column_by_name("samples")
            .ok_or_else(|| Error::MissingField("samples column".to_string()))?;

        let read_id_array = read_id_col
            .as_any()
            .downcast_ref::<FixedSizeBinaryArray>()
            .ok_or_else(|| Error::InvalidField {
                field: "read_id".to_string(),
                message: "Expected FixedSizeBinaryArray".to_string(),
            })?;

        let signal_array = signal_col
            .as_any()
            .downcast_ref::<LargeBinaryArray>()
            .ok_or_else(|| Error::InvalidField {
                field: "signal".to_string(),
                message: "Expected LargeBinaryArray".to_string(),
            })?;

        let samples_array = samples_col
            .as_any()
            .downcast_ref::<UInt32Array>()
            .ok_or_else(|| Error::InvalidField {
                field: "samples".to_string(),
                message: "Expected UInt32Array".to_string(),
            })?;

        for row in 0..batch.num_rows() {
            let read_id_bytes = read_id_array.value(row);
            let read_id =
                Uuid::from_slice(read_id_bytes).map_err(|e| Error::InvalidUuid(e.to_string()))?;
            let compressed_data = signal_array.value(row);
            let samples = samples_array.value(row);

            chunks.push(CompressedSignalChunk {
                read_id,
                samples,
                data: Arc::from(compressed_data),
            });
        }

        Ok(())
    }

    /// Extract signal samples from a signal table batch row.
    fn extract_signal_from_batch(&self, batch: &RecordBatch, row: usize) -> Result<Vec<i16>> {
        use arrow::array::{Array, LargeBinaryArray, UInt32Array};

        // Get signal column (LargeBinary with VBZ data)
        let signal_col = batch
            .column_by_name("signal")
            .ok_or_else(|| Error::MissingField("signal column".to_string()))?;

        // Get samples column for count
        let samples_col = batch
            .column_by_name("samples")
            .ok_or_else(|| Error::MissingField("samples column".to_string()))?;

        let samples_array = samples_col
            .as_any()
            .downcast_ref::<UInt32Array>()
            .ok_or_else(|| Error::InvalidField {
                field: "samples".to_string(),
                message: "Expected UInt32Array".to_string(),
            })?;

        let sample_count = samples_array.value(row) as usize;

        // Handle signal data (could be LargeBinary for VBZ)
        let signal_array = signal_col
            .as_any()
            .downcast_ref::<LargeBinaryArray>()
            .ok_or_else(|| Error::InvalidField {
                field: "signal".to_string(),
                message: "Expected LargeBinaryArray".to_string(),
            })?;

        let compressed_data = signal_array.value(row);

        // Decompress VBZ data
        compression::decompress_signal(compressed_data, sample_count)
    }

    /// Extract a read from a record batch at the given row.
    ///
    /// This is useful for batch-level parallel processing where you want to
    /// process batches in parallel using rayon.
    pub fn read_from_batch(batch: &RecordBatch, row: usize) -> Result<ReadData> {
        extract_read_from_batch(batch, row, true)
    }

    /// Get all read IDs from the file efficiently (reads only the read_id column).
    ///
    /// This is much faster than iterating over all reads when you only need the IDs,
    /// as it uses Arrow column projection to avoid loading other columns.
    pub fn read_ids(&self) -> Result<Vec<Uuid>> {
        use arrow::array::{Array, FixedSizeBinaryArray};

        let embedded = self
            .footer
            .reads_table()
            .ok_or_else(|| Error::MissingField("reads table".to_string()))?;

        // Create reader with projection for just the read_id column (index 0)
        let reader = self.create_arrow_reader_with_projection(embedded, Some(vec![0]))?;

        let mut read_ids = Vec::new();
        for batch_result in reader {
            let batch = batch_result?;
            // The projected batch will have read_id as column 0
            if let Some(col) = batch
                .column(0)
                .as_any()
                .downcast_ref::<FixedSizeBinaryArray>()
            {
                for row in 0..col.len() {
                    if let Ok(uuid) = Uuid::from_slice(col.value(row)) {
                        read_ids.push(uuid);
                    }
                }
            }
        }

        Ok(read_ids)
    }

    /// Get read IDs from a specific batch efficiently (reads only the read_id column).
    pub fn read_ids_from_batch(&self, batch_idx: usize) -> Result<Vec<Uuid>> {
        use arrow::array::{Array, FixedSizeBinaryArray};

        let embedded = self
            .footer
            .reads_table()
            .ok_or_else(|| Error::MissingField("reads table".to_string()))?;

        // Create reader with projection for just the read_id column (index 0)
        let mut reader = self.create_arrow_reader_with_projection(embedded, Some(vec![0]))?;

        if batch_idx >= reader.num_batches() {
            return Err(Error::BatchIndexOutOfBounds {
                index: batch_idx,
                max: reader.num_batches(),
            });
        }

        // Skip to the desired batch
        for _ in 0..batch_idx {
            reader.next();
        }

        let batch = reader.next().ok_or_else(|| Error::BatchIndexOutOfBounds {
            index: batch_idx,
            max: reader.num_batches(),
        })??;

        let mut read_ids = Vec::new();
        if let Some(col) = batch
            .column(0)
            .as_any()
            .downcast_ref::<FixedSizeBinaryArray>()
        {
            for row in 0..col.len() {
                if let Ok(uuid) = Uuid::from_slice(col.value(row)) {
                    read_ids.push(uuid);
                }
            }
        }

        Ok(read_ids)
    }

    /// Create an Arrow IPC file reader for an embedded file.
    fn create_arrow_reader(
        &self,
        embedded: &crate::footer::EmbeddedFile,
    ) -> Result<ArrowFileReader<Cursor<&[u8]>>> {
        self.create_arrow_reader_with_projection(embedded, None)
    }

    /// Create an Arrow IPC file reader with optional column projection.
    fn create_arrow_reader_with_projection(
        &self,
        embedded: &crate::footer::EmbeddedFile,
        projection: Option<Vec<usize>>,
    ) -> Result<ArrowFileReader<Cursor<&[u8]>>> {
        let start = embedded.offset as usize;
        let end = start + embedded.length as usize;

        if end > self.mmap.len() {
            return Err(Error::InvalidFooter(format!(
                "Embedded file extends beyond file end: {} + {} > {}",
                start,
                embedded.length,
                self.mmap.len()
            )));
        }

        let slice = &self.mmap[start..end];
        let cursor = Cursor::new(slice);
        ArrowFileReader::try_new(cursor, projection).map_err(Error::from)
    }

    /// Load run info from the run info table.
    fn load_run_info(mmap: &Mmap, footer: &Footer) -> Result<Vec<RunInfoData>> {
        let embedded = match footer.run_info_table() {
            Some(e) => e,
            None => return Ok(Vec::new()),
        };

        let start = embedded.offset as usize;
        let end = start + embedded.length as usize;

        if end > mmap.len() {
            return Err(Error::InvalidFooter(
                "Run info table extends beyond file".to_string(),
            ));
        }

        let slice = &mmap[start..end];
        let cursor = Cursor::new(slice);
        let reader = ArrowFileReader::try_new(cursor, None)?;

        let mut run_infos = Vec::new();
        for batch_result in reader {
            let batch = batch_result?;
            for row in 0..batch.num_rows() {
                run_infos.push(Self::run_info_from_batch(&batch, row)?);
            }
        }

        Ok(run_infos)
    }

    /// Extract RunInfoData from a batch row.
    fn run_info_from_batch(batch: &RecordBatch, row: usize) -> Result<RunInfoData> {
        let ext = BatchFieldExtractor::new(batch, row);

        // Parse context_tags map if present
        let context_tags = Self::parse_map_column(batch, "context_tags", row);

        // Parse tracking_id map if present
        let tracking_id = Self::parse_map_column(batch, "tracking_id", row);

        Ok(RunInfoData {
            acquisition_id: ext.get_string("acquisition_id")?,
            acquisition_start_time: ext.get_timestamp("acquisition_start_time")?,
            adc_max: ext.get_i16("adc_max")?,
            adc_min: ext.get_i16("adc_min")?,
            context_tags,
            experiment_name: ext.get_string("experiment_name").unwrap_or_default(),
            flow_cell_id: ext.get_string("flow_cell_id").unwrap_or_default(),
            flow_cell_product_code: ext.get_string("flow_cell_product_code").unwrap_or_default(),
            protocol_name: ext.get_string("protocol_name").unwrap_or_default(),
            protocol_run_id: ext.get_string("protocol_run_id").unwrap_or_default(),
            protocol_start_time: ext.get_timestamp("protocol_start_time").unwrap_or(0),
            sample_id: ext.get_string("sample_id").unwrap_or_default(),
            sample_rate: ext.get_u16("sample_rate")?,
            sequencing_kit: ext.get_string("sequencing_kit").unwrap_or_default(),
            sequencer_position: ext.get_string("sequencer_position").unwrap_or_default(),
            sequencer_position_type: ext
                .get_string("sequencer_position_type")
                .unwrap_or_default(),
            software: ext.get_string("software").unwrap_or_default(),
            system_name: ext.get_string("system_name").unwrap_or_default(),
            system_type: ext.get_string("system_type").unwrap_or_default(),
            tracking_id,
        })
    }

    /// Parse a Map column into a HashMap.
    fn parse_map_column(batch: &RecordBatch, name: &str, row: usize) -> HashMap<String, String> {
        use arrow::array::{Array, MapArray, StringArray, StructArray};

        let Some(col) = batch.column_by_name(name) else {
            return HashMap::new();
        };

        let Some(map_array) = col.as_any().downcast_ref::<MapArray>() else {
            return HashMap::new();
        };

        let mut result = HashMap::new();

        // Get the entries for this row as a StructArray
        let entries = map_array.value(row);
        let Some(struct_array) = entries.as_any().downcast_ref::<StructArray>() else {
            return HashMap::new();
        };

        if struct_array.num_columns() >= 2 {
            if let (Some(keys), Some(values)) = (
                struct_array
                    .column(0)
                    .as_any()
                    .downcast_ref::<StringArray>(),
                struct_array
                    .column(1)
                    .as_any()
                    .downcast_ref::<StringArray>(),
            ) {
                for i in 0..struct_array.len() {
                    if !keys.is_null(i) && !values.is_null(i) {
                        result.insert(keys.value(i).to_string(), values.value(i).to_string());
                    }
                }
            }
        }

        result
    }
}

/// Thread-safe signal extractor for parallel per-read signal extraction.
///
/// Holds an immutable reference to the memory-mapped signal table bytes and
/// a pre-parsed Arrow IPC footer. Because it contains only immutable data,
/// it is `Send + Sync` and can be shared across rayon threads.
pub struct SignalExtractor<'a> {
    signal_bytes: &'a [u8],
    footer: crate::arrow_ipc::ArrowIpcFooter,
}

impl<'a> SignalExtractor<'a> {
    /// Extract and decompress signal for a single read's signal rows.
    ///
    /// Thread-safe: no shared mutable state.
    pub fn get_signal(&self, signal_rows: &[u64]) -> Result<Vec<i16>> {
        use crate::compression::vbz::decompress_signal;

        let raw_chunks = self
            .footer
            .extract_signal_rows(signal_rows, self.signal_bytes)?;
        let total_samples: usize = raw_chunks.iter().map(|c| c.samples as usize).sum();
        let mut result = Vec::with_capacity(total_samples);

        for chunk in &raw_chunks {
            let decompressed = decompress_signal(chunk.signal, chunk.samples as usize)?;
            result.extend_from_slice(&decompressed);
        }

        Ok(result)
    }
}

/// Iterator over reads in a POD5 file.
pub struct ReadIterator<'a> {
    #[allow(dead_code)]
    pod5_reader: &'a Reader,
    arrow_reader: ArrowFileReader<Cursor<&'a [u8]>>,
    current_batch: Option<RecordBatch>,
    batch_row: usize,
}

impl<'a> Iterator for ReadIterator<'a> {
    type Item = Result<ReadData>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // Check if we need a new batch
            let need_new_batch = match &self.current_batch {
                None => true,
                Some(batch) => self.batch_row >= batch.num_rows(),
            };

            if need_new_batch {
                match self.arrow_reader.next() {
                    Some(Ok(batch)) => {
                        self.current_batch = Some(batch);
                        self.batch_row = 0;
                    }
                    Some(Err(e)) => return Some(Err(Error::from(e))),
                    None => return None,
                }
            }

            // Extract read from current batch
            if let Some(batch) = &self.current_batch {
                let row = self.batch_row;
                self.batch_row += 1;
                return Some(Self::read_from_batch(batch, row));
            }
        }
    }
}

impl<'a> ReadIterator<'a> {
    fn read_from_batch(batch: &RecordBatch, row: usize) -> Result<ReadData> {
        extract_read_from_batch(batch, row, false)
    }
}

/// Extract a read from a record batch at the given row.
///
/// This is the shared implementation used by both `Reader::read_from_batch`
/// and `ReadIterator::read_from_batch`.
///
/// The `try_alternate_field_names` parameter controls whether to try alternate
/// field names for compatibility with different POD5 versions:
/// - `start_sample` vs `start`
/// - `predicted_scaling_open_pore_level` vs `open_pore_level`
fn extract_read_from_batch(
    batch: &RecordBatch,
    row: usize,
    try_alternate_field_names: bool,
) -> Result<ReadData> {
    let ext = BatchFieldExtractor::new(batch, row);

    // Handle start_sample field name variations
    let start_sample = if try_alternate_field_names {
        ext.get_u64("start_sample")
            .or_else(|_| ext.get_u64("start"))?
    } else {
        ext.get_u64("start")?
    };

    // Handle open_pore_level field name variations
    let open_pore_level = if try_alternate_field_names {
        ext.get_f32("predicted_scaling_open_pore_level")
            .or_else(|_| ext.get_f32("open_pore_level"))
            .unwrap_or(0.0)
    } else {
        ext.get_f32("open_pore_level").unwrap_or(0.0)
    };

    // Get run_info index from dictionary key
    let run_info_index = ext
        .get_dict_index("run_info")
        .map(|idx| idx as u32)
        .unwrap_or(0);

    // Parse end_reason - use FromStr which returns Infallible
    let end_reason_str = ext.get_dict_string("end_reason").unwrap_or_default();
    let end_reason = end_reason_str.parse().unwrap_or_default();

    Ok(ReadData {
        read_id: ext.get_uuid("read_id")?,
        read_number: ext.get_u32("read_number")?,
        start_sample,
        channel: ext.get_u16("channel")?,
        well: ext.get_u8("well")?,
        pore_type: ext.get_dict_string("pore_type").unwrap_or_default(),
        calibration_offset: ext.get_f32("calibration_offset")?,
        calibration_scale: ext.get_f32("calibration_scale")?,
        median_before: ext.get_f32("median_before")?,
        end_reason,
        end_reason_forced: ext.get_bool("end_reason_forced")?,
        run_info_index,
        num_minknow_events: ext.get_u64("num_minknow_events")?,
        num_samples: ext.get_u64("num_samples")?,
        open_pore_level,
        signal_rows: ext.get_signal_rows()?,
    })
}
