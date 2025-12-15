//! Main POD5 file reader.

use crate::compression;
use crate::error::{Error, Result};
use crate::footer::{self, Footer};
use crate::types::{ReadData, RunInfoData, Uuid, POD5_SIGNATURE};
use crate::CompressedSignalChunk;
use arrow::ipc::reader::FileReader as ArrowFileReader;
use arrow::record_batch::RecordBatch;
use memmap2::Mmap;
use std::cell::RefCell;
use std::collections::HashMap;
use std::fs::File;
use std::io::Cursor;
use std::path::Path;
use std::sync::Arc;

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
    /// Signal batch metadata for O(1) batch lookup.
    signal_metadata: Option<SignalBatchMetadata>,
    /// LRU cache for signal batches (interior mutability for read operations).
    signal_cache: RefCell<SignalBatchCache>,
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

        // Load signal batch metadata (batch size from batch 0, like C++ implementation)
        let signal_metadata = Self::load_signal_metadata(&mmap, &footer)?;

        Ok(Self {
            mmap,
            footer,
            run_info_cache,
            signal_metadata,
            signal_cache: RefCell::new(SignalBatchCache::new(cache_size)),
        })
    }

    /// Load signal batch metadata for O(1) batch lookup.
    fn load_signal_metadata(mmap: &Mmap, footer: &Footer) -> Result<Option<SignalBatchMetadata>> {
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
        let cursor = Cursor::new(slice);
        let reader = ArrowFileReader::try_new(cursor, None)?;

        let num_batches = reader.num_batches();
        if num_batches == 0 {
            return Ok(None);
        }

        // Read batch 0 to determine batch size (like C++ implementation)
        let mut reader_iter = reader.into_iter();
        let batch_size = match reader_iter.next() {
            Some(Ok(batch)) => batch.num_rows(),
            Some(Err(e)) => return Err(Error::from(e)),
            None => return Ok(None),
        };

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
        // Use optimized path if we have signal metadata
        if let Some(ref metadata) = self.signal_metadata {
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
                let mut cache = self.signal_cache.borrow_mut();
                if let Some(batch) = cache.get(batch_idx) {
                    // Cache hit - extract signal directly
                    self.extract_signal_from_batch(batch, local_row)?
                } else {
                    // Cache miss - need to load the batch
                    drop(cache); // Release borrow before loading

                    let batch = self.load_signal_batch(embedded, batch_idx)?;
                    let samples = self.extract_signal_from_batch(&batch, local_row)?;

                    // Insert into cache
                    self.signal_cache.borrow_mut().insert(batch_idx, batch);

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

    /// Get compressed signal chunks for specific row indices only.
    /// This is more efficient than get_all_signal_compressed() when only a subset
    /// of reads are needed (e.g., for filter operations).
    /// Uses O(1) batch lookup and LRU caching for repeated access.
    pub fn get_compressed_signal_for_rows(
        &self,
        signal_rows: &[u64],
    ) -> Result<Vec<CompressedSignalChunk>> {
        // Use optimized path if we have signal metadata
        if let Some(ref metadata) = self.signal_metadata {
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
                let mut cache = self.signal_cache.borrow_mut();
                if let Some(batch) = cache.get(batch_idx) {
                    self.extract_single_compressed_chunk(batch, local_row)?
                } else {
                    drop(cache);
                    let batch = self.load_signal_batch(embedded, batch_idx)?;
                    let chunk = self.extract_single_compressed_chunk(&batch, local_row)?;
                    self.signal_cache.borrow_mut().insert(batch_idx, batch);
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
            let read_id = Uuid::from_slice(read_id_bytes)
                .map_err(|e| Error::InvalidUuid(e.to_string()))?;
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

    /// Create an Arrow IPC file reader for an embedded file.
    fn create_arrow_reader(
        &self,
        embedded: &crate::footer::EmbeddedFile,
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
        ArrowFileReader::try_new(cursor, None).map_err(Error::from)
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
        use arrow::array::{
            Array, Int16Array, StringArray, TimestampMillisecondArray, UInt16Array,
        };

        let get_string = |name: &str| -> Result<String> {
            let col = batch
                .column_by_name(name)
                .ok_or_else(|| Error::MissingField(name.to_string()))?;
            let arr =
                col.as_any()
                    .downcast_ref::<StringArray>()
                    .ok_or_else(|| Error::InvalidField {
                        field: name.to_string(),
                        message: "Expected StringArray".to_string(),
                    })?;
            Ok(arr.value(row).to_string())
        };

        let get_i16 = |name: &str| -> Result<i16> {
            let col = batch
                .column_by_name(name)
                .ok_or_else(|| Error::MissingField(name.to_string()))?;
            let arr =
                col.as_any()
                    .downcast_ref::<Int16Array>()
                    .ok_or_else(|| Error::InvalidField {
                        field: name.to_string(),
                        message: "Expected Int16Array".to_string(),
                    })?;
            Ok(arr.value(row))
        };

        let get_u16 = |name: &str| -> Result<u16> {
            let col = batch
                .column_by_name(name)
                .ok_or_else(|| Error::MissingField(name.to_string()))?;
            let arr =
                col.as_any()
                    .downcast_ref::<UInt16Array>()
                    .ok_or_else(|| Error::InvalidField {
                        field: name.to_string(),
                        message: "Expected UInt16Array".to_string(),
                    })?;
            Ok(arr.value(row))
        };

        let get_timestamp = |name: &str| -> Result<i64> {
            let col = batch
                .column_by_name(name)
                .ok_or_else(|| Error::MissingField(name.to_string()))?;
            let arr = col
                .as_any()
                .downcast_ref::<TimestampMillisecondArray>()
                .ok_or_else(|| Error::InvalidField {
                    field: name.to_string(),
                    message: "Expected TimestampMillisecondArray".to_string(),
                })?;
            Ok(arr.value(row))
        };

        Ok(RunInfoData {
            acquisition_id: get_string("acquisition_id")?,
            acquisition_start_time: get_timestamp("acquisition_start_time")?,
            adc_max: get_i16("adc_max")?,
            adc_min: get_i16("adc_min")?,
            context_tags: HashMap::new(), // TODO: parse map
            experiment_name: get_string("experiment_name").unwrap_or_default(),
            flow_cell_id: get_string("flow_cell_id").unwrap_or_default(),
            flow_cell_product_code: get_string("flow_cell_product_code").unwrap_or_default(),
            protocol_name: get_string("protocol_name").unwrap_or_default(),
            protocol_run_id: get_string("protocol_run_id").unwrap_or_default(),
            protocol_start_time: get_timestamp("protocol_start_time").unwrap_or(0),
            sample_id: get_string("sample_id").unwrap_or_default(),
            sample_rate: get_u16("sample_rate")?,
            sequencing_kit: get_string("sequencing_kit").unwrap_or_default(),
            sequencer_position: get_string("sequencer_position").unwrap_or_default(),
            sequencer_position_type: get_string("sequencer_position_type").unwrap_or_default(),
            software: get_string("software").unwrap_or_default(),
            system_name: get_string("system_name").unwrap_or_default(),
            system_type: get_string("system_type").unwrap_or_default(),
            tracking_id: HashMap::new(), // TODO: parse map
        })
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
        use arrow::array::{
            Array, BooleanArray, DictionaryArray, FixedSizeBinaryArray, Float32Array, ListArray,
            UInt16Array, UInt32Array, UInt64Array, UInt8Array,
        };

        // Helper functions
        let get_uuid = |name: &str| -> Result<Uuid> {
            let col = batch
                .column_by_name(name)
                .ok_or_else(|| Error::MissingField(name.to_string()))?;
            let arr = col
                .as_any()
                .downcast_ref::<FixedSizeBinaryArray>()
                .ok_or_else(|| Error::InvalidField {
                    field: name.to_string(),
                    message: "Expected FixedSizeBinaryArray".to_string(),
                })?;
            let bytes = arr.value(row);
            Uuid::from_slice(bytes).map_err(|e| Error::InvalidUuid(e.to_string()))
        };

        let get_u8 = |name: &str| -> Result<u8> {
            let col = batch
                .column_by_name(name)
                .ok_or_else(|| Error::MissingField(name.to_string()))?;
            let arr =
                col.as_any()
                    .downcast_ref::<UInt8Array>()
                    .ok_or_else(|| Error::InvalidField {
                        field: name.to_string(),
                        message: "Expected UInt8Array".to_string(),
                    })?;
            Ok(arr.value(row))
        };

        let get_u16 = |name: &str| -> Result<u16> {
            let col = batch
                .column_by_name(name)
                .ok_or_else(|| Error::MissingField(name.to_string()))?;
            let arr =
                col.as_any()
                    .downcast_ref::<UInt16Array>()
                    .ok_or_else(|| Error::InvalidField {
                        field: name.to_string(),
                        message: "Expected UInt16Array".to_string(),
                    })?;
            Ok(arr.value(row))
        };

        let get_u32 = |name: &str| -> Result<u32> {
            let col = batch
                .column_by_name(name)
                .ok_or_else(|| Error::MissingField(name.to_string()))?;
            let arr =
                col.as_any()
                    .downcast_ref::<UInt32Array>()
                    .ok_or_else(|| Error::InvalidField {
                        field: name.to_string(),
                        message: "Expected UInt32Array".to_string(),
                    })?;
            Ok(arr.value(row))
        };

        let get_u64 = |name: &str| -> Result<u64> {
            let col = batch
                .column_by_name(name)
                .ok_or_else(|| Error::MissingField(name.to_string()))?;
            let arr =
                col.as_any()
                    .downcast_ref::<UInt64Array>()
                    .ok_or_else(|| Error::InvalidField {
                        field: name.to_string(),
                        message: "Expected UInt64Array".to_string(),
                    })?;
            Ok(arr.value(row))
        };

        let get_f32 = |name: &str| -> Result<f32> {
            let col = batch
                .column_by_name(name)
                .ok_or_else(|| Error::MissingField(name.to_string()))?;
            let arr =
                col.as_any()
                    .downcast_ref::<Float32Array>()
                    .ok_or_else(|| Error::InvalidField {
                        field: name.to_string(),
                        message: "Expected Float32Array".to_string(),
                    })?;
            Ok(arr.value(row))
        };

        let get_bool = |name: &str| -> Result<bool> {
            let col = batch
                .column_by_name(name)
                .ok_or_else(|| Error::MissingField(name.to_string()))?;
            let arr =
                col.as_any()
                    .downcast_ref::<BooleanArray>()
                    .ok_or_else(|| Error::InvalidField {
                        field: name.to_string(),
                        message: "Expected BooleanArray".to_string(),
                    })?;
            Ok(arr.value(row))
        };

        // Get dictionary-encoded string value
        let get_dict_string = |name: &str| -> Result<String> {
            let col = batch
                .column_by_name(name)
                .ok_or_else(|| Error::MissingField(name.to_string()))?;

            // Try Int16 dictionary first
            if let Some(dict) = col
                .as_any()
                .downcast_ref::<DictionaryArray<arrow::datatypes::Int16Type>>()
            {
                let keys = dict.keys();
                let values = dict.values();
                let values = values
                    .as_any()
                    .downcast_ref::<arrow::array::StringArray>()
                    .ok_or_else(|| Error::InvalidField {
                        field: name.to_string(),
                        message: "Expected String dictionary values".to_string(),
                    })?;
                let key = keys.value(row);
                return Ok(values.value(key as usize).to_string());
            }

            Err(Error::InvalidField {
                field: name.to_string(),
                message: "Expected DictionaryArray".to_string(),
            })
        };

        // Extract signal row indices from list
        let signal_rows = {
            let col = batch
                .column_by_name("signal")
                .ok_or_else(|| Error::MissingField("signal".to_string()))?;
            let list_arr =
                col.as_any()
                    .downcast_ref::<ListArray>()
                    .ok_or_else(|| Error::InvalidField {
                        field: "signal".to_string(),
                        message: "Expected ListArray".to_string(),
                    })?;
            let values = list_arr.value(row);
            let u64_arr = values
                .as_any()
                .downcast_ref::<UInt64Array>()
                .ok_or_else(|| Error::InvalidField {
                    field: "signal".to_string(),
                    message: "Expected UInt64Array values".to_string(),
                })?;
            u64_arr.values().to_vec()
        };

        Ok(ReadData {
            read_id: get_uuid("read_id")?,
            read_number: get_u32("read_number")?,
            start_sample: get_u64("start")?,
            channel: get_u16("channel")?,
            well: get_u8("well")?,
            pore_type: get_dict_string("pore_type").unwrap_or_default(),
            calibration_offset: get_f32("calibration_offset")?,
            calibration_scale: get_f32("calibration_scale")?,
            median_before: get_f32("median_before")?,
            end_reason: get_dict_string("end_reason")
                .unwrap_or_default()
                .parse()
                .unwrap(),
            end_reason_forced: get_bool("end_reason_forced")?,
            run_info_index: 0, // TODO: parse from dictionary
            num_minknow_events: get_u64("num_minknow_events")?,
            num_samples: get_u64("num_samples")?,
            open_pore_level: get_f32("open_pore_level").unwrap_or(0.0),
            signal_rows,
        })
    }
}
