//! Main POD5 file reader.

use crate::CompressedSignalChunk;
use crate::arrow_helpers::{BatchFieldExtractor, ReadColumns, ReadsBatchView};
use crate::arrow_ipc::ArrowIpcFooter;
use crate::compression;
use crate::error::{Error, Result};
use crate::footer::{self, Footer};
use crate::types::{POD5_SIGNATURE, ReadData, RunInfoData, SECTION_MARKER_LENGTH, Uuid};
use arrow::ipc::reader::FileReader as ArrowFileReader;
use arrow::record_batch::RecordBatch;
use memmap2::Mmap;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::File;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::OnceLock;

use super::read_index::ReadIndex;
use super::read_iter::{ReadIterator, extract_read_from_batch};
use super::signal_extractor::SignalExtractor;

/// Signal-row + calibration data for a single read, returned by
/// `find_signal_rows_with_calibration_by_ids` and helpers.
#[allow(dead_code)]
pub(crate) struct SignalCalibration {
    pub read_id: Uuid,
    pub signal_rows: Vec<u64>,
    pub calibration_offset: f32,
    pub calibration_scale: f32,
}

/// Default maximum number of signal batches to cache.
const DEFAULT_MAX_CACHED_BATCHES: usize = 10;

/// A signal batch whose row count breaks the file's stride.
///
/// See [`Reader::nonuniform_signal_batch`] — a portability fault, not a read
/// error: escapepod resolves such a file correctly, other readers do not.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NonUniformSignalBatch {
    /// Index of the offending batch.
    pub index: usize,
    /// Rows it actually holds.
    pub rows: u64,
    /// Rows the stride implies (the first batch's count).
    pub expected: u64,
}

/// First non-final batch whose row count differs from the first batch's.
///
/// Split out from [`Reader::nonuniform_signal_batch`] so the rule is testable
/// without a malformed file — which the writer can no longer produce
/// (escapepod-rs#195), leaving no way to build one through the public API.
fn first_nonuniform_batch(counts: &[u64]) -> Option<NonUniformSignalBatch> {
    // Under 3 batches there is at most one non-final batch, and that one
    // defines the stride, so it cannot disagree with it.
    if counts.len() < 3 {
        return None;
    }
    let expected = counts[0];
    counts[..counts.len() - 1]
        .iter()
        .enumerate()
        .find(|(_, n)| **n != expected)
        .map(|(index, rows)| NonUniformSignalBatch {
            index,
            rows: *rows,
            expected,
        })
}

/// A reader for POD5 files.
pub struct Reader {
    /// Memory-mapped file data.
    mmap: Mmap,
    /// Parsed file footer.
    footer: Footer,
    /// Cached run info data.
    run_info_cache: Vec<RunInfoData>,
    /// Parsed Arrow IPC footer of the signal table (lazy — computed on first
    /// signal access). Owns only per-batch offset/row-count descriptors, so
    /// signal fetches locate a row and slice its compressed bytes straight out
    /// of the mmap via [`ArrowIpcFooter::extract_signal_rows`] — no whole-batch
    /// Arrow deserialization. `None` when the file has no signal table or the
    /// footer can't be parsed (callers fall back to the Arrow reader path).
    signal_ipc_footer: OnceLock<Option<ArrowIpcFooter>>,
    /// Cached read UUID index: UUID → (batch_idx, row_within_batch).
    /// Lazily built on first lookup via `.p5s` sidecar or column-projected scan.
    read_index: OnceLock<ReadIndex>,
    /// Path to the POD5 file (for locating `.p5s` sidecar).
    file_path: Option<PathBuf>,
}

/// Probe the POD5 header and footer through ordinary buffered I/O *before* the
/// file is memory-mapped.
///
/// Memory-mapping a truncated file or an archive "stub" — a placeholder whose
/// size is correct in metadata but whose data is not actually resident, common
/// on HSM / tape-backed filesystems — and then faulting a page whose backing
/// store is unavailable raises SIGBUS, an uncatchable signal that aborts the
/// process. Reading the same bytes with `read()` instead surfaces a recoverable
/// [`std::io::Error`]. We touch only the header and footer, where POD5 keeps its
/// structural metadata; this catches the overwhelmingly common stub / truncation
/// case at `open()` for the cost of two small reads, mirroring upstream pod5
/// 0.3.37. It does not guarantee the interior signal pages are resident, so a
/// partially-materialised stub can still SIGBUS on a later scan — the same
/// scope upstream chose. Structural validation is left to `parse_footer`, which
/// runs against the mmap immediately afterwards.
fn probe_header_footer(file: &File) -> Result<()> {
    use std::io::{Read, Seek, SeekFrom};

    // Trailer laid out at end-of-file: footer_len(8) + section_marker + trailing
    // signature(8).
    const TRAILER: u64 = 8 + SECTION_MARKER_LENGTH as u64 + 8;

    let file_len = file.metadata()?.len();
    // Too small to be a POD5 file — let the signature / footer checks that run
    // after mmap produce the precise diagnostic.
    if file_len < 8 + TRAILER {
        return Ok(());
    }

    let mut reader: &File = file;

    // Header signature page.
    let mut header = [0u8; 8];
    reader.seek(SeekFrom::Start(0))?;
    reader.read_exact(&mut header)?;

    // Footer trailer (footer_len + section marker + trailing signature).
    let mut trailer = [0u8; TRAILER as usize];
    reader.seek(SeekFrom::Start(file_len - TRAILER))?;
    reader.read_exact(&mut trailer)?;
    let footer_len = i64::from_le_bytes(trailer[0..8].try_into().unwrap());

    // Fault the FlatBuffer footer + FOOTER magic too when the recorded length is
    // plausible. Bounded by the file itself and capped at 1 MiB so a corrupt
    // length can never trigger a huge read — parse_footer re-validates from the
    // mmap regardless.
    if (0..=file_len as i64).contains(&footer_len) {
        let region = ((footer_len as u64) + 8).min(1 << 20);
        if let Some(start) = (file_len - TRAILER).checked_sub(region) {
            let mut sink = vec![0u8; region as usize];
            reader.seek(SeekFrom::Start(start))?;
            reader.read_exact(&mut sink)?;
        }
    }

    Ok(())
}

/// Read count above which building the in-memory read-id index automatically
/// is worth announcing. Overridable via `ESCAPEPOD_AUTOINDEX_MAX`.
///
/// This is a threshold on *speculative* work, not on demanded work, and the
/// distinction is the whole point. A caller that has asked for random access
/// — [`Reader::reads_by_ids`] and friends — always gets an index, however
/// large the file, because the only alternative is a full scan of the reads
/// table on every call, which is orders of magnitude worse than the memory it
/// would save. Above this threshold that build is reported at `warn` rather
/// than `debug`, naming `escpod index`, because it is a cost worth knowing
/// about even though it is the right cost to pay.
///
/// A caller that is merely *guessing* that random access is coming — the
/// Python context-manager warm-up — should still gate on it, so that opening
/// a huge file and only iterating it does not pay for an index nobody asked
/// for.
///
/// It is deliberately not a memory guard: [`Reader::read_index`] loads a
/// `.p5s` sidecar of any size uncapped, so the same file with a sidecar
/// already holds the same entries in memory. The threshold decides where an
/// index comes from and how loudly, never whether one exists.
pub fn autoindex_max() -> usize {
    std::env::var("ESCAPEPOD_AUTOINDEX_MAX")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5_000_000)
}

impl Reader {
    /// Open a POD5 file for reading.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::open_with_cache_size(path, DEFAULT_MAX_CACHED_BATCHES)
    }

    /// Open a POD5 file. The `cache_size` argument is retained for API
    /// compatibility but no longer used: signal is now read via a zero-copy,
    /// lock-free path that slices compressed bytes directly from the mmap, so
    /// there is no per-batch cache to size.
    pub fn open_with_cache_size<P: AsRef<Path>>(path: P, _cache_size: usize) -> Result<Self> {
        let file_path = path.as_ref().to_path_buf();
        let file = File::open(&file_path)?;

        // Defensive pre-mmap probe (see `probe_header_footer`). Reading the
        // structural metadata through ordinary I/O first turns the common
        // truncated-file / archive-stub failure into a recoverable error
        // instead of an uncatchable SIGBUS on first page fault. Set
        // `POD5_DISABLE_MMAP_OPEN=1` to skip it and map straight away.
        if std::env::var_os("POD5_DISABLE_MMAP_OPEN").is_none() {
            probe_header_footer(&file)?;
        }

        let mmap = unsafe { Mmap::map(&file)? };

        // Hint the OS to stream the mapping. POD5 access is overwhelmingly a
        // single front-to-back scan of the (large) signal + reads tables —
        // `view`, `filter`, `merge`, `repack`, and `demux` all read the whole
        // file once. Without a readahead hint the kernel faults mmap pages 4 KiB
        // at a time on demand; on a network filesystem (BeeGFS/Lustre/NFS) each
        // fault is a server round-trip, collapsing effective throughput to a few
        // MB/s even though the FS streams sequentially at ~1 GB/s — the dominant
        // cost (and the multi-minute "slow to get going" stall) on large remote
        // POD5 files. `MADV_SEQUENTIAL` widens the readahead window so the
        // mapping streams near raw bandwidth. It is a hint (ignored where
        // unsupported); single-read random access pays only a negligible
        // over-readahead cost.
        #[cfg(unix)]
        let _ = mmap.advise(memmap2::Advice::Sequential);

        // Verify signature at start
        if mmap.len() < 8 || mmap[..8] != POD5_SIGNATURE {
            return Err(Error::InvalidSignature);
        }

        // Parse footer
        let footer = footer::parse_footer(&mmap)?;

        // Load run info eagerly (it's usually small)
        let run_info_cache = Self::load_run_info(&mmap, &footer)?;

        let reader = Self {
            mmap,
            footer,
            run_info_cache,
            signal_ipc_footer: OnceLock::new(),
            read_index: OnceLock::new(),
            file_path: Some(file_path),
        };
        Ok(reader)
    }

    /// Lazily parse and cache the signal table's Arrow IPC footer.
    ///
    /// Reads only the footer (a few KB at the end of the signal table) to
    /// enumerate per-batch offsets and row counts, avoiding deserialization of
    /// any signal batch (which can be 50-100 MB on large files). The parsed
    /// footer owns nothing but these descriptors, so signal fetches locate a
    /// row and slice its compressed bytes straight out of the mmap. Returns
    /// `None` when the file has no signal table, the table extends past EOF, or
    /// the footer is empty/unparseable — callers fall back to the Arrow reader.
    fn signal_ipc_footer(&self) -> Option<&ArrowIpcFooter> {
        self.signal_ipc_footer
            .get_or_init(|| {
                let embedded = self.footer.signal_table()?;
                let start = embedded.offset as usize;
                let end = start + embedded.length as usize;
                let slice = self.mmap.get(start..end)?;
                let footer = ArrowIpcFooter::parse(slice).ok()?;
                if footer.record_batches.is_empty() || footer.total_rows == 0 {
                    return None;
                }
                Some(footer)
            })
            .as_ref()
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

        // Seek directly to the batch via the IPC footer's block offsets (O(1))
        // instead of decoding every preceding batch.
        reader.set_index(index)?;

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

    /// Collect every read in the file into a single `Vec<ReadData>`.
    ///
    /// Functionally equivalent to `reads()?.collect()`, but resolves
    /// column lookups once per batch via `ReadsBatchView` instead of once
    /// per row. This is the hot path for merge (which materializes every
    /// read of every input file) and for filter's non-UUID criteria path.
    pub fn collect_all_reads(&self) -> Result<Vec<ReadData>> {
        let mut out: Vec<ReadData> = Vec::new();
        for batch_result in self.read_batches()? {
            let batch = batch_result?;
            let view = ReadsBatchView::new(&batch, false)?;
            out.reserve(view.num_rows());
            for row in 0..view.num_rows() {
                out.push(view.read(row)?);
            }
        }
        Ok(out)
    }

    /// Collect every read's metadata into a [`ReadColumns`] struct-of-arrays.
    ///
    /// The columnar counterpart to [`Self::collect_all_reads`]: it skips the
    /// per-read `ReadData` (and its `signal_rows` `Vec`) and fills numeric
    /// columns by a bulk slice copy from each batch's Arrow buffers. This is the
    /// fast backing for the Python `to_dict`/`to_pandas`/`to_polars` metadata
    /// exports, where `signal_rows` is never used.
    pub fn read_columns(&self) -> Result<ReadColumns> {
        let mut cols = ReadColumns::default();
        if let Ok(n) = self.read_count() {
            cols.reserve(n);
        }
        for batch_result in self.read_batches()? {
            let batch = batch_result?;
            let view = ReadsBatchView::new(&batch, false)?;
            view.append_columns(&mut cols)?;
        }
        Ok(cols)
    }

    /// Iterate the reads-table batches as raw Arrow `RecordBatch`es.
    ///
    /// This is the streaming counterpart to `collect_all_reads`. Hot
    /// consumers that don't want to materialize every read up front
    /// (e.g. `repack`, `resquiggle`'s read indexer, `demux fingerprint`'s
    /// pre-filter) should iterate batches and build a `ReadsBatchView`
    /// per batch:
    ///
    /// ```ignore
    /// for batch_result in reader.read_batches()? {
    ///     let batch = batch_result?;
    ///     let view = ReadsBatchView::new(&batch, false)?;
    ///     for row in 0..view.num_rows() {
    ///         let read = view.read(row)?;
    ///         // ...
    ///     }
    /// }
    /// ```
    ///
    /// This avoids the per-row `column_by_name` lookups that
    /// `Reader::reads()`'s row-yielding iterator pays.
    pub fn read_batches(&self) -> Result<impl Iterator<Item = Result<RecordBatch>> + '_> {
        let embedded = self
            .footer
            .reads_table()
            .ok_or_else(|| Error::MissingField("reads table".to_string()))?;
        let reader = self.create_arrow_reader(embedded)?;
        Ok(reader.map(|r| r.map_err(Error::from)))
    }

    /// Distinct `pore_type` and end_reason dictionary labels for the reads
    /// table. Truly O(dict): a POD5 reads table is a single Arrow IPC stream
    /// with one shared dictionary per dictionary-encoded column (no IPC
    /// dictionary replacement/deltas), so every record batch carries the same
    /// `pore_type` / end_reason dictionary values. We therefore read only the
    /// **first** batch's dictionaries rather than scanning the whole table.
    ///
    /// Scanning every batch (the previous implementation) decoded the entire
    /// reads table — including the large `signal` row-index list column — up
    /// front, which stalled `demux` for minutes before the first read was
    /// processed on multi-GB / multi-million-read files. Useful for
    /// pre-declaring a writer's dictionaries when block-copying reads into one
    /// or more output files.
    pub fn reads_dictionaries(&self) -> Result<(Vec<String>, Vec<String>)> {
        let Some(batch) = self.read_batches()?.next() else {
            // Empty reads table (no batches) → no dictionary labels.
            return Ok((Vec::new(), Vec::new()));
        };
        let batch = batch?;
        let view = crate::arrow_helpers::ReadsBatchView::new(&batch, false)?;
        Ok((view.pore_type_dict(), view.end_reason_dict()))
    }

    /// Get the total number of reads.
    ///
    /// Parses the reads-table Arrow IPC footer to sum each
    /// `BatchBlock::row_count` — O(num_batches), not O(num_reads). On a
    /// 2.96M-read POD5 this is microseconds versus tens of milliseconds
    /// for the previous full-scan implementation.
    pub fn read_count(&self) -> Result<usize> {
        let bytes = self.reads_table_bytes()?;
        let footer = crate::arrow_ipc::ArrowIpcFooter::parse(bytes)?;
        Ok(footer.total_rows as usize)
    }

    /// Raw bytes of the reads table (Arrow IPC stream slice into the mmap).
    fn reads_table_bytes(&self) -> Result<&[u8]> {
        let embedded = self
            .footer
            .reads_table()
            .ok_or_else(|| Error::MissingField("reads table".to_string()))?;
        let start = embedded.offset as usize;
        let end = start + embedded.length as usize;
        if end > self.mmap.len() {
            return Err(Error::InvalidFooter(format!(
                "Reads table extends beyond file: {}..{} > {}",
                start,
                end,
                self.mmap.len()
            )));
        }
        Ok(&self.mmap[start..end])
    }

    /// Get signal data for a read.
    ///
    /// The `signal_rows` parameter should be the signal row indices from the
    /// read record. Slices each row's compressed bytes directly out of the
    /// mmap via the cached signal footer and VBZ-decodes only that row — it
    /// never deserializes the surrounding batch. Thread-safe and lock-free:
    /// the same primitive backs the parallel [`SignalExtractor`].
    pub fn get_signal(&self, signal_rows: &[u64]) -> Result<Vec<i16>> {
        self.get_signal_prefix(signal_rows, usize::MAX)
    }

    /// Like [`Self::get_signal`] but decodes at most the first `max_samples`
    /// samples of the read — identical to `get_signal(..)[..max_samples]`, and
    /// shorter when the read is.
    ///
    /// For a consumer that already knows its boundary (adapter detection,
    /// fingerprinting, barcode windows) this skips the SVB16 decode of the
    /// unread tail, plus whole 128 KiB ZSTD blocks once a read is long enough
    /// to span several. It cannot skip a partial ZSTD block, so a read that
    /// fits in one — anything up to ~110k samples — saves the SVB16 stage only.
    pub fn get_signal_prefix(&self, signal_rows: &[u64], max_samples: usize) -> Result<Vec<i16>> {
        let Some(footer) = self.signal_ipc_footer() else {
            // No parseable signal footer (missing table / edge case): fall back
            // to Arrow's own IPC reader, which has no prefix path of its own.
            let mut all = self.get_signal_fallback(signal_rows)?;
            all.truncate(max_samples);
            return Ok(all);
        };

        let signal_bytes = self.signal_table_bytes()?;
        let raw_chunks = footer.extract_signal_rows(signal_rows, signal_bytes)?;
        super::signal_extractor::decode_chunks(&raw_chunks, max_samples)
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
    /// Row count of every signal batch, straight from the Arrow IPC footer.
    ///
    /// O(number of batches) and no signal decode — the footer already carries
    /// the descriptors. Empty when the file has no signal table.
    pub fn signal_batch_row_counts(&self) -> Vec<u64> {
        self.signal_ipc_footer()
            .map(|f| f.record_batches.iter().map(|b| b.row_count).collect())
            .unwrap_or_default()
    }

    /// The first non-final signal batch whose row count differs from the
    /// first batch's, if the file has one.
    ///
    /// This crate reads signal by walking cumulative per-batch row counts, so a
    /// file like this reads back correctly HERE. Other readers — including the
    /// official `pod5` library and dorado — resolve a read's global signal
    /// index by assuming a constant batch stride, and for them one oversized
    /// batch shifts every index after it. So this is a **portability** fault,
    /// not a corruption we cannot handle, and the distinction matters: such a
    /// file looks perfect to `escpod` and silently loses reads in dorado.
    ///
    /// The final batch is exempt — it is legitimately short.
    pub fn nonuniform_signal_batch(&self) -> Option<NonUniformSignalBatch> {
        first_nonuniform_batch(&self.signal_batch_row_counts())
    }

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
        self.get_signal_bulk_prefix(reads, usize::MAX)
    }

    /// Like [`Self::get_signal_bulk`] but decodes at most the first
    /// `max_samples` samples of each read. See [`Self::get_signal_prefix`] for
    /// what a prefix does and does not save.
    pub fn get_signal_bulk_prefix<K: Clone + Send>(
        &self,
        reads: &[(K, Vec<u64>)],
        max_samples: usize,
    ) -> Result<Vec<(K, Vec<i16>)>> {
        use crate::arrow_ipc::ArrowIpcFooter;
        use rayon::prelude::*;

        let signal_bytes = self.signal_table_bytes()?;
        let signal_footer = ArrowIpcFooter::parse(signal_bytes)?;

        // Flatten every read's rows into one list so the extraction is a single
        // batch-grouped, ascending-order sweep (see `get_compressed_signal_bulk`
        // for why that matters on a cold network filesystem).
        let row_indices: Vec<u64> = reads
            .iter()
            .flat_map(|(_key, rows)| rows.iter().copied())
            .collect();
        let raw_chunks = signal_footer.extract_signal_rows(&row_indices, signal_bytes)?;
        if raw_chunks.len() != row_indices.len() {
            // A requested row was out of bounds, so the returned chunks no
            // longer line up positionally with the reads that asked for them.
            return Err(Error::InvalidFooter(format!(
                "signal table returned {} chunks for {} requested rows",
                raw_chunks.len(),
                row_indices.len()
            )));
        }

        // Re-slice the flat chunk list back into per-read runs — the extraction
        // preserved input order, so each read's chunks are contiguous and in
        // the order its `signal_rows` listed them.
        let mut per_read: Vec<&[crate::arrow_ipc::RawSignalChunk<'_>]> =
            Vec::with_capacity(reads.len());
        let mut offset = 0usize;
        for (_key, rows) in reads {
            per_read.push(&raw_chunks[offset..offset + rows.len()]);
            offset += rows.len();
        }

        // Decompress in parallel; VBZ decode is CPU-bound and reads are
        // independent. One read per task, so the per-read prefix budget stays
        // local to the task that spends it.
        let decoded: Vec<Result<Vec<i16>>> = per_read
            .par_iter()
            .map(|chunks| super::signal_extractor::decode_chunks(chunks, max_samples))
            .collect();

        reads
            .iter()
            .zip(decoded)
            .map(|((key, _), signal)| Ok((key.clone(), signal?)))
            .collect()
    }

    /// Bulk-extract **compressed** signal chunks for many reads in one pass.
    ///
    /// Like [`Self::get_signal_bulk`] but returns the raw VBZ bytes without
    /// decompressing — the primitive a demux/scan front-end wants when it will
    /// decompress on its own worker threads.
    ///
    /// Crucially, this uses the batch-grouped raw-byte path
    /// ([`ArrowIpcFooter::extract_signal_rows`]): every requested row is grouped
    /// by batch and each batch's bytes are read exactly once, in **ascending
    /// file order**. On a cold network filesystem that turns the scattered,
    /// per-read page faults of [`Self::get_compressed_signal_for_rows`] (which
    /// defeat kernel readahead) into a single sequential sweep. Doing one bulk
    /// call per worker chunk — instead of N parallel single-read calls — is the
    /// difference between single-digit MB/s and near-disk bandwidth on BeeGFS.
    pub fn get_compressed_signal_bulk<K: Clone>(
        &self,
        reads: &[(K, Vec<u64>)],
    ) -> Result<Vec<(K, Vec<CompressedSignalChunk>)>> {
        use crate::arrow_ipc::ArrowIpcFooter;

        let signal_bytes = self.signal_table_bytes()?;
        let footer = ArrowIpcFooter::parse(signal_bytes)?;

        // Flatten to a single row list, keeping a back-reference to which read
        // and which position-within-read each row came from.
        let mut back_refs: Vec<(usize, usize)> = Vec::new();
        let mut row_indices: Vec<u64> = Vec::new();
        for (read_idx, (_key, rows)) in reads.iter().enumerate() {
            for (chunk_idx, &row) in rows.iter().enumerate() {
                back_refs.push((read_idx, chunk_idx));
                row_indices.push(row);
            }
        }

        // One batch-grouped, ascending-order sweep over the signal table.
        let raw_chunks = footer.extract_signal_rows(&row_indices, signal_bytes)?;
        if raw_chunks.len() != row_indices.len() {
            // A requested row was out of bounds; positional alignment with
            // `back_refs` would be wrong. Fall back to the per-read path.
            return reads
                .iter()
                .map(|(k, rows)| {
                    self.get_compressed_signal_for_rows(rows)
                        .map(|c| (k.clone(), c))
                })
                .collect();
        }

        // Reassemble per read, preserving chunk order within each read.
        let mut per_read: Vec<Vec<(usize, CompressedSignalChunk)>> = vec![Vec::new(); reads.len()];
        for (i, raw) in raw_chunks.iter().enumerate() {
            let (read_idx, chunk_idx) = back_refs[i];
            per_read[read_idx].push((
                chunk_idx,
                CompressedSignalChunk {
                    read_id: Uuid::from_bytes(raw.read_id),
                    samples: raw.samples,
                    data: Arc::from(raw.signal),
                },
            ));
        }

        let mut results = Vec::with_capacity(reads.len());
        for ((key, _), mut chunks) in reads.iter().zip(per_read) {
            chunks.sort_by_key(|(idx, _)| *idx);
            let v: Vec<CompressedSignalChunk> = chunks.into_iter().map(|(_, c)| c).collect();
            results.push((key.clone(), v));
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
    ///
    /// Slices each requested row's compressed bytes directly out of the mmap via
    /// the cached signal footer — no whole-batch Arrow deserialization. For many
    /// rows in one call, prefer [`Self::get_compressed_signal_bulk`], which reads
    /// batches in ascending file order for sequential I/O.
    pub fn get_compressed_signal_for_rows(
        &self,
        signal_rows: &[u64],
    ) -> Result<Vec<CompressedSignalChunk>> {
        let Some(footer) = self.signal_ipc_footer() else {
            // Fallback: load all and filter (less efficient)
            let all_signal = self.get_all_signal_compressed()?;
            let mut result = Vec::with_capacity(signal_rows.len());
            for &idx in signal_rows {
                if let Some(chunk) = all_signal.get(idx as usize) {
                    result.push(chunk.clone());
                }
            }
            return Ok(result);
        };

        let signal_bytes = self.signal_table_bytes()?;
        let raw_chunks = footer.extract_signal_rows(signal_rows, signal_bytes)?;
        Ok(raw_chunks
            .iter()
            .map(|raw| CompressedSignalChunk {
                read_id: Uuid::from_bytes(raw.read_id),
                samples: raw.samples,
                data: Arc::from(raw.signal),
            })
            .collect())
    }

    /// Extract compressed signal chunks from a batch.
    fn extract_compressed_signal_from_batch(
        &self,
        batch: &RecordBatch,
        chunks: &mut Vec<CompressedSignalChunk>,
    ) -> Result<()> {
        use arrow::array::AsArray;
        use arrow::datatypes::UInt32Type;

        let read_id_col = batch
            .column_by_name("read_id")
            .ok_or_else(|| Error::MissingField("read_id column".to_string()))?;
        let signal_col = batch
            .column_by_name("signal")
            .ok_or_else(|| Error::MissingField("signal column".to_string()))?;
        let samples_col = batch
            .column_by_name("samples")
            .ok_or_else(|| Error::MissingField("samples column".to_string()))?;

        let read_id_array =
            read_id_col
                .as_fixed_size_binary_opt()
                .ok_or_else(|| Error::InvalidField {
                    field: "read_id".to_string(),
                    message: "Expected FixedSizeBinaryArray".to_string(),
                })?;

        let signal_array =
            signal_col
                .as_binary_opt::<i64>()
                .ok_or_else(|| Error::InvalidField {
                    field: "signal".to_string(),
                    message: "Expected LargeBinaryArray".to_string(),
                })?;

        let samples_array = samples_col
            .as_primitive_opt::<UInt32Type>()
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
        use arrow::array::AsArray;
        use arrow::datatypes::UInt32Type;

        // Get signal column (LargeBinary with VBZ data)
        let signal_col = batch
            .column_by_name("signal")
            .ok_or_else(|| Error::MissingField("signal column".to_string()))?;

        // Get samples column for count
        let samples_col = batch
            .column_by_name("samples")
            .ok_or_else(|| Error::MissingField("samples column".to_string()))?;

        let samples_array = samples_col
            .as_primitive_opt::<UInt32Type>()
            .ok_or_else(|| Error::InvalidField {
                field: "samples".to_string(),
                message: "Expected UInt32Array".to_string(),
            })?;

        let sample_count = samples_array.value(row) as usize;

        // Handle signal data (could be LargeBinary for VBZ)
        let signal_array =
            signal_col
                .as_binary_opt::<i64>()
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
        let embedded = self
            .footer
            .reads_table()
            .ok_or_else(|| Error::MissingField("reads table".to_string()))?;

        // Create reader with projection for just the read_id column (index 0)
        let reader = self.create_arrow_reader_with_projection(embedded, Some(vec![0]))?;

        let mut read_ids = Vec::new();
        for batch_result in reader {
            Self::extract_uuids_from_batch(&batch_result?, &mut read_ids);
        }

        Ok(read_ids)
    }

    /// Append every valid read-id UUID from column 0 (a `FixedSizeBinaryArray`)
    /// of a read_id-projected reads batch to `out`. Rows that don't parse as a
    /// UUID are skipped; a non-binary column 0 contributes nothing.
    fn extract_uuids_from_batch(batch: &RecordBatch, out: &mut Vec<Uuid>) {
        use arrow::array::{Array, AsArray};
        if let Some(col) = batch.column(0).as_fixed_size_binary_opt() {
            out.reserve(col.len());
            for row in 0..col.len() {
                if let Ok(uuid) = Uuid::from_slice(col.value(row)) {
                    out.push(uuid);
                }
            }
        }
    }

    /// Get read IDs from a specific batch efficiently (reads only the read_id column).
    pub fn read_ids_from_batch(&self, batch_idx: usize) -> Result<Vec<Uuid>> {
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

        // Seek directly to the batch via the IPC footer's block offsets (O(1))
        // instead of decoding every preceding batch.
        reader.set_index(batch_idx)?;

        let batch = reader.next().ok_or_else(|| Error::BatchIndexOutOfBounds {
            index: batch_idx,
            max: reader.num_batches(),
        })??;

        let mut read_ids = Vec::new();
        Self::extract_uuids_from_batch(&batch, &mut read_ids);

        Ok(read_ids)
    }

    // ------------------------------------------------------------------
    // Read index: cached UUID → (batch_idx, row) mapping
    // ------------------------------------------------------------------

    /// Get the path to the `.p5s` sidecar file for this POD5.
    fn p5s_path(&self) -> Option<PathBuf> {
        self.file_path.as_ref().map(crate::sidecar::sidecar_path)
    }

    /// The identity a `.p5s` sidecar for this file must be bound to.
    pub fn sidecar_identity(&self) -> Result<crate::sidecar::Pod5Identity> {
        let file_id = Uuid::parse_str(self.file_identifier())
            .map_err(|e| Error::InvalidUuid(e.to_string()))?;
        Ok(crate::sidecar::Pod5Identity {
            file_id,
            size: self.mmap.len() as u64,
        })
    }

    /// Build the read index from a column-projected scan of the reads table.
    ///
    /// Projects only column 0 (read_id) to avoid parsing all 22 columns.
    pub(crate) fn build_read_index_from_scan(&self) -> Result<ReadIndex> {
        use arrow::array::{Array, AsArray};

        let embedded = self
            .footer
            .reads_table()
            .ok_or_else(|| Error::MissingField("reads table".to_string()))?;

        let started = std::time::Instant::now();
        let reader = self.create_arrow_reader_with_projection(embedded, Some(vec![0]))?;

        let mut batches = 0usize;
        let mut entries = Vec::new();
        for (batch_idx, batch_result) in reader.enumerate() {
            let batch = batch_result?;
            let col =
                batch
                    .column(0)
                    .as_fixed_size_binary_opt()
                    .ok_or_else(|| Error::InvalidField {
                        field: "read_id".to_string(),
                        message: "Expected FixedSizeBinaryArray".to_string(),
                    })?;
            batches += 1;
            entries.reserve(col.len());
            for row in 0..col.len() {
                let bytes = col.value(row);
                if bytes.len() == 16 {
                    let uuid_bytes: [u8; 16] = bytes.try_into().unwrap();
                    entries.push((uuid_bytes, batch_idx as u32, row as u32));
                }
            }
        }
        entries.sort_unstable_by_key(|e| e.0);
        tracing::debug!(
            reads = entries.len(),
            batches,
            elapsed_ms = started.elapsed().as_millis(),
            "read index built from a projected scan"
        );
        Ok(ReadIndex { entries })
    }

    /// Try to load the read index from the `.p5s` sidecar.
    ///
    /// Returns `Ok(None)` if no sidecar exists.
    /// Returns `Err` if one exists but is invalid or stale.
    fn load_sidecar_index(&self) -> Result<Option<ReadIndex>> {
        let p5s_path = match self.p5s_path() {
            Some(p) => p,
            None => return Ok(None),
        };
        let sidecar = match crate::sidecar::read_sidecar_file(&p5s_path, &self.sidecar_identity()?)?
        {
            Some(s) => s,
            None => return Ok(None),
        };
        // Sidecar entries are already UUID-sorted.
        Ok(Some(ReadIndex {
            entries: sidecar.entries().to_vec(),
        }))
    }

    /// Get or lazily build the read UUID index.
    ///
    /// Checks for a `.p5s` sidecar first and falls back to a column-projected
    /// scan of the reads table. Either way the result is cached on the reader,
    /// so the cost is paid once per `Reader`, not once per lookup.
    ///
    /// A sidecar that exists but is bound to another POD5 is an **error**, not
    /// a reason to fall back: it is never silently stepped over in favour of a
    /// scan that would have worked, because that would turn a stale index into
    /// a slow success instead of a loud failure.
    pub fn read_index(&self) -> Result<&ReadIndex> {
        if let Some(idx) = self.read_index.get() {
            return Ok(idx);
        }
        // Build outside the lock — may race with another thread, but
        // get_or_init will discard the extra copy.
        let index = match self.load_sidecar_index()? {
            Some(index) => index,
            None => {
                // Loud above the threshold, because a build of that size is
                // worth knowing about — but still a build. Falling back to a
                // scan here is the bug this replaced (escapepod-rs#251).
                let reads = self.read_count().unwrap_or(0);
                if reads > autoindex_max() {
                    tracing::warn!(
                        file = ?self.file_path,
                        reads,
                        "no .p5s sidecar; building an in-memory read index for a \
                         large file — run `escpod index` to persist it and skip \
                         this on every future open"
                    );
                } else {
                    tracing::debug!(
                        file = ?self.file_path,
                        reads,
                        "no .p5s sidecar; building the read index in memory \
                         (`escpod index` persists it across processes)"
                    );
                }
                self.build_read_index_from_scan()?
            }
        };
        Ok(self.read_index.get_or_init(|| index))
    }

    /// Build the read index and write it to the `.p5s` sidecar.
    ///
    /// Annotations in a valid existing sidecar are preserved; a stale or
    /// unreadable sidecar is replaced outright (its contents were bound to
    /// a different file and unusable anyway). This is called by the
    /// `escpod index` CLI command.
    pub fn build_and_write_index<P: AsRef<Path>>(&self, output: P) -> Result<usize> {
        let index = self.build_read_index_from_scan()?;
        let count = index.len();
        let identity = self.sidecar_identity()?;

        let mut sidecar = match crate::sidecar::read_sidecar_file(output.as_ref(), &identity) {
            Ok(Some(existing)) => existing,
            Ok(None) | Err(_) => crate::sidecar::Sidecar::default(),
        };
        sidecar.set_entries(index.entries.clone());
        crate::sidecar::write_sidecar_file(output.as_ref(), &identity, &sidecar)?;
        Ok(count)
    }

    // ------------------------------------------------------------------
    // Targeted batch access — indexed or single-pass
    // ------------------------------------------------------------------

    /// Look up signal rows for a set of target UUIDs.
    ///
    /// Goes through the read index, which is built on first use and cached
    /// (see [`Self::read_index`]), so only batches holding a target are
    /// touched. See [`Self::reads_by_ids`] for why this is preferred to a
    /// scan even on the first call.
    pub fn find_signal_rows_by_ids(
        &self,
        target_ids: &HashSet<Uuid>,
    ) -> Result<Vec<(Uuid, Vec<u64>)>> {
        self.find_signal_rows_indexed(target_ids)
    }

    /// Look up signal rows and calibration data for a set of target UUIDs.
    ///
    /// Same strategy as [`Self::find_signal_rows_by_ids`].
    #[allow(dead_code)]
    pub(crate) fn find_signal_rows_with_calibration_by_ids(
        &self,
        target_ids: &HashSet<Uuid>,
    ) -> Result<Vec<SignalCalibration>> {
        self.find_signal_rows_with_calibration_indexed(target_ids)
    }

    /// Retrieve full `ReadData` for a set of target UUIDs.
    ///
    /// Groups targets by batch through the read index, opens a full-column
    /// reader, and seeks directly to each target batch — only batches that
    /// contain a target UUID are deserialized.
    ///
    /// This always goes through [`Self::read_index`], building the index on
    /// the first call when no `.p5s` sidecar exists, rather than falling back
    /// to a scan. Building is not the more expensive option even for a single
    /// call: the index *is* a scan, projected to the `read_id` column, so it
    /// moves strictly fewer bytes than one execution of the full 22-column
    /// scan it replaces — and it is cached on the reader, so every later
    /// lookup is a seek instead of another pass over the file.
    ///
    /// The early exit a scan offers ("stop once all targets are found") does
    /// not rescue it in the access pattern that matters: targets typically
    /// arrive in BAM order, which is unrelated to POD5 storage order, so the
    /// last one sits near EOF and the scan runs to the end of the file anyway.
    ///
    /// Persist the index with `escpod index` to skip the build entirely.
    pub fn reads_by_ids(&self, target_ids: &HashSet<Uuid>) -> Result<Vec<ReadData>> {
        if target_ids.is_empty() {
            return Ok(Vec::new());
        }
        self.reads_by_ids_indexed(target_ids)
    }

    fn reads_by_ids_indexed(&self, target_ids: &HashSet<Uuid>) -> Result<Vec<ReadData>> {
        let index = self.read_index()?;

        let mut batch_targets: BTreeMap<usize, Vec<(Uuid, usize)>> = BTreeMap::new();
        for uuid in target_ids {
            if let Some((batch_idx, row_idx)) = index.get(uuid) {
                batch_targets
                    .entry(batch_idx)
                    .or_default()
                    .push((*uuid, row_idx));
            }
        }

        let embedded = self
            .footer
            .reads_table()
            .ok_or_else(|| Error::MissingField("reads table".to_string()))?;
        let mut reader = self.create_arrow_reader(embedded)?;

        let mut results = Vec::with_capacity(target_ids.len());
        for (batch_idx, targets) in batch_targets {
            reader.set_index(batch_idx)?;
            let batch = reader.next().ok_or_else(|| Error::BatchIndexOutOfBounds {
                index: batch_idx,
                max: reader.num_batches(),
            })??;
            // Resolve columns once per batch, then loop targets.
            let view = ReadsBatchView::new(&batch, true)?;
            for (uuid, row) in targets {
                view.verify_row(uuid, batch_idx, row)?;
                results.push(view.read(row)?);
            }
        }
        Ok(results)
    }

    // ---- Indexed path (two-pass: index lookup → targeted batch fetch) ----

    fn find_signal_rows_indexed(
        &self,
        target_ids: &HashSet<Uuid>,
    ) -> Result<Vec<(Uuid, Vec<u64>)>> {
        use arrow::array::AsArray;
        use arrow::datatypes::UInt64Type;

        let index = self.read_index()?;

        let mut batch_targets: BTreeMap<usize, Vec<(Uuid, usize)>> = BTreeMap::new();
        for uuid in target_ids {
            if let Some((batch_idx, row_idx)) = index.get(uuid) {
                batch_targets
                    .entry(batch_idx)
                    .or_default()
                    .push((*uuid, row_idx));
            }
        }

        let embedded = self
            .footer
            .reads_table()
            .ok_or_else(|| Error::MissingField("reads table".to_string()))?;
        let mut reader = self.create_arrow_reader_with_projection(embedded, Some(vec![0, 1]))?;

        let mut results = Vec::with_capacity(target_ids.len());
        for (batch_idx, targets) in batch_targets {
            reader.set_index(batch_idx)?;
            let batch = reader.next().ok_or_else(|| Error::BatchIndexOutOfBounds {
                index: batch_idx,
                max: reader.num_batches(),
            })??;
            let signal_col =
                batch
                    .column(1)
                    .as_list_opt::<i32>()
                    .ok_or_else(|| Error::InvalidField {
                        field: "signal".to_string(),
                        message: "Expected ListArray".to_string(),
                    })?;
            // read_id is already in the projection; without this the returned
            // signal would carry the *queried* UUID whatever row it came from.
            let read_ids = crate::arrow_helpers::read_id_column(&batch)?;
            for (uuid, row) in targets {
                crate::arrow_helpers::verify_index_row(read_ids, uuid, batch_idx, row)?;
                let values = signal_col.value(row);
                let u64_arr =
                    values
                        .as_primitive_opt::<UInt64Type>()
                        .ok_or_else(|| Error::InvalidField {
                            field: "signal".to_string(),
                            message: "Expected UInt64Array values".to_string(),
                        })?;
                results.push((uuid, u64_arr.values().to_vec()));
            }
        }
        Ok(results)
    }

    fn find_signal_rows_with_calibration_indexed(
        &self,
        target_ids: &HashSet<Uuid>,
    ) -> Result<Vec<SignalCalibration>> {
        use arrow::array::AsArray;
        use arrow::datatypes::{Float32Type, UInt64Type};

        let index = self.read_index()?;

        let mut batch_targets: BTreeMap<usize, Vec<(Uuid, usize)>> = BTreeMap::new();
        for uuid in target_ids {
            if let Some((batch_idx, row_idx)) = index.get(uuid) {
                batch_targets
                    .entry(batch_idx)
                    .or_default()
                    .push((*uuid, row_idx));
            }
        }

        let embedded = self
            .footer
            .reads_table()
            .ok_or_else(|| Error::MissingField("reads table".to_string()))?;
        let mut reader =
            self.create_arrow_reader_with_projection(embedded, Some(vec![0, 1, 16, 17]))?;

        let mut results = Vec::with_capacity(target_ids.len());
        for (batch_idx, targets) in batch_targets {
            reader.set_index(batch_idx)?;
            let batch = reader.next().ok_or_else(|| Error::BatchIndexOutOfBounds {
                index: batch_idx,
                max: reader.num_batches(),
            })??;
            let signal_col =
                batch
                    .column(1)
                    .as_list_opt::<i32>()
                    .ok_or_else(|| Error::InvalidField {
                        field: "signal".to_string(),
                        message: "Expected ListArray".to_string(),
                    })?;
            let cal_offset_col = batch
                .column(2)
                .as_primitive_opt::<Float32Type>()
                .ok_or_else(|| Error::InvalidField {
                    field: "calibration_offset".to_string(),
                    message: "Expected Float32Array".to_string(),
                })?;
            let cal_scale_col = batch
                .column(3)
                .as_primitive_opt::<Float32Type>()
                .ok_or_else(|| Error::InvalidField {
                    field: "calibration_scale".to_string(),
                    message: "Expected Float32Array".to_string(),
                })?;
            // read_id is already in the projection; without this the returned
            // signal and calibration would carry the *queried* UUID whatever
            // row they came from.
            let read_ids = crate::arrow_helpers::read_id_column(&batch)?;
            for (uuid, row) in targets {
                crate::arrow_helpers::verify_index_row(read_ids, uuid, batch_idx, row)?;
                let values = signal_col.value(row);
                let u64_arr =
                    values
                        .as_primitive_opt::<UInt64Type>()
                        .ok_or_else(|| Error::InvalidField {
                            field: "signal".to_string(),
                            message: "Expected UInt64Array values".to_string(),
                        })?;
                results.push(SignalCalibration {
                    read_id: uuid,
                    signal_rows: u64_arr.values().to_vec(),
                    calibration_offset: cal_offset_col.value(row),
                    calibration_scale: cal_scale_col.value(row),
                });
            }
        }
        Ok(results)
    }

    // ---- Single-pass path (no index — column-projected scan with inline filter) ----

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
        use arrow::array::{Array, AsArray};

        let Some(col) = batch.column_by_name(name) else {
            return HashMap::new();
        };

        let Some(map_array) = col.as_map_opt() else {
            return HashMap::new();
        };

        let mut result = HashMap::new();

        // Get the entries for this row as a StructArray. `MapArray::value`
        // already yields a concrete `StructArray`, so no downcast is needed.
        let struct_array = map_array.value(row);

        if struct_array.num_columns() >= 2
            && let (Some(keys), Some(values)) = (
                struct_array.column(0).as_string_opt::<i32>(),
                struct_array.column(1).as_string_opt::<i32>(),
            )
        {
            for i in 0..struct_array.len() {
                if !keys.is_null(i) && !values.is_null(i) {
                    result.insert(keys.value(i).to_string(), values.value(i).to_string());
                }
            }
        }

        result
    }
}

#[cfg(test)]
mod uniformity_tests {
    use super::*;

    #[test]
    fn uniform_and_short_files_are_clean() {
        // The last batch is legitimately short.
        assert_eq!(first_nonuniform_batch(&[1000, 1000, 1000, 373]), None);
        assert_eq!(first_nonuniform_batch(&[1000, 1000, 1000, 1000]), None);
        // Too few batches for a stride to be contradicted.
        assert_eq!(first_nonuniform_batch(&[]), None);
        assert_eq!(first_nonuniform_batch(&[7]), None);
        assert_eq!(first_nonuniform_batch(&[1000, 42]), None);
    }

    #[test]
    fn oversized_batch_is_reported_with_its_position() {
        // The shape of the real production failure: 57x1000, one 1003, short
        // last (barcode_nbc15.pod5, escapepod-rs#195).
        let mut counts = vec![1000u64; 58];
        counts[1] = 1003;
        counts.push(410);
        let bad = first_nonuniform_batch(&counts).expect("must be flagged");
        assert_eq!(bad.index, 1);
        assert_eq!(bad.rows, 1003);
        assert_eq!(bad.expected, 1000);
    }

    #[test]
    fn an_undersized_middle_batch_counts_too() {
        // Short is just as index-shifting as long; only the FINAL batch is
        // allowed to be short.
        let bad = first_nonuniform_batch(&[100, 100, 99, 100, 50]).expect("must be flagged");
        assert_eq!((bad.index, bad.rows, bad.expected), (2, 99, 100));
    }

    #[test]
    fn the_first_offender_is_the_one_reported() {
        let bad = first_nonuniform_batch(&[10, 12, 10, 15, 3]).expect("must be flagged");
        assert_eq!(
            bad.index, 1,
            "report the earliest divergence, not the worst"
        );
    }
}
