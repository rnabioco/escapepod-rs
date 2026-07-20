//! High-performance filter operation for POD5 files.
//!
//! Uses raw byte extraction from mmap without Arrow deserialization.

use crate::arrow_helpers::ReadsBatchView;
use crate::arrow_ipc::ArrowIpcFooter;
use crate::error::{Error, Result};
use crate::reader::Reader;
use crate::types::{EndReason, POD5_SIGNATURE, ReadData, RunInfoData, Uuid};
use crate::utils::parse_uuid_flexible;
use crate::utils::pod5_assembler::{
    FlatReadRef, deduplicate_run_infos, write_post_signal_sections,
};
use crate::utils::table_builders::{SchemaMetadata, build_reads_table_remapped};
use crate::writer::atomic::{AtomicFile, Durability};
use arrow::record_batch::RecordBatch;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::sync::Arc;

/// Criteria for filtering reads.
#[derive(Debug, Clone, Default)]
pub struct FilterCriteria {
    /// Only include reads with these IDs.
    pub read_ids: Option<HashSet<Uuid>>,
    /// Minimum number of samples (inclusive).
    pub min_samples: Option<u64>,
    /// Maximum number of samples (inclusive).
    pub max_samples: Option<u64>,
    /// Only include reads with these end reasons.
    pub include_end_reasons: Option<HashSet<EndReason>>,
    /// Exclude reads with these end reasons.
    pub exclude_end_reasons: Option<HashSet<EndReason>>,
}

impl FilterCriteria {
    /// Check if a read matches all the filter criteria.
    pub fn matches(&self, read: &ReadData) -> bool {
        // Check read ID filter
        if let Some(ref ids) = self.read_ids
            && !ids.contains(&read.read_id)
        {
            return false;
        }

        // Check min samples
        if let Some(min) = self.min_samples
            && read.num_samples < min
        {
            return false;
        }

        // Check max samples
        if let Some(max) = self.max_samples
            && read.num_samples > max
        {
            return false;
        }

        // Check include end reasons (if specified, read must have one of these)
        if let Some(ref include) = self.include_end_reasons
            && !include.contains(&read.end_reason)
        {
            return false;
        }

        // Check exclude end reasons
        if let Some(ref exclude) = self.exclude_end_reasons
            && exclude.contains(&read.end_reason)
        {
            return false;
        }

        true
    }

    /// Returns true if the only active criterion is a read-ID set
    /// (no sample-count or end-reason filters). When true, the
    /// accelerated `reads_by_ids()` path can be used.
    pub fn is_uuid_only(&self) -> bool {
        self.read_ids.is_some()
            && self.min_samples.is_none()
            && self.max_samples.is_none()
            && self.include_end_reasons.is_none()
            && self.exclude_end_reasons.is_none()
    }

    /// Returns true if no criteria are set (matches all reads).
    pub fn is_empty(&self) -> bool {
        self.read_ids.is_none()
            && self.min_samples.is_none()
            && self.max_samples.is_none()
            && self.include_end_reasons.is_none()
            && self.exclude_end_reasons.is_none()
    }
}

/// Options for the filter operation.
#[derive(Debug, Clone)]
pub struct FilterOptions {
    /// Signal chunks per batch.
    pub signal_batch_size: u32,
    /// Reads per batch.
    pub read_batch_size: u32,
    /// How hard to push bytes to stable storage before renaming into place.
    pub durability: Durability,
}

impl Default for FilterOptions {
    fn default() -> Self {
        Self {
            signal_batch_size: 1_000,
            read_batch_size: 10_000,
            durability: Durability::default(),
        }
    }
}

/// Result of a filter operation.
#[derive(Debug, Clone, Default)]
pub struct FilterResult {
    /// Total number of reads processed.
    pub total_reads: u64,
    /// Number of reads that matched the filter.
    pub matched_reads: u64,
    /// Number of read errors encountered.
    pub read_errors: u64,
    /// Number of signal errors encountered.
    pub signal_errors: u64,
}

impl FilterResult {
    /// Returns the percentage of reads that matched.
    pub fn match_percentage(&self) -> f64 {
        if self.total_reads > 0 {
            (self.matched_reads as f64 / self.total_reads as f64) * 100.0
        } else {
            0.0
        }
    }
}

use crate::progress::{Progress, ProgressCallback};

/// Collected metadata from a single file for filtering.
struct FileMetadata {
    reader: Reader,
    signal_footer: ArrowIpcFooter,
    run_infos: Vec<RunInfoData>,
    /// Only reads that match the filter criteria.
    matching_reads: Vec<ReadData>,
    /// Total number of reads in the file (for statistics).
    total_read_count: usize,
}

/// Filter reads from POD5 files based on a set of read IDs.
///
/// This function uses raw byte extraction from mmap - no Arrow deserialization.
/// It extracts only the specific signal rows needed for matching reads.
///
/// This is a convenience wrapper around `filter_files_with_criteria` for
/// backwards compatibility.
pub fn filter_files<P: AsRef<Path> + Sync>(
    input_files: &[P],
    output_path: impl AsRef<Path>,
    filter_ids: &HashSet<Uuid>,
    options: FilterOptions,
    progress: Option<ProgressCallback>,
) -> Result<FilterResult> {
    let criteria = FilterCriteria {
        read_ids: Some(filter_ids.clone()),
        ..Default::default()
    };
    filter_files_with_criteria(input_files, output_path, &criteria, options, progress)
}

/// Filter reads from POD5 files based on filter criteria.
///
/// This function uses raw byte extraction from mmap - no Arrow deserialization.
/// It extracts only the specific signal rows needed for matching reads.
pub fn filter_files_with_criteria<P: AsRef<Path> + Sync>(
    input_files: &[P],
    output_path: impl AsRef<Path>,
    criteria: &FilterCriteria,
    options: FilterOptions,
    progress: Option<ProgressCallback>,
) -> Result<FilterResult> {
    if input_files.is_empty() {
        return Err(Error::InvalidState("No input files specified".into()));
    }

    if criteria.is_empty() {
        return Err(Error::InvalidState("No filter criteria specified".into()));
    }

    // Phase 1: Open files and identify matching reads in parallel
    let input_paths: Vec<&Path> = input_files.iter().map(|p| p.as_ref()).collect();

    let metadata_results: Vec<Result<FileMetadata>> = input_paths
        .par_iter()
        .map(|path| {
            let reader = Reader::open(path)?;
            let signal_bytes = reader.signal_table_bytes()?;
            let signal_footer = ArrowIpcFooter::parse(signal_bytes)?;
            let run_infos = reader.run_infos().to_vec();

            // Fast path: when only filtering by UUID, use reads_by_ids()
            // to skip non-matching batches entirely.
            let (matching_reads, total_read_count) = if criteria.is_uuid_only() {
                let total = reader.read_count()?;
                let target_ids = criteria.read_ids.as_ref().unwrap();
                let matching = reader.reads_by_ids(target_ids)?;
                (matching, total)
            } else {
                // Scan the reads table in parallel across its record batches.
                // Matching is independent per batch, and concatenating the
                // per-batch results in batch order preserves file read order —
                // which the downstream signal-row prefix sums (`flat_reads`)
                // rely on. Nested under the outer per-file `par_iter`; the
                // global rayon pool bounds the combined width (CLI `-t`).
                let batches: Vec<RecordBatch> =
                    reader.read_batches()?.collect::<Result<Vec<_>>>()?;
                let total: usize = batches.iter().map(|b| b.num_rows()).sum();

                let per_batch: Vec<Vec<ReadData>> = batches
                    .par_iter()
                    .map(|batch| -> Result<Vec<ReadData>> {
                        let view = ReadsBatchView::new(batch, false)?;
                        let mut matched = Vec::new();
                        for row in 0..view.num_rows() {
                            let read = view.read(row)?;
                            if criteria.matches(&read) {
                                matched.push(read);
                            }
                        }
                        Ok(matched)
                    })
                    .collect::<Result<Vec<_>>>()?;

                let matching: Vec<ReadData> = per_batch.into_iter().flatten().collect();
                (matching, total)
            };

            Ok(FileMetadata {
                reader,
                signal_footer,
                run_infos,
                matching_reads,
                total_read_count,
            })
        })
        .collect();

    let file_metadata: Vec<FileMetadata> =
        metadata_results.into_iter().collect::<Result<Vec<_>>>()?;

    // Count total reads for statistics
    let total_read_count: u64 = file_metadata
        .iter()
        .map(|m| m.total_read_count as u64)
        .sum();
    // Phases 2 & 3: extract signal and assemble the single output file.
    // Borrow each file's owned reader/footer/reads so the shared assembler
    // can hold the mmap-backed chunks without cloning.
    let sources: Vec<OutputSource<'_>> = file_metadata
        .iter()
        .map(|m| OutputSource {
            reader: &m.reader,
            signal_footer: &m.signal_footer,
            run_infos: &m.run_infos,
            matching_reads: &m.matching_reads,
        })
        .collect();

    let matched = assemble_output(output_path.as_ref(), &sources, &options, progress.as_ref())?;

    Ok(FilterResult {
        total_reads: total_read_count,
        matched_reads: matched,
        read_errors: 0,
        signal_errors: 0,
    })
}

/// One source file's borrowed contribution to an output POD5 file: the reader
/// (which owns the mmap the extracted chunks borrow from), its parsed signal
/// footer, its run infos, and the reads selected for this output (in the order
/// they should appear).
struct OutputSource<'a> {
    reader: &'a Reader,
    signal_footer: &'a ArrowIpcFooter,
    run_infos: &'a [RunInfoData],
    matching_reads: &'a [ReadData],
}

/// Extract the selected signal and assemble one POD5 output file from a set of
/// borrowed source files, returning the number of reads written.
///
/// Shared by the multi-file `filter` path (many sources → one output) and the
/// single-pass `subset` path (one shared reader → one source per output). The
/// per-source extraction runs in parallel; chunks are concatenated in source
/// order so the reads-table's per-read signal-row prefix sums line up with the
/// signal table.
fn assemble_output(
    output_path: &Path,
    sources: &[OutputSource<'_>],
    options: &FilterOptions,
    progress: Option<&ProgressCallback>,
) -> Result<u64> {
    let num_sources = sources.len();
    let matching_count: u64 = sources.iter().map(|s| s.matching_reads.len() as u64).sum();

    // Phase 2: Extract signal data in parallel across sources.
    //
    // Each source's mmap is independent so extraction parallelizes per source.
    // Each chunk is held as a `&[u8]` into the source mmap (kept alive by the
    // caller's readers) — never copied to the heap. On a 26 GB POD5 this avoids
    // ~26 GB of `Arc<[u8]>` heap allocation that would otherwise OOM modest
    // SLURM allocations.
    type SignalChunks<'a> = Vec<(&'a [u8], u32)>;

    let extractions: Vec<Result<SignalChunks<'_>>> = sources
        .par_iter()
        .map(|source| {
            if source.matching_reads.is_empty() {
                return Ok(Vec::new());
            }

            let signal_bytes = source.reader.signal_table_bytes()?;

            // Collect all signal rows needed from this source
            let signal_row_indices: Vec<u64> = source
                .matching_reads
                .iter()
                .flat_map(|read| read.signal_rows.iter().copied())
                .collect();

            let raw_chunks = source
                .signal_footer
                .extract_signal_rows(&signal_row_indices, signal_bytes)?;

            let chunks: SignalChunks<'_> = raw_chunks
                .into_iter()
                .map(|chunk| (chunk.signal, chunk.samples))
                .collect();

            Ok(chunks)
        })
        .collect();

    let source_extractions: Vec<SignalChunks<'_>> =
        extractions.into_iter().collect::<Result<Vec<_>>>()?;

    // Flatten source extractions into a single chunk list in source order.
    // Order matters: the reads-table's per-read signal-row prefix sums (built
    // below into `flat_reads`) walk sources in this same order, so chunk N of
    // the signal table must correspond to signal-row N of the output.
    let total_chunks: usize = source_extractions.iter().map(|e| e.len()).sum();
    let mut signal_chunks: Vec<(&[u8], u32)> = Vec::with_capacity(total_chunks);
    for (source_idx, chunks) in source_extractions.iter().enumerate() {
        signal_chunks.extend_from_slice(chunks);
        if let Some(cb) = progress {
            cb(Progress {
                current: source_idx as u64 + 1,
                total: num_sources as u64,
            });
        }
    }

    // Phase 3: Build output file

    let schema_meta = SchemaMetadata::new();

    // Stage into a temp file alongside the destination; nothing appears at
    // `output_path` until the commit below. Any `?` between here and there
    // drops the guard, which unlinks the partial file and leaves an existing
    // destination untouched.
    let atomic = AtomicFile::with_durability(output_path, options.durability)?;
    // 128 MiB buffer matches merge.rs and avoids many small syscalls when
    // the reads/run_info tables are flushed at the end.
    let mut file = BufWriter::with_capacity(128 * 1024 * 1024, atomic.reopen()?);

    // Write POD5 header
    file.write_all(&POD5_SIGNATURE)?;
    let section_marker = Uuid::new_v4();
    file.write_all(section_marker.as_bytes())?;

    // Stream the signal table directly to the file. Previously this built
    // the full IPC bytes into a `Vec<u8>` and then wrote it — for a 26 GB
    // input that meant a ~26 GB heap allocation. Streaming keeps peak
    // memory bounded by `signal_batch_size` chunks in flight.
    let signal_table_bytes_written = write_raw_signal_table(
        &mut file,
        &signal_chunks,
        options.signal_batch_size,
        &schema_meta,
    )?;

    // Track signal_end manually to avoid `stream_position()`, which would
    // force a flush of the 128 MiB BufWriter.
    let signal_end = POD5_SIGNATURE.len() + 16 + signal_table_bytes_written;

    // Borrow each source's run_infos for dedup — avoids deep-cloning every
    // RunInfoData (each carries 13 Strings + 2 HashMaps).
    let per_source_run_infos: Vec<&[RunInfoData]> = sources.iter().map(|s| s.run_infos).collect();
    let (all_run_infos, run_info_map) = deduplicate_run_infos(&per_source_run_infos);

    // Build a flat view of every matching read with the prefix-sum
    // signal-row offset and remapped run_info dict key baked in. Each
    // `FlatReadRef` is borrowed pointers + two scalars (~32 B/entry) —
    // for a 30M-match workload this is ~960 MB of refs, vs ~6 GB if we
    // materialized cloned `ProcessedRead`s here. The remap is applied
    // here once per read; `build_reads_table_remapped` then walks the
    // flat list during its parallel partition build with no further
    // remap work per row.
    let mut flat_reads: Vec<FlatReadRef<'_>> = Vec::with_capacity(matching_count as usize);
    let mut signal_row_cursor: u64 = 0;
    for source in sources {
        for read in source.matching_reads {
            let run_info_key = source
                .run_infos
                .get(read.run_info_index as usize)
                .and_then(|ri| run_info_map.get(&ri.acquisition_id).copied())
                .unwrap_or(0);
            flat_reads.push(FlatReadRef {
                read,
                new_signal_rows_start: signal_row_cursor,
                run_info_key: i16::try_from(run_info_key).unwrap_or(0),
            });
            signal_row_cursor += read.signal_rows.len() as u64;
        }
    }

    let reads_table_bytes = build_reads_table_remapped(&flat_reads, &all_run_infos, &schema_meta)?;

    // Free the borrow refs before the (large) reads_table_bytes buffer
    // hits the writer, so peak RSS is dominated by the IPC-encoded form
    // (~50–100 B/read) rather than carrying both shapes in memory.
    drop(flat_reads);

    write_post_signal_sections(
        &mut file,
        &section_marker,
        &schema_meta,
        signal_end,
        &all_run_infos,
        &reads_table_bytes,
    )?;

    // Release the buffered handle before the rename so nothing is left unwritten.
    file.flush()?;
    drop(file);
    atomic.commit()?;

    Ok(matching_count)
}

/// Scan one reader's reads table a single time, partitioning the reads present
/// in `read_to_group` into `group name -> reads` (in input read order).
///
/// Batches are scanned in parallel; per-batch partitions are merged in batch
/// order so each group keeps input read order (its signal-row prefix sums
/// depend on it). Returned group keys borrow the names from `read_to_group`.
fn partition_reads_by_group<'a>(
    reader: &Reader,
    read_to_group: &'a HashMap<Uuid, String>,
) -> Result<HashMap<&'a str, Vec<ReadData>>> {
    let batches: Vec<RecordBatch> = reader.read_batches()?.collect::<Result<Vec<_>>>()?;

    let per_batch: Vec<HashMap<&str, Vec<ReadData>>> = batches
        .par_iter()
        .map(|batch| -> Result<HashMap<&str, Vec<ReadData>>> {
            let view = ReadsBatchView::new(batch, false)?;
            let mut local: HashMap<&str, Vec<ReadData>> = HashMap::new();
            for row in 0..view.num_rows() {
                let read = view.read(row)?;
                if let Some(group) = read_to_group.get(&read.read_id) {
                    local.entry(group.as_str()).or_default().push(read);
                }
            }
            Ok(local)
        })
        .collect::<Result<Vec<_>>>()?;

    let mut groups: HashMap<&str, Vec<ReadData>> = HashMap::new();
    for local in per_batch {
        for (group, reads) in local {
            groups.entry(group).or_default().extend(reads);
        }
    }
    Ok(groups)
}

/// Outcome of a subset/split, reported per group.
///
/// Groups are written in parallel and independently, so one failing group does
/// not invalidate the others. Both lists are sorted by group name.
#[derive(Debug, Default)]
pub struct SubsetOutcome {
    /// Groups written successfully, as (output name, reads written).
    pub groups: Vec<(String, u64)>,
    /// Groups that failed. These produced no output file at all.
    pub failures: Vec<(String, Error)>,
}

impl SubsetOutcome {
    /// Total reads written across all successful groups.
    pub fn reads_written(&self) -> u64 {
        self.groups.iter().map(|(_, n)| n).sum()
    }
}

/// Split the reads of a single POD5 file into multiple outputs in one pass.
///
/// Convenience wrapper around [`subset_files`] for a single input. See there
/// for semantics.
pub fn subset_file<P: AsRef<Path> + Sync>(
    input: P,
    read_to_group: &HashMap<Uuid, String>,
    output_dir: &Path,
    options: FilterOptions,
) -> Result<SubsetOutcome> {
    subset_files(
        std::slice::from_ref(&input),
        read_to_group,
        output_dir,
        options,
    )
}

/// Split the reads of one or more POD5 files into multiple outputs in one pass.
///
/// Unlike calling [`filter_files`] once per output group — which re-opens and
/// re-scans every input for every group — this opens each input once, scans its
/// reads table a single time to partition matching reads by their target group,
/// then writes every group's file in parallel. A group whose reads span several
/// inputs is assembled from all of them (in input order); reads absent from
/// `read_to_group` are dropped.
///
/// `read_to_group` maps each read UUID to the basename of its output file
/// (joined onto `output_dir`).
pub fn subset_files<P: AsRef<Path> + Sync>(
    inputs: &[P],
    read_to_group: &HashMap<Uuid, String>,
    output_dir: &Path,
    options: FilterOptions,
) -> Result<SubsetOutcome> {
    if inputs.is_empty() {
        return Ok(SubsetOutcome::default());
    }

    // Phase 1: open each input and partition its reads by group, in parallel
    // across inputs. Each reader stays alive so its mmap backs the chunks the
    // assembler borrows in phase 2.
    type InputCtx<'a> = (
        Reader,
        ArrowIpcFooter,
        Vec<RunInfoData>,
        HashMap<&'a str, Vec<ReadData>>,
    );
    let input_paths: Vec<&Path> = inputs.iter().map(|p| p.as_ref()).collect();
    let per_input: Vec<InputCtx<'_>> = input_paths
        .par_iter()
        .map(|path| -> Result<InputCtx<'_>> {
            let reader = Reader::open(path)?;
            let signal_bytes = reader.signal_table_bytes()?;
            let signal_footer = ArrowIpcFooter::parse(signal_bytes)?;
            let run_infos = reader.run_infos().to_vec();
            let groups = partition_reads_by_group(&reader, read_to_group)?;
            Ok((reader, signal_footer, run_infos, groups))
        })
        .collect::<Result<Vec<_>>>()?;

    // Union of every group name that received at least one read.
    let mut all_groups: HashSet<&str> = HashSet::new();
    for (_, _, _, groups) in &per_input {
        all_groups.extend(groups.keys().copied());
    }

    // Phase 2: write each group's output in parallel. Each output gathers one
    // `OutputSource` per input that contributed reads (in input order), so a
    // group spanning multiple inputs is assembled from all their mmaps at once.
    // Nested rayon (across groups, then across sources inside `assemble_output`)
    // is bounded by the global pool (CLI `-t`).
    let group_list: Vec<&str> = all_groups.into_iter().collect();
    type GroupResult = std::result::Result<(String, u64), (String, Error)>;
    let results: Vec<GroupResult> = group_list
        .par_iter()
        .map(|&group| {
            let sources: Vec<OutputSource<'_>> = per_input
                .iter()
                .filter_map(|(reader, signal_footer, run_infos, groups)| {
                    groups.get(group).map(|reads| OutputSource {
                        reader,
                        signal_footer,
                        run_infos: run_infos.as_slice(),
                        matching_reads: reads.as_slice(),
                    })
                })
                .collect();
            let output_path = output_dir.join(group);
            match assemble_output(&output_path, &sources, &options, None) {
                Ok(matched) => Ok((group.to_string(), matched)),
                Err(e) => Err((group.to_string(), e)),
            }
        })
        .collect();

    // Report every group's outcome rather than collapsing to the first error
    // and discarding the rest: groups are written in parallel, so "the first
    // error" is nondeterministic and says nothing about what else happened.
    // Failed groups leave no file behind, so a re-run is clean.
    let mut outcome = SubsetOutcome::default();
    for result in results {
        match result {
            Ok(group) => outcome.groups.push(group),
            Err(failure) => outcome.failures.push(failure),
        }
    }
    outcome.groups.sort_by(|a, b| a.0.cmp(&b.0));
    outcome.failures.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(outcome)
}

/// `Write` adapter that counts the bytes flowing through. The Arrow IPC
/// writer needs a `Write` sink, but we also want to know how many bytes
/// it emitted so we can compute `signal_end` without `stream_position()`.
struct CountingWriter<'a, W: Write> {
    inner: &'a mut W,
    count: usize,
}

impl<'a, W: Write> CountingWriter<'a, W> {
    fn new(inner: &'a mut W) -> Self {
        Self { inner, count: 0 }
    }
}

impl<W: Write> Write for CountingWriter<'_, W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let n = self.inner.write(buf)?;
        self.count += n;
        Ok(n)
    }

    fn write_all(&mut self, buf: &[u8]) -> std::io::Result<()> {
        self.inner.write_all(buf)?;
        self.count += buf.len();
        Ok(())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

/// Stream the signal IPC table directly to `file`, returning the number of
/// bytes written. Peak memory is bounded by one in-flight `RecordBatch`
/// (≤ `batch_size` chunks), independent of the total signal volume.
fn write_raw_signal_table<W: Write>(
    file: &mut W,
    chunks: &[(&[u8], u32)],
    batch_size: u32,
    meta: &SchemaMetadata,
) -> Result<usize> {
    use crate::schema::signal_schema;
    use arrow::array::{ArrayRef, FixedSizeBinaryBuilder, LargeBinaryBuilder, UInt32Builder};
    use arrow::ipc::writer::FileWriter;
    use arrow::record_batch::RecordBatch;

    let schema = Arc::new(meta.apply(signal_schema()));

    let mut counter = CountingWriter::new(file);
    {
        let mut writer = FileWriter::try_new(&mut counter, &schema)?;

        let total_rows = chunks.len();
        let mut offset = 0;
        while offset < total_rows {
            let end = std::cmp::min(offset + batch_size as usize, total_rows);
            let batch_chunks = &chunks[offset..end];

            let mut read_id_builder = FixedSizeBinaryBuilder::with_capacity(batch_chunks.len(), 16);
            let mut signal_builder = LargeBinaryBuilder::new();
            let mut samples_builder = UInt32Builder::with_capacity(batch_chunks.len());

            for (signal_data, samples) in batch_chunks {
                // The actual read_id lives in the reads table; the signal
                // table's read_id column is unused by the POD5 reader.
                read_id_builder.append_value([0u8; 16])?;
                signal_builder.append_value(*signal_data);
                samples_builder.append_value(*samples);
            }

            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(read_id_builder.finish()) as ArrayRef,
                    Arc::new(signal_builder.finish()) as ArrayRef,
                    Arc::new(samples_builder.finish()) as ArrayRef,
                ],
            )?;

            writer.write(&batch)?;
            offset = end;
        }

        writer.finish()?;
    }

    Ok(counter.count)
}

/// Read read IDs from a text file or stdin (one per line).
///
/// If `path` is "-" or "stdin" (case-insensitive), reads from stdin.
/// Otherwise reads from the specified file.
///
/// Supports UUIDs in various formats:
/// - Standard: `a1b2c3d4-e5f6-7890-abcd-ef1234567890`
/// - No dashes: `a1b2c3d4e5f67890abcdef1234567890`
///
/// Lines starting with `#` are treated as comments and skipped.
/// Empty lines are also skipped.
pub fn read_ids_from_file(path: impl AsRef<Path>) -> Result<HashSet<Uuid>> {
    let path_str = path.as_ref().to_string_lossy();

    if path_str == "-" || path_str.eq_ignore_ascii_case("stdin") {
        read_ids_from_reader(std::io::stdin().lock())
    } else {
        let file = File::open(path.as_ref())?;
        read_ids_from_reader(BufReader::new(file))
    }
}

/// Read read IDs from any `BufRead` source.
fn read_ids_from_reader<R: BufRead>(reader: R) -> Result<HashSet<Uuid>> {
    let mut ids = HashSet::new();

    for (line_num, line_result) in reader.lines().enumerate() {
        let line = line_result?;
        let line = line.trim();

        // Skip empty lines and comments
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        match parse_uuid_flexible(line) {
            Ok(uuid) => {
                ids.insert(uuid);
            }
            Err(_) => {
                return Err(Error::Parse(format!(
                    "Invalid UUID on line {}: '{}'",
                    line_num + 1,
                    line
                )));
            }
        }
    }

    Ok(ids)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ReadData;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn make_test_read(num_samples: u64, end_reason: EndReason) -> ReadData {
        ReadData {
            read_id: Uuid::new_v4(),
            num_samples,
            end_reason,
            ..Default::default()
        }
    }

    #[test]
    fn test_filter_criteria_is_empty() {
        let criteria = FilterCriteria::default();
        assert!(criteria.is_empty());

        let criteria = FilterCriteria {
            min_samples: Some(100),
            ..Default::default()
        };
        assert!(!criteria.is_empty());
    }

    #[test]
    fn test_filter_criteria_is_uuid_only() {
        // UUID-only: should be true
        let criteria = FilterCriteria {
            read_ids: Some(HashSet::new()),
            ..Default::default()
        };
        assert!(criteria.is_uuid_only());

        // No criteria at all: not UUID-only
        assert!(!FilterCriteria::default().is_uuid_only());

        // UUID + min_samples: not UUID-only
        let criteria = FilterCriteria {
            read_ids: Some(HashSet::new()),
            min_samples: Some(100),
            ..Default::default()
        };
        assert!(!criteria.is_uuid_only());

        // Only min_samples, no UUIDs: not UUID-only
        let criteria = FilterCriteria {
            min_samples: Some(100),
            ..Default::default()
        };
        assert!(!criteria.is_uuid_only());
    }

    #[test]
    fn test_filter_criteria_matches_read_ids() {
        let read = make_test_read(1000, EndReason::SignalPositive);
        let mut ids = HashSet::new();
        ids.insert(read.read_id);

        let criteria = FilterCriteria {
            read_ids: Some(ids),
            ..Default::default()
        };
        assert!(criteria.matches(&read));

        // Different read ID should not match
        let other_read = make_test_read(1000, EndReason::SignalPositive);
        assert!(!criteria.matches(&other_read));
    }

    #[test]
    fn test_filter_criteria_matches_min_samples() {
        let read = make_test_read(5000, EndReason::SignalPositive);

        // Read with 5000 samples should match min_samples=4000
        let criteria = FilterCriteria {
            min_samples: Some(4000),
            ..Default::default()
        };
        assert!(criteria.matches(&read));

        // Read with 5000 samples should match min_samples=5000 (inclusive)
        let criteria = FilterCriteria {
            min_samples: Some(5000),
            ..Default::default()
        };
        assert!(criteria.matches(&read));

        // Read with 5000 samples should not match min_samples=6000
        let criteria = FilterCriteria {
            min_samples: Some(6000),
            ..Default::default()
        };
        assert!(!criteria.matches(&read));
    }

    #[test]
    fn test_filter_criteria_matches_max_samples() {
        let read = make_test_read(5000, EndReason::SignalPositive);

        // Read with 5000 samples should match max_samples=6000
        let criteria = FilterCriteria {
            max_samples: Some(6000),
            ..Default::default()
        };
        assert!(criteria.matches(&read));

        // Read with 5000 samples should match max_samples=5000 (inclusive)
        let criteria = FilterCriteria {
            max_samples: Some(5000),
            ..Default::default()
        };
        assert!(criteria.matches(&read));

        // Read with 5000 samples should not match max_samples=4000
        let criteria = FilterCriteria {
            max_samples: Some(4000),
            ..Default::default()
        };
        assert!(!criteria.matches(&read));
    }

    #[test]
    fn test_filter_criteria_matches_include_end_reasons() {
        let read = make_test_read(1000, EndReason::SignalPositive);

        // Should match when end_reason is in the include set
        let mut include = HashSet::new();
        include.insert(EndReason::SignalPositive);
        include.insert(EndReason::SignalNegative);
        let criteria = FilterCriteria {
            include_end_reasons: Some(include),
            ..Default::default()
        };
        assert!(criteria.matches(&read));

        // Should not match when end_reason is not in the include set
        let mut include = HashSet::new();
        include.insert(EndReason::MuxChange);
        let criteria = FilterCriteria {
            include_end_reasons: Some(include),
            ..Default::default()
        };
        assert!(!criteria.matches(&read));
    }

    #[test]
    fn test_filter_criteria_matches_exclude_end_reasons() {
        let read = make_test_read(1000, EndReason::SignalPositive);

        // Should match when end_reason is not in the exclude set
        let mut exclude = HashSet::new();
        exclude.insert(EndReason::MuxChange);
        exclude.insert(EndReason::UnblockMuxChange);
        let criteria = FilterCriteria {
            exclude_end_reasons: Some(exclude),
            ..Default::default()
        };
        assert!(criteria.matches(&read));

        // Should not match when end_reason is in the exclude set
        let mut exclude = HashSet::new();
        exclude.insert(EndReason::SignalPositive);
        let criteria = FilterCriteria {
            exclude_end_reasons: Some(exclude),
            ..Default::default()
        };
        assert!(!criteria.matches(&read));
    }

    #[test]
    fn test_filter_criteria_matches_combined() {
        let read = make_test_read(5000, EndReason::SignalPositive);

        // Should match when all criteria are satisfied
        let criteria = FilterCriteria {
            min_samples: Some(4000),
            max_samples: Some(6000),
            include_end_reasons: Some([EndReason::SignalPositive].into_iter().collect()),
            ..Default::default()
        };
        assert!(criteria.matches(&read));

        // Should not match when one criterion fails (samples too low)
        let criteria = FilterCriteria {
            min_samples: Some(6000),
            max_samples: Some(10000),
            include_end_reasons: Some([EndReason::SignalPositive].into_iter().collect()),
            ..Default::default()
        };
        assert!(!criteria.matches(&read));

        // Should not match when one criterion fails (wrong end reason)
        let criteria = FilterCriteria {
            min_samples: Some(4000),
            max_samples: Some(6000),
            include_end_reasons: Some([EndReason::MuxChange].into_iter().collect()),
            ..Default::default()
        };
        assert!(!criteria.matches(&read));
    }

    #[test]
    fn test_filter_criteria_matches_sample_range() {
        // Test combined min and max samples
        let criteria = FilterCriteria {
            min_samples: Some(4000),
            max_samples: Some(6000),
            ..Default::default()
        };

        assert!(!criteria.matches(&make_test_read(3999, EndReason::SignalPositive)));
        assert!(criteria.matches(&make_test_read(4000, EndReason::SignalPositive)));
        assert!(criteria.matches(&make_test_read(5000, EndReason::SignalPositive)));
        assert!(criteria.matches(&make_test_read(6000, EndReason::SignalPositive)));
        assert!(!criteria.matches(&make_test_read(6001, EndReason::SignalPositive)));
    }

    #[test]
    fn test_read_ids_from_file() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "a1b2c3d4-e5f6-7890-abcd-ef1234567890").unwrap();
        writeln!(temp_file, "# This is a comment").unwrap();
        writeln!(temp_file).unwrap(); // Empty line
        writeln!(temp_file, "b2c3d4e5f6a78901bcdef12345678901").unwrap(); // No dashes
        temp_file.flush().unwrap();

        let ids = read_ids_from_file(temp_file.path()).unwrap();
        assert_eq!(ids.len(), 2);
    }

    #[test]
    fn test_read_ids_invalid_uuid() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "not-a-uuid").unwrap();
        temp_file.flush().unwrap();

        let result = read_ids_from_file(temp_file.path());
        assert!(result.is_err());
    }

    #[test]
    fn test_filter_result_percentage() {
        let result = FilterResult {
            total_reads: 100,
            matched_reads: 25,
            read_errors: 0,
            signal_errors: 0,
        };
        assert!((result.match_percentage() - 25.0).abs() < 0.001);
    }

    #[test]
    fn test_filter_result_empty() {
        let result = FilterResult::default();
        assert_eq!(result.match_percentage(), 0.0);
    }
}
