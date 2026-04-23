//! Fingerprint subcommand - extract signal features from adapter regions.

use super::utils::{configure_thread_pool, parse_boundaries_csv, parse_norm_method};
use crate::progress::create_progress_bar;
use crate::style;
use arrow::array::{ArrayRef, Float64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use escapepod_demux::{ReadFingerprint, extract_fingerprint_from_signal};
use escapepod_signal::Reader;
use escapepod_signal::dtw::NormMethod;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Arguments for the fingerprint subcommand.
#[derive(Debug, clap::Args)]
pub struct FingerprintArgs {
    /// Input POD5 file(s)
    #[arg(required = true, value_name = "FILES")]
    pub input: Vec<PathBuf>,

    /// Detected boundaries CSV (from detect command)
    #[arg(long, required = true, value_name = "FILE")]
    pub boundaries: PathBuf,

    /// Output fingerprints file
    #[arg(short, long, required = true, value_name = "FILE")]
    pub output: PathBuf,

    /// Start sample offset within adapter region for fingerprinting
    #[arg(
        long,
        default_value = "1000",
        value_name = "N",
        help_heading = "Advanced Options"
    )]
    pub segment_start: usize,

    /// End sample offset within adapter region for fingerprinting
    #[arg(
        long,
        default_value = "2000",
        value_name = "N",
        help_heading = "Advanced Options"
    )]
    pub segment_end: usize,

    /// Number of segments for fingerprint
    #[arg(
        long,
        default_value = "10",
        value_name = "N",
        help_heading = "Advanced Options"
    )]
    pub num_segments: usize,

    /// Window width for t-test segmentation
    #[arg(
        long,
        default_value = "5",
        value_name = "N",
        help_heading = "Advanced Options"
    )]
    pub window_width: usize,

    /// Normalization method (zscore, minmax, median, mean, none)
    #[arg(
        long,
        default_value = "zscore",
        value_name = "METHOD",
        help_heading = "Advanced Options"
    )]
    pub normalize: String,

    /// WarpDemuX-compatible fingerprinting mode.
    /// Uses full adapter region, 110 t-test events (window=12, min_sep=6),
    /// keeps last 25 segment means, and applies mean normalization.
    #[arg(long, help_heading = "Advanced Options")]
    pub warpdemux_compat: bool,

    /// Append per-segment dwell time (in samples) as a second feature
    /// channel, normalized with `log1p + z-score`. Doubles the output
    /// feature width from N to 2N columns (`fp_0..N-1, dwell_0..N-1`).
    /// DTW distance in downstream train-svm / classify operates on the
    /// concatenated vector — means first, dwells after. Opt-in because
    /// existing WarpDemux-compat models are 25-wide; turning this on
    /// breaks binary compatibility with them by design.
    #[arg(long, help_heading = "Advanced Options")]
    pub emit_dwell: bool,

    /// Number of threads for parallel processing (default: all CPUs)
    #[arg(short = 't', long, visible_short_alias = 'j', value_name = "N")]
    pub threads: Option<usize>,

    /// Print per-phase timing breakdown after completion
    #[arg(long)]
    pub profile: bool,
}

/// Run the fingerprint subcommand.
pub fn run(args: FingerprintArgs) -> anyhow::Result<()> {
    use crate::commands::profile::PhaseTimer;
    let mut timer = PhaseTimer::new();
    timer.phase("Fingerprint");
    let profile = args.profile;
    // Resolve effective parameters (WarpDemuX-compat overrides defaults)
    let (num_segments, window_width, norm_method, min_separation, keep_last, use_full_adapter) =
        if args.warpdemux_compat {
            (
                111_usize,          // 110 changepoints → 111 segments (WDX num_events=110)
                12_usize,           // running_stat_width=12
                NormMethod::ZScore, // WarpDemuX "mean" norm = z-score (mean/std)
                Some(6_usize),      // min_obs_per_base=6
                Some(25_usize),     // keep last 25 segment means
                true,               // use full adapter region
            )
        } else {
            (
                args.num_segments,
                args.window_width,
                parse_norm_method(&args.normalize)?,
                None,
                None,
                false,
            )
        };

    println!("{} barcode fingerprints", style::action("Extracting"));
    println!(
        "{} {} POD5 file(s)",
        style::label("Input:"),
        style::count(args.input.len())
    );
    println!(
        "{} {}",
        style::label("Boundaries:"),
        style::path(args.boundaries.display())
    );
    println!(
        "{} {}",
        style::label("Output:"),
        style::path(args.output.display())
    );
    if args.warpdemux_compat {
        println!(
            "{} WarpDemuX-compatible (110 events, window=12, keep_last=25, zscore norm)",
            style::label("Mode:"),
        );
    }

    // Set thread pool size
    configure_thread_pool(args.threads);

    // Read boundaries CSV (auto-detects escapepod vs WarpDemuX format)
    let boundaries_map = parse_boundaries_csv(&args.boundaries)?;

    println!(
        "{} {} boundary records with valid adapters",
        style::label("Loaded:"),
        style::count(boundaries_map.len())
    );

    // Upper bound for the progress bar: every boundary record is a candidate
    // read. Some may not appear in the POD5 corpus, in which case the bar
    // stops short of 100% — acceptable for a progress indicator.
    println!(
        "{} up to {} reads to fingerprint",
        style::label("Processing:"),
        style::count(boundaries_map.len())
    );

    let progress_bar = create_progress_bar(boundaries_map.len() as u64, "Fingerprinting")?;

    // Fan out across POD5 files with the outer par_iter; within each file,
    // do the expensive signal decompress + fingerprint with an inner
    // par_iter. Nested rayon work-stealing keeps all CPUs busy even when the
    // read count is skewed across files. Unlike the previous serial-load +
    // par_chunks pattern, there is no single-threaded phase — I/O,
    // SVB16+ZSTD decompression, and fingerprint compute all overlap.
    //
    // Batched progress updates: one pb.inc(64) per 64 reads instead of one
    // atomic RMW per read keeps contention off the hot path.
    const PROGRESS_BATCH: usize = 64;
    let pb_counter = AtomicUsize::new(0);
    let pb_ref = &progress_bar;
    let pb_counter_ref = &pb_counter;

    let fingerprints: Vec<ReadFingerprint> = args
        .input
        .par_iter()
        .map(|path| -> Vec<ReadFingerprint> {
            let Ok(reader) = Reader::open(path) else {
                return Vec::new();
            };
            let Ok(read_iter) = reader.reads() else {
                return Vec::new();
            };
            // Metadata-only pre-filter: boundaries + non-empty signal_rows.
            // No signal I/O yet — the signal decode is the expensive part
            // and happens in the inner par_iter below.
            let reads: Vec<_> = read_iter
                .filter_map(Result::ok)
                .filter(|r| !r.signal_rows.is_empty() && boundaries_map.contains_key(&r.read_id))
                .collect();
            let Ok(extractor) = reader.signal_extractor() else {
                return Vec::new();
            };

            reads
                .par_iter()
                .filter_map(|r| {
                    let signal = extractor.get_signal(&r.signal_rows).ok()?;
                    let boundaries = boundaries_map.get(&r.read_id)?;
                    let (region_start, region_end) = if use_full_adapter {
                        (boundaries.adapter_start, boundaries.adapter_end)
                    } else {
                        let start = boundaries.adapter_start + args.segment_start;
                        let end = (boundaries.adapter_start + args.segment_end)
                            .min(boundaries.adapter_end);
                        (start, end)
                    };
                    if region_end <= region_start {
                        return None;
                    }
                    let fp = extract_fingerprint_from_signal(
                        &signal,
                        region_start,
                        region_end,
                        num_segments,
                        window_width,
                        norm_method,
                        r.read_id,
                        min_separation,
                        keep_last,
                        args.emit_dwell,
                    );
                    // Count attempts (not only successes) so the bar advances
                    // even when a read fails to fingerprint.
                    let count = pb_counter_ref.fetch_add(1, Ordering::Relaxed) + 1;
                    if count.is_multiple_of(PROGRESS_BATCH) {
                        pb_ref.inc(PROGRESS_BATCH as u64);
                    }
                    fp
                })
                .collect()
        })
        .reduce(Vec::new, |mut a, b| {
            a.extend(b);
            a
        });

    // Advance the bar by any reads counted but not yet reflected (tail of
    // the last PROGRESS_BATCH-sized group).
    let total_counted = pb_counter.load(Ordering::Relaxed);
    let remainder = total_counted % PROGRESS_BATCH;
    if remainder > 0 {
        progress_bar.inc(remainder as u64);
    }
    progress_bar.finish_with_message("complete");

    // Write fingerprints — dispatch by extension so downstream consumers
    // (join_labels.py, concat_labeled.py, escpod's own fp_io reader)
    // work with either CSV or parquet. Parquet is 3-4x smaller, faster to
    // read, and avoids the text parse roundtrip through polars.
    write_fingerprints(&args.output, &fingerprints)?;

    eprintln!(
        "{} {} fingerprints written to {}",
        style::action("Extracted"),
        style::count(fingerprints.len()),
        style::path(args.output.display())
    );

    timer.report(profile);

    Ok(())
}

/// Write fingerprints, choosing parquet or CSV by output extension.
///
/// `.parquet` → columnar zstd-compressed arrow/parquet (3-4x smaller on disk
/// than CSV, ~10x faster downstream read). Anything else falls back to the
/// flat CSV writer. No format sniffing — the caller picks the shape by
/// naming their output `*.parquet` vs `*.csv`.
fn write_fingerprints(path: &Path, fingerprints: &[ReadFingerprint]) -> anyhow::Result<()> {
    match path.extension().and_then(|s| s.to_str()) {
        Some("parquet") => write_fingerprints_parquet(path, fingerprints),
        _ => write_fingerprints_csv(path, fingerprints),
    }
}

/// Write fingerprints as an Arrow/parquet table, zstd-compressed.
///
/// Schema matches the CSV writer (`read_id, fp_0..N-1` plus optional
/// `dwell_0..N-1`) so `escpod`'s own `fp_io::read_query_fingerprints` and
/// the Python `polars.scan_parquet` path in `scripts/join_labels.py` read
/// the same column set regardless of which format produced it.
///
/// Single-batch write: the full fingerprint set is materialized as one
/// `RecordBatch` and handed to `ArrowWriter`. For 11M rows × 50 columns
/// peak memory is ~4.4 GB of column buffers plus Arrow overhead — well
/// inside the `fingerprint_run` SLURM allocation. If this ever outgrows
/// that, swap in a row-group loop that batches N rows at a time.
fn write_fingerprints_parquet(path: &Path, fingerprints: &[ReadFingerprint]) -> anyhow::Result<()> {
    let (fp_width, dwell_width) = match fingerprints.first() {
        Some(first) => (
            first.values.len(),
            first.dwell_times.as_ref().map_or(0, |d| d.len()),
        ),
        None => (0, 0),
    };

    // Schema: read_id + fp_* + dwell_* (dwell_* only if present).
    let mut fields: Vec<Field> = Vec::with_capacity(1 + fp_width + dwell_width);
    fields.push(Field::new("read_id", DataType::Utf8, false));
    for i in 0..fp_width {
        fields.push(Field::new(format!("fp_{i}"), DataType::Float64, false));
    }
    for i in 0..dwell_width {
        fields.push(Field::new(format!("dwell_{i}"), DataType::Float64, false));
    }
    let schema = Arc::new(Schema::new(fields));

    // Single-pass columnar accumulation — one iteration over
    // `fingerprints` fills every column, much cache-friendlier than 1 +
    // fp_width + dwell_width separate iterations.
    let n = fingerprints.len();
    let mut read_ids: Vec<String> = Vec::with_capacity(n);
    let mut fp_cols: Vec<Vec<f64>> = (0..fp_width).map(|_| Vec::with_capacity(n)).collect();
    let mut dwell_cols: Vec<Vec<f64>> = (0..dwell_width).map(|_| Vec::with_capacity(n)).collect();

    for fp in fingerprints {
        read_ids.push(fp.read_id.to_string());
        for (i, v) in fp.values.iter().enumerate() {
            fp_cols[i].push(*v);
        }
        if let Some(d) = &fp.dwell_times {
            for (i, v) in d.iter().enumerate() {
                dwell_cols[i].push(*v);
            }
        }
    }

    let mut columns: Vec<ArrayRef> = Vec::with_capacity(1 + fp_width + dwell_width);
    columns.push(Arc::new(StringArray::from(read_ids)));
    for col in fp_cols {
        columns.push(Arc::new(Float64Array::from(col)));
    }
    for col in dwell_cols {
        columns.push(Arc::new(Float64Array::from(col)));
    }

    let batch = RecordBatch::try_new(schema.clone(), columns)?;

    let file = File::create(path)?;
    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(Default::default()))
        .build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

/// Write fingerprints to a CSV file.
///
/// Schema:
///   without dwell: `read_id, fp_0, ..., fp_{N-1}`
///   with dwell:    `read_id, fp_0, ..., fp_{N-1}, dwell_0, ..., dwell_{N-1}`
///
/// Dwell presence is sampled from the first row; mixing dwell-carrying and
/// plain `ReadFingerprint`s in the same call would produce jagged rows and
/// is caller error — the extractor always emits one shape per run.
fn write_fingerprints_csv(path: &Path, fingerprints: &[ReadFingerprint]) -> anyhow::Result<()> {
    let output_file = File::create(path)?;
    let mut writer = BufWriter::new(output_file);

    let (fp_width, dwell_width) = match fingerprints.first() {
        Some(first) => (
            first.values.len(),
            first.dwell_times.as_ref().map_or(0, |d| d.len()),
        ),
        None => (0, 0),
    };

    write!(writer, "read_id")?;
    for i in 0..fp_width {
        write!(writer, ",fp_{}", i)?;
    }
    for i in 0..dwell_width {
        write!(writer, ",dwell_{}", i)?;
    }
    writeln!(writer)?;

    for fp in fingerprints {
        write!(writer, "{}", fp.read_id)?;
        for val in &fp.values {
            write!(writer, ",{:.6}", val)?;
        }
        if let Some(dwells) = &fp.dwell_times {
            for val in dwells {
                write!(writer, ",{:.6}", val)?;
            }
        }
        writeln!(writer)?;
    }

    writer.flush()?;
    Ok(())
}
