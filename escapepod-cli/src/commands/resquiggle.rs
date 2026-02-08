//! Resquiggle command: refine signal-to-base mapping using banded DP.

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::bail;
use bstr::ByteSlice;
use clap::Args;
use rayon::prelude::*;

use escapepod::parse_uuid_flexible;
use escapepod::resquiggle::{
    calculate_initial_scaling, refine_signal_map, BandingAlgo, KmerTable, RefineAlgo,
    RefineSettings, RescaleAlgo, RoughRescaleAlgo,
};
use noodles_bam as bam;
use noodles_bgzf as bgzf;
use noodles_sam as sam;
use sam::alignment::record::data::field::Tag;
use sam::alignment::record_buf::data::field::value::Array;
use sam::alignment::record_buf::data::field::Value;
use sam::alignment::RecordBuf;
use sam::header::record::value::map::{program::tag as pg_tag, Map, Program};

use crate::progress::create_spinner;
use crate::style;
use crate::util::resolve_pod5_inputs;

#[derive(Args)]
pub struct ResquiggleArgs {
    /// Input POD5 file or directory
    pub input: PathBuf,

    /// Input BAM file with move table (mv tag)
    #[arg(short, long, required = true)]
    pub bam: PathBuf,

    /// Tab-delimited kmer level table file
    #[arg(short, long, required = true)]
    pub kmer_table: PathBuf,

    /// Output BAM file
    #[arg(short, long, required = true)]
    pub output: PathBuf,

    /// Refinement algorithm
    #[arg(long, default_value = "dwell-penalty", value_parser = parse_algo, help_heading = "Advanced Options")]
    pub algo: RefineAlgo,

    /// Number of refinement iterations
    #[arg(long, default_value = "2", help_heading = "Advanced Options")]
    pub iterations: usize,

    /// Half bandwidth for banded DP
    #[arg(long, default_value = "5", help_heading = "Advanced Options")]
    pub half_bandwidth: usize,

    /// Rescale algorithm
    #[arg(long, default_value = "theil-sen", value_parser = parse_rescale, help_heading = "Advanced Options")]
    pub rescale: RescaleAlgo,

    /// Target dwell time per base for dwell-penalty algorithm (0 = auto from move table)
    #[arg(long, default_value = "0", help_heading = "Advanced Options")]
    pub dwell_target: f32,

    /// Dwell penalty weight for dwell-penalty algorithm
    #[arg(long, default_value = "0.5", help_heading = "Advanced Options")]
    pub dwell_weight: f32,

    /// Normalization mode for kmer levels (e.g., 'mad')
    #[arg(long, value_parser = parse_normalize, value_name = "MODE", help_heading = "Advanced Options")]
    pub normalize: Option<NormalizeMode>,

    /// Banding algorithm: 'fixed' (default) or 'adaptive'
    #[arg(long, default_value = "fixed", value_parser = parse_banding, help_heading = "Advanced Options")]
    pub banding: BandingAlgo,

    /// X-drop threshold for adaptive banding early termination (optional).
    /// When the best per-base DP score exceeds the global best by more than
    /// this value, the DP bails out and returns the initial map.
    #[arg(long, help_heading = "Advanced Options")]
    pub x_drop: Option<f32>,

    /// Number of threads for parallel processing
    #[arg(short = 'j', long)]
    pub threads: Option<usize>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum NormalizeMode {
    Mad,
}

fn parse_normalize(s: &str) -> Result<NormalizeMode, String> {
    match s {
        "mad" => Ok(NormalizeMode::Mad),
        _ => Err(format!(
            "unknown normalization mode '{}', expected 'mad'",
            s
        )),
    }
}

fn parse_algo(s: &str) -> Result<RefineAlgo, String> {
    match s {
        "viterbi" => Ok(RefineAlgo::Viterbi),
        // target/weight are placeholders; overridden by --dwell-target/--dwell-weight
        "dwell-penalty" => Ok(RefineAlgo::DwellPenalty {
            target: 0.0,
            weight: 0.5,
        }),
        _ => Err(format!(
            "unknown algorithm '{}', expected 'viterbi' or 'dwell-penalty'",
            s
        )),
    }
}

fn parse_banding(s: &str) -> Result<BandingAlgo, String> {
    match s {
        "fixed" => Ok(BandingAlgo::Fixed),
        "adaptive" => Ok(BandingAlgo::Adaptive {
            bandwidth: 0, // sentinel: will use half_bandwidth * 2
            x_drop: None, // sentinel: will use --x-drop flag
        }),
        _ => Err(format!(
            "unknown banding algorithm '{}', expected 'fixed' or 'adaptive'",
            s
        )),
    }
}

fn parse_rescale(s: &str) -> Result<RescaleAlgo, String> {
    match s {
        "theil-sen" => Ok(RescaleAlgo::default()),
        "least-squares" => Ok(RescaleAlgo::LeastSquares {
            dwell_filter_lower_percentile: 0.1,
            dwell_filter_upper_percentile: 0.9,
            min_abs_level: 0.2,
            n_bases_truncate: 10,
            min_num_filtered_levels: 10,
        }),
        _ => Err(format!(
            "unknown rescale algorithm '{}', expected 'theil-sen' or 'least-squares'",
            s
        )),
    }
}

pub fn run(args: ResquiggleArgs) -> anyhow::Result<()> {
    // Configure thread pool
    if let Some(threads) = args.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .ok(); // Ignore error if pool is already initialized
    }

    // Resolve adaptive bandwidth from half_bandwidth if sentinel value
    let banding = match args.banding {
        BandingAlgo::Adaptive { bandwidth: 0, .. } => BandingAlgo::Adaptive {
            bandwidth: args.half_bandwidth * 2,
            x_drop: args.x_drop,
        },
        other => other,
    };

    // Apply --dwell-target and --dwell-weight to the algo if dwell-penalty was chosen
    let algo = match args.algo {
        RefineAlgo::DwellPenalty { .. } => RefineAlgo::DwellPenalty {
            target: args.dwell_target,
            weight: args.dwell_weight,
        },
        other => other,
    };

    let settings = RefineSettings {
        refinement_algo: algo,
        n_refinement_iters: args.iterations,
        half_bandwidth: args.half_bandwidth,
        adjust_band_min_size: 2,
        rescale_algo: args.rescale.clone(),
        rough_rescale_algo: RoughRescaleAlgo::default(),
        normalize_levels: args.normalize == Some(NormalizeMode::Mad),
        banding_algo: banding,
    };

    // --- Phase 1: Load kmer table ---
    println!(
        "{} kmer table from {}",
        style::action("Loading"),
        style::path(args.kmer_table.display())
    );
    let mut kmer_table = KmerTable::from_file(&args.kmer_table)?;
    if settings.normalize_levels {
        kmer_table.fix_gauge()?;
        println!("  Applied MAD normalization to kmer levels");
    }

    // --- Phase 2: Index all POD5 reads ---
    let pod5_files = resolve_pod5_inputs(&args.input)?;
    let pod5_spinner = create_spinner("Indexing")?;
    pod5_spinner.set_message(format!(
        "POD5 data from {} ({})",
        args.input.display(),
        if pod5_files.len() > 1 {
            format!("{} files", pod5_files.len())
        } else {
            "1 file".to_string()
        }
    ));

    let mut pod5_reads: HashMap<uuid::Uuid, Pod5ReadInfo> = HashMap::new();
    let mut pod5_readers: Vec<escapepod::Reader> = Vec::new();

    for (reader_idx, path) in pod5_files.iter().enumerate() {
        let reader = escapepod::Reader::open(path)?;
        for read_result in reader.reads()? {
            let read = read_result?;
            pod5_reads.insert(
                read.read_id,
                Pod5ReadInfo {
                    reader_idx,
                    calibration_scale: read.calibration_scale,
                    calibration_offset: read.calibration_offset,
                    signal_rows: read.signal_rows.clone(),
                },
            );
        }
        pod5_readers.push(reader);
    }
    pod5_spinner.finish_with_message(format!(
        "{} reads indexed from POD5",
        style::count(pod5_reads.len())
    ));
    eprintln!(
        "[resquiggle] {} reads indexed from {} POD5 file(s)",
        pod5_reads.len(),
        pod5_files.len()
    );

    // Create signal extractors (one per reader) for parallel on-demand extraction
    let signal_extractors: Vec<_> = pod5_readers
        .iter()
        .map(|r| r.signal_extractor())
        .collect::<escapepod::Result<_>>()?;

    // --- Phase 3: Stream BAM, refine in parallel, write asynchronously ---
    let file = std::fs::File::open(&args.bam)?;
    let worker_count = args
        .threads
        .and_then(std::num::NonZeroUsize::new)
        .or_else(|| std::thread::available_parallelism().ok())
        .unwrap_or(std::num::NonZeroUsize::MIN);
    let decoder = bgzf::io::MultithreadedReader::with_worker_count(worker_count, file);
    let mut bam_reader = bam::io::Reader::from(decoder);
    let mut header = bam_reader.read_header()?;

    // Add @PG record with resquiggle parameters
    let normalize_str = match &args.normalize {
        Some(NormalizeMode::Mad) => " --normalize mad",
        None => "",
    };
    let dwell_str = match &settings.refinement_algo {
        RefineAlgo::DwellPenalty { target, weight } => {
            format!(" --dwell-target {} --dwell-weight {}", target, weight)
        }
        RefineAlgo::Viterbi => String::new(),
    };
    let command_line = format!(
        "escpod resquiggle --algo {} --iterations {} --half-bandwidth {} --rescale {}{}{}",
        match &settings.refinement_algo {
            RefineAlgo::Viterbi => "viterbi",
            RefineAlgo::DwellPenalty { .. } => "dwell-penalty",
        },
        settings.n_refinement_iters,
        settings.half_bandwidth,
        match &settings.rescale_algo {
            RescaleAlgo::TheilSen { .. } => "theil-sen",
            RescaleAlgo::LeastSquares { .. } => "least-squares",
        },
        dwell_str,
        normalize_str,
    );
    let pg = Map::<Program>::builder()
        .insert(pg_tag::NAME, "escpod")
        .insert(pg_tag::VERSION, env!("CARGO_PKG_VERSION"))
        .insert(pg_tag::COMMAND_LINE, command_line)
        .build()?;
    header.programs_mut().add("escpod-resquiggle", pg)?;

    let stats = RefineStats::new();

    // Spawn writer thread — receives ordered chunks via channel
    let (tx, rx) = std::sync::mpsc::sync_channel::<Vec<RecordBuf>>(2);
    let header_clone = header.clone();
    let output_path = args.output.clone();
    let writer_handle = std::thread::spawn(move || -> anyhow::Result<usize> {
        let output_file = std::fs::File::create(&output_path)?;
        let encoder = bgzf::io::MultithreadedWriter::with_worker_count(worker_count, output_file);
        let mut writer = bam::io::Writer::from(encoder);
        writer.write_header(&header_clone)?;

        let mut count = 0;
        for chunk in rx {
            for record in &chunk {
                use sam::alignment::io::Write as _;
                writer.write_alignment_record(&header_clone, record)?;
                count += 1;
            }
        }

        let mut inner = writer.into_inner();
        inner.finish()?;
        Ok(count)
    });

    // Stream BAM records, filter against POD5 index, and process in chunks
    // Each chunk entry carries the pre-parsed UUID to avoid double parsing.
    const CHUNK_SIZE: usize = 10_000;
    let stream_spinner = create_spinner("Processing")?;
    let mut chunk: Vec<(uuid::Uuid, RecordBuf)> = Vec::with_capacity(CHUNK_SIZE);
    let mut total_bam: usize = 0;
    let mut matched: usize = 0;

    loop {
        let mut record_buf = RecordBuf::default();
        if bam_reader.read_record_buf(&header, &mut record_buf)? == 0 {
            break;
        }
        total_bam += 1;

        // Parse UUID once and check for POD5 match
        let read_id = record_buf.name().and_then(|name| {
            let name_bytes: &[u8] = name.as_ref();
            let name_str = name_bytes.to_str().ok()?;
            let uuid = parse_uuid_flexible(name_str).ok()?;
            if pod5_reads.contains_key(&uuid) {
                Some(uuid)
            } else {
                None
            }
        });

        let read_id = match read_id {
            Some(id) => id,
            None => continue,
        };
        matched += 1;
        chunk.push((read_id, record_buf));

        if total_bam % 50_000 == 0 {
            stream_spinner.set_message(format!(
                "{} matched / {} scanned from BAM",
                style::count(matched),
                style::count(total_bam)
            ));
            eprintln!(
                "[resquiggle] {} matched / {} scanned from BAM",
                matched, total_bam
            );
        }

        if chunk.len() >= CHUNK_SIZE {
            refine_and_send_chunk(
                &mut chunk,
                &pod5_reads,
                &signal_extractors,
                &kmer_table,
                &settings,
                &stats,
                &tx,
            )?;
            chunk = Vec::with_capacity(CHUNK_SIZE);
        }
    }

    // Flush remaining records
    if !chunk.is_empty() {
        refine_and_send_chunk(
            &mut chunk,
            &pod5_reads,
            &signal_extractors,
            &kmer_table,
            &settings,
            &stats,
            &tx,
        )?;
    }
    drop(tx);

    stream_spinner.finish_with_message(format!(
        "{} matched / {} scanned from BAM",
        style::count(matched),
        style::count(total_bam)
    ));
    eprintln!(
        "[resquiggle] processing complete: {} matched / {} scanned from BAM",
        matched, total_bam
    );

    // Wait for writer to finish
    let written = writer_handle
        .join()
        .map_err(|_| anyhow::anyhow!("writer thread panicked"))??;

    let refined = stats
        .refined_count
        .load(std::sync::atomic::Ordering::Relaxed);
    let errors = stats.error_count.load(std::sync::atomic::Ordering::Relaxed);

    println!(
        "{} {} refined, {} errors, {} written to {}",
        style::action("Done:"),
        style::count(refined),
        errors,
        style::count(written),
        style::path(args.output.display())
    );
    if errors > 0 {
        let reasons = stats.skip_reasons.lock().unwrap();
        for (reason, count) in reasons.iter() {
            eprintln!("  error ({}x): {}", count, reason);
        }
    }

    Ok(())
}

/// Counters for tracking refinement progress and errors.
struct RefineStats {
    refined_count: std::sync::atomic::AtomicUsize,
    error_count: std::sync::atomic::AtomicUsize,
    skip_reasons: std::sync::Mutex<HashMap<String, usize>>,
}

impl RefineStats {
    fn new() -> Self {
        Self {
            refined_count: std::sync::atomic::AtomicUsize::new(0),
            error_count: std::sync::atomic::AtomicUsize::new(0),
            skip_reasons: std::sync::Mutex::new(HashMap::new()),
        }
    }
}

/// Refine a chunk of BAM records in parallel and send to the writer thread.
fn refine_and_send_chunk(
    chunk: &mut Vec<(uuid::Uuid, RecordBuf)>,
    pod5_reads: &HashMap<uuid::Uuid, Pod5ReadInfo>,
    signal_extractors: &[escapepod::SignalExtractor<'_>],
    kmer_table: &KmerTable,
    settings: &RefineSettings,
    stats: &RefineStats,
    tx: &std::sync::mpsc::SyncSender<Vec<RecordBuf>>,
) -> anyhow::Result<()> {
    chunk.par_iter_mut().for_each(|(read_id, record)| {
        match refine_single_read(
            *read_id,
            record,
            pod5_reads,
            signal_extractors,
            kmer_table,
            settings,
        ) {
            Ok(true) => {
                stats
                    .refined_count
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            Ok(false) => {}
            Err(e) => {
                stats
                    .error_count
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let reason = format!("{}", e);
                stats
                    .skip_reasons
                    .lock()
                    .unwrap()
                    .entry(reason)
                    .and_modify(|c| *c += 1)
                    .or_insert(1);
            }
        }
    });
    // Extract just the RecordBufs for the writer
    let records: Vec<RecordBuf> = std::mem::take(chunk).into_iter().map(|(_, r)| r).collect();
    tx.send(records)?;
    Ok(())
}

/// POD5 read metadata needed for refinement.
struct Pod5ReadInfo {
    reader_idx: usize,
    calibration_scale: f32,
    calibration_offset: f32,
    signal_rows: Vec<u64>,
}

/// Refine a single BAM record's signal-to-base mapping.
/// Returns Ok(true) if refined, Ok(false) if skipped.
fn refine_single_read(
    read_id: uuid::Uuid,
    record: &mut RecordBuf,
    pod5_reads: &HashMap<uuid::Uuid, Pod5ReadInfo>,
    signal_extractors: &[escapepod::SignalExtractor<'_>],
    kmer_table: &KmerTable,
    settings: &RefineSettings,
) -> anyhow::Result<bool> {
    // Look up POD5 read (UUID already parsed by caller)
    let pod5_info = match pod5_reads.get(&read_id) {
        Some(info) => info,
        None => bail!("no matching POD5 read"),
    };

    // Extract move table from mv tag
    let mv_tag = Tag::new(b'm', b'v');
    let (stride, moves) = match record.data().get(&mv_tag) {
        Some(Value::Array(Array::UInt8(data))) => {
            if data.len() < 2 {
                bail!("mv tag too short (UInt8)");
            }
            (data[0] as usize, data[1..].to_vec())
        }
        Some(Value::Array(Array::Int8(data))) => {
            if data.len() < 2 {
                bail!("mv tag too short (Int8)");
            }
            (
                data[0] as usize,
                data[1..].iter().map(|&b| b as u8).collect::<Vec<u8>>(),
            )
        }
        _ => bail!("no mv tag"),
    };

    if stride == 0 {
        bail!("stride is 0");
    }

    // Extract sequence (noodles already decodes BAM 4-bit to ASCII)
    let sequence: &[u8] = record.sequence().as_ref();
    if sequence.is_empty() {
        bail!("empty sequence");
    }

    // Build initial query-to-signal map from move table
    let seq_to_signal = build_query_to_signal_map(&moves, stride, sequence.len())?;

    // Extract signal on-demand (parallel decompression) — kept as i16
    let signal_i16 = signal_extractors[pod5_info.reader_idx].get_signal(&pod5_info.signal_rows)?;

    // Compute signal trimming bounds on the i16 signal (no conversion yet)
    let sp_tag = Tag::new(b's', b'p');
    let ts_tag = Tag::new(b't', b's');
    let ns_tag = Tag::new(b'n', b's');

    let parent_signal_offset = get_int_tag(record, &sp_tag).unwrap_or(0) as usize;
    let trimmed_signal_len = get_int_tag(record, &ts_tag).unwrap_or(0) as usize;
    let subread_signal_len = get_int_tag(record, &ns_tag);

    let signal_start = parent_signal_offset + trimmed_signal_len;
    let signal_end = match subread_signal_len {
        Some(ns) => parent_signal_offset + ns as usize,
        None => signal_i16.len(),
    };

    if signal_start >= signal_end || signal_end > signal_i16.len() {
        bail!(
            "invalid signal trim: start={}, end={}, signal_len={}",
            signal_start,
            signal_end,
            signal_i16.len()
        );
    }

    // Convert only the trimmed range from i16 to f32
    let signal_f32: Vec<f32> = signal_i16[signal_start..signal_end]
        .iter()
        .map(|&s| s as f32)
        .collect();

    // Get initial scaling from POD5 calibration + BAM sm/sd tags
    let sm_tag = Tag::new(b's', b'm');
    let sd_tag = Tag::new(b's', b'd');
    let shift_pa = get_float_tag(record, &sm_tag).unwrap_or(0.0);
    let scale_pa = get_float_tag(record, &sd_tag).unwrap_or(1.0);

    let (initial_scale, initial_shift) = calculate_initial_scaling(
        pod5_info.calibration_scale,
        pod5_info.calibration_offset,
        scale_pa,
        shift_pa,
    );

    // Extract expected levels from kmer table
    let levels = match kmer_table.extract_levels(sequence) {
        Ok(l) => l,
        Err(e) => bail!("kmer levels: {}", e),
    };

    // Verify map is valid for the trimmed signal
    if let Some(&last) = seq_to_signal.last() {
        if last > signal_f32.len() {
            bail!(
                "map end {} > trimmed signal len {} (seq_len={}, moves={}, stride={}, trim_start={}, trim_end={})",
                last,
                signal_f32.len(),
                sequence.len(),
                moves.len(),
                stride,
                signal_start,
                signal_end,
            );
        }
    }

    // Run refinement on trimmed signal
    let result = match refine_signal_map(
        settings,
        &signal_f32,
        &seq_to_signal,
        &levels,
        initial_scale,
        initial_shift,
    ) {
        Ok(r) => r,
        Err(e) => bail!("refinement failed: {}", e),
    };

    // Insert refined tags into the BAM record
    // Add signal_start offset back so boundaries are in full-signal coordinates
    let rs_tag = Tag::new(b'r', b's');
    let rc_tag = Tag::new(b'r', b'c');
    let ro_tag = Tag::new(b'r', b'o');

    let boundaries: Vec<u32> = result
        .seq_to_signal_map
        .iter()
        .map(|&x| (x + signal_start) as u32)
        .collect();

    let rd_tag = Tag::new(b'r', b'd');

    record
        .data_mut()
        .insert(rs_tag, Value::Array(Array::UInt32(boundaries)));
    record.data_mut().insert(rc_tag, Value::Float(result.scale));
    record.data_mut().insert(ro_tag, Value::Float(result.shift));
    record.data_mut().insert(rd_tag, Value::Float(result.drift));

    Ok(true)
}

/// Build a query-to-signal map from the BAM move table.
///
/// The move table has one entry per stride-sized signal block.
/// A value of 1 means "move to next base", 0 means "stay".
fn build_query_to_signal_map(
    moves: &[u8],
    stride: usize,
    seq_len: usize,
) -> anyhow::Result<Vec<usize>> {
    let mut map = Vec::with_capacity(seq_len + 1);

    for (i, &m) in moves.iter().enumerate() {
        if m == 1 {
            map.push(i * stride);
        }
    }

    // End boundary
    map.push(moves.len() * stride);

    if map.len() != seq_len + 1 {
        bail!(
            "move table produced {} boundaries, expected {} (seq_len={}, moves_len={}, stride={})",
            map.len(),
            seq_len + 1,
            seq_len,
            moves.len(),
            stride
        );
    }

    Ok(map)
}

/// Extract a float value from a BAM auxiliary tag.
fn get_float_tag(record: &RecordBuf, tag: &Tag) -> Option<f32> {
    match record.data().get(tag) {
        Some(Value::Float(f)) => Some(*f),
        _ => None,
    }
}

/// Extract an integer value from a BAM auxiliary tag.
fn get_int_tag(record: &RecordBuf, tag: &Tag) -> Option<i64> {
    match record.data().get(tag) {
        Some(Value::Int8(v)) => Some(*v as i64),
        Some(Value::UInt8(v)) => Some(*v as i64),
        Some(Value::Int16(v)) => Some(*v as i64),
        Some(Value::UInt16(v)) => Some(*v as i64),
        Some(Value::Int32(v)) => Some(*v as i64),
        Some(Value::UInt32(v)) => Some(*v as i64),
        _ => None,
    }
}
