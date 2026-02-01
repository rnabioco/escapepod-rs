//! Resquiggle command: refine signal-to-base mapping using banded DP.

use std::collections::HashMap;
use std::io::BufReader;
use std::path::PathBuf;

use anyhow::bail;
use bstr::ByteSlice;
use clap::Args;
use rayon::prelude::*;

use escapepod::parse_uuid_flexible;
use escapepod::resquiggle::{
    calculate_initial_scaling, refine_signal_map, KmerTable, RefineAlgo, RefineSettings,
    RescaleAlgo, RoughRescaleAlgo,
};

use noodles_bam as bam;
use noodles_sam as sam;
use sam::alignment::record::data::field::Tag;
use sam::alignment::record_buf::data::field::value::Array;
use sam::alignment::record_buf::data::field::Value;
use sam::alignment::RecordBuf;

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
    #[arg(long, default_value = "dwell-penalty", value_parser = parse_algo)]
    pub algo: RefineAlgo,

    /// Number of refinement iterations
    #[arg(long, default_value = "1")]
    pub iterations: usize,

    /// Half bandwidth for banded DP
    #[arg(long, default_value = "5")]
    pub half_bandwidth: usize,

    /// Rescale algorithm
    #[arg(long, default_value = "theil-sen", value_parser = parse_rescale)]
    pub rescale: RescaleAlgo,

    /// Apply MAD normalization to kmer levels
    #[arg(long)]
    pub normalize_levels: bool,

    /// Number of threads for parallel processing
    #[arg(short = 'j', long)]
    pub threads: Option<usize>,
}

fn parse_algo(s: &str) -> Result<RefineAlgo, String> {
    match s {
        "viterbi" => Ok(RefineAlgo::Viterbi),
        "dwell-penalty" => Ok(RefineAlgo::default()),
        _ => Err(format!(
            "unknown algorithm '{}', expected 'viterbi' or 'dwell-penalty'",
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

    let settings = RefineSettings {
        refinement_algo: args.algo.clone(),
        n_refinement_iters: args.iterations,
        half_bandwidth: args.half_bandwidth,
        adjust_band_min_size: 2,
        rescale_algo: args.rescale.clone(),
        rough_rescale_algo: RoughRescaleAlgo::None,
        normalize_levels: args.normalize_levels,
    };

    // --- Phase 1: Load ---
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

    // Load POD5 reads into a map by UUID
    let pod5_files = resolve_pod5_inputs(&args.input)?;
    println!(
        "{} POD5 data from {} ({})",
        style::action("Loading"),
        style::path(args.input.display()),
        if pod5_files.len() > 1 {
            format!("{} files", pod5_files.len())
        } else {
            "1 file".to_string()
        }
    );

    // Open all POD5 readers and collect reads
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
    println!(
        "  {} reads indexed from POD5",
        style::count(pod5_reads.len())
    );

    // Scan BAM file
    println!(
        "{} BAM records from {}",
        style::action("Loading"),
        style::path(args.bam.display())
    );
    let file = std::fs::File::open(&args.bam)?;
    let mut bam_reader = bam::io::Reader::new(BufReader::new(file));
    let header = bam_reader.read_header()?;

    let mut records: Vec<RecordBuf> = Vec::new();
    let mut record_buf = RecordBuf::default();
    while bam_reader.read_record_buf(&header, &mut record_buf)? != 0 {
        records.push(record_buf.clone());
        record_buf = RecordBuf::default();
    }
    println!("  {} BAM records loaded", style::count(records.len()));

    // --- Phase 2: Refine (parallel) ---
    println!(
        "{} signal-to-base mappings (half_bandwidth={}, iterations={}, algo={:?})",
        style::action("Refining"),
        settings.half_bandwidth,
        settings.n_refinement_iters,
        settings.refinement_algo,
    );

    let refined_count = std::sync::atomic::AtomicUsize::new(0);
    let skip_count = std::sync::atomic::AtomicUsize::new(0);
    let error_count = std::sync::atomic::AtomicUsize::new(0);

    records.par_iter_mut().for_each(|record| {
        match refine_single_read(record, &pod5_readers, &pod5_reads, &kmer_table, &settings) {
            Ok(true) => {
                refined_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            Ok(false) => {
                skip_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            Err(_) => {
                error_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }
    });

    let refined = refined_count.load(std::sync::atomic::Ordering::Relaxed);
    let skipped = skip_count.load(std::sync::atomic::Ordering::Relaxed);
    let errors = error_count.load(std::sync::atomic::Ordering::Relaxed);

    println!(
        "  {} refined, {} skipped, {} errors",
        style::count(refined),
        skipped,
        errors
    );

    // --- Phase 3: Write ---
    println!(
        "{} output BAM to {}",
        style::action("Writing"),
        style::path(args.output.display())
    );

    let output_file = std::fs::File::create(&args.output)?;
    let mut writer = bam::io::Writer::new(output_file);
    writer.write_header(&header)?;

    for record in &records {
        use sam::alignment::io::Write as _;
        writer.write_alignment_record(&header, record)?;
    }

    // Finish BGZF
    let inner = writer.into_inner();
    inner.finish()?;

    println!(
        "{} {} records written to {}",
        style::action("Done:"),
        style::count(records.len()),
        style::path(args.output.display())
    );

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
    record: &mut RecordBuf,
    pod5_readers: &[escapepod::Reader],
    pod5_reads: &HashMap<uuid::Uuid, Pod5ReadInfo>,
    kmer_table: &KmerTable,
    settings: &RefineSettings,
) -> anyhow::Result<bool> {
    // Extract read ID from query name
    let read_id = match record.name() {
        Some(name) => {
            let name_bytes: &[u8] = name.as_ref();
            let name_str = name_bytes.to_str()?;
            parse_uuid_flexible(name_str)?
        }
        None => return Ok(false),
    };

    // Look up POD5 read
    let pod5_info = match pod5_reads.get(&read_id) {
        Some(info) => info,
        None => return Ok(false), // No matching POD5 read
    };

    // Extract move table from mv tag
    let mv_tag = Tag::new(b'm', b'v');
    let (stride, moves) = match record.data().get(&mv_tag) {
        Some(Value::Array(Array::UInt8(data))) => {
            if data.len() < 2 {
                return Ok(false);
            }
            (data[0] as usize, data[1..].to_vec())
        }
        Some(Value::Array(Array::Int8(data))) => {
            if data.len() < 2 {
                return Ok(false);
            }
            (
                data[0] as usize,
                data[1..].iter().map(|&b| b as u8).collect::<Vec<u8>>(),
            )
        }
        _ => return Ok(false), // No move table
    };

    if stride == 0 {
        return Ok(false);
    }

    // Extract sequence
    let sequence = decode_sequence(record.sequence());
    if sequence.is_empty() {
        return Ok(false);
    }

    // Build initial query-to-signal map from move table
    let seq_to_signal = build_query_to_signal_map(&moves, stride, sequence.len())?;

    // Get raw signal from POD5
    let reader = &pod5_readers[pod5_info.reader_idx];
    let signal_i16 = reader.get_signal(&pod5_info.signal_rows)?;
    let signal_f32: Vec<f32> = signal_i16.iter().map(|&s| s as f32).collect();

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
    let levels = match kmer_table.extract_levels(&sequence) {
        Ok(l) => l,
        Err(_) => return Ok(false), // Sequence too short or contains non-ACGT
    };

    // Verify map is valid for the signal
    if let Some(&last) = seq_to_signal.last() {
        if last > signal_f32.len() {
            return Ok(false);
        }
    }

    // Run refinement
    let result = match refine_signal_map(
        settings,
        &signal_f32,
        &seq_to_signal,
        &levels,
        initial_scale,
        initial_shift,
    ) {
        Ok(r) => r,
        Err(_) => return Ok(false),
    };

    // Insert refined tags into the BAM record
    let rs_tag = Tag::new(b'r', b's');
    let rc_tag = Tag::new(b'r', b'c');
    let ro_tag = Tag::new(b'r', b'o');

    let boundaries: Vec<u32> = result.seq_to_signal_map.iter().map(|&x| x as u32).collect();

    record
        .data_mut()
        .insert(rs_tag, Value::Array(Array::UInt32(boundaries)));
    record.data_mut().insert(rc_tag, Value::Float(result.scale));
    record.data_mut().insert(ro_tag, Value::Float(result.shift));

    Ok(true)
}

/// Decode a RecordBuf sequence from BAM 4-bit encoding to ASCII bytes.
fn decode_sequence(seq: &sam::alignment::record_buf::Sequence) -> Vec<u8> {
    seq.as_ref()
        .iter()
        .map(|&b| match b {
            1 => b'A',
            2 => b'C',
            4 => b'G',
            8 => b'T',
            _ => b'N',
        })
        .collect()
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
