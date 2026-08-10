//! `escpod classify` — tRNA charging (aminoacylation) classification from
//! POD5 + aligned BAM, writing the call as a `cl` tag on the BAM directly.
//!
//! Unlike `escpod demux`, which anchors on a signal-derived `adapter_end`,
//! the charging model anchors on the CCA–aa junction, which only exists in
//! reference coordinates — hence the aligned BAM with move tables (the same
//! input pair `remora infer from_pod5_and_bam` takes, so this drops into the
//! existing aa-tRNA-seq pipeline where the BAM already exists).
//!
//! The feature recipe (offsets, stat layout, k-mer table pinned by sha256,
//! recommended operating point) comes from the model bundle's
//! `metadata.json`, not from flags — a caller computing the features
//! differently gets a wrong answer, not an error. See
//! `escapepod_classify::bundle` for the contract.

use anyhow::{Context, bail};
use clap::Args;
use rayon::prelude::*;
use std::collections::HashMap;
use std::io::Write as _;
use std::path::PathBuf;
use tracing::{info, warn};

use noodles_bam as bam;
use noodles_bgzf as bgzf;
use noodles_sam as sam;
use sam::alignment::RecordBuf;
use sam::alignment::record::data::field::Tag;
use sam::alignment::record_buf::data::field::Value;
use sam::header::record::value::map::{Map, Program, program::tag as pg_tag};

use escapepod_classify::{
    AnchoredRead, ChargingBundle, Orientation, OrientationVotes, ScanOutcome, cl_from_probability,
    expected_levels_z, junction_features, junction_positions, resolve_orientation,
};
use escapepod_demux::GbmPredictor;

use crate::progress::create_spinner;
use crate::style;
use crate::util::resolve_pod5_inputs;

#[derive(Args)]
pub struct ClassifyArgs {
    /// Input POD5 file or directory
    #[arg(value_name = "POD5")]
    pub input: PathBuf,

    /// Aligned BAM with move tables (dorado --emit-moves, tags preserved
    /// through alignment)
    #[arg(short, long)]
    pub bam: PathBuf,

    /// Reference FASTA the BAM was aligned to; the CCA|adapter junction is
    /// located in every record
    #[arg(short, long)]
    pub reference: PathBuf,

    /// Model bundle directory (or its metadata.json)
    #[arg(short, long)]
    pub model: PathBuf,

    /// Output BAM: input records with `cl` (uint8, round(P(charged)·255))
    /// added to every record of each classified read
    #[arg(short, long)]
    pub output: PathBuf,

    /// Also write per-read calls as TSV (read_id, reference, p, cl)
    #[arg(long, value_name = "PATH")]
    pub tsv: Option<PathBuf>,

    /// Minimum mapping quality for a read to be classified
    #[arg(long, default_value = "1")]
    pub min_mapq: u8,

    /// Move-table signal frame: detect from the data (auto, requires >= 50
    /// informative reads and a 95% consensus) or force for small batches
    #[arg(long, default_value = "auto", value_parser = parse_orientation)]
    pub orientation: OrientationArg,

    /// Number of threads for parallel processing
    #[arg(short = 't', long, visible_short_alias = 'j', value_name = "N")]
    pub threads: Option<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OrientationArg {
    Auto,
    Time,
    Reversed,
}

fn parse_orientation(s: &str) -> Result<OrientationArg, String> {
    match s {
        "auto" => Ok(OrientationArg::Auto),
        "time" => Ok(OrientationArg::Time),
        "reversed" => Ok(OrientationArg::Reversed),
        _ => Err(format!(
            "unknown orientation '{}', expected auto|time|reversed",
            s
        )),
    }
}

/// Per-read signal lookup info from the POD5 index.
struct Pod5ReadInfo {
    reader_idx: usize,
    calibration_scale: f32,
    calibration_offset: f32,
    signal_rows: Vec<u64>,
}

pub fn run(args: ClassifyArgs) -> anyhow::Result<()> {
    // --- Bundle ---------------------------------------------------------
    let bundle = ChargingBundle::load(&args.model)?;
    info!(
        "model {}{}: {} features over offsets {}..{}, classes [{}, {}]",
        bundle.model_id,
        bundle
            .model_version
            .as_deref()
            .map(|v| format!(" v{v}"))
            .unwrap_or_default(),
        bundle.columns.len(),
        bundle.offsets.first().copied().unwrap_or(0),
        bundle.offsets.last().copied().unwrap_or(0),
        bundle.classes[0],
        bundle.classes[1],
    );
    match &bundle.operating_point {
        Some(op) => info!(
            "recommended operating point: P({}) >= {:.4} (cl >= {}){}",
            bundle.classes[1],
            op.probability,
            op.cl.unwrap_or_else(|| cl_from_probability(op.probability)),
            op.source
                .as_deref()
                .map(|s| format!(" — {s}"))
                .unwrap_or_default(),
        ),
        None => warn!(
            "bundle carries no operating point; downstream thresholds are the \
             caller's responsibility (do not assume the legacy 200)"
        ),
    }

    // --- Reference geometry ----------------------------------------------
    let geometry = junction_positions(
        &args.reference,
        &bundle.anchor.motif,
        bundle.anchor.motif_offset,
        &bundle.anchor.common_arm,
    )?;
    info!(
        "junction located in {} reference records ({} + {} at motif offset {})",
        geometry.len(),
        bundle.anchor.motif,
        bundle.anchor.common_arm,
        bundle.anchor.motif_offset,
    );

    // --- Pass 1: scan the BAM, anchor reads, vote on orientation ---------
    let spinner = create_spinner("scanning BAM")?;
    let file = std::fs::File::open(&args.bam)
        .with_context(|| format!("cannot open BAM {}", args.bam.display()))?;
    let decoder = bgzf::io::MultithreadedReader::new(file);
    let mut reader = bam::io::Reader::from(decoder);
    let header = reader.read_header()?;
    let ref_names: Vec<String> = header
        .reference_sequences()
        .keys()
        .map(|k| k.to_string())
        .collect();

    let mut votes = OrientationVotes::default();
    let mut anchored: HashMap<uuid::Uuid, AnchoredRead> = HashMap::new();
    let mut skips: HashMap<&'static str, u64> = HashMap::new();
    let mut total: u64 = 0;
    let mut record = RecordBuf::default();
    loop {
        if reader.read_record_buf(&header, &mut record)? == 0 {
            break;
        }
        total += 1;
        let Some(ref_name) = record
            .reference_sequence_id()
            .and_then(|id| ref_names.get(id))
        else {
            *skips.entry("unmapped/filtered").or_default() += 1;
            continue;
        };
        match escapepod_classify::anchor::scan_record(
            &record,
            ref_name,
            &geometry,
            &bundle.offsets,
            args.min_mapq,
        ) {
            ScanOutcome::Anchored(read) => {
                votes.add(&read);
                // One record per read, best alignment wins.
                match anchored.entry(read.read_id) {
                    std::collections::hash_map::Entry::Occupied(mut e) => {
                        if read.mapq > e.get().mapq {
                            e.insert(*read);
                        }
                    }
                    std::collections::hash_map::Entry::Vacant(e) => {
                        e.insert(*read);
                    }
                }
            }
            ScanOutcome::Skip(reason) => {
                use escapepod_classify::anchor::SkipReason::*;
                let key = match reason {
                    Filtered => "unmapped/filtered",
                    LowMapq => "low mapq",
                    NoGeometry => "reference without junction",
                    NoTags => "missing mv/ns tags",
                    Unanchored => "junction not aligned",
                    QueryOutOfRange => "query outside move table",
                    BadName => "non-UUID read name",
                };
                *skips.entry(key).or_default() += 1;
            }
        }
    }
    spinner.finish_with_message(format!(
        "{} BAM records scanned, {} reads anchored",
        style::count(total as usize),
        style::count(anchored.len())
    ));
    info!(
        "{} records scanned; {} unique anchored reads",
        total,
        anchored.len()
    );
    for (reason, n) in &skips {
        info!("  skipped ({}): {}", reason, n);
    }
    if anchored.is_empty() {
        bail!("no reads could be anchored; nothing to classify");
    }

    // --- Orientation ------------------------------------------------------
    let orientation = match args.orientation {
        OrientationArg::Time => Orientation::Time,
        OrientationArg::Reversed => Orientation::Reversed,
        OrientationArg::Auto => resolve_orientation(&votes, 50)?,
    };
    info!(
        "move-table frame: {} (votes: time={}, reversed={}{})",
        match orientation {
            Orientation::Time => "time-ordered",
            Orientation::Reversed => "reversed",
        },
        votes.time,
        votes.reversed,
        if args.orientation == OrientationArg::Auto {
            ""
        } else {
            "; forced by --orientation"
        },
    );

    // --- POD5 index -------------------------------------------------------
    let pod5_files = resolve_pod5_inputs(&args.input)?;
    let mut pod5_reads: HashMap<uuid::Uuid, Pod5ReadInfo> = HashMap::new();
    let mut pod5_readers: Vec<escapepod_signal::Reader> = Vec::new();
    for (reader_idx, path) in pod5_files.iter().enumerate() {
        let reader = escapepod_signal::Reader::open(path)?;
        for batch_result in reader.read_batches()? {
            let batch = batch_result?;
            let view = escapepod_signal::ReadsBatchView::new(&batch, false)?;
            for row in 0..view.num_rows() {
                let read = view.read(row)?;
                if anchored.contains_key(&read.read_id) {
                    pod5_reads.insert(
                        read.read_id,
                        Pod5ReadInfo {
                            reader_idx,
                            calibration_scale: read.calibration_scale,
                            calibration_offset: read.calibration_offset,
                            signal_rows: read.signal_rows,
                        },
                    );
                }
            }
        }
        pod5_readers.push(reader);
    }
    info!(
        "{} of {} anchored reads have signal in {} POD5 file(s)",
        pod5_reads.len(),
        anchored.len(),
        pod5_files.len()
    );
    let signal_extractors: Vec<_> = pod5_readers
        .iter()
        .map(|r| r.signal_extractor())
        .collect::<escapepod_signal::Result<_>>()?;

    // --- Features + prediction -------------------------------------------
    let predictor = GbmPredictor::new(&bundle.gbm);
    let reads: Vec<&AnchoredRead> = anchored.values().collect();
    let mut ns_mismatch = 0u64;
    let mut no_signal = 0u64;
    let results: Vec<Option<(uuid::Uuid, String, f64)>> = reads
        .par_iter()
        .map(|read| {
            let info = pod5_reads.get(&read.read_id)?;
            let raw = signal_extractors[info.reader_idx]
                .get_signal(&info.signal_rows)
                .ok()?;
            if raw.len() as i64 != read.ns {
                // The move-table frame is defined over `ns` samples; a
                // different signal length (e.g. a split read) would put
                // every span in the wrong place.
                return Some((read.read_id, String::new(), f64::NAN));
            }
            let sig_pa: Vec<f32> = raw
                .iter()
                .map(|&adc| (adc as f32 + info.calibration_offset) * info.calibration_scale)
                .collect();
            let coords = escapepod_classify::anchor::finalize(read, orientation, &bundle.offsets);
            let expected = bundle.kmer.as_ref().map(|k| {
                expected_levels_z(&read.seq, &k.map, k.k, k.center_idx, &read.qf, read.nb)
            });
            let grid = junction_features(&sig_pa, &coords, expected.as_deref());
            let features = bundle.select_columns(&grid);
            let (probs, _) = predictor.predict(&features).ok()?;
            Some((read.read_id, read.reference.clone(), probs[1]))
        })
        .collect();

    let mut calls: HashMap<uuid::Uuid, (String, f64, u8)> = HashMap::new();
    for r in results {
        match r {
            None => no_signal += 1,
            Some((_, _, p)) if p.is_nan() => ns_mismatch += 1,
            Some((id, reference, p)) => {
                calls.insert(id, (reference, p, cl_from_probability(p)));
            }
        }
    }
    if no_signal > 0 {
        warn!(
            "{} anchored reads had no fetchable signal (dorado read splitting \
             mints child ids absent from the POD5; see --disable-read-splitting)",
            no_signal
        );
    }
    if ns_mismatch > 0 {
        warn!(
            "{} reads skipped: signal length != ns tag (split or trimmed reads)",
            ns_mismatch
        );
    }
    if calls.is_empty() {
        bail!("no reads could be classified");
    }

    let mut ps: Vec<f64> = calls.values().map(|(_, p, _)| *p).collect();
    ps.sort_unstable_by(|a, b| a.total_cmp(b));
    let median_p = ps[ps.len() / 2];
    info!(
        "{} reads classified; median P({}) = {:.3}",
        calls.len(),
        bundle.classes[1],
        median_p
    );
    if let Some(op) = &bundle.operating_point {
        let n_pos = ps.iter().filter(|&&p| p >= op.probability).count();
        info!(
            "{} / {} reads ({:.1}%) at or above the bundle operating point ({:.4})",
            n_pos,
            ps.len(),
            100.0 * n_pos as f64 / ps.len() as f64,
            op.probability
        );
    }

    // --- TSV --------------------------------------------------------------
    if let Some(tsv_path) = &args.tsv {
        let mut w = std::io::BufWriter::new(std::fs::File::create(tsv_path)?);
        writeln!(w, "read_id\treference\tp_{}\tcl", bundle.classes[1])?;
        let mut rows: Vec<_> = calls.iter().collect();
        rows.sort_by_key(|(id, _)| *id);
        for (id, (reference, p, cl)) in rows {
            writeln!(w, "{}\t{}\t{:.6}\t{}", id, reference, p, cl)?;
        }
        info!("wrote {} calls to {}", calls.len(), tsv_path.display());
    }

    // --- Pass 2: write the BAM with `cl` ----------------------------------
    let file = std::fs::File::open(&args.bam)?;
    let decoder = bgzf::io::MultithreadedReader::new(file);
    let mut reader = bam::io::Reader::from(decoder);
    let mut out_header = reader.read_header()?;
    let pg = Map::<Program>::builder()
        .insert(pg_tag::NAME, "escpod")
        .insert(pg_tag::VERSION, env!("CARGO_PKG_VERSION"))
        .insert(
            pg_tag::COMMAND_LINE,
            format!(
                "escpod classify --model {} (cl = round(P({}) * 255))",
                bundle.model_id, bundle.classes[1]
            ),
        )
        .build()?;
    out_header.programs_mut().add("escpod-classify", pg)?;

    let out_file = std::fs::File::create(&args.output)?;
    let encoder = bgzf::io::MultithreadedWriter::new(out_file);
    let mut writer = bam::io::Writer::from(encoder);
    writer.write_header(&out_header)?;

    let cl_tag = Tag::new(b'c', b'l');
    let mut tagged: u64 = 0;
    let mut record = RecordBuf::default();
    loop {
        if reader.read_record_buf(&out_header, &mut record)? == 0 {
            break;
        }
        let call = record
            .name()
            .and_then(|n| std::str::from_utf8(n.as_ref()).ok())
            .and_then(|s| escapepod_signal::parse_uuid_flexible(s).ok())
            .and_then(|id| calls.get(&id));
        if let Some((_, _, cl)) = call {
            record.data_mut().insert(cl_tag, Value::UInt8(*cl));
            tagged += 1;
        }
        {
            use sam::alignment::io::Write as _;
            writer.write_alignment_record(&out_header, &record)?;
        }
    }
    writer.into_inner().finish()?;
    info!(
        "wrote {}: {} records, {} tagged with cl",
        args.output.display(),
        total,
        tagged
    );

    Ok(())
}
