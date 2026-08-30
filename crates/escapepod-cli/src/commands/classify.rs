//! `escpod classify` — tRNA charging (aminoacylation) classification from
//! POD5 + aligned BAM, writing the call as a `cl` tag on the BAM directly.
//! (`escpod signal classify` remains as a hidden deprecated alias; see
//! [`crate::commands::signal`].)
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
//! `escapepod_classify::bundle` for the contract. The scan → index →
//! classify orchestration lives in `escapepod_classify::pipeline` (shared
//! with the golden-parity tests); this command adds flags, progress,
//! reporting, and the output BAM/TSV writing.

use anyhow::bail;
use clap::Args;
use std::collections::{HashMap, HashSet};
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

use escapepod_classify::anchor::SkipReason;
use escapepod_classify::{
    ChargingBundle, Orientation, Pod5Index, cl_from_probability, classify_reads,
    junction_positions, resolve_orientation, scan_bam,
};

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

fn skip_label(reason: SkipReason) -> &'static str {
    match reason {
        SkipReason::Filtered => "unmapped/filtered",
        SkipReason::LowMapq => "low mapq",
        SkipReason::NoGeometry => "reference without junction",
        SkipReason::NoTags => "missing mv/ns tags",
        SkipReason::Unanchored => "junction not aligned",
        SkipReason::QueryOutOfRange => "query outside move table",
        SkipReason::BadName => "non-UUID read name",
    }
}

pub fn run(args: ClassifyArgs) -> anyhow::Result<()> {
    // --- Bundle ---------------------------------------------------------
    let bundle = ChargingBundle::load(&args.model)?;
    info!(
        "model {}{} [{}]: {} features over offsets {}..{}, classes [{}, {}]",
        bundle.model_id,
        bundle
            .model_version
            .as_deref()
            .map(|v| format!(" v{v}"))
            .unwrap_or_default(),
        // Which of the two scorers the directory holds. Both read the same
        // features, so nothing else in this line distinguishes them.
        bundle.scorer.kind(),
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
    match &bundle.abstain {
        Some(ab) => info!("abstain rule: {} — those reads get no call", ab.rule),
        // Not a defect, but worth saying: every bundle escapepod-models ships
        // carries one, so its absence usually means an old or hand-made bundle.
        None => info!("bundle declares no abstain rule; every anchored read is scored"),
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
    let scan = scan_bam(&args.bam, &geometry, &bundle.offsets, args.min_mapq)?;
    spinner.finish_with_message(format!(
        "{} BAM records scanned, {} reads anchored",
        style::count(scan.records_scanned as usize),
        style::count(scan.anchored.len())
    ));
    info!(
        "{} records scanned; {} unique anchored reads",
        scan.records_scanned,
        scan.anchored.len()
    );
    for (reason, n) in &scan.skips {
        info!("  skipped ({}): {}", skip_label(*reason), n);
    }
    if scan.anchored.is_empty() {
        bail!("no reads could be anchored; nothing to classify");
    }

    // --- Orientation ------------------------------------------------------
    let orientation = match args.orientation {
        OrientationArg::Time => Orientation::Time,
        OrientationArg::Reversed => Orientation::Reversed,
        OrientationArg::Auto => resolve_orientation(&scan.votes, 50)?,
    };
    info!(
        "move-table frame: {} (votes: time={}, reversed={}{})",
        match orientation {
            Orientation::Time => "time-ordered",
            Orientation::Reversed => "reversed",
        },
        scan.votes.time,
        scan.votes.reversed,
        if args.orientation == OrientationArg::Auto {
            ""
        } else {
            "; forced by --orientation"
        },
    );

    // --- POD5 index + classification --------------------------------------
    let pod5_files = resolve_pod5_inputs(&args.input)?;
    let wanted: HashSet<uuid::Uuid> = scan.anchored.keys().copied().collect();
    let pod5 = Pod5Index::build(&pod5_files, &wanted)?;
    info!(
        "{} of {} anchored reads have signal in {} POD5 file(s)",
        pod5.reads().len(),
        scan.anchored.len(),
        pod5.n_files()
    );

    let (calls, stats) = classify_reads(&bundle, &scan.anchored, &pod5, orientation)?;
    if stats.no_signal > 0 {
        warn!(
            "{} anchored reads had no fetchable signal (dorado read splitting \
             mints child ids absent from the POD5; see --disable-read-splitting)",
            stats.no_signal
        );
    }
    if stats.ns_mismatch > 0 {
        warn!(
            "{} reads skipped: signal length != ns tag (split or trimmed reads)",
            stats.ns_mismatch
        );
    }
    // The no-call rate is reported at info level rather than buried, because
    // the bundle asks for it: arm resolvability is correlated with charging,
    // so a charging fraction over called reads alone is biased low, and the
    // reader cannot correct for a number they never saw.
    if let Some(ab) = &bundle.abstain {
        let total = stats.abstained + calls.len() as u64;
        info!(
            "{} of {} scoreable reads ({:.1}%) were no-called by the bundle's \
             abstain rule ({}); report this rate beside any charging fraction",
            stats.abstained,
            total,
            100.0 * stats.abstained as f64 / total.max(1) as f64,
            ab.rule,
        );
    }
    if calls.is_empty() {
        bail!("no reads could be classified");
    }

    let mut ps: Vec<f64> = calls.iter().map(|c| c.p).collect();
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
    //
    // Every anchored read gets a row: a probability, or an empty one and the
    // reason it has none. A read that simply vanishes from the output is a
    // drop nobody can chase later — the failure
    // `rnabioco/aa-tRNA-seq-pipeline#110` had to reverse-engineer from the gap
    // between two QC rows, since remora reports no per-read reason. The
    // `reason` column is empty for a call, so the file still reads as
    // `read_id, reference, p, cl` for anything that only wants calls.
    if let Some(tsv_path) = &args.tsv {
        let mut w = std::io::BufWriter::new(std::fs::File::create(tsv_path)?);
        writeln!(w, "read_id\treference\tp_{}\tcl\treason", bundle.classes[1])?;
        for c in &calls {
            writeln!(w, "{}\t{}\t{:.6}\t{}\t", c.read_id, c.reference, c.p, c.cl)?;
        }
        for n in &stats.no_calls {
            writeln!(
                w,
                "{}\t{}\t\t\t{}",
                n.read_id,
                n.reference,
                n.reason.as_str()
            )?;
        }
        info!(
            "wrote {} calls + {} no-calls to {}",
            calls.len(),
            stats.no_calls.len(),
            tsv_path.display()
        );
    }

    // --- Pass 2: write the BAM with `cl` ----------------------------------
    let cl_by_id: HashMap<uuid::Uuid, u8> = calls.iter().map(|c| (c.read_id, c.cl)).collect();

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
        let cl = record
            .name()
            .and_then(|n| std::str::from_utf8(n.as_ref()).ok())
            .and_then(|s| escapepod_signal::parse_uuid_flexible(s).ok())
            .and_then(|id| cl_by_id.get(&id));
        if let Some(&cl) = cl {
            record.data_mut().insert(cl_tag, Value::UInt8(cl));
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
        scan.records_scanned,
        tagged
    );

    Ok(())
}
