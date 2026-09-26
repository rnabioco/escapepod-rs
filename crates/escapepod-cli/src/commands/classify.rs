//! `escpod classify` — tRNA charging (aminoacylation) classification from
//! POD5 + aligned BAM, writing the call as a `cl` tag on the BAM directly.
//! (`escpod signal classify`, the deprecated alias this command was briefly
//! spelled as between 0.11.0 and 0.18.1, was removed in favor of this
//! top-level spelling.)
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
use escapepod_classify::pipeline::{ClassifyStats, ReadCall};
use escapepod_classify::{
    ChargingBundle, Orientation, Pod5Index, cl_from_probability, classify_reads,
    junction_positions, resolve_orientation, scan_bam, waveform,
};

use crate::device::Device;
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

    /// Where the windowed (`waveform_model`) variant's TCN inference runs
    /// (`auto` by default, which prefers the GPU here at production batch
    /// sizes). Has no effect on the GBM / feature-network variants, which
    /// have no GPU path — see [`crate::device::note_cpu_only`].
    #[command(flatten)]
    pub device: crate::device::DeviceArgs,
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

/// One line describing what the loaded bundle's model reads.
///
/// The two input spaces have nothing in common to summarise — one is a column
/// vector over base offsets, the other three tensors over a signal window — so
/// this says which, rather than forcing both into one sentence's fields.
fn input_summary(bundle: &ChargingBundle) -> String {
    if let Some(w) = &bundle.waveform {
        let sig = w.tensor_shape(escapepod_classify::WaveformTensor::Signal);
        let seq = w.tensor_shape(escapepod_classify::WaveformTensor::Sequence);
        let feat = w.tensor_shape(escapepod_classify::WaveformTensor::Features);
        format!(
            "a [{}, {}] signal window + [{}, {}] sequence + [{}, {}] per-base features \
             ({} offsets from the anchor{})",
            sig[0],
            sig[1],
            seq[0],
            seq[1],
            feat[0],
            feat[1],
            feat[1],
            if w.refine.is_some() {
                ", map refined"
            } else {
                ""
            },
        )
    } else if let Some(f) = &bundle.features {
        format!(
            "{} features over offsets {}..{}",
            f.columns.len(),
            f.offsets.first().copied().unwrap_or(0),
            f.offsets.last().copied().unwrap_or(0),
        )
    } else {
        "an input space this build cannot describe".to_string()
    }
}

/// The windowed variant's scan → index → classify, in place of the column
/// path's. Returns what [`finish`] reports on.
///
/// `gpu` is a placement decision already made by [`run`] (via
/// [`crate::device::place_and_report`]) — this function only acts on it.
fn run_waveform(
    args: &ClassifyArgs,
    bundle: &ChargingBundle,
    geometry: &HashMap<String, escapepod_classify::RefGeometry>,
    gpu: bool,
    #[cfg_attr(not(feature = "gpu"), allow(unused_variables))] device: Device,
) -> anyhow::Result<(Vec<ReadCall>, ClassifyStats, u64, DeviceProvenance)> {
    if args.orientation != OrientationArg::Auto {
        // Not silently ignored: the flag exists to override a *vote*, and this
        // variant does not vote — its frame is the one the model was trained
        // in, declared in the bundle. Honouring the flag would mirror every
        // window away from the model.
        warn!(
            "--orientation is ignored for this bundle: the windowed variant takes its \
             signal frame from the model's own `reverse_signal`, not from a vote"
        );
    }
    let spinner = create_spinner("scanning BAM")?;
    let scan = waveform::scan_bam(
        &args.bam,
        geometry,
        bundle.anchor.motif_offset,
        args.min_mapq,
    )?;
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

    let pod5_files = resolve_pod5_inputs(&args.input)?;
    let wanted: HashSet<uuid::Uuid> = scan.anchored.keys().copied().collect();
    let pod5 = Pod5Index::build(&pod5_files, &wanted)?;
    info!(
        "{} of {} anchored reads have signal in {} POD5 file(s)",
        pod5.reads().len(),
        scan.anchored.len(),
        pod5.n_files()
    );

    let (calls, stats, device_prov) = if gpu {
        #[cfg(feature = "gpu")]
        {
            // Both the load (cuBLAS pairing) and the run (per-batch GPU/CPU
            // parity on real reads) can refuse the GPU with `GpuRefused`
            // (#416). `--device gpu` makes that an error; `auto` only allowed
            // the GPU, so it falls back to the CPU scorer for the whole run.
            // Nested rather than chained with `.and_then`/`.map`: `pairing`
            // has to survive a `classify_reads_gpu` failure too (the
            // fallback arm below), which a `Result`-flattening chain would
            // have dropped along with the error (#425).
            match bundle.waveform_net_gpu(waveform_gpu_batch()) {
                Ok(net) => {
                    let pairing = net.cublas_pairing().clone();
                    match waveform::classify_reads_gpu(bundle, &scan.anchored, &pod5, &net) {
                        // `groups_scored == 0`: every read landed in the
                        // CPU-fallback tail and `gpu.logits()` was never
                        // actually called (the run was smaller than one GPU
                        // batch) — report the device that actually scored
                        // these reads, not the one that merely loaded (#425).
                        Ok((calls, stats, parity)) if parity.groups_scored > 0 => {
                            (calls, stats, DeviceProvenance::gpu(&pairing, parity))
                        }
                        Ok((calls, stats, _parity)) => (calls, stats, DeviceProvenance::cpu()),
                        Err(e)
                            if device != Device::Gpu
                                && e
                                    .downcast_ref::<escapepod_classify::waveform_net_gpu::GpuRefused>()
                                    .is_some() =>
                        {
                            warn!(
                                "{e:#}; --device {device}: scoring every read on the CPU instead"
                            );
                            let (calls, stats) =
                                waveform::classify_reads(bundle, &scan.anchored, &pod5)?;
                            // The pairing loaded and agreed fine — a
                            // real-batch parity divergence is what triggered
                            // this fallback, not a bad pairing — so it is
                            // still worth recording rather than discarded: a
                            // #416-class event should leave more evidence
                            // behind, not less.
                            (calls, stats, DeviceProvenance::cpu_gpu_refused(&pairing))
                        }
                        Err(e) => return Err(e),
                    }
                }
                Err(e)
                    if device != Device::Gpu
                        && e.downcast_ref::<escapepod_classify::waveform_net_gpu::GpuRefused>()
                            .is_some() =>
                {
                    // Refused before a pairing was even resolved (the
                    // load-time check itself failed) — nothing to attach.
                    warn!("{e:#}; --device {device}: scoring every read on the CPU instead");
                    let (calls, stats) = waveform::classify_reads(bundle, &scan.anchored, &pod5)?;
                    (calls, stats, DeviceProvenance::cpu())
                }
                Err(e) => return Err(e),
            }
        }
        #[cfg(not(feature = "gpu"))]
        {
            unreachable!("place_and_report only returns Placement::Gpu when Stage::compiled_in()")
        }
    } else {
        let (calls, stats) = waveform::classify_reads(bundle, &scan.anchored, &pod5)?;
        (calls, stats, DeviceProvenance::cpu())
    };
    Ok((calls, stats, scan.records_scanned, device_prov))
}

/// The GPU scorer's fixed batch size — a hardware-tuning knob, not a routine
/// flag, so it is an env var like `ESCAPEPOD_LSTM_BACKEND` rather than a
/// `--batch` option. 128 is the default: `examples/tcn_cuda_probe.rs`'s sweep
/// plateaus by there (11.1x at 128 vs 11.7x at 256), and a smaller compiled
/// batch means a cheaper worst-case CPU-remainder fallback (see
/// `waveform::classify_reads_gpu`) and a smaller fixed GPU memory reservation.
#[cfg(feature = "gpu")]
fn waveform_gpu_batch() -> usize {
    escapepod_signal::pod5::env::positive_usize("ESCAPEPOD_WAVEFORM_GPU_BATCH").unwrap_or(128)
}

/// Replace any byte the SAM header-value grammar refuses (§ 1.3, `[ -~]+` —
/// noodles-sam's own `is_valid_value`, which every `@PG` field including
/// `DS` is checked against at write time) with `?`.
///
/// Every other field `provenance_ds` writes comes from the bundle's own
/// schema-controlled `metadata.json` — plain ASCII identifiers by
/// construction. `DeviceProvenance`'s cuBLAS fields are the one exception:
/// they can carry [`escapepod_classify::cuda_libs::release_of`]'s
/// unversioned fallback, which embeds a filesystem path this crate does not
/// control. Without this, an unusual byte in that path would fail
/// `writer.write_header` at the very end of [`finish`] — after the BAM scan,
/// POD5 indexing and the whole classification pass have already run — rather
/// than degrade the one field that can't be trusted.
#[cfg(feature = "gpu")]
fn sam_header_safe(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii() && (' '..='~').contains(&c) {
                c
            } else {
                '?'
            }
        })
        .collect()
}

/// Which device actually scored a run, and — when the GPU was involved at
/// all — the cuBLAS/cuBLASLt pairing and GPU/CPU parity evidence #425
/// attaches to the output BAM. #416 showed a mismatched pairing returns
/// wrong GPU probabilities with no error, so "GPU or CPU", "which physical
/// libraries were paired", and "was the running parity check clean" are
/// exactly the facts an after-the-fact audit needs and, before this, could
/// only ever recover from a caller's own log capture (if it kept one) or
/// Slurm accounting (if the run is still recent enough).
///
/// `requested` is always the *actual* device, never the flag: a GPU run
/// smaller than one GPU batch (every read scored by
/// [`waveform::classify_reads_gpu`]'s own CPU-fallback tail,
/// `GpuParitySummary::groups_scored == 0`) or one that fell back after a
/// [`escapepod_classify::waveform_net_gpu::GpuRefused`] (`auto` only)
/// reports `cpu` here, because that is what scored the reads — even though a
/// GPU scorer loaded successfully in both cases. The `cublas_*` fields are
/// independent of that: a pairing that loaded and was then made moot by a
/// real-batch parity refusal is still worth keeping ([`Self::cpu_gpu_refused`]),
/// since a #416-class event is exactly what this record exists to catch.
struct DeviceProvenance {
    requested: &'static str,
    cublas_path: Option<String>,
    cublas_version: Option<String>,
    cublaslt_path: Option<String>,
    cublaslt_version: Option<String>,
    cublas_repaired: Option<bool>,
    gpu_batches_scored: Option<usize>,
    parity_checked_batches: Option<usize>,
    parity_worst_abs_dp: Option<f64>,
}

impl DeviceProvenance {
    fn cpu() -> Self {
        Self {
            requested: "cpu",
            cublas_path: None,
            cublas_version: None,
            cublaslt_path: None,
            cublaslt_version: None,
            cublas_repaired: None,
            gpu_batches_scored: None,
            parity_checked_batches: None,
            parity_worst_abs_dp: None,
        }
    }

    #[cfg(feature = "gpu")]
    fn gpu(
        pairing: &escapepod_classify::cuda_libs::CublasPairing,
        parity: waveform::GpuParitySummary,
    ) -> Self {
        Self {
            requested: "gpu",
            // The denominator `parity_checked_batches` is a fraction of:
            // only 1 in `ESCAPEPOD_WAVEFORM_GPU_PARITY_EVERY` real GPU
            // batches (default 64) is ever cross-checked against the CPU
            // scorer, so `parity_checked_batches` alone cannot say how much
            // of the run that coverage represents.
            gpu_batches_scored: Some(parity.groups_scored),
            parity_checked_batches: Some(parity.checked_batches),
            parity_worst_abs_dp: Some(parity.worst_abs_dp),
            ..Self::pairing_only(pairing)
        }
    }

    /// The GPU scorer loaded and its cuBLAS/cuBLASLt pairing resolved fine,
    /// but a real-batch parity check caught a divergence mid-run (a
    /// #416-class event, not a load-time pairing failure) and the whole run
    /// was rescored on the CPU — `requested` is `cpu` (that is what actually
    /// scored these reads), but the pairing that was resolved is kept rather
    /// than silently discarded, and no parity fields are set (the check that
    /// caught the divergence is what ended the GPU run, not a summary of it).
    #[cfg(feature = "gpu")]
    fn cpu_gpu_refused(pairing: &escapepod_classify::cuda_libs::CublasPairing) -> Self {
        Self {
            requested: "cpu",
            ..Self::pairing_only(pairing)
        }
    }

    #[cfg(feature = "gpu")]
    fn pairing_only(pairing: &escapepod_classify::cuda_libs::CublasPairing) -> Self {
        Self {
            requested: "cpu", // overwritten by every caller of this helper
            // `sam_header_safe`, not a plain `.clone()`: `cublas_version`/
            // `lt_version` fall back to `release_of`'s `"(unversioned, in
            // {path})"` when a loaded library's file name carries no
            // parseable version, embedding a filesystem path this crate
            // never controls. The rest of `CublasPairing` is a diagnostic
            // type with no such constraint (its `Display` is fine for
            // `tracing::debug!`) — only the copies that reach a SAM header
            // value need to hold to that grammar.
            cublas_path: Some(sam_header_safe(&pairing.cublas_path.display().to_string())),
            cublas_version: Some(sam_header_safe(&pairing.cublas_version)),
            cublaslt_path: Some(sam_header_safe(&pairing.lt_path.display().to_string())),
            cublaslt_version: Some(sam_header_safe(&pairing.lt_version)),
            cublas_repaired: Some(pairing.repaired),
            gpu_batches_scored: None,
            parity_checked_batches: None,
            parity_worst_abs_dp: None,
        }
    }
}

/// The `@PG` record's `DS` field: the bundle identity that is already
/// `info!`-logged in [`run`] but otherwise dropped once the run's stdout is
/// gone — the field the same run's aggregate charged fraction has nothing to
/// show it was scored against `bundle.model_id` at all (#370). Attached to
/// the output BAM itself rather than a sidecar, so provenance survives
/// whatever survives the BAM.
///
/// Fields the bundle does not carry are omitted, not null: a bundle with no
/// `basecaller`/`operating_point`/`abstain` block must not have one
/// fabricated for it just to fill a slot. Same rule for `device`'s
/// GPU-only fields (#425): omitted entirely under `requested == "cpu"`.
fn provenance_ds(bundle: &ChargingBundle, device: &DeviceProvenance) -> String {
    let mut ds = serde_json::Map::new();
    ds.insert(
        "model_id".to_string(),
        serde_json::Value::String(bundle.model_id.clone()),
    );
    if let Some(v) = &bundle.model_version {
        ds.insert(
            "model_version".to_string(),
            serde_json::Value::String(v.clone()),
        );
    }
    ds.insert(
        "scorer_sha256".to_string(),
        serde_json::Value::String(bundle.scorer_sha256.clone()),
    );
    // The class `cl`/`operating_point.probability` are stated on — the old
    // hand-templated `@PG` `CL` string used to be the only place this was
    // recorded, and #425 replaces that string with the real argv, so this
    // is where it now lives instead of being dropped entirely.
    ds.insert(
        "positive_class".to_string(),
        serde_json::Value::String(bundle.classes[1].clone()),
    );
    if let Some(bc) = &bundle.basecaller {
        let mut bc_obj = serde_json::Map::new();
        bc_obj.insert(
            "model".to_string(),
            serde_json::Value::String(bc.model.clone()),
        );
        if let Some(dv) = &bc.dorado_version {
            bc_obj.insert(
                "dorado_version".to_string(),
                serde_json::Value::String(dv.clone()),
            );
        }
        if let Some(sha) = &bc.model_sha256 {
            bc_obj.insert(
                "model_sha256".to_string(),
                serde_json::Value::String(sha.clone()),
            );
        }
        ds.insert("basecaller".to_string(), serde_json::Value::Object(bc_obj));
    }
    if let Some(op) = &bundle.operating_point {
        let mut op_obj = serde_json::Map::new();
        op_obj.insert(
            "probability".to_string(),
            serde_json::Value::from(op.probability),
        );
        if let Some(cl) = op.cl {
            op_obj.insert("cl".to_string(), serde_json::Value::from(cl));
        }
        ds.insert(
            "operating_point".to_string(),
            serde_json::Value::Object(op_obj),
        );
    }
    ds.insert(
        "calibration".to_string(),
        serde_json::Value::Bool(bundle.calibration.is_some()),
    );
    if let Some(ab) = &bundle.abstain {
        ds.insert(
            "abstain_rule".to_string(),
            serde_json::Value::String(ab.rule.clone()),
        );
    }
    let mut dev = serde_json::Map::new();
    dev.insert(
        "requested".to_string(),
        serde_json::Value::String(device.requested.to_string()),
    );
    if let Some(v) = &device.cublas_path {
        dev.insert(
            "cublas_path".to_string(),
            serde_json::Value::String(v.clone()),
        );
    }
    if let Some(v) = &device.cublas_version {
        dev.insert(
            "cublas_version".to_string(),
            serde_json::Value::String(v.clone()),
        );
    }
    if let Some(v) = &device.cublaslt_path {
        dev.insert(
            "cublaslt_path".to_string(),
            serde_json::Value::String(v.clone()),
        );
    }
    if let Some(v) = &device.cublaslt_version {
        dev.insert(
            "cublaslt_version".to_string(),
            serde_json::Value::String(v.clone()),
        );
    }
    if let Some(v) = device.cublas_repaired {
        dev.insert("cublas_repaired".to_string(), serde_json::Value::Bool(v));
    }
    if let Some(v) = device.gpu_batches_scored {
        dev.insert("gpu_batches_scored".to_string(), serde_json::Value::from(v));
    }
    if let Some(v) = device.parity_checked_batches {
        dev.insert(
            "parity_checked_batches".to_string(),
            serde_json::Value::from(v),
        );
    }
    if let Some(v) = device.parity_worst_abs_dp {
        dev.insert(
            "parity_worst_abs_dp".to_string(),
            serde_json::Value::from(v),
        );
    }
    ds.insert("device".to_string(), serde_json::Value::Object(dev));
    serde_json::Value::Object(ds).to_string()
}

fn skip_label(reason: SkipReason) -> &'static str {
    match reason {
        SkipReason::Filtered => "unmapped/filtered",
        SkipReason::NoMdTag => "no MD tag (aligned reference unavailable)",
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
        "model {}{} [{}]: {}, classes [{}, {}]",
        bundle.model_id,
        bundle
            .model_version
            .as_deref()
            .map(|v| format!(" v{v}"))
            .unwrap_or_default(),
        // Which scorer the directory holds, and what it reads. The two input
        // spaces are different enough that one line cannot describe both.
        bundle.scorer.kind(),
        input_summary(&bundle),
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
    if bundle.calibration.is_some() {
        // Carried, never applied. Saying so matters because the operating
        // point above is stated on the *uncalibrated* probability the graph
        // emits, so silently calibrating would move the scale out from under
        // the very threshold printed beside it.
        info!(
            "bundle ships a Platt calibration of the raw logit; it is NOT applied — the \
             operating point above is stated on the uncalibrated probability"
        );
    }
    match &bundle.abstain {
        Some(ab) => info!("abstain rule: {} — those reads get no call", ab.rule),
        // Not a defect, but worth saying: every bundle escapepod-models ships
        // carries one, so its absence usually means an old or hand-made bundle.
        None => info!("bundle declares no abstain rule; every anchored read is scored"),
    }
    // Stated, never checked: escpod would have to read the BAM's `@PG` to know
    // what called it. Saying it out loud is nonetheless the whole point of the
    // field — a charging model substantially detects how the basecaller fails
    // at the adduct, so calling the same reads with another model flips ~3.9%
    // of calls while the aggregate charged fraction moves 0.04 pp, and the
    // 20260825 RLA QC scored v6 data with a v5.3.0-trained bundle with nothing
    // in the run to show it.
    match &bundle.basecaller {
        Some(bc) => info!(
            "bundle was trained on basecalls from {}{}; scoring reads called with \
             another model is a domain shift on the k-mer residual — escpod does not \
             check this",
            bc.model,
            bc.dorado_version
                .as_deref()
                .map(|v| format!(" (dorado {v})"))
                .unwrap_or_default(),
        ),
        None => info!(
            "bundle does not declare which basecaller called its training corpus; \
             comparability with the reads being scored cannot be checked"
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

    // Resolved once, early — placement and the CPU-cost warnings both hang
    // off it, and the point of the warnings is that they arrive before the
    // scan rather than after a run that already ran on the wrong device.
    let device = args.device.resolve();

    if bundle.waveform.is_some() {
        // GPU placement at batch >= 2 (`WaveformNetGpu::load` refuses batch
        // < 2 outright regardless of this call, so batch 1 is not reachable
        // here — see its module doc's "Batch 1 is refused outright" section
        // for the confirmed, reproducible bug that guards against). #343's
        // original real-chunk batch-2 divergence did not survive follow-up:
        // eleven runs against the exact checksummed fixture bundle (bit-for-
        // bit identical, zero flipped calls), then four more at the actual
        // production batch size (128) against an independent ~5,000-read
        // real dataset (rnabioco/2026-aars-in-vitro), also bit-for-bit
        // identical, zero flips. The "real feature magnitude breaks the
        // fused kernel" hypothesis behind the original report is refuted (a
        // 500x synthetic magnitude sweep at batch 2 shows no degradation),
        // and the "only this bundle's graph fuses into RmsNorm" claim is
        // refuted too (both bundles fuse identically). See
        // rnabioco/escapepod-rs#343 for the full writeup — including a
        // separate, rare (1-in-51 trials), non-deterministic anomaly found
        // at batch 1 on a different bundle during that investigation, never
        // reproduced at batch >= 2 in ~15 total runs across two independent
        // real datasets and dozens of synthetic ones. Root cause still
        // unknown; flagged there for anyone who hits it again.
        let placement = crate::device::place_and_report(device, crate::device::Stage::WaveformTcn)?;
        let (calls, stats, records, device_prov) =
            run_waveform(&args, &bundle, &geometry, placement.is_gpu(), device)?;
        return finish(&args, &bundle, calls, stats, records, device_prov);
    }
    crate::device::note_cpu_only(
        device,
        "GBM / feature-network classification",
        "the tree walk and the small per-base network have no GPU path and are not \
         expected to get one — the feature network is ~5 MFLOP/read and latency-bound, \
         so a GPU would add launch overhead rather than remove a bottleneck",
    );

    // --- Pass 1: scan the BAM, anchor reads, vote on orientation ---------
    let spinner = create_spinner("scanning BAM")?;
    let scan = scan_bam(
        &args.bam,
        &geometry,
        &bundle.feature_space()?.offsets,
        args.min_mapq,
    )?;
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
    // GBM / feature-network bundles have no GPU path at all (see
    // `note_cpu_only` above) — always CPU, never the `--device` flag's value.
    finish(
        &args,
        &bundle,
        calls,
        stats,
        scan.records_scanned,
        DeviceProvenance::cpu(),
    )
}

/// Report, write the TSV, and write the `cl`-tagged BAM.
///
/// Shared by both input spaces: what a bundle reads changes how a read is
/// scored, not what a call is or where it is written.
fn finish(
    args: &ClassifyArgs,
    bundle: &ChargingBundle,
    calls: Vec<ReadCall>,
    stats: ClassifyStats,
    records_scanned: u64,
    device: DeviceProvenance,
) -> anyhow::Result<()> {
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
    if stats.no_chunk > 0 {
        warn!(
            "{} reads yielded no window at the anchor (the alignment does not reach it, \
             or its map covers no signal)",
            stats.no_chunk
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

    // `MultithreadedReader::new` / `MultithreadedWriter::new` are ONE worker
    // each, whatever the name suggests. This pass is decode -> add one tag ->
    // re-encode, and the BGZF deflate of the output is 5.7% of the whole
    // command's CPU on one thread — so the writer gets the pool's width and the
    // reader, whose inflate is far cheaper, a quarter of it.
    let threads = rayon::current_num_threads().max(1);
    let reader_workers = std::num::NonZero::new(threads.div_ceil(4)).expect("at least one");
    let writer_workers = std::num::NonZero::new(threads).expect("at least one");

    let file = std::fs::File::open(&args.bam)?;
    let decoder = bgzf::io::MultithreadedReader::with_worker_count(reader_workers, file);
    let mut reader = bam::io::Reader::from(decoder);
    let mut out_header = reader.read_header()?;
    // The real invoked argv, mirroring `align`/`resquiggle` (#409) — the
    // `cl` scale note this used to stand in for belongs in the DS blob (it's
    // implicit in `operating_point.cl`/the docs), not in place of the actual
    // command line an audit would otherwise have to reconstruct. Named
    // `argv`, not `cl`: this function's `cl` is already the charging-call
    // byte (see the tagging loop below), and `cl_tag`/`cl_by_id` are already
    // that name's established meaning here.
    let argv: Vec<String> = std::env::args().collect();
    let pg = Map::<Program>::builder()
        .insert(pg_tag::NAME, "escpod")
        .insert(pg_tag::VERSION, env!("CARGO_PKG_VERSION"))
        .insert(pg_tag::COMMAND_LINE, argv.join(" "))
        .insert(pg_tag::DESCRIPTION, provenance_ds(bundle, &device))
        .build()?;
    out_header.programs_mut().add("escpod-classify", pg)?;

    let out_file = std::fs::File::create(&args.output)?;
    let encoder = bgzf::io::MultithreadedWriter::with_worker_count(writer_workers, out_file);
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
        records_scanned,
        tagged
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The fixture bundle `classify_e2e.rs`'s end-to-end tests already load
    /// from `../escapepod-classify/tests/fixtures/bundle` — reused here so
    /// this pins `provenance_ds`'s JSON shape without needing a GPU, a real
    /// `CublasPairing`, or a real `waveform::GpuParitySummary` run: `device`
    /// is plain data by the time `provenance_ds` sees it, so a run-level
    /// `DeviceProvenance` value is all either case needs (#425).
    fn fixture_bundle() -> ChargingBundle {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("escapepod-classify/tests/fixtures/bundle");
        ChargingBundle::load(&dir).unwrap()
    }

    const GPU_FIELDS: [&str; 5] = [
        "cublas_path",
        "cublas_version",
        "cublaslt_path",
        "cublaslt_version",
        "cublas_repaired",
    ];
    const PARITY_FIELDS: [&str; 3] = [
        "gpu_batches_scored",
        "parity_checked_batches",
        "parity_worst_abs_dp",
    ];

    #[test]
    fn provenance_ds_carries_the_positive_class() {
        let bundle = fixture_bundle();
        let ds = provenance_ds(&bundle, &DeviceProvenance::cpu());
        let v: serde_json::Value = serde_json::from_str(&ds).unwrap();
        assert_eq!(v["positive_class"], bundle.classes[1]);
    }

    #[test]
    fn provenance_ds_cpu_device_carries_no_gpu_fields() {
        let bundle = fixture_bundle();
        let ds = provenance_ds(&bundle, &DeviceProvenance::cpu());
        let v: serde_json::Value = serde_json::from_str(&ds).unwrap();
        assert_eq!(v["device"]["requested"], "cpu");
        for key in GPU_FIELDS.iter().chain(&PARITY_FIELDS) {
            assert!(
                v["device"].get(key).is_none(),
                "device.{key} must be omitted, not null, under --device cpu: {v}"
            );
        }
    }

    /// A GPU run's provenance, from a hand-built `DeviceProvenance` — the
    /// mocked `CublasPairing`/parity-summary input the acceptance criteria
    /// ask for. This does not need the `gpu` feature: `DeviceProvenance`
    /// itself is plain data, and only its constructors (which pull the real
    /// types in) are feature-gated.
    #[test]
    fn provenance_ds_gpu_device_carries_pairing_and_parity() {
        let bundle = fixture_bundle();
        let device = DeviceProvenance {
            requested: "gpu",
            cublas_path: Some("/opt/cuda-12.8/lib64/libcublas.so.12".to_string()),
            cublas_version: Some("12.8.93".to_string()),
            cublaslt_path: Some("/opt/cuda-12.8/lib64/libcublasLt.so.12".to_string()),
            cublaslt_version: Some("12.8.93".to_string()),
            cublas_repaired: Some(false),
            gpu_batches_scored: Some(400),
            parity_checked_batches: Some(12),
            parity_worst_abs_dp: Some(1.3e-4),
        };
        let ds = provenance_ds(&bundle, &device);
        let v: serde_json::Value = serde_json::from_str(&ds).unwrap();
        assert_eq!(v["device"]["requested"], "gpu");
        assert_eq!(
            v["device"]["cublas_path"],
            "/opt/cuda-12.8/lib64/libcublas.so.12"
        );
        assert_eq!(v["device"]["cublas_version"], "12.8.93");
        assert_eq!(
            v["device"]["cublaslt_path"],
            "/opt/cuda-12.8/lib64/libcublasLt.so.12"
        );
        assert_eq!(v["device"]["cublaslt_version"], "12.8.93");
        assert_eq!(v["device"]["cublas_repaired"], false);
        assert_eq!(v["device"]["gpu_batches_scored"], 400);
        assert_eq!(v["device"]["parity_checked_batches"], 12);
        assert_eq!(v["device"]["parity_worst_abs_dp"], 1.3e-4);
    }

    /// `DeviceProvenance::gpu`/`cpu_gpu_refused` themselves, against a real
    /// (if fictitious) `CublasPairing` — not just a hand-built
    /// `DeviceProvenance` — closing the gap the mocked-input test above
    /// leaves: a field renamed inside either constructor would still pass
    /// every test that only builds `DeviceProvenance` by hand.
    #[cfg(feature = "gpu")]
    #[test]
    fn device_provenance_constructors_map_pairing_fields_correctly() {
        use escapepod_classify::cuda_libs::CublasPairing;

        let pairing = CublasPairing {
            cublas_path: std::path::PathBuf::from("/opt/cuda-12.8/lib64/libcublas.so.12"),
            cublas_version: "12.8.93".to_string(),
            lt_path: std::path::PathBuf::from("/opt/cuda-12.8/lib64/libcublasLt.so.12"),
            lt_version: "12.8.93".to_string(),
            repaired: true,
        };
        let parity = waveform::GpuParitySummary {
            groups_scored: 40,
            checked_batches: 1,
            worst_abs_dp: 2.5e-5,
        };

        let gpu = DeviceProvenance::gpu(&pairing, parity);
        assert_eq!(gpu.requested, "gpu");
        assert_eq!(
            gpu.cublas_path.as_deref(),
            Some("/opt/cuda-12.8/lib64/libcublas.so.12")
        );
        assert_eq!(gpu.cublas_version.as_deref(), Some("12.8.93"));
        assert_eq!(
            gpu.cublaslt_path.as_deref(),
            Some("/opt/cuda-12.8/lib64/libcublasLt.so.12")
        );
        assert_eq!(gpu.cublaslt_version.as_deref(), Some("12.8.93"));
        assert_eq!(gpu.cublas_repaired, Some(true));
        assert_eq!(gpu.gpu_batches_scored, Some(40));
        assert_eq!(gpu.parity_checked_batches, Some(1));
        assert_eq!(gpu.parity_worst_abs_dp, Some(2.5e-5));

        let refused = DeviceProvenance::cpu_gpu_refused(&pairing);
        assert_eq!(refused.requested, "cpu");
        assert_eq!(refused.cublas_version.as_deref(), Some("12.8.93"));
        assert_eq!(refused.gpu_batches_scored, None);
        assert_eq!(refused.parity_checked_batches, None);
        assert_eq!(refused.parity_worst_abs_dp, None);
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn sam_header_safe_replaces_bytes_the_sam_grammar_refuses() {
        assert_eq!(sam_header_safe("12.8.93"), "12.8.93");
        assert_eq!(
            sam_header_safe("(unversioned, in /opt/lib)"),
            "(unversioned, in /opt/lib)"
        );
        // Tab, newline and DEL (0x7F) are all outside SAM's `[ -~]+` — one
        // rogue byte in a path escpod does not control must not corrupt the
        // rest of the value or crash the writer, just degrade to `?`.
        assert_eq!(sam_header_safe("a\tb\nc"), "a?b?c");
        assert_eq!(sam_header_safe("caf\u{e9}"), "caf?"); // non-ASCII (é)
    }

    /// A pathological `CublasPairing` — an embedded control character, as
    /// `release_of`'s unversioned fallback could in principle carry from an
    /// unusual filesystem path — must not reach `provenance_ds` unsanitized
    /// (rnabioco/escapepod-rs#426 review finding): `write_header` validates
    /// every `@PG` field byte-for-byte against the same SAM grammar, and a
    /// value that fails it errors out at the very end of a full run.
    #[cfg(feature = "gpu")]
    #[test]
    fn device_provenance_sanitizes_a_pathological_pairing() {
        use escapepod_classify::cuda_libs::CublasPairing;

        let pairing = CublasPairing {
            cublas_path: std::path::PathBuf::from("/opt/lib\ncublas.so"),
            cublas_version: "(unversioned, in /opt/weird\tdir)".to_string(),
            lt_path: std::path::PathBuf::from("/opt/lib/libcublasLt.so.12"),
            lt_version: "12.8.93".to_string(),
            repaired: false,
        };
        let dp = DeviceProvenance::cpu_gpu_refused(&pairing);
        for v in [
            dp.cublas_path.as_deref(),
            dp.cublas_version.as_deref(),
            dp.cublaslt_path.as_deref(),
        ]
        .into_iter()
        .flatten()
        {
            assert!(
                v.bytes().all(|b| (b' '..=b'~').contains(&b)),
                "every byte of {v:?} must be within the SAM header-value grammar"
            );
        }
    }
}
