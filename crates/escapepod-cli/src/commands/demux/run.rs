//! Fused, streaming demux pipeline: decode each read's signal **once**, run
//! detect → fingerprint → classify in a single pass, and route the read
//! (block-level compressed copy, no re-decode/re-compress) into its barcode's
//! output POD5. No intermediate boundaries/fingerprints/classifications files
//! are written unless explicitly requested (`--classifications`).
//!
//! Pipeline (all stages overlap):
//!   A. rayon pool decodes + detects + fingerprints reads in parallel (per
//!      Arrow batch, bounded memory).
//!   B. classify — CPU per-read (in stage A), or, with `--gpu`, a dedicated GPU
//!      thread that is continuously fed fingerprint blocks through a bounded
//!      channel (double-buffered, so the GPU isn't idle between batches).
//!   C. one writer thread **per barcode** does the serial block-copy for that
//!      barcode — writes parallelize across barcodes instead of one global
//!      writer being the bottleneck.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc::SyncSender;

use crate::progress::create_progress_bar;
use crate::style;
use escapepod_demux::{
    AnyModel, DtwSvmModel, GbmModel, GbmPredictor, SvmPredictor, SvmWorkspace,
    extract_fingerprint_from_signal, load_any_model,
};
#[cfg(feature = "crf-decode")]
use escapepod_demux::crf::{BarcodeRefs, CrfEncoder, CrfScratch};
use escapepod_signal::dtw::NormMethod;
use escapepod_signal::segmentation::{detect_adapter, downscale_normalize_into};
use escapepod_signal::{
    CompressedSignalChunk, PredefinedDictionaries, ReadData, Reader, ReadsBatchView, RunInfoData,
    Uuid, Writer, WriterOptions,
};
use rayon::prelude::*;
use tracing::info;

const UNCLASSIFIED: &str = "unclassified";


/// Arguments for the fused demux pipeline.
#[derive(Debug, clap::Args)]
pub struct RunArgs {
    /// Input POD5 file(s). (Required for the fused pipeline; validated at
    /// runtime so the advanced subcommands aren't forced to provide it.)
    #[arg(value_name = "FILES")]
    pub input: Vec<PathBuf>,

    /// Trained classifier — a DTW-SVM / GBM JSON (auto-detected by JSON
    /// shape), or a CTC-CRF encoder bundle directory (`metadata.json` + the
    /// ONNX graph it names). A CRF bundle also needs `--barcodes`.
    #[arg(long, value_name = "FILE|DIR")]
    pub model: Option<PathBuf>,

    /// Barcode reference CSV (`name,sequence`) for the CTC-CRF head. Required
    /// with a CRF bundle, ignored otherwise: the fingerprint heads carry their
    /// own barcode set in the model JSON, whereas the CRF emits sequence and
    /// has to be told what to match it against.
    ///
    /// These must be the sequences the model actually EMITS, which is not the
    /// training target: `state_len` leading bases only fix the initial CRF
    /// state and are never produced, so a 40-nt target emits 36 nt. Matching
    /// against full-length targets still calls the same barcode, but inflates
    /// every distance and compresses the confidence margin that `--min-margin`
    /// gates on (escapepod-models#36).
    #[cfg(feature = "crf-decode")]
    #[arg(long, value_name = "FILE")]
    pub barcodes: Option<PathBuf>,

    /// Call a read `unclassified` when its edit-distance margin to the
    /// second-best reference is below this (CRF head only). 0 keeps every
    /// call, including outright ties.
    #[cfg(feature = "crf-decode")]
    #[arg(
        long,
        default_value = "0",
        value_name = "N",
        help_heading = "Advanced Options"
    )]
    pub min_margin: u32,

    /// Describe the model and exit: identity, signal geometry, bundled
    /// references, pinned boundary detector, published metrics, and the exact
    /// command line it needs. Reads no POD5, so it is safe to run against a
    /// model before trusting it.
    #[arg(long)]
    pub info: bool,

    /// Output directory for the per-barcode demultiplexed POD5 files
    #[arg(short = 'd', long, value_name = "DIR")]
    pub output_dir: Option<PathBuf>,

    /// Also write a per-read classifications table (CSV). Off by default —
    /// the pipeline streams in memory and only writes demuxed POD5.
    #[arg(long, value_name = "FILE", help_heading = "Advanced Options")]
    pub classifications: Option<PathBuf>,

    /// Prefix for the per-barcode output filenames (`<prefix>_<barcode>.pod5`)
    #[arg(long, default_value = "barcode", help_heading = "Advanced Options")]
    pub prefix: String,

    /// Adapter detection method: `cnn` or `llr`. **No default** — LLR is
    /// opt-in, never inferred.
    ///
    /// LLR boundaries cost 17.2 points of barcode recall against the same
    /// classifier (0.9928 -> 0.8196, escapepod-models#16) and the failure is
    /// silent: it runs and produces plausible output. So a model bundle that
    /// pins its detector supplies this, and otherwise you have to say which
    /// you want. Passing it explicitly overrides a bundle's choice, except
    /// that a bundle pinning `cnn` refuses to be downgraded to `llr`.
    #[arg(long, value_name = "{cnn,llr}", help_heading = "Advanced Options")]
    pub method: Option<String>,

    /// Path to the ADAPTed CNN ONNX model (only with `--method cnn`).
    #[cfg(feature = "cnn-detect")]
    #[arg(long, value_name = "FILE", help_heading = "Advanced Options")]
    pub cnn_model: Option<PathBuf>,

    /// Minimum observations for the adapter segment (LLR detect).
    #[arg(
        long,
        default_value = "200",
        value_name = "N",
        help_heading = "Advanced Options"
    )]
    pub min_adapter: usize,

    /// Border trim size (LLR detect).
    #[arg(
        long,
        default_value = "50",
        value_name = "N",
        help_heading = "Advanced Options"
    )]
    pub border_trim: usize,

    /// Downscale factor for LLR signal processing. Default 10 is the
    /// WarpDemuX-native mode (~5× faster detect, the dominant prep stage,
    /// with ~98% barcode agreement vs full resolution). Set 1 for
    /// full-resolution detect.
    #[arg(
        long,
        default_value = "10",
        value_name = "N",
        help_heading = "Advanced Options"
    )]
    pub downscale: usize,

    /// [experimental] Use the GPU where this pipeline supports it: batched
    /// DTW-SVM classify (`--features gpu`) and/or batched CNN adapter detection
    /// with `--method cnn` (`--features cnn-gpu`). CPU prep stays parallel and
    /// feeds the GPU; CPU falls back automatically for stages without a GPU
    /// path (e.g. GBM classify).
    ///
    /// GPU DTW classify is NOT recommended on a full node — the CPU DTW is
    /// faster there (measured 113 s CPU on 64 cores vs 132 s with `--gpu` on an
    /// A30, plus ~2.2 GB more RSS). It may help when cores are scarce. GPU CNN
    /// detection (`--method cnn --gpu`) is the case that does pay off.
    #[cfg(any(feature = "gpu", feature = "cnn-gpu"))]
    #[arg(long, help_heading = "Advanced Options")]
    pub gpu: bool,

    /// Number of threads for parallel processing (default: 16, or all available CPUs if fewer)
    #[arg(short = 't', long, visible_short_alias = 'j', value_name = "N")]
    pub threads: Option<usize>,

    /// Print per-phase timing breakdown after completion
    #[arg(long)]
    pub profile: bool,
}

/// A classified read handed to its barcode's writer thread (block-copy).
struct Routed {
    read: ReadData,
    chunks: Vec<CompressedSignalChunk>,
    run_infos: Arc<Vec<RunInfoData>>,
}

/// Fingerprint parameters (WarpDemuX-compatible — the parity default).
#[derive(Clone, Copy)]
struct FpParams {
    num_segments: usize,
    window_width: usize,
    min_separation: Option<usize>,
    keep_last: Option<usize>,
}

impl Default for FpParams {
    fn default() -> Self {
        Self {
            num_segments: 111,
            window_width: 12,
            min_separation: Some(6),
            keep_last: Some(25),
        }
    }
}

/// Adapter detector — LLR (always available), CPU CNN (`cnn-detect`), or
/// batched GPU CNN (`cnn-gpu`). The fused pipeline always detects through
/// [`Detector::detect_batch`] so the GPU variant runs as one onnxruntime call
/// per block instead of per read.
enum Detector {
    Llr {
        min_adapter: usize,
        border_trim: usize,
        downscale: usize,
    },
    #[cfg(feature = "cnn-detect")]
    Cnn(Box<escapepod_demux::AdapterCnn>),
    #[cfg(feature = "cnn-gpu")]
    CnnGpu(Box<escapepod_demux::AdapterCnnGpu>),
}

/// Per-worker scratch for the LLR detect prep (normalize + downscale).
#[derive(Default)]
struct DetectScratch {
    prep: escapepod_signal::segmentation::SignalPrepScratch,
    processed: Vec<f32>,
}

impl Detector {
    /// Detect `(start, end)` for one read, reusing caller-owned buffers. The
    /// LLR prep otherwise allocates three full-length `f32` buffers per read,
    /// which is a real RSS spike on the long tail of the read-length
    /// distribution (medians of ~8 k samples, maxima in the millions).
    fn detect_with(&self, signal: &[i16], scratch: &mut DetectScratch) -> (usize, usize) {
        match self {
            Detector::Llr {
                min_adapter,
                border_trim,
                downscale: ds,
            } => {
                downscale_normalize_into(signal, *ds, &mut scratch.prep, &mut scratch.processed);
                let scale = if *ds > 1 { *ds } else { 1 };
                let (s, e) = detect_adapter(
                    &scratch.processed,
                    (min_adapter / scale).max(1),
                    (border_trim / scale).max(1),
                );
                (s * scale, e * scale)
            }
            #[cfg(feature = "cnn-detect")]
            Detector::Cnn(cnn) => {
                let sig_f32: Vec<f32> = signal.iter().map(|&s| s as f32).collect();
                (0, cnn.detect_adapter_end(&sig_f32).unwrap_or(0))
            }
            // Per-read is a degenerate single-read batch; the producers always go
            // through `detect_batch`, so this is only a correctness fallback.
            #[cfg(feature = "cnn-gpu")]
            Detector::CnnGpu(gpu) => {
                let sig_f32: Vec<f32> = signal.iter().map(|&s| s as f32).collect();
                let end = gpu
                    .detect_adapter_end_batch(&[&sig_f32])
                    .into_iter()
                    .next()
                    .and_then(Result::ok)
                    .unwrap_or(0);
                (0, end)
            }
        }
    }

    /// Per-read `(start, end)` for a whole window of decoded signals (`None` =
    /// decode failed → `(0, 0)`, routed as unclassified). GPU CNN runs as one
    /// batched onnxruntime call (length-grouped); LLR and CPU CNN run per read
    /// in parallel. Bit-identical to calling [`Self::detect_with`] on each signal.
    fn detect_batch(&self, signals: &[Option<Vec<i16>>]) -> Vec<(usize, usize)> {
        #[cfg(feature = "cnn-gpu")]
        if let Detector::CnnGpu(gpu) = self {
            let cfg = gpu.config();
            let prepped: Vec<Option<Vec<f32>>> = signals
                .par_iter()
                .map(|s| {
                    s.as_ref().and_then(|v| {
                        let f: Vec<f32> = v.iter().map(|&x| x as f32).collect();
                        cfg.prep(&f)
                    })
                })
                .collect();
            return gpu
                .detect_prepped(&prepped)
                .into_iter()
                .map(|r| (0usize, r.unwrap_or(0)))
                .collect();
        }
        signals
            .par_iter()
            .map_init(DetectScratch::default, |scratch, s| {
                s.as_ref().map_or((0, 0), |v| self.detect_with(v, scratch))
            })
            .collect()
    }

    /// Leading samples this detector needs decoded (`None` = the whole read).
    /// CNN only looks at `[min_obs_adapter:max_obs_trace]`, so long reads (mRNA)
    /// can skip decompressing the rest of the signal; LLR normalizes over the
    /// whole read, so it needs all of it.
    fn signal_decode_bound(&self) -> Option<usize> {
        match self {
            Detector::Llr { .. } => None,
            #[cfg(feature = "cnn-detect")]
            Detector::Cnn(c) => Some(c.config().max_obs_trace),
            #[cfg(feature = "cnn-gpu")]
            Detector::CnnGpu(g) => Some(g.config().max_obs_trace),
        }
    }
}

fn barcode_label(predicted: i32) -> String {
    // `predicted` is already -1 when the SVM call was below threshold.
    if predicted >= 0 {
        format!("BC{predicted:02}")
    } else {
        UNCLASSIFIED.to_string()
    }
}

/// The set of output barcode labels (model barcodes + unclassified). Takes the
/// raw `label_mapper` so it serves both the SVM and GBM heads.
fn barcode_set(label_mapper: &HashMap<usize, i32>) -> Vec<String> {
    let mut set: Vec<String> = label_mapper
        .values()
        .filter(|&&id| id >= 0)
        .map(|&id| format!("BC{id:02}"))
        .collect();
    set.sort();
    set.dedup();
    set.push(UNCLASSIFIED.to_string());
    set
}

/// Union the pore_type / end_reason dictionaries across all input files.
fn collect_dictionaries(input: &[PathBuf]) -> anyhow::Result<(Vec<String>, Vec<String>)> {
    use std::collections::BTreeSet;
    let mut pores: BTreeSet<String> = BTreeSet::new();
    let mut ends: BTreeSet<String> = BTreeSet::new();
    for path in input {
        let (p, e) = Reader::open(path)?.reads_dictionaries()?;
        pores.extend(p);
        ends.extend(e);
    }
    Ok((pores.into_iter().collect(), ends.into_iter().collect()))
}

/// Channels to the per-barcode writer threads. Cloneable senders are `Sync`,
/// so producers on any thread can route concurrently.
type Routers = HashMap<String, SyncSender<Routed>>;

/// Route one classified read to its barcode writer + (optionally) the
/// classifications CSV.
fn route(
    routers: &Routers,
    class_tx: Option<&SyncSender<(Uuid, String, f64)>>,
    read: ReadData,
    barcode: String,
    chunks: Vec<CompressedSignalChunk>,
    run_infos: Arc<Vec<RunInfoData>>,
    confidence: f64,
) {
    if let Some(ctx) = class_tx {
        let _ = ctx.send((read.read_id, barcode.clone(), confidence));
    }
    let tx = routers
        .get(&barcode)
        .or_else(|| routers.get(UNCLASSIFIED))
        .expect("unclassified router always present");
    let _ = tx.send(Routed {
        read,
        chunks,
        run_infos,
    });
}

/// Which classifier head the fused pipeline drives.
///
/// The two fingerprint heads (DTW-SVM, with an optional GPU DTW path, and the
/// CPU-only GBM tree walk) share everything up to classify: detect, then a
/// fingerprint of the adapter region, then a model that maps features to a
/// class index.
///
/// The CRF head is a different shape. It does not fingerprint at all — it
/// basecalls the barcode out of the raw pA window `[adapter_end - chunk,
/// adapter_end]` and matches the decoded sequence to a reference set by edit
/// distance. So its barcode set comes from the reference CSV rather than a
/// `label_mapper`, and its confidence is an edit-distance margin rather than a
/// probability.
enum ClassifyModel {
    Svm(DtwSvmModel),
    Gbm(GbmModel),
    #[cfg(feature = "crf-decode")]
    Crf(Box<CrfHead>),
}

/// The CTC-CRF head: encoder bundle plus the references its decodes are matched
/// against.
#[cfg(feature = "crf-decode")]
struct CrfHead {
    encoder: CrfEncoder,
    refs: BarcodeRefs,
    min_margin: u32,
}

impl ClassifyModel {
    /// The set of output barcode labels, before `unclassified` is added.
    ///
    /// The fingerprint heads name barcodes positionally from the model's
    /// `label_mapper` (`BC00`, `BC01`, ...); the CRF head uses the reference
    /// names, so its output files are `barcode_nbc01.pod5` rather than
    /// `barcode_BC00.pod5`.
    fn barcode_names(&self) -> Vec<String> {
        match self {
            ClassifyModel::Svm(m) => barcode_set(&m.label_mapper),
            ClassifyModel::Gbm(m) => barcode_set(&m.label_mapper),
            #[cfg(feature = "crf-decode")]
            ClassifyModel::Crf(h) => {
                // Reference order is the CSV's, but every label here becomes a
                // router key and an output file, so a duplicate name (or one
                // literally called `unclassified`) would leave a writer thread
                // with no sender. Dedup rather than trusting the CSV.
                let mut v: Vec<String> = Vec::with_capacity(h.refs.len() + 1);
                for n in h.refs.names() {
                    if n != UNCLASSIFIED && !v.iter().any(|s| s == n) {
                        v.push(n.clone());
                    }
                }
                v.push(UNCLASSIFIED.to_string());
                v
            }
        }
    }
}

/// Is this `--model` a CTC-CRF encoder bundle rather than a classifier JSON?
///
/// A bundle is a directory holding `metadata.json`, or that `metadata.json`
/// itself. Sniff the `format` key rather than trusting the extension, so a
/// stray `.json` cannot be mistaken for either kind: the CRF sidecar declares
/// `"format": "escapepod-crf-encoder/N"`, which no classifier JSON carries.
#[cfg(feature = "crf-decode")]
pub(super) fn crf_bundle_dir(path: &Path) -> Option<PathBuf> {
    let (dir, meta) = if path.is_dir() {
        (path.to_path_buf(), path.join("metadata.json"))
    } else if path.file_name().is_some_and(|n| n == "metadata.json") {
        (path.parent()?.to_path_buf(), path.to_path_buf())
    } else {
        return None;
    };
    let text = std::fs::read_to_string(&meta).ok()?;
    let json: serde_json::Value = serde_json::from_str(&text).ok()?;
    json.get("format")?
        .as_str()?
        .starts_with("escapepod-crf-encoder/")
        .then_some(dir)
}

pub fn run(args: RunArgs) -> anyhow::Result<()> {
    use crate::commands::profile::PhaseTimer;
    let mut timer = PhaseTimer::new();
    timer.phase("Fused demux");
    let profile = args.profile;

    // Validate the fused-pipeline args here (not via clap `required`) so the
    // advanced subcommands aren't forced to supply them.
    let model_path = args
        .model
        .clone()
        .ok_or_else(|| anyhow::anyhow!("--model <FILE|DIR> is required"))?;
    // `--info` describes the model and exits: no input, no output dir, no POD5
    // touched. Checked before the input/output validation below so you can
    // interrogate a model without inventing arguments for a run you are not
    // making.
    if args.info {
        return super::info::run(&model_path);
    }
    if args.input.is_empty() {
        anyhow::bail!("no input POD5 file(s) given");
    }
    let output_dir = args
        .output_dir
        .clone()
        .ok_or_else(|| anyhow::anyhow!("-d/--output-dir <DIR> is required"))?;

    // Three heads: DTW-SVM (with an optional GPU DTW path), the native GBM tree
    // ensemble (CPU-only), and the CTC-CRF basecaller. A CRF bundle is a
    // directory, so check for that before trying to parse `--model` as JSON.
    // Only the legacy reference-bank WarpDemux JSON is rejected here.
    #[cfg(feature = "crf-decode")]
    let crf_dir = crf_bundle_dir(&model_path);
    #[cfg(not(feature = "crf-decode"))]
    let crf_dir: Option<PathBuf> = None;
    // Kept because a pinned boundary model's ONNX path is relative to the
    // bundle directory, and `crf_dir` is consumed building the head below.
    #[cfg(feature = "crf-decode")]
    let crf_dir_for_pin = crf_dir.clone();

    let model = match crf_dir {
        #[cfg(feature = "crf-decode")]
        Some(dir) => {
            let encoder = CrfEncoder::load_bundle(&dir)?;
            // References come from the bundle unless the caller overrides them.
            // Carrying them in the bundle is what makes the plain
            // `--model <bundle> -d out/` form work, and it fixes the
            // emitted-vs-target trimming once at export instead of at every
            // call site (escapepod-models#36).
            let refs = match (&args.barcodes, &encoder.metadata().barcodes) {
                (Some(csv), _) => {
                    let r = BarcodeRefs::from_csv(csv)?;
                    if encoder.metadata().barcodes.is_some() {
                        info!(
                            "{} overriding the {} references in the bundle",
                            style::label("Barcodes:"),
                            style::count(encoder.metadata().barcodes.as_ref().unwrap().len())
                        );
                    }
                    r
                }
                (None, Some(entries)) => BarcodeRefs::from_pairs(
                    entries.iter().map(|e| (e.name.clone(), e.sequence.clone())),
                )?,
                (None, None) => anyhow::bail!(
                    "this CTC-CRF bundle carries no barcode references, so --barcodes \
                     <FILE> is required. The CRF emits sequence rather than a class \
                     index and has to be told what to match it against — and those must \
                     be the sequences the model EMITS (target[state_len:]), not the \
                     full-length training targets."
                ),
            };
            info!(
                "{} {} references, minimum pairwise edit distance {}",
                style::label("Barcodes:"),
                style::count(refs.len()),
                refs.min_pairwise_distance()
                    .map_or_else(|| "n/a".to_string(), |d| d.to_string()),
            );
            ClassifyModel::Crf(Box::new(CrfHead {
                encoder,
                refs,
                min_margin: args.min_margin,
            }))
        }
        #[cfg(not(feature = "crf-decode"))]
        Some(_) => unreachable!("crf_dir is always None without the crf-decode feature"),
        None => match load_any_model(&model_path)? {
            AnyModel::Svm(m) => ClassifyModel::Svm(m),
            AnyModel::Gbm(m) => ClassifyModel::Gbm(m),
            AnyModel::WarpDemux(_) => anyhow::bail!(
                "`demux` needs an SVM, GBM or CTC-CRF model (DtwSvmModel / converted \
                 WarpDemuX / native GBM / CRF bundle directory). The reference-bank path \
                 is only on `demux classify --reference`."
            ),
        },
    };
    // A CRF bundle may pin the boundary detector it was trained against; the
    // ONNX path in the sidecar is relative to the bundle directory.
    #[cfg(feature = "crf-decode")]
    let boundary_pin = match (&model, &crf_dir_for_pin) {
        (ClassifyModel::Crf(h), Some(dir)) => h.encoder.metadata().boundary.as_ref().map(|b| {
            if let Some(id) = &b.model_id {
                info!(
                    "{} {} (pinned by the model bundle)",
                    style::label("Boundary model:"),
                    style::value(id)
                );
            }
            (b.method.as_str(), b.onnx.as_ref().map(|o| dir.join(o)))
        }),
        _ => None,
    };
    #[cfg(not(feature = "crf-decode"))]
    let boundary_pin: Option<(&str, Option<PathBuf>)> = None;

    let detector = build_detector(&args, boundary_pin)?;
    let fp = FpParams::default();

    std::fs::create_dir_all(&output_dir)?;

    info!("{} fused streaming demux", style::action("Running"));
    info!(
        "{} {} POD5 file(s)",
        style::label("Input:"),
        style::count(args.input.len())
    );
    info!(
        "{} {}",
        style::label("Model:"),
        style::path(model_path.display())
    );
    info!(
        "{} {}",
        style::label("Output:"),
        style::path(output_dir.display())
    );

    let total = super::utils::total_read_count(&args.input);
    let pb = create_progress_bar(total as u64, "Demuxing")?;

    // Pre-declare the output dictionaries (pore_type / end_reason) so each
    // block-copy writer has a fixed dictionary across all batches — Arrow IPC
    // forbids the dictionary changing between batches. Read straight from the
    // source files' Arrow dictionaries (O(dict), not O(reads)).
    let (pore_types, end_reasons) = collect_dictionaries(&args.input)?;
    let predefined = PredefinedDictionaries {
        pore_types: Some(pore_types),
        end_reasons: Some(end_reasons),
    };

    // ---- Stage C: one writer thread per barcode (sharded) ----
    //
    // Channel depth is budgeted across barcodes rather than fixed per barcode.
    // Each queued `Routed` owns that read's compressed bytes (~10 KB), so a
    // fixed 4096 per barcode meant worst-case in-flight memory scaled with the
    // barcode count — 0.5 GB at 12 barcodes, ~3.8 GB at 96. The depth still
    // needs to be generous enough to absorb bursts: real libraries are heavily
    // skewed (one benchmark routed 86% of 1.22 M reads to a single barcode),
    // and when a channel fills, rayon workers block inside `for_each` on
    // `send` — and blocked rayon workers cannot be stolen from, so one
    // saturated writer stalls the whole pool.
    let barcodes = model.barcode_names();
    let router_depth = (ROUTER_TOTAL_SLOTS / barcodes.len().max(1)).clamp(256, 4096);

    let mut routers: Routers = HashMap::new();
    let mut writer_handles: Vec<(String, std::thread::JoinHandle<anyhow::Result<usize>>)> =
        Vec::new();
    for bc in barcodes {
        let (tx, rx) = std::sync::mpsc::sync_channel::<Routed>(router_depth);
        let path = output_dir.join(format!("{}_{}.pod5", args.prefix, bc));
        let dicts = predefined.clone();
        let handle = std::thread::spawn(move || writer_thread(rx, &path, dicts));
        routers.insert(bc.clone(), tx);
        writer_handles.push((bc, handle));
    }
    let routers = Arc::new(routers);

    // Optional classifications CSV writer (a single small-record stream).
    let (class_tx, class_handle) = spawn_class_writer(args.classifications.as_deref())?;

    // ---- Stages A/B: produce classified reads ----
    let produce_result = match &model {
        ClassifyModel::Svm(svm) => {
            #[cfg(feature = "gpu")]
            {
                if args.gpu {
                    produce_gpu(&args, &detector, svm, fp, &routers, class_tx.as_ref(), &pb)
                } else {
                    produce_cpu(&args, &detector, svm, fp, &routers, class_tx.as_ref(), &pb)
                }
            }
            #[cfg(not(feature = "gpu"))]
            {
                produce_cpu(&args, &detector, svm, fp, &routers, class_tx.as_ref(), &pb)
            }
        }
        ClassifyModel::Gbm(gbm) => {
            // GBM classify is CPU-only; with `--method cnn` the GPU still
            // accelerates adapter detection, so only warn when `--gpu` can do
            // nothing (CPU classify + CPU detect).
            #[cfg(any(feature = "gpu", feature = "cnn-gpu"))]
            if args.gpu && args.method.as_deref() != Some("cnn") {
                tracing::warn!(
                    "--gpu has no effect here: GBM classify is CPU-only and \
                     `--method {}` detection is CPU-only (use `--method cnn` for \
                     GPU adapter detection).",
                    args.method.as_deref().unwrap_or("<from model>")
                );
            }
            produce_cpu_gbm(&args, &detector, gbm, fp, &routers, class_tx.as_ref(), &pb)
        }
        #[cfg(feature = "crf-decode")]
        ClassifyModel::Crf(head) => {
            // Encoder inference here is tract on the CPU, one read per rayon
            // worker. `demux basecall --gpu` has an onnxruntime path; it is not
            // wired through the fused pipeline, so say so rather than silently
            // ignoring the flag.
            #[cfg(any(feature = "gpu", feature = "cnn-gpu"))]
            if args.gpu && args.method.as_deref() != Some("cnn") {
                tracing::warn!(
                    "--gpu has no effect here: fused CRF basecalling is CPU-only and \
                     `--method {}` detection is CPU-only (use `--method cnn` for GPU \
                     adapter detection, or `demux basecall --gpu` for a GPU encoder).",
                    args.method.as_deref().unwrap_or("<from model>")
                );
            }
            produce_cpu_crf(&args, &detector, head, &routers, class_tx.as_ref(), &pb)
        }
    };

    // Drop all senders so the writer threads see EOF. The producers only
    // borrowed `&Routers`, so this is the last `Arc` reference.
    drop(class_tx);
    match Arc::try_unwrap(routers) {
        Ok(map) => drop(map),
        Err(_) => unreachable!("router Arc still shared after producers returned"),
    }

    // Join writers, collect counts.
    let mut summary = DemuxSummary::default();
    for (bc, handle) in writer_handles {
        let n = handle
            .join()
            .map_err(|e| anyhow::anyhow!("writer thread for {bc} panicked: {e:?}"))??;
        if n > 0 {
            summary.per_barcode.push((bc, n));
        }
    }
    if let Some(h) = class_handle {
        h.join()
            .map_err(|e| anyhow::anyhow!("classifications writer panicked: {e:?}"))??;
    }
    summary.per_barcode.sort();
    produce_result?;

    pb.finish_with_message("complete");
    print_summary(&summary);
    timer.report(profile);
    Ok(())
}

/// Total queued reads allowed across *all* per-barcode writer channels. Each
/// queued read owns its compressed signal (~10 KB), so this bounds router
/// memory at roughly 500 MB regardless of how many barcodes the model defines.
/// Split evenly per barcode and clamped — see the router setup in `run`.
const ROUTER_TOTAL_SLOTS: usize = 49_152;

/// Upper bound on reads accumulated into one detect+classify block. POD5 Arrow
/// read-batches are small (~1000 reads), so detecting per batch makes GPU CNN
/// fire many tiny calls; accumulating across batches into a large block groups
/// far more same-length reads per onnxruntime call. The on-device batch is
/// separately capped by `gpu_batch_elems`.
const DETECT_WINDOW: usize = 65_536;

/// Upper bound on the *decoded signal bytes* held in one block.
///
/// A read-count cap alone does not bound memory: LLR sets no decode bound (it
/// normalizes over the whole read), so a block holds whole reads. At 65,536
/// reads that is ~1.8 GB on a short-read RNA library and ~9.5 GB on a long-read
/// one — measured 13.5 GB peak RSS on a 1.22 M-read file, of which ~2.4 GB was
/// this block (the rest is mmap'd input pages). Capping by bytes keeps the
/// footprint flat across libraries with wildly different read lengths, while
/// the read cap above still lets the GPU-CNN path (which decodes only ~806
/// samples per read) fill a full 65,536-read block before this binds.
///
/// 128 MB measured best on 1.22 M reads; larger is worse on both wall time and
/// RSS (512 MB: 74.4 s / 13.07 GB vs 128 MB: 63.8 s / 12.68 GB).
const BLOCK_TARGET_BYTES: usize = 128 * 1024 * 1024;

/// Blocks queued between the reader threads and the processing loop.
///
/// Enough to cover a reader hiccup without inflating peak memory — measured
/// worth ~3% on its own (56.9 s -> 55.1 s at one filler); the filler count is
/// the larger lever.
const BLOCK_QUEUE_DEPTH: usize = 2;

/// Number of block-filling reader threads (`ESCAPEPOD_DEMUX_FILLERS`).
///
/// Two is the measured sweet spot; see [`fill_shard`] for the access-pattern
/// caveat. On 1.22 M reads (BeeGFS input, 48 cores):
///
/// ```text
/// fillers/queue   wall     CPU    peak RSS
///   1 / 1        56.9 s    771%    12.2 GB
///   2 / 2        48.4 s    918%    13.0 GB   <- default
///   4 / 4        48.1 s    931%    14.5 GB
///   8 / 4        48.5 s    923%    17.3 GB
/// ```
///
/// One filler leaves the pipeline waiting on a single sequential sweep; a
/// second covers that stall. Past two it is flat, and the extra blocks in
/// flight cost real memory. Set `ESCAPEPOD_DEMUX_FILLERS=1` on a cold network
/// filesystem where concurrent streams are known to hurt.
fn filler_threads() -> usize {
    use std::sync::OnceLock;
    static CACHED: OnceLock<usize> = OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("ESCAPEPOD_DEMUX_FILLERS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or(2)
            .min(32)
    })
}

/// A block row for the GBM producer: decoded signal, detected adapter bounds,
/// and the read's write-side context.
type GbmRow = (Option<Vec<i16>>, (usize, usize), BlockItem);

/// One read's context carried alongside its decoded signal through a block:
/// the writable read record, its compressed chunks (for the block-level write),
/// and its run-info table.
type BlockItem = (ReadData, Vec<CompressedSignalChunk>, Arc<Vec<RunInfoData>>);

/// One filled block handed from the reader thread to the processing loop.
type SignalBlock = (Vec<Option<Vec<i16>>>, Vec<BlockItem>);

/// Stream reads through the fused pipeline in blocks: per Arrow batch do the
/// single-stream, ascending-order signal sweep (#72) and decode in parallel,
/// accumulate across batches up to [`DETECT_WINDOW`] reads or
/// [`BLOCK_TARGET_BYTES`] of decoded signal, then hand each full block (decoded
/// signals + per-read context, aligned) to `process_block`. Accumulating across
/// the file's small Arrow batches is what lets GPU detection batch many reads
/// per call.
///
/// Filling runs on its own thread and feeds `process_block` through a depth-1
/// channel, so the next block's I/O and decode overlap with the current block's
/// detect + classify. Previously these were serial: no I/O was in flight for
/// the entire detect+classify phase, which on a 1.22 M-read benchmark left ~43
/// of 48 cores idle and made the pipeline *slower* at 48 threads than at 8.
fn drive_blocks(
    input: &[std::path::PathBuf],
    decode_to: Option<usize>,
    mut process_block: impl FnMut(Vec<Option<Vec<i16>>>, Vec<BlockItem>),
) -> anyhow::Result<()> {
    let shards = filler_threads();
    let (tx, rx) = std::sync::mpsc::sync_channel::<SignalBlock>(BLOCK_QUEUE_DEPTH);

    std::thread::scope(|scope| -> anyhow::Result<()> {
        let mut fillers = Vec::with_capacity(shards);
        for shard in 0..shards {
            let tx = tx.clone();
            fillers.push(scope.spawn(move || fill_shard(input, decode_to, shard, shards, tx)));
        }
        // Drop the extra sender so `rx` ends once every shard has finished.
        drop(tx);

        for (sigs, items) in rx {
            process_block(sigs, items);
        }

        for f in fillers {
            f.join()
                .map_err(|e| anyhow::anyhow!("demux reader thread panicked: {e:?}"))??;
        }
        Ok(())
    })
}

/// Fill blocks from the Arrow batches assigned to `shard` (every `shards`-th
/// batch) and send them downstream.
///
/// With `shards == 1` this is one sequential ascending sweep of the whole file
/// — the #72 access pattern. Sharding hands each thread a strided subset of
/// batches: still coarse-grained sequential I/O per thread, qualitatively
/// unlike the per-read demand paging #72 fixed (48 threads faulting per read
/// measured 0.3 MB/s against 288 MB/s for one sweep), but it is N concurrent
/// streams rather than one.
///
/// The default of 2 was measured on BeeGFS input that had been read repeatedly
/// and so may have been partly cached; the cold case is **not** independently
/// validated. `ESCAPEPOD_DEMUX_FILLERS=1` restores the strict single-stream
/// behavior if concurrent streams turn out to hurt on a cold mount.
fn fill_shard(
    input: &[std::path::PathBuf],
    decode_to: Option<usize>,
    shard: usize,
    shards: usize,
    tx: std::sync::mpsc::SyncSender<SignalBlock>,
) -> anyhow::Result<()> {
    let mut sigs: Vec<Option<Vec<i16>>> = Vec::new();
    let mut items: Vec<BlockItem> = Vec::new();
    let mut block_bytes = 0usize;

    for path in input {
        let reader = Reader::open(path)?;
        let run_infos = Arc::new(reader.run_infos().to_vec());
        for (bi, batch) in reader.read_batches()?.enumerate() {
            if bi % shards != shard {
                continue;
            }
            let batch = batch?;
            let view = ReadsBatchView::new(&batch, false)?;
            let reads: Vec<ReadData> = (0..view.num_rows())
                .filter_map(|row| view.read(row).ok())
                .filter(|r| !r.signal_rows.is_empty())
                .collect();
            // One sequential, ascending-order sweep pulls this batch's
            // signal (grouped by Arrow record-batch — see #72), then
            // decode in parallel.
            let keyed: Vec<(usize, Vec<u64>)> = reads
                .iter()
                .enumerate()
                .map(|(i, r)| (i, r.signal_rows.clone()))
                .collect();
            let bulk = reader.get_compressed_signal_bulk(&keyed)?;
            let decoded: Vec<Option<Vec<i16>>> = bulk
                .par_iter()
                .map(|(_, chunks)| super::utils::decode_chunks_to(chunks, decode_to))
                .collect();

            let mut reads_opt: Vec<Option<ReadData>> = reads.into_iter().map(Some).collect();
            for ((i, chunks), sig) in bulk.into_iter().zip(decoded) {
                let read = reads_opt[i].take().expect("each read consumed once");
                block_bytes += sig.as_ref().map_or(0, |s| s.len() * 2)
                    + chunks.iter().map(|c| c.data.len()).sum::<usize>();
                sigs.push(sig);
                items.push((read, chunks, run_infos.clone()));
            }

            if sigs.len() >= DETECT_WINDOW || block_bytes >= BLOCK_TARGET_BYTES {
                block_bytes = 0;
                // A send error means the consumer is gone (it returned early or
                // panicked); stop filling rather than block.
                if tx
                    .send((std::mem::take(&mut sigs), std::mem::take(&mut items)))
                    .is_err()
                {
                    return Ok(());
                }
            }
        }
    }
    if !sigs.is_empty() {
        tx.send((sigs, items)).ok();
    }
    Ok(())
}

/// CPU-classify producer (DTW-SVM): stream reads in large blocks
/// ([`drive_blocks`]), batch-detect, then fingerprint + classify + route each
/// read in parallel. `--method cnn --gpu` makes detection a batched GPU stage.
fn produce_cpu(
    args: &RunArgs,
    detector: &Detector,
    model: &DtwSvmModel,
    fp: FpParams,
    routers: &Routers,
    class_tx: Option<&SyncSender<(Uuid, String, f64)>>,
    pb: &indicatif::ProgressBar,
) -> anyhow::Result<()> {
    let predictor = SvmPredictor::new(model);
    drive_blocks(
        &args.input,
        detector.signal_decode_bound(),
        |sigs, items| {
            // Batch-detect the whole block (GPU CNN = grouped onnxruntime calls; LLR
            // / CPU-CNN = parallel per read), then classify reusing each decoded
            // signal. One SVM workspace per rayon worker (not per read): classify
            // scores each read against tens of thousands of training fingerprints,
            // and `SvmWorkspace` holds the reusable scratch (DTW rows, distances,
            // kernel, coupling).
            let bounds = detector.detect_batch(&sigs);
            sigs.into_par_iter().zip(bounds).zip(items).for_each_init(
                || SvmWorkspace::for_model(predictor.model()),
                |ws, ((signal, (s, e)), (read, chunks, run_infos))| {
                    if let Some(signal) = signal {
                        let (barcode, conf) =
                            classify_one_cpu(&read, &signal, s, e, &predictor, fp, ws);
                        route(
                            routers,
                            class_tx,
                            read.for_writing(read.run_info_index),
                            barcode,
                            chunks,
                            run_infos,
                            conf,
                        );
                    }
                    pb.inc(1);
                },
            );
        },
    )
}

/// Classify a single read from its decoded signal and precomputed adapter
/// boundaries (fingerprint → SVM). Returns `(barcode, confidence)`; the caller
/// holds the chunks for routing. Detection is done in batch by the producer.
fn classify_one_cpu(
    read: &ReadData,
    signal: &[i16],
    s: usize,
    e: usize,
    predictor: &SvmPredictor,
    fp: FpParams,
    ws: &mut SvmWorkspace,
) -> (String, f64) {
    if e <= s {
        return (UNCLASSIFIED.to_string(), 0.0);
    }
    let Some(features) = extract_fingerprint_from_signal(
        signal,
        s,
        e,
        fp.num_segments,
        fp.window_width,
        NormMethod::ZScore,
        read.read_id,
        fp.min_separation,
        fp.keep_last,
        false,
    ) else {
        return (UNCLASSIFIED.to_string(), 0.0);
    };
    let (_probs, result) = predictor.predict_with_workspace(&features.values, ws);
    (barcode_label(result.predicted_barcode), result.confidence)
}

/// GBM producer: same fused, single-stream-I/O structure as [`produce_cpu`],
/// but the per-read classifier is the native tree ensemble. The GBM predictor
/// holds no per-read mutable state (it's `Sync`, read-only), so — unlike the SVM
/// path's `for_each_init` workspace — this is a plain `par_iter`. No GPU branch:
/// GBM inference is CPU-only.
fn produce_cpu_gbm(
    args: &RunArgs,
    detector: &Detector,
    model: &GbmModel,
    fp: FpParams,
    routers: &Routers,
    class_tx: Option<&SyncSender<(Uuid, String, f64)>>,
    pb: &indicatif::ProgressBar,
) -> anyhow::Result<()> {
    let predictor = GbmPredictor::new(model);
    drive_blocks(
        &args.input,
        detector.signal_decode_bound(),
        |sigs, items| {
            // Batch-detect the whole block, then fingerprint + GBM-classify in
            // chunks. The chunking exists so each rayon task can run the batched
            // `predict_many`, which walks 8 reads in lockstep down each tree and
            // hides the per-node L2 latency that bottlenecks a serial walk
            // (~2.9× over per-read `predict`, bit-identical). GBM is `Sync` and
            // read-only, so no per-worker workspace is needed.
            const GBM_CHUNK: usize = 1024;
            let bounds = detector.detect_batch(&sigs);
            let rows: Vec<GbmRow> = sigs
                .into_iter()
                .zip(bounds)
                .zip(items)
                .map(|((sig, b), item)| (sig, b, item))
                .collect();

            rows.into_par_iter().chunks(GBM_CHUNK).for_each(|chunk| {
                // Fingerprint the chunk first so the classifier sees a batch.
                // `None` = undetected adapter or unfingerprintable read; those
                // route as unclassified without reaching the model.
                let features: Vec<Option<Vec<f64>>> = chunk
                    .iter()
                    .map(|(signal, (s, e), (read, _, _))| {
                        let signal = signal.as_ref()?;
                        fingerprint_for_gbm(read, signal, *s, *e, fp)
                    })
                    .collect();

                let present: Vec<&[f64]> = features.iter().filter_map(|f| f.as_deref()).collect();
                let predicted = predictor.predict_many(&present).ok();

                let mut next = 0usize;
                for ((_, _, (read, chunks, run_infos)), feat) in chunk.into_iter().zip(&features) {
                    let (barcode, conf) = match feat {
                        Some(_) => {
                            let out = match &predicted {
                                // Batched call succeeded: take this read's slot.
                                Some(results) => {
                                    let (_probs, r) = &results[next];
                                    (barcode_label(r.predicted_barcode), r.confidence)
                                }
                                // The batch is all-or-nothing; retry this read
                                // alone so one bad fingerprint cannot discard
                                // the whole chunk.
                                None => match predictor.predict(present[next]) {
                                    Ok((_probs, r)) => {
                                        (barcode_label(r.predicted_barcode), r.confidence)
                                    }
                                    Err(_) => (UNCLASSIFIED.to_string(), 0.0),
                                },
                            };
                            next += 1;
                            out
                        }
                        None => (UNCLASSIFIED.to_string(), 0.0),
                    };
                    route(
                        routers,
                        class_tx,
                        read.for_writing(read.run_info_index),
                        barcode,
                        chunks,
                        run_infos,
                        conf,
                    );
                }
                pb.inc(features.len() as u64);
            });
        },
    )
}

/// CRF producer: detect → prep the raw-pA window → CTC-CRF basecall → match the
/// decoded sequence to the references by edit distance.
///
/// Unlike the fingerprint heads this needs **calibrated pA**, not ADC counts,
/// and it needs the window `[adapter_end - chunk, adapter_end]` to lie inside
/// the decoded prefix. It always does: the detector bounds its decode by
/// `max_obs_trace` and can only report an `adapter_end` inside what it saw, so
/// any read whose adapter was detected at all has its window available.
/// `meta.prep` still returns `None` when `adapter_end < chunk` (the adapter sits
/// too close to the read start), and those route as unclassified.
///
/// Batching mirrors `produce_cpu_gbm`: detect the whole block at once, then
/// chunk so each rayon task preps, encodes and matches a run of reads. The
/// encode is the dominant cost and tract has no batched LSTM, so parallelism is
/// per read within the chunk.
#[cfg(feature = "crf-decode")]
fn produce_cpu_crf(
    args: &RunArgs,
    detector: &Detector,
    head: &CrfHead,
    routers: &Routers,
    class_tx: Option<&SyncSender<(Uuid, String, f64)>>,
    pb: &indicatif::ProgressBar,
) -> anyhow::Result<()> {
    let meta = head.encoder.metadata();
    drive_blocks(
        &args.input,
        detector.signal_decode_bound(),
        |sigs, items| {
            const CRF_CHUNK: usize = 256;
            let bounds = detector.detect_batch(&sigs);
            let rows: Vec<_> = sigs
                .into_iter()
                .zip(bounds)
                .zip(items)
                .map(|((sig, b), item)| (sig, b, item))
                .collect();

            rows.into_par_iter().chunks(CRF_CHUNK).for_each(|chunk| {
                let n = chunk.len();
                let mut scratch = CrfScratch::new();
                for (signal, (_s, adapter_end), (read, chunks, run_infos)) in chunk {
                    let (barcode, conf) = (|| {
                        let adc = signal.as_ref()?;
                        // The detector reports `adapter_end` as an index into
                        // the decoded prefix, which is what `prep` wants.
                        let pa = adc_to_pa(adc, read.calibration_offset, read.calibration_scale);
                        let window = meta.prep(&pa, adapter_end)?;
                        let seq = head
                            .encoder
                            .basecall_prepped(&window, &mut scratch)
                            .inspect_err(|e| tracing::warn!("encoder: {e}"))
                            .ok()?;
                        let m = head.refs.match_sequence(seq.as_bytes())?;
                        // Same gate as `demux basecall --min-margin`, including
                        // its treatment of a single reference: with no runner-up
                        // there is no margin to test, so the call stands.
                        if !m.margin.is_none_or(|v| v >= head.min_margin) {
                            return None;
                        }
                        // Confidence is the margin, matching `demux basecall`
                        // and `eval_recovery.py`. A lone reference reports 0
                        // rather than a fabricated distance.
                        Some((
                            head.refs.name(m.index).to_string(),
                            f64::from(m.margin.unwrap_or(0)),
                        ))
                    })()
                    .unwrap_or_else(|| (UNCLASSIFIED.to_string(), 0.0));

                    route(
                        routers,
                        class_tx,
                        read.for_writing(read.run_info_index),
                        barcode,
                        chunks,
                        run_infos,
                        conf,
                    );
                }
                pb.inc(n as u64);
            });
        },
    )
}

/// ADC counts to picoamps, fused so there is one rounding step.
///
/// Matches `escapepod_python::adc_to_pa` and `demux basecall`'s own conversion;
/// the reference `pod5` package computes it unfused, which differs by ~1 ulp —
/// thousands of times below the standardisation scale and irrelevant to the
/// decode.
#[cfg(feature = "crf-decode")]
fn adc_to_pa(raw: &[i16], offset: f32, scale: f32) -> Vec<f32> {
    raw.iter().map(|&v| (f32::from(v) + offset) * scale).collect()
}

/// GBM counterpart to [`classify_one_cpu`]: fingerprint → GBM tree walk from a
/// decoded signal and precomputed boundaries. Returns `(barcode, confidence)`;
/// unfingerprintable reads route to `unclassified` (matching the SVM path).
/// Fingerprint one read for the GBM path. `None` when the adapter window is
/// empty or the read cannot produce a full-width fingerprint.
fn fingerprint_for_gbm(
    read: &ReadData,
    signal: &[i16],
    s: usize,
    e: usize,
    fp: FpParams,
) -> Option<Vec<f64>> {
    if e <= s {
        return None;
    }
    extract_fingerprint_from_signal(
        signal,
        s,
        e,
        fp.num_segments,
        fp.window_width,
        NormMethod::ZScore,
        read.read_id,
        fp.min_separation,
        fp.keep_last,
        false,
    )
    .map(|f| f.values)
}

/// GPU producer: parallel CPU prep (decode + detect + fingerprint) feeds a
/// dedicated GPU classify thread through a bounded channel, so the GPU is kept
/// continuously fed (double-buffered) rather than going idle between batches.
#[cfg(feature = "gpu")]
fn produce_gpu(
    args: &RunArgs,
    detector: &Detector,
    model: &DtwSvmModel,
    fp: FpParams,
    routers: &Routers,
    class_tx: Option<&SyncSender<(Uuid, String, f64)>>,
    pb: &indicatif::ProgressBar,
) -> anyhow::Result<()> {
    use escapepod_signal::dtw::GpuDtwContext;

    type Meta = (ReadData, Vec<CompressedSignalChunk>, Arc<Vec<RunInfoData>>);
    type Block = (Vec<Vec<f64>>, Vec<Meta>);
    const GPU_BATCH: usize = 65_536;

    // Bounded so CPU prep stays ~2 blocks ahead of the GPU (double-buffering)
    // without unbounded memory.
    let (block_tx, block_rx) = std::sync::mpsc::sync_channel::<Block>(2);

    // GPU classify thread: pull blocks, classify, route. Runs concurrently with
    // CPU prep.
    std::thread::scope(|scope| -> anyhow::Result<()> {
        let model_ref = &*model;
        let routers_ref = routers;
        let class_ref = class_tx;
        let gpu = scope.spawn(move || -> anyhow::Result<()> {
            let ctx = GpuDtwContext::new().map_err(|e| anyhow::anyhow!("GPU init: {e}"))?;
            for (fps, metas) in block_rx.iter() {
                let results = escapepod_demux::classify_with_svm_batch_gpu_with_ctx(
                    &ctx,
                    model_ref,
                    &fps,
                    escapepod_demux::DEFAULT_GPU_CHUNK_CELLS,
                )
                .map_err(|e| anyhow::anyhow!("GPU classify: {e}"))?;
                for ((read, chunks, run_infos), (_p, result)) in metas.into_iter().zip(results) {
                    route(
                        routers_ref,
                        class_ref,
                        read,
                        barcode_label(result.predicted_barcode),
                        chunks,
                        run_infos,
                        result.confidence,
                    );
                }
            }
            Ok(())
        });

        // CPU prep (parallel) — accumulate fingerprint blocks and push them.
        let mut fps: Vec<Vec<f64>> = Vec::with_capacity(GPU_BATCH);
        let mut metas: Vec<Meta> = Vec::with_capacity(GPU_BATCH);
        for path in &args.input {
            let reader = Reader::open(path)?;
            let run_infos = Arc::new(reader.run_infos().to_vec());
            for batch in reader.read_batches()? {
                let batch = batch?;
                let view = ReadsBatchView::new(&batch, false)?;
                let reads: Vec<ReadData> = (0..view.num_rows())
                    .filter_map(|row| view.read(row).ok())
                    .filter(|r| !r.signal_rows.is_empty())
                    .collect();

                // One sequential sweep pulls this read-batch's compressed signal
                // (see produce_cpu for why single-stream I/O beats per-worker
                // faulting on a network FS, #72), then the CPU prep parallelizes
                // over the in-memory chunks.
                let keyed: Vec<(usize, Vec<u64>)> = reads
                    .iter()
                    .enumerate()
                    .map(|(i, r)| (i, r.signal_rows.clone()))
                    .collect();
                let bulk = reader.get_compressed_signal_bulk(&keyed)?;

                type Prepped = (ReadData, Option<Vec<f64>>, Vec<CompressedSignalChunk>);
                // Windowed: decode once, batch-detect (GPU CNN = one call/window;
                // LLR / CPU-CNN = parallel per read), then parallel fingerprint.
                let decode_to = detector.signal_decode_bound();
                for window in bulk.chunks(DETECT_WINDOW) {
                    let signals: Vec<Option<Vec<i16>>> = window
                        .par_iter()
                        .map(|(_, chunks)| super::utils::decode_chunks_to(chunks, decode_to))
                        .collect();
                    let bounds = detector.detect_batch(&signals);
                    let prepped: Vec<Option<Prepped>> = window
                        .par_iter()
                        .enumerate()
                        .map(|(k, (i, chunks))| -> Option<Prepped> {
                            let read = &reads[*i];
                            let signal = signals[k].as_ref()?;
                            let (s, e) = bounds[k];
                            let features = if e > s {
                                extract_fingerprint_from_signal(
                                    signal,
                                    s,
                                    e,
                                    fp.num_segments,
                                    fp.window_width,
                                    NormMethod::ZScore,
                                    read.read_id,
                                    fp.min_separation,
                                    fp.keep_last,
                                    false,
                                )
                                .map(|f| f.values)
                            } else {
                                None
                            };
                            Some((
                                read.for_writing(read.run_info_index),
                                features,
                                chunks.clone(),
                            ))
                        })
                        .collect();
                    pb.inc(window.len() as u64);

                    for (read, fp_opt, chunks) in prepped.into_iter().flatten() {
                        match fp_opt {
                            Some(values) => {
                                fps.push(values);
                                metas.push((read, chunks, run_infos.clone()));
                            }
                            None => route(
                                routers,
                                class_tx,
                                read,
                                UNCLASSIFIED.to_string(),
                                chunks,
                                run_infos.clone(),
                                0.0,
                            ),
                        }
                    }
                    if fps.len() >= GPU_BATCH {
                        let block = (std::mem::take(&mut fps), std::mem::take(&mut metas));
                        block_tx
                            .send(block)
                            .map_err(|_| anyhow::anyhow!("GPU thread hung up"))?;
                    }
                }
            }
        }
        if !fps.is_empty() {
            let _ = block_tx.send((fps, metas));
        }
        drop(block_tx);
        gpu.join()
            .map_err(|e| anyhow::anyhow!("GPU thread panicked: {e:?}"))?
    })
}

/// Per-barcode writer thread: lazily create the output POD5 on the first read
/// (so empty barcodes produce no file), block-copy each read, remap run_info.
fn writer_thread(
    rx: std::sync::mpsc::Receiver<Routed>,
    path: &Path,
    predefined: PredefinedDictionaries,
) -> anyhow::Result<usize> {
    let mut writer: Option<Writer> = None;
    let mut ri_index: HashMap<String, u32> = HashMap::new();
    let mut count = 0usize;
    for Routed {
        read,
        chunks,
        run_infos,
    } in rx.iter()
    {
        let w = match writer.as_mut() {
            Some(w) => w,
            None => {
                // Match `filter`/`repack` rather than taking the writer's
                // conservative 100/1000 defaults. At 100 the writer flushes a
                // signal batch every 100 reads, rebuilding the Arrow schema and
                // emitting an IPC message + footer entry each time — and every
                // downstream reader then pays per-batch parse cost over that
                // many batches for the life of the file.
                let opts = WriterOptions {
                    predefined_dictionaries: Some(predefined.clone()),
                    signal_batch_size: 1_000,
                    read_batch_size: 10_000,
                    ..Default::default()
                };
                writer = Some(Writer::create(path, opts)?);
                writer.as_mut().unwrap()
            }
        };
        let src = &run_infos[read.run_info_index as usize];
        let widx = match ri_index.get(&src.acquisition_id) {
            Some(&i) => i,
            None => {
                let i = w.add_run_info(src.clone())?;
                ri_index.insert(src.acquisition_id.clone(), i);
                i
            }
        };
        w.add_read_with_compressed_signal(read.for_writing(widx), &chunks)?;
        count += 1;
    }
    if let Some(w) = writer {
        w.finish()?;
    }
    Ok(count)
}

/// Optional classifications-CSV writer thread.
#[allow(clippy::type_complexity)]
fn spawn_class_writer(
    path: Option<&Path>,
) -> anyhow::Result<(
    Option<SyncSender<(Uuid, String, f64)>>,
    Option<std::thread::JoinHandle<anyhow::Result<()>>>,
)> {
    let Some(path) = path else {
        return Ok((None, None));
    };
    let (tx, rx) = std::sync::mpsc::sync_channel::<(Uuid, String, f64)>(16_384);
    let path = path.to_path_buf();
    let handle = std::thread::spawn(move || -> anyhow::Result<()> {
        use std::io::Write;
        let file = std::fs::File::create(&path)?;
        let mut w = std::io::BufWriter::with_capacity(256 * 1024, file);
        writeln!(w, "read_id,barcode,confidence")?;
        for (read_id, barcode, conf) in rx.iter() {
            writeln!(w, "{read_id},{barcode},{conf:.6}")?;
        }
        w.flush()?;
        Ok(())
    });
    Ok((Some(tx), Some(handle)))
}

/// Build the adapter detector: the model bundle's pinned choice, or an explicit
/// `--method`, and an error rather than a silent guess when neither says.
///
/// `pin` is the detector a CRF bundle declares itself calibrated against
/// (`(method, Some(onnx_path))`). The training window is defined relative to
/// that detector's `adapter_end`, so a pin is a hard requirement rather than a
/// preference.
///
/// LLR is never inferred. It costs 17.2 points of barcode recall against the
/// same classifier and fails silently (escapepod-models#16), so it has to be
/// asked for by name. Explicit `--method` overrides a pin — that has to stay
/// possible to evaluate a new boundary model — except that a bundle pinning
/// `cnn` refuses the downgrade, which is #16's runtime guard.
fn build_detector(args: &RunArgs, pin: Option<(&str, Option<PathBuf>)>) -> anyhow::Result<Detector> {
    let pinned_method = pin.as_ref().map(|(m, _)| *m);
    let pinned_onnx = pin.and_then(|(_, p)| p);
    let method = match (args.method.as_deref(), pinned_method) {
        (Some("llr"), Some("cnn")) => anyhow::bail!(
            "this model is calibrated against CNN adapter boundaries and refuses \
             `--method llr`: LLR costs 17.2 points of barcode recall on the same \
             classifier (0.9928 -> 0.8196) and fails silently. Drop `--method llr`, \
             or use a model that does not pin a detector."
        ),
        (Some(m), _) => m,
        (None, Some(m)) => m,
        (None, None) => anyhow::bail!(
            "--method {{cnn,llr}} is required: this model does not pin a boundary \
             detector, and LLR is never chosen for you. Use `--method cnn --cnn-model \
             <FILE>` for the accuracy the shipped barcode models were measured at, or \
             `--method llr` to opt into the classical detector (17.2 points worse on \
             barcode recall — escapepod-models#16)."
        ),
    };
    match method {
        "llr" => Ok(Detector::Llr {
            min_adapter: args.min_adapter,
            border_trim: args.border_trim,
            downscale: args.downscale.max(1),
        }),
        "cnn" => {
            #[cfg(feature = "cnn-detect")]
            {
                let path = args
                    .cnn_model
                    .as_ref()
                    .or(pinned_onnx.as_ref())
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "--method cnn requires --cnn-model <FILE> (this model bundle \
                             does not ship a boundary model)"
                        )
                    })?;
                // `--gpu` with `--method cnn` runs detection on the GPU (one
                // batched onnxruntime call per block) when built with cnn-gpu.
                #[cfg(feature = "cnn-gpu")]
                if args.gpu {
                    return Ok(Detector::CnnGpu(Box::new(
                        escapepod_demux::AdapterCnnGpu::load_with_threads(
                            path,
                            crate::threads::width(),
                        )
                        .map_err(|e| anyhow::anyhow!("loading CNN model on GPU: {e}"))?,
                    )));
                }
                Ok(Detector::Cnn(Box::new(
                    escapepod_demux::AdapterCnn::load(path)
                        .map_err(|e| anyhow::anyhow!("loading CNN model: {e}"))?,
                )))
            }
            #[cfg(not(feature = "cnn-detect"))]
            {
                let _ = pinned_onnx;
                anyhow::bail!("--method cnn requires a build with `--features cnn-detect`")
            }
        }
        other => anyhow::bail!("unknown --method `{other}`; expected `llr` or `cnn`"),
    }
}

#[derive(Default)]
struct DemuxSummary {
    per_barcode: Vec<(String, usize)>,
}

fn print_summary(summary: &DemuxSummary) {
    // Styled multi-line report; gate on verbosity instead of per-line tracing events.
    if tracing::enabled!(tracing::Level::INFO) {
        let total: usize = summary.per_barcode.iter().map(|(_, n)| n).sum();
        println!("\n{}", style::action("Demux summary:"));
        for (barcode, n) in &summary.per_barcode {
            println!("  {} {}", style::label(barcode), style::count(*n));
        }
        println!(
            "{} {} reads across {} barcode file(s)",
            style::action("Total:"),
            style::count(total),
            summary.per_barcode.len()
        );
    }
}

#[cfg(all(test, feature = "crf-decode"))]
mod tests {
    use super::crf_bundle_dir;

    /// `--model` sniffing must not depend on the extension or the file name
    /// alone: a CRF bundle is identified by its sidecar's `format` key, so a
    /// classifier JSON living next to one, or a directory without a sidecar,
    /// still routes to `load_any_model`.
    #[test]
    fn crf_bundle_detected_by_format_key_not_by_name() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();

        // Not a bundle: no metadata.json at all.
        assert!(crf_bundle_dir(root).is_none());

        // Not a bundle: metadata.json exists but declares something else.
        let meta = root.join("metadata.json");
        std::fs::write(&meta, r#"{"format":"something-else/1"}"#).unwrap();
        assert!(crf_bundle_dir(root).is_none());
        assert!(crf_bundle_dir(&meta).is_none());

        // A bundle: recognised via the directory and via the sidecar itself,
        // and both resolve to the directory the ONNX is loaded from.
        std::fs::write(&meta, r#"{"format":"escapepod-crf-encoder/1"}"#).unwrap();
        assert_eq!(crf_bundle_dir(root).as_deref(), Some(root));
        assert_eq!(crf_bundle_dir(&meta).as_deref(), Some(root));

        // A classifier JSON is never mistaken for a bundle, even beside one.
        let svm = root.join("model.json");
        std::fs::write(&svm, r#"{"label_mapper":{}}"#).unwrap();
        assert!(crf_bundle_dir(&svm).is_none());

        // Malformed sidecar: fall through to the JSON loader rather than
        // failing here, so the error the user sees comes from the real parse.
        std::fs::write(&meta, "not json").unwrap();
        assert!(crf_bundle_dir(root).is_none());
    }
}
