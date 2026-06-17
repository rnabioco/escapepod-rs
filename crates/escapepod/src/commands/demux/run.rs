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

use super::utils::configure_thread_pool;
use crate::progress::create_progress_bar;
use crate::style;
use escapepod_demux::{
    AnyModel, DtwSvmModel, GbmModel, GbmPredictor, SvmPredictor, SvmWorkspace,
    extract_fingerprint_from_signal, load_any_model,
};
use escapepod_signal::dtw::NormMethod;
use escapepod_signal::segmentation::{detect_adapter, downscale, normalize_signal};
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

    /// Trained classifier JSON — DTW-SVM (`demux train-svm` / converted
    /// WarpDemuX) or native GBM tree ensemble. Auto-detected by JSON shape.
    #[arg(long, value_name = "FILE")]
    pub model: Option<PathBuf>,

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

    /// Adapter detection method: `llr` (default) or `cnn`.
    #[arg(
        long,
        default_value = "llr",
        value_name = "{llr,cnn}",
        help_heading = "Advanced Options"
    )]
    pub method: String,

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

    /// Use the GPU where this pipeline supports it: batched DTW-SVM classify
    /// (`--features gpu`) and/or batched CNN adapter detection with
    /// `--method cnn` (`--features cnn-gpu`). CPU prep stays parallel and feeds
    /// the GPU; CPU falls back automatically for stages without a GPU path
    /// (e.g. GBM classify).
    #[cfg(any(feature = "gpu", feature = "cnn-gpu"))]
    #[arg(long)]
    pub gpu: bool,

    /// Number of threads (default: all CPUs)
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

impl Detector {
    fn detect(&self, signal: &[i16]) -> (usize, usize) {
        match self {
            Detector::Llr {
                min_adapter,
                border_trim,
                downscale: ds,
            } => {
                let normalized = normalize_signal(signal);
                let (processed, scale) = if *ds > 1 {
                    let trunc = (normalized.len() / ds) * ds;
                    (downscale(&normalized[..trunc], *ds), *ds)
                } else {
                    (normalized, 1)
                };
                let (s, e) = detect_adapter(
                    &processed,
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
    /// in parallel. Bit-identical to calling [`Self::detect`] on each signal.
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
            .map(|s| s.as_ref().map_or((0, 0), |v| self.detect(v)))
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

/// Either classifier head the fused pipeline can drive. Detect + fingerprint are
/// model-agnostic; only the per-read classify differs — DTW-SVM (with an optional
/// GPU DTW path) or the CPU-only GBM tree walk.
enum ClassifyModel {
    Svm(DtwSvmModel),
    Gbm(GbmModel),
}

impl ClassifyModel {
    /// Class-index → barcode-id map (same shape on both heads), for the output
    /// barcode set.
    fn label_mapper(&self) -> &HashMap<usize, i32> {
        match self {
            ClassifyModel::Svm(m) => &m.label_mapper,
            ClassifyModel::Gbm(m) => &m.label_mapper,
        }
    }
}

pub fn run(args: RunArgs) -> anyhow::Result<()> {
    use crate::commands::profile::PhaseTimer;
    let mut timer = PhaseTimer::new();
    timer.phase("Fused demux");
    let profile = args.profile;

    // Validate the fused-pipeline args here (not via clap `required`) so the
    // advanced subcommands aren't forced to supply them.
    if args.input.is_empty() {
        anyhow::bail!("no input POD5 file(s) given");
    }
    let model_path = args
        .model
        .clone()
        .ok_or_else(|| anyhow::anyhow!("--model <FILE> is required"))?;
    let output_dir = args
        .output_dir
        .clone()
        .ok_or_else(|| anyhow::anyhow!("-d/--output-dir <DIR> is required"))?;

    // The fused pipeline supports both classifier heads: DTW-SVM (with an
    // optional GPU DTW path) and the native GBM tree ensemble (CPU-only). Only
    // the legacy reference-bank WarpDemux JSON is rejected here.
    let model = match load_any_model(&model_path)? {
        AnyModel::Svm(m) => ClassifyModel::Svm(m),
        AnyModel::Gbm(m) => ClassifyModel::Gbm(m),
        AnyModel::WarpDemux(_) => anyhow::bail!(
            "`demux` needs an SVM or GBM model (DtwSvmModel / converted WarpDemuX \
             / native GBM). The reference-CSV path is only on `demux classify --reference`."
        ),
    };
    let detector = build_detector(&args)?;
    let fp = FpParams::default();

    configure_thread_pool(args.threads);
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
    let mut routers: Routers = HashMap::new();
    let mut writer_handles: Vec<(String, std::thread::JoinHandle<anyhow::Result<usize>>)> =
        Vec::new();
    for bc in barcode_set(model.label_mapper()) {
        let (tx, rx) = std::sync::mpsc::sync_channel::<Routed>(4096);
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
            if args.gpu && args.method != "cnn" {
                tracing::warn!(
                    "--gpu has no effect here: GBM classify is CPU-only and \
                     `--method {}` detection is CPU-only (use `--method cnn` for \
                     GPU adapter detection).",
                    args.method
                );
            }
            produce_cpu_gbm(&args, &detector, gbm, fp, &routers, class_tx.as_ref(), &pb)
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

/// Reads accumulated into one detect+classify block. POD5 Arrow read-batches
/// are small (~1000 reads), so detecting per batch makes GPU CNN fire many tiny
/// calls; accumulating across batches into a large block groups far more
/// same-length reads per onnxruntime call. Host cost is one block of decoded
/// signal (bounded; trivial relative to typical node RAM). The on-device batch
/// is separately capped by `gpu_batch_elems`.
const DETECT_WINDOW: usize = 65_536;

/// One read's context carried alongside its decoded signal through a block:
/// the writable read record, its compressed chunks (for the block-level write),
/// and its run-info table.
type BlockItem = (ReadData, Vec<CompressedSignalChunk>, Arc<Vec<RunInfoData>>);

/// Stream reads through the fused pipeline in large blocks: per Arrow batch do
/// the single-stream, ascending-order signal sweep (#72) and decode in
/// parallel, accumulate across batches up to [`DETECT_WINDOW`] reads, then hand
/// each full block (decoded signals + per-read context, aligned) to
/// `process_block`. Accumulating across the file's small Arrow batches is what
/// lets GPU detection batch many reads per call.
fn drive_blocks(
    input: &[std::path::PathBuf],
    decode_to: Option<usize>,
    mut process_block: impl FnMut(Vec<Option<Vec<i16>>>, Vec<BlockItem>),
) -> anyhow::Result<()> {
    let mut sigs: Vec<Option<Vec<i16>>> = Vec::with_capacity(DETECT_WINDOW);
    let mut items: Vec<BlockItem> = Vec::with_capacity(DETECT_WINDOW);
    for path in input {
        let reader = Reader::open(path)?;
        let run_infos = Arc::new(reader.run_infos().to_vec());
        for batch in reader.read_batches()? {
            let batch = batch?;
            let view = ReadsBatchView::new(&batch, false)?;
            let reads: Vec<ReadData> = (0..view.num_rows())
                .filter_map(|row| view.read(row).ok())
                .filter(|r| !r.signal_rows.is_empty())
                .collect();
            // One sequential, ascending-order sweep pulls this batch's signal
            // (grouped by Arrow record-batch — see #72), then decode in parallel.
            let keyed: Vec<(usize, Vec<u64>)> = reads
                .iter()
                .enumerate()
                .map(|(i, r)| (i, r.signal_rows.clone()))
                .collect();
            let bulk = reader.get_compressed_signal_bulk(&keyed)?;
            let decoded: Vec<Option<Vec<i16>>> = bulk
                .par_iter()
                .map(|(_, chunks)| decode_chunks_to(chunks, decode_to))
                .collect();

            let mut reads_opt: Vec<Option<ReadData>> = reads.into_iter().map(Some).collect();
            for ((i, chunks), sig) in bulk.into_iter().zip(decoded) {
                let read = reads_opt[i].take().expect("each read consumed once");
                sigs.push(sig);
                items.push((read, chunks, run_infos.clone()));
            }
            if sigs.len() >= DETECT_WINDOW {
                process_block(std::mem::take(&mut sigs), std::mem::take(&mut items));
            }
        }
    }
    if !sigs.is_empty() {
        process_block(sigs, items);
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

/// Decode (decompress) a read's in-memory compressed signal chunks into a
/// single sample buffer, concatenated in chunk order.
fn decode_chunks(chunks: &[CompressedSignalChunk]) -> Option<Vec<i16>> {
    let total: usize = chunks.iter().map(|c| c.samples as usize).sum();
    let mut signal = Vec::with_capacity(total);
    for c in chunks {
        let decoded =
            escapepod_signal::pod5::compression::decompress_signal(&c.data, c.samples as usize)
                .ok()?;
        signal.extend_from_slice(&decoded);
    }
    Some(signal)
}

/// Like [`decode_chunks`] but, when `decode_to` is `Some(max)`, decompresses
/// only the first `max` samples — decoding just the needed prefix of the chunk
/// that crosses the boundary (the rest of the ZSTD stream is skipped). For long
/// reads (mRNA) where the adapter detector only looks at the first
/// `max_obs_trace` samples, this avoids decompressing the whole transcript.
/// `None` decodes the full signal (e.g. LLR, which normalizes over the read).
fn decode_chunks_to(
    chunks: &[CompressedSignalChunk],
    decode_to: Option<usize>,
) -> Option<Vec<i16>> {
    let Some(max) = decode_to else {
        return decode_chunks(chunks);
    };
    let mut signal = Vec::with_capacity(max);
    let mut remaining = max;
    for c in chunks {
        if remaining == 0 {
            break;
        }
        let cs = c.samples as usize;
        let take = cs.min(remaining);
        let decoded = if take == cs {
            escapepod_signal::pod5::compression::decompress_signal(&c.data, cs).ok()?
        } else {
            escapepod_signal::pod5::compression::decompress_signal_prefix(&c.data, cs, take).ok()?
        };
        signal.extend_from_slice(&decoded);
        remaining -= take;
    }
    Some(signal)
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
            // Batch-detect the whole block, then GBM-classify reusing each decoded
            // signal. GBM is `Sync` and read-only, so a plain parallel walk (no
            // per-worker workspace like the SVM path).
            let bounds = detector.detect_batch(&sigs);
            sigs.into_par_iter().zip(bounds).zip(items).for_each(
                |((signal, (s, e)), (read, chunks, run_infos))| {
                    if let Some(signal) = signal {
                        let (barcode, conf) =
                            classify_one_cpu_gbm(&read, &signal, s, e, &predictor, fp);
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

/// GBM counterpart to [`classify_one_cpu`]: fingerprint → GBM tree walk from a
/// decoded signal and precomputed boundaries. Returns `(barcode, confidence)`;
/// unfingerprintable reads route to `unclassified` (matching the SVM path).
fn classify_one_cpu_gbm(
    read: &ReadData,
    signal: &[i16],
    s: usize,
    e: usize,
    predictor: &GbmPredictor,
    fp: FpParams,
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
    match predictor.predict(&features.values) {
        Ok((_probs, result)) => (barcode_label(result.predicted_barcode), result.confidence),
        Err(_) => (UNCLASSIFIED.to_string(), 0.0),
    }
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
                        .map(|(_, chunks)| decode_chunks_to(chunks, decode_to))
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
                let opts = WriterOptions {
                    predefined_dictionaries: Some(predefined.clone()),
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

fn build_detector(args: &RunArgs) -> anyhow::Result<Detector> {
    match args.method.as_str() {
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
                    .ok_or_else(|| anyhow::anyhow!("--method cnn requires --cnn-model <FILE>"))?;
                // `--gpu` with `--method cnn` runs detection on the GPU (one
                // batched onnxruntime call per block) when built with cnn-gpu.
                #[cfg(feature = "cnn-gpu")]
                if args.gpu {
                    return Ok(Detector::CnnGpu(Box::new(
                        escapepod_demux::AdapterCnnGpu::load(path)
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
