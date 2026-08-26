//! Fused, streaming demux pipeline: decode each read's signal **once**, run
//! detect → fingerprint → classify in a single pass, and route the read
//! (block-level compressed copy, no re-decode/re-compress) into its barcode's
//! output POD5. No intermediate boundaries/fingerprints/classifications files
//! are written unless explicitly requested (`--classifications`).
//!
//! Pipeline (all stages overlap):
//!   A. rayon pool decodes + detects + fingerprints reads in parallel (per
//!      Arrow batch, bounded memory).
//!   B. classify — CPU per-read (in stage A), or, on the GPU, a dedicated GPU
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
#[cfg(feature = "gpu")]
use escapepod_demux::crf::CrfEncoderGpu;
#[cfg(feature = "crf-decode")]
use escapepod_demux::crf::{
    BarcodeRefs, CrfEncoder, CrfError, CrfScratch, RefChains, ScoredDecode,
};
use escapepod_demux::{
    AnyModel, DtwSvmModel, GbmModel, GbmPredictor, SvmPredictor, SvmWorkspace,
    extract_fingerprint_from_signal, load_any_model,
};
use escapepod_signal::dtw::NormMethod;
use escapepod_signal::operations::{ColumnValues, ColumnWrite};
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

    /// Score every reference against the lattice, giving each read a real
    /// `log P(barcode | signal)` (CRF head only).
    ///
    /// `confidence` is the edit-distance margin to the runner-up, which on a
    /// designed panel measures how far apart the references are, not how sure
    /// the model is: over one production 16-plex flowcell 99% of classified
    /// reads take one of three values, so `--min-margin` cannot trade recall
    /// for precision (#241). This computes what the lattice actually thought,
    /// which is continuous, and adds it to `--classifications`.
    ///
    /// Costs +3.6% on this pipeline; on the GPU more, because the
    /// constrained scan reads the raw scores and so brings the decode back to
    /// the host while the encoder stays on the device.
    #[cfg(feature = "crf-decode")]
    #[arg(long, help_heading = "Advanced Options")]
    pub ref_scores: bool,

    /// Call a read `unclassified` when the lattice's log-odds for the called
    /// barcode against its best alternative are below this, in nats
    /// (implies `--ref-scores`).
    ///
    /// The dial `--min-margin` is not: it is continuous, and it goes negative
    /// when the lattice prefers a different reference, so any positive
    /// threshold drops those too. 0.7 nats is 2:1 odds, 2.3 is 10:1, 4.6 is
    /// 100:1.
    #[cfg(feature = "crf-decode")]
    #[arg(long, value_name = "NATS", help_heading = "Advanced Options")]
    pub min_crf_margin: Option<f32>,

    /// Call a read `unclassified` when `P(called barcode | signal)` is below
    /// this (implies `--ref-scores`).
    ///
    /// Asks whether the model is confident in absolute terms, where
    /// `--min-crf-margin` asks whether it can tell the call from the next
    /// reference. A read can be sure it is not any of the other barcodes and
    /// still put little mass on any reference at all.
    #[cfg(feature = "crf-decode")]
    #[arg(long, value_name = "P", help_heading = "Advanced Options")]
    pub min_crf_prob: Option<f32>,

    /// Overrule the bundle's declared `boundary.margin` — the samples of
    /// `adapter_end` a read needs beyond the model's `chunk` before it decodes
    /// (CRF head only). Unset uses the bundle, which falls back to 200.
    ///
    /// The margin records how the training corpus was FILTERED, not what the
    /// encoder needs: `adapter_end >= chunk` already yields a full window, and
    /// the extra samples exist only because `extract_chunks.py` demanded them.
    /// Reads in `[chunk, chunk + margin)` are therefore dropped undecoded even
    /// though they are decodable. Lowering it recovers them at the cost of a
    /// window reaching into the read's opening samples, which the model never
    /// saw in training. Measured on a 1.0M-read RNA004 nbc16 run: 0 recovered
    /// 36,921 reads (demux yield 85.44% -> 89.13%), all decoding at median edit
    /// distance 0 with 98.3% within 2 edits — cleaner than the reads that
    /// already passed.
    ///
    /// This is an escape hatch for evaluating a change before baking it into an
    /// export. The bundle should state the margin its corpus was built with;
    /// once measured, set `boundary.margin` there rather than passing this on
    /// every run.
    #[cfg(feature = "crf-decode")]
    #[arg(long, value_name = "N", help_heading = "Advanced Options")]
    pub boundary_margin: Option<usize>,

    /// Overrule the bundle's declared `boundary.clamp_max_shift`: decode a read
    /// whose adapter ends before the model's `chunk` from the window `[0,
    /// chunk]`, provided `chunk - adapter_end` is at most N. 0 disables it.
    ///
    /// `--boundary-margin` cannot reach these reads. Their window would start
    /// before sample 0, so there is nothing to relax — the signal genuinely runs
    /// out. Clamping keeps the window width and anchors it at the read start
    /// instead, sliding `chunk - adapter_end` samples of downstream signal into
    /// the tail. Such reads are truncated mid-adapter, so part of the barcode is
    /// already gone; the bound is how much of that you accept.
    ///
    /// Measured on the RNA004 nbc16 run: known-good reads deliberately slid
    /// forward still call the same barcode 98.6% of the time at shift 0 and
    /// 93.5% at 500. Applying it to the real `adapter_end` 2,500-2,999 band
    /// recovered 42,404 reads, all decoding, median edit distance 0, but with
    /// agreement falling 97.4% -> 92.9% within 2 edits across that range and the
    /// share still aligning to a tRNA falling 78.5% -> 50.0%.
    #[cfg(feature = "crf-decode")]
    #[arg(long, value_name = "N", help_heading = "Advanced Options")]
    pub clamp_max_shift: Option<usize>,

    /// Describe the model and exit: identity, signal geometry, bundled
    /// references, pinned boundary detector, published metrics, and the exact
    /// command line it needs. Reads no POD5, so it is safe to run against a
    /// model before trusting it.
    #[arg(long)]
    pub info: bool,

    /// Output directory for the per-barcode demultiplexed POD5 files
    /// (optional with --annotate: sidecar-only, no split files written)
    #[arg(short = 'd', long, value_name = "DIR")]
    pub output_dir: Option<PathBuf>,

    /// Record each read's barcode assignment in the input's .p5s sidecar
    /// (`escpod annotate` format). With no -d/--output-dir this is the only
    /// output: demux without duplicating the POD5, split on demand later
    /// with `demux split --sidecar`
    #[arg(long)]
    pub annotate: bool,

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

    /// Where this pipeline's GPU-capable stages run — `auto` (default), `cpu`,
    /// or `gpu`. CPU prep stays parallel and feeds the device either way.
    ///
    /// Under `auto` the two stages that measurably win go to the GPU when one is
    /// usable: CNN adapter detection with `--method cnn` (~7x) and
    /// CTC-CRF encoder inference (~4x). With a CRF bundle the encoder
    /// is the case that pays off most — it is ~91% of that head's CPU cost
    /// (13.9 ms/read against a 1.19 ms AVX-512 lattice decode), so leaving it on
    /// the CPU leaves the device idle even with `--method cnn` detecting there.
    ///
    /// DTW-SVM classify stays on the **CPU** under `auto` because the CPU
    /// is faster: 113 s on 64 cores vs 132 s on an A30 for 1.22M reads, plus
    /// ~2.2 GB more RSS. `--device gpu` opts into it anyway, which is worth doing
    /// only when cores are scarce. GBM classify has no GPU path at all.
    ///
    /// `--device gpu` is a requirement: a missing Cargo feature, an absent
    /// device, or an onnxruntime that cannot register its CUDA execution
    /// provider all fail the run instead of falling back silently.
    #[command(flatten)]
    pub device: crate::device::DeviceArgs,

    /// Number of threads for parallel processing (default: 16, or all available CPUs if fewer)
    #[arg(short = 't', long, visible_short_alias = 'j', value_name = "N")]
    pub threads: Option<usize>,

    /// Print per-phase timing breakdown after completion
    #[arg(long)]
    pub profile: bool,
}

/// Owns `T` and, when `leak` is set, forgets it at scope exit instead of
/// dropping it.
///
/// Exists for one upstream bug: onnxruntime's CUDA provider reads freed memory
/// during onnxruntime's *own* at-exit teardown (pykeio/ort#609), and glibc
/// aborts on it with "corrupted double-linked list" — measured 5 runs in 10,
/// always after every read had already been written. `release_env_on_exit` calls
/// `ReleaseEnv` only once the last `Arc<Environment>` drops and every live
/// `Session` holds one, so keeping our sessions alive past `main` means the
/// faulty path never runs.
///
/// The reason this is a guard and not a `mem::forget` at the end of the happy
/// path: `run` is full of `?`, so a trailing forget is skipped on precisely the
/// error paths — where an ordinary, reportable failure would then be masked by
/// an exit-134 abort. `Drop` runs on every path.
///
/// Narrow by construction: `leak` is false unless a GPU path actually created
/// ORT sessions, our own destructors elsewhere still run, and the process is
/// exiting anyway, so the kernel reclaims the memory. Drop it once `ort` /
/// onnxruntime ship a fix.
pub(super) struct LeakIf<T> {
    inner: Option<T>,
    leak: bool,
}

impl<T> LeakIf<T> {
    pub(super) fn new(inner: T, leak: bool) -> Self {
        Self {
            inner: Some(inner),
            leak,
        }
    }
}

impl<T> std::ops::Deref for LeakIf<T> {
    type Target = T;
    fn deref(&self) -> &T {
        self.inner.as_ref().expect("present until drop")
    }
}

impl<T> Drop for LeakIf<T> {
    fn drop(&mut self) {
        if let Some(inner) = self.inner.take()
            && self.leak
        {
            std::mem::forget(inner);
        }
    }
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
/// batched GPU CNN (`gpu`). The fused pipeline always detects through
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
    #[cfg(feature = "gpu")]
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
            #[cfg(feature = "gpu")]
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
        self.detect_batch_traced(signals, None)
    }

    /// [`detect_batch`](Self::detect_batch), splitting the GPU-CNN path's cost
    /// into its host and device halves.
    ///
    /// Worth separating because they call for opposite fixes: host-side prep
    /// scales with threads and pipelines against the device, while the batched
    /// onnxruntime call serialises on one session mutex and contends with the
    /// encoder for the same GPU. Detect is this pipeline's bottleneck, and
    /// without this split there is no way to tell which half to attack.
    // `split` is only read by the GPU-CNN branch, so it is unused whenever that
    // branch is compiled out. Plain atomics rather than a `&GpuTrace` so this
    // signature needs no feature gate of its own.
    #[cfg_attr(not(feature = "gpu"), allow(unused_variables))]
    fn detect_batch_traced(
        &self,
        signals: &[Option<Vec<i16>>],
        split: Option<(&std::sync::atomic::AtomicU64, &std::sync::atomic::AtomicU64)>,
    ) -> Vec<(usize, usize)> {
        #[cfg(feature = "gpu")]
        if let Detector::CnnGpu(gpu) = self {
            let cfg = gpu.config();
            let t_prep = std::time::Instant::now();
            let prepped: Vec<Option<escapepod_demux::PreppedWindow>> = signals
                .par_iter()
                .map(|s| {
                    s.as_ref().and_then(|v| {
                        let f: Vec<f32> = v.iter().map(|&x| x as f32).collect();
                        cfg.prep(&f)
                    })
                })
                .collect();
            if let Some((prep_ms, _)) = split {
                prep_ms.fetch_add(
                    t_prep.elapsed().as_millis() as u64,
                    std::sync::atomic::Ordering::Relaxed,
                );
            }
            let t_infer = std::time::Instant::now();
            let out: Vec<(usize, usize)> = gpu
                .detect_prepped(&prepped)
                .into_iter()
                .map(|r| (0usize, r.unwrap_or(0)))
                .collect();
            if let Some((_, infer_ms)) = split {
                infer_ms.fetch_add(
                    t_infer.elapsed().as_millis() as u64,
                    std::sync::atomic::Ordering::Relaxed,
                );
            }
            return out;
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
            #[cfg(feature = "gpu")]
            Detector::CnnGpu(g) => Some(g.config().max_obs_trace),
        }
    }

    /// Whether this detector holds an onnxruntime CUDA session.
    ///
    /// Answerable in every build, unlike `matches!(self, Detector::CnnGpu(_))`,
    /// whose variant does not exist without `gpu` — which is why the
    /// `LeakIf` predicate calls this instead of pattern-matching.
    fn on_gpu(&self) -> bool {
        #[cfg(feature = "gpu")]
        {
            matches!(self, Detector::CnnGpu(_))
        }
        #[cfg(not(feature = "gpu"))]
        {
            false
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

/// What a head decided about one read.
///
/// One type rather than a widening tuple: every head produces a barcode and a
/// confidence, and the CRF head under `--ref-scores` also produces what the
/// lattice thought, which has to reach the classifications writer through the
/// same channel.
struct Call {
    barcode: String,
    confidence: f64,
    crf: Option<CrfRowScores>,
}

impl Call {
    /// A read no head could place.
    fn unclassified() -> Self {
        Self {
            barcode: UNCLASSIFIED.to_string(),
            confidence: 0.0,
            crf: None,
        }
    }

    /// A call from a head that has no lattice to consult — the two fingerprint
    /// heads, and the CRF head without `--ref-scores`.
    fn scoreless(barcode: String, confidence: f64) -> Self {
        Self {
            barcode,
            confidence,
            crf: None,
        }
    }
}

/// The lattice's opinion of one read, as it reaches the classifications CSV.
///
/// `Copy` and reference *indices* rather than names: this crosses a channel
/// once per read, and at a million reads a per-read `String` for a name the
/// writer can look up is pure allocator traffic.
#[derive(Clone, Copy)]
#[cfg_attr(not(feature = "crf-decode"), allow(dead_code))]
struct CrfRowScores {
    /// `log P(called barcode | signal)`.
    logp: f32,
    /// The called barcode's log-odds in nats against its best alternative;
    /// `None` with a single reference. Negative when the lattice prefers
    /// something else.
    margin: Option<f32>,
    /// Index of the reference the lattice itself prefers.
    best: u32,
    /// Mean per-timestep log-posterior of the decoded path.
    mean_logpost: f32,
}

/// One classified read on its way to the classifications CSV and the
/// `--annotate` sidecar.
struct ClassRow {
    read_id: Uuid,
    barcode: String,
    confidence: f64,
    crf: Option<CrfRowScores>,
}

/// The classifications channel. Named because it appears in eight producer
/// signatures and widening it once should not mean editing all of them.
type ClassTx = SyncSender<ClassRow>;

/// Route one classified read to its barcode writer + (optionally) the
/// classifications CSV.
fn route(
    routers: &Routers,
    class_tx: Option<&ClassTx>,
    read: ReadData,
    call: Call,
    chunks: Vec<CompressedSignalChunk>,
    run_infos: Arc<Vec<RunInfoData>>,
) {
    let Call {
        barcode,
        confidence,
        crf,
    } = call;
    if let Some(ctx) = class_tx {
        let _ = ctx.send(ClassRow {
            read_id: read.read_id,
            barcode: barcode.clone(),
            confidence,
            crf,
        });
    }
    // Empty on sidecar-only runs (--annotate without -d): no POD5 leg.
    if let Some(tx) = routers.get(&barcode).or_else(|| routers.get(UNCLASSIFIED)) {
        let _ = tx.send(Routed {
            read,
            chunks,
            run_infos,
        });
    }
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
    encoder: CrfEncoderAny,
    refs: BarcodeRefs,
    min_margin: u32,
    /// The constrained lattices for `refs`, under `--ref-scores`. Built once
    /// per run: the structure depends on the panel and the model geometry, not
    /// on a read.
    chains: Option<RefChains>,
    min_crf_margin: Option<f32>,
    min_crf_prob: Option<f32>,
}

/// Where CRF encoder inference runs. The lattice decode is on the CPU either
/// way — see `escapepod_demux::crf::encoder_gpu` for why.
///
/// The two variants drive genuinely different producers rather than hiding
/// behind one `basecall` method: tract has no efficient batched LSTM, so the CPU
/// path interleaves prep/encode/match per read inside one rayon pass, while the
/// GPU path has to accumulate a batch before it can submit anything. Collapsing
/// them into a common batched interface would force the CPU path to materialise
/// a whole block of 3000-sample windows it has no use for.
#[cfg(feature = "crf-decode")]
enum CrfEncoderAny {
    Cpu(Box<CrfEncoder>),
    #[cfg(feature = "gpu")]
    Gpu(Box<CrfEncoderGpu>),
}

#[cfg(feature = "crf-decode")]
impl CrfEncoderAny {
    fn metadata(&self) -> &escapepod_demux::crf::CrfMetadata {
        match self {
            Self::Cpu(e) => e.metadata(),
            #[cfg(feature = "gpu")]
            Self::Gpu(e) => e.metadata(),
        }
    }

    fn set_boundary_margin(&mut self, margin: usize) {
        match self {
            Self::Cpu(e) => e.set_boundary_margin(margin),
            #[cfg(feature = "gpu")]
            Self::Gpu(e) => e.set_boundary_margin(margin),
        }
    }

    fn set_clamp_max_shift(&mut self, shift: usize) {
        match self {
            Self::Cpu(e) => e.set_clamp_max_shift(shift),
            #[cfg(feature = "gpu")]
            Self::Gpu(e) => e.set_clamp_max_shift(shift),
        }
    }

    fn ref_chains(&self, seqs: &[&[u8]]) -> Result<RefChains, CrfError> {
        match self {
            Self::Cpu(e) => e.ref_chains(seqs),
            #[cfg(feature = "gpu")]
            Self::Gpu(e) => e.ref_chains(seqs),
        }
    }
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
    // Resolved once, before anything is loaded: the deprecated `--gpu` alias
    // warns exactly once, and every stage below asks the same question of the
    // same answer.
    let device = args.device.resolve();
    let output_dir = match args.output_dir.clone() {
        Some(dir) => Some(dir),
        None if args.annotate => None, // sidecar-only: no split outputs
        None => anyhow::bail!(
            "-d/--output-dir <DIR> is required (or --annotate for sidecar-only demux)"
        ),
    };

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

    // Whether the CRF head put its encoder on the device. Only the CRF arm can
    // set it, and only one arm runs, so it is written at most once — but it has
    // to outlive the `match` because `LeakIf` below needs to know whether any
    // ORT session exists.
    #[cfg_attr(not(feature = "crf-decode"), allow(unused_mut, unused_variables))]
    let mut crf_encoder_on_gpu = false;

    let model = match crf_dir {
        #[cfg(feature = "crf-decode")]
        Some(dir) => {
            // The encoder is ~91% of this head's CPU cost, which is why `auto`
            // sends it to the device; the lattice decode's own placement is the
            // encoder's business. `--threads` bounds onnxruntime's intra-op pool,
            // which is otherwise spawned `available_parallelism()` wide on top of
            // rayon's.
            crf_encoder_on_gpu =
                crate::device::place_and_report(device, crate::device::Stage::CrfEncoder)?.is_gpu();
            #[cfg(feature = "gpu")]
            let encoder = if crf_encoder_on_gpu {
                // Must match `produce_gpu_crf`'s placement: this instance becomes
                // worker 0, so it has to land on the first *encoder* device — GPU 1
                // when detection has GPU 0 to itself.
                let enc_device = crf_encoder_devices(
                    escapepod_demux::crf::lattice_gpu::visible_device_count()
                        .unwrap_or(1)
                        .max(1),
                )[0];
                let enc = CrfEncoderGpu::load_bundle_on_device(&dir, args.threads, enc_device)?;
                if enc.gpu_decode_active() {
                    info!(
                        "{} GPU (onnxruntime CUDA), lattice decode GPU (batched), \
                         scores {}",
                        style::label("CRF encoder:"),
                        if enc.zero_copy_active() {
                            "decoded in place on the device"
                        } else {
                            "round-tripped through host memory"
                        }
                    );
                    // Zero-copy is *requested* from a cudarc probe, which proves
                    // only that the CUDA driver works — not that onnxruntime
                    // registered its CUDA EP. When the load-time probe finds the
                    // encoder output on the host, the run stays correct but the
                    // line above overstates it, so say what happened. Most often
                    // this means ORT_DYLIB_PATH points at a CPU-only build.
                    if let Some(why) = enc.zero_copy_fallback_reason() {
                        tracing::warn!(
                            "Zero-copy scores unavailable, using the copying path \
                             (same results, slower): {why}"
                        );
                    }
                } else {
                    // The decode is the larger half of this path's host cost, so
                    // running it on the CPU is a ~3x end-to-end difference. Say
                    // so rather than letting the run look fully accelerated.
                    tracing::warn!(
                        "CRF lattice decode fell back to the CPU ({}); the encoder is \
                         still on the GPU, but expect roughly 3x the wall time.",
                        enc.decode_fallback_reason().unwrap_or("reason unavailable")
                    );
                }
                CrfEncoderAny::Gpu(Box::new(enc))
            } else {
                CrfEncoderAny::Cpu(Box::new(CrfEncoder::load_bundle(&dir)?))
            };
            #[cfg(not(feature = "gpu"))]
            let encoder = {
                // Unreachable — placement returns CPU under `auto` and errors
                // under `--device gpu` when `gpu` is absent — but read, so
                // the binding is live in this build too.
                debug_assert!(!crf_encoder_on_gpu);
                CrfEncoderAny::Cpu(Box::new(CrfEncoder::load_bundle(&dir)?))
            };
            #[allow(unused_mut)]
            let mut encoder = encoder;
            if let Some(margin) = args.boundary_margin {
                let was = encoder.metadata().min_adapter_end();
                encoder.set_boundary_margin(margin);
                info!(
                    "{} adapter_end >= {} (was {}); reads between the two decode instead \
                     of routing to unclassified",
                    style::label("Boundary margin:"),
                    style::count(encoder.metadata().min_adapter_end()),
                    style::count(was),
                );
            }
            if let Some(shift) = args.clamp_max_shift {
                encoder.set_clamp_max_shift(shift);
            }
            let shift = encoder.metadata().clamp_max_shift();
            if shift > 0 {
                info!(
                    "{} reads with adapter_end down to {} decode from [0, {}], sliding up \
                     to {} samples past the adapter",
                    style::label("Window clamp:"),
                    style::count(encoder.metadata().signal.chunk.saturating_sub(shift)),
                    style::count(encoder.metadata().signal.chunk),
                    style::count(shift),
                );
            }
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
            // A gate implies the scores it gates on, rather than erroring: the
            // flags are not independent choices, and `--min-crf-margin` alone
            // silently doing nothing would be the worse failure.
            let want_scores =
                args.ref_scores || args.min_crf_margin.is_some() || args.min_crf_prob.is_some();
            let chains = want_scores
                .then(|| encoder.ref_chains(&refs.sequences()))
                .transpose()?;
            if let Some(c) = &chains {
                info!(
                    "{} {} references over {} shared lattice cells",
                    style::label("Lattice scoring:"),
                    style::count(c.len()),
                    style::count(c.cells()),
                );
            }
            ClassifyModel::Crf(Box::new(CrfHead {
                encoder,
                refs,
                min_margin: args.min_margin,
                chains,
                min_crf_margin: args.min_crf_margin,
                min_crf_prob: args.min_crf_prob,
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
    // ONNX path in the sidecar is relative to the bundle directory, and the
    // sidecar may declare the input tensor that detector consumes and the
    // sha256 of the weights it shipped (#187).
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
            BoundaryPin {
                method: b.method.clone(),
                onnx: b.onnx.as_ref().map(|o| dir.join(o)),
                input: b.input,
                sha256: b.sha256.clone(),
            }
        }),
        _ => None,
    };
    #[cfg(not(feature = "crf-decode"))]
    let boundary_pin: Option<BoundaryPin> = None;

    let detector = build_detector(&args, boundary_pin, device)?;

    // Neutralise onnxruntime's at-exit CUDA teardown *here*, at construction,
    // rather than at the end of a successful run.
    //
    // The bug this dodges is upstream (pykeio/ort#609): onnxruntime's CUDA
    // provider reads freed memory inside onnxruntime's own `.fini_array`
    // teardown, and glibc aborts on it with "corrupted double-linked list" —
    // measured 5 runs in 10, always *after* every read was written. The
    // mechanism is that `release_env_on_exit` calls `ReleaseEnv` only when the
    // last `Arc<Environment>` drops, and every live `Session` holds one, so
    // keeping ours alive means the faulty path never runs.
    //
    // A drop guard rather than a `mem::forget` at the end of `run`, because this
    // function is full of `?`. A trailing forget is skipped on exactly the error
    // paths — a bad POD5 batch, ENOSPC on an output shard, a writer panic — so a
    // run that failed for an ordinary, reportable reason would abort at exit
    // with 134 and bury its own diagnosis under a glibc message. That is the
    // case that most needs a readable error, and it was the one case the
    // mitigation missed. `LeakIf` fires on every path, and only when a GPU path
    // actually built ORT sessions.
    //
    // Scoped to the paths that actually build an ORT session: GPU CNN detection
    // and the GPU CRF encoder. The GPU DTW path is cudarc/NVRTC and has no
    // onnxruntime environment to keep alive, so it does not qualify — the old
    // `args.gpu` predicate leaked on its behalf for nothing.
    let leak_ort = detector.on_gpu() || crf_encoder_on_gpu;
    let model = LeakIf::new(model, leak_ort);
    let detector = LeakIf::new(detector, leak_ort);

    // Classify-head placement, decided here rather than inside the dispatch far
    // below, because a `--device gpu` failure has to surface before any writer
    // thread is spawned: a `?` down there would leave them detached.
    //
    // The CRF head is absent on purpose. Its encoder was placed when it was
    // loaded — an ORT session cannot be moved to another device after the fact —
    // so the dispatch reads that decision off the `CrfEncoderAny` variant and
    // this only echoes it.
    let classify_on_gpu = match &*model {
        ClassifyModel::Svm(_) => {
            crate::device::place_and_report(device, crate::device::Stage::Dtw)?.is_gpu()
        }
        ClassifyModel::Gbm(_) => {
            crate::device::note_cpu_only(
                device,
                "GBM classification",
                "the tree walk is CPU-only and expected to stay that way — a 32-core \
                 CPU pool beats a single GPU stream on it by roughly 20x. Adapter \
                 detection with `--method cnn` still runs on the device.",
            );
            false
        }
        #[cfg(feature = "crf-decode")]
        ClassifyModel::Crf(_) => crf_encoder_on_gpu,
    };

    let fp = FpParams::default();

    if let Some(dir) = &output_dir {
        std::fs::create_dir_all(dir)?;
    }

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
    match &output_dir {
        Some(dir) => info!("{} {}", style::label("Output:"), style::path(dir.display())),
        None => info!(
            "{} .p5s sidecar annotations only (no split files)",
            style::label("Output:")
        ),
    }

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
    //
    // The floor is what makes the budget approximate rather than absolute: at
    // more than `ROUTER_TOTAL_SLOTS / MIN_ROUTER_DEPTH` barcodes the per-barcode
    // share falls below it and total slots start scaling with the barcode count
    // again. That is deliberate — a channel too shallow to absorb a burst stalls
    // the pool, which costs more than the memory — but it is why the floor is
    // 64 and not the 256 it started as. The CRF head takes its barcode set from
    // a user-supplied reference CSV, so the count is unbounded, and a 256 floor
    // put a 384-plex set at ~1 GB, twice the budget this is supposed to enforce.
    // No shipping design is affected: the floor does not bind until 768
    // barcodes, so 4-, 5-, 12-, 16- and 96-code models all get the same depth
    // they got before.
    let barcodes = model.barcode_names();
    let n_barcodes = barcodes.len().max(1);
    let router_depth = (ROUTER_TOTAL_SLOTS / n_barcodes).clamp(MIN_ROUTER_DEPTH, 4096);
    if ROUTER_TOTAL_SLOTS / n_barcodes < MIN_ROUTER_DEPTH {
        tracing::debug!(
            "{n_barcodes} barcodes x depth {router_depth} exceeds the {ROUTER_TOTAL_SLOTS}-slot \
             router budget; peak router memory will scale with the barcode count"
        );
    }

    // Sidecar-only runs (--annotate without -d) spawn no writers: `routers`
    // stays empty and `route` simply drops the POD5 leg.
    let mut routers: Routers = HashMap::new();
    let mut writer_handles: Vec<(String, std::thread::JoinHandle<anyhow::Result<usize>>)> =
        Vec::new();
    if let Some(dir) = &output_dir {
        for bc in barcodes {
            let (tx, rx) = std::sync::mpsc::sync_channel::<Routed>(router_depth);
            let path = dir.join(format!("{}_{}.pod5", args.prefix, bc));
            let dicts = predefined.clone();
            let handle = std::thread::spawn(move || writer_thread(rx, &path, dicts));
            routers.insert(bc.clone(), tx);
            writer_handles.push((bc, handle));
        }
    }
    let routers = Arc::new(routers);

    // Optional classifications CSV writer (a single small-record stream);
    // with --annotate the same thread also collects the assignment map the
    // sidecar write needs.
    // The writer resolves `crf_best` from an index, so it needs the panel's
    // names — and their presence is also what tells it to emit the score
    // columns at all.
    #[cfg(feature = "crf-decode")]
    let ref_names = match &*model {
        ClassifyModel::Crf(head) if head.chains.is_some() => Some(head.refs.names().to_vec()),
        _ => None,
    };
    #[cfg(not(feature = "crf-decode"))]
    let ref_names = None;
    let (class_tx, class_handle) =
        spawn_class_writer(args.classifications.as_deref(), args.annotate, ref_names)?;

    // ---- Stages A/B: produce classified reads ----
    let produce_result = match &*model {
        ClassifyModel::Svm(svm) => {
            // No "the GPU does nothing on this head" warning to emit: `gpu`
            // is atomic, so a build that can place work on a device always
            // carries the batched DTW-SVM classify kernel this arm uses. It
            // used to be reachable on a `cnn-gpu`/`crf-gpu`-only build, which
            // no longer exists.
            #[cfg(feature = "gpu")]
            {
                if classify_on_gpu {
                    produce_gpu(&args, &detector, svm, fp, &routers, class_tx.as_ref(), &pb)
                } else {
                    produce_cpu(&args, &detector, svm, fp, &routers, class_tx.as_ref(), &pb)
                }
            }
            #[cfg(not(feature = "gpu"))]
            {
                // Placement already refused `--device gpu` on this build and
                // returned CPU under `auto`; asserting it here is what keeps the
                // two arms from drifting.
                debug_assert!(!classify_on_gpu);
                produce_cpu(&args, &detector, svm, fp, &routers, class_tx.as_ref(), &pb)
            }
        }
        ClassifyModel::Gbm(gbm) => {
            // CPU-only head; the `--device gpu` note was emitted with the other
            // placements, before the writer threads existed.
            produce_cpu_gbm(&args, &detector, gbm, fp, &routers, class_tx.as_ref(), &pb)
        }
        #[cfg(feature = "crf-decode")]
        ClassifyModel::Crf(head) => {
            // No "the encoder stayed on the CPU anyway" warning either. That
            // warned about a build with a GPU feature but not the CRF one,
            // where a GPU request silently left ~91% of this head's cost on
            // tract; `gpu` is atomic now, so placement moves the encoder
            // whenever it says GPU. Whether the device is usable is a *runtime*
            // question, and the encoder loader reports its own fallbacks.
            match &head.encoder {
                #[cfg(feature = "gpu")]
                CrfEncoderAny::Gpu(enc) => {
                    // Extra encoder workers load their own session from the same
                    // bundle, so the pool needs the directory the head came from.
                    let bundle = crf_dir_for_pin
                        .as_deref()
                        .ok_or_else(|| anyhow::anyhow!("CRF head without a bundle directory"))?;
                    produce_gpu_crf(
                        &args,
                        &detector,
                        head,
                        enc,
                        bundle,
                        &routers,
                        class_tx.as_ref(),
                        &pb,
                    )
                }
                CrfEncoderAny::Cpu(enc) => produce_cpu_crf(
                    &args,
                    &detector,
                    head,
                    enc,
                    &routers,
                    class_tx.as_ref(),
                    &pb,
                ),
            }
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
    let mut assignments: Option<Collected> = None;
    if let Some(h) = class_handle {
        assignments = h
            .join()
            .map_err(|e| anyhow::anyhow!("classifications writer panicked: {e:?}"))??;
    }
    summary.per_barcode.sort();
    produce_result?;

    pb.finish_with_message("complete");

    // --annotate: record the collected assignments in each input's sidecar.
    // write_annotation intersects the map with the reads actually present in
    // each file, so one global map serves all inputs without provenance
    // tracking. Runs after produce_result? so a failed run writes nothing.
    if args.annotate {
        let collected = assignments.unwrap_or_default();
        // Kept for the barcode summary below, which counts labels rather than
        // reading them back off the sidecar.
        let assignments = collected.barcode.clone();
        let columns = collected.into_columns();
        for path in &args.input {
            // One read-modify-write for all five columns: five separate ones
            // would rewrite the sidecar five times and leave four intermediate
            // states on disk describing a run that never happened.
            let result = escapepod_signal::operations::write_columns(path, &columns, false)
                .map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
            let barcode = result.columns.first();
            info!(
                "{} {} — {} of {} reads assigned across {} labels{}",
                style::action("wrote"),
                style::path(result.sidecar_path.display()),
                style::count(barcode.map_or(0, |c| c.assigned)),
                style::count(result.total_reads),
                style::count(barcode.map_or(0, |c| c.labels)),
                if result.columns.len() > 1 {
                    format!(
                        ", plus {}",
                        result.columns[1..]
                            .iter()
                            .map(|c| c.name.as_str())
                            .collect::<Vec<_>>()
                            .join("/")
                    )
                } else {
                    String::new()
                },
            );
        }
        // Sidecar-only runs have no writer threads; fill the summary from
        // the assignment map so the barcode table still prints.
        if summary.per_barcode.is_empty() {
            let mut counts: HashMap<&String, usize> = HashMap::new();
            for barcode in assignments.values() {
                *counts.entry(barcode).or_default() += 1;
            }
            summary.per_barcode = counts.into_iter().map(|(bc, n)| (bc.clone(), n)).collect();
            summary.per_barcode.sort();
        }
    }

    print_summary(&summary);
    timer.report(profile);

    // The onnxruntime sessions are kept alive past the end of `main` by the
    // `LeakIf` guards wrapping `model` and `detector` — see that type, and
    // pykeio/ort#609. Nothing to do here: the guards fire on this path and on
    // every `?` above, which is the point.
    Ok(())
}

/// Total queued reads allowed across *all* per-barcode writer channels. Each
/// queued read owns its compressed signal (~10 KB), so this bounds router
/// memory at roughly 500 MB regardless of how many barcodes the model defines.
/// Split evenly per barcode and clamped — see the router setup in `run`.
const ROUTER_TOTAL_SLOTS: usize = 49_152;

/// Shallowest per-barcode writer channel the router will hand out.
///
/// Below this a burst on one barcode blocks a rayon worker inside `send`, and
/// blocked workers cannot be stolen from. This floor is what makes
/// [`ROUTER_TOTAL_SLOTS`] a target rather than a hard cap: past
/// `ROUTER_TOTAL_SLOTS / MIN_ROUTER_DEPTH` = 768 barcodes the budget is
/// knowingly exceeded, and `run` logs when that happens.
const MIN_ROUTER_DEPTH: usize = 64;

/// Upper bound on reads accumulated into one detect+classify block. POD5 Arrow
/// read-batches are small (~1000 reads), so detecting per batch makes GPU CNN
/// fire many tiny calls; accumulating across batches amortises the per-call
/// overhead over more rows. The on-device batch is separately capped by
/// `gpu_batch_elems`, and on the CNN path [`BLOCK_TARGET_BYTES`] usually binds
/// first.
///
/// This used to also be how same-length reads found each other, back when prep
/// gave every read its own length; #187 made the model input one fixed shape,
/// so grouping is no longer a reason to accumulate.
const DETECT_WINDOW: usize = 65_536;

/// Upper bound on the *decoded signal bytes* held in one block.
///
/// A read-count cap alone does not bound memory: LLR sets no decode bound (it
/// normalizes over the whole read), so a block holds whole reads. At 65,536
/// reads that is ~1.8 GB on a short-read RNA library and ~9.5 GB on a long-read
/// one — measured 13.5 GB peak RSS on a 1.22 M-read file, of which ~2.4 GB was
/// this block (the rest is mmap'd input pages). Capping by bytes keeps the
/// footprint flat across libraries with wildly different read lengths.
///
/// Note which cap actually binds: the CNN path decodes up to `max_obs_trace`
/// (16,000 samples = 32 KB) per read, so on a long-read library this cap binds
/// at a few thousand reads and [`DETECT_WINDOW`] is never reached. That is
/// fine — since #187 fixed the model input to one shape, batching quality no
/// longer depends on accumulating many reads to find same-length peers, and
/// smaller blocks pipeline better against the encoder pool.
///
/// 128 MB measured best on 1.22 M reads; larger is worse on both wall time and
/// RSS (512 MB: 74.4 s / 13.07 GB vs 128 MB: 63.8 s / 12.68 GB).
///
/// This bounds **one block**, not the process. [`fill_shard`] enforces it per
/// filler thread, so blocks in flight are roughly
/// `fillers * (BLOCK_QUEUE_DEPTH + 1) + 1` — about 0.9 GB at [`CPU_FILLERS`] and
/// 3.2 GB at [`GPU_CRF_FILLERS`], which is the whole of the measured ~4 GB gap
/// between those two settings. That is already what the peak-RSS columns in
/// both tables are measuring, so the 128 MB figure is tuned *with* the
/// multiplier, not against it; scale the filler count to trade throughput for
/// footprint rather than shrinking this.
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
///
/// # This is also flat for the GPU CRF pipeline — do not re-litigate it
///
/// Once the encoder ran on the device, its workers traced as 47% blocked on
/// `recv`, which reads like reader starvation, and a sweep at 2/4/8 fillers on
/// 1 M reads returned a clean monotone 179.6 s → 136.4 s → 119.6 s with GPU
/// utilisation climbing 49% → 65% → 75% alongside. Every column agreed.
///
/// It was an artefact. The arms ran in ascending order against a cold page
/// cache, so each one re-read less of the 12 GB input than the last. Re-run
/// **interleaved** on the same node — 2, 8, 2, 8 — the effect vanishes:
///
/// ```text
/// fillers=2  114.6 s      fillers=8  112.2 s
/// fillers=2  109.8 s      fillers=8  109.1 s
/// ```
///
/// A second node confirmed it (116.2 s at 8 vs 114.2 s at 2). Eight fillers cost
/// ~4 GB of resident memory for nothing. Interleave the arms before believing
/// any I/O-adjacent sweep here.
const DEFAULT_FILLERS: usize = 2;

fn filler_threads() -> usize {
    use std::sync::OnceLock;
    static CACHED: OnceLock<usize> = OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("ESCAPEPOD_DEMUX_FILLERS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or(DEFAULT_FILLERS)
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
    class_tx: Option<&ClassTx>,
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
                            Call::scoreless(barcode, conf),
                            chunks,
                            run_infos,
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
    class_tx: Option<&ClassTx>,
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
                        Call::scoreless(barcode, conf),
                        chunks,
                        run_infos,
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
/// Detection is batched over the whole block, but the per-read work is *not*
/// chunked the way `produce_cpu_gbm` chunks: that head batches because
/// `predict_many` is a genuinely batched kernel, whereas tract has no batched
/// LSTM, so a chunk here would only ever run its reads serially. At ~14 ms per
/// read (13 ms encode + 1.2 ms decode) even a modest chunk is seconds of work a
/// starved worker cannot steal, so this fans out per read and keeps one
/// `CrfScratch` per *worker* via `for_each_init` — the same shape as
/// `produce_cpu`.
#[cfg(feature = "crf-decode")]
fn produce_cpu_crf(
    args: &RunArgs,
    detector: &Detector,
    head: &CrfHead,
    encoder: &CrfEncoder,
    routers: &Routers,
    class_tx: Option<&ClassTx>,
    pb: &indicatif::ProgressBar,
) -> anyhow::Result<()> {
    let meta = encoder.metadata();
    drive_blocks(
        &args.input,
        detector.signal_decode_bound(),
        |sigs, items| {
            let bounds = detector.detect_batch(&sigs);
            sigs.into_par_iter().zip(bounds).zip(items).for_each_init(
                || (CrfScratch::new(), Vec::<f32>::new()),
                |(scratch, window), ((signal, (_s, adapter_end)), (read, chunks, run_infos))| {
                    let call = (|| {
                        let adc = signal.as_ref()?;
                        // The detector reports `adapter_end` as an index into
                        // the decoded prefix, which is what `prep` wants. Only
                        // the `chunk` samples ending there are converted — the
                        // prefix itself can be the whole read under LLR.
                        if !meta.prep_adc_into(
                            adc,
                            adapter_end,
                            read.calibration_offset,
                            read.calibration_scale,
                            window,
                        ) {
                            return None;
                        }
                        match &head.chains {
                            Some(chains) => encoder
                                .basecall_prepped_with_refs(window, scratch, chains)
                                .inspect_err(|e| tracing::warn!("encoder: {e}"))
                                .ok()
                                .map(|s| call_barcode_scored(head, &s)),
                            None => {
                                let seq = encoder
                                    .basecall_prepped(window, scratch)
                                    .inspect_err(|e| tracing::warn!("encoder: {e}"))
                                    .ok()?;
                                call_barcode(head, &seq)
                                    .map(|(b, c)| Call::scoreless(b, c))
                                    .or_else(|| Some(Call::unclassified()))
                            }
                        }
                    })()
                    .unwrap_or_else(Call::unclassified);

                    route(
                        routers,
                        class_tx,
                        read.for_writing(read.run_info_index),
                        call,
                        chunks,
                        run_infos,
                    );
                    pb.inc(1);
                },
            );
        },
    )
}

/// Match one decoded sequence to a reference, applying `--min-margin`.
///
/// `None` means "no call" — either nothing matched, or the runner-up was too
/// close — and the caller routes those to `unclassified`. Shared by the CPU and
/// GPU CRF producers so the two cannot drift on the gate or on what confidence
/// means.
///
/// Same gate as `demux basecall --min-margin`, including its treatment of a
/// single reference: with no runner-up there is no margin to test, so the call
/// stands. Confidence is the margin, matching `demux basecall` and
/// `eval_recovery.py`; a lone reference reports 0 rather than a fabricated
/// distance.
#[cfg(feature = "crf-decode")]
fn call_barcode(head: &CrfHead, seq: &str) -> Option<(String, f64)> {
    let m = head.refs.match_sequence(seq.as_bytes())?;
    if !m.margin.is_none_or(|v| v >= head.min_margin) {
        return None;
    }
    Some((
        head.refs.name(m.index).to_string(),
        f64::from(m.margin.unwrap_or(0)),
    ))
}

/// [`call_barcode`] with the lattice consulted as well — `--ref-scores`.
///
/// The edit distance still makes the assignment, so a run that only adds
/// `--ref-scores` calls exactly what it called before and merely records what
/// the lattice thought. The lattice gates (`--min-crf-margin`,
/// `--min-crf-prob`) are what turn that record into a decision, and they only
/// ever *reject*: an assignment the edit distance did not make cannot be
/// created here.
///
/// Scores are attached to rejected reads too, so a `unclassified` row in the
/// classifications CSV says which gate dropped it and by how much.
#[cfg(feature = "crf-decode")]
fn call_barcode_scored(head: &CrfHead, scored: &ScoredDecode) -> Call {
    let Some(m) = head.refs.match_sequence(scored.sequence.as_bytes()) else {
        return Call::unclassified();
    };
    let crf = scored.call(m.index).map(|(logp, margin)| CrfRowScores {
        logp,
        margin,
        best: scored.best().map_or(m.index, |(i, _, _)| i) as u32,
        mean_logpost: scored.mean_logpost,
    });
    let gated = !m.margin.is_none_or(|v| v >= head.min_margin)
        || crf.is_some_and(|c| {
            head.min_crf_margin
                .is_some_and(|t| c.margin.is_some_and(|m| m < t))
                || head.min_crf_prob.is_some_and(|t| c.logp.exp() < t)
        });
    Call {
        barcode: if gated {
            UNCLASSIFIED.to_string()
        } else {
            head.refs.name(m.index).to_string()
        },
        confidence: f64::from(m.margin.unwrap_or(0)),
        crf,
    }
}

/// Reads whose prepped windows are handed to the GPU encoder in one go
/// (`ESCAPEPOD_CRF_GPU_BLOCK`).
///
/// This is a *host* memory bound, not a device one — `CrfEncoderGpu` splits each
/// call into `ESCAPEPOD_CRF_GPU_BATCH_ROWS` (512) rows for the device and
/// decodes each of those before encoding the next, so device activations and the
/// 1.5 MB/read score buffers are already capped underneath this. Raising this
/// does not enlarge the device batch; raise `ESCAPEPOD_CRF_GPU_BATCH_ROWS` for
/// that.
///
/// What this sizes is the prepped windows in flight: `chunk` f32 per read, so
/// 48 MB per block at the RNA004 geometry (3000 samples), and the channel holds
/// two. A whole [`DETECT_WINDOW`] block would be 786 MB.
///
/// 4096 is 8 device batches, which measured enough on an A30: the encoder thread
/// is the bottleneck there, so the producer is always ahead and deeper buffering
/// buys nothing. Worth raising only if prep becomes the slower side (many cores
/// starved of I/O, or a much faster device).
#[cfg(feature = "gpu")]
fn crf_gpu_block() -> usize {
    use std::sync::OnceLock;
    static CACHED: OnceLock<usize> = OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("ESCAPEPOD_CRF_GPU_BLOCK")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or(4096)
    })
}

/// The CUDA ordinals the CRF encoder pool may use.
///
/// With one visible device everything shares it. With more, **device 0 is
/// reserved for adapter detection** and the encoders take the rest.
///
/// # This reservation is now mis-tuned — measured, unfixed
///
/// It was justified by detect and the encoder costing about the same device time
/// (5.4 s vs 6.4 s over 40 k reads): if two roles are comparable, sharing a card
/// makes the pipeline's ceiling their sum and separating them makes it the
/// larger of the two. #187 destroyed that premise. Matching the model's training
/// convention cut detect's device time ~20x (401 s -> 20.2 s over 1 M reads)
/// while the encoder stayed put, so the roles now cost ~19 s against ~170 s —
/// and reserving a whole card for 10% of the work costs more than the contention
/// it avoids. At exactly two devices it is pathological: the encoder pool does
/// not grow at all, so the second GPU only offloads detection. Measured on 1 M
/// reads, interleaved, 2 reps each:
///
/// ```text
/// GPUs   encoder pool          wall     vs 1 GPU
///   1    2 workers on [0]     107.5 s     --
///   2    2 workers on [1]      96.5 s    1.11x    <- barely worth the card
///   4    6 workers on [1,2,3]  43.7 s    2.46x
/// ```
///
/// The likely fix is to encode on *every* visible device and let detection share
/// device 0 — workers pull from one channel rather than being handed a
/// partition, so whichever device is slowed by carrying detection simply takes
/// fewer blocks, with nothing to balance explicitly. It is **not** applied here
/// because it is unmeasured, and the four-device column is the reason for
/// caution: at 4 GPUs the producer is already the constraint (busy 71%, workers
/// blocked on `recv` 38.6 s), and that producer *is* detection. Adding encoder
/// work to device 0 could slow the stage that is already the ceiling and regress
/// the case that currently scales, to fix the case that does not. Measure both
/// columns before changing it.
#[cfg(feature = "gpu")]
fn crf_encoder_devices(visible: usize) -> Vec<i32> {
    if visible > 1 {
        (1..visible as i32).collect()
    } else {
        vec![0]
    }
}

/// Encoder workers, and how they are spread over the visible devices
/// (`ESCAPEPOD_CRF_GPU_WORKERS`).
///
/// Default is two per visible device. One worker leaves the device idle through
/// its own per-call overhead — an onnxruntime `Run` is bracketed by cuDNN plan
/// setup, two stream syncs, the barcode match and the routing, and profiling put
/// 34% of wall time in cuDNN's engine/graph setup rather than in convolution. A
/// second worker on the *same* device overlaps that setup with the first's
/// device work; further workers across devices add real parallelism.
///
/// Capped so a many-GPU node does not spawn an encoder session per device
/// unbounded — each one holds its own ONNX graph and scores buffer.
///
/// Raising this does **not** give the device more memory to work with: workers
/// on one device split its row budget (`CrfEncoderGpu::share_device_with`), so
/// past a point each does proportionally smaller calls and the per-call overhead
/// this exists to hide grows back. Before that fix, four workers on one 24 GB
/// A30 simply exhausted the card and killed the run.
#[cfg(feature = "gpu")]
fn crf_gpu_workers(devices: usize) -> usize {
    std::env::var("ESCAPEPOD_CRF_GPU_WORKERS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or_else(|| (2 * devices).min(8))
        .max(1)
}

/// GPU producer for the CRF head: parallel CPU prep (decode + detect + window
/// standardisation) feeds a pool of encoder workers through a bounded channel,
/// so the device is kept fed while the next block is prepped.
///
/// The overlap is the point. Running the encoder inline in the block loop the
/// way [`produce_cpu_crf`] does would alternate device and host phases with
/// neither saturated: the GPU would idle through prep and the rayon pool would
/// idle through inference. `drive_blocks` already hands blocks over from its own
/// reader threads, so this adds the second half of the pipeline.
///
/// Workers pull from one channel rather than being handed a partition, so a slow
/// device or an unlucky block cannot leave one worker holding the tail. Blocks
/// are independent and output row order is already nondeterministic here (the
/// per-barcode writers interleave), so which worker takes which does not matter.
/// Where the fused GPU CRF pipeline's wall time goes, in milliseconds summed
/// across threads (`ESCAPEPOD_CRF_GPU_TRACE=1`).
///
/// Blocked-vs-working is the whole point: a stage that is mostly *blocked* is
/// being starved by its neighbour, and a stage that is mostly *working* is the
/// constraint. Six plausible fixes for this pipeline's idle GPU were tried and
/// measured negative before this existed, which is the argument for having it.
#[cfg(feature = "gpu")]
#[derive(Default)]
struct GpuTrace {
    /// Producer: batched adapter-CNN detect over a whole block (prep + infer).
    detect_ms: std::sync::atomic::AtomicU64,
    /// Detect, host half: i16 -> f32, normalise, median, across rayon.
    detect_prep_ms: std::sync::atomic::AtomicU64,
    /// Detect, device half: the batched onnxruntime call.
    detect_infer_ms: std::sync::atomic::AtomicU64,
    /// Producer: window standardisation across rayon.
    prep_ms: std::sync::atomic::AtomicU64,
    /// Producer blocked handing a sub-block over — the encoders are behind.
    send_blocked_ms: std::sync::atomic::AtomicU64,
    /// Worker blocked waiting for a sub-block — the producer is behind.
    recv_blocked_ms: std::sync::atomic::AtomicU64,
    /// Worker: onnxruntime encode plus the lattice decode.
    encode_decode_ms: std::sync::atomic::AtomicU64,
    /// Worker: barcode match across rayon.
    match_ms: std::sync::atomic::AtomicU64,
    /// Worker blocked in `route` — a per-barcode writer channel is full.
    route_ms: std::sync::atomic::AtomicU64,
}

#[cfg(feature = "gpu")]
impl GpuTrace {
    fn enabled() -> bool {
        std::env::var("ESCAPEPOD_CRF_GPU_TRACE").as_deref() == Ok("1")
    }

    /// Add `t`'s elapsed millis to `field`, but only when tracing is on — the
    /// `Instant::now()` calls are cheap next to a block but not free.
    fn add(field: &std::sync::atomic::AtomicU64, t: std::time::Instant) {
        field.fetch_add(
            t.elapsed().as_millis() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
    }

    fn report(&self, wall: std::time::Duration, workers: usize) {
        use std::sync::atomic::Ordering::Relaxed;
        let g = |f: &std::sync::atomic::AtomicU64| f.load(Relaxed) as f64 / 1000.0;
        info!("{}", style::label("GPU CRF stage trace (thread-seconds):"));
        info!(
            "  producer   detect {:>7.1}s (host {:>6.1}s + device {:>6.1}s)  \
             prep {:>5.1}s  BLOCKED on send {:>6.1}s",
            g(&self.detect_ms),
            g(&self.detect_prep_ms),
            g(&self.detect_infer_ms),
            g(&self.prep_ms),
            g(&self.send_blocked_ms)
        );
        info!(
            "  {} worker(s)  encode+decode {:>7.1}s  match {:>6.1}s  route {:>6.1}s  BLOCKED on recv {:>7.1}s",
            workers,
            g(&self.encode_decode_ms),
            g(&self.match_ms),
            g(&self.route_ms),
            g(&self.recv_blocked_ms)
        );
        let wall_s = wall.as_secs_f64();
        info!(
            "  wall {:.1}s — worker busy {:.0}% of its wall, producer busy {:.0}%",
            wall_s,
            100.0 * (g(&self.encode_decode_ms) + g(&self.match_ms) + g(&self.route_ms))
                / (wall_s * workers as f64).max(1e-9),
            100.0 * (g(&self.detect_ms) + g(&self.prep_ms)) / wall_s.max(1e-9),
        );
    }
}

// One over clippy's limit: the sibling producers take the same seven, and this
// one additionally needs the bundle directory to load its extra workers.
#[allow(clippy::too_many_arguments)]
#[cfg(feature = "gpu")]
fn produce_gpu_crf(
    args: &RunArgs,
    detector: &Detector,
    head: &CrfHead,
    encoder: &CrfEncoderGpu,
    bundle: &Path,
    routers: &Routers,
    class_tx: Option<&ClassTx>,
    pb: &indicatif::ProgressBar,
) -> anyhow::Result<()> {
    /// Prepped windows (`None` = no usable window) aligned with their reads.
    type Block = (Vec<Option<Vec<f32>>>, Vec<BlockItem>);

    let meta = encoder.metadata();
    let gpu_block = crf_gpu_block();

    // Visible, so `CUDA_VISIBLE_DEVICES` already applies: under SLURM
    // `--gres=gpu:1` this is 1 and the pool collapses to same-device workers.
    let devices = escapepod_demux::crf::lattice_gpu::visible_device_count()
        .unwrap_or(1)
        .max(1);
    let enc_devices = crf_encoder_devices(devices);
    let workers = crf_gpu_workers(enc_devices.len());
    // Worker 0 reuses the encoder already loaded for its metadata, which `run`
    // placed on `enc_devices[0]`; the rest get their own session, round-robin
    // over the encoder devices.
    let extra: Vec<CrfEncoderGpu> = (1..workers)
        .map(|w| {
            CrfEncoderGpu::load_bundle_on_device(
                bundle,
                args.threads,
                enc_devices[w % enc_devices.len()],
            )
        })
        .collect::<Result<_, _>>()?;
    // Every worker sharing a device allocates its LSTM activations from that
    // device's VRAM, so they must split its row budget rather than each take a
    // full one. Without this, `ESCAPEPOD_CRF_GPU_WORKERS=4` on a single 24 GB
    // A30 exhausted the card and killed the run.
    let per_device = workers.div_ceil(enc_devices.len());
    encoder.share_device_with(per_device);
    for e in &extra {
        e.share_device_with(per_device);
    }
    if workers > 1 || devices > 1 {
        info!(
            "{} {} worker(s) on GPU {:?}, {} reads/call{}",
            style::label("CRF encoder:"),
            style::count(workers),
            enc_devices,
            style::count(encoder.batch_rows()),
            if devices > 1 {
                "; adapter detection has GPU 0 to itself"
            } else {
                ""
            }
        );
    }

    // Depth scales with the pool so every worker can hold one block and still
    // find another queued behind it.
    let (block_tx, block_rx) = std::sync::mpsc::sync_channel::<Block>(2 * workers);
    let block_rx = Arc::new(std::sync::Mutex::new(block_rx));

    let trace = GpuTrace::default();
    let trace = &trace;
    let tracing_on = GpuTrace::enabled();
    let t_wall = std::time::Instant::now();

    std::thread::scope(|scope| -> anyhow::Result<()> {
        let mut gpus = Vec::with_capacity(workers);
        for w in 0..workers {
            let enc: &CrfEncoderGpu = if w == 0 { encoder } else { &extra[w - 1] };
            let rx = Arc::clone(&block_rx);
            gpus.push(scope.spawn(move || -> anyhow::Result<()> {
                loop {
                    // Held only across `recv`, never across the encode.
                    let t_wait = std::time::Instant::now();
                    let next = {
                        let guard = rx.lock().unwrap_or_else(|p| p.into_inner());
                        guard.recv()
                    };
                    if tracing_on {
                        GpuTrace::add(&trace.recv_blocked_ms, t_wait);
                    }
                    let Ok((windows, items)) = next else { break };

                    // Encodes on the device, then fans the lattice decode back
                    // out across rayon. `None` windows never reach the device
                    // and come back `None`, so this stays aligned with `items`.
                    let t_enc = std::time::Instant::now();
                    let scored = match &head.chains {
                        Some(chains) => Some(
                            enc.basecall_batch_with_refs(&windows, chains)
                                .map_err(|e| anyhow::anyhow!("GPU encoder (worker {w}): {e}"))?,
                        ),
                        None => None,
                    };
                    let seqs = match &scored {
                        Some(_) => None,
                        None => Some(
                            enc.basecall_batch(&windows)
                                .map_err(|e| anyhow::anyhow!("GPU encoder (worker {w}): {e}"))?,
                        ),
                    };
                    if tracing_on {
                        GpuTrace::add(&trace.encode_decode_ms, t_enc);
                    }
                    // 96 references x one wavefront alignment each is small next
                    // to the decode, but it is still per-read work worth fanning
                    // out.
                    let t_match = std::time::Instant::now();
                    let calls: Vec<Call> = match (&scored, &seqs) {
                        (Some(scored), _) => scored
                            .par_iter()
                            .map(|s| {
                                s.as_ref().map_or_else(Call::unclassified, |s| {
                                    call_barcode_scored(head, s)
                                })
                            })
                            .collect(),
                        (None, Some(seqs)) => seqs
                            .par_iter()
                            .map(|seq| {
                                seq.as_deref()
                                    .and_then(|s| call_barcode(head, s))
                                    .map_or_else(Call::unclassified, |(b, c)| Call::scoreless(b, c))
                            })
                            .collect(),
                        (None, None) => unreachable!("one of the two arms always ran"),
                    };
                    if tracing_on {
                        GpuTrace::add(&trace.match_ms, t_match);
                    }
                    let n = items.len() as u64;
                    let t_route = std::time::Instant::now();
                    for ((read, chunks, run_infos), call) in items.into_iter().zip(calls) {
                        route(
                            routers,
                            class_tx,
                            read.for_writing(read.run_info_index),
                            call,
                            chunks,
                            run_infos,
                        );
                    }
                    if tracing_on {
                        GpuTrace::add(&trace.route_ms, t_route);
                    }
                    pb.inc(n);
                }
                Ok(())
            }));
        }

        // CPU prep. A send failure means the encoder thread is gone; stop
        // feeding and let the join below report why rather than masking it with
        // a channel error.
        let mut hung_up = false;
        let drive = drive_blocks(
            &args.input,
            detector.signal_decode_bound(),
            |sigs, items| {
                if hung_up {
                    return;
                }
                // Detect over the whole block (one batched GPU CNN call), then
                // hand the encoder bounded sub-blocks — see `CRF_GPU_BLOCK`.
                let t_det = std::time::Instant::now();
                let bounds = detector.detect_batch_traced(
                    &sigs,
                    tracing_on.then_some((&trace.detect_prep_ms, &trace.detect_infer_ms)),
                );
                if tracing_on {
                    GpuTrace::add(&trace.detect_ms, t_det);
                }
                let mut rows = sigs.into_iter().zip(bounds).zip(items);
                loop {
                    let chunk: Vec<_> = rows.by_ref().take(gpu_block).collect();
                    if chunk.is_empty() {
                        break;
                    }
                    let t_prep = std::time::Instant::now();
                    let windows: Vec<Option<Vec<f32>>> = chunk
                        .par_iter()
                        .map(|((signal, (_s, adapter_end)), (read, _, _))| {
                            let adc = signal.as_ref()?;
                            let mut w = Vec::new();
                            // Same conversion as the CPU path: only the `chunk`
                            // samples ending at `adapter_end` are calibrated.
                            meta.prep_adc_into(
                                adc,
                                *adapter_end,
                                read.calibration_offset,
                                read.calibration_scale,
                                &mut w,
                            )
                            .then_some(w)
                        })
                        .collect();
                    let items: Vec<BlockItem> = chunk.into_iter().map(|(_, item)| item).collect();
                    if tracing_on {
                        GpuTrace::add(&trace.prep_ms, t_prep);
                    }
                    let t_send = std::time::Instant::now();
                    let sent = block_tx.send((windows, items));
                    if tracing_on {
                        GpuTrace::add(&trace.send_blocked_ms, t_send);
                    }
                    if sent.is_err() {
                        hung_up = true;
                        return;
                    }
                }
            },
        );

        drop(block_tx);
        // An encoder's error is the root cause when the channel hung up, so
        // report it ahead of whatever `drive_blocks` returned.
        for (w, g) in gpus.into_iter().enumerate() {
            g.join()
                .map_err(|e| anyhow::anyhow!("GPU encoder worker {w} panicked: {e:?}"))??;
        }
        if tracing_on {
            trace.report(t_wall.elapsed(), workers);
        }
        drive
    })
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
    class_tx: Option<&ClassTx>,
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
                        Call::scoreless(barcode_label(result.predicted_barcode), result.confidence),
                        chunks,
                        run_infos,
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
                                Call::unclassified(),
                                chunks,
                                run_infos.clone(),
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

/// What `--annotate` records in the sidecar, accumulated by the classifications
/// writer thread as rows go past.
///
/// Sidecar-only demux writes no CSV, so without this the lattice scores would
/// be computed and then dropped on exactly the runs that have nowhere else to
/// keep them (#241).
#[derive(Default)]
struct Collected {
    barcode: HashMap<Uuid, String>,
    /// Only populated under `--ref-scores`.
    crf: Option<CollectedScores>,
}

/// The lattice columns, held as reference *indices* for `crf_best` and resolved
/// to names once at the end — a million short `String`s to say `nbc07` is a
/// million allocations for sixteen distinct values.
#[derive(Default)]
struct CollectedScores {
    names: Vec<String>,
    best: HashMap<Uuid, u32>,
    logp: HashMap<Uuid, f32>,
    margin: HashMap<Uuid, f32>,
    mean_logpost: HashMap<Uuid, f32>,
}

impl Collected {
    /// The sidecar columns this run produced, in write order.
    fn into_columns(self) -> Vec<ColumnWrite> {
        let mut out = vec![ColumnWrite {
            name: "barcode".to_string(),
            values: ColumnValues::Labels(self.barcode),
        }];
        let Some(crf) = self.crf else { return out };
        out.push(ColumnWrite {
            name: "crf_best".to_string(),
            values: ColumnValues::Labels(
                crf.best
                    .into_iter()
                    .filter_map(|(id, i)| crf.names.get(i as usize).map(|n| (id, n.clone())))
                    .collect(),
            ),
        });
        for (name, values) in [
            ("crf_logp", crf.logp),
            ("crf_margin", crf.margin),
            ("mean_logpost", crf.mean_logpost),
        ] {
            out.push(ColumnWrite {
                name: name.to_string(),
                values: ColumnValues::Scores(values),
            });
        }
        out
    }
}

/// Optional classifications-CSV writer thread. With `collect` it also
/// accumulates what the `--annotate` sidecar write records, returned through
/// the join handle.
#[allow(clippy::type_complexity)]
fn spawn_class_writer(
    path: Option<&Path>,
    collect: bool,
    ref_names: Option<Vec<String>>,
) -> anyhow::Result<(
    Option<ClassTx>,
    Option<std::thread::JoinHandle<anyhow::Result<Option<Collected>>>>,
)> {
    if path.is_none() && !collect {
        return Ok((None, None));
    }
    let (tx, rx) = std::sync::mpsc::sync_channel::<ClassRow>(16_384);
    let path = path.map(Path::to_path_buf);
    let handle = std::thread::spawn(move || -> anyhow::Result<Option<Collected>> {
        use std::io::Write;
        // The score columns exist for the whole file or not at all — whether
        // the run scores is a property of the head, not of a read — so the
        // header is decided once here rather than per row.
        let names = ref_names.unwrap_or_default();
        let scored = !names.is_empty();
        let mut writer = match &path {
            Some(path) => {
                let file = std::fs::File::create(path)?;
                let mut w = std::io::BufWriter::with_capacity(256 * 1024, file);
                writeln!(
                    w,
                    "read_id,barcode,confidence{}",
                    if scored {
                        ",crf_logp,crf_margin,crf_best,mean_logpost"
                    } else {
                        ""
                    }
                )?;
                Some(w)
            }
            None => None,
        };
        let mut collected = collect.then(|| Collected {
            crf: scored.then(|| CollectedScores {
                names: names.clone(),
                ..Default::default()
            }),
            ..Default::default()
        });
        for row in rx.iter() {
            let ClassRow {
                read_id,
                barcode,
                confidence,
                crf,
            } = row;
            if let Some(w) = &mut writer {
                write!(w, "{read_id},{barcode},{confidence:.6}")?;
                if scored {
                    // Empty rather than absent for a read that never decoded:
                    // the column count has to stay constant, and a zero would
                    // read as a confident score of 1.0.
                    match crf {
                        Some(c) => writeln!(
                            w,
                            ",{:.4},{},{},{:.4}",
                            c.logp,
                            c.margin.map(|m| format!("{m:.4}")).unwrap_or_default(),
                            names.get(c.best as usize).map_or("", String::as_str),
                            c.mean_logpost,
                        )?,
                        None => writeln!(w, ",,,,")?,
                    }
                } else {
                    writeln!(w)?;
                }
            }
            if let Some(out) = &mut collected {
                out.barcode.insert(read_id, barcode);
                // Recorded for gated reads too, matching the CSV: a sidecar
                // that says `unclassified` should still say what it was
                // rejected for.
                if let (Some(dst), Some(c)) = (out.crf.as_mut(), crf) {
                    dst.best.insert(read_id, c.best);
                    dst.logp.insert(read_id, c.logp);
                    if let Some(m) = c.margin {
                        dst.margin.insert(read_id, m);
                    }
                    dst.mean_logpost.insert(read_id, c.mean_logpost);
                }
            }
        }
        if let Some(w) = &mut writer {
            w.flush()?;
        }
        Ok(collected)
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
/// What a CRF bundle pins for boundary detection, lifted out of the sidecar
/// with the ONNX path resolved against the bundle directory.
struct BoundaryPin {
    method: String,
    onnx: Option<PathBuf>,
    input: Option<escapepod_demux::crf::BoundaryInputSpec>,
    sha256: Option<String>,
}

fn build_detector(
    args: &RunArgs,
    pin: Option<BoundaryPin>,
    device: crate::device::Device,
) -> anyhow::Result<Detector> {
    let (pinned_method, pinned_onnx, pinned_input, pinned_sha) = match pin {
        Some(p) => (Some(p.method), p.onnx, p.input, p.sha256),
        None => (None, None, None, None),
    };
    let pinned_method = pinned_method.as_deref();
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
        "llr" => {
            // Not a placement: LLR is a CPU changepoint search with no device
            // path, so there is nothing for `place` to decide.
            crate::device::note_cpu_only(
                device,
                "`--method llr`",
                "the LLR detector has no GPU path. `--method cnn` is the detector \
                 that runs on the device, and it is also the one the shipped barcode \
                 models were measured against.",
            );
            Ok(Detector::Llr {
                min_adapter: args.min_adapter,
                border_trim: args.border_trim,
                downscale: args.downscale.max(1),
            })
        }
        "cnn" => {
            #[cfg(feature = "cnn-detect")]
            {
                // Decided before the ONNX file is opened so the CPU-cost warning
                // lands at second one, not after a 37-minute run (#270).
                let on_gpu =
                    crate::device::place_and_report(device, crate::device::Stage::CnnDetect)?
                        .is_gpu();
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
                // The bundle names the exact bytes it pinned; refuse to run
                // different ones (a truncated fetch, a hand-edited bundle).
                // Only when the pinned file is what loads — an explicit
                // --cnn-model already chose different weights deliberately.
                #[cfg(feature = "model-fetch")]
                if args.cnn_model.is_none()
                    && let (Some(expect), Some(p)) = (pinned_sha.as_deref(), pinned_onnx.as_ref())
                {
                    verify_pinned_sha256(p, expect)?;
                }
                #[cfg(not(feature = "model-fetch"))]
                if pinned_sha.is_some() && args.cnn_model.is_none() {
                    tracing::warn!(
                        "the bundle declares its pinned boundary model's sha256, but this \
                         build lacks `model-fetch` and cannot verify it"
                    );
                }
                // Prep with the geometry the bundle declares for its pinned
                // model (#187) — but only when that model is what's running.
                // An explicit --cnn-model is a different set of weights, whose
                // geometry the bundle cannot speak for; those get the legacy
                // defaults, as does a bundle from before the contract existed
                // (the defaults are what its model trained with).
                let config = match (&args.cnn_model, pinned_input) {
                    (None, Some(spec)) => {
                        escapepod_demux::AdapterCnnConfig::from_bundle_input(&spec)
                            .map_err(|e| anyhow::anyhow!("bundle boundary.input: {e}"))?
                    }
                    _ => escapepod_demux::AdapterCnnConfig::default(),
                };
                // GPU detection is one batched onnxruntime call per block.
                #[cfg(feature = "gpu")]
                if on_gpu {
                    return Ok(Detector::CnnGpu(Box::new(
                        escapepod_demux::AdapterCnnGpu::load_with_config(path, config)
                            .map_err(|e| anyhow::anyhow!("loading CNN model on GPU: {e}"))?,
                    )));
                }
                // True only on the branch above, which returned. Reading it
                // here keeps the binding live in a `cnn-detect`-without-`gpu`
                // build and states the invariant in one place.
                debug_assert!(!on_gpu, "GPU detection reached the CPU loader");
                Ok(Detector::Cnn(Box::new(
                    escapepod_demux::AdapterCnn::load_with_config(path, config)
                        .map_err(|e| anyhow::anyhow!("loading CNN model: {e}"))?,
                )))
            }
            #[cfg(not(feature = "cnn-detect"))]
            {
                let _ = (pinned_onnx, pinned_input, pinned_sha, device);
                anyhow::bail!("--method cnn requires a build with `--features cnn-detect`")
            }
        }
        other => anyhow::bail!("unknown --method `{other}`; expected `llr` or `cnn`"),
    }
}

/// Refuse to run pinned boundary weights whose bytes are not the ones the
/// bundle was built with. The pinned copy is the one bundle file the registry
/// manifest does not hash, which is why the sidecar declares it
/// (escapepod-models#56); a mismatch means a corrupt or edited bundle, and
/// running it anyway would silently degrade every downstream barcode call.
#[cfg(all(feature = "cnn-detect", feature = "model-fetch"))]
fn verify_pinned_sha256(path: &Path, expect: &str) -> anyhow::Result<()> {
    use sha2::{Digest, Sha256};
    let bytes = std::fs::read(path)
        .map_err(|e| anyhow::anyhow!("reading pinned boundary model {}: {e}", path.display()))?;
    // Same formatting as models.rs::sha256_hex, not a call to it: that helper
    // sits behind `demux-models`, a feature this path does not otherwise need.
    let digest = Sha256::digest(&bytes);
    let mut got = String::with_capacity(digest.len() * 2);
    for b in digest {
        use std::fmt::Write;
        let _ = write!(got, "{b:02x}");
    }
    if !got.eq_ignore_ascii_case(expect) {
        anyhow::bail!(
            "pinned boundary model {} hashes {got} but the bundle declares {expect}: the \
             bundle is corrupt or was edited after it was built; re-fetch it (`escpod demux \
             models fetch`)",
            path.display()
        );
    }
    Ok(())
}

#[cfg(all(test, feature = "cnn-detect", feature = "model-fetch"))]
mod pinned_sha_tests {
    use super::*;

    #[test]
    fn verify_accepts_the_built_bytes_and_refuses_others() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("adapter.onnx");
        std::fs::write(&p, b"weights").unwrap();

        // sha256("weights"), computed independently of the code under test.
        let expect = "9a129038d9a00aed0cf6a7ea059ca50a813449061ab87848cf1a13eafdf33b2c";
        verify_pinned_sha256(&p, expect).expect("matching bytes verify");
        verify_pinned_sha256(&p, &expect.to_uppercase()).expect("hex case is not a mismatch");

        let err = verify_pinned_sha256(&p, "deadbeef")
            .unwrap_err()
            .to_string();
        assert!(err.contains("declares deadbeef"), "unhelpful error: {err}");

        let err = verify_pinned_sha256(&dir.path().join("missing.onnx"), expect)
            .unwrap_err()
            .to_string();
        assert!(err.contains("reading pinned boundary model"), "{err}");
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

#[cfg(all(test, feature = "gpu"))]
mod gpu_placement_tests {
    use super::{crf_encoder_devices, crf_gpu_workers};

    /// Pins today's placement, including the part that is known mis-tuned: at
    /// two devices the encoder pool is `[1]` alone, which is why the second GPU
    /// measures 1.11x rather than ~2x. Written so that changing the policy has
    /// to change this test deliberately rather than silently.
    #[test]
    fn device_zero_is_reserved_for_detection_above_one_device() {
        assert_eq!(crf_encoder_devices(1), vec![0]);
        assert_eq!(crf_encoder_devices(2), vec![1]);
        assert_eq!(crf_encoder_devices(4), vec![1, 2, 3]);
        // A zero count can only come from a failed probe; never hand back an
        // empty pool, since the caller indexes into it.
        assert_eq!(crf_encoder_devices(0), vec![0]);
    }

    /// The pool must divide evenly over the devices it is spread across, or the
    /// per-device row budget is computed against the wrong denominator.
    #[test]
    fn workers_spread_evenly_over_the_encoder_devices() {
        if std::env::var("ESCAPEPOD_CRF_GPU_WORKERS").is_ok() {
            return;
        }
        for visible in [1usize, 2, 4] {
            let devices = crf_encoder_devices(visible);
            let workers = crf_gpu_workers(devices.len());
            assert_eq!(
                workers % devices.len(),
                0,
                "{workers} workers do not divide over {} devices",
                devices.len()
            );
            // Two per device is what overlaps one worker's per-call setup with
            // another's device work; the row budget is sized for it.
            assert_eq!(workers / devices.len(), 2);
        }
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
