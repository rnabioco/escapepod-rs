//! Detect subcommand - LLR-based adapter boundary detection.

use super::utils::{process_reads_par, total_read_count};
use crate::progress::create_progress_bar;
use crate::style;
use escapepod_demux::ReadBoundaries;
use escapepod_signal::segmentation::{SignalPrepScratch, detect_adapter, downscale_normalize_into};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use tracing::info;
#[cfg(feature = "cnn-detect")]
use tracing::warn;
#[cfg(feature = "cnn-detect")]
use uuid::Uuid;

/// Arguments for the detect subcommand.
#[derive(Debug, clap::Args)]
pub struct DetectArgs {
    /// Input POD5 file(s)
    #[arg(required = true, value_name = "FILES")]
    pub input: Vec<PathBuf>,

    /// Output CSV file for detected boundaries
    #[arg(short, long, required = true, value_name = "FILE")]
    pub output: PathBuf,

    /// Minimum observations for adapter segment
    #[arg(
        long,
        default_value = "200",
        value_name = "N",
        help_heading = "Advanced Options"
    )]
    pub min_adapter: usize,

    /// Border trim size
    #[arg(
        long,
        default_value = "50",
        value_name = "N",
        help_heading = "Advanced Options"
    )]
    pub border_trim: usize,

    /// Downscale factor for signal processing. Default 10 is the
    /// WarpDemuX-native mode; set 1 for full-resolution (no downscaling).
    #[arg(
        long,
        default_value = "10",
        value_name = "N",
        help_heading = "Advanced Options"
    )]
    pub downscale: usize,

    /// Adapter detection method.
    ///
    /// `llr` (default) uses the built-in log-likelihood ratio detector.
    /// `cnn` runs a boundary-CNN ONNX graph through tract-onnx (in the default
    /// build). Supply any model with the `[B,1,L] -> [B,2,L]`
    /// contract via `--cnn-model` — e.g. escapepod-models' `adapter_rna004`
    /// (CC-BY), or an ADAPTed `BoundariesCNN` exported with
    /// `scripts/export_adapter_cnn_to_onnx.py` (those weights are CC BY-NC 4.0
    /// and not bundled). Runs batched through onnxruntime CUDA when a GPU is
    /// available (`--device auto`, the default, on a `--features gpu`
    /// build); otherwise per-read through tract on the CPU.
    ///
    /// **No default** — LLR is opt-in, never inferred. It costs 17.2 points of
    /// downstream barcode recall against the same classifier (0.9928 -> 0.8196,
    /// escapepod-models#16) and the failure is silent: it runs and produces
    /// plausible-looking boundaries.
    #[arg(long, value_name = "{cnn,llr}", help_heading = "Advanced Options")]
    pub method: Option<String>,

    /// Path to the boundary-CNN ONNX model (only used with `--method cnn`).
    #[cfg(feature = "cnn-detect")]
    #[arg(long, value_name = "FILE", help_heading = "Advanced Options")]
    pub cnn_model: Option<PathBuf>,

    /// Named boundary model resolved from the local cache, e.g.
    /// `adapter_rna004`. Prefetch it with `escpod demux models fetch`;
    /// resolution never touches the network. Mutually exclusive with
    /// `--cnn-model`.
    #[cfg(all(feature = "cnn-detect", feature = "demux-models"))]
    #[arg(long, value_name = "NAME", help_heading = "Advanced Options")]
    pub cnn_model_name: Option<String>,

    /// Where `--method cnn` inference runs (`auto` by default, which prefers
    /// the GPU here — CNN detection is the path that pays off most).
    #[command(flatten)]
    pub device: crate::device::DeviceArgs,

    /// Also run the LLR detector and emit its boundaries alongside the CNN's
    /// (`--method cnn` only).
    ///
    /// Adds `llr_adapter_start`, `llr_adapter_end`, and `end_delta` columns, so
    /// the two independent detectors can be compared per read. This is the only
    /// boundary quality gate that works in production, where EDX labels are not
    /// available.
    ///
    /// Opt-in because of I/O, not compute: LLR itself is nearly free next to CNN
    /// inference, but it normalizes over the whole read, so the CNN path can no
    /// longer decode just its leading `max_obs_trace` samples. Every read is
    /// decompressed in full while this is on.
    #[cfg(feature = "cnn-detect")]
    #[arg(long, help_heading = "Advanced Options")]
    pub emit_llr_delta: bool,

    /// Number of threads for parallel processing (default: 16, or all available CPUs if fewer)
    #[arg(short = 't', long, visible_short_alias = 'j', value_name = "N")]
    pub threads: Option<usize>,

    /// Print per-phase timing breakdown after completion
    #[arg(long)]
    pub profile: bool,
}

/// One output row: the CNN's boundaries, plus the LLR arm's `(start, end)` when
/// `--emit-llr-delta` is on.
#[cfg(feature = "cnn-detect")]
type DetectRow = (ReadBoundaries, Option<(usize, usize)>);

/// LLR adapter boundaries for one read, in full-resolution coordinates.
///
/// Shared by `--method llr` and by the `--emit-llr-delta` arm of `--method cnn`,
/// and shared deliberately: the delta is only meaningful if its LLR side is the
/// same detector `--method llr` reports, so the two must not be able to drift
/// apart. Calling one function is a stronger guarantee of that than two copies
/// that agree today.
fn llr_boundaries(
    signal: &[i16],
    (min_adapter, border_trim, downscale): (usize, usize, usize),
) -> (usize, usize) {
    // Per-worker scratch via a thread-local: `process_reads_par` takes an `Fn`,
    // so there is no init hook to thread buffers through. The unfused
    // normalize+downscale chain allocates three full-length f32 buffers per
    // read, and the read-length tail (medians of ~8 k with maxima in the
    // millions) makes that a real RSS spike.
    thread_local! {
        static PREP: std::cell::RefCell<(SignalPrepScratch, Vec<f32>)> =
            const { std::cell::RefCell::new((SignalPrepScratch::new(), Vec::new())) };
    }

    // A 0 or 1 factor both mean "no downscaling"; folding them together here is
    // what keeps the two callers from disagreeing on `--downscale 0`.
    let scale_factor = downscale.max(1);

    let (start, end) = PREP.with(|cell| {
        let (prep, processed) = &mut *cell.borrow_mut();
        downscale_normalize_into(signal, scale_factor, prep, processed);
        detect_adapter(
            processed,
            (min_adapter / scale_factor).max(1),
            (border_trim / scale_factor).max(1),
        )
    });
    (start * scale_factor, end * scale_factor)
}

/// Wall-clock spent in — and waiting between — the GPU pipeline's stages,
/// reported by `--profile`.
///
/// The producer and the GPU consumer run concurrently, so these do not sum to
/// the run: what they answer is *which* stage the other is waiting on.
/// `index`/`read_decode`/`prep`/`gpu_infer` are time doing work; `*_blocked` is
/// a stage stalled because the next one is behind, and `gpu_wait` is the GPU
/// starved because the producer is. #239 was filed after inferring an idle GPU
/// from `nvidia-smi` sampling, which is what this replaces.
///
/// `read_decode` and `prep` are summed across the rayon workers that ran them,
/// so they measure CPU time and can exceed the wall-clock the producer took.
#[cfg(feature = "gpu")]
#[derive(Default)]
struct StageTimes {
    /// Producer: reading the reads table and sorting it, per file. Wall.
    index: Stage,
    /// Producer: per-read signal extraction + bounded VBZ decode. Summed over
    /// workers.
    read_decode: Stage,
    /// Producer: `i16 -> f32` and the CNN's prep. Summed over workers.
    prep: Stage,
    /// Producer: one block's parallel decode+prep. Wall.
    block: Stage,
    /// Producer stalled — the GPU has not drained a block.
    block_blocked: Stage,
    /// GPU starved — no prepped block yet. This is what #239 measured as an
    /// idle GPU.
    gpu_wait: Stage,
    /// GPU: batched onnxruntime inference.
    gpu_infer: Stage,
}

#[cfg(feature = "gpu")]
#[derive(Default)]
struct Stage(std::sync::atomic::AtomicU64);

#[cfg(feature = "gpu")]
impl Stage {
    /// Add the time since `since` to this stage's total.
    fn add(&self, since: std::time::Instant) {
        self.add_nanos(since.elapsed().as_nanos() as u64);
    }

    fn add_nanos(&self, nanos: u64) {
        self.0
            .fetch_add(nanos, std::sync::atomic::Ordering::Relaxed);
    }

    fn secs(&self) -> f64 {
        self.0.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e9
    }
}

#[cfg(feature = "gpu")]
impl StageTimes {
    fn report(&self, enabled: bool) {
        if !enabled {
            return;
        }
        eprintln!();
        eprintln!("{}", style::action("GPU pipeline"));
        for (name, s, wall) in [
            ("index (reads table)", &self.index, true),
            ("read + decode (cpu-time)", &self.read_decode, false),
            ("prep (cpu-time)", &self.prep, false),
            ("producer block", &self.block, true),
            ("producer blocked on GPU", &self.block_blocked, true),
            ("GPU starved for blocks", &self.gpu_wait, true),
            ("GPU inference", &self.gpu_infer, true),
        ] {
            eprintln!(
                "  {:<30} {:>8.2}s{}",
                name,
                s.secs(),
                if wall { "" } else { " (summed)" }
            );
        }
    }
}

/// Run the detect subcommand.
pub fn run(args: DetectArgs) -> anyhow::Result<()> {
    let Some(method) = args.method.clone() else {
        anyhow::bail!(
            "--method {{cnn,llr}} is required: LLR is never chosen for you. Use \
             `--method cnn --cnn-model <FILE>` for the accuracy the shipped barcode \
             models were measured at, or `--method llr` to opt into the classical \
             detector (17.2 points worse on downstream barcode recall — \
             escapepod-models#16)."
        );
    };
    // Resolved once, here, so the `--gpu` deprecation warning is emitted exactly
    // once no matter which arm runs.
    let device = args.device.resolve();
    match method.as_str() {
        "llr" => {
            crate::device::note_cpu_only(
                device,
                "`--method llr`",
                "the LLR detector is a CPU changepoint search with no GPU path. \
                 `--method cnn` is the detector that runs on the device.",
            );
            run_llr(args)
        }
        "cnn" => {
            #[cfg(feature = "cnn-detect")]
            {
                run_cnn(args, device)
            }
            #[cfg(not(feature = "cnn-detect"))]
            {
                let _ = args;
                let _ = device;
                anyhow::bail!(
                    "--method cnn requires a build with `--features cnn-detect`. \
                     Rebuild with: cargo build --release -p escapepod-cli \
                     --features \"demux cnn-detect\"."
                );
            }
        }
        other => anyhow::bail!("unknown --method `{other}`; expected `llr` or `cnn`"),
    }
}

/// Run the detect subcommand using LLR boundary detection.
fn run_llr(args: DetectArgs) -> anyhow::Result<()> {
    use crate::commands::profile::PhaseTimer;
    let mut timer = PhaseTimer::new();
    timer.phase("Detect adapters");
    let profile = args.profile;
    info!(
        "{} adapter boundaries using LLR algorithm",
        style::action("Detecting"),
    );
    info!(
        "{} {} POD5 file(s)",
        style::label("Input:"),
        style::count(args.input.len())
    );
    info!(
        "{} {}",
        style::label("Output:"),
        style::path(args.output.display())
    );
    info!(
        "{} min_adapter={}, border_trim={}, downscale={}",
        style::label("Parameters:"),
        style::value(args.min_adapter),
        style::value(args.border_trim),
        style::value(args.downscale)
    );

    // Set thread pool size

    let total = total_read_count(&args.input);
    info!(
        "{} {} reads to process",
        style::label("Found:"),
        style::count(total)
    );

    let progress_bar = create_progress_bar(total as u64, "Detecting")?;

    let downscale_factor = args.downscale.max(1);
    let min_adapter = args.min_adapter;
    let border_trim = args.border_trim;

    let results: Vec<ReadBoundaries> = process_reads_par(
        &args.input,
        Some(&progress_bar),
        None, // LLR scans the full read
        |read_id, num_samples, signal| {
            let (adapter_start, adapter_end) =
                llr_boundaries(signal, (min_adapter, border_trim, downscale_factor));

            // `llr_boundaries` already rescales to full-resolution coordinates.
            ReadBoundaries {
                read_id,
                num_samples,
                adapter_start,
                adapter_end,
            }
        },
    )?;

    progress_bar.finish_with_message("complete");

    // Write results
    let output_file = File::create(&args.output)?;
    let mut writer = BufWriter::with_capacity(256 * 1024, output_file);

    writeln!(writer, "read_id,num_samples,adapter_start,adapter_end")?;

    let mut detected_count = 0;
    for boundaries in &results {
        writeln!(
            writer,
            "{},{},{},{}",
            boundaries.read_id,
            boundaries.num_samples,
            boundaries.adapter_start,
            boundaries.adapter_end
        )?;
        if boundaries.has_valid_adapter() {
            detected_count += 1;
        }
    }

    writer.flush()?;

    info!(
        "{} boundaries written to {}",
        style::action("Detected"),
        style::path(args.output.display())
    );
    info!(
        "{} {} reads with detected adapters",
        style::label("Result:"),
        style::count(detected_count)
    );

    timer.report(profile);

    Ok(())
}

/// Run the detect subcommand using a boundary-CNN ONNX model (opt-in).
///
/// CPU runs the model one read at a time through tract-onnx; the GPU placement
/// (the default under `--device auto` on a `gpu` build) runs it batched
/// through onnxruntime's CUDA execution provider, which is where the large
/// speedup lives — the TCN is inference-bound and tract has no efficient batched
/// conv. Works with any model on the `[B,1,L] -> [B,2,L]` contract —
/// escapepod-models' `adapter_rna004` (CC-BY) or an ADAPTed `BoundariesCNN`
/// export (CC BY-NC; not bundled). See `scripts/export_adapter_cnn_to_onnx.py`.
#[cfg(feature = "cnn-detect")]
fn run_cnn(args: DetectArgs, device: crate::device::Device) -> anyhow::Result<()> {
    use crate::commands::profile::PhaseTimer;
    use escapepod_demux::AdapterCnnError;
    use std::sync::atomic::{AtomicUsize, Ordering};

    let mut timer = PhaseTimer::new();
    timer.phase("Detect adapters (CNN)");
    let profile = args.profile;

    #[cfg(feature = "demux-models")]
    let resolved_model;
    #[cfg(feature = "demux-models")]
    let cnn_model_path = match (&args.cnn_model, &args.cnn_model_name) {
        (Some(_), Some(_)) => {
            anyhow::bail!("pass either --cnn-model or --cnn-model-name, not both")
        }
        (None, None) => anyhow::bail!(
            "--method cnn requires a model: --cnn-model <FILE> or --cnn-model-name <NAME> \
             (see 'escpod demux models list')"
        ),
        (Some(path), None) => path,
        (None, Some(name)) => {
            resolved_model = super::models::resolve(name)?;
            &resolved_model
        }
    };
    #[cfg(not(feature = "demux-models"))]
    let cnn_model_path = args
        .cnn_model
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("--method cnn requires --cnn-model <FILE>"))?;

    // Decided (and reported) before the model is even opened: the whole point of
    // the warning is that it beats the 37-minute CPU run rather than trailing it.
    //
    // The GPU producer prepares reads through `get_signal_prefix`, so the whole
    // signal an LLR arm would need is never decoded there. Silently running LLR
    // on the prefix would report a delta that measures truncation instead of
    // detector disagreement, so the two cannot be combined — but under `auto`
    // that means CPU detection, not a failed run.
    let use_gpu = if args.emit_llr_delta {
        crate::device::place_ruled_out(
            device,
            crate::device::Stage::CnnDetect,
            "`--emit-llr-delta` needs whole-read signal, and the GPU producer decodes \
             only each read's leading samples",
        )?
        .is_gpu()
    } else {
        crate::device::place_and_report(device, crate::device::Stage::CnnDetect)?.is_gpu()
    };

    warn!(
        "boundary CNN runs the model you supply via --cnn-model; respect that \
         model's license (e.g. ADAPTed-derived weights are CC BY-NC 4.0).",
    );

    info!(
        "{} adapter boundaries using boundary CNN ({})",
        style::action("Detecting"),
        if use_gpu { "GPU" } else { "CPU" },
    );
    info!(
        "{} {} POD5 file(s)",
        style::label("Input:"),
        style::count(args.input.len())
    );
    info!(
        "{} {}",
        style::label("Model:"),
        style::path(cnn_model_path.display())
    );
    info!(
        "{} {}",
        style::label("Output:"),
        style::path(args.output.display())
    );

    let total = total_read_count(&args.input);
    info!(
        "{} {} reads to process",
        style::label("Found:"),
        style::count(total)
    );

    let progress_bar = create_progress_bar(total as u64, "Detecting (CNN)")?;

    // Count failures so a broken model surfaces loudly instead of silently
    // writing adapter_end=0 for every read (the v1.0.0 static-shape trap).
    let too_short = AtomicUsize::new(0);
    let failed = AtomicUsize::new(0);
    // ADAPTed's CNN sets adapter_start=0 always — this path is single-ended.
    let boundaries = |read_id: Uuid, num_samples: u64, end: Result<usize, AdapterCnnError>| {
        let adapter_end = match end {
            Ok(e) => e,
            Err(AdapterCnnError::SignalTooShort { .. }) => {
                too_short.fetch_add(1, Ordering::Relaxed);
                0
            }
            Err(_) => {
                failed.fetch_add(1, Ordering::Relaxed);
                0
            }
        };
        ReadBoundaries {
            read_id,
            num_samples,
            adapter_start: 0,
            adapter_end,
        }
    };

    let results: Vec<DetectRow> = if use_gpu {
        // GPU: a dedicated consumer thread builds the ort session — overlapping
        // the (multi-second) CUDA/cuDNN init with the producers' first decodes —
        // then runs each prepped block batched. CPU producers use the full pool
        // to decode + prep in parallel and feed blocks through a small bounded
        // channel (double-buffered, so the GPU isn't starved).
        #[cfg(feature = "gpu")]
        {
            use escapepod_demux::AdapterCnnConfig;
            use escapepod_signal::Reader;
            use rayon::prelude::*;
            use std::sync::mpsc::sync_channel;

            // Reads per block: bigger ⇒ bigger same-length groups ⇒ fewer, larger
            // onnxruntime calls. Bounded at 2 in flight to cap memory.
            const GPU_BLOCK: usize = 16_384;
            let cfg = AdapterCnnConfig::default();
            type Block = (
                Vec<(Uuid, u64)>,
                Vec<Option<escapepod_demux::PreppedWindow>>,
            );
            let (tx, rx) = sync_channel::<Block>(2);

            let model_path = cnn_model_path.clone();
            let pb = &progress_bar;
            let bnd = &boundaries;
            let stages = StageTimes::default();
            let st = &stages;
            let rows = std::thread::scope(|scope| -> anyhow::Result<Vec<DetectRow>> {
                let gpu_handle = scope.spawn(move || -> anyhow::Result<Vec<DetectRow>> {
                    // Each session pins its own onnxruntime thread pool to one
                    // non-spinning thread — see `AdapterCnnGpu`. Left at ORT's
                    // default it is `available_parallelism()` wide, spawned
                    // alongside rayon's, so `--threads` would not bound the
                    // process (#155, GPU half) and the pool would take a third
                    // of the CPU from the producers (#239). That bound is per
                    // session, so it holds as the device pool grows.
                    //
                    // One session per visible device. Loading them here rather
                    // than on the main thread is what it always was — the point
                    // of this thread is to overlap CUDA/cuDNN init with the
                    // producers' first decodes, and an N-device pool has N times
                    // as much of it to hide.
                    let devices = super::run::cnn_detect_devices(super::run::visible_devices());
                    if devices.len() > 1 {
                        tracing::info!(
                            "{} boundary CNN on GPU {:?}",
                            crate::style::label("Device:"),
                            devices
                        );
                    }
                    let gpu = escapepod_demux::AdapterCnnGpuPool::load_on_devices(
                        &model_path,
                        escapepod_demux::AdapterCnnConfig::default(),
                        &devices,
                    )
                    .map_err(|e| anyhow::anyhow!("loading CNN model on GPU: {e}"))?;
                    let mut out = Vec::new();
                    loop {
                        let waited = std::time::Instant::now();
                        let Ok((meta, prepped)) = rx.recv() else {
                            break;
                        };
                        st.gpu_wait.add(waited);
                        let inferring = std::time::Instant::now();
                        let ends = gpu.detect_prepped(&prepped);
                        st.gpu_infer.add(inferring);
                        pb.inc(meta.len() as u64);
                        for (end, (read_id, num_samples)) in ends.into_iter().zip(meta) {
                            // `--emit-llr-delta` never reaches the GPU path
                            // (see `place_ruled_out` above), so the LLR arm is
                            // always absent here.
                            out.push((bnd(read_id, num_samples, end), None));
                        }
                    }
                    // Keep the ORT session alive past process exit, for the
                    // same reason the fused `demux` GPU path does — onnxruntime's
                    // CUDA provider
                    // reads freed memory during onnxruntime's own at-exit
                    // teardown, and glibc aborts on it. See the long comment in
                    // `run.rs` and pykeio/ort#609. `release_env_on_exit` only
                    // fires when the last `Arc<Environment>` drops and every live
                    // `Session` holds one, so leaking this keeps the count above
                    // zero. All output is produced from `out`, not from `gpu`.
                    std::mem::forget(gpu);
                    Ok(out)
                });

                // Producers: per file, sort reads by length, then decode +
                // i16→f32 + prep in parallel and push blocks. MAD-norm is
                // scale-invariant, so raw i16 → f32 matches the pA path
                // bit-for-bit post-normalization.
                //
                // The sort's original purpose is gone: it grouped reads so each
                // block shared a prepped length and formed big exact-length GPU
                // groups, and #187 made prep emit one fixed length for every
                // read, so that grouping is now automatic. It is kept anyway,
                // because removing it was measured and is not an improvement.
                //
                // Dropping it looks like it should help — sorting scatters the
                // per-read `get_signal_prefix` across the file, which is the
                // access pattern #72 measured at 0.3 MB/s against 288 MB/s for
                // one ascending sweep. It does not: over 503 k reads, warmed and
                // interleaved, unsorted ran 18.0/15.9/16.7 s against sorted
                // 18.3/17.3/16.6 s — within-arm spread larger than the
                // difference. (An earlier cold-cache run showed 185.7 s -> 21.6 s
                // and meant nothing; see `DEFAULT_FILLERS` for the same trap.)
                //
                // And it is not free to change: read order decides which reads
                // share a device batch, and that changes results. Same binary
                // twice is bit-identical, but sorted vs unsorted disagreed on
                // **7 of 503,076** boundaries — cuDNN's batch-shape-dependent
                // kernel choice, not a bug here. No gain is worth perturbing
                // output for.
                for path in &args.input {
                    let indexing = std::time::Instant::now();
                    let reader = Reader::open(path)?;
                    let mut reads: Vec<_> = reader
                        .reads()?
                        .filter_map(Result::ok)
                        .filter(|r| !r.signal_rows.is_empty())
                        .collect();
                    reads.sort_by_key(|r| r.num_samples);
                    let extractor = reader.signal_extractor()?;
                    st.index.add(indexing);
                    for window in reads.chunks(GPU_BLOCK) {
                        let filling = std::time::Instant::now();
                        let prepped: Vec<(Uuid, u64, Option<escapepod_demux::PreppedWindow>)> =
                            window
                                .par_iter()
                                .map(|r| {
                                    // Only the leading `max_obs_trace` samples feed the
                                    // CNN; skip decompressing the rest (matters for long
                                    // mRNA reads).
                                    let decoding = std::time::Instant::now();
                                    let signal = extractor
                                        .get_signal_prefix(&r.signal_rows, cfg.max_obs_trace)
                                        .ok();
                                    let decoded = decoding.elapsed();
                                    let p = signal.and_then(|s| {
                                        let f: Vec<f32> = s.iter().map(|&x| x as f32).collect();
                                        cfg.prep(&f)
                                    });
                                    // Summed across workers, so the split between
                                    // I/O+decode and prep survives the parallelism.
                                    st.read_decode.add_nanos(decoded.as_nanos() as u64);
                                    st.prep
                                        .add_nanos((decoding.elapsed() - decoded).as_nanos() as u64);
                                    (r.read_id, r.num_samples, p)
                                })
                                .collect();
                        st.block.add(filling);
                        let mut meta = Vec::with_capacity(prepped.len());
                        let mut preps = Vec::with_capacity(prepped.len());
                        for (id, ns, p) in prepped {
                            meta.push((id, ns));
                            preps.push(p);
                        }
                        let sending = std::time::Instant::now();
                        let sent = tx.send((meta, preps)).is_ok();
                        st.block_blocked.add(sending);
                        if !sent {
                            break; // GPU thread died; its error surfaces at join
                        }
                    }
                }
                drop(tx);
                gpu_handle
                    .join()
                    .map_err(|_| anyhow::anyhow!("GPU detection thread panicked"))?
            })?;
            stages.report(profile);
            rows
        }
        #[cfg(not(feature = "gpu"))]
        unreachable!("device placement never returns GPU without the gpu feature")
    } else {
        // CPU: per-read tract. tract has no efficient batched conv (batching it
        // measured *slower*), so the fine-grained per-read parallelism across
        // many reads is the better CPU schedule.
        let cnn = escapepod_demux::AdapterCnn::load(cnn_model_path)
            .map_err(|e| anyhow::anyhow!("loading CNN model: {e}"))?;
        // The CNN alone needs only the leading `max_obs_trace` samples, which is
        // what lets long mRNA reads skip most of their decompression. The LLR
        // arm normalizes over the *whole* read, so with `--emit-llr-delta` that
        // saving has to go: scoring LLR on the CNN's prefix would not be the
        // detector `--method llr` runs, and the delta would measure the
        // truncation rather than the disagreement.
        let decode_bound = if args.emit_llr_delta {
            None
        } else {
            Some(cnn.config().max_obs_trace)
        };
        let llr_params = (args.min_adapter, args.border_trim, args.downscale);
        process_reads_par(
            &args.input,
            Some(&progress_bar),
            decode_bound,
            |read_id, num_samples, signal| {
                let sig_f32: Vec<f32> = signal.iter().map(|&s| s as f32).collect();
                let b = boundaries(read_id, num_samples, cnn.detect_adapter_end(&sig_f32));
                let llr = args
                    .emit_llr_delta
                    .then(|| llr_boundaries(signal, llr_params));
                (b, llr)
            },
        )?
    };

    progress_bar.finish_with_message("complete");

    // Every read failing on the GPU is not a per-read problem — a batched
    // onnxruntime call has no per-read reason to fail uniformly — so it means the
    // device path is broken, and the boundaries CSV about to be written would be
    // `adapter_end=0` from top to bottom. Downstream reads that file happily; it
    // is the silent-failure this command's `--device` work exists to remove, so
    // it is an error, not a warning plus a useless artifact.
    //
    // Checked *here*, before `File::create`, so a failed run leaves no output
    // behind to be mistaken for a real one.
    //
    // The likely cause is named because ort cannot name it: registering the CUDA
    // execution provider only appends a factory to the session options, and
    // `error_on_failure` catches exactly that step. The runtime libraries the
    // kernels need are dlopened later, inside the first `Conv`, and a missing
    // libcudnn surfaces as `NOT_IMPLEMENTED : cuDNN is unavailable` per node with
    // the session already built and `--device gpu` already satisfied.
    //
    // GPU only. On CPU, tract runs one read at a time and an all-fail can
    // legitimately be a property of the input (every read too short, say), so
    // that path keeps the warning it has always had.
    let failures = failed.load(Ordering::Relaxed);
    if use_gpu && failures > 0 && failures == results.len() {
        anyhow::bail!(
            "every one of the {failures} read(s) failed CNN inference on the GPU. \
             The CUDA execution provider registered, so this is a runtime library \
             the kernels could not load — typically libcudnn or libcublasLt. Run \
             inside the pixi `gpu` environment so `LD_LIBRARY_PATH` includes them \
             (see docs/cli/demux.md), or pass `--device cpu`. Re-run with \
             `RUST_LOG=ort=error` to see onnxruntime's own message. No boundaries \
             file was written."
        );
    }

    let output_file = File::create(&args.output)?;
    let mut writer = BufWriter::with_capacity(256 * 1024, output_file);
    if args.emit_llr_delta {
        writeln!(
            writer,
            "read_id,num_samples,adapter_start,adapter_end,llr_adapter_start,llr_adapter_end,end_delta"
        )?;
    } else {
        writeln!(writer, "read_id,num_samples,adapter_start,adapter_end")?;
    }

    let mut detected = 0;
    let mut abs_deltas: Vec<u64> = Vec::new();
    for (b, llr) in &results {
        write!(
            writer,
            "{},{},{},{}",
            b.read_id, b.num_samples, b.adapter_start, b.adapter_end
        )?;
        if let Some((llr_start, llr_end)) = llr {
            // `adapter_end == 0` is the shared "no adapter / too short /
            // inference failed" sentinel on both arms. Subtracting it would give
            // the distance from a position to a sentinel, which reads as a huge
            // disagreement and is not one — so `end_delta` is left empty unless
            // both detectors actually found a boundary.
            if b.adapter_end > 0 && *llr_end > 0 {
                let delta = b.adapter_end as i64 - *llr_end as i64;
                abs_deltas.push(delta.unsigned_abs());
                write!(writer, ",{llr_start},{llr_end},{delta}")?;
            } else {
                write!(writer, ",{llr_start},{llr_end},")?;
            }
        }
        writeln!(writer)?;
        if b.has_valid_adapter() {
            detected += 1;
        }
    }
    writer.flush()?;

    let too_short = too_short.into_inner();
    let failed = failures;
    if too_short > 0 {
        warn!("{too_short} read(s) too short for CNN detection — wrote adapter_end=0");
    }
    if failed > 0 {
        warn!(
            "{failed} read(s) failed CNN inference — wrote adapter_end=0; \
             check that the model honours the [B, 1, L] -> [B, 2, L] contract",
        );
    }

    info!(
        "{} boundaries written to {}",
        style::action("Detected"),
        style::path(args.output.display())
    );
    info!(
        "{} {} reads with detected adapters",
        style::label("Result:"),
        style::count(detected)
    );

    if args.emit_llr_delta {
        if abs_deltas.is_empty() {
            warn!(
                "--emit-llr-delta: no read had a boundary from both detectors, \
                 so every end_delta is empty"
            );
        } else {
            // Percentiles rather than a "within N samples" count: any such N
            // would be a threshold invented here, and the point of the gate is
            // to let the distribution speak.
            abs_deltas.sort_unstable();
            let at = |q: f64| abs_deltas[((abs_deltas.len() - 1) as f64 * q) as usize];
            info!(
                "{} {} reads comparable; |end_delta| median {}, p95 {}, max {}",
                style::label("LLR vs CNN:"),
                style::count(abs_deltas.len()),
                at(0.5),
                at(0.95),
                abs_deltas[abs_deltas.len() - 1],
            );
        }
    }

    timer.report(profile);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::llr_boundaries;

    /// A synthetic read shaped like what the LLR detector looks for: one level
    /// followed by another, with a deterministic wobble so neither segment has
    /// zero variance (which is degenerate for a log-likelihood ratio).
    fn synthetic_read() -> Vec<i16> {
        (0..4000)
            .map(|i| {
                let base: i16 = if i < 1200 { 400 } else { 900 };
                base + ((i * 37) % 21) as i16 - 10
            })
            .collect()
    }

    /// `--downscale 0` and `--downscale 1` both mean "no downscaling", and the
    /// two callers of this function must agree on that.
    ///
    /// Regression guard with a specific history: `--method llr` folds the 0 case
    /// away with `args.downscale.max(1)` in its caller, so a second copy of this
    /// logic that took the raw argument would hand a factor of 0 to
    /// `downscale_normalize_into` and diverge from the detector it claims to
    /// reproduce — which would surface as fictional LLR-vs-CNN disagreement in
    /// `--emit-llr-delta`. Sharing one function is what prevents it; this pins
    /// the behaviour.
    #[test]
    fn downscale_zero_and_one_agree() {
        let sig = synthetic_read();
        assert_eq!(
            llr_boundaries(&sig, (200, 50, 0)),
            llr_boundaries(&sig, (200, 50, 1)),
        );
    }

    /// Boundaries come back in full-resolution sample coordinates, not in
    /// downscaled units — the callers write them straight into the CSV
    /// alongside `num_samples`.
    #[test]
    fn downscaled_output_is_rescaled_to_full_resolution() {
        let sig = synthetic_read();
        let (start, end) = llr_boundaries(&sig, (200, 50, 10));
        assert_eq!(start % 10, 0, "start not rescaled by the downscale factor");
        assert_eq!(end % 10, 0, "end not rescaled by the downscale factor");
        assert!(end <= sig.len(), "boundary past the end of the read");
        assert!(start <= end, "start after end");
    }
}
