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
    /// and not bundled). Runs batched on the CPU by default; pass `--gpu` (with
    /// a `--features cnn-gpu` build) for onnxruntime CUDA inference.
    #[arg(
        long,
        default_value = "llr",
        value_name = "{llr,cnn}",
        help_heading = "Advanced Options"
    )]
    pub method: String,

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

    /// Run `--method cnn` inference on the GPU via onnxruntime CUDA, instead of
    /// the batched CPU tract path. Requires a `--features cnn-gpu` build and a
    /// visible CUDA device + onnxruntime shared library at runtime.
    #[cfg(feature = "cnn-gpu")]
    #[arg(long, help_heading = "Advanced Options")]
    pub gpu: bool,

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

/// Run the detect subcommand.
pub fn run(args: DetectArgs) -> anyhow::Result<()> {
    match args.method.as_str() {
        "llr" => run_llr(args),
        "cnn" => {
            #[cfg(feature = "cnn-detect")]
            {
                run_cnn(args)
            }
            #[cfg(not(feature = "cnn-detect"))]
            {
                let _ = args;
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
/// CPU runs the model one read at a time through tract-onnx; `--gpu` (on a
/// `cnn-gpu` build) runs it batched through onnxruntime's CUDA execution
/// provider, which is where the large speedup lives — the TCN is
/// inference-bound and tract has no efficient batched conv. Works with any
/// model on the `[B,1,L] -> [B,2,L]` contract — escapepod-models'
/// `adapter_rna004` (CC-BY) or an ADAPTed `BoundariesCNN` export (CC BY-NC; not
/// bundled). See `scripts/export_adapter_cnn_to_onnx.py`.
#[cfg(feature = "cnn-detect")]
fn run_cnn(args: DetectArgs) -> anyhow::Result<()> {
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

    #[cfg(feature = "cnn-gpu")]
    let use_gpu = args.gpu;
    #[cfg(not(feature = "cnn-gpu"))]
    let use_gpu = false;

    // The GPU producer prepares reads through `get_signal_prefix`, so the whole
    // signal an LLR arm would need is never decoded there. Rejecting the
    // combination is honest; silently running LLR on the prefix would report a
    // delta that measures truncation instead of detector disagreement.
    if args.emit_llr_delta && use_gpu {
        anyhow::bail!(
            "--emit-llr-delta is not supported with --gpu: the GPU path decodes \
             only each read's leading samples, and the LLR detector normalizes \
             over the whole read. Drop --gpu to compare the two detectors."
        );
    }

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
        #[cfg(feature = "cnn-gpu")]
        {
            use escapepod_demux::AdapterCnnConfig;
            use escapepod_signal::Reader;
            use rayon::prelude::*;
            use std::sync::mpsc::sync_channel;

            // Reads per block: bigger ⇒ bigger same-length groups ⇒ fewer, larger
            // onnxruntime calls. Bounded at 2 in flight to cap memory.
            const GPU_BLOCK: usize = 16_384;
            let cfg = AdapterCnnConfig::default();
            type Block = (Vec<(Uuid, u64)>, Vec<Option<Vec<f32>>>);
            let (tx, rx) = sync_channel::<Block>(2);

            let model_path = cnn_model_path.clone();
            let pb = &progress_bar;
            let bnd = &boundaries;
            std::thread::scope(|scope| -> anyhow::Result<Vec<DetectRow>> {
                let gpu_handle = scope.spawn(move || -> anyhow::Result<Vec<DetectRow>> {
                    // Bound onnxruntime's intra-op pool too. Left unset it
                    // spawns `available_parallelism()` threads alongside
                    // rayon's, so `--threads` would not bound the process
                    // (#155, GPU half).
                    let gpu = escapepod_demux::AdapterCnnGpu::load_with_threads(
                        &model_path,
                        crate::threads::width(),
                    )
                    .map_err(|e| anyhow::anyhow!("loading CNN model on GPU: {e}"))?;
                    let mut out = Vec::new();
                    for (meta, prepped) in rx.iter() {
                        let ends = gpu.detect_prepped(&prepped);
                        pb.inc(meta.len() as u64);
                        for (end, (read_id, num_samples)) in ends.into_iter().zip(meta) {
                            // `--emit-llr-delta` is rejected with `--gpu` above, so the
                            // LLR arm is always absent here.
                            out.push((bnd(read_id, num_samples, end), None));
                        }
                    }
                    Ok(out)
                });

                // Producers: per file, sort reads by length (so each block's reads
                // share lengths ⇒ big exact-length groups), then decode + i16→f32 +
                // prep in parallel and push blocks. MAD-norm is scale-invariant, so
                // raw i16 → f32 matches the pA path bit-for-bit post-normalization.
                for path in &args.input {
                    let reader = Reader::open(path)?;
                    let mut reads: Vec<_> = reader
                        .reads()?
                        .filter_map(Result::ok)
                        .filter(|r| !r.signal_rows.is_empty())
                        .collect();
                    reads.sort_by_key(|r| r.num_samples);
                    let extractor = reader.signal_extractor()?;
                    for window in reads.chunks(GPU_BLOCK) {
                        let prepped: Vec<(Uuid, u64, Option<Vec<f32>>)> = window
                            .par_iter()
                            .map(|r| {
                                // Only the leading `max_obs_trace` samples feed the
                                // CNN; skip decompressing the rest (matters for long
                                // mRNA reads).
                                let p = extractor
                                    .get_signal_prefix(&r.signal_rows, cfg.max_obs_trace)
                                    .ok()
                                    .and_then(|s| {
                                        let f: Vec<f32> = s.iter().map(|&x| x as f32).collect();
                                        cfg.prep(&f)
                                    });
                                (r.read_id, r.num_samples, p)
                            })
                            .collect();
                        let mut meta = Vec::with_capacity(prepped.len());
                        let mut preps = Vec::with_capacity(prepped.len());
                        for (id, ns, p) in prepped {
                            meta.push((id, ns));
                            preps.push(p);
                        }
                        if tx.send((meta, preps)).is_err() {
                            break; // GPU thread died; its error surfaces at join
                        }
                    }
                }
                drop(tx);
                gpu_handle
                    .join()
                    .map_err(|_| anyhow::anyhow!("GPU detection thread panicked"))?
            })?
        }
        #[cfg(not(feature = "cnn-gpu"))]
        unreachable!("--gpu is unavailable without the cnn-gpu feature")
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
    let failed = failed.into_inner();
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
