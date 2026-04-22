//! Classify subcommand - barcode classification using DTW distance.

use super::fp_io::{read_query_fingerprints_f32, read_query_fingerprints_f64};
use super::utils::{configure_thread_pool, parse_reference_csv};
use crate::style;
use anyhow::Context;
use escapepod_demux::{AnyModel, DtwSvmModel, SvmPredictor, SvmWorkspace, load_any_model};
use escapepod_signal::dtw::dtw_distance_matrix;
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use uuid::Uuid;

/// Open a streaming output writer. Detects `.gz` extension and transparently
/// gzip-compresses the output; otherwise writes plain bytes. Both paths
/// front-buffer with a 256 KiB BufWriter so per-row `writeln!` calls don't
/// hit the filesystem. `GzEncoder::Drop` finalizes the gzip trailer on
/// thread teardown; callers should still call `flush()` explicitly to
/// surface I/O errors during streaming.
fn open_output_writer(path: &Path) -> anyhow::Result<Box<dyn Write + Send>> {
    let file = File::create(path)
        .with_context(|| format!("Failed to create output file '{}'", path.display()))?;
    let buf = BufWriter::with_capacity(256 * 1024, file);
    if path.extension().and_then(|s| s.to_str()) == Some("gz") {
        Ok(Box::new(flate2::write::GzEncoder::new(
            buf,
            flate2::Compression::default(),
        )))
    } else {
        Ok(Box::new(buf))
    }
}

/// Arguments for the classify subcommand.
#[derive(Debug, clap::Args)]
pub struct ClassifyArgs {
    /// Input fingerprints file
    #[arg(value_name = "FILE")]
    pub fingerprints: PathBuf,

    /// Reference barcode fingerprints (training data, CSV format).
    #[arg(long, value_name = "FILE")]
    pub reference: Option<PathBuf>,

    /// Trained model JSON. Auto-detects between `DtwSvmModel` (from
    /// `escpod demux train-svm`) and the legacy `WarpDemuxModel` based on
    /// the JSON shape — users no longer need to pick the matching flag.
    #[arg(long, value_name = "FILE")]
    pub model: Option<PathBuf>,

    /// Deprecated alias for `--model`. Kept for compatibility with existing
    /// scripts; emits a warning when used.
    #[arg(long, value_name = "FILE", hide = true)]
    pub svm_model: Option<PathBuf>,

    /// Output classifications file
    #[arg(short, long, required = true, value_name = "FILE")]
    pub output: PathBuf,

    /// Number of threads for parallel processing (default: all CPUs, or
    /// whatever the rayon global pool picks up from `RAYON_NUM_THREADS`).
    #[arg(long, short = 'j', value_name = "N", help_heading = "Advanced Options")]
    pub threads: Option<usize>,

    /// GPU DTW batch size in matrix cells (queries × refs). Default
    /// 536_870_912 (~2 GB of f32 distance matrix per call), which fits
    /// on a 24 GB A30. Lower if you see VRAM OOMs on a smaller card
    /// (T4 at 16 GB), higher on H100/A100-80G with more headroom.
    #[cfg(feature = "gpu")]
    #[arg(long, value_name = "N", help_heading = "Advanced Options")]
    pub gpu_chunk_cells: Option<usize>,

    /// DTW window constraint (Sakoe-Chiba band width)
    #[arg(long, value_name = "N", help_heading = "Advanced Options")]
    pub window: Option<usize>,

    /// Minimum distance ratio for confident classification (CSV mode only)
    #[arg(
        long,
        default_value = "0.8",
        value_name = "RATIO",
        help_heading = "Advanced Options"
    )]
    pub min_ratio: f32,

    /// Output per-class probabilities (SVM model only)
    #[arg(long, help_heading = "Advanced Options")]
    pub probabilities: bool,

    /// Run DTW on the GPU (requires build with `--features gpu` and a CUDA device).
    ///
    /// Applies to all three classification modes: `--reference`, `--model`, and
    /// `--svm-model`. Only the DTW distance step moves to GPU; SVM kernel /
    /// decision / probability math stays on CPU.
    #[cfg(feature = "gpu")]
    #[arg(long, help_heading = "Advanced Options")]
    pub gpu: bool,

    /// Print per-phase timing breakdown after completion
    #[arg(long)]
    pub profile: bool,
}

/// Whether the user requested the GPU path. Expands to `false` in builds
/// compiled without the `gpu` feature.
#[inline]
fn gpu_requested(args: &ClassifyArgs) -> bool {
    #[cfg(feature = "gpu")]
    {
        args.gpu
    }
    #[cfg(not(feature = "gpu"))]
    {
        let _ = args;
        false
    }
}

/// Classification result for output.
struct ClassifyResult {
    read_id: Uuid,
    barcode: String,
    confidence: f64,
    best_distance: f64,
    second_best_distance: f64,
    is_confident: bool,
}

/// SVM classification result with probabilities.
struct SvmClassifyResult {
    read_id: Uuid,
    predicted_barcode: i32,
    confidence: f64,
    is_confident: bool,
    probabilities: Vec<f64>,
}

/// Run the classify subcommand.
pub fn run(mut args: ClassifyArgs) -> anyhow::Result<()> {
    use crate::commands::profile::PhaseTimer;
    let mut timer = PhaseTimer::new();
    timer.phase("Classify");
    let profile = args.profile;

    // Fold the deprecated --svm-model alias into --model, since the model
    // loader now auto-detects the JSON shape. Emit a warning so scripts
    // start migrating.
    if let Some(p) = args.svm_model.take() {
        eprintln!(
            "{} --svm-model is deprecated; use --model (auto-detects SVM vs WarpDemux JSON).",
            style::label("warning:"),
        );
        if args.model.is_some() {
            anyhow::bail!("Specify only one of --model / --svm-model (they are aliases now).");
        }
        args.model = Some(p);
    }

    configure_thread_pool(args.threads);

    // Validate that exactly one input source was provided.
    match (args.model.is_some(), args.reference.is_some()) {
        (false, false) => anyhow::bail!("One of --reference or --model must be provided"),
        (true, true) => anyhow::bail!("Only one of --reference or --model can be specified"),
        _ => {}
    }

    let result = if let Some(model_path) = args.model.take() {
        match load_any_model(&model_path)? {
            AnyModel::Svm(model) => run_with_svm_model(args, model_path, model),
            AnyModel::WarpDemux(_) => run_with_model(args, model_path),
        }
    } else if let Some(reference_path) = args.reference.take() {
        run_with_csv(args, reference_path)
    } else {
        unreachable!()
    };

    timer.report(profile);
    result
}

/// Run classification using a trained SVM model. `model` is the already-parsed
/// JSON (the dispatcher in `run()` detects the file's schema via
/// `load_any_model` to pick this path vs the WarpDemux path, so re-reading it
/// here would be wasted I/O).
fn run_with_svm_model(
    args: ClassifyArgs,
    svm_model_path: PathBuf,
    model: DtwSvmModel,
) -> anyhow::Result<()> {
    println!("{} reads using SVM model", style::action("Classifying"));
    println!(
        "{} {}",
        style::label("Fingerprints:"),
        style::path(args.fingerprints.display())
    );
    println!(
        "{} {}",
        style::label("SVM Model:"),
        style::path(svm_model_path.display())
    );
    println!(
        "{} {}",
        style::label("Output:"),
        style::path(args.output.display())
    );
    if args.probabilities {
        println!("{} per-class probabilities", style::label("Including:"));
    }

    println!(
        "{} {} classes, {} training samples, {} support vectors",
        style::label("Model:"),
        style::count(model.n_classes),
        style::count(model.n_samples()),
        style::count(model.support_indices.len())
    );

    // Read query fingerprints
    let query_fps = read_query_fingerprints_f64(&args.fingerprints)?;

    println!(
        "{} {} query fingerprints",
        style::label("Loaded:"),
        style::count(query_fps.len())
    );

    if query_fps.is_empty() {
        anyhow::bail!("No valid query fingerprints found");
    }

    // Classify each query (GPU batched if requested, else parallel CPU).
    // Both paths feed a producer/consumer stream: classifications go into a
    // bounded mpsc channel, a dedicated writer thread formats+writes as
    // results arrive. This overlaps DTW/SVM work with I/O and lets us drop
    // per-read probability vectors once they're serialized (saving ~8 · k ·
    // N_reads bytes of peak heap, which for a 1M-read classify with k=20
    // barcodes is ~160 MB that never gets buffered).
    let (confident_count, total_count) = if gpu_requested(&args) {
        #[cfg(feature = "gpu")]
        {
            use escapepod_demux::{
                DEFAULT_GPU_CHUNK_CELLS, classify_with_svm_batch_gpu_with_ctx,
            };
            println!("{} reads with SVM on GPU...", style::action("Classifying"));
            let chunk_cells = args.gpu_chunk_cells.unwrap_or(DEFAULT_GPU_CHUNK_CELLS);
            let read_ids: Vec<Uuid> = query_fps.iter().map(|(id, _)| *id).collect();
            let fps: Vec<Vec<f64>> = query_fps.into_iter().map(|(_, fp)| fp).collect();
            let ctx = escapepod_signal::dtw::GpuDtwContext::new()
                .map_err(|e| anyhow::anyhow!("GPU init failed: {e}"))?;
            let gpu_results = classify_with_svm_batch_gpu_with_ctx(
                &ctx,
                &model,
                &fps,
                chunk_cells,
            )
            .map_err(|e| anyhow::anyhow!("GPU DTW failed: {e}"))?;

            // GPU batch already produces the full Vec; stream it to the
            // writer so we still avoid buffering n_pairs × probabilities
            // for the write step.
            stream_svm_classifications(
                &args.output,
                &model,
                args.probabilities,
                gpu_results.len(),
                |tx| {
                    for ((probs, result), read_id) in gpu_results.into_iter().zip(read_ids) {
                        tx.send(SvmClassifyResult {
                            read_id,
                            predicted_barcode: result.predicted_barcode,
                            confidence: result.confidence,
                            is_confident: result.is_confident,
                            probabilities: probs,
                        })
                        .ok();
                    }
                    Ok(())
                },
            )?
        }
        #[cfg(not(feature = "gpu"))]
        {
            unreachable!("--gpu flag is only defined when the `gpu` feature is enabled")
        }
    } else {
        println!("{} reads with SVM...", style::action("Classifying"));
        // Build the predictor once (label→class-index tables reused across
        // every read) and give each rayon worker its own workspace so the
        // k×k coupling matrices never re-allocate per read.
        let predictor = SvmPredictor::new(&model);
        let n = query_fps.len();
        stream_svm_classifications(&args.output, &model, args.probabilities, n, |tx| {
            query_fps
                .par_iter()
                .map_init(
                    || SvmWorkspace::for_model(&model),
                    |ws, (read_id, fingerprint)| {
                        let (probs, result) = predictor.predict_with_workspace(fingerprint, ws);
                        SvmClassifyResult {
                            read_id: *read_id,
                            predicted_barcode: result.predicted_barcode,
                            confidence: result.confidence,
                            is_confident: result.is_confident,
                            probabilities: probs,
                        }
                    },
                )
                .for_each_with(tx.clone(), |tx, r| {
                    tx.send(r).ok();
                });
            Ok(())
        })?
    };

    let unclassified_count = total_count - confident_count;

    println!(
        "{} classifications written to {}",
        style::action("Wrote"),
        style::path(args.output.display())
    );
    println!(
        "{} {} confident, {} unclassified",
        style::label("Result:"),
        style::count(confident_count),
        style::warning(unclassified_count)
    );

    Ok(())
}

/// Run classification using a trained WarpDemuX model.
fn run_with_model(args: ClassifyArgs, model_path: PathBuf) -> anyhow::Result<()> {
    use escapepod_demux::{classify_read, load_model};

    println!(
        "{} reads using WarpDemuX model",
        style::action("Classifying")
    );
    println!(
        "{} {}",
        style::label("Fingerprints:"),
        style::path(args.fingerprints.display())
    );
    println!(
        "{} {}",
        style::label("Model:"),
        style::path(model_path.display())
    );
    println!(
        "{} {}",
        style::label("Output:"),
        style::path(args.output.display())
    );

    // Load the model
    println!("{} model...", style::action("Loading"));
    let model = load_model(&model_path)?;

    println!(
        "{} {} training samples, {} features, threshold={:.3} ({})",
        style::label("Model:"),
        style::count(model.num_samples()),
        style::value(model.feature_dim()),
        style::value(model.threshold),
        model.threshold_type
    );

    // Read query fingerprints
    let query_fps = read_query_fingerprints_f64(&args.fingerprints)?;

    println!(
        "{} {} query fingerprints",
        style::label("Loaded:"),
        style::count(query_fps.len())
    );

    if query_fps.is_empty() {
        anyhow::bail!("No valid query fingerprints found");
    }

    // Classify each query (GPU if requested, else parallel CPU). Streams
    // results through a bounded mpsc channel to a dedicated writer thread;
    // classification workers don't buffer a full `Vec<ClassifyResult>`.
    let (confident_count, total_count) = if gpu_requested(&args) {
        #[cfg(feature = "gpu")]
        {
            use escapepod_demux::classify_reads_gpu;
            println!("{} reads on GPU...", style::action("Classifying"));
            let read_ids: Vec<Uuid> = query_fps.iter().map(|(id, _)| *id).collect();
            let fps: Vec<Vec<f64>> = query_fps.into_iter().map(|(_, fp)| fp).collect();
            let gpu_results = classify_reads_gpu(&model, &fps)
                .map_err(|e| anyhow::anyhow!("GPU DTW failed: {e}"))?;

            stream_model_classifications(&args.output, gpu_results.len(), |tx| {
                for (result, read_id) in gpu_results.into_iter().zip(read_ids) {
                    tx.send(ClassifyResult {
                        read_id,
                        barcode: result.barcode,
                        confidence: result.confidence,
                        best_distance: result.best_distance,
                        second_best_distance: result.second_best_distance,
                        is_confident: result.is_confident,
                    })
                    .ok();
                }
                Ok(())
            })?
        }
        #[cfg(not(feature = "gpu"))]
        {
            unreachable!("--gpu flag is only defined when the `gpu` feature is enabled")
        }
    } else {
        println!("{} reads...", style::action("Classifying"));
        let n = query_fps.len();
        stream_model_classifications(&args.output, n, |tx| {
            query_fps
                .par_iter()
                .map(|(read_id, fingerprint)| {
                    let result = classify_read(&model, fingerprint);
                    ClassifyResult {
                        read_id: *read_id,
                        barcode: result.barcode,
                        confidence: result.confidence,
                        best_distance: result.best_distance,
                        second_best_distance: result.second_best_distance,
                        is_confident: result.is_confident,
                    }
                })
                .for_each_with(tx.clone(), |tx, r| {
                    tx.send(r).ok();
                });
            Ok(())
        })?
    };

    let unclassified_count = total_count - confident_count;

    println!(
        "{} classifications written to {}",
        style::action("Wrote"),
        style::path(args.output.display())
    );
    println!(
        "{} {} confident, {} unclassified",
        style::label("Result:"),
        style::count(confident_count),
        style::warning(unclassified_count)
    );

    Ok(())
}

/// Run classification using CSV reference fingerprints.
fn run_with_csv(args: ClassifyArgs, reference_path: PathBuf) -> anyhow::Result<()> {
    println!(
        "{} reads by barcode using DTW",
        style::action("Classifying")
    );
    println!(
        "{} {}",
        style::label("Fingerprints:"),
        style::path(args.fingerprints.display())
    );
    println!(
        "{} {}",
        style::label("Reference:"),
        style::path(reference_path.display())
    );
    println!(
        "{} {}",
        style::label("Output:"),
        style::path(args.output.display())
    );
    if let Some(w) = args.window {
        println!("{} {}", style::label("DTW window:"), style::value(w));
    }

    // Read reference fingerprints
    let reference_fps = parse_reference_csv(&reference_path)?;

    println!(
        "{} {} reference barcodes",
        style::label("Loaded:"),
        style::count(reference_fps.len())
    );

    if reference_fps.is_empty() {
        anyhow::bail!("No valid reference fingerprints found");
    }

    // Read query fingerprints
    let query_fps = read_query_fingerprints_f32(&args.fingerprints)?;

    println!(
        "{} {} query fingerprints",
        style::label("Loaded:"),
        style::count(query_fps.len())
    );

    if query_fps.is_empty() {
        anyhow::bail!("No valid query fingerprints found");
    }

    // Pass the existing fingerprint buffers to dtw_distance_matrix by slice
    // instead of cloning every query + reference Vec. For 100k queries at
    // 150 f32 each, this saves ~60 MB of peak heap (only ~1.6 MB of
    // `&[f32]` fat-pointers remain).
    let query_slices: Vec<&[f32]> = query_fps.iter().map(|(_, v)| v.as_slice()).collect();
    let ref_slices: Vec<&[f32]> = reference_fps
        .iter()
        .map(|fp| fp.values.as_slice())
        .collect();

    let distances = if gpu_requested(&args) {
        #[cfg(feature = "gpu")]
        {
            use escapepod_signal::dtw::dtw_distance_matrix_gpu;
            println!("{} DTW distances on GPU...", style::action("Computing"));
            dtw_distance_matrix_gpu(&query_slices, &ref_slices, args.window)
                .map_err(|e| anyhow::anyhow!("GPU DTW failed: {e}"))?
        }
        #[cfg(not(feature = "gpu"))]
        {
            unreachable!("--gpu flag is only defined when the `gpu` feature is enabled")
        }
    } else {
        println!("{} DTW distances...", style::action("Computing"));
        dtw_distance_matrix(&query_slices, &ref_slices, args.window)
    };

    // Classify each query
    let results: Vec<ClassifyResult> = query_fps
        .iter()
        .enumerate()
        .map(|(i, (read_id, _))| {
            let row = distances.row(i);

            // Find best and second-best matches in a single linear pass.
            // We only care about the top two, so a full sort is wasted work
            // once the reference set gets past a handful of barcodes.
            let mut best_idx: Option<usize> = None;
            let mut best_dist = f32::INFINITY;
            let mut second_best_dist = f32::INFINITY;
            for (j, d) in row.iter().copied().enumerate() {
                if d < best_dist {
                    second_best_dist = best_dist;
                    best_dist = d;
                    best_idx = Some(j);
                } else if d < second_best_dist {
                    second_best_dist = d;
                }
            }
            let best_idx = match best_idx {
                Some(idx) => idx,
                None => {
                    return ClassifyResult {
                        read_id: *read_id,
                        barcode: "unclassified".to_string(),
                        confidence: 0.0,
                        best_distance: f64::INFINITY,
                        second_best_distance: f64::INFINITY,
                        is_confident: false,
                    };
                }
            };

            let ratio = if second_best_dist > 0.0 {
                best_dist / second_best_dist
            } else {
                0.0
            };

            let confident = ratio <= args.min_ratio;
            let barcode_name = reference_fps[best_idx].barcode.clone();

            ClassifyResult {
                read_id: *read_id,
                barcode: barcode_name,
                confidence: (1.0 - ratio) as f64,
                best_distance: best_dist as f64,
                second_best_distance: second_best_dist as f64,
                is_confident: confident,
            }
        })
        .collect();

    // Write output
    write_csv_classifications(&args.output, &results)?;

    let confident_count = results.iter().filter(|r| r.is_confident).count();
    let unclassified_count = results.len() - confident_count;

    println!(
        "{} classifications written to {}",
        style::action("Wrote"),
        style::path(args.output.display())
    );
    println!(
        "{} {} confident, {} unclassified",
        style::label("Result:"),
        style::count(confident_count),
        style::warning(unclassified_count)
    );

    Ok(())
}

// Query fingerprint parsing moved to `super::fp_io` so the CSV path and
// the new Parquet path share the same dispatch + barcode-column-skip
// logic. See `read_query_fingerprints_f64` / `_f32` there.

/// Streaming variant of the legacy `write_model_classifications` writer.
/// Same producer/consumer pattern as [`stream_svm_classifications`].
fn stream_model_classifications<F>(
    path: &Path,
    _expected: usize,
    produce: F,
) -> anyhow::Result<(usize, usize)>
where
    F: FnOnce(&std::sync::mpsc::SyncSender<ClassifyResult>) -> anyhow::Result<()>,
{
    use std::sync::mpsc;
    let (tx, rx) = mpsc::sync_channel::<ClassifyResult>(4096);
    let path_buf = path.to_path_buf();

    let writer_thread = std::thread::spawn(move || -> anyhow::Result<(usize, usize)> {
        let mut writer = open_output_writer(&path_buf)?;
        writeln!(
            writer,
            "read_id,barcode,confidence,best_distance,second_best_distance,is_confident"
        )?;
        let mut confident = 0usize;
        let mut total = 0usize;
        for result in rx.iter() {
            total += 1;
            if result.is_confident {
                confident += 1;
            }
            writeln!(
                writer,
                "{},{},{:.6},{:.4},{:.4},{}",
                result.read_id,
                result.barcode,
                result.confidence,
                result.best_distance,
                result.second_best_distance,
                result.is_confident,
            )?;
        }
        writer.flush()?;
        Ok((confident, total))
    });

    produce(&tx)?;
    drop(tx);

    let counts = writer_thread
        .join()
        .map_err(|e| anyhow::anyhow!("writer thread panicked: {:?}", e))??;
    Ok(counts)
}

/// Write CSV classification results to CSV.
fn write_csv_classifications(path: &PathBuf, results: &[ClassifyResult]) -> anyhow::Result<()> {
    let mut writer = open_output_writer(path)?;

    writeln!(
        writer,
        "read_id,barcode,distance,second_best_distance,ratio,confident"
    )?;

    for result in results {
        let ratio = if result.second_best_distance > 0.0 {
            result.best_distance / result.second_best_distance
        } else {
            0.0
        };

        writeln!(
            writer,
            "{},{},{:.4},{:.4},{:.4},{}",
            result.read_id,
            result.barcode,
            result.best_distance,
            result.second_best_distance,
            ratio,
            result.is_confident
        )?;
    }

    writer.flush()?;
    Ok(())
}

/// Stream SVM classification results through a producer/consumer channel.
///
/// Spawns a writer thread that opens the output file, writes the header, and
/// drains a bounded mpsc channel of [`SvmClassifyResult`]s — formatting each
/// row with direct `write!` calls (no per-row `format!` + `Vec<String>` +
/// `join` allocations). `produce` runs on the calling thread (typically a
/// rayon `par_iter().for_each_with(tx, ...)`).
///
/// Returns `(confident_count, total_count)`.
fn stream_svm_classifications<F>(
    path: &Path,
    model: &DtwSvmModel,
    include_probabilities: bool,
    _expected: usize,
    produce: F,
) -> anyhow::Result<(usize, usize)>
where
    F: FnOnce(&std::sync::mpsc::SyncSender<SvmClassifyResult>) -> anyhow::Result<()>,
{
    use std::sync::mpsc;

    // Bounded channel provides backpressure: if I/O stalls, classification
    // workers block instead of piling up results in unbounded queue memory.
    // 4096 is deep enough to hide short syscall hiccups without letting the
    // buffer grow faster than the writer can drain it.
    let (tx, rx) = mpsc::sync_channel::<SvmClassifyResult>(4096);

    // Pre-build the barcode label strings once (instead of `format!("BC{:02}", id)`
    // per read) and the probability header once.
    let label_mapper = model.label_mapper.clone();
    let n_classes = model.n_classes;
    let path_buf = path.to_path_buf();

    let writer_thread = std::thread::spawn(move || -> anyhow::Result<(usize, usize)> {
        let mut writer = open_output_writer(&path_buf)?;

        if include_probabilities {
            write!(writer, "read_id,predicted_barcode,confidence,is_confident")?;
            for i in 0..n_classes {
                let barcode_id = label_mapper.get(&i).copied().unwrap_or(i as i32);
                write!(writer, ",p{:02}", barcode_id)?;
            }
            writeln!(writer)?;
        } else {
            writeln!(writer, "read_id,predicted_barcode,confidence,is_confident")?;
        }

        let mut confident = 0usize;
        let mut total = 0usize;

        for result in rx.iter() {
            total += 1;
            if result.is_confident {
                confident += 1;
            }

            // Direct `write!` → BufWriter: no intermediate `String` per row,
            // no `Vec<String>` + `join(",")` for the probability columns.
            if result.predicted_barcode >= 0 {
                write!(
                    writer,
                    "{},BC{:02},{:.6},{}",
                    result.read_id,
                    result.predicted_barcode,
                    result.confidence,
                    result.is_confident,
                )?;
            } else {
                write!(
                    writer,
                    "{},unclassified,{:.6},{}",
                    result.read_id, result.confidence, result.is_confident,
                )?;
            }

            if include_probabilities {
                for &p in &result.probabilities {
                    write!(writer, ",{:.6}", p)?;
                }
            }
            writeln!(writer)?;
        }

        writer.flush()?;
        Ok((confident, total))
    });

    produce(&tx)?;
    drop(tx); // close sender so the writer thread's `rx.iter()` terminates.

    let counts = writer_thread
        .join()
        .map_err(|e| anyhow::anyhow!("writer thread panicked: {:?}", e))??;
    Ok(counts)
}
