//! Basecall subcommand — CTC-CRF barcode basecalling from detected boundaries.
//!
//! Consumes `escpod demux detect`'s boundaries CSV and emits one row per read
//! with the decoded barcode-region sequence. That is the step the Stage-1
//! hybrid pipeline (rnabioco/escapepod-models#27) had to drop into Python for:
//!
//! ```text
//! escpod demux detect --method cnn   ->  boundaries.csv
//! escpod demux basecall              ->  sequences.csv     <- this command
//! <match sequences to barcode references>
//! escpod demux split --classifications ...
//! ```
//!
//! It stops at sequences rather than barcode calls on purpose: assigning a
//! sequence to a reference needs an edit-distance aligner, and escapepod-rs has
//! none yet (#27 tracks lifting `fqxv-align` for that). Emitting the decode is
//! the piece that was blocked on the Viterbi; matching is a separate decision.

use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use escapepod_demux::crf::{CrfEncoder, CrfScratch};
use escapepod_signal::{Reader, ReadsBatchView};
use rayon::prelude::*;
use tracing::{info, warn};

use super::utils::{decode_chunks_to, parse_boundaries_csv};
use crate::progress::create_progress_bar;
use crate::style;

/// Arguments for the basecall subcommand.
#[derive(Debug, clap::Args)]
pub struct BasecallArgs {
    /// Input POD5 file(s)
    #[arg(required = true, value_name = "FILES")]
    pub input: Vec<PathBuf>,

    /// Detected boundaries CSV (from `escpod demux detect`)
    #[arg(long, required = true, value_name = "FILE")]
    pub boundaries: PathBuf,

    /// CRF encoder bundle directory (`metadata.json` + the ONNX graph it names)
    #[arg(long, required = true, value_name = "DIR")]
    pub model: PathBuf,

    /// Output CSV of decoded sequences
    #[arg(short, long, required = true, value_name = "FILE")]
    pub output: PathBuf,

    /// Run encoder inference on the GPU (onnxruntime CUDA execution provider).
    /// The lattice decode stays on the CPU either way.
    #[cfg(feature = "crf-gpu")]
    #[arg(long)]
    pub gpu: bool,

    /// Number of threads for parallel processing
    #[arg(short = 't', long, visible_short_alias = 'j', value_name = "N")]
    pub threads: Option<usize>,
}

/// Where encoder inference runs. The lattice decode is on the CPU in both
/// cases — see `escapepod_demux::crf::encoder_gpu` for why.
enum Basecaller {
    Cpu(CrfEncoder),
    #[cfg(feature = "crf-gpu")]
    Gpu(Box<escapepod_demux::crf::CrfEncoderGpu>),
}

impl Basecaller {
    fn metadata(&self) -> &escapepod_demux::crf::CrfMetadata {
        match self {
            Self::Cpu(e) => e.metadata(),
            #[cfg(feature = "crf-gpu")]
            Self::Gpu(e) => e.metadata(),
        }
    }

    fn layout(&self) -> &escapepod_demux::crf::CrfLayout {
        match self {
            Self::Cpu(e) => e.layout(),
            #[cfg(feature = "crf-gpu")]
            Self::Gpu(e) => e.layout(),
        }
    }

    /// Basecall a batch of already-prepped windows (`None` = no usable window).
    ///
    /// The CPU path runs one read per rayon worker, because tract has no
    /// efficient batched LSTM; the GPU path submits the whole batch in one
    /// onnxruntime call and then fans the decode back out across workers.
    fn basecall_prepped(
        &self,
        prepped: &[Option<Vec<f32>>],
    ) -> anyhow::Result<Vec<Option<String>>> {
        match self {
            Self::Cpu(encoder) => Ok(prepped
                .par_iter()
                .map_init(CrfScratch::new, |scratch, w| {
                    let w = w.as_ref()?;
                    encoder
                        .basecall_prepped(w, scratch)
                        .inspect_err(|e| warn!("encoder: {e}"))
                        .ok()
                })
                .collect()),
            #[cfg(feature = "crf-gpu")]
            Self::Gpu(encoder) => Ok(encoder.basecall_batch(prepped)?),
        }
    }
}

/// One decoded read.
struct Decoded {
    read_id: uuid::Uuid,
    adapter_end: usize,
    sequence: String,
}

/// Run the basecall subcommand.
pub fn run(args: BasecallArgs) -> anyhow::Result<()> {
    info!("{} barcode regions", style::action("Basecalling"));
    info!(
        "{} {} POD5 file(s)",
        style::label("Input:"),
        style::count(args.input.len())
    );
    info!(
        "{} {}",
        style::label("Model:"),
        style::path(args.model.display())
    );

    #[cfg(feature = "crf-gpu")]
    let encoder = if args.gpu {
        info!(
            "{} GPU (onnxruntime CUDA)",
            style::label("Encoder runs on:")
        );
        Basecaller::Gpu(Box::new(escapepod_demux::crf::CrfEncoderGpu::load_bundle(
            &args.model,
            args.threads,
        )?))
    } else {
        Basecaller::Cpu(CrfEncoder::load_bundle(&args.model)?)
    };
    #[cfg(not(feature = "crf-gpu"))]
    let encoder = Basecaller::Cpu(CrfEncoder::load_bundle(&args.model)?);

    let meta = encoder.metadata();
    info!(
        "{} chunk={} stride={} t_len={} states={}",
        style::label("Encoder:"),
        style::value(meta.signal.chunk),
        style::value(meta.signal.stride),
        style::value(meta.t_len()),
        style::value(encoder.layout().n_states),
    );
    // Worth surfacing: these come from the sidecar, not config.toml, and using
    // the wrong pair degrades the decode silently.
    info!(
        "{} mean={:.3} stdev={:.3}",
        style::label("Standardisation:"),
        meta.standardisation.mean,
        meta.standardisation.stdev,
    );

    let boundaries = parse_boundaries_csv(&args.boundaries)?;
    info!(
        "{} {} boundary records",
        style::label("Loaded:"),
        style::count(boundaries.len())
    );

    let progress = create_progress_bar(boundaries.len() as u64, "Basecalling")?;
    let skipped = AtomicUsize::new(0);
    let mut out = BufWriter::new(std::fs::File::create(&args.output)?);
    writeln!(out, "read_id,adapter_end,decoded_len,decoded_seq")?;
    let mut written = 0usize;

    // One Arrow batch at a time, then decode + basecall in parallel — the same
    // single-sequential-sweep shape as `fingerprint`, for the same reason
    // (fanning per-read signal reads across workers collapses throughput on a
    // network filesystem).
    for path in &args.input {
        let Ok(reader) = Reader::open(path) else {
            warn!("skipping unreadable file {}", path.display());
            continue;
        };
        let Ok(batches) = reader.read_batches() else {
            continue;
        };
        for batch_result in batches {
            let Ok(batch) = batch_result else { continue };
            let Ok(view) = ReadsBatchView::new(&batch, false) else {
                continue;
            };
            let reads: Vec<_> = (0..view.num_rows())
                .filter_map(|row| view.read(row).ok())
                .filter(|r| !r.signal_rows.is_empty() && boundaries.contains_key(&r.read_id))
                .collect();
            if reads.is_empty() {
                continue;
            }

            let keyed: Vec<(usize, Vec<u64>)> = reads
                .iter()
                .enumerate()
                .map(|(i, r)| (i, r.signal_rows.clone()))
                .collect();
            let Ok(bulk) = reader.get_compressed_signal_bulk(&keyed) else {
                continue;
            };

            // Signal decompression, calibration and standardisation are all
            // CPU work and independent per read, so they fan out here; the
            // encoder then sees a ready batch (which is what lets the GPU path
            // submit one call per Arrow batch).
            let prepped: Vec<(uuid::Uuid, usize, Option<Vec<f32>>)> = bulk
                .par_iter()
                .map(|(i, chunks)| {
                    let read = &reads[*i];
                    let adapter_end = boundaries
                        .get(&read.read_id)
                        .map(|b| b.adapter_end)
                        .unwrap_or(0);
                    let window = (|| {
                        // Only the window ending at the adapter is ever read,
                        // so decode that prefix rather than a whole transcript.
                        let adc = decode_chunks_to(chunks, Some(adapter_end))?;
                        let signal =
                            adc_to_pa(&adc, read.calibration_offset, read.calibration_scale);
                        meta.prep(&signal, adapter_end)
                    })();
                    (read.read_id, adapter_end, window)
                })
                .collect();

            let windows: Vec<Option<Vec<f32>>> =
                prepped.iter().map(|(_, _, w)| w.clone()).collect();
            let sequences = encoder.basecall_prepped(&windows)?;

            let decoded: Vec<Decoded> = prepped
                .iter()
                .zip(sequences)
                .filter_map(|((read_id, adapter_end, _), seq)| {
                    Some(Decoded {
                        read_id: *read_id,
                        adapter_end: *adapter_end,
                        sequence: seq?,
                    })
                })
                .collect();

            skipped.fetch_add(reads.len() - decoded.len(), Ordering::Relaxed);
            for d in &decoded {
                writeln!(
                    out,
                    "{},{},{},{}",
                    d.read_id,
                    d.adapter_end,
                    d.sequence.len(),
                    d.sequence
                )?;
            }
            written += decoded.len();
            progress.inc(reads.len() as u64);
        }
    }
    out.flush()?;
    progress.finish_and_clear();

    let skipped = skipped.load(Ordering::Relaxed);
    info!(
        "{} {} reads to {}",
        style::action("Wrote"),
        style::count(written),
        style::path(args.output.display())
    );
    if skipped > 0 {
        // Overwhelmingly `adapter_end` below the training window — the detector
        // overloads 0 for "no adapter" / "too short" / "inference failed", and
        // those reads are dropped rather than decoded from a partial window.
        info!(
            "{} {} reads without a usable {}-sample window",
            style::label("Skipped:"),
            style::count(skipped),
            style::value(encoder.metadata().signal.chunk),
        );
    }
    Ok(())
}

/// ADC counts to picoamps, fused so there is one rounding step.
///
/// Matches `escapepod_python::adc_to_pa`; the reference `pod5` package computes
/// it unfused, which differs by ~1 ulp — thousands of times below the
/// standardisation scale and irrelevant to the decode.
fn adc_to_pa(raw: &[i16], offset: f32, scale: f32) -> Vec<f32> {
    let bias = offset * scale;
    raw.iter()
        .map(|&adc| f32::from(adc).mul_add(scale, bias))
        .collect()
}
