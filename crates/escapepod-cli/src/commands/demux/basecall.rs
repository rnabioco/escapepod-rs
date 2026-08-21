//! Basecall subcommand — CTC-CRF barcode basecalling from detected boundaries.
//!
//! Consumes `escpod demux detect`'s boundaries CSV and emits one row per read
//! with the decoded barcode-region sequence. That is the step the Stage-1
//! hybrid pipeline (rnabioco/escapepod-models#27) had to drop into Python for:
//!
//! ```text
//! escpod demux detect --method cnn   ->  boundaries.csv
//! escpod demux basecall --barcodes   ->  classifications.csv   <- this command
//! escpod demux split --classifications ...
//! ```
//!
//! With `--barcodes` the output carries `read_id`/`barcode`, which is exactly
//! what `escpod demux split` reads, so the pipeline runs end to end with no
//! Python in the middle. Without it, only decoded sequences are emitted — useful
//! when the references are not settled, or for QC on the decode alone.
//!
//! Confidence is the edit-distance margin to the second-best reference, which
//! is the definition the model's published precision-at-recovery numbers were
//! computed with.

use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use escapepod_demux::crf::{
    BarcodeMatch, BarcodeRefs, CrfEncoder, CrfScratch, RefChains, ScoredDecode,
};
use escapepod_signal::{Reader, ReadsBatchView};
use rayon::prelude::*;
use tracing::{debug, info, warn};

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

    /// Barcode reference CSV (`name,sequence` columns). Given this, each read
    /// is assigned to its closest reference by edit distance and the output
    /// gains `barcode`/`confidence` columns that `escpod demux split` consumes
    /// directly. Without it, only decoded sequences are emitted.
    #[arg(long, value_name = "FILE")]
    pub barcodes: Option<PathBuf>,

    /// Call a read `unclassified` when its edit-distance margin to the
    /// second-best reference is below this. 0 keeps every call, including
    /// outright ties; gate downstream instead if you want the raw margins.
    #[arg(long, default_value = "0", value_name = "N", requires = "barcodes")]
    pub min_margin: u32,

    /// Also score every reference against the lattice itself, adding
    /// `crf_logp`, `crf_margin`, `crf_best` and `mean_logpost` columns.
    ///
    /// `crf_logp` is `log P(called barcode | signal)` — a real probability,
    /// unlike `confidence`, which is an edit distance between two strings and
    /// on a designed panel measures how far apart the references are rather
    /// than how sure the model is (#241). `crf_margin` is the called barcode's
    /// log-odds in nats against its best alternative, so it goes negative when
    /// the lattice prefers something else; `crf_best` names what that is.
    ///
    /// Measured at +7.6% on this command over 20k RNA004 reads. Off by default
    /// anyway: the columns are an output change, and with `--gpu` it costs more
    /// than that — the constrained scan needs the raw scores, so the decode
    /// comes back to the host while the encoder stays on the device.
    #[arg(long, requires = "barcodes")]
    pub ref_scores: bool,

    /// Call a read `unclassified` when the lattice's log-odds for the called
    /// barcode against its best alternative are below this, in nats
    /// (implies `--ref-scores`).
    ///
    /// The gate `--min-margin` cannot be. Edit distance to a designed panel
    /// takes a handful of values — over one production flowcell, 99% of reads
    /// land on three of them — so `--min-margin` is a cliff, not a dial. This
    /// is continuous, and because the margin is measured against the *called*
    /// barcode it is negative whenever the lattice prefers a different one, so
    /// any positive threshold also drops those.
    ///
    /// Rough scale: 0.7 nats is 2:1 odds, 2.3 is 10:1, 4.6 is 100:1.
    #[arg(long, value_name = "NATS", requires = "barcodes")]
    pub min_crf_margin: Option<f32>,

    /// Call a read `unclassified` when `P(called barcode | signal)` is below
    /// this (implies `--ref-scores`).
    ///
    /// A different question from `--min-crf-margin`: this asks whether the
    /// model is confident in absolute terms, rather than whether it can tell
    /// the call apart from the next reference. A read can be certain it is
    /// *not* the other 15 barcodes and still put only half its mass on any
    /// reference at all.
    #[arg(long, value_name = "P", requires = "barcodes")]
    pub min_crf_prob: Option<f32>,

    /// Overrule the bundle's declared `boundary.margin` for this run.
    ///
    /// Reads below the threshold emit an empty sequence rather than a poor one,
    /// so a run full of `decoded_len=0` means the gate, not the decode. See
    /// `escpod demux --help`; prefer fixing the bundle's declaration.
    #[arg(long, value_name = "N", help_heading = "Advanced Options")]
    pub boundary_margin: Option<usize>,

    /// Overrule the bundle's declared `boundary.clamp_max_shift`: decode a read
    /// whose adapter ends before the model's `chunk` from `[0, chunk]`, provided
    /// `chunk - adapter_end` is at most N. 0 disables it.
    ///
    /// Reaches reads `--boundary-margin` cannot: their window would start before
    /// sample 0. See `escpod demux --help` for the recovery/quality tradeoff.
    #[arg(long, value_name = "N", help_heading = "Advanced Options")]
    pub clamp_max_shift: Option<usize>,

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
    Cpu(Box<CrfEncoder>),
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

    fn set_boundary_margin(&mut self, margin: usize) {
        match self {
            Self::Cpu(e) => e.set_boundary_margin(margin),
            #[cfg(feature = "crf-gpu")]
            Self::Gpu(e) => e.set_boundary_margin(margin),
        }
    }

    fn set_clamp_max_shift(&mut self, shift: usize) {
        match self {
            Self::Cpu(e) => e.set_clamp_max_shift(shift),
            #[cfg(feature = "crf-gpu")]
            Self::Gpu(e) => e.set_clamp_max_shift(shift),
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

    fn ref_chains(&self, seqs: &[&[u8]]) -> anyhow::Result<RefChains> {
        Ok(match self {
            Self::Cpu(e) => e.ref_chains(seqs)?,
            #[cfg(feature = "crf-gpu")]
            Self::Gpu(e) => e.ref_chains(seqs)?,
        })
    }

    /// [`Self::basecall_prepped`], additionally scoring every reference in
    /// `chains` against the lattice — see `--ref-scores`.
    fn basecall_prepped_scored(
        &self,
        prepped: &[Option<Vec<f32>>],
        chains: &RefChains,
    ) -> anyhow::Result<Vec<Option<ScoredDecode>>> {
        match self {
            Self::Cpu(encoder) => Ok(prepped
                .par_iter()
                .map_init(CrfScratch::new, |scratch, w| {
                    let w = w.as_ref()?;
                    encoder
                        .basecall_prepped_with_refs(w, scratch, chains)
                        .inspect_err(|e| warn!("encoder: {e}"))
                        .ok()
                })
                .collect()),
            #[cfg(feature = "crf-gpu")]
            Self::Gpu(encoder) => Ok(encoder.basecall_batch_with_refs(prepped, chains)?),
        }
    }
}

/// One read's result. `sequence` is `None` when the read had no usable window.
struct Decoded {
    read_id: uuid::Uuid,
    adapter_end: usize,
    sequence: Option<String>,
    call: Option<BarcodeMatch>,
    /// The lattice's own view of the panel, under `--ref-scores`.
    scored: Option<ScoredDecode>,
}

/// escpod's own sentinel for a read that could not be placed (`demux/run.rs`).
const UNCLASSIFIED: &str = "unclassified";

/// The thresholds that can turn a matched read into `unclassified`.
///
/// Grouped rather than passed loose because they compose: a read has to clear
/// all of them, and which ones are set is a property of the run, not of the row
/// being written.
struct Gates {
    min_margin: u32,
    min_crf_margin: Option<f32>,
    min_crf_prob: Option<f32>,
}

impl Gates {
    /// Whether `call` survives every gate.
    ///
    /// A reference with no runner-up passes the margin gates rather than
    /// failing them: with nothing to compare against there is no margin to
    /// test, which is not the same as a margin of zero. Both the edit-distance
    /// and the lattice margin treat it that way, so `--min-margin` and
    /// `--min-crf-margin` cannot disagree about a single-reference panel.
    fn passes(&self, call: &BarcodeMatch, scored: Option<&ScoredDecode>) -> bool {
        if !call.margin.is_none_or(|m| m >= self.min_margin) {
            return false;
        }
        // The CRF gates are `requires = "ref_scores"`, so a run that sets one
        // always has the scores to test it against.
        let Some((logp, margin)) = scored.and_then(|s| s.call(call.index)) else {
            return true;
        };
        if let Some(t) = self.min_crf_margin
            && margin.is_some_and(|m| m < t)
        {
            return false;
        }
        if let Some(t) = self.min_crf_prob
            && logp.exp() < t
        {
            return false;
        }
        true
    }
}

/// Write one row, in whichever of the two column sets is in effect.
///
/// Every read that reached the decoder gets a row, including ones that could
/// not be decoded — they are emitted as `unclassified` rather than dropped, so
/// `escpod demux split` cannot silently lose reads and the output row count
/// stays reconcilable against the boundaries file.
fn write_row(
    out: &mut impl Write,
    d: &Decoded,
    refs: Option<&BarcodeRefs>,
    gates: &Gates,
) -> std::io::Result<()> {
    let seq = d.sequence.as_deref().unwrap_or("");
    let Some(refs) = refs else {
        return writeln!(out, "{},{},{},{}", d.read_id, d.adapter_end, seq.len(), seq);
    };
    let dist = |v: Option<u32>| v.map(|v| v.to_string()).unwrap_or_default();
    let (barcode, best, second, margin) = match &d.call {
        Some(c) => (
            if gates.passes(c, d.scored.as_ref()) {
                refs.name(c.index)
            } else {
                UNCLASSIFIED
            },
            c.best_dist.to_string(),
            dist(c.second_best_dist),
            dist(c.margin),
        ),
        None => (UNCLASSIFIED, String::new(), String::new(), String::new()),
    };
    write!(
        out,
        "{},{},{},{},{},{},{},{}",
        d.read_id,
        barcode,
        margin,
        best,
        second,
        d.adapter_end,
        seq.len(),
        seq
    )?;
    let Some(scored) = &d.scored else {
        return writeln!(out);
    };
    // These describe the barcode the edit distance matched, whether or not a
    // gate then rejected it: a row that reads `unclassified` should say what it
    // was rejected for, the same way `best_dist` stays populated on a row
    // `--min-margin` dropped.
    let (logp, call_margin) = d.call.as_ref().and_then(|c| scored.call(c.index)).map_or(
        (String::new(), String::new()),
        |(l, m)| {
            (
                format!("{l:.4}"),
                m.map(|m| format!("{m:.4}")).unwrap_or_default(),
            )
        },
    );
    writeln!(
        out,
        ",{},{},{},{:.4}",
        logp,
        call_margin,
        scored.best().map(|(i, _, _)| refs.name(i)).unwrap_or(""),
        scored.mean_logpost,
    )
}

/// Run the basecall subcommand.
/// One batch handed from the reader to the encoder: per-read `(id, adapter_end)`
/// and the prepped windows, in the same order and the same length.
type Block = (Vec<uuid::Uuid>, Vec<usize>, Vec<Option<Vec<f32>>>);

/// Batches read ahead of the encoder (`ESCAPEPOD_BASECALL_READAHEAD`).
///
/// **Deeper was measured and does not help.** A block here is a whole Arrow
/// batch — ~9,900 reads, ~118 MB of windows, ~1.2 s of encoder work — so depth
/// 16 looks like it should absorb ~19 s of reader stall, which is the scale of
/// the stalls a cold network filesystem produces. It does not pay off.
///
/// Warm (one 3.0 GB POD5, two rounds, depth order reversed in the second):
/// 30.7-31.2 s at every depth from 2 to 16, GPU mean flat at 62-64%. The trace
/// says why — the encoder waited 1.4 s on the reader all run, so the buffer was
/// never the constraint and deepening it only let the reader idle further ahead.
///
/// Cold is the regime that motivated the question (the encoder waits 145-179 s
/// there), and it still does not help. A 2x2 over {compgpu01, compgpu03} x
/// {depth 2, depth 16}, each run reading a file that node had never touched:
///
/// | node | depth 2 | depth 16 |
/// |---|---|---|
/// | compgpu01 | 268.4 s | 294.9 s |
/// | compgpu03 | 120.4 s | 131.9 s |
///
/// Depth 16 was *slower* on both nodes; the 2.2x spread is the node, not the
/// depth. Reading depth-2-vs-16 off a single pair of runs would have shown a
/// spurious 2.03x — run the control before believing an I/O-adjacent sweep here.
/// Costs ~1.4 GB of resident memory at depth 16 for nothing.
const DEFAULT_READAHEAD: usize = 2;

/// How many prepped batches may sit between the reader and the encoder.
///
/// Kept as a knob because the sweep above is one workload on one filesystem,
/// not because a bigger number is expected to win. Run with `-v` first: the
/// logged waits say which side is starving, and if the encoder is not waiting
/// on the reader, no depth will change anything.
fn readahead_blocks() -> usize {
    use std::sync::OnceLock;
    static CACHED: OnceLock<usize> = OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("ESCAPEPOD_BASECALL_READAHEAD")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or(DEFAULT_READAHEAD)
            .min(64)
    })
}

/// The reader half of the pipeline: one sequential sweep of the inputs, prepping
/// each Arrow batch and handing it off.
///
/// The sweep is single-streamed on purpose — fanning per-read signal reads across
/// workers collapses throughput on a network filesystem (#72), which is why this
/// has the same shape as `fingerprint`. Only the *handoff* is concurrent, so the
/// encoder can run batch N while this reads batch N+1.
/// Returns the time spent parked on a full channel — i.e. how long the reader
/// was ahead and waiting for the encoder to catch up.
fn produce_blocks(
    inputs: &[PathBuf],
    boundaries: &std::collections::HashMap<uuid::Uuid, escapepod_demux::ReadBoundaries>,
    meta: &escapepod_demux::crf::CrfMetadata,
    tx: &std::sync::mpsc::SyncSender<Block>,
) -> std::time::Duration {
    let mut blocked = std::time::Duration::ZERO;
    for path in inputs {
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
                        // so decode that prefix rather than a whole transcript
                        // — and calibrate only the window, not the prefix.
                        //
                        // Under clamping the window can extend PAST the adapter,
                        // to `[0, chunk]`, so the prefix has to reach `chunk` or
                        // the read arrives one sample short of its own window and
                        // is refused for a reason that looks like geometry.
                        let need = if meta.clamp_max_shift() > 0 {
                            adapter_end.max(meta.signal.chunk)
                        } else {
                            adapter_end
                        };
                        let adc = decode_chunks_to(chunks, Some(need))?;
                        let mut w = Vec::new();
                        meta.prep_adc_into(
                            &adc,
                            adapter_end,
                            read.calibration_offset,
                            read.calibration_scale,
                            &mut w,
                        )
                        .then_some(w)
                    })();
                    (read.read_id, adapter_end, window)
                })
                .collect();

            // Split rather than clone: the encoder wants the windows by
            // themselves, and they are the large half of the block.
            let mut ids = Vec::with_capacity(prepped.len());
            let mut ends = Vec::with_capacity(prepped.len());
            let mut windows = Vec::with_capacity(prepped.len());
            for (id, adapter_end, window) in prepped {
                ids.push(id);
                ends.push(adapter_end);
                windows.push(window);
            }
            let parked = std::time::Instant::now();
            let sent = tx.send((ids, ends, windows));
            blocked += parked.elapsed();
            if sent.is_err() {
                return blocked; // consumer stopped early; its error surfaces at the join
            }
        }
    }
    blocked
}

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
        Basecaller::Cpu(Box::new(CrfEncoder::load_bundle(&args.model)?))
    };
    #[cfg(not(feature = "crf-gpu"))]
    let encoder = Basecaller::Cpu(Box::new(CrfEncoder::load_bundle(&args.model)?));
    #[allow(unused_mut)]
    let mut encoder = encoder;
    if let Some(margin) = args.boundary_margin {
        let was = encoder.metadata().min_adapter_end();
        encoder.set_boundary_margin(margin);
        info!(
            "{} adapter_end >= {} (was {})",
            style::label("Boundary margin:"),
            style::count(encoder.metadata().min_adapter_end()),
            style::count(was),
        );
    }
    if let Some(shift) = args.clamp_max_shift {
        encoder.set_clamp_max_shift(shift);
    }
    let clamp = encoder.metadata().clamp_max_shift();
    if clamp > 0 {
        info!(
            "{} adapter_end down to {} decodes from [0, {}]",
            style::label("Window clamp:"),
            style::count(encoder.metadata().signal.chunk.saturating_sub(clamp)),
            style::count(encoder.metadata().signal.chunk),
        );
    }

    // Keep the ORT session alive past process exit when there is one, for the
    // reason spelled out on `run::LeakIf` (pykeio/ort#609). Applied here, at
    // construction, rather than after the last write: the rest of this function
    // is full of `?`, and a trailing `mem::forget` is skipped on exactly the
    // error paths — where an ordinary failure would then be masked by an
    // exit-134 glibc abort instead of reporting itself.
    #[cfg(feature = "crf-gpu")]
    let leak_ort = matches!(encoder, Basecaller::Gpu(_));
    #[cfg(not(feature = "crf-gpu"))]
    let leak_ort = false;
    let encoder = super::run::LeakIf::new(encoder, leak_ort);

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

    let refs = args
        .barcodes
        .as_ref()
        .map(BarcodeRefs::from_csv)
        .transpose()?;
    if let Some(r) = &refs {
        info!(
            "{} {} barcode references",
            style::label("Loaded:"),
            style::count(r.len())
        );
        // The floor on what any call can mean: a read cannot be resolved to
        // better than half this, so a best_dist above it is a warning sign.
        if let Some(floor) = r.min_pairwise_distance() {
            info!(
                "{} minimum pairwise edit distance {}",
                style::label("References:"),
                style::value(floor)
            );
        }
    }

    let progress = create_progress_bar(boundaries.len() as u64, "Basecalling")?;
    let skipped = AtomicUsize::new(0);
    let mut out = BufWriter::new(std::fs::File::create(&args.output)?);
    // A gate implies the scores it gates on, matching the fused `demux`: the
    // flags are not independent choices, and `--min-crf-margin` silently doing
    // nothing would be the worse failure. Scoring needs the panel, so
    // `--ref-scores` without `--barcodes` is a clap `requires` error.
    let want_scores =
        args.ref_scores || args.min_crf_margin.is_some() || args.min_crf_prob.is_some();
    let chains = match (&refs, want_scores) {
        (Some(r), true) => Some(encoder.ref_chains(&r.sequences())?),
        _ => None,
    };
    if let Some(c) = &chains {
        info!(
            "{} {} references over {} shared lattice cells",
            style::label("Scoring:"),
            style::count(c.len()),
            style::count(c.cells()),
        );
    }
    let gates = Gates {
        min_margin: args.min_margin,
        min_crf_margin: args.min_crf_margin,
        min_crf_prob: args.min_crf_prob,
    };

    writeln!(
        out,
        "{}{}",
        if refs.is_some() {
            "read_id,barcode,confidence,best_dist,second_best_dist,adapter_end,decoded_len,decoded_seq"
        } else {
            "read_id,adapter_end,decoded_len,decoded_seq"
        },
        if chains.is_some() {
            ",crf_logp,crf_margin,crf_best,mean_logpost"
        } else {
            ""
        }
    )?;
    let mut written = 0usize;

    // Read ahead by one batch so the encoder is not idle across the signal read.
    //
    // The read itself is unchanged — still one sequential sweep, for the reason
    // in `produce_blocks` — but it now runs on its own thread, because the two
    // stages alternating in lockstep is what starves the encoder. Measured on a
    // 136 GB / 20-file input (A30, `--gpu`): 62% mean GPU utilisation with 20% of
    // samples below 5%, the dead windows being the reader stalled in a page fault
    // at ~2.4 MB/s while the device had nothing queued.
    //
    // Depth 2 is the whole buffer: one block in flight, one being built, so the
    // extra memory is bounded by a single batch of windows. The CPU encoder gets
    // the same overlap — it wants it for the same reason, and splitting the paths
    // would mean two copies of this loop.
    let (tx, rx) = std::sync::mpsc::sync_channel::<Block>(readahead_blocks());
    let inputs = &args.input;
    let bounds = &boundaries;
    std::thread::scope(|scope| -> anyhow::Result<()> {
        let reader = scope.spawn(move || produce_blocks(inputs, bounds, meta, &tx));

        // Which side waits is the whole question when tuning the read-ahead, and
        // it is not guessable: encoder-waiting means the reader cannot keep up
        // (a deeper buffer only helps if the stalls are bursty), reader-waiting
        // means the buffer is already deep enough and the encoder is the floor.
        // Timed as loop-total minus work, so `rx` still moves into the loop.
        let loop_start = std::time::Instant::now();
        let mut working = std::time::Duration::ZERO;

        // Encode, match and write on this thread: `out`/`written` stay plain
        // locals, and blocks arrive in sweep order, so the output is unchanged.
        //
        // Consuming `rx` (rather than `rx.iter()`) is load-bearing on the error
        // path: it moves the receiver into this closure, so an early `?` below
        // drops it, the producer's next `send` fails, and the scope's join can
        // finish. Borrowed from outside, the receiver would outlive the closure
        // and a failed write would deadlock the join against a producer parked
        // on a full channel — turning an I/O error into a hang.
        for (ids, ends, windows) in rx {
            let batch_start = std::time::Instant::now();
            let n = ids.len();
            // Two shapes of the same call: with `--ref-scores` the decode also
            // returns the panel scores, so the reference scan runs inside it
            // rather than over scores nothing kept.
            let results: Vec<(Option<String>, Option<ScoredDecode>)> = match &chains {
                Some(c) => encoder
                    .basecall_prepped_scored(&windows, c)?
                    .into_iter()
                    .map(|s| (s.as_ref().map(|s| s.sequence.clone()), s))
                    .collect(),
                None => encoder
                    .basecall_prepped(&windows)?
                    .into_iter()
                    .map(|s| (s, None))
                    .collect(),
            };
            drop(windows);

            // Matching is independent per read and cheap next to the decode,
            // but 96 references x one alignment each is still worth fanning out.
            let decoded: Vec<Decoded> = ids
                .into_iter()
                .zip(ends)
                .zip(results)
                .collect::<Vec<_>>()
                .into_par_iter()
                .map(|((read_id, adapter_end), (seq, scored))| {
                    let call = seq
                        .as_ref()
                        .and_then(|s| refs.as_ref().and_then(|r| r.match_sequence(s.as_bytes())));
                    Decoded {
                        read_id,
                        adapter_end,
                        sequence: seq,
                        call,
                        scored,
                    }
                })
                .collect();

            skipped.fetch_add(
                decoded.iter().filter(|d| d.sequence.is_none()).count(),
                Ordering::Relaxed,
            );
            for d in &decoded {
                write_row(&mut out, d, refs.as_ref(), &gates)?;
            }
            written += decoded.len();
            progress.inc(n as u64);
            working += batch_start.elapsed();
        }
        let waiting_on_reader = loop_start.elapsed().saturating_sub(working);
        let waiting_on_encoder = reader.join().unwrap_or_default();
        debug!(
            "read-ahead {}: encoder waited {:.1}s on the reader, reader waited {:.1}s on the encoder",
            readahead_blocks(),
            waiting_on_reader.as_secs_f64(),
            waiting_on_encoder.as_secs_f64(),
        );
        Ok(())
    })?;
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
