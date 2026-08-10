//! POD5 file CLI tools.

use clap::builder::styling::{AnsiColor, Effects, Styles};
use clap::{Parser, Subcommand};
use escapepod_signal::Durability;
use std::path::PathBuf;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::fmt::format::{self, FormatEvent, FormatFields};
use tracing_subscriber::registry::LookupSpan;

mod commands;
mod progress;
mod style;
#[cfg(test)]
mod test_env;
mod threads;
mod util;

/// Terse event formatter for `escpod` logs: `YYYY-MM-DD HH:MM:SS  LEVEL [target] message`.
///
/// The CLI's own events (target `escpod` / `escpod::*`) are the primary status
/// channel, so their target is omitted entirely to keep status lines clean
/// (`… INFO Merging 2 files …`). `escapepod_signal::` targets are shown without
/// the crate prefix; all other (library/external) targets are printed verbatim
/// so `-v` can attribute them.
struct EscpodFormatter;

impl<S, N> FormatEvent<S, N> for EscpodFormatter
where
    S: tracing::Subscriber + for<'a> LookupSpan<'a>,
    N: for<'a> FormatFields<'a> + 'static,
{
    fn format_event(
        &self,
        ctx: &tracing_subscriber::fmt::FmtContext<'_, S, N>,
        mut writer: format::Writer<'_>,
        event: &tracing::Event<'_>,
    ) -> std::fmt::Result {
        let now = chrono::Local::now();
        write!(writer, "{}", now.format("%Y-%m-%d %H:%M:%S"))?;
        write!(writer, " {:>5}", event.metadata().level())?;

        let target = event.metadata().target();
        if target == "escpod" || target.starts_with("escpod::") {
            // CLI's own status output — no target label.
        } else if let Some(module) = target.strip_prefix("escapepod_signal::") {
            write!(writer, " [{module}]")?;
        } else if target != "escapepod_cli" && target != "escapepod" {
            write!(writer, " [{target}]")?;
        }

        write!(writer, " ")?;
        ctx.field_format().format_fields(writer.by_ref(), event)?;
        writeln!(writer)
    }
}

const STYLES: Styles = Styles::styled()
    .header(AnsiColor::Yellow.on_default().effects(Effects::BOLD))
    .usage(AnsiColor::Yellow.on_default().effects(Effects::BOLD))
    .literal(AnsiColor::Green.on_default().effects(Effects::BOLD))
    .placeholder(AnsiColor::Cyan.on_default());

#[derive(Parser)]
#[command(name = "escpod")]
#[command(author, version = env!("ESCPOD_VERSION"), styles = STYLES)]
#[command(about = "A fast, pure-Rust toolkit for POD5 files (Oxford Nanopore sequencing data)")]
#[command(
    long_about = "A fast, pure-Rust toolkit for POD5 files (Oxford Nanopore sequencing data).\n\n\
POD5 is the native file format for Oxford Nanopore sequencing devices. This tool \
provides commands for viewing, inspecting, merging, filtering, and subsetting POD5 files."
)]
#[command(after_help = "\
Examples:
  escpod view input.pod5                    View all reads as TSV
  escpod view input.pod5 --ids              Extract just read IDs
  escpod inspect summary input.pod5         Show file summary
  escpod merge *.pod5 -o merged.pod5        Merge multiple files
  escpod filter in.pod5 -i ids.txt -o out.pod5   Filter by read IDs
  escpod summary input.pod5                 Comprehensive statistics

Experimental commands (repack, resquiggle, index, demux) are not built by default.
Rebuild with `--features experimental` and/or `--features demux` to enable them.
")]
struct Cli {
    /// Silence all output except errors. Overrides `--verbose`.
    #[arg(short = 'q', long, global = true)]
    quiet: bool,

    /// Increase log verbosity. Status messages show by default (info);
    /// `-v` = debug, `-vv` = trace. `RUST_LOG` takes precedence when set.
    #[arg(short = 'v', long, global = true, action = clap::ArgAction::Count)]
    verbose: u8,

    /// How hard to push output to stable storage before it is renamed into
    /// place. Output files are always staged and renamed, so an interrupted
    /// run never leaves a partial archive at the destination; this controls
    /// only whether the bytes are also durable against a machine crash.
    ///
    /// `none` (default) is fastest and appropriate on scratch filesystems
    /// where output is regenerable. `file` fsyncs each output; `full` also
    /// fsyncs the directory, so the rename itself survives a crash.
    #[arg(long, global = true, value_enum, default_value_t = FsyncMode::None)]
    fsync: FsyncMode,

    #[command(subcommand)]
    command: Commands,
}

/// CLI spelling of [`Durability`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, clap::ValueEnum)]
enum FsyncMode {
    /// Rename only — no fsync.
    None,
    /// fsync each output file before renaming it into place.
    File,
    /// Also fsync the parent directory after the rename.
    Full,
}

impl From<FsyncMode> for Durability {
    fn from(m: FsyncMode) -> Self {
        match m {
            FsyncMode::None => Durability::None,
            FsyncMode::File => Durability::File,
            FsyncMode::Full => Durability::FileAndDir,
        }
    }
}

// The `Demux` variant carries the fused-pipeline arg struct, which is wide by
// nature (one field per pipeline knob). Boxing a clap `Args` variant is awkward,
// and a `Commands` value is constructed exactly once at startup — the size
// asymmetry is irrelevant here.
#[allow(clippy::large_enum_variant)]
#[derive(Subcommand)]
enum Commands {
    /// View read summaries from a POD5 file as TSV
    #[command(after_help = "\
Examples:
  escpod view input.pod5                      Output all fields as TSV
  escpod view input.pod5 --ids                Output only read IDs
  escpod view input.pod5 --include read_id,channel
  escpod view input.pod5 --exclude signal_rows,pore_type
  escpod view input.pod5 -o reads.tsv         Write to file
  escpod view input.pod5 --separator ','      Use comma separator
")]
    View {
        /// Input POD5 file
        input: PathBuf,

        /// Fields to include (comma-separated)
        #[arg(long, value_name = "FIELDS")]
        include: Option<String>,

        /// Fields to exclude (comma-separated)
        #[arg(long, value_name = "FIELDS")]
        exclude: Option<String>,

        /// Output only read IDs
        #[arg(long)]
        ids: bool,

        /// Output file (stdout if not specified)
        #[arg(short, long, value_name = "FILE")]
        output: Option<PathBuf>,

        /// Field separator
        #[arg(long, default_value = "\t", value_name = "SEP")]
        separator: String,

        /// Don't print header row
        #[arg(long)]
        no_header: bool,
    },

    /// Inspect POD5 file contents
    #[command(after_help = "\
Examples:
  escpod inspect summary input.pod5           Show file summary
  escpod inspect reads input.pod5             List all read IDs
  escpod inspect read input.pod5 <READ_ID>    Show details for one read
")]
    Inspect {
        #[command(subcommand)]
        command: InspectCommands,
    },

    /// Merge multiple POD5 files into one
    #[command(after_help = "\
Examples:
  escpod merge *.pod5 -o merged.pod5          Merge all POD5 files
  escpod merge a.pod5 b.pod5 -o out.pod5      Merge specific files
  escpod merge *.pod5 -o out.pod5 --duplicate-ok
")]
    Merge {
        /// Input POD5 files
        #[arg(required = true, value_name = "FILES")]
        inputs: Vec<PathBuf>,

        /// Output POD5 file
        #[arg(short, long, required = true, value_name = "FILE")]
        output: PathBuf,

        /// Allow duplicate read IDs (skip duplicate checking)
        #[arg(long)]
        duplicate_ok: bool,

        /// Number of threads for parallel processing (default: 16, or all available CPUs if fewer)
        #[arg(short = 't', long, visible_short_alias = 'j', value_name = "N")]
        threads: Option<usize>,

        /// Overwrite the output file if it already exists
        #[arg(short, long)]
        force: bool,

        /// Print profiling information (phase timing, throughput)
        #[arg(long)]
        profile: bool,
    },

    /// Filter reads by various criteria
    #[command(after_help = "\
Examples:
  escpod filter input.pod5 -i ids.txt -o filtered.pod5
  escpod filter input.pod5 --min-samples 4000 -o long_reads.pod5
  escpod filter input.pod5 --exclude-end-reason unblock_mux_change -o no_rejects.pod5
  escpod filter input.pod5 --end-reason signal_positive,signal_negative -o normal.pod5
  cat ids.txt | escpod filter input.pod5 -i - -o filtered.pod5

At least one filter criterion must be specified.
")]
    Filter {
        /// Input POD5 files and/or directories
        #[arg(required = true, num_args = 1..)]
        input: Vec<PathBuf>,

        /// File containing read IDs (one per line), or '-'/'stdin' to read from stdin
        #[arg(short, long, value_name = "FILE")]
        ids: Option<PathBuf>,

        /// Minimum number of samples (inclusive)
        #[arg(long, value_name = "N")]
        min_samples: Option<u64>,

        /// Maximum number of samples (inclusive)
        #[arg(long, value_name = "N")]
        max_samples: Option<u64>,

        /// Only include reads with these end reasons (comma-separated)
        #[arg(long, value_name = "REASONS", value_delimiter = ',')]
        end_reason: Option<Vec<String>>,

        /// Exclude reads with these end reasons (comma-separated)
        #[arg(long, value_name = "REASONS", value_delimiter = ',')]
        exclude_end_reason: Option<Vec<String>>,

        /// Keep reads whose .p5s sidecar annotation matches (NAME=LABEL,
        /// e.g. barcode=nbc01 or condition=fresh). Repeatable: same NAME =
        /// any-of, different NAMEs = all-of
        #[arg(long, value_name = "NAME=LABEL")]
        annotation: Option<Vec<String>>,

        /// Output POD5 file
        #[arg(short, long, required = true, value_name = "FILE")]
        output: PathBuf,

        /// Number of threads for parallel processing (default: 16, or all available CPUs if fewer)
        #[arg(short = 't', long, visible_short_alias = 'j', value_name = "N")]
        threads: Option<usize>,

        /// Overwrite the output file if it already exists
        #[arg(short, long)]
        force: bool,

        /// Print per-phase timing breakdown after completion
        #[arg(long)]
        profile: bool,
    },

    /// Filter reads based on paired BAM file
    #[command(after_help = "\
Examples:
  escpod bam-filter reads.pod5 -b aligned.bam -o mapped.pod5 --mapped
  escpod bam-filter reads.pod5 -b aligned.bam -o chr1.pod5 --region chr1
  escpod bam-filter reads.pod5 -b aligned.bam -o hq.pod5 --quality 20

Region queries (`--region`) require a BAI index next to the BAM file; one is \
created automatically if not present (written as <bam>.bai).
")]
    BamFilter {
        /// Input POD5 file or directory
        input: PathBuf,

        /// Input BAM file (auto-indexes BAM.bai if --region is used)
        #[arg(short, long, required = true, value_name = "FILE")]
        bam: PathBuf,

        /// Output POD5 file
        #[arg(short, long, required = true, value_name = "FILE")]
        output: PathBuf,

        /// Keep only mapped reads
        #[arg(long)]
        mapped: bool,

        /// Filter by region (chr or chr:start-end)
        #[arg(long, value_name = "REGION")]
        region: Option<String>,

        /// Minimum mapping quality
        #[arg(long, value_name = "N")]
        quality: Option<u8>,

        /// Overwrite the output file if it already exists
        #[arg(short, long)]
        force: bool,

        /// Print per-phase timing breakdown after completion
        #[arg(long)]
        profile: bool,
    },

    /// Repack POD5 files to optimize storage (experimental; requires `--features experimental`)
    #[cfg(feature = "experimental")]
    #[command(after_help = "\
Examples:
  escpod repack input.pod5 -o output_dir/
  escpod repack *.pod5 -o repacked/ --force
")]
    Repack {
        /// Input POD5 files
        #[arg(required = true, value_name = "FILES")]
        inputs: Vec<PathBuf>,

        /// Output directory
        #[arg(short, long, required = true, value_name = "DIR")]
        output_dir: PathBuf,

        /// Overwrite existing files
        #[arg(short, long)]
        force: bool,

        /// Print per-phase timing breakdown after completion
        #[arg(long)]
        profile: bool,
    },

    /// Repack POD5 files to optimize storage (rebuild with `--features experimental` to enable)
    #[cfg(not(feature = "experimental"))]
    #[command(hide = true)]
    Repack {
        /// Repack arguments (ignored; feature not enabled)
        #[arg(
            trailing_var_arg = true,
            allow_hyphen_values = true,
            value_name = "ARGS"
        )]
        args: Vec<String>,
    },

    /// Subset reads into multiple files based on CSV mapping
    #[command(after_help = "\
Examples:
  escpod subset input.pod5 --csv mapping.csv -o output_dir/

The CSV file should have columns: read_id,output
Each unique 'output' value creates a separate POD5 file.
")]
    Subset {
        /// Input POD5 file
        input: PathBuf,

        /// CSV file with read_id,output columns
        #[arg(long, required = true, value_name = "FILE")]
        csv: PathBuf,

        /// Output directory
        #[arg(short, long, default_value = ".", value_name = "DIR")]
        output_dir: PathBuf,

        /// Number of threads for parallel processing (default: 16, or all available CPUs if fewer)
        #[arg(short = 't', long, visible_short_alias = 'j', value_name = "N")]
        threads: Option<usize>,

        /// Overwrite existing files
        #[arg(short, long)]
        force: bool,

        /// Print per-phase timing breakdown after completion
        #[arg(long)]
        profile: bool,
    },

    /// Show comprehensive summary of POD5 file(s)
    #[command(after_help = "\
Examples:
  escpod summary input.pod5                   Summary for one file
  escpod summary *.pod5                       Summary across all files
  escpod summary input.pod5 --json            Output as JSON
")]
    Summary(commands::summary::SummaryArgs),

    /// Barcode demultiplexing workflow
    #[cfg(feature = "demux")]
    #[command(after_help = "\
Examples:
  escpod demux input.pod5 --model model.json -d out/   Fused pipeline (recommended)
  escpod demux input.pod5 --model crf_bundle/ --barcodes refs.csv -d out/ \\
      --method cnn --cnn-model adapter.onnx            Fused, CTC-CRF head
  escpod demux detect input.pod5 -o boundaries.csv
  escpod demux fingerprint input.pod5 --boundaries boundaries.csv -o fingerprints.csv
  escpod demux classify fingerprints.csv --reference barcodes.csv -o classifications.csv
  escpod demux basecall input.pod5 --boundaries boundaries.csv --model crf_bundle/ \\
      --barcodes refs.csv -o classifications.csv

The fused pipeline drives any of the three classifier heads -- DTW-SVM, GBM or
CTC-CRF -- and decodes each read's signal once. The detect/fingerprint/classify/
basecall subcommands are those same stages run separately, which re-reads the
POD5 per stage; prefer the fused form unless you want the intermediate files.
")]
    Demux(commands::demux::DemuxArgs),

    /// Barcode demultiplexing workflow (rebuild with `--features demux` to enable)
    #[cfg(not(feature = "demux"))]
    #[command(hide = true)]
    Demux {
        /// Demux subcommand and arguments (ignored; feature not enabled)
        #[arg(
            trailing_var_arg = true,
            allow_hyphen_values = true,
            value_name = "ARGS"
        )]
        args: Vec<String>,
    },

    /// Classify reads against a model bundle (tRNA charging) from POD5 + aligned BAM
    #[cfg(feature = "classify")]
    Classify(commands::classify::ClassifyArgs),

    /// Classify reads against a model bundle (rebuild with `--features classify` to enable)
    #[cfg(not(feature = "classify"))]
    #[command(hide = true)]
    Classify {
        /// Classify arguments (ignored; feature not enabled)
        #[arg(
            trailing_var_arg = true,
            allow_hyphen_values = true,
            value_name = "ARGS"
        )]
        args: Vec<String>,
    },

    /// Refine signal-to-base mapping using banded DP (experimental; requires `--features experimental`)
    #[cfg(feature = "experimental")]
    Resquiggle(commands::resquiggle::ResquiggleArgs),

    /// Refine signal-to-base mapping using banded DP (rebuild with `--features experimental` to enable)
    #[cfg(not(feature = "experimental"))]
    #[command(hide = true)]
    Resquiggle {
        /// Resquiggle arguments (ignored; feature not enabled)
        #[arg(
            trailing_var_arg = true,
            allow_hyphen_values = true,
            value_name = "ARGS"
        )]
        args: Vec<String>,
    },

    /// Build .p5s read index for fast UUID lookup (experimental; requires `--features experimental`)
    #[cfg(feature = "experimental")]
    #[command(after_help = "\
Examples:
  escpod index input.pod5                   Index one file
  escpod index *.pod5                       Index all POD5 files
  escpod index data_dir/                    Index directory recursively
  escpod index input.pod5 --force           Overwrite existing index
")]
    Index {
        /// Input POD5 file(s) or directory
        #[arg(required = true, value_name = "FILES")]
        inputs: Vec<PathBuf>,

        /// Rebuild existing .p5s indexes (annotations are preserved)
        #[arg(short, long)]
        force: bool,

        /// Number of threads for parallel processing (default: 16, or all available CPUs if fewer)
        #[arg(short = 't', long, visible_short_alias = 'j', value_name = "N")]
        threads: Option<usize>,
    },

    /// Build .p5s read index for fast UUID lookup (rebuild with `--features experimental` to enable)
    #[cfg(not(feature = "experimental"))]
    #[command(hide = true)]
    Index {
        /// Index arguments (ignored; feature not enabled)
        #[arg(
            trailing_var_arg = true,
            allow_hyphen_values = true,
            value_name = "ARGS"
        )]
        args: Vec<String>,
    },

    /// Record per-read annotations (e.g. demux barcodes) in the .p5s sidecar (experimental; requires `--features experimental`)
    #[cfg(feature = "experimental")]
    #[command(after_help = "\
Writes a read → label mapping into the .p5s sidecar next to each POD5 file.
The POD5 itself is never modified. The sidecar is a plain Arrow IPC table
(read_id, batch_idx, row_idx, plus one dictionary column per annotation) —
readable directly with pyarrow/polars — and is bound to its POD5 by file
identifier + size, so a stale or misplaced sidecar fails loudly. Existing
sidecar contents (the read index, other annotations) are preserved.

Examples:
  escpod annotate -a demux.csv input.pod5        Annotate with demux assignments
  escpod annotate -a demux.csv data_dir/         Annotate a directory recursively
  escpod annotate -a samples.csv --name sample input.pod5
  escpod annotate --design samplesheet.csv input.pod5
                                                 Map barcode (combinations) to
                                                 experimental conditions
")]
    #[command(group(
        clap::ArgGroup::new("action")
            .required(true)
            .args(["assignments", "design", "list", "remove", "remove_design"])
    ))]
    Annotate {
        /// Input POD5 file(s) or directory
        #[arg(required = true, value_name = "FILES")]
        inputs: Vec<PathBuf>,

        /// CSV mapping reads to labels (read_id + barcode/predicted_barcode
        /// columns — the output of `demux classify` or `demux basecall --barcodes`)
        #[arg(short = 'a', long, value_name = "CSV")]
        assignments: Option<PathBuf>,

        /// Experimental-design CSV: one column per key annotation (e.g.
        /// `barcode`, or `ldx,edx` for combinations) plus one column per
        /// experimental variable (e.g. `condition`). Stored in the sidecar
        /// and materialized as derived per-read columns
        #[arg(long, value_name = "CSV")]
        design: Option<PathBuf>,

        /// With --design: which CSV columns are the keys (default: every
        /// column whose name matches an existing annotation)
        #[arg(long, value_name = "COLS", requires = "design", value_delimiter = ',')]
        keys: Option<Vec<String>>,

        /// Annotation name (the sidecar column to write, with -a)
        #[arg(long, default_value = escapepod_signal::pod5::sidecar::DEFAULT_ANNOTATION_NAME)]
        name: String,

        /// List each file's sidecar contents (index, annotations, design)
        #[arg(long)]
        list: bool,

        /// Remove an annotation by name (design columns are refused —
        /// remove or update the design instead)
        #[arg(long, value_name = "NAME")]
        remove: Option<String>,

        /// Remove the experimental design and its derived columns
        #[arg(long)]
        remove_design: bool,

        /// Replace a stale or unreadable sidecar instead of erroring
        #[arg(long)]
        force: bool,

        /// Number of threads for parallel processing (default: 16, or all available CPUs if fewer)
        #[arg(short = 't', long, visible_short_alias = 'j', value_name = "N")]
        threads: Option<usize>,
    },

    /// Record per-read annotations in the .p5s sidecar (rebuild with `--features experimental` to enable)
    #[cfg(not(feature = "experimental"))]
    #[command(hide = true)]
    Annotate {
        /// Annotate arguments (ignored; feature not enabled)
        #[arg(
            trailing_var_arg = true,
            allow_hyphen_values = true,
            value_name = "ARGS"
        )]
        args: Vec<String>,
    },
}

/// The `-t/--threads` value the user asked for, whichever command they ran.
///
/// `main` sizes the rayon pool once, before dispatch (see [`threads`]), so it
/// needs the flag before it knows which command will consume it. Reading it
/// back out of the parsed enum keeps each subcommand's own `--threads` arg
/// intact — a `global = true` arg on `Cli` would collide with them.
///
/// This match is deliberately exhaustive: a new command is a compile error
/// until someone states whether it takes `--threads`. That is what stops #155
/// from being reintroduced by a command that quietly sizes its own pool.
fn requested_threads(command: &Commands) -> Option<usize> {
    match command {
        Commands::Merge { threads, .. }
        | Commands::Filter { threads, .. }
        | Commands::Subset { threads, .. } => *threads,

        #[cfg(feature = "experimental")]
        Commands::Index { threads, .. } => *threads,
        #[cfg(not(feature = "experimental"))]
        Commands::Index { .. } => None,

        #[cfg(feature = "experimental")]
        Commands::Annotate { threads, .. } => *threads,
        #[cfg(not(feature = "experimental"))]
        Commands::Annotate { .. } => None,

        // Like `demux`, resquiggle flattens its run args under a subcommand.
        #[cfg(feature = "experimental")]
        Commands::Resquiggle(args) => args.run.threads,
        #[cfg(not(feature = "experimental"))]
        Commands::Resquiggle { .. } => None,

        #[cfg(feature = "demux")]
        Commands::Demux(args) => commands::demux::requested_threads(args),
        #[cfg(not(feature = "demux"))]
        Commands::Demux { .. } => None,

        #[cfg(feature = "classify")]
        Commands::Classify(args) => args.threads,
        #[cfg(not(feature = "classify"))]
        Commands::Classify { .. } => None,

        // No `--threads` flag; these run on the default pool.
        Commands::View { .. }
        | Commands::Inspect { .. }
        | Commands::BamFilter { .. }
        | Commands::Summary(_)
        | Commands::Repack { .. } => None,
    }
}

#[derive(Subcommand)]
enum InspectCommands {
    /// Show file summary (batches, reads, run info)
    #[command(after_help = "\
Examples:
  escpod inspect summary input.pod5           Summary for one file
  escpod inspect summary data_dir/            Aggregate summary across a directory
")]
    Summary {
        /// Input POD5 file or directory
        #[arg(value_name = "INPUT")]
        input: PathBuf,
    },

    /// List all read IDs in the file
    #[command(after_help = "\
Examples:
  escpod inspect reads input.pod5 | head      First rows as plain text
  escpod inspect reads data_dir/ > all.tsv    Aggregate across a directory
")]
    Reads {
        /// Input POD5 file or directory
        #[arg(value_name = "INPUT")]
        input: PathBuf,
    },

    /// Show detailed info for a specific read
    #[command(after_help = "\
Examples:
  escpod inspect read input.pod5 a1b2c3d4-e5f6-7890-abcd-ef1234567890
  escpod inspect read data_dir/ <READ_ID>     Search across a directory

READ_ID accepts canonical UUID (8-4-4-4-12) or 32 hex chars without dashes.
")]
    Read {
        /// Input POD5 file or directory
        #[arg(value_name = "INPUT")]
        input: PathBuf,

        /// Read ID (UUID with or without dashes; case-insensitive)
        #[arg(value_name = "READ_ID")]
        read_id: String,
    },
}

/// Emit a clear error when a subcommand was invoked in a build that
/// didn't enable its feature flag.
#[allow(dead_code)]
fn feature_disabled(command: &str, feature: &str) -> anyhow::Result<()> {
    anyhow::bail!(
        "The `{command}` command is not available in this build.\n\
         Rebuild escpod with `--features {feature}` to enable it \
         (e.g., `cargo install --features {feature} escapepod-cli`)."
    )
}

/// Remove staging files on interrupt or termination.
///
/// Output is staged and renamed, so a signal already leaves the destination
/// alone — but the process dies before any `Drop` runs, so the staging files
/// themselves would be left behind. On a cluster the SIGTERM case is the
/// common one: SLURM sends it on `scancel` and at walltime expiry.
fn install_signal_handler() {
    use std::sync::atomic::{AtomicBool, Ordering};
    static HANDLING: AtomicBool = AtomicBool::new(false);

    // `ctrlc` dispatches this on a thread it spawned rather than in real
    // signal context, so locking and file removal here are legitimate.
    let result = ctrlc::set_handler(|| {
        if HANDLING.swap(true, Ordering::SeqCst) {
            // A second signal while the first cleanup is still running (for
            // example, wedged on an unresponsive network filesystem) must
            // still get the user out.
            std::process::exit(130);
        }
        eprintln!("\nInterrupted — discarding incomplete output...");
        escapepod_signal::abort_all_in_flight_writes();
        std::process::exit(130);
    });

    if let Err(e) = result {
        tracing::debug!("could not install signal handler: {e}");
    }
}

/// Restore the default `SIGPIPE` disposition.
///
/// The Rust runtime ignores `SIGPIPE`, which turns a closed downstream reader
/// into an `EPIPE` that our `writeln!`/`println!` calls surface as
/// `Error: Broken pipe`. Resetting to `SIG_DFL` makes `escpod view … | head`
/// (or `| less`, etc.) terminate quietly like any other Unix tool.
#[cfg(unix)]
fn reset_sigpipe() {
    // SAFETY: `signal(2)` with `SIG_DFL` is async-signal-safe and called once,
    // before any threads are spawned.
    unsafe {
        libc::signal(libc::SIGPIPE, libc::SIG_DFL);
    }
}

#[cfg(not(unix))]
fn reset_sigpipe() {}

fn main() -> anyhow::Result<()> {
    reset_sigpipe();

    let cli = Cli::parse();

    // Verbosity → log level. `RUST_LOG` always wins if set.
    // Status/progress output is emitted at INFO, so INFO is the default level
    // (status visible out of the box); `-q` drops to errors only.
    let level = match (cli.quiet, cli.verbose) {
        (true, _) => "error",
        (_, 0) => "info",
        (_, 1) => "debug",
        _ => "trace",
    };
    // Scope the level to our own crates and hold dependencies at `warn`.
    // Verbosity flags are about escpod's own progress, but some dependencies
    // are chatty at info through the `log` bridge — tract announces every SIMD
    // kernel it probes on each `demux detect --method cnn` run. `RUST_LOG`
    // remains the escape hatch for dependency logs. `-q` silences everything
    // but errors, dependencies included, so it stays a flat global level.
    let filter = if cli.quiet {
        level.to_string()
    } else {
        format!(
            "warn,escpod={level},escapepod_cli={level},escapepod_demux={level},\
             escapepod_signal={level},escapepod_pod5={level}"
        )
    };
    tracing_subscriber::fmt()
        // Our status lines carry ANSI from the `style::*` helpers (already
        // TTY/NO_COLOR-gated). tracing-subscriber 0.3.20+ sanitizes ANSI in the
        // `message` field by default, which would render those escapes as
        // literal `\x1b[...]` text; opt out so trusted color passes through.
        // Must precede `event_format` — the setter is only exposed while the
        // event format is still the default `Format`.
        .with_ansi_sanitization(false)
        .event_format(EscpodFormatter)
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(filter)),
        )
        .with_writer(std::io::stderr)
        .init();

    // Size the rayon pool before any command code runs. rayon's global pool
    // can only be built once, and only before anything touches it — doing it
    // here is what keeps `--threads` from being silently ignored (#155).
    // After the subscriber, so the diagnostics in `init` are visible.
    threads::init(requested_threads(&cli.command))?;

    install_signal_handler();

    let durability = Durability::from(cli.fsync);

    match cli.command {
        Commands::View {
            input,
            include,
            exclude,
            ids,
            output,
            separator,
            no_header,
        } => commands::view::run(input, include, exclude, ids, output, separator, no_header),

        Commands::Inspect { command } => match command {
            InspectCommands::Summary { input } => commands::inspect::summary(input),
            InspectCommands::Reads { input } => commands::inspect::reads(input),
            InspectCommands::Read { input, read_id } => commands::inspect::read(input, read_id),
        },

        Commands::Merge {
            inputs,
            output,
            duplicate_ok,
            threads: _,
            force,
            profile,
        } => commands::merge::run(inputs, output, duplicate_ok, force, profile, durability),

        Commands::Filter {
            input,
            ids,
            min_samples,
            max_samples,
            end_reason,
            exclude_end_reason,
            annotation,
            output,
            threads: _,
            force,
            profile,
        } => commands::filter::run(
            input,
            ids,
            min_samples,
            max_samples,
            end_reason,
            exclude_end_reason,
            annotation,
            output,
            force,
            profile,
            durability,
        ),

        Commands::BamFilter {
            input,
            bam,
            output,
            mapped,
            region,
            quality,
            force,
            profile,
        } => commands::bam_filter::run(
            input, bam, output, mapped, region, quality, force, profile, durability,
        ),

        #[cfg(feature = "experimental")]
        Commands::Repack {
            inputs,
            output_dir,
            force,
            profile,
        } => commands::repack::run(inputs, output_dir, force, profile, durability),

        #[cfg(not(feature = "experimental"))]
        Commands::Repack { .. } => feature_disabled("repack", "experimental"),

        Commands::Subset {
            input,
            csv,
            output_dir,
            threads: _,
            force,
            profile,
        } => commands::subset::run(input, csv, output_dir, force, profile, durability),

        Commands::Summary(args) => commands::summary::run(args),

        #[cfg(feature = "demux")]
        Commands::Demux(args) => commands::demux::run(args),

        #[cfg(not(feature = "demux"))]
        Commands::Demux { .. } => feature_disabled("demux", "demux"),

        #[cfg(feature = "classify")]
        Commands::Classify(args) => commands::classify::run(args),

        #[cfg(not(feature = "classify"))]
        Commands::Classify { .. } => feature_disabled("classify", "classify"),

        #[cfg(feature = "experimental")]
        Commands::Resquiggle(args) => commands::resquiggle::run(args),

        #[cfg(not(feature = "experimental"))]
        Commands::Resquiggle { .. } => feature_disabled("resquiggle", "experimental"),

        #[cfg(feature = "experimental")]
        Commands::Index {
            inputs,
            force,
            threads: _,
        } => commands::index::run(inputs, force),

        #[cfg(not(feature = "experimental"))]
        Commands::Index { .. } => feature_disabled("index", "experimental"),

        #[cfg(feature = "experimental")]
        Commands::Annotate {
            inputs,
            assignments,
            design,
            keys,
            name,
            list,
            remove,
            remove_design,
            force,
            threads: _,
        } => commands::annotate::run(commands::annotate::AnnotateCmd {
            inputs,
            assignments,
            design,
            keys,
            name,
            list,
            remove,
            remove_design,
            force,
        }),

        #[cfg(not(feature = "experimental"))]
        Commands::Annotate { .. } => feature_disabled("annotate", "experimental"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::sync::{Arc, Mutex};
    use tracing_subscriber::fmt::MakeWriter;

    /// In-memory sink so we can inspect the bytes the subscriber actually writes.
    #[derive(Clone)]
    struct BufWriter(Arc<Mutex<Vec<u8>>>);

    impl Write for BufWriter {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl<'a> MakeWriter<'a> for BufWriter {
        type Writer = BufWriter;
        fn make_writer(&'a self) -> Self::Writer {
            self.clone()
        }
    }

    /// Regression guard for the literal `\x1b[...]` bug.
    ///
    /// tracing-subscriber 0.3.20+ sanitizes ANSI in the `message` field by
    /// default, which escapes the trusted color codes our `style::*` helpers
    /// emit into the literal 4-char text `\x1b[...]` (visible verbatim in the
    /// demo GIFs and any color terminal). The subscriber opts out via
    /// `.with_ansi_sanitization(false)`; assert that a raw ESC byte in a message
    /// passes through unchanged and is not textualized.
    #[test]
    fn message_ansi_passes_through_unsanitized() {
        let buf = Arc::new(Mutex::new(Vec::new()));
        let subscriber = tracing_subscriber::fmt()
            .with_ansi_sanitization(false)
            .event_format(EscpodFormatter)
            .with_env_filter(EnvFilter::new("info"))
            .with_writer(BufWriter(buf.clone()))
            .finish();

        tracing::subscriber::with_default(subscriber, || {
            // Embed a raw ESC directly so the assertion is independent of
            // `style::use_color()`'s cached TTY detection.
            tracing::info!("{} done", "\u{1b}[32mgreen\u{1b}[0m");
        });

        let out = String::from_utf8(buf.lock().unwrap().clone()).unwrap();
        assert!(out.contains('\u{1b}'), "expected raw ESC byte in: {out:?}");
        assert!(
            !out.contains("\\x1b"),
            "message ANSI was sanitized to literal text: {out:?}"
        );
    }

    /// Parse a real argv and report the thread count `main` would apply.
    fn threads_for(argv: &[&str]) -> Option<usize> {
        let cli = Cli::try_parse_from(argv).unwrap_or_else(|e| panic!("{argv:?}: {e}"));
        requested_threads(&cli.command)
    }

    /// The regression guard for #155.
    ///
    /// `--method cnn` used to run a full-width pool whatever `-j` said, because
    /// its pool-sizing call landed *after* a `par_iter()` had already built the
    /// global pool. Sizing now happens in `main` from this accessor, so the
    /// flag reaching it is the property worth pinning — and it is identical for
    /// both methods, which is exactly what was untrue before.
    #[test]
    fn detect_threads_reach_main_for_every_method() {
        let base = ["escpod", "demux", "detect", "in.pod5", "-o", "b.csv"];
        for method in ["llr", "cnn"] {
            let argv = [&base[..], &["--method", method, "-j", "3"]].concat();
            assert_eq!(threads_for(&argv), Some(3), "method {method} dropped -j");
        }
    }

    #[test]
    fn detect_accepts_both_t_and_j() {
        let base = ["escpod", "demux", "detect", "in.pod5", "-o", "b.csv"];
        for flag in ["-j", "-t", "--threads"] {
            let argv = [&base[..], &[flag, "5"]].concat();
            assert_eq!(threads_for(&argv), Some(5), "{flag} dropped");
        }
    }

    #[test]
    fn absent_flag_defers_to_the_default() {
        assert_eq!(
            threads_for(&["escpod", "demux", "detect", "in.pod5", "-o", "b.csv"]),
            None
        );
        assert_eq!(threads_for(&["escpod", "view", "in.pod5"]), None);
    }

    /// With no subcommand, `demux` is the fused pipeline and its flags live in
    /// the flattened `RunArgs` — a separate path through the accessor.
    #[test]
    fn fused_demux_pipeline_threads_are_read() {
        assert_eq!(
            threads_for(&[
                "escpod", "demux", "in.pod5", "--model", "m.json", "-d", "out", "-j", "7"
            ]),
            Some(7)
        );
    }

    #[test]
    fn every_demux_subcommand_forwards_threads() {
        let cases: [&[&str]; 4] = [
            &[
                "fingerprint",
                "in.pod5",
                "--boundaries",
                "b.csv",
                "-o",
                "f.csv",
            ],
            &["classify", "f.csv", "--reference", "r.csv", "-o", "c.csv"],
            &[
                "split",
                "in.pod5",
                "--classifications",
                "c.csv",
                "-d",
                "out",
            ],
            &["train", "--assignments", "a.csv", "-o", "r.json"],
        ];
        for case in cases {
            let argv = [&["escpod", "demux"][..], case, &["-j", "6"]].concat();
            assert_eq!(threads_for(&argv), Some(6), "{case:?} dropped -j");
        }
    }

    #[test]
    fn top_level_commands_forward_threads() {
        let cases: [&[&str]; 3] = [
            &["merge", "a.pod5", "-o", "out.pod5"],
            &["filter", "in.pod5", "--min-samples", "10", "-o", "out.pod5"],
            &["subset", "in.pod5", "--csv", "m.csv", "-o", "out"],
        ];
        for case in cases {
            for flag in ["-t", "-j"] {
                let argv = [&["escpod"][..], case, &[flag, "2"]].concat();
                assert_eq!(threads_for(&argv), Some(2), "{case:?} dropped {flag}");
            }
        }
    }

    #[cfg(feature = "experimental")]
    #[test]
    fn experimental_commands_forward_threads() {
        assert_eq!(
            threads_for(&["escpod", "index", "in.pod5", "-t", "4"]),
            Some(4)
        );
        // resquiggle flattens its run args under a subcommand, like demux.
        assert_eq!(
            threads_for(&[
                "escpod",
                "resquiggle",
                "in.pod5",
                "-b",
                "a.bam",
                "-k",
                "k.tsv",
                "-o",
                "o.bam",
                "-j",
                "4",
            ]),
            Some(4)
        );
    }
}
