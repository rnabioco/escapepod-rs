//! POD5 file CLI tools.

use clap::builder::styling::{AnsiColor, Effects, Styles};
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::fmt::format::{self, FormatEvent, FormatFields};
use tracing_subscriber::registry::LookupSpan;

mod commands;
mod progress;
mod style;
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

    #[command(subcommand)]
    command: Commands,
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

        /// Number of threads for parallel processing (default: all CPUs)
        #[arg(short = 't', long, value_name = "N")]
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
        /// Input POD5 file or directory
        input: PathBuf,

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

        /// Output POD5 file
        #[arg(short, long, required = true, value_name = "FILE")]
        output: PathBuf,

        /// Number of threads for parallel processing (default: 8)
        #[arg(short = 't', long, value_name = "N")]
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

        /// Number of threads for parallel processing (default: 8)
        #[arg(short = 't', long, value_name = "N")]
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

    /// Barcode demultiplexing workflow (experimental; requires `--features demux`)
    #[cfg(feature = "demux")]
    #[command(after_help = "\
Examples:
  escpod demux detect input.pod5 -o boundaries.csv
  escpod demux fingerprint input.pod5 --boundaries boundaries.csv -o fingerprints.csv
  escpod demux classify fingerprints.csv --reference barcodes.csv -o classifications.csv
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

    /// Build .p5i read index for fast UUID lookup (experimental; requires `--features experimental`)
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

        /// Overwrite existing .p5i files
        #[arg(short, long)]
        force: bool,

        /// Number of threads for parallel processing (default: all CPUs)
        #[arg(short = 't', long, value_name = "N")]
        threads: Option<usize>,
    },

    /// Build .p5i read index for fast UUID lookup (rebuild with `--features experimental` to enable)
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

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Verbosity → log level. `RUST_LOG` always wins if set.
    // Status/progress output is emitted at INFO, so INFO is the default level
    // (status visible out of the box); `-q` drops to errors only.
    let filter = match (cli.quiet, cli.verbose) {
        (true, _) => "error",
        (_, 0) => "info",
        (_, 1) => "debug",
        _ => "trace",
    };
    tracing_subscriber::fmt()
        .event_format(EscpodFormatter)
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(filter)),
        )
        .with_writer(std::io::stderr)
        .init();

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
            threads,
            force,
            profile,
        } => commands::merge::run(inputs, output, duplicate_ok, threads, force, profile),

        Commands::Filter {
            input,
            ids,
            min_samples,
            max_samples,
            end_reason,
            exclude_end_reason,
            output,
            threads,
            force,
            profile,
        } => commands::filter::run(
            input,
            ids,
            min_samples,
            max_samples,
            end_reason,
            exclude_end_reason,
            output,
            threads,
            force,
            profile,
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
        } => commands::bam_filter::run(input, bam, output, mapped, region, quality, force, profile),

        #[cfg(feature = "experimental")]
        Commands::Repack {
            inputs,
            output_dir,
            force,
            profile,
        } => commands::repack::run(inputs, output_dir, force, profile),

        #[cfg(not(feature = "experimental"))]
        Commands::Repack { .. } => feature_disabled("repack", "experimental"),

        Commands::Subset {
            input,
            csv,
            output_dir,
            threads,
            force,
            profile,
        } => commands::subset::run(input, csv, output_dir, threads, force, profile),

        Commands::Summary(args) => commands::summary::run(args),

        #[cfg(feature = "demux")]
        Commands::Demux(args) => commands::demux::run(args),

        #[cfg(not(feature = "demux"))]
        Commands::Demux { .. } => feature_disabled("demux", "demux"),

        #[cfg(feature = "experimental")]
        Commands::Resquiggle(args) => commands::resquiggle::run(args),

        #[cfg(not(feature = "experimental"))]
        Commands::Resquiggle { .. } => feature_disabled("resquiggle", "experimental"),

        #[cfg(feature = "experimental")]
        Commands::Index {
            inputs,
            force,
            threads,
        } => commands::index::run(inputs, force, threads),

        #[cfg(not(feature = "experimental"))]
        Commands::Index { .. } => feature_disabled("index", "experimental"),
    }
}
