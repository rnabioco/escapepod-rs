//! POD5 file CLI tools.

use clap::builder::styling::{AnsiColor, Effects, Styles};
use clap::{Parser, Subcommand};
use std::path::PathBuf;

mod commands;
mod progress;
mod style;
mod util;

const STYLES: Styles = Styles::styled()
    .header(AnsiColor::Yellow.on_default().effects(Effects::BOLD))
    .usage(AnsiColor::Yellow.on_default().effects(Effects::BOLD))
    .literal(AnsiColor::Green.on_default().effects(Effects::BOLD))
    .placeholder(AnsiColor::Cyan.on_default());

#[derive(Parser)]
#[command(name = "escpod")]
#[command(author, version, styles = STYLES)]
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
")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

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
    },

    /// Filter reads based on paired BAM file
    BamFilter {
        /// Input POD5 file or directory
        input: PathBuf,

        /// Input BAM file (requires .bai index for region queries)
        #[arg(short, long, required = true)]
        bam: PathBuf,

        /// Output POD5 file
        #[arg(short, long, required = true)]
        output: PathBuf,

        /// Keep only mapped reads
        #[arg(long)]
        mapped: bool,

        /// Filter by region (chr or chr:start-end)
        #[arg(long)]
        region: Option<String>,

        /// Minimum mapping quality
        #[arg(short = 'q', long)]
        quality: Option<u8>,
    },

    /// Repack POD5 files to optimize storage
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

        /// Overwrite existing files
        #[arg(short, long)]
        force: bool,
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
    #[cfg(feature = "experimental")]
    #[command(after_help = "\
Examples:
  escpod demux detect input.pod5 -o boundaries.csv
  escpod demux fingerprint input.pod5 --boundaries boundaries.csv -o fingerprints.csv
  escpod demux classify fingerprints.csv --reference barcodes.csv -o classifications.csv
")]
    Demux(commands::demux::DemuxArgs),

    /// Refine signal-to-base mapping using banded DP
    #[cfg(feature = "experimental")]
    Resquiggle(commands::resquiggle::ResquiggleArgs),

    /// Build .p5i read index for fast UUID lookup
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
}

#[derive(Subcommand)]
enum InspectCommands {
    /// Show file summary (batches, reads, run info)
    Summary {
        /// Input POD5 file
        #[arg(value_name = "FILE")]
        input: PathBuf,
    },

    /// List all read IDs in the file
    Reads {
        /// Input POD5 file
        #[arg(value_name = "FILE")]
        input: PathBuf,
    },

    /// Show detailed info for a specific read
    Read {
        /// Input POD5 file
        #[arg(value_name = "FILE")]
        input: PathBuf,

        /// Read ID to inspect (UUID format)
        #[arg(value_name = "READ_ID")]
        read_id: String,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

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
            profile,
        } => commands::merge::run(inputs, output, duplicate_ok, threads, profile),

        Commands::Filter {
            input,
            ids,
            min_samples,
            max_samples,
            end_reason,
            exclude_end_reason,
            output,
        } => commands::filter::run(
            input,
            ids,
            min_samples,
            max_samples,
            end_reason,
            exclude_end_reason,
            output,
        ),

        Commands::BamFilter {
            input,
            bam,
            output,
            mapped,
            region,
            quality,
        } => commands::bam_filter::run(input, bam, output, mapped, region, quality),

        #[cfg(feature = "experimental")]
        Commands::Repack {
            inputs,
            output_dir,
            force,
        } => commands::repack::run(inputs, output_dir, force),

        Commands::Subset {
            input,
            csv,
            output_dir,
            force,
        } => commands::subset::run(input, csv, output_dir, force),

        Commands::Summary(args) => commands::summary::run(args),

        #[cfg(feature = "experimental")]
        Commands::Demux(args) => commands::demux::run(args),

        #[cfg(feature = "experimental")]
        Commands::Resquiggle(args) => commands::resquiggle::run(args),

        Commands::Index {
            inputs,
            force,
            threads,
        } => commands::index::run(inputs, force, threads),
    }
}
