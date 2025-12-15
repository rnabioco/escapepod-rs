//! POD5 file CLI tools.

use clap::{Parser, Subcommand};
use std::path::PathBuf;

mod commands;
mod util;

#[derive(Parser)]
#[command(name = "podfive")]
#[command(author, version, about = "CLI tools for POD5 files", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// View read summaries from a POD5 file
    View {
        /// Input POD5 file
        input: PathBuf,

        /// Fields to include (comma-separated)
        #[arg(long)]
        include: Option<String>,

        /// Fields to exclude (comma-separated)
        #[arg(long)]
        exclude: Option<String>,

        /// Output only read IDs
        #[arg(long)]
        ids: bool,

        /// Output file (stdout if not specified)
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Field separator (default: tab)
        #[arg(long, default_value = "\t")]
        separator: String,

        /// Don't print header row
        #[arg(long)]
        no_header: bool,
    },

    /// Inspect POD5 file contents
    Inspect {
        #[command(subcommand)]
        command: InspectCommands,
    },

    /// Merge multiple POD5 files
    Merge {
        /// Input POD5 files
        #[arg(required = true)]
        inputs: Vec<PathBuf>,

        /// Output POD5 file
        #[arg(short, long, required = true)]
        output: PathBuf,

        /// Allow duplicate read IDs (skip duplicate checking)
        #[arg(long)]
        duplicate_ok: bool,

        /// Number of parallel file readers (default: number of CPUs)
        #[arg(short, long)]
        threads: Option<usize>,
    },

    /// Filter reads by ID
    Filter {
        /// Input POD5 file
        input: PathBuf,

        /// File containing read IDs (one per line)
        #[arg(short, long, required = true)]
        ids: PathBuf,

        /// Output POD5 file
        #[arg(short, long, required = true)]
        output: PathBuf,
    },

    /// Repack POD5 files to optimize storage
    Repack {
        /// Input POD5 files
        #[arg(required = true)]
        inputs: Vec<PathBuf>,

        /// Output directory
        #[arg(short, long, required = true)]
        output_dir: PathBuf,

        /// Overwrite existing files
        #[arg(short, long)]
        force: bool,
    },

    /// Subset reads into multiple files based on CSV mapping
    Subset {
        /// Input POD5 file
        input: PathBuf,

        /// CSV file with read_id,output columns
        #[arg(long, required = true)]
        csv: PathBuf,

        /// Output directory
        #[arg(short, long, default_value = ".")]
        output_dir: PathBuf,

        /// Overwrite existing files
        #[arg(short, long)]
        force: bool,
    },

    /// Show comprehensive summary of POD5 file(s)
    Summary(commands::summary::SummaryArgs),
}

#[derive(Subcommand)]
enum InspectCommands {
    /// Show file summary
    Summary {
        /// Input POD5 file
        input: PathBuf,
    },

    /// List all reads
    Reads {
        /// Input POD5 file
        input: PathBuf,
    },

    /// Show details for a specific read
    Read {
        /// Input POD5 file
        input: PathBuf,

        /// Read ID to inspect
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
        } => commands::merge::run(inputs, output, duplicate_ok, threads),

        Commands::Filter { input, ids, output } => commands::filter::run(input, ids, output),

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
    }
}
