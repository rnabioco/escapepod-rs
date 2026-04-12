//! BAM-based filter command implementation.
//!
//! Filters reads from a POD5 file based on a paired BAM file.
//! Supports filtering by mapped status, region, and mapping quality.
//! Uses the optimized filter_files() for maximum performance.

use crate::progress::{create_progress_bar, create_spinner};
use crate::style;
use crate::util::{ensure_bai_index, resolve_pod5_inputs};
use bstr::ByteSlice;
use escapepod::operations::{FilterOptions, filter_files};
use escapepod::parse_uuid_flexible;
use noodles_bam as bam;
use noodles_core::Region;
use std::collections::HashSet;
use std::io::BufReader;
use std::path::PathBuf;
use uuid::Uuid;

/// Run the bam-filter command.
pub fn run(
    input: PathBuf,
    bam_path: PathBuf,
    output: PathBuf,
    mapped_only: bool,
    region: Option<String>,
    min_quality: Option<u8>,
) -> anyhow::Result<()> {
    // Resolve input to list of POD5 files (supports directories)
    let files = resolve_pod5_inputs(&input)?;
    let is_directory = files.len() > 1;

    println!(
        "{} {} using BAM {}",
        style::action("Filtering"),
        if is_directory {
            format!(
                "{} ({} files)",
                style::path(input.display()),
                style::value(files.len())
            )
        } else {
            style::path(input.display())
        },
        style::path(bam_path.display())
    );
    println!(
        "{} {}",
        style::label("Output:"),
        style::path(output.display())
    );

    // Print filter criteria
    if mapped_only {
        println!("  Filter: {}", style::value("mapped reads only"));
    }
    if let Some(ref r) = region {
        println!("  Filter: region {}", style::value(r));
    }
    if let Some(q) = min_quality {
        println!("  Filter: MAPQ >= {}", style::value(q));
    }

    // Read IDs from BAM file
    let bam_spinner = create_spinner("Scanning")?;
    bam_spinner.set_message("BAM file...");
    let (ids, bam_records_scanned) =
        read_ids_from_bam(&bam_path, mapped_only, region.as_deref(), min_quality)?;
    bam_spinner.finish_with_message(format!(
        "{} read IDs from {} BAM records",
        style::count(ids.len()),
        bam_records_scanned
    ));

    if ids.is_empty() {
        anyhow::bail!(
            "No matching reads found in BAM file with the specified filters. \
             Check that the BAM file contains reads matching your criteria."
        );
    }

    // Set up progress bar
    let filter_bar = create_progress_bar(0, "Filtering")?;
    filter_bar.set_length(0); // Will be set by first progress callback

    let bar_for_callback = filter_bar.clone();

    // Create progress callback
    let progress: escapepod::ProgressCallback = Box::new(move |p: escapepod::Progress| {
        bar_for_callback.set_length(p.total);
        bar_for_callback.set_position(p.current);
    });

    // Use the core library's optimized filter
    let options = FilterOptions {
        signal_batch_size: 1_000,
        read_batch_size: 10_000,
    };

    let result = filter_files(&files, &output, &ids, options, Some(progress))?;

    filter_bar.finish_with_message(format!("{} matched", result.matched_reads));

    // Check for BAM/POD5 mismatch
    if result.matched_reads == 0 && result.total_reads > 0 {
        anyhow::bail!(
            "No overlap between BAM and POD5 files. \
             The BAM file may not correspond to this POD5 data. \
             POD5 contained {} reads, BAM filter matched {} read IDs.",
            result.total_reads,
            ids.len()
        );
    }

    let percentage = result.match_percentage();
    println!(
        "{} {} reads from {} total ({})",
        style::action("Filtered"),
        style::count(result.matched_reads),
        result.total_reads,
        style::percentage(format!("{:.1}%", percentage))
    );

    let not_found = (ids.len() as u64).saturating_sub(result.matched_reads);
    if not_found > 0 {
        println!(
            "{} {} BAM read IDs were not found in POD5 file(s)",
            style::note_label("Note:"),
            style::warning(not_found)
        );
    }

    // Report any errors encountered
    if result.read_errors > 0 || result.signal_errors > 0 {
        eprintln!(
            "{} encountered {} read error(s) and {} signal error(s)",
            style::error_label("Warning:"),
            style::error(result.read_errors),
            style::error(result.signal_errors)
        );
    }

    Ok(())
}

/// Read read IDs from a BAM file, applying optional filters.
///
/// Returns a tuple of (matching read IDs, total records scanned).
fn read_ids_from_bam(
    bam_path: &PathBuf,
    mapped_only: bool,
    region: Option<&str>,
    min_quality: Option<u8>,
) -> anyhow::Result<(HashSet<Uuid>, u64)> {
    let mut ids = HashSet::new();
    let mut records_scanned = 0u64;

    if let Some(region_str) = region {
        // Region query requires index
        read_ids_from_bam_region(
            bam_path,
            region_str,
            mapped_only,
            min_quality,
            &mut ids,
            &mut records_scanned,
        )?;
    } else {
        // Full file scan
        read_ids_from_bam_full(
            bam_path,
            mapped_only,
            min_quality,
            &mut ids,
            &mut records_scanned,
        )?;
    }

    Ok((ids, records_scanned))
}

/// Read IDs from BAM using indexed region query.
fn read_ids_from_bam_region(
    bam_path: &PathBuf,
    region_str: &str,
    mapped_only: bool,
    min_quality: Option<u8>,
    ids: &mut HashSet<Uuid>,
    records_scanned: &mut u64,
) -> anyhow::Result<()> {
    // Ensure BAI index exists (create if needed)
    ensure_bai_index(bam_path)?;

    // Build indexed reader
    let mut reader = bam::io::indexed_reader::Builder::default()
        .build_from_path(bam_path)
        .map_err(|e| {
            anyhow::anyhow!(
                "Cannot open indexed BAM file '{}': {}",
                bam_path.display(),
                e
            )
        })?;

    let header = reader.read_header()?;

    // Parse region
    let region: Region = region_str.parse().map_err(|e| {
        anyhow::anyhow!(
            "Invalid region '{}': {}. Expected format: chr or chr:start-end",
            region_str,
            e
        )
    })?;

    // Query the region
    let query = reader.query(&header, &region)?;

    for result in query.records() {
        let record = result?;
        *records_scanned += 1;

        if should_include_record(&record, mapped_only, min_quality)?
            && let Some(uuid) = extract_read_id(&record)?
        {
            ids.insert(uuid);
        }
    }

    Ok(())
}

/// Read IDs from BAM by scanning the full file.
fn read_ids_from_bam_full(
    bam_path: &PathBuf,
    mapped_only: bool,
    min_quality: Option<u8>,
    ids: &mut HashSet<Uuid>,
    records_scanned: &mut u64,
) -> anyhow::Result<()> {
    let file = std::fs::File::open(bam_path)?;
    let mut reader = bam::io::Reader::new(BufReader::new(file));

    let header = reader.read_header()?;

    for result in reader.records() {
        let record = result?;
        *records_scanned += 1;

        if should_include_record(&record, mapped_only, min_quality)?
            && let Some(uuid) = extract_read_id(&record)?
        {
            ids.insert(uuid);
        }
    }

    // Silence unused variable warning
    let _ = header;

    Ok(())
}

/// Check if a BAM record should be included based on filters.
fn should_include_record(
    record: &bam::Record,
    mapped_only: bool,
    min_quality: Option<u8>,
) -> anyhow::Result<bool> {
    let flags = record.flags();

    // Check mapped status
    if mapped_only && flags.is_unmapped() {
        return Ok(false);
    }

    // Check mapping quality
    if let Some(min_q) = min_quality {
        let mapq = record.mapping_quality();
        // Mapping quality is Option<MappingQuality>
        match mapq {
            Some(q) => {
                if u8::from(q) < min_q {
                    return Ok(false);
                }
            }
            None => {
                // Unmapped reads or reads with unavailable MAPQ
                if min_q > 0 {
                    return Ok(false);
                }
            }
        }
    }

    Ok(true)
}

/// Extract read ID (UUID) from BAM record's query name.
///
/// Oxford Nanopore read names are UUIDs, e.g., `a1b2c3d4-e5f6-7890-abcd-ef1234567890`
fn extract_read_id(record: &bam::Record) -> anyhow::Result<Option<Uuid>> {
    let name = record.name();
    match name {
        Some(n) => {
            // BStr derefs to [u8], use ByteSlice trait for to_str()
            let name_str = n.to_str()?;
            match parse_uuid_flexible(name_str) {
                Ok(uuid) => Ok(Some(uuid)),
                Err(_) => {
                    // Not a valid UUID - skip silently (might be non-ONT data)
                    Ok(None)
                }
            }
        }
        None => Ok(None),
    }
}
