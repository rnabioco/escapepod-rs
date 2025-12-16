//! BAM-based filter command implementation.
//!
//! Filters reads from a POD5 file based on a paired BAM file.
//! Supports filtering by mapped status, region, and mapping quality.
//! Uses lazy signal loading and block-level copying for maximum performance.

use crate::progress::{create_progress_bar, create_spinner};
use crate::style;
use crate::util::{
    batch_sizes, get_reads_iter_with_warning, open_reader_with_warning, resolve_pod5_inputs,
    LimitedWarningReporter, OpenResult,
};
use bstr::ByteSlice;
use noodles_bam as bam;
use noodles_core::Region;
use podfive_core::utils::{
    add_run_infos_deduplicated, map_run_info_index, parse_uuid_flexible, scan_dictionary_values,
};
use podfive_core::{PredefinedDictionaries, Writer, WriterOptions};
use std::collections::{HashMap, HashSet};
use std::io::BufReader;
use std::path::{Path, PathBuf};
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

    // Pre-scan files to collect unique dictionary values and count total reads
    let scan_spinner = create_spinner("Scanning")?;
    scan_spinner.set_message("POD5 files for dictionary values...");
    let scanned = scan_dictionary_values(&files, Some(&ids));
    scan_spinner.finish_with_message(format!(
        "{} reads found",
        style::count(scanned.total_read_count)
    ));

    // Create writer with predefined dictionaries for consistent multi-batch writes
    let options = WriterOptions {
        signal_batch_size: batch_sizes::SIGNAL_BATCH_SIZE,
        read_batch_size: batch_sizes::READ_BATCH_SIZE,
        predefined_dictionaries: Some(PredefinedDictionaries {
            pore_types: Some(scanned.pore_types.into_iter().collect()),
            end_reasons: Some(scanned.end_reasons.into_iter().collect()),
        }),
        ..WriterOptions::default()
    };
    let mut writer = Writer::create(&output, options)?;

    // Track run infos across all files
    let mut run_info_map: HashMap<String, u32> = HashMap::new();

    // Filter reads from all files
    let filter_bar = create_progress_bar(scanned.total_read_count, "Filtering")?;
    let mut matched = 0u64;
    let mut total = 0u64;
    let mut read_warnings = LimitedWarningReporter::new(3);
    let mut signal_warnings = LimitedWarningReporter::new(3);

    for file_path in &files {
        let reader = match open_reader_with_warning(file_path, is_directory) {
            OpenResult::Ok(r) => r,
            OpenResult::Skip => continue,
            OpenResult::Err(e) => return Err(e),
        };

        // Add run infos (deduplicated by acquisition_id)
        add_run_infos_deduplicated(&reader, &mut writer, &mut run_info_map)?;

        let reads_iter = match get_reads_iter_with_warning(&reader, file_path, is_directory) {
            OpenResult::Ok(iter) => iter,
            OpenResult::Skip => continue,
            OpenResult::Err(e) => return Err(e),
        };

        for read_result in reads_iter {
            let read = match read_result {
                Ok(r) => r,
                Err(e) => {
                    read_warnings.warn(&format!("error reading read record: {}", e));
                    continue;
                }
            };
            total += 1;
            filter_bar.inc(1);
            filter_bar.set_message(format!("{} matched", matched));

            // Check if this read's ID is in the BAM-derived filter set
            if ids.contains(&read.read_id) {
                // Lazy load: only fetch signal for matching reads
                let compressed_signal =
                    match reader.get_compressed_signal_for_rows(&read.signal_rows) {
                        Ok(s) => s,
                        Err(e) => {
                            signal_warnings.warn(&format!(
                                "cannot read signal for read {}: {}",
                                read.read_id, e
                            ));
                            continue;
                        }
                    };

                // Map run_info index
                let new_run_info_idx =
                    map_run_info_index(&reader, read.run_info_index, &run_info_map);

                // Create new read data for writing
                let new_read = read.for_writing(new_run_info_idx);

                writer.add_read_with_compressed_signal(new_read, &compressed_signal)?;
                matched += 1;
            }
        }
    }

    filter_bar.finish_with_message(format!("{} matched", matched));

    // Check for BAM/POD5 mismatch
    if matched == 0 && total > 0 {
        anyhow::bail!(
            "No overlap between BAM and POD5 files. \
             The BAM file may not correspond to this POD5 data. \
             POD5 contained {} reads, BAM filter matched {} read IDs.",
            total,
            ids.len()
        );
    }

    // Finalize output
    writer.finish()?;

    let percentage = if total > 0 {
        (matched as f64 / total as f64) * 100.0
    } else {
        0.0
    };
    println!(
        "{} {} reads from {} total ({})",
        style::action("Filtered"),
        style::count(matched),
        total,
        style::percentage(format!("{:.1}%", percentage))
    );

    let not_found = (ids.len() as u64).saturating_sub(matched);
    if not_found > 0 {
        println!(
            "{} {} BAM read IDs were not found in POD5 file(s)",
            style::note_label("Note:"),
            style::warning(not_found)
        );
    }

    // Report any errors encountered
    let read_errors = read_warnings.count();
    let signal_errors = signal_warnings.count();
    if read_errors > 0 || signal_errors > 0 {
        eprintln!(
            "{} encountered {} read error(s) and {} signal error(s)",
            style::error_label("Warning:"),
            style::error(read_errors),
            style::error(signal_errors)
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

/// Ensure a BAI index exists for the given BAM file, creating one if necessary.
///
/// Returns the path to the BAI file (either existing or newly created).
fn ensure_bai_index(bam_path: &Path) -> anyhow::Result<PathBuf> {
    // noodles expects the index at path.bam.bai
    let bai_path = bam_path.with_extension("bam.bai");

    if bai_path.exists() {
        return Ok(bai_path);
    }

    // Also check for path.bai (alternative naming convention)
    let alt_bai_path = bam_path.with_extension("bai");
    if alt_bai_path.exists() {
        // noodles indexed_reader expects .bam.bai, so we need to create it
        // or we could copy/symlink, but creating is safer
        eprintln!(
            "{} Found index at {} but noodles expects {}",
            style::note_label("Note:"),
            style::path(alt_bai_path.display()),
            style::path(bai_path.display())
        );
    }

    eprintln!(
        "{} BAI index not found, creating {}...",
        style::info("Info:"),
        style::path(bai_path.display())
    );

    // Build the index from the BAM file
    let index = bam::fs::index(bam_path)?;

    // Write the index to file
    bam::bai::fs::write(&bai_path, &index)?;

    eprintln!(
        "{} Created BAI index: {}",
        style::action("Done:"),
        style::path(bai_path.display())
    );

    Ok(bai_path)
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

        if should_include_record(&record, mapped_only, min_quality)? {
            if let Some(uuid) = extract_read_id(&record)? {
                ids.insert(uuid);
            }
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

        if should_include_record(&record, mapped_only, min_quality)? {
            if let Some(uuid) = extract_read_id(&record)? {
                ids.insert(uuid);
            }
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
