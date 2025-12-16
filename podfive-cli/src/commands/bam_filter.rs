//! BAM-based filter command implementation.
//!
//! Filters reads from a POD5 file based on a paired BAM file.
//! Supports filtering by mapped status, region, and mapping quality.
//! Uses lazy signal loading and block-level copying for maximum performance.

use crate::util::{parse_uuid_flexible, resolve_pod5_inputs};
use bstr::ByteSlice;
use noodles_bam as bam;
use noodles_core::Region;
use podfive_core::{Reader, Writer, WriterOptions};
use std::collections::{HashMap, HashSet};
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
        "Filtering {} using BAM {}",
        if is_directory {
            format!("{} ({} files)", input.display(), files.len())
        } else {
            input.display().to_string()
        },
        bam_path.display()
    );
    println!("Output: {}", output.display());

    // Print filter criteria
    if mapped_only {
        println!("  Filter: mapped reads only");
    }
    if let Some(ref r) = region {
        println!("  Filter: region {}", r);
    }
    if let Some(q) = min_quality {
        println!("  Filter: MAPQ >= {}", q);
    }

    // Read IDs from BAM file
    let (ids, bam_records_scanned) =
        read_ids_from_bam(&bam_path, mapped_only, region.as_deref(), min_quality)?;
    println!(
        "Found {} read IDs from {} BAM records",
        ids.len(),
        bam_records_scanned
    );

    if ids.is_empty() {
        anyhow::bail!(
            "No matching reads found in BAM file with the specified filters. \
             Check that the BAM file contains reads matching your criteria."
        );
    }

    // Create writer with optimized batch sizes
    let options = WriterOptions {
        signal_batch_size: 1_000,
        read_batch_size: 500_000,
        ..Default::default()
    };
    let mut writer = Writer::create(&output, options)?;

    // Track run infos across all files
    let mut run_info_map: HashMap<String, u32> = HashMap::new();

    // Filter reads from all files
    let mut matched = 0u64;
    let mut total = 0u64;
    let mut read_errors = 0u64;
    let mut signal_errors = 0u64;

    for file_path in &files {
        let reader = match Reader::open(file_path) {
            Ok(r) => r,
            Err(e) => {
                if is_directory {
                    eprintln!(
                        "Warning: skipping {} ({})",
                        file_path.file_name().unwrap_or_default().to_string_lossy(),
                        e
                    );
                    continue;
                } else {
                    return Err(e.into());
                }
            }
        };

        // Add run infos (deduplicated by acquisition_id)
        for run_info in reader.run_infos() {
            if !run_info_map.contains_key(&run_info.acquisition_id) {
                let idx = writer.add_run_info(run_info.clone())?;
                run_info_map.insert(run_info.acquisition_id.clone(), idx);
            }
        }

        let reads_iter = match reader.reads() {
            Ok(iter) => iter,
            Err(e) => {
                if is_directory {
                    eprintln!(
                        "Warning: cannot read {} ({})",
                        file_path.file_name().unwrap_or_default().to_string_lossy(),
                        e
                    );
                    continue;
                } else {
                    return Err(e.into());
                }
            }
        };

        for read_result in reads_iter {
            let read = match read_result {
                Ok(r) => r,
                Err(e) => {
                    read_errors += 1;
                    if read_errors <= 3 {
                        eprintln!("Warning: error reading read record: {}", e);
                    }
                    continue;
                }
            };
            total += 1;

            // Check if this read's ID is in the BAM-derived filter set
            if ids.contains(&read.read_id) {
                // Lazy load: only fetch signal for matching reads
                let compressed_signal = match reader.get_compressed_signal_for_rows(&read.signal_rows)
                {
                    Ok(s) => s,
                    Err(e) => {
                        signal_errors += 1;
                        if signal_errors <= 3 {
                            eprintln!(
                                "Warning: cannot read signal for read {}: {}",
                                read.read_id, e
                            );
                        }
                        continue;
                    }
                };

                // Map run_info index
                let new_run_info_idx = if let Some(original_run_info) =
                    reader.get_run_info(read.run_info_index as usize)
                {
                    *run_info_map
                        .get(&original_run_info.acquisition_id)
                        .unwrap_or(&0)
                } else {
                    0
                };

                // Create new read data for writing
                let new_read = read.for_writing(new_run_info_idx);

                writer.add_read_with_compressed_signal(new_read, &compressed_signal)?;
                matched += 1;
            }
        }
    }

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

    println!(
        "Filtered {} reads from {} total ({:.1}%)",
        matched,
        total,
        if total > 0 {
            (matched as f64 / total as f64) * 100.0
        } else {
            0.0
        }
    );

    let not_found = (ids.len() as u64).saturating_sub(matched);
    if not_found > 0 {
        println!(
            "Note: {} BAM read IDs were not found in POD5 file(s)",
            not_found
        );
    }

    // Report any errors encountered
    if read_errors > 0 || signal_errors > 0 {
        eprintln!(
            "Warning: encountered {} read error(s) and {} signal error(s)",
            read_errors, signal_errors
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
        read_ids_from_bam_full(bam_path, mapped_only, min_quality, &mut ids, &mut records_scanned)?;
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
    // Build indexed reader
    let mut reader = bam::io::indexed_reader::Builder::default()
        .build_from_path(bam_path)
        .map_err(|e| {
            anyhow::anyhow!(
                "Cannot open indexed BAM file '{}': {}. \
                 For region queries, ensure a .bai index exists (run `samtools index`)",
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
