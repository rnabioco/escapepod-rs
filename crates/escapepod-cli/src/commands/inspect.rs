//! Inspect command implementation.

use crate::style;
use crate::util::resolve_pod5_inputs;
use escapepod_signal::{Reader, ReadsBatchView};
use std::path::PathBuf;
use tracing::warn;

pub fn summary(input: PathBuf) -> anyhow::Result<()> {
    let files = resolve_pod5_inputs(&input)?;
    let is_directory = files.len() > 1;

    println!("{}", style::header("POD5 File Summary"));
    println!("=================");
    println!();

    if is_directory {
        println!(
            "{} {}",
            style::key("Directory:"),
            style::path(input.display())
        );
        println!("{} {}", style::key("Files:"), style::count(files.len()));
    }

    let mut total_reads = 0usize;
    let mut total_batches = 0usize;

    for file_path in &files {
        let reader = match Reader::open(file_path) {
            Ok(r) => r,
            Err(e) => {
                if is_directory {
                    warn!(
                        "skipping {} ({})",
                        style::path(file_path.file_name().unwrap_or_default().to_string_lossy()),
                        e
                    );
                    continue;
                } else {
                    return Err(e.into());
                }
            }
        };

        if !is_directory {
            println!(
                "{} {}",
                style::key("File:"),
                style::path(file_path.display())
            );
            println!(
                "{} {}",
                style::key("File ID:"),
                style::value(reader.file_identifier())
            );
            println!(
                "{} {}",
                style::key("POD5 version:"),
                style::value(reader.pod5_version())
            );
            println!(
                "{} {}",
                style::key("Software:"),
                style::value(reader.software())
            );
            println!();
        }

        let read_count = reader.read_count().unwrap_or(0);
        let batch_count = reader.read_batch_count().unwrap_or(0);
        total_reads += read_count;
        total_batches += batch_count;

        if is_directory {
            println!(
                "  {}: {} reads, {} batches",
                style::path(file_path.file_name().unwrap_or_default().to_string_lossy()),
                style::count(read_count),
                batch_count
            );
        } else {
            println!("{} {}", style::key("Reads:"), style::count(read_count));
            println!("{} {}", style::key("Read batches:"), batch_count);
            // Say it here too, not only as the open-time warning: `inspect` is
            // where someone looks to decide whether a file is sound, and this
            // is the one fault escapepod reads straight through.
            if let Some(bad) = reader.nonuniform_signal_batch() {
                println!(
                    "{} signal batch {} has {} rows, expected {} — reads \
                     correctly here, but the official pod5 library and dorado \
                     assume a constant stride and will mis-resolve every signal \
                     index after it. Rewrite with `escpod repack`.",
                    style::warning("Signal batches: NOT PORTABLE —"),
                    bad.index,
                    bad.rows,
                    bad.expected,
                );
            }
            println!();

            println!(
                "{} {}",
                style::key("Run info entries:"),
                style::value(reader.run_info_count())
            );
            for (i, run_info) in reader.run_infos().iter().enumerate() {
                println!(
                    "  [{}] {}: {}",
                    i,
                    style::key("acquisition_id"),
                    style::value(&run_info.acquisition_id)
                );
                println!(
                    "      {}: {} Hz",
                    style::key("sample_rate"),
                    style::value(run_info.sample_rate)
                );
                println!(
                    "      {}: {}",
                    style::key("flow_cell_id"),
                    style::value(&run_info.flow_cell_id)
                );
            }

            println!();
            print_sidecar_summary(&reader, file_path);
        }
    }

    if is_directory {
        println!();
        println!(
            "{} {}",
            style::key("Total reads:"),
            style::count(total_reads)
        );
        println!("{} {}", style::key("Total batches:"), total_batches);
    }

    Ok(())
}

/// One-look summary of the `.p5s` sidecar: index, annotations, design.
fn print_sidecar_summary(reader: &Reader, pod5_path: &std::path::Path) {
    use escapepod_signal::pod5::sidecar::{read_sidecar_file, sidecar_path};

    let p5s_path = sidecar_path(pod5_path);
    if !p5s_path.exists() {
        println!("{} none", style::key("Sidecar:"));
        return;
    }

    let sidecar = reader
        .sidecar_identity()
        .map_err(anyhow::Error::from)
        .and_then(|identity| read_sidecar_file(&p5s_path, &identity).map_err(Into::into));
    let sidecar = match sidecar {
        Ok(Some(s)) => s,
        Ok(None) => {
            println!("{} none", style::key("Sidecar:"));
            return;
        }
        Err(e) => {
            println!(
                "{} {} — {}",
                style::key("Sidecar:"),
                style::path(p5s_path.display()),
                e
            );
            return;
        }
    };

    println!(
        "{} {}",
        style::key("Sidecar:"),
        style::path(p5s_path.display())
    );
    println!(
        "  {}: {} reads",
        style::key("index"),
        style::count(sidecar.len())
    );
    if sidecar.annotations().is_empty() {
        println!("  {}: none", style::key("annotations"));
    } else {
        let described: Vec<String> = sidecar
            .annotations()
            .iter()
            .map(|a| {
                format!(
                    "{} ({} labels, {} reads)",
                    a.name(),
                    a.labels().len(),
                    a.len()
                )
            })
            .collect();
        println!(
            "  {}: {}",
            style::key("annotations"),
            style::value(described.join(", "))
        );
    }
    if let Some(design) = sidecar.design() {
        println!(
            "  {}: [{}] → [{}], {} rows",
            style::key("design"),
            design.key_columns.join(","),
            design.value_columns.join(","),
            style::count(design.rows.len())
        );
    }
}

pub fn reads(input: PathBuf) -> anyhow::Result<()> {
    let files = resolve_pod5_inputs(&input)?;
    let is_directory = files.len() > 1;

    println!(
        "{:<36} {:>8} {:>4} {:>10} {:>12}",
        "read_id", "channel", "well", "samples", "end_reason"
    );
    println!("{}", "-".repeat(76));

    for file_path in &files {
        let reader = match Reader::open(file_path) {
            Ok(r) => r,
            Err(e) => {
                if is_directory {
                    warn!(
                        "skipping {} ({})",
                        style::path(file_path.file_name().unwrap_or_default().to_string_lossy()),
                        e
                    );
                    continue;
                } else {
                    return Err(e.into());
                }
            }
        };

        let batches = match reader.read_batches() {
            Ok(b) => b,
            Err(e) => {
                if is_directory {
                    warn!(
                        "cannot read {} ({})",
                        style::path(file_path.file_name().unwrap_or_default().to_string_lossy()),
                        e
                    );
                    continue;
                } else {
                    return Err(e.into());
                }
            }
        };

        for batch_result in batches {
            let Ok(batch) = batch_result else { continue };
            let Ok(view) = ReadsBatchView::new(&batch, false) else {
                continue;
            };
            for row in 0..view.num_rows() {
                let Ok(read) = view.read(row) else { continue };
                println!(
                    "{:<36} {:>8} {:>4} {:>10} {:>12}",
                    read.read_id, read.channel, read.well, read.num_samples, read.end_reason
                );
            }
        }
    }

    Ok(())
}

pub fn read(input: PathBuf, read_id: String) -> anyhow::Result<()> {
    let files = resolve_pod5_inputs(&input)?;
    let is_directory = files.len() > 1;
    let target_id: uuid::Uuid = read_id.parse()?;

    for file_path in &files {
        let reader = match Reader::open(file_path) {
            Ok(r) => r,
            Err(_) if is_directory => continue,
            Err(e) => return Err(e.into()),
        };

        // Use the indexed by-id lookup — an O(log n) binary search that
        // touches one batch when a .p5s sidecar exists — instead of
        // scanning every batch.
        let mut targets = std::collections::HashSet::new();
        targets.insert(target_id);
        let matches = match reader.reads_by_ids(&targets) {
            Ok(m) => m,
            Err(_) if is_directory => continue,
            Err(e) => return Err(e.into()),
        };

        for read in matches {
            if read.read_id == target_id {
                println!("{}", style::header("Read Details"));
                println!("============");
                println!();
                if is_directory {
                    println!(
                        "{}: {}",
                        style::key("file"),
                        style::path(file_path.display())
                    );
                }
                println!("{}: {}", style::key("read_id"), style::value(read.read_id));
                println!(
                    "{}: {}",
                    style::key("read_number"),
                    style::value(read.read_number)
                );
                println!("{}: {}", style::key("channel"), style::value(read.channel));
                println!("{}: {}", style::key("well"), style::value(read.well));
                println!(
                    "{}: {}",
                    style::key("start_sample"),
                    style::value(read.start_sample)
                );
                println!(
                    "{}: {}",
                    style::key("num_samples"),
                    style::count(read.num_samples)
                );
                println!(
                    "{}: {}",
                    style::key("num_minknow_events"),
                    style::value(read.num_minknow_events)
                );
                println!();
                println!(
                    "{}: {}",
                    style::key("pore_type"),
                    style::value(&read.pore_type)
                );
                println!(
                    "{}: {}",
                    style::key("calibration_offset"),
                    style::value(read.calibration_offset)
                );
                println!(
                    "{}: {}",
                    style::key("calibration_scale"),
                    style::value(read.calibration_scale)
                );
                println!(
                    "{}: {}",
                    style::key("median_before"),
                    style::value(read.median_before)
                );
                println!(
                    "{}: {}",
                    style::key("open_pore_level"),
                    style::value(read.open_pore_level)
                );
                println!(
                    "{}: {}",
                    style::key("expected_open_pore_level"),
                    style::value(read.expected_open_pore_level)
                );
                println!(
                    "{}: {}",
                    style::key("selected_read_level"),
                    style::value(read.selected_read_level)
                );
                println!();
                println!(
                    "{}: {}",
                    style::key("end_reason"),
                    style::value(read.end_reason)
                );
                println!(
                    "{}: {}",
                    style::key("end_reason_forced"),
                    style::value(read.end_reason_forced)
                );
                return Ok(());
            }
        }
    }

    anyhow::bail!("Read not found: {}", read_id)
}
