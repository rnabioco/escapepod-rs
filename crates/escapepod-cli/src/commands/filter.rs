//! Filter command implementation.
//!
//! Filters reads from POD5 files based on various criteria including read IDs,
//! sample count (read length), and end reasons.
//! Uses batch-level parallelism with rayon and block-level copying for maximum performance.

use crate::commands::profile::PhaseTimer;
use crate::progress::create_progress_bar;
use crate::style;
use crate::util::{
    check_output_not_input, check_output_writable, collect_pod5_inputs, warn_if_not_portable,
};
use escapepod_signal::Durability;
use escapepod_signal::operations::{
    FilterCriteria, FilterOptions, filter_files_with_criteria, read_annotation, read_ids_from_file,
};
use escapepod_signal::types::EndReason;
use std::collections::HashSet;
use std::path::PathBuf;
use tracing::{info, warn};

#[allow(clippy::too_many_arguments)] // args mirror the CLI subcommand surface
pub fn run(
    input: Vec<PathBuf>,
    ids_file: Option<PathBuf>,
    min_samples: Option<u64>,
    max_samples: Option<u64>,
    end_reason: Option<Vec<String>>,
    exclude_end_reason: Option<Vec<String>>,
    annotation: Option<Vec<String>>,
    output: PathBuf,
    force: bool,
    profile: bool,
    durability: Durability,
) -> anyhow::Result<()> {
    check_output_writable(&output, force)?;

    let mut timer = PhaseTimer::new();
    timer.phase("Resolve inputs");

    // Resolve inputs to a list of POD5 files (files and/or directories). Taking
    // several is what makes this a drop-in for `pod5 filter`, which accepts a
    // list — a pipeline splitting one logical run across per-flowcell
    // directories has to name them all in one call or the read-ID list is
    // filtered against only part of the run.
    let files = collect_pod5_inputs(&input)?;
    warn_if_not_portable(&files);
    check_output_not_input(&output, &files)?;
    let is_directory = files.len() > 1;

    // Build filter criteria
    let mut criteria = FilterCriteria::default();

    // Load read IDs if specified
    if let Some(ref ids_path) = ids_file {
        timer.phase("Load read IDs");
        let ids = read_ids_from_file(ids_path)?;
        if ids.is_empty() {
            anyhow::bail!("No read IDs found in {}", ids_path.display());
        }
        criteria.read_ids = Some(ids);
    }

    // Set sample count filters
    criteria.min_samples = min_samples;
    criteria.max_samples = max_samples;

    // Parse end reason filters
    if let Some(reasons) = end_reason {
        let parsed: HashSet<EndReason> = reasons
            .iter()
            .map(|s| EndReason::from(s.as_str()))
            .collect();
        criteria.include_end_reasons = Some(parsed);
    }

    if let Some(reasons) = exclude_end_reason {
        let parsed: HashSet<EndReason> = reasons
            .iter()
            .map(|s| EndReason::from(s.as_str()))
            .collect();
        criteria.exclude_end_reasons = Some(parsed);
    }

    // Resolve sidecar-annotation filters into read IDs. Same-name pairs are
    // any-of, different names all-of, and an explicit --ids list further
    // intersects.
    if let Some(pairs) = annotation {
        timer.phase("Resolve annotations");
        let mut by_name: std::collections::HashMap<String, HashSet<String>> = Default::default();
        for pair in &pairs {
            let (name, label) = pair
                .split_once('=')
                .filter(|(n, l)| !n.is_empty() && !l.is_empty())
                .ok_or_else(|| anyhow::anyhow!("--annotation takes NAME=LABEL, got '{pair}'"))?;
            by_name
                .entry(name.to_string())
                .or_default()
                .insert(label.to_string());
        }
        let mut selected: Option<HashSet<escapepod_signal::Uuid>> = None;
        for (name, labels) in &by_name {
            let mut ids = HashSet::new();
            let mut known_labels: HashSet<&str> = HashSet::new();
            let annotations: Vec<_> = files
                .iter()
                .map(|file| {
                    read_annotation(file, Some(name))
                        .map_err(|e| anyhow::anyhow!("{}: {e}", file.display()))
                })
                .collect::<anyhow::Result<_>>()?;
            for ann in &annotations {
                known_labels.extend(ann.labels().iter().map(String::as_str));
                for (uuid, label) in ann.iter() {
                    if labels.contains(label) {
                        ids.insert(uuid);
                    }
                }
            }
            for label in labels {
                if !known_labels.contains(label.as_str()) {
                    warn!("label '{label}' does not occur in annotation '{name}'");
                }
            }
            selected = Some(match selected {
                None => ids,
                Some(prev) => prev.intersection(&ids).copied().collect(),
            });
        }
        if let Some(selected) = selected {
            criteria.read_ids = Some(match criteria.read_ids.take() {
                None => selected,
                Some(user) => user.intersection(&selected).copied().collect(),
            });
        }
    }

    // Validate that at least one criterion is set
    if criteria.is_empty() {
        anyhow::bail!(
            "No filter criteria specified. Use --ids, --min-samples, --max-samples, \
             --end-reason, --exclude-end-reason, or --annotation"
        );
    }

    // Print filtering info
    info!(
        "{} {}",
        style::action("Filtering"),
        if is_directory {
            // Name what the user typed, not the expansion — one directory can
            // stand for hundreds of files, and several inputs are now possible.
            let named = input
                .iter()
                .map(|p| p.display().to_string())
                .collect::<Vec<_>>()
                .join(", ");
            format!(
                "{} ({} files)",
                style::path(named),
                style::value(files.len())
            )
        } else {
            style::path(files[0].display()).to_string()
        }
    );

    // Print active criteria
    if let Some(ref ids) = criteria.read_ids {
        match &ids_file {
            Some(path) => info!(
                "{} {} read IDs from {}",
                style::label("IDs:"),
                style::count(ids.len()),
                style::path(path.display())
            ),
            None => info!(
                "{} {} read IDs from sidecar annotations",
                style::label("IDs:"),
                style::count(ids.len()),
            ),
        }
    }
    if let Some(min) = criteria.min_samples {
        info!("{} >= {}", style::label("Samples:"), style::value(min));
    }
    if let Some(max) = criteria.max_samples {
        info!("{} <= {}", style::label("Samples:"), style::value(max));
    }
    if let Some(ref reasons) = criteria.include_end_reasons {
        let reason_strs: Vec<_> = reasons.iter().map(|r| r.as_str()).collect();
        info!(
            "{} {}",
            style::label("End reasons:"),
            reason_strs.join(", ")
        );
    }
    if let Some(ref reasons) = criteria.exclude_end_reasons {
        let reason_strs: Vec<_> = reasons.iter().map(|r| r.as_str()).collect();
        info!(
            "{} {}",
            style::label("Exclude end reasons:"),
            reason_strs.join(", ")
        );
    }

    info!(
        "{} {}",
        style::label("Output:"),
        style::path(output.display())
    );

    // Estimate total reads for progress bar (we'll update as we go)
    let filter_bar = create_progress_bar(0, "Filtering")?;
    filter_bar.set_length(0); // Will be set by first progress callback

    let bar_for_callback = filter_bar.clone();

    // Create progress callback
    let progress: escapepod_signal::ProgressCallback =
        Box::new(move |p: escapepod_signal::Progress| {
            bar_for_callback.set_length(p.total);
            bar_for_callback.set_position(p.current);
        });

    // Use the core library's parallel filter
    let options = FilterOptions {
        signal_batch_size: 1_000,
        durability,
        // `read_batch_size` is deliberately left at the library default, so
        // the reads-table geometry has one definition rather than six.
        ..Default::default()
    };

    timer.phase("Filter & write");
    let result = filter_files_with_criteria(&files, &output, &criteria, options, Some(progress))?;

    filter_bar.finish_with_message(format!("{} matched", result.matched_reads));

    let percentage = result.match_percentage();
    info!(
        "{} {} reads from {} total ({})",
        style::action("Filtered"),
        style::count(result.matched_reads),
        result.total_reads,
        style::percentage(format!("{:.1}%", percentage))
    );

    // ID-specific warnings only if filtering by IDs
    if let Some(ref ids) = criteria.read_ids {
        let not_found = (ids.len() as u64).saturating_sub(result.matched_reads);
        if not_found > 0 {
            warn!(
                "{} requested IDs were not found in the input",
                style::warning(not_found)
            );
        }
        if result.matched_reads > ids.len() as u64 {
            warn!(
                "{} duplicate reads matched across multiple files",
                style::warning(result.matched_reads - ids.len() as u64)
            );
        }
    }

    // Report any errors encountered
    if result.read_errors > 0 || result.signal_errors > 0 {
        warn!(
            "encountered {} read error(s) and {} signal error(s)",
            style::error(result.read_errors),
            style::error(result.signal_errors)
        );
    }

    timer.report(profile);

    Ok(())
}
