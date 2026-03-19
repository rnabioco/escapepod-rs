//! Build `.p5i` sidecar indexes for fast read lookup in POD5 files.

use std::path::PathBuf;
use std::time::Instant;

use crate::style;
use crate::util::resolve_pod5_inputs;

/// Build `.p5i` read index for one or more POD5 files.
///
/// The index maps each read UUID to its location in the reads table,
/// enabling O(1) lookup instead of a full-table scan. The sidecar is
/// written next to the POD5 file by appending `.p5i` to the full
/// filename (e.g. `reads.pod5` → `reads.pod5.p5i`).
pub fn run(inputs: Vec<PathBuf>, force: bool) -> anyhow::Result<()> {
    let mut files = Vec::new();
    for input in &inputs {
        files.extend(resolve_pod5_inputs(input)?);
    }

    if files.is_empty() {
        anyhow::bail!("No POD5 files found");
    }

    let total = files.len();
    eprintln!(
        "{} Building read indexes for {} file(s)...",
        style::action("Index:"),
        style::count(total),
    );

    let mut indexed = 0usize;
    let mut skipped = 0usize;

    for pod5_path in &files {
        let p5i_path = {
            let mut s = pod5_path.as_os_str().to_owned();
            s.push(".p5i");
            PathBuf::from(s)
        };

        if p5i_path.exists() && !force {
            eprintln!(
                "  {} {} (already exists, use --force to overwrite)",
                style::info("skip"),
                style::path(p5i_path.display()),
            );
            skipped += 1;
            continue;
        }

        let t0 = Instant::now();
        let reader = escapepod::Reader::open(pod5_path)?;
        let count = reader.build_and_write_index(&p5i_path)?;
        let elapsed = t0.elapsed();

        eprintln!(
            "  {} {} — {} reads in {:.1}s",
            style::action("wrote"),
            style::path(p5i_path.display()),
            style::count(count),
            elapsed.as_secs_f64(),
        );
        indexed += 1;
    }

    eprintln!(
        "{} {} indexed, {} skipped",
        style::action("Done:"),
        style::count(indexed),
        style::count(skipped),
    );

    Ok(())
}
