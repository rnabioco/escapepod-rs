// SPDX-License-Identifier: MIT

//! What opening a POD5 costs before a single read is fetched.
//!
//! escapepod-rs#334 measured a `classify` run at 17.6 reads/s and 236 major
//! faults/s at 0.00 cores, and the work since has gone into the two things the
//! issue named — the lookup and the read order. Neither is what a cold run
//! spends its first minutes on. Instrumenting the probe that measured them
//! showed 85-95% of its wall clock outside every timer it had: not in the
//! lookup, not in the fetch, but between them, in the calls that *construct*
//! the reader.
//!
//! Two of those, and this times them apart because their fixes are different:
//!
//! * `read_index()` — the reads-table scan that maps read id to batch and row,
//!   or the `.p5s` sidecar that already holds it. Front-to-back over one
//!   projected column, so the kernel's readahead serves it well; ~120-160 ms
//!   on a 326 k-read file, and near zero with a sidecar.
//! * `signal_extractor()` — the Arrow IPC footer parse, which must recover
//!   each signal batch's row count. That number is the one field the footer's
//!   block list does not carry, so it means reading every batch's own message
//!   header: one scattered touch per batch, thousands on a large file, each a
//!   server round trip on BeeGFS. This is the stall.
//!
//! Neither is charged to a read, so neither shows up in reads/s, and both are
//! paid per file per process — which is why a run over 70 POD5s can sit for
//! minutes at no CPU before it looks like it has started.
//!
//! ```text
//! cargo run --release -p escapepod-pod5 --example open_cost -- <dir-or-files>
//! ESCAPEPOD_POD5_FOOTER_SERIAL=1 cargo run … # the pre-parallel walk, for A/B
//! ```
//!
//! Arms belong in separate processes and on files nothing has touched: the
//! setting is read once per process, and a second look at the same file is
//! served from the page cache and measures nothing.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use escapepod_pod5::Reader;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        return Err("usage: open_cost <pod5 file or directory>...".into());
    }

    let mut paths: Vec<PathBuf> = Vec::new();
    for arg in &args {
        let p = PathBuf::from(arg);
        if p.is_dir() {
            for entry in std::fs::read_dir(&p)? {
                let path = entry?.path();
                if path.extension().is_some_and(|e| e == "pod5") {
                    paths.push(path);
                }
            }
        } else {
            paths.push(p);
        }
    }
    paths.sort();

    println!(
        "{} file(s), footer walk {}",
        paths.len(),
        match std::env::var("ESCAPEPOD_POD5_FOOTER_SERIAL").as_deref() {
            Ok("1") | Ok("true") => "SERIAL",
            _ => "parallel",
        }
    );

    let (mut t_open, mut t_index, mut t_extract) = (Duration::ZERO, Duration::ZERO, Duration::ZERO);
    let faults0 = major_faults();

    for path in &paths {
        let t0 = Instant::now();
        let reader = Reader::open(path)?;
        let open = t0.elapsed();

        // Ordered as a caller pays them, and timed apart because they are two
        // different costs with two different fixes.
        let t1 = Instant::now();
        let reads = reader.read_index()?.len();
        let index = t1.elapsed();

        let t2 = Instant::now();
        let extractor = reader.signal_extractor()?;
        let extract = t2.elapsed();

        println!(
            "  {:<44}  open {:>8.1?}  read_index {:>9.1?} ({reads} reads)  \
             signal_extractor {:>9.1?}",
            path.file_name().unwrap_or_default().to_string_lossy(),
            open,
            index,
            extract,
        );
        // Keep the extractor alive across the print so nothing above is
        // measuring a drop, and use it so it cannot be optimised away.
        std::hint::black_box(&extractor);

        t_open += open;
        t_index += index;
        t_extract += extract;
    }

    let total = t_open + t_index + t_extract;
    println!(
        "\ntotals over {} file(s):  open {:.2?}  read_index {:.2?}  \
         signal_extractor {:.2?}  =  {:.2?}   ({} major faults)",
        paths.len(),
        t_open,
        t_index,
        t_extract,
        total,
        major_faults().saturating_sub(faults0),
    );
    Ok(())
}

/// Major faults this process has taken (field 12 of `/proc/self/stat`).
///
/// A POD5 is an mmap, so its I/O never appears in `read_bytes`, and a process
/// stalled on faults shows no CPU either. This is the quantity that makes the
/// stall visible at all.
fn major_faults() -> u64 {
    let Ok(stat) = std::fs::read_to_string("/proc/self/stat") else {
        return 0;
    };
    // Field 2 (comm) may contain spaces inside parentheses; split after it.
    let Some((_, rest)) = stat.rsplit_once(") ") else {
        return 0;
    };
    rest.split_whitespace()
        .nth(9)
        .and_then(|v| v.parse().ok())
        .unwrap_or(0)
}
