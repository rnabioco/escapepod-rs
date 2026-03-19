//! Benchmark: reads_by_ids() (indexed vs scan) vs reads().collect() + filter
//!
//! Usage: cargo run --release --example bench_reads_by_ids -- <pod5_file> [num_target_ids]
//!
//! Temporarily hides the .p5i sidecar to measure the scan path separately.

use escapepod::Reader;
use std::collections::HashSet;
use std::path::PathBuf;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <pod5_file> [num_target_ids]", args[0]);
        std::process::exit(1);
    }

    let pod5_path = &args[1];
    let num_targets: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(100);

    let p5i_path = PathBuf::from(format!("{}.p5i", pod5_path));
    let p5i_hidden = PathBuf::from(format!("{}.p5i.bench_hidden", pod5_path));
    let has_index = p5i_path.exists();

    println!("Opening: {}", pod5_path);
    println!("Index:   {}", if has_index { "yes (.p5i)" } else { "no" });

    let reader = Reader::open(pod5_path)?;

    // Get total read count
    let t0 = Instant::now();
    let total = reader.read_count()?;
    println!("Total reads: {} (counted in {:.2?})", total, t0.elapsed());

    // Get a sample of read IDs using the fast projected scan
    let t0 = Instant::now();
    let all_ids = reader.read_ids()?;
    println!(
        "Fetched {} read IDs via read_ids() in {:.2?}",
        all_ids.len(),
        t0.elapsed()
    );

    // Pick target IDs (evenly spaced through the file)
    let step = (all_ids.len() / num_targets).max(1);
    let target_ids: HashSet<_> = all_ids
        .iter()
        .step_by(step)
        .take(num_targets)
        .copied()
        .collect();
    let actual_targets = target_ids.len();
    println!("Selected {} target IDs (step={})\n", actual_targets, step);

    // --- Benchmark 1: reads_by_ids() with index ---
    if has_index {
        println!("=== reads_by_ids() [indexed] ===");
        let reader = Reader::open(pod5_path)?;
        // Warm up the index load so we can measure it separately
        let t_idx = Instant::now();
        let _ = reader.read_index()?;
        let idx_load_time = t_idx.elapsed();
        println!("  Index load: {:.2?}", idx_load_time);
        let t0 = Instant::now();
        let matched = reader.reads_by_ids(&target_ids)?;
        let elapsed = t0.elapsed();
        println!(
            "  Found {} reads in {:.2?} ({:.2?} excluding index load)",
            matched.len(),
            idx_load_time + elapsed,
            elapsed,
        );

        // --- Benchmark 2: reads_by_ids() without index (hide .p5i) ---
        println!("\n=== reads_by_ids() [scan, no index] ===");
        std::fs::rename(&p5i_path, &p5i_hidden)?;
        let reader = Reader::open(pod5_path)?;
        let t0 = Instant::now();
        let matched_scan = reader.reads_by_ids(&target_ids)?;
        let elapsed_scan = t0.elapsed();
        std::fs::rename(&p5i_hidden, &p5i_path)?; // restore
        println!(
            "  Found {} reads in {:.2?}",
            matched_scan.len(),
            elapsed_scan,
        );

        // --- Benchmark 3: reads().collect() + filter ---
        println!("\n=== reads().collect() + filter ===");
        let reader = Reader::open(pod5_path)?;
        let t0 = Instant::now();
        let all_reads: Vec<_> = reader.reads()?.collect::<Result<Vec<_>, _>>()?;
        let elapsed_collect = t0.elapsed();
        let matched_slow: Vec<_> = all_reads
            .iter()
            .filter(|r| target_ids.contains(&r.read_id))
            .collect();
        let elapsed_total = t0.elapsed();
        println!(
            "  Collected {} reads in {:.2?}, filtered to {} in {:.2?} total",
            all_reads.len(),
            elapsed_collect,
            matched_slow.len(),
            elapsed_total
        );

        // --- Summary ---
        let indexed_total = idx_load_time + elapsed;
        println!("\n=== Summary ===");
        println!(
            "  reads_by_ids() indexed:    {:.2?} (index load {:.2?} + query {:.2?})",
            indexed_total, idx_load_time, elapsed
        );
        println!("  reads_by_ids() scan:       {:.2?}", elapsed_scan);
        println!("  reads()+filter:            {:.2?}", elapsed_total);
        println!(
            "  Speedup indexed vs old:    {:.1}x",
            elapsed_total.as_secs_f64() / indexed_total.as_secs_f64()
        );
        println!(
            "  Speedup scan vs old:       {:.1}x",
            elapsed_total.as_secs_f64() / elapsed_scan.as_secs_f64()
        );

        assert_eq!(matched.len(), matched_slow.len(), "indexed count mismatch");
        assert_eq!(
            matched_scan.len(),
            matched_slow.len(),
            "scan count mismatch"
        );
        println!("  Results match: ✓");
    } else {
        // No index — just compare scan vs reads().collect()
        println!("=== reads_by_ids() [scan] ===");
        let reader = Reader::open(pod5_path)?;
        let t0 = Instant::now();
        let matched = reader.reads_by_ids(&target_ids)?;
        let elapsed = t0.elapsed();
        println!("  Found {} reads in {:.2?}", matched.len(), elapsed,);

        println!("\n=== reads().collect() + filter ===");
        let reader = Reader::open(pod5_path)?;
        let t0 = Instant::now();
        let all_reads: Vec<_> = reader.reads()?.collect::<Result<Vec<_>, _>>()?;
        let elapsed_collect = t0.elapsed();
        let matched_slow: Vec<_> = all_reads
            .iter()
            .filter(|r| target_ids.contains(&r.read_id))
            .collect();
        let elapsed_total = t0.elapsed();
        println!(
            "  Collected {} reads in {:.2?}, filtered to {} in {:.2?} total",
            all_reads.len(),
            elapsed_collect,
            matched_slow.len(),
            elapsed_total
        );

        println!("\n=== Summary ===");
        println!("  reads_by_ids() scan:    {:.2?}", elapsed);
        println!("  reads()+filter:         {:.2?}", elapsed_total);
        println!(
            "  Speedup scan vs old:    {:.1}x",
            elapsed_total.as_secs_f64() / elapsed.as_secs_f64()
        );

        assert_eq!(matched.len(), matched_slow.len(), "count mismatch");
        println!("  Results match: ✓");
    }

    Ok(())
}
