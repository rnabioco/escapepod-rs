//! Cold-read A/B: explicit `pread` (buffered read syscalls, like `dd`) vs
//! `mmap` page-touch, over disjoint cold regions of the same file. Tests the
//! hypothesis that mmap demand-paging — not the access pattern — is what
//! collapses cold throughput on BeeGFS (where `dd` streams at ~775 MB/s but
//! every mmap-based reader path crawls at single-digit MB/s).
//!
//! Run: cargo run --release --example mmap_vs_read -- <file> [mb_per_phase]

use std::fs::File;
use std::io::Read;
use std::time::Instant;

use memmap2::Mmap;

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .expect("usage: mmap_vs_read <file> [mb_per_phase]");
    let mb: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(256);
    let region = mb * 1024 * 1024;

    let file_len = std::fs::metadata(&path).expect("stat").len() as usize;
    assert!(
        file_len >= 2 * region,
        "file too small for two disjoint regions"
    );

    // Phase 1: pread the FIRST region with explicit read() syscalls (8 MiB bufs).
    let mut f = File::open(&path).expect("open");
    let mut buf = vec![0u8; 8 * 1024 * 1024];
    let mut got = 0usize;
    let mut checksum = 0u64;
    let t0 = Instant::now();
    while got < region {
        let n = f.read(&mut buf).expect("read");
        if n == 0 {
            break;
        }
        checksum = checksum.wrapping_add(buf[..n].iter().map(|&b| b as u64).sum::<u64>());
        got += n;
    }
    let dt_read = t0.elapsed().as_secs_f64();

    // Phase 2: mmap, advise sequential (matches Reader::open), touch every 4 KiB
    // page of the SECOND, disjoint region — pure demand-paging, no read().
    let mmap = unsafe { Mmap::map(&File::open(&path).expect("open2")).expect("mmap") };
    #[cfg(unix)]
    let _ = mmap.advise(memmap2::Advice::Sequential);
    let start = region;
    let end = (start + region).min(mmap.len());
    let t1 = Instant::now();
    let mut sum: u64 = 0;
    let mut i = start;
    while i < end {
        sum = sum.wrapping_add(mmap[i] as u64);
        i += 4096;
    }
    let dt_mmap = t1.elapsed().as_secs_f64();

    let rate = |bytes: usize, dt: f64| bytes as f64 / 1e6 / dt;
    println!("region per phase: {mb} MiB   (checksum {checksum}, sum {sum})");
    println!(
        "pread (read syscalls): {:.2}s = {:.0} MB/s",
        dt_read,
        rate(got, dt_read)
    );
    println!(
        "mmap (page-touch):     {:.2}s = {:.0} MB/s",
        dt_mmap,
        rate(end - start, dt_mmap)
    );
    println!(
        "pread / mmap ratio:    {:.0}x",
        rate(got, dt_read) / rate(end - start, dt_mmap).max(0.001)
    );
}
