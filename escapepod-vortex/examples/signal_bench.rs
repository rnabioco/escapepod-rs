//! POD5 vs Vortex micro-benchmark with multiple codec variants + encoding introspection.
//!
//! Usage:
//!   cargo run --release -p escapepod-vortex --example signal_bench -- \
//!       [--max-reads N] [--concat] <input.pod5> [more.pod5 ...]
//!
//! Codecs tested:
//!   pod5            : VBZ baseline (SVB16 + ZSTD)
//!   vortex-default  : BtrBlocks default cascade
//!   vortex-pco      : default + with_compact() (adds Pco scheme)
//!   vortex-delta    : in-process delta then default cascade
//!   vortex-delta+pco: in-process delta then default + Pco

use std::path::{Path, PathBuf};
use std::time::Instant;

use escapepod::Reader;
use escapepod_vortex::signal::{self, Codec};

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut max_reads: Option<usize> = None;
    let mut concat = false;
    if let Some(idx) = args.iter().position(|a| a == "--max-reads") {
        let n: usize = args.get(idx + 1).unwrap().parse()?;
        max_reads = Some(n);
        args.drain(idx..idx + 2);
    }
    if let Some(idx) = args.iter().position(|a| a == "--concat") {
        concat = true;
        args.remove(idx);
    }
    let inputs: Vec<PathBuf> = args.into_iter().map(PathBuf::from).collect();
    if inputs.is_empty() {
        anyhow::bail!("usage: signal_bench [--max-reads N] [--concat] <input.pod5> ...");
    }

    println!(
        "# settings: max_reads={:?}, concat_into_single_chunk={}",
        max_reads, concat
    );

    for input in &inputs {
        run_one(input, max_reads, concat).await?;
        println!();
    }
    Ok(())
}

async fn run_one(input: &Path, max_reads: Option<usize>, concat: bool) -> anyhow::Result<()> {
    let name = input
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("?")
        .to_string();
    println!("=== {} ===", name);

    // --- Experiment A: introspect what scheme each codec picks on a sample chunk
    {
        let reader = Reader::open(input)?;
        let first_read = reader
            .reads()?
            .next()
            .ok_or_else(|| anyhow::anyhow!("no reads"))??;
        let sample: Vec<i16> = reader.get_signal(&first_read.signal_rows)?;
        println!(
            "# scheme-pick on first read ({} samples, raw {} bytes):",
            sample.len(),
            sample.len() * 2
        );
        for codec in [
        Codec::Default,
        Codec::Pco,
        Codec::DeltaDefault,
        Codec::DeltaPco,
        Codec::DeltaSchemeOnly,
        Codec::DeltaSchemePco,
    ] {
            let (eid, sz) = signal::inspect_encoding(&sample, codec)?;
            println!(
                "    {:<18} {:>20}  bytes={}  ratio={:.2}x",
                codec.label(),
                eid,
                sz,
                (sample.len() * 2) as f64 / sz as f64
            );
        }
        println!();
    }

    // --- Sizes + decode throughput
    let pod5_size_full = std::fs::metadata(input)?.len();
    let total_samples;
    let pod5_size;
    let pod5_secs;
    if max_reads.is_some() {
        let tmp_pod5 = tempfile::Builder::new().suffix(".pod5").tempfile()?;
        let path = tmp_pod5.path().to_path_buf();
        drop(tmp_pod5);
        let (sz, samples) = repack_first_n(input, &path, max_reads.unwrap())?;
        pod5_size = sz;
        total_samples = samples;
        pod5_secs = bench_pod5_decode(&path, total_samples)?;
        let _ = std::fs::remove_file(&path);
    } else {
        pod5_size = pod5_size_full;
        total_samples = signal::pod5_total_samples(input)?;
        pod5_secs = bench_pod5_decode(input, total_samples)?;
    }
    let raw = total_samples * 2;

    println!(
        "{:<18} {:>14} {:>10} {:>10} {:>10} {:>14} {:>10}",
        "format", "bytes", "vs_raw", "vs_pod5", "decode_s", "MS/s", "samples"
    );
    println!("{}", "-".repeat(95));
    print_row("pod5", pod5_size, raw, pod5_size, pod5_secs, total_samples);

    for codec in [
        Codec::Default,
        Codec::Pco,
        Codec::DeltaDefault,
        Codec::DeltaPco,
    ] {
        let tmp = tempfile::Builder::new().suffix(".vortex").tempfile()?;
        let path = tmp.path().to_path_buf();
        drop(tmp);
        let (vx_size, _, _) =
            signal::convert_signal_only(input, &path, codec, max_reads, concat).await?;
        let needs_undelta = matches!(codec, Codec::DeltaDefault | Codec::DeltaPco);
        let vx_secs = bench_vortex_decode(&path, total_samples, needs_undelta).await?;
        print_row(codec.label(), vx_size, raw, pod5_size, vx_secs, total_samples);
        let _ = std::fs::remove_file(&path);
    }

    // ---- Random access (List<i16> layout)
    println!();
    println!("# Random access: 100 random reads, time to fetch + decode each");
    let n_reads = if let Some(n) = max_reads {
        n
    } else {
        signal::pod5_read_count(input)?
    };
    let n_random = std::cmp::min(100usize, n_reads);
    use rand::seq::SliceRandom;
    let mut rng = rand::rng();
    let mut indices: Vec<u64> = (0..n_reads as u64).collect();
    indices.shuffle(&mut rng);
    indices.truncate(n_random);

    // Resolve POD5 read UUIDs in row order so we can index them.
    let pod5_uuids = collect_pod5_uuids(input, max_reads)?;
    let chosen_uuids: Vec<_> = indices.iter().map(|&i| pod5_uuids[i as usize]).collect();

    let pod5_secs = bench_pod5_random(input, &chosen_uuids)?;
    println!(
        "{:<18} {:>12.4}  {:>10.2}  reads/s",
        "pod5",
        pod5_secs,
        n_random as f64 / pod5_secs
    );

    for codec in [Codec::Default, Codec::DeltaPco] {
        let tmp = tempfile::Builder::new().suffix(".vortex").tempfile()?;
        let path = tmp.path().to_path_buf();
        drop(tmp);
        let (sz, _, _) =
            signal::convert_signal_as_list(input, &path, codec, max_reads).await?;
        let needs_undelta = matches!(codec, Codec::DeltaDefault | Codec::DeltaPco);
        // 3 trials, take min
        let mut best = f64::INFINITY;
        for _ in 0..3 {
            let t0 = Instant::now();
            let _ = signal::random_access_list(&path, &indices, needs_undelta).await?;
            best = best.min(t0.elapsed().as_secs_f64());
        }
        println!(
            "{:<18} {:>12.4}  {:>10.2}  reads/s   (file {} bytes)",
            codec.label(),
            best,
            n_random as f64 / best,
            sz
        );
        let _ = std::fs::remove_file(&path);
    }

    Ok(())
}

fn collect_pod5_uuids(
    input: &Path,
    max_reads: Option<usize>,
) -> anyhow::Result<Vec<escapepod::Uuid>> {
    let reader = Reader::open(input)?;
    let mut out = Vec::new();
    for r in reader.reads()? {
        if let Some(n) = max_reads {
            if out.len() >= n {
                break;
            }
        }
        out.push(r?.read_id);
    }
    Ok(out)
}

fn bench_pod5_random(input: &Path, uuids: &[escapepod::Uuid]) -> anyhow::Result<f64> {
    use std::collections::HashSet;
    let mut best = f64::INFINITY;
    for _ in 0..3 {
        let reader = Reader::open(input)?;
        let t0 = Instant::now();
        let mut chk: i64 = 0;
        // Each random fetch is one HashSet lookup → real "find one read" cost.
        for &uuid in uuids {
            let mut targets = HashSet::new();
            targets.insert(uuid);
            let rows = reader.find_signal_rows_by_ids(&targets)?;
            for (_uuid, sig_rows) in &rows {
                let signal = reader.get_signal(sig_rows)?;
                for &v in signal.iter() {
                    chk = chk.wrapping_add(v as i64);
                }
            }
        }
        let e = t0.elapsed().as_secs_f64();
        std::hint::black_box(chk);
        best = best.min(e);
    }
    Ok(best)
}

fn print_row(fmt: &str, bytes: u64, raw: u64, pod5: u64, secs: f64, samples: u64) {
    let vs_raw = raw as f64 / bytes as f64;
    let vs_pod5 = bytes as f64 / pod5 as f64;
    let mss = (samples as f64) / secs / 1e6;
    println!(
        "{:<18} {:>14} {:>10.3} {:>10.3} {:>10.4} {:>14.1} {:>10}",
        fmt, bytes, vs_raw, vs_pod5, secs, mss, samples
    );
}

fn bench_pod5_decode(input: &Path, expected: u64) -> anyhow::Result<f64> {
    let mut best = f64::INFINITY;
    for _ in 0..3 {
        let reader = Reader::open(input)?;
        let t0 = Instant::now();
        let mut decoded: u64 = 0;
        let mut chk: i64 = 0;
        for r in reader.reads()? {
            let read = r?;
            let s = reader.get_signal(&read.signal_rows)?;
            decoded += s.len() as u64;
            for &v in s.iter() {
                chk = chk.wrapping_add(v as i64);
            }
        }
        let e = t0.elapsed().as_secs_f64();
        std::hint::black_box(chk);
        anyhow::ensure!(decoded == expected);
        best = best.min(e);
    }
    Ok(best)
}

async fn bench_vortex_decode(input: &Path, expected: u64, undelta: bool) -> anyhow::Result<f64> {
    let mut best = f64::INFINITY;
    for _ in 0..3 {
        let t0 = Instant::now();
        let total = signal::decode_all(input, undelta).await?;
        let e = t0.elapsed().as_secs_f64();
        anyhow::ensure!(total == expected);
        best = best.min(e);
    }
    Ok(best)
}

fn repack_first_n(input: &Path, output: &Path, n: usize) -> anyhow::Result<(u64, u64)> {
    use escapepod::{Writer, WriterOptions};

    let reader = Reader::open(input)?;
    let mut writer = Writer::create(output, WriterOptions::default())?;
    for ri in reader.run_infos() {
        writer.add_run_info(ri.clone())?;
    }
    let mut count = 0;
    let mut samples: u64 = 0;
    for r in reader.reads()? {
        if count >= n {
            break;
        }
        let read = r?;
        samples += read.num_samples;
        let comp = reader.get_compressed_signal_for_rows(&read.signal_rows)?;
        let new_read = read.for_writing_same_run();
        writer.add_read_with_compressed_signal(new_read, &comp)?;
        count += 1;
    }
    writer.finish()?;
    Ok((std::fs::metadata(output)?.len(), samples))
}
