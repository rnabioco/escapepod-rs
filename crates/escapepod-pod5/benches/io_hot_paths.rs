//! Microbenchmarks for POD5 format I/O hot paths.
//!
//! Covers the paths that dominate real-world throughput but were previously
//! unmeasured:
//! - `Reader::open` (mmap + footer parse)
//! - Read-table iteration (no signal decompression)
//! - Signal extraction (VBZ decompress per read)
//! - Parallel signal extraction via `SignalExtractor` + rayon
//! - `Writer` end-to-end (add_read + VBZ compress + finalize)
//! - `merge_files` (two-file zero-copy merge)
//! - `filter_files_with_criteria` (UUID fast path vs. sample-range slow path)
//! - `repack_files` (block-level signal copy)
//!
//! Benches build a synthetic fixture once (LazyLock) into a tempdir, so CI
//! can run them without relying on the gitignored `data/drna/` fixture.
//!
//! Run with:
//!   cargo bench -p escapepod-pod5 --bench io_hot_paths
//!   cargo bench -p escapepod-pod5 --bench io_hot_paths -- reader
//!
//! Env vars:
//!   ESCAPEPOD_BENCH_READS=N       Reads in the fixture file (default 200).
//!   ESCAPEPOD_BENCH_SAMPLES=N     criterion sample_size override for slow
//!                                 groups (merge/filter/repack). Default 30.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use tempfile::TempDir;

use escapepod_pod5::operations::{
    FilterCriteria, FilterOptions, RepackOptions, filter_files_with_criteria, repack_files,
};
use escapepod_pod5::{
    EndReason, MergeOptions, ReadData, Reader, RunInfoData, Uuid, Writer, WriterOptions,
    merge_files,
};

const DEFAULT_SAMPLES_PER_READ: usize = 4_000;

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn fixture_read_count() -> usize {
    env_usize("ESCAPEPOD_BENCH_READS", 200)
}

fn bench_sample_size(default: usize) -> usize {
    env_usize("ESCAPEPOD_BENCH_SAMPLES", default)
}

/// Build a nanopore-like i16 signal: slow drift + per-sample noise.
fn pseudo_signal(n: usize, seed: u64) -> Vec<i16> {
    let mut state = seed | 1;
    let mut base: i32 = 500;
    (0..n)
        .map(|i| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let noise = ((state as i32) >> 24) % 15;
            if i % 200 == 0 {
                base = 400 + ((i as i32 / 200) % 4) * 50;
            }
            (base + noise).clamp(i16::MIN as i32, i16::MAX as i32) as i16
        })
        .collect()
}

fn sample_run_info(acq_id: &str) -> RunInfoData {
    use std::collections::HashMap;
    RunInfoData {
        acquisition_id: acq_id.to_string(),
        acquisition_start_time: 1_609_459_200_000,
        adc_max: 2047,
        adc_min: -2048,
        context_tags: HashMap::from([("experiment_type".to_string(), "genomic_dna".to_string())]),
        experiment_name: "bench".to_string(),
        flow_cell_id: "FAK00000".to_string(),
        flow_cell_product_code: "FLO-MIN106".to_string(),
        protocol_name: "bench_protocol".to_string(),
        protocol_run_id: "protocol_bench".to_string(),
        protocol_start_time: 1_609_459_200_000,
        sample_id: "bench_sample".to_string(),
        sample_rate: 4_000,
        sequencing_kit: "SQK-LSK109".to_string(),
        sequencer_position: "MN00000".to_string(),
        sequencer_position_type: "minion".to_string(),
        software: "escapepod-bench".to_string(),
        system_name: "bench_system".to_string(),
        system_type: "minion".to_string(),
        tracking_id: HashMap::new(),
    }
}

fn sample_read(run_info_idx: u32, read_number: u32, num_samples: u64) -> ReadData {
    ReadData {
        read_id: Uuid::new_v4(),
        read_number,
        start_sample: (read_number as u64 - 1) * num_samples,
        channel: 1,
        well: 1,
        pore_type: "not_set".into(),
        calibration_offset: 0.5,
        calibration_scale: 0.95,
        median_before: 200.0,
        end_reason: EndReason::SignalPositive,
        end_reason_forced: false,
        run_info_index: run_info_idx,
        num_minknow_events: 100,
        tracked_scaling_scale: 1.0,
        tracked_scaling_shift: 0.0,
        predicted_scaling_scale: 1.0,
        predicted_scaling_shift: 0.0,
        num_reads_since_mux_change: 0,
        time_since_mux_change: 0.0,
        num_samples,
        open_pore_level: 220.0,
        expected_open_pore_level: 0.0,
        selected_read_level: 0.0,
        signal_rows: Vec::new(),
    }
}

/// Write a POD5 file with `n_reads` reads, each with `samples_per_read` samples.
fn build_fixture_at(
    path: &Path,
    n_reads: usize,
    samples_per_read: usize,
    acq_id: &str,
) -> Vec<Uuid> {
    let mut writer =
        Writer::create(path, WriterOptions::default()).expect("failed to create writer");
    let run_idx = writer
        .add_run_info(sample_run_info(acq_id))
        .expect("failed to add run info");

    let mut ids = Vec::with_capacity(n_reads);
    for i in 0..n_reads {
        let read = sample_read(run_idx, i as u32 + 1, samples_per_read as u64);
        ids.push(read.read_id);
        let signal = pseudo_signal(samples_per_read, 0x5EED_0000 + i as u64);
        writer.add_read(read, &signal).expect("failed to add read");
    }
    writer.finish().expect("failed to finalize writer");
    ids
}

/// A shared on-disk fixture. Built once per process, lives for the whole run.
struct Fixture {
    _tmp: TempDir,
    path: PathBuf,
    /// Second file with disjoint read IDs (for merge/filter multi-input tests).
    path_b: PathBuf,
    read_ids: Vec<Uuid>,
    read_ids_b: Vec<Uuid>,
    samples_per_read: usize,
}

static FIXTURE: LazyLock<Fixture> = LazyLock::new(|| {
    let tmp = TempDir::new().expect("failed to create tempdir");
    let path = tmp.path().join("bench_a.pod5");
    let path_b = tmp.path().join("bench_b.pod5");
    let n = fixture_read_count();
    let read_ids = build_fixture_at(&path, n, DEFAULT_SAMPLES_PER_READ, "bench_acq_a");
    let read_ids_b = build_fixture_at(&path_b, n, DEFAULT_SAMPLES_PER_READ, "bench_acq_b");
    Fixture {
        _tmp: tmp,
        path,
        path_b,
        read_ids,
        read_ids_b,
        samples_per_read: DEFAULT_SAMPLES_PER_READ,
    }
});

fn bench_reader_open(c: &mut Criterion) {
    let mut group = c.benchmark_group("reader_open");
    let fixture = &*FIXTURE;
    group.bench_function("open_mmap_and_footer", |bench| {
        bench.iter(|| {
            let reader = Reader::open(black_box(&fixture.path)).expect("open failed");
            black_box(reader.run_info_count());
        });
    });
    group.finish();
}

fn bench_reader_iteration(c: &mut Criterion) {
    let mut group = c.benchmark_group("reader_iterate_reads");
    let fixture = &*FIXTURE;
    group.throughput(Throughput::Elements(fixture.read_ids.len() as u64));
    group.bench_function("metadata_only", |bench| {
        bench.iter(|| {
            let reader = Reader::open(&fixture.path).unwrap();
            let count = reader
                .reads()
                .unwrap()
                .map(|r| r.map(|_| ()))
                .filter(Result::is_ok)
                .count();
            black_box(count);
        });
    });
    group.finish();
}

fn bench_reader_signal_extraction(c: &mut Criterion) {
    let mut group = c.benchmark_group("reader_extract_signal");
    let fixture = &*FIXTURE;
    let total_bytes = fixture.read_ids.len() * fixture.samples_per_read * 2;
    group.throughput(Throughput::Bytes(total_bytes as u64));

    group.bench_function("sequential", |bench| {
        bench.iter(|| {
            let reader = Reader::open(&fixture.path).unwrap();
            let mut total_samples = 0usize;
            for read_result in reader.reads().unwrap() {
                let read = read_result.unwrap();
                let signal = reader.get_signal(&read.signal_rows).unwrap();
                total_samples += signal.len();
            }
            black_box(total_samples);
        });
    });

    group.bench_function("parallel_extractor", |bench| {
        use rayon::prelude::*;
        bench.iter(|| {
            let reader = Reader::open(&fixture.path).unwrap();
            let reads: Vec<_> = reader
                .reads()
                .unwrap()
                .collect::<Result<Vec<_>, _>>()
                .unwrap();
            let extractor = reader.signal_extractor().unwrap();
            let total: usize = reads
                .par_iter()
                .map(|r| extractor.get_signal(&r.signal_rows).unwrap().len())
                .sum();
            black_box(total);
        });
    });

    group.finish();
}

fn bench_writer(c: &mut Criterion) {
    let mut group = c.benchmark_group("writer_end_to_end");
    group.sample_size(bench_sample_size(30));
    let n_reads = fixture_read_count();
    group.throughput(Throughput::Bytes(
        (n_reads * DEFAULT_SAMPLES_PER_READ * 2) as u64,
    ));
    group.bench_with_input(BenchmarkId::new("reads", n_reads), &n_reads, |bench, &n| {
        bench.iter_batched(
            || TempDir::new().expect("tempdir"),
            |tmp| {
                let path = tmp.path().join("out.pod5");
                build_fixture_at(&path, n, DEFAULT_SAMPLES_PER_READ, "bench_writer");
                tmp
            },
            criterion::BatchSize::SmallInput,
        );
    });
    group.finish();
}

fn bench_merge(c: &mut Criterion) {
    let mut group = c.benchmark_group("merge_two_files");
    group.sample_size(bench_sample_size(30));
    let fixture = &*FIXTURE;
    let inputs = vec![fixture.path.clone(), fixture.path_b.clone()];
    let total_reads = (fixture.read_ids.len() + fixture.read_ids_b.len()) as u64;
    group.throughput(Throughput::Elements(total_reads));
    group.bench_function("zero_copy", |bench| {
        bench.iter_batched(
            || TempDir::new().expect("tempdir"),
            |tmp| {
                let out = tmp.path().join("merged.pod5");
                merge_files(&inputs, &out, &MergeOptions::default(), None).unwrap();
                tmp
            },
            criterion::BatchSize::SmallInput,
        );
    });
    group.finish();
}

fn bench_filter(c: &mut Criterion) {
    let mut group = c.benchmark_group("filter");
    group.sample_size(bench_sample_size(30));
    let fixture = &*FIXTURE;
    let inputs = vec![fixture.path.clone()];

    // Fast path: UUID-only filter (half of the reads).
    let half: HashSet<Uuid> = fixture
        .read_ids
        .iter()
        .take(fixture.read_ids.len() / 2)
        .copied()
        .collect();
    let uuid_criteria = FilterCriteria {
        read_ids: Some(half),
        ..Default::default()
    };
    group.bench_function("by_uuid", |bench| {
        bench.iter_batched(
            || TempDir::new().expect("tempdir"),
            |tmp| {
                let out = tmp.path().join("filtered.pod5");
                filter_files_with_criteria(
                    &inputs,
                    &out,
                    &uuid_criteria,
                    FilterOptions::default(),
                    None,
                )
                .unwrap();
                tmp
            },
            criterion::BatchSize::SmallInput,
        );
    });

    // Slow path: sample-count range triggers full read-table deserialization.
    let range_criteria = FilterCriteria {
        min_samples: Some(1),
        max_samples: Some(u64::MAX),
        ..Default::default()
    };
    group.bench_function("by_sample_range", |bench| {
        bench.iter_batched(
            || TempDir::new().expect("tempdir"),
            |tmp| {
                let out = tmp.path().join("filtered.pod5");
                filter_files_with_criteria(
                    &inputs,
                    &out,
                    &range_criteria,
                    FilterOptions::default(),
                    None,
                )
                .unwrap();
                tmp
            },
            criterion::BatchSize::SmallInput,
        );
    });
    group.finish();
}

fn bench_repack(c: &mut Criterion) {
    let mut group = c.benchmark_group("repack");
    group.sample_size(bench_sample_size(30));
    let fixture = &*FIXTURE;
    group.bench_function("single_file_block_copy", |bench| {
        bench.iter_batched(
            || {
                let tmp = TempDir::new().expect("tempdir");
                let out = tmp.path().join("repacked.pod5");
                (tmp, out)
            },
            |(tmp, out)| {
                let pairs = vec![(fixture.path.clone(), out)];
                let opts = RepackOptions {
                    force: true,
                    ..Default::default()
                };
                let _result = repack_files(&pairs, opts, None);
                tmp
            },
            criterion::BatchSize::SmallInput,
        );
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_reader_open,
    bench_reader_iteration,
    bench_reader_signal_extraction,
    bench_writer,
    bench_merge,
    bench_filter,
    bench_repack,
);
criterion_main!(benches);
