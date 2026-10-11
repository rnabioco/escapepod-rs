//! #467 probe: time `Reader::reads()` and hash every field of every `ReadData`.
//!
//! `reads_probe <pod5> digest` prints an FNV-1a digest over all fields (bit-exact
//! floats) so two builds can be compared read for read; `reads_probe <pod5>
//! time <reps>` prints ns/read for the row iterator.
use escapepod_pod5::{ReadData, Reader};
use std::time::Instant;

fn fnv(h: &mut u64, bytes: &[u8]) {
    for b in bytes {
        *h ^= u64::from(*b);
        *h = h.wrapping_mul(0x100000001b3);
    }
}

fn digest(h: &mut u64, r: &ReadData) {
    fnv(h, r.read_id.as_bytes());
    fnv(h, &r.read_number.to_le_bytes());
    fnv(h, &r.start_sample.to_le_bytes());
    fnv(h, &r.channel.to_le_bytes());
    fnv(h, &[r.well]);
    fnv(h, r.pore_type.as_str().as_bytes());
    for f in [r.calibration_offset, r.calibration_scale, r.median_before] {
        fnv(h, &f.to_bits().to_le_bytes());
    }
    fnv(h, format!("{:?}", r.end_reason).as_bytes());
    fnv(h, &[r.end_reason_forced as u8]);
    fnv(h, &r.run_info_index.to_le_bytes());
    fnv(h, &r.num_minknow_events.to_le_bytes());
    for f in [
        r.tracked_scaling_scale,
        r.tracked_scaling_shift,
        r.predicted_scaling_scale,
        r.predicted_scaling_shift,
    ] {
        fnv(h, &f.to_bits().to_le_bytes());
    }
    fnv(h, &r.num_reads_since_mux_change.to_le_bytes());
    fnv(h, &r.time_since_mux_change.to_bits().to_le_bytes());
    fnv(h, &r.num_samples.to_le_bytes());
    for f in [
        r.open_pore_level,
        r.expected_open_pore_level,
        r.selected_read_level,
    ] {
        fnv(h, &f.to_bits().to_le_bytes());
    }
    for s in &r.signal_rows {
        fnv(h, &s.to_le_bytes());
    }
    fnv(h, &[0xff]);
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let reader = Reader::open(&a[1]).unwrap();
    match a[2].as_str() {
        "digest" => {
            let mut h = 0xcbf29ce484222325u64;
            let mut n = 0u64;
            for r in reader.reads().unwrap() {
                digest(&mut h, &r.unwrap());
                n += 1;
            }
            println!("{n} {h:016x}");
        }
        "collect" => {
            let reps: usize = a[3].parse().unwrap();
            for i in 0..reps {
                let t = Instant::now();
                let all = reader.collect_all_reads().unwrap();
                let e = t.elapsed();
                println!(
                    "rep {i} collect_all_reads {} reads {:.1} ns/read",
                    all.len(),
                    e.as_nanos() as f64 / all.len() as f64
                );
            }
        }
        "columns" => {
            let reps: usize = a[3].parse().unwrap();
            for i in 0..reps {
                let t = Instant::now();
                let c = reader.read_columns().unwrap();
                let e = t.elapsed();
                println!(
                    "rep {i} read_columns {} reads {:.1} ns/read",
                    c.len(),
                    e.as_nanos() as f64 / c.len() as f64
                );
            }
        }
        _ => {
            let reps: usize = a[3].parse().unwrap();
            for i in 0..reps {
                let t = Instant::now();
                let mut n = 0u64;
                let mut s = 0u64;
                for r in reader.reads().unwrap() {
                    s = s.wrapping_add(u64::from(r.unwrap().read_number));
                    n += 1;
                }
                let e = t.elapsed();
                println!(
                    "rep {i} reads() {n} reads {:.1} ns/read ({s})",
                    e.as_nanos() as f64 / n as f64
                );
            }
        }
    }
}
