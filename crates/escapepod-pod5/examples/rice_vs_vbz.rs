//! Prototype: fixed-predictor + partitioned adaptive Rice codec for POD5 signal,
//! benchmarked head-to-head against the shipping VBZ path (delta+zigzag → SVB16 → ZSTD).
//!
//! This is option 2 from the compression design discussion — "FLAC for squiggles":
//! a per-block FLAC-style fixed polynomial predictor (order 0–3) whose residuals are
//! Rice-coded with a per-partition adaptive `k`, and NO zstd stage at all. The point
//! is to measure whether pulling sub-byte entropy coding into the fast codec beats
//! VBZ on ratio *and* decode speed (VBZ decode is dominated by its zstd pass).
//!
//! Run:
//!   cargo run --release --example rice_vs_vbz -- [file1.pod5 ...] [--max-reads N]
//! Defaults to the in-repo dRNA fixture if no path is given.
//!
//! Every read is round-trip verified (rice decode == input) before timing, so a
//! wrong ratio can never be reported for a broken codec.

use escapepod_pod5::Reader;
use escapepod_pod5::compression::{svb16, vbz};
use std::time::Instant;

/// Same front end as VBZ (delta+zigzag+SVB16) but zstd at a higher level.
/// Zero-engineering archival lever: decode speed is identical to VBZ since
/// zstd decompress cost is level-independent. Size only.
fn svb16_zstd_size(samples: &[i16], level: i32) -> usize {
    let svb = svb16::encode(samples).unwrap();
    zstd::encode_all(svb.as_slice(), level).unwrap().len()
}

// ───────────────────────── Rice codec ─────────────────────────

const BLOCK: usize = 4096; // predictor-order decision granularity
const PART: usize = 256; // Rice-parameter decision granularity
const MAX_K: u32 = 24; // cap on Rice remainder bits (31 is the escape sentinel)
const ESCAPE: u32 = 31; // per-partition verbatim-32-bit escape

#[inline]
fn zigzag32(n: i32) -> u32 {
    ((n << 1) ^ (n >> 31)) as u32
}
#[inline]
fn unzigzag32(u: u32) -> i32 {
    ((u >> 1) as i32) ^ -((u & 1) as i32)
}

/// FLAC-style fixed predictor. Uses `min(order, i)` warmup so the first few
/// samples of the whole stream degrade gracefully to lower orders — no separate
/// verbatim warmup section needed. Encoder and decoder apply this identically.
#[inline]
fn predict(x: &[i32], i: usize, order: usize) -> i32 {
    match order.min(i) {
        0 => 0,
        1 => x[i - 1],
        2 => 2 * x[i - 1] - x[i - 2],
        _ => 3 * x[i - 1] - 3 * x[i - 2] + x[i - 3],
    }
}

// ---- bit I/O (MSB-first) ----

struct BitWriter {
    out: Vec<u8>,
    acc: u64,
    nbits: u32,
}
impl BitWriter {
    fn with_capacity(cap: usize) -> Self {
        Self {
            out: Vec::with_capacity(cap),
            acc: 0,
            nbits: 0,
        }
    }
    #[inline]
    fn put(&mut self, val: u64, n: u32) {
        if n == 0 {
            return;
        }
        self.acc = (self.acc << n) | (val & ((1u64 << n) - 1));
        self.nbits += n;
        while self.nbits >= 8 {
            self.nbits -= 8;
            self.out.push((self.acc >> self.nbits) as u8);
        }
        self.acc &= (1u64 << self.nbits) - 1;
    }
    #[inline]
    fn put_unary(&mut self, q: u32) {
        let mut r = q;
        while r >= 24 {
            self.put(0, 24);
            r -= 24;
        }
        self.put(1, r + 1); // r zero-bits then a terminating 1
    }
    fn finish(mut self) -> Vec<u8> {
        if self.nbits > 0 {
            self.out.push((self.acc << (8 - self.nbits)) as u8);
        }
        self.out
    }
}

struct BitReader<'a> {
    data: &'a [u8],
    byte: usize,
    acc: u64,
    nbits: u32,
}
impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            byte: 0,
            acc: 0,
            nbits: 0,
        }
    }
    #[inline]
    fn refill(&mut self) {
        while self.nbits <= 56 && self.byte < self.data.len() {
            self.acc = (self.acc << 8) | self.data[self.byte] as u64;
            self.byte += 1;
            self.nbits += 8;
        }
    }
    #[inline]
    fn get(&mut self, n: u32) -> u64 {
        if n == 0 {
            return 0;
        }
        if self.nbits < n {
            self.refill();
        }
        self.nbits -= n;
        (self.acc >> self.nbits) & ((1u64 << n) - 1)
    }
    #[inline]
    fn get_unary(&mut self) -> u32 {
        let mut q = 0u32;
        loop {
            if self.nbits == 0 {
                self.refill();
                if self.nbits == 0 {
                    break; // stream exhausted
                }
            }
            // Align the window's MSB (next bit to read) to bit 63, then count
            // leading zeros in one instruction instead of looping bit-by-bit.
            let v = self.acc << (64 - self.nbits);
            let lz = v.leading_zeros();
            if lz < self.nbits {
                q += lz;
                self.nbits -= lz + 1; // consume the zeros + the terminating 1
                break;
            }
            q += self.nbits; // whole window was zeros; refill and continue
            self.nbits = 0;
        }
        q
    }
}

/// Pick the fixed-predictor order (0–3) minimizing summed |residual| over a block.
fn best_order(x: &[i32], start: usize, end: usize) -> usize {
    let mut best = 0usize;
    let mut best_cost = i64::MAX;
    for order in 0..=3usize {
        let mut cost = 0i64;
        for i in start..end {
            let res = (x[i] - predict(x, i, order)) as i64;
            cost += res.unsigned_abs() as i64;
        }
        if cost < best_cost {
            best_cost = cost;
            best = order;
        }
    }
    best
}

/// Best Rice `k` for a partition of zigzagged residuals, plus its exact bit cost.
fn best_k(zz: &[u32]) -> (u32, u64) {
    let mut best_k = 0u32;
    let mut best_cost = u64::MAX;
    for k in 0..=MAX_K {
        // cost per value = (u >> k) unary zeros + 1 stop bit + k remainder bits
        let mut cost = 0u64;
        for &u in zz {
            cost += (u >> k) as u64 + 1 + k as u64;
        }
        if cost < best_cost {
            best_cost = cost;
            best_k = k;
        }
    }
    (best_k, best_cost)
}

fn rice_compress(samples: &[i16]) -> Vec<u8> {
    if samples.is_empty() {
        return Vec::new();
    }
    let x: Vec<i32> = samples.iter().map(|&s| s as i32).collect();
    let n = x.len();
    let mut w = BitWriter::with_capacity(n); // ~8 bits/sample ballpark
    let mut zz = vec![0u32; BLOCK];

    let mut i = 0;
    while i < n {
        let end = (i + BLOCK).min(n);
        let order = best_order(&x, i, end);
        w.put(order as u64, 2);

        // Precompute zigzag residuals for the whole block once.
        for j in i..end {
            zz[j - i] = zigzag32(x[j] - predict(&x, j, order));
        }

        let mut p = i;
        while p < end {
            let pend = (p + PART).min(end);
            let seg = &zz[p - i..pend - i];
            let (k, rice_cost) = best_k(seg);
            let esc_cost = 32 * seg.len() as u64;
            if esc_cost < rice_cost {
                w.put(ESCAPE as u64, 5);
                for &u in seg {
                    w.put(u as u64, 32);
                }
            } else {
                w.put(k as u64, 5);
                for &u in seg {
                    w.put_unary(u >> k);
                    w.put((u & ((1u32 << k) - 1)) as u64, k);
                }
            }
            p = pend;
        }
        i = end;
    }
    w.finish()
}

fn rice_decompress(data: &[u8], n: usize) -> Vec<i16> {
    if n == 0 {
        return Vec::new();
    }
    let mut r = BitReader::new(data);
    let mut x: Vec<i32> = Vec::with_capacity(n);

    let mut i = 0;
    while i < n {
        let end = (i + BLOCK).min(n);
        let order = r.get(2) as usize;
        let mut p = i;
        while p < end {
            let pend = (p + PART).min(end);
            let k = r.get(5) as u32;
            if k == ESCAPE {
                for _ in p..pend {
                    let u = r.get(32) as u32;
                    let pred = predict(&x, x.len(), order);
                    x.push(pred + unzigzag32(u));
                }
            } else {
                for _ in p..pend {
                    let q = r.get_unary();
                    let rem = r.get(k) as u32;
                    let u = (q << k) | rem;
                    let pred = predict(&x, x.len(), order);
                    x.push(pred + unzigzag32(u));
                }
            }
            p = pend;
        }
        i = end;
    }
    x.iter().map(|&v| v as i16).collect()
}

// ───────────────────────── harness ─────────────────────────

const DEFAULT_POD5: &str = "data/drna/yeast_trna_reads.pod5";

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut paths: Vec<String> = Vec::new();
    let mut max_reads = usize::MAX;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if a == "--max-reads" {
            max_reads = it.next().and_then(|s| s.parse().ok()).unwrap_or(usize::MAX);
        } else {
            paths.push(a.clone());
        }
    }
    if paths.is_empty() {
        paths.push(DEFAULT_POD5.to_string());
    }

    // Load all read signals into memory (this is I/O; excluded from codec timing).
    let mut reads: Vec<Vec<i16>> = Vec::new();
    for path in &paths {
        let reader = Reader::open(path).unwrap_or_else(|e| panic!("open {path}: {e}"));
        for rd in reader.reads().expect("reads()") {
            let rd = rd.expect("read");
            let sig = reader.get_signal(&rd.signal_rows).expect("signal");
            if !sig.is_empty() {
                reads.push(sig);
            }
            if reads.len() >= max_reads {
                break;
            }
        }
        if reads.len() >= max_reads {
            break;
        }
    }
    assert!(!reads.is_empty(), "no signal loaded");

    let total_samples: usize = reads.iter().map(|r| r.len()).sum();
    let raw_bytes = total_samples * 2;

    // Correctness: round-trip every read through the Rice codec.
    let mut rice_bytes = 0usize;
    for r in &reads {
        let enc = rice_compress(r);
        let dec = rice_decompress(&enc, r.len());
        assert!(dec == *r, "rice round-trip mismatch (len {})", r.len());
        rice_bytes += enc.len();
    }

    // VBZ size (deterministic).
    let mut vbz_bytes = 0usize;
    for r in &reads {
        vbz_bytes += vbz::compress_signal(r).expect("vbz encode").len();
    }

    // SVB16 + zstd at higher levels — archival baseline that rice must beat.
    let mut vbz9_bytes = 0usize;
    let mut vbz19_bytes = 0usize;
    for r in &reads {
        vbz9_bytes += svb16_zstd_size(r, 9);
        vbz19_bytes += svb16_zstd_size(r, 19);
    }

    // Our SVB16 front end WITHOUT zstd (isolates what the zstd pass buys).
    let mut ours_svb_bytes = 0usize;
    for r in &reads {
        ours_svb_bytes += svb16::encode(r).unwrap().len();
    }

    // Third-party `svb` crate (Psy-Fer): SIMD VBZ/SVB16 front end + exzd (patched
    // exceptions ~ PForDelta). Round-trip verified; both are zstd-free layers.
    let mut svb_vbz_bytes = 0usize;
    let mut svb_exzd_bytes = 0usize;
    for r in &reads {
        let e_vbz = svb::encode_vbz(r);
        assert!(
            svb::decode_vbz(&e_vbz, r.len()).unwrap() == *r,
            "svb vbz round-trip"
        );
        svb_vbz_bytes += e_vbz.len();
        let e_exzd = svb::encode_exzd(r);
        assert!(
            svb::decode_exzd(&e_exzd).unwrap() == *r,
            "svb exzd round-trip"
        );
        svb_exzd_bytes += e_exzd.len();
    }

    // Timing: best-of-N total wall time over all reads.
    const ROUNDS: usize = 5;
    let bench = |f: &dyn Fn()| {
        let mut best = f64::INFINITY;
        for _ in 0..ROUNDS {
            let t = Instant::now();
            f();
            best = best.min(t.elapsed().as_secs_f64());
        }
        best
    };

    // Pre-encode once for decode benches.
    let vbz_enc: Vec<Vec<u8>> = reads
        .iter()
        .map(|r| vbz::compress_signal(r).unwrap())
        .collect();
    let rice_enc: Vec<Vec<u8>> = reads.iter().map(|r| rice_compress(r)).collect();

    let vbz_enc_t = bench(&|| {
        for r in &reads {
            std::hint::black_box(vbz::compress_signal(r).unwrap());
        }
    });
    let rice_enc_t = bench(&|| {
        for r in &reads {
            std::hint::black_box(rice_compress(r));
        }
    });
    let vbz_dec_t = bench(&|| {
        for (e, r) in vbz_enc.iter().zip(&reads) {
            std::hint::black_box(vbz::decompress_signal(e, r.len()).unwrap());
        }
    });
    let rice_dec_t = bench(&|| {
        for (e, r) in rice_enc.iter().zip(&reads) {
            std::hint::black_box(rice_decompress(e, r.len()));
        }
    });

    // Our SVB16-only (no zstd) enc/dec.
    let ours_svb_enc: Vec<Vec<u8>> = reads.iter().map(|r| svb16::encode(r).unwrap()).collect();
    let ours_svb_enc_t = bench(&|| {
        for r in &reads {
            std::hint::black_box(svb16::encode(r).unwrap());
        }
    });
    let ours_svb_dec_t = bench(&|| {
        for (e, r) in ours_svb_enc.iter().zip(&reads) {
            std::hint::black_box(svb16::decode(e, r.len()).unwrap());
        }
    });

    // svb crate: vbz + exzd enc/dec.
    let svb_vbz_enc: Vec<Vec<u8>> = reads.iter().map(|r| svb::encode_vbz(r)).collect();
    let svb_exzd_enc: Vec<Vec<u8>> = reads.iter().map(|r| svb::encode_exzd(r)).collect();
    let svb_vbz_enc_t = bench(&|| {
        for r in &reads {
            std::hint::black_box(svb::encode_vbz(r));
        }
    });
    let svb_vbz_dec_t = bench(&|| {
        for (e, r) in svb_vbz_enc.iter().zip(&reads) {
            std::hint::black_box(svb::decode_vbz(e, r.len()).unwrap());
        }
    });
    let svb_exzd_enc_t = bench(&|| {
        for r in &reads {
            std::hint::black_box(svb::encode_exzd(r));
        }
    });
    let svb_exzd_dec_t = bench(&|| {
        for e in svb_exzd_enc.iter() {
            std::hint::black_box(svb::decode_exzd(e).unwrap());
        }
    });

    let mbps = |t: f64| raw_bytes as f64 / 1e6 / t;
    let ratio = |b: usize| raw_bytes as f64 / b as f64;
    let bps = |b: usize| b as f64 * 8.0 / total_samples as f64;

    println!("\n=== rice (fixed-predictor + adaptive Rice) vs VBZ ===");
    println!("files      : {}", paths.join(", "));
    println!("reads      : {}", reads.len());
    println!(
        "samples    : {total_samples}  (raw {:.2} MB)",
        raw_bytes as f64 / 1e6
    );
    println!();
    println!(
        "{:<10} {:>10} {:>8} {:>10} {:>12} {:>12}",
        "codec", "bytes", "ratio", "bits/samp", "enc MB/s", "dec MB/s"
    );
    println!(
        "{:<10} {:>10} {:>8.3} {:>10.3} {:>12.1} {:>12.1}",
        "VBZ(z1)",
        vbz_bytes,
        ratio(vbz_bytes),
        bps(vbz_bytes),
        mbps(vbz_enc_t),
        mbps(vbz_dec_t)
    );
    println!(
        "{:<10} {:>10} {:>8.3} {:>10.3} {:>12} {:>12}",
        "+z9",
        vbz9_bytes,
        ratio(vbz9_bytes),
        bps(vbz9_bytes),
        "~vbz",
        "~vbz"
    );
    println!(
        "{:<10} {:>10} {:>8.3} {:>10.3} {:>12} {:>12}",
        "+z19",
        vbz19_bytes,
        ratio(vbz19_bytes),
        bps(vbz19_bytes),
        "~vbz",
        "~vbz"
    );
    println!(
        "{:<10} {:>10} {:>8.3} {:>10.3} {:>12.1} {:>12.1}",
        "svb16(ours)",
        ours_svb_bytes,
        ratio(ours_svb_bytes),
        bps(ours_svb_bytes),
        mbps(ours_svb_enc_t),
        mbps(ours_svb_dec_t)
    );
    println!(
        "{:<10} {:>10} {:>8.3} {:>10.3} {:>12.1} {:>12.1}",
        "svb::vbz",
        svb_vbz_bytes,
        ratio(svb_vbz_bytes),
        bps(svb_vbz_bytes),
        mbps(svb_vbz_enc_t),
        mbps(svb_vbz_dec_t)
    );
    println!(
        "{:<10} {:>10} {:>8.3} {:>10.3} {:>12.1} {:>12.1}",
        "svb::exzd",
        svb_exzd_bytes,
        ratio(svb_exzd_bytes),
        bps(svb_exzd_bytes),
        mbps(svb_exzd_enc_t),
        mbps(svb_exzd_dec_t)
    );
    println!(
        "{:<10} {:>10} {:>8.3} {:>10.3} {:>12.1} {:>12.1}",
        "rice",
        rice_bytes,
        ratio(rice_bytes),
        bps(rice_bytes),
        mbps(rice_enc_t),
        mbps(rice_dec_t)
    );
    println!();
    println!(
        "svb::vbz vs our svb16(no-zstd): size {:+.1}%   enc {:.2}x   dec {:.2}x",
        (svb_vbz_bytes as f64 / ours_svb_bytes as f64 - 1.0) * 100.0,
        ours_svb_enc_t / svb_vbz_enc_t,
        ours_svb_dec_t / svb_vbz_dec_t
    );
    println!(
        "svb::exzd vs VBZ(z1): size {:+.1}%   enc {:.2}x   dec {:.2}x",
        (svb_exzd_bytes as f64 / vbz_bytes as f64 - 1.0) * 100.0,
        vbz_enc_t / svb_exzd_enc_t,
        vbz_dec_t / svb_exzd_dec_t
    );
    println!(
        "rice vs +z19 size: {:+.1}%   (decode: +z19 ~= VBZ speed, rice {:.2}x VBZ)",
        (rice_bytes as f64 / vbz19_bytes as f64 - 1.0) * 100.0,
        rice_dec_t / vbz_dec_t
    );
    println!(
        "rice vs VBZ: size {:+.1}%   enc {:.2}x   dec {:.2}x",
        (rice_bytes as f64 / vbz_bytes as f64 - 1.0) * 100.0,
        vbz_enc_t / rice_enc_t,
        vbz_dec_t / rice_dec_t
    );
}
