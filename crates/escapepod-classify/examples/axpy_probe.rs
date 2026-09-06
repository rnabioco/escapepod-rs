//! Where the BiLSTM recurrence's time goes.
//!
//! The kernel's inner loop measured on synthetic weights sized so a pass's
//! working set sits in L1, the whole matrix in L2, or beyond it in L3, for
//! every shape the crate ships or considered: the single-read AVX2 loop,
//! the three-read AVX2 loop as written (accumulator array, `zip`) and with
//! named accumulators, AVX2 over 16-gate slices at 4–6 reads, AVX-512 over
//! 64-, 32- and 16-gate slices at 4–16 reads. Cycles per row that do not
//! move with the size are the loop; cycles that do are the memory system.
//! Compare shapes by cycles per read per *timestep* — cycles per row per
//! read × rows per step — since a narrower slice means more rows. Run on an
//! `rna` node:
//!
//! ```text
//! srun -p rna -c 4 pixi run cargo run --release --example axpy_probe -p escapepod-classify
//! ```
//!
//! What it found (2026-09-06, `rna`, 144 KB = the shipped `Rᵀ`): the loop
//! is L2→L1 bandwidth bound (5.0 cycles/row from L1, 11.2 from L2), array
//! and named accumulators are identical, and AVX-512 at eight reads over
//! 32-gate slices is the cheapest shape per read-step. The numbers are in
//! `benchmarks/README.md`.
#[cfg(target_arch = "x86_64")]
mod probe {
    use std::arch::x86_64::*;

    pub const H: usize = 96;
    /// Floats per row of one block: a 64-gate slice.
    pub const ROW: usize = 64;

    /// Single read: `acc[0..64] += h[j] * rows[j][0..64]`, eight named
    /// accumulators, memory-operand FMAs — `run_avx2`'s loop.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn single(rows: *const f32, h: &[f32], acc: &mut [f32; 64]) {
        unsafe {
            let p = acc.as_mut_ptr();
            let mut a0 = _mm256_loadu_ps(p);
            let mut a1 = _mm256_loadu_ps(p.add(8));
            let mut a2 = _mm256_loadu_ps(p.add(16));
            let mut a3 = _mm256_loadu_ps(p.add(24));
            let mut a4 = _mm256_loadu_ps(p.add(32));
            let mut a5 = _mm256_loadu_ps(p.add(40));
            let mut a6 = _mm256_loadu_ps(p.add(48));
            let mut a7 = _mm256_loadu_ps(p.add(56));
            let mut row = rows;
            for &hj in h {
                let hv = _mm256_set1_ps(hj);
                a0 = _mm256_fmadd_ps(hv, _mm256_loadu_ps(row), a0);
                a1 = _mm256_fmadd_ps(hv, _mm256_loadu_ps(row.add(8)), a1);
                a2 = _mm256_fmadd_ps(hv, _mm256_loadu_ps(row.add(16)), a2);
                a3 = _mm256_fmadd_ps(hv, _mm256_loadu_ps(row.add(24)), a3);
                a4 = _mm256_fmadd_ps(hv, _mm256_loadu_ps(row.add(32)), a4);
                a5 = _mm256_fmadd_ps(hv, _mm256_loadu_ps(row.add(40)), a5);
                a6 = _mm256_fmadd_ps(hv, _mm256_loadu_ps(row.add(48)), a6);
                a7 = _mm256_fmadd_ps(hv, _mm256_loadu_ps(row.add(56)), a7);
                row = row.add(ROW);
            }
            _mm256_storeu_ps(p, a0);
            _mm256_storeu_ps(p.add(8), a1);
            _mm256_storeu_ps(p.add(16), a2);
            _mm256_storeu_ps(p.add(24), a3);
            _mm256_storeu_ps(p.add(32), a4);
            _mm256_storeu_ps(p.add(40), a5);
            _mm256_storeu_ps(p.add(48), a6);
            _mm256_storeu_ps(p.add(56), a7);
        }
    }

    /// Three reads over a 32-gate slice, as `run_avx2_batch::<3>` is
    /// written: `acc[v][r]` array, one row load per `v` shared by the reads.
    /// `h` is `[3][H]`; `acc` is `[3][32]`. Reads the first 32 floats of
    /// each 64-float row.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn batch3_array(rows: *const f32, h: &[f32], acc: &mut [f32; 96]) {
        unsafe {
            let gp = acc.as_mut_ptr();
            let hc = h.as_ptr();
            let mut a = [[_mm256_setzero_ps(); 3]; 4];
            for (v, av) in a.iter_mut().enumerate() {
                for (r, x) in av.iter_mut().enumerate() {
                    *x = _mm256_loadu_ps(gp.add(r * 32 + v * 8));
                }
            }
            let mut row = rows;
            for j in 0..H {
                let mut hv = [_mm256_setzero_ps(); 3];
                for (r, x) in hv.iter_mut().enumerate() {
                    *x = _mm256_set1_ps(*hc.add(r * H + j));
                }
                for (v, av) in a.iter_mut().enumerate() {
                    let w = _mm256_loadu_ps(row.add(v * 8));
                    for (x, hr) in av.iter_mut().zip(hv.iter()) {
                        *x = _mm256_fmadd_ps(*hr, w, *x);
                    }
                }
                row = row.add(ROW);
            }
            for (v, av) in a.iter().enumerate() {
                for (r, x) in av.iter().enumerate() {
                    _mm256_storeu_ps(gp.add(r * 32 + v * 8), *x);
                }
            }
        }
    }

    /// The same, twelve named accumulators.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn batch3_named(rows: *const f32, h: &[f32], acc: &mut [f32; 96]) {
        unsafe {
            let gp = acc.as_mut_ptr();
            let (h0, h1, h2) = (h.as_ptr(), h.as_ptr().add(H), h.as_ptr().add(2 * H));
            let (p0, p1, p2) = (gp, gp.add(32), gp.add(64));
            let mut a00 = _mm256_loadu_ps(p0);
            let mut a10 = _mm256_loadu_ps(p0.add(8));
            let mut a20 = _mm256_loadu_ps(p0.add(16));
            let mut a30 = _mm256_loadu_ps(p0.add(24));
            let mut a01 = _mm256_loadu_ps(p1);
            let mut a11 = _mm256_loadu_ps(p1.add(8));
            let mut a21 = _mm256_loadu_ps(p1.add(16));
            let mut a31 = _mm256_loadu_ps(p1.add(24));
            let mut a02 = _mm256_loadu_ps(p2);
            let mut a12 = _mm256_loadu_ps(p2.add(8));
            let mut a22 = _mm256_loadu_ps(p2.add(16));
            let mut a32 = _mm256_loadu_ps(p2.add(24));
            let mut row = rows;
            for j in 0..H {
                let b0 = _mm256_set1_ps(*h0.add(j));
                let b1 = _mm256_set1_ps(*h1.add(j));
                let b2 = _mm256_set1_ps(*h2.add(j));
                let w0 = _mm256_loadu_ps(row);
                a00 = _mm256_fmadd_ps(b0, w0, a00);
                a01 = _mm256_fmadd_ps(b1, w0, a01);
                a02 = _mm256_fmadd_ps(b2, w0, a02);
                let w1 = _mm256_loadu_ps(row.add(8));
                a10 = _mm256_fmadd_ps(b0, w1, a10);
                a11 = _mm256_fmadd_ps(b1, w1, a11);
                a12 = _mm256_fmadd_ps(b2, w1, a12);
                let w2 = _mm256_loadu_ps(row.add(16));
                a20 = _mm256_fmadd_ps(b0, w2, a20);
                a21 = _mm256_fmadd_ps(b1, w2, a21);
                a22 = _mm256_fmadd_ps(b2, w2, a22);
                let w3 = _mm256_loadu_ps(row.add(24));
                a30 = _mm256_fmadd_ps(b0, w3, a30);
                a31 = _mm256_fmadd_ps(b1, w3, a31);
                a32 = _mm256_fmadd_ps(b2, w3, a32);
                row = row.add(ROW);
            }
            _mm256_storeu_ps(p0, a00);
            _mm256_storeu_ps(p0.add(8), a10);
            _mm256_storeu_ps(p0.add(16), a20);
            _mm256_storeu_ps(p0.add(24), a30);
            _mm256_storeu_ps(p1, a01);
            _mm256_storeu_ps(p1.add(8), a11);
            _mm256_storeu_ps(p1.add(16), a21);
            _mm256_storeu_ps(p1.add(24), a31);
            _mm256_storeu_ps(p2, a02);
            _mm256_storeu_ps(p2.add(8), a12);
            _mm256_storeu_ps(p2.add(16), a22);
            _mm256_storeu_ps(p2.add(24), a32);
        }
    }

    /// `N` reads over a 16-gate slice, eight lanes: two accumulators per
    /// read, so more reads share each row load. `h` is `[N][H]`; `acc` is
    /// `[N][16]`. Reads the first 16 floats of each row.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn batch_avx2_16<const N: usize>(rows: *const f32, h: &[f32], acc: &mut [f32]) {
        unsafe {
            let gp = acc.as_mut_ptr();
            let hc = h.as_ptr();
            let mut a = [[_mm256_setzero_ps(); N]; 2];
            for (v, av) in a.iter_mut().enumerate() {
                for (r, x) in av.iter_mut().enumerate() {
                    *x = _mm256_loadu_ps(gp.add(r * 16 + v * 8));
                }
            }
            let mut row = rows;
            for j in 0..H {
                let mut hv = [_mm256_setzero_ps(); N];
                for (r, x) in hv.iter_mut().enumerate() {
                    *x = _mm256_set1_ps(*hc.add(r * H + j));
                }
                let w = [_mm256_loadu_ps(row), _mm256_loadu_ps(row.add(8))];
                for (av, wv) in a.iter_mut().zip(w.iter()) {
                    for (x, hr) in av.iter_mut().zip(hv.iter()) {
                        *x = _mm256_fmadd_ps(*hr, *wv, *x);
                    }
                }
                row = row.add(ROW);
            }
            for (v, av) in a.iter().enumerate() {
                for (r, x) in av.iter().enumerate() {
                    _mm256_storeu_ps(gp.add(r * 16 + v * 8), *x);
                }
            }
        }
    }

    /// `N` reads over a 32-gate slice, sixteen lanes: two accumulators per
    /// read. `h` is `[N][H]`; `acc` is `[N][32]`.
    #[target_feature(enable = "avx512f")]
    pub unsafe fn batch_512_32<const N: usize>(rows: *const f32, h: &[f32], acc: &mut [f32]) {
        unsafe {
            let gp = acc.as_mut_ptr();
            let hc = h.as_ptr();
            let mut a = [[_mm512_setzero_ps(); N]; 2];
            for (v, av) in a.iter_mut().enumerate() {
                for (r, x) in av.iter_mut().enumerate() {
                    *x = _mm512_loadu_ps(gp.add(r * 32 + v * 16));
                }
            }
            let mut row = rows;
            for j in 0..H {
                let mut hv = [_mm512_setzero_ps(); N];
                for (r, x) in hv.iter_mut().enumerate() {
                    *x = _mm512_set1_ps(*hc.add(r * H + j));
                }
                let w = [_mm512_loadu_ps(row), _mm512_loadu_ps(row.add(16))];
                for (av, wv) in a.iter_mut().zip(w.iter()) {
                    for (x, hr) in av.iter_mut().zip(hv.iter()) {
                        *x = _mm512_fmadd_ps(*hr, *wv, *x);
                    }
                }
                row = row.add(ROW);
            }
            for (v, av) in a.iter().enumerate() {
                for (r, x) in av.iter().enumerate() {
                    _mm512_storeu_ps(gp.add(r * 32 + v * 16), *x);
                }
            }
        }
    }

    /// `N` reads over a 16-gate slice, sixteen lanes: one accumulator per
    /// read, one row load shared by all of them. `h` is `[N][H]`; `acc` is
    /// `[N][16]`.
    #[target_feature(enable = "avx512f")]
    pub unsafe fn batch_512_16<const N: usize>(rows: *const f32, h: &[f32], acc: &mut [f32]) {
        unsafe {
            let gp = acc.as_mut_ptr();
            let hc = h.as_ptr();
            let mut a = [_mm512_setzero_ps(); N];
            for (r, x) in a.iter_mut().enumerate() {
                *x = _mm512_loadu_ps(gp.add(r * 16));
            }
            let mut row = rows;
            for j in 0..H {
                let w = _mm512_loadu_ps(row);
                for (r, x) in a.iter_mut().enumerate() {
                    *x = _mm512_fmadd_ps(_mm512_set1_ps(*hc.add(r * H + j)), w, *x);
                }
                row = row.add(ROW);
            }
            for (r, x) in a.iter().enumerate() {
                _mm512_storeu_ps(gp.add(r * 16), *x);
            }
        }
    }

    /// Four reads over a 64-gate slice, sixteen lanes: the AVX-512 kernel's
    /// first cut, kept because it lost. `h` is `[4][H]`; `acc` `[4][64]`.
    #[target_feature(enable = "avx512f")]
    pub unsafe fn batch4_512_array(rows: *const f32, h: &[f32], acc: &mut [f32; 256]) {
        unsafe {
            let gp = acc.as_mut_ptr();
            let hc = h.as_ptr();
            let mut a = [[_mm512_setzero_ps(); 4]; 4];
            for (v, av) in a.iter_mut().enumerate() {
                for (r, x) in av.iter_mut().enumerate() {
                    *x = _mm512_loadu_ps(gp.add(r * 64 + v * 16));
                }
            }
            let mut row = rows;
            for j in 0..H {
                let mut hv = [_mm512_setzero_ps(); 4];
                for (r, x) in hv.iter_mut().enumerate() {
                    *x = _mm512_set1_ps(*hc.add(r * H + j));
                }
                let w = [
                    _mm512_loadu_ps(row),
                    _mm512_loadu_ps(row.add(16)),
                    _mm512_loadu_ps(row.add(32)),
                    _mm512_loadu_ps(row.add(48)),
                ];
                for (av, wv) in a.iter_mut().zip(w.iter()) {
                    for (x, hr) in av.iter_mut().zip(hv.iter()) {
                        *x = _mm512_fmadd_ps(*hr, *wv, *x);
                    }
                }
                row = row.add(ROW);
            }
            for (v, av) in a.iter().enumerate() {
                for (r, x) in av.iter().enumerate() {
                    _mm512_storeu_ps(gp.add(r * 64 + v * 16), *x);
                }
            }
        }
    }

    /// The same, sixteen named accumulators, memory-operand FMAs where the
    /// compiler will.
    #[target_feature(enable = "avx512f")]
    pub unsafe fn batch4_512_named(rows: *const f32, h: &[f32], acc: &mut [f32; 256]) {
        unsafe {
            let gp = acc.as_mut_ptr();
            let hc = h.as_ptr();
            macro_rules! ld {
                ($r:expr, $v:expr) => {
                    _mm512_loadu_ps(gp.add($r * 64 + $v * 16))
                };
            }
            let (mut a00, mut a10, mut a20, mut a30) = (ld!(0, 0), ld!(0, 1), ld!(0, 2), ld!(0, 3));
            let (mut a01, mut a11, mut a21, mut a31) = (ld!(1, 0), ld!(1, 1), ld!(1, 2), ld!(1, 3));
            let (mut a02, mut a12, mut a22, mut a32) = (ld!(2, 0), ld!(2, 1), ld!(2, 2), ld!(2, 3));
            let (mut a03, mut a13, mut a23, mut a33) = (ld!(3, 0), ld!(3, 1), ld!(3, 2), ld!(3, 3));
            let mut row = rows;
            for j in 0..H {
                let b0 = _mm512_set1_ps(*hc.add(j));
                let b1 = _mm512_set1_ps(*hc.add(H + j));
                let b2 = _mm512_set1_ps(*hc.add(2 * H + j));
                let b3 = _mm512_set1_ps(*hc.add(3 * H + j));
                let w0 = _mm512_loadu_ps(row);
                a00 = _mm512_fmadd_ps(b0, w0, a00);
                a01 = _mm512_fmadd_ps(b1, w0, a01);
                a02 = _mm512_fmadd_ps(b2, w0, a02);
                a03 = _mm512_fmadd_ps(b3, w0, a03);
                let w1 = _mm512_loadu_ps(row.add(16));
                a10 = _mm512_fmadd_ps(b0, w1, a10);
                a11 = _mm512_fmadd_ps(b1, w1, a11);
                a12 = _mm512_fmadd_ps(b2, w1, a12);
                a13 = _mm512_fmadd_ps(b3, w1, a13);
                let w2 = _mm512_loadu_ps(row.add(32));
                a20 = _mm512_fmadd_ps(b0, w2, a20);
                a21 = _mm512_fmadd_ps(b1, w2, a21);
                a22 = _mm512_fmadd_ps(b2, w2, a22);
                a23 = _mm512_fmadd_ps(b3, w2, a23);
                let w3 = _mm512_loadu_ps(row.add(48));
                a30 = _mm512_fmadd_ps(b0, w3, a30);
                a31 = _mm512_fmadd_ps(b1, w3, a31);
                a32 = _mm512_fmadd_ps(b2, w3, a32);
                a33 = _mm512_fmadd_ps(b3, w3, a33);
                row = row.add(ROW);
            }
            macro_rules! st {
                ($r:expr, $v:expr, $a:expr) => {
                    _mm512_storeu_ps(gp.add($r * 64 + $v * 16), $a)
                };
            }
            st!(0, 0, a00);
            st!(0, 1, a10);
            st!(0, 2, a20);
            st!(0, 3, a30);
            st!(1, 0, a01);
            st!(1, 1, a11);
            st!(1, 2, a21);
            st!(1, 3, a31);
            st!(2, 0, a02);
            st!(2, 1, a12);
            st!(2, 2, a22);
            st!(2, 3, a32);
            st!(3, 0, a03);
            st!(3, 1, a13);
            st!(3, 2, a23);
            st!(3, 3, a33);
        }
    }
}

#[cfg(target_arch = "x86_64")]
fn main() {
    use probe::*;
    use std::hint::black_box;
    use std::time::Instant;

    assert!(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma"));
    let avx512 = is_x86_feature_detected!("avx512f");
    let ghz: f64 = std::env::var("PROBE_GHZ")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2.9);
    println!(
        "h = {H}; one block = {} KB; cycles at {ghz} GHz; avx512f = {avx512}",
        H * ROW * 4 / 1024
    );
    println!(
        "{:<24} {:>6} {:>10} {:>10} {:>14}",
        "loop", "reads", "ns/row", "cyc/row", "cyc/row/read"
    );
    // 1 block = 24 KB (fits L1); 6 = 147 KB, the shipped Rᵀ (L2); 40 = 1 MB
    // (the L2); 240 = 5.9 MB (L3).
    let hv: Vec<f32> = (0..16 * H).map(|i| (i as f32 * 0.37).sin() * 0.1).collect();
    for &blocks in &[1usize, 6, 40, 240] {
        let w: Vec<f32> = (0..blocks * H * ROW)
            .map(|i| ((i * 7919) % 1000) as f32 * 1e-3 - 0.5)
            .collect();
        let passes = 40_000usize;
        let kb = blocks * H * ROW * 4 / 1024;
        println!("--- {blocks} blocks, {kb} KB");
        let report = |name: &str, reads: usize, f: &mut dyn FnMut(*const f32)| {
            for p in 0..passes / 10 {
                f(unsafe { w.as_ptr().add((p % blocks) * H * ROW) });
            }
            let t = Instant::now();
            for p in 0..passes {
                f(unsafe { w.as_ptr().add((p % blocks) * H * ROW) });
            }
            let ns_row = t.elapsed().as_nanos() as f64 / (passes * H) as f64;
            println!(
                "{name:<24} {reads:>6} {ns_row:>10.2} {:>10.1} {:>14.2}",
                ns_row * ghz,
                ns_row * ghz / reads as f64
            );
        };
        let mut acc1 = [0.0f32; 64];
        report("single avx2", 1, &mut |r| unsafe {
            single(r, black_box(&hv[..H]), &mut acc1)
        });
        let mut acc3 = [0.0f32; 96];
        report("batch3 avx2 array", 3, &mut |r| unsafe {
            batch3_array(r, black_box(&hv[..3 * H]), &mut acc3)
        });
        report("batch3 avx2 named", 3, &mut |r| unsafe {
            batch3_named(r, black_box(&hv[..3 * H]), &mut acc3)
        });
        let mut acc16 = vec![0.0f32; 16 * 8];
        report("batch4 avx2 16-gate", 4, &mut |r| unsafe {
            batch_avx2_16::<4>(r, black_box(&hv), &mut acc16)
        });
        report("batch5 avx2 16-gate", 5, &mut |r| unsafe {
            batch_avx2_16::<5>(r, black_box(&hv), &mut acc16)
        });
        report("batch6 avx2 16-gate", 6, &mut |r| unsafe {
            batch_avx2_16::<6>(r, black_box(&hv), &mut acc16)
        });
        if avx512 {
            let mut acc32 = vec![0.0f32; 32 * 8];
            report("batch4 avx512 32-gate", 4, &mut |r| unsafe {
                batch_512_32::<4>(r, black_box(&hv), &mut acc32)
            });
            report("batch6 avx512 32-gate", 6, &mut |r| unsafe {
                batch_512_32::<6>(r, black_box(&hv), &mut acc32)
            });
            report("batch8 avx512 32-gate", 8, &mut |r| unsafe {
                batch_512_32::<8>(r, black_box(&hv), &mut acc32)
            });
            black_box(&acc32);
            let mut acc16z = vec![0.0f32; 16 * 16];
            report("batch8 avx512 16-gate", 8, &mut |r| unsafe {
                batch_512_16::<8>(r, black_box(&hv), &mut acc16z)
            });
            report("batch12 avx512 16-gate", 12, &mut |r| unsafe {
                batch_512_16::<12>(r, black_box(&hv), &mut acc16z)
            });
            report("batch14 avx512 16-gate", 14, &mut |r| unsafe {
                batch_512_16::<14>(r, black_box(&hv), &mut acc16z)
            });
            report("batch16 avx512 16-gate", 16, &mut |r| unsafe {
                batch_512_16::<16>(r, black_box(&hv), &mut acc16z)
            });
            black_box(&acc16z);
            let mut acc4 = [0.0f32; 256];
            report("batch4 avx512 array", 4, &mut |r| unsafe {
                batch4_512_array(r, black_box(&hv[..4 * H]), &mut acc4)
            });
            report("batch4 avx512 named", 4, &mut |r| unsafe {
                batch4_512_named(r, black_box(&hv[..4 * H]), &mut acc4)
            });
            black_box(acc4);
        }
        black_box((acc1, acc3, &acc16));
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn main() {
    eprintln!("x86_64 only");
}
