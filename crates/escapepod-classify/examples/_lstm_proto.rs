//! Prototype: a hand-written BiLSTM against tract's `Scan`.
//!
//! ONNX LSTM semantics, opset 17, `direction=bidirectional`, no peepholes:
//! `W [D, 4H, IN]`, `R [D, 4H, H]`, `B [D, 8H]`, gate order **i, o, f, c**.
//!
//!   i = σ(Wi·x + Ri·h + Wbi + Rbi)      f = σ(Wf·x + Rf·h + Wbf + Rbf)
//!   c̃ = tanh(Wc·x + Rc·h + Wbc + Rbc)   o = σ(Wo·x + Ro·h + Wbo + Rbo)
//!   C = f⊙C + i⊙c̃                       H = o⊙tanh(C)
use std::time::Instant;
use tract_onnx::prelude::*;

const SEQ: usize = 33;
const IN: usize = 4;
const H: usize = 96;
const G: usize = 4 * H; // 384 gate rows
const D: usize = 2;

fn read_f32(p: &str) -> Vec<f32> {
    std::fs::read(p)
        .unwrap()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Weights repacked once into the layout the step loop wants.
struct Lstm {
    /// Per direction: input contribution for every timestep, `[SEQ][G]`,
    /// with both bias halves already folded in. One matmul, done up front —
    /// it does not depend on the recurrence.
    xw: Vec<Vec<f32>>,
    /// Per direction: `R` transposed to `[H][G]` so the step is an axpy over
    /// the previous hidden state — sequential in memory, no horizontal sums.
    rt: Vec<Vec<f32>>,
}

impl Lstm {
    fn new(w: &[f32], r: &[f32], b: &[f32], x: &[f32]) -> Self {
        let (mut xw, mut rt) = (Vec::new(), Vec::new());
        for d in 0..D {
            let mut xwd = vec![0.0f32; SEQ * G];
            for t in 0..SEQ {
                for g in 0..G {
                    let mut acc = b[d * 8 * H + g] + b[d * 8 * H + G + g];
                    for k in 0..IN {
                        acc += x[t * IN + k] * w[(d * G + g) * IN + k];
                    }
                    xwd[t * G + g] = acc;
                }
            }
            xw.push(xwd);
            let mut rtd = vec![0.0f32; H * G];
            for g in 0..G {
                for j in 0..H {
                    rtd[j * G + g] = r[(d * G + g) * H + j];
                }
            }
            rt.push(rtd);
        }
        Self { xw, rt }
    }
}

#[inline(always)]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Scalar reference. `out` receives the final hidden state of each direction.
fn scalar(l: &Lstm, gates: &mut [f32], out: &mut [f32]) {
    for d in 0..D {
        let (mut h, mut c) = ([0.0f32; H], [0.0f32; H]);
        for step in 0..SEQ {
            // Direction 1 walks the sequence backwards.
            let t = if d == 0 { step } else { SEQ - 1 - step };
            gates.copy_from_slice(&l.xw[d][t * G..(t + 1) * G]);
            for j in 0..H {
                let hj = h[j];
                if hj == 0.0 {
                    continue;
                }
                let row = &l.rt[d][j * G..(j + 1) * G];
                for g in 0..G {
                    gates[g] += hj * row[g];
                }
            }
            for u in 0..H {
                let i = sigmoid(gates[u]);
                let o = sigmoid(gates[H + u]);
                let f = sigmoid(gates[2 * H + u]);
                let ct = gates[3 * H + u].tanh();
                c[u] = f * c[u] + i * ct;
                h[u] = o * c[u].tanh();
            }
        }
        out[d * H..(d + 1) * H].copy_from_slice(&h);
    }
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::*;
    use std::arch::x86_64::*;

    /// Cephes-style `exp`, the same construction `crf/avx2.rs` uses.
    #[inline(always)]
    unsafe fn exp8(x: __m256) -> __m256 {
        unsafe {
            let x = _mm256_min_ps(_mm256_set1_ps(88.376_26), x);
            let x = _mm256_max_ps(_mm256_set1_ps(-88.376_26), x);
            let fx = _mm256_fmadd_ps(x, _mm256_set1_ps(1.442_695_f32), _mm256_set1_ps(0.5));
            let fx = _mm256_floor_ps(fx);
            let r = _mm256_fnmadd_ps(fx, _mm256_set1_ps(0.693_359_4), x);
            let r = _mm256_fnmadd_ps(fx, _mm256_set1_ps(-2.121_944_4e-4), r);
            let r2 = _mm256_mul_ps(r, r);
            let mut y = _mm256_set1_ps(1.987_569_1e-4);
            y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(1.398_199_9e-3));
            y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(8.333_452e-3));
            y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(4.166_579_6e-2));
            y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(1.666_666_6e-1));
            y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(5e-1));
            y = _mm256_fmadd_ps(y, r2, r);
            y = _mm256_add_ps(y, _mm256_set1_ps(1.0));
            // 2^fx by assembling the exponent field directly.
            let imm = _mm256_cvtps_epi32(fx);
            let pow2 = _mm256_castsi256_ps(_mm256_slli_epi32(
                _mm256_add_epi32(imm, _mm256_set1_epi32(0x7f)),
                23,
            ));
            _mm256_mul_ps(y, pow2)
        }
    }

    #[inline(always)]
    unsafe fn sigmoid8(x: __m256) -> __m256 {
        unsafe {
            let e = exp8(_mm256_sub_ps(_mm256_setzero_ps(), x));
            _mm256_div_ps(
                _mm256_set1_ps(1.0),
                _mm256_add_ps(_mm256_set1_ps(1.0), e),
            )
        }
    }

    /// `tanh(x) = 2σ(2x) − 1`, so it costs one `exp` like the sigmoid.
    #[inline(always)]
    unsafe fn tanh8(x: __m256) -> __m256 {
        unsafe {
            let s = sigmoid8(_mm256_add_ps(x, x));
            _mm256_fmsub_ps(_mm256_set1_ps(2.0), s, _mm256_set1_ps(1.0))
        }
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn run(l: &Lstm, gates: &mut [f32], out: &mut [f32]) {
        unsafe {
            for d in 0..D {
                let (mut h, mut c) = ([0.0f32; H], [0.0f32; H]);
                for step in 0..SEQ {
                    let t = if d == 0 { step } else { SEQ - 1 - step };
                    gates.copy_from_slice(&l.xw[d][t * G..(t + 1) * G]);

                    // gates += h[j] * Rᵀ[j][..] — an axpy per hidden unit, so
                    // every load is sequential and nothing is horizontally
                    // reduced. Accumulators stay in registers a chunk at a
                    // time; G = 384 is 48 vectors, far more than 16 YMM.
                    const CH: usize = 64; // 8 vectors held live
                    for chunk in 0..G / CH {
                        let base = chunk * CH;
                        let mut acc = [_mm256_setzero_ps(); CH / 8];
                        for (v, a) in acc.iter_mut().enumerate() {
                            *a = _mm256_loadu_ps(gates.as_ptr().add(base + v * 8));
                        }
                        for j in 0..H {
                            let hj = _mm256_set1_ps(h[j]);
                            let row = l.rt[d].as_ptr().add(j * G + base);
                            for (v, a) in acc.iter_mut().enumerate() {
                                *a = _mm256_fmadd_ps(hj, _mm256_loadu_ps(row.add(v * 8)), *a);
                            }
                        }
                        for (v, a) in acc.iter().enumerate() {
                            _mm256_storeu_ps(gates.as_mut_ptr().add(base + v * 8), *a);
                        }
                    }

                    for u in (0..H).step_by(8) {
                        let gi = sigmoid8(_mm256_loadu_ps(gates.as_ptr().add(u)));
                        let go = sigmoid8(_mm256_loadu_ps(gates.as_ptr().add(H + u)));
                        let gf = sigmoid8(_mm256_loadu_ps(gates.as_ptr().add(2 * H + u)));
                        let gc = tanh8(_mm256_loadu_ps(gates.as_ptr().add(3 * H + u)));
                        let cprev = _mm256_loadu_ps(c.as_ptr().add(u));
                        let cn = _mm256_fmadd_ps(gf, cprev, _mm256_mul_ps(gi, gc));
                        _mm256_storeu_ps(c.as_mut_ptr().add(u), cn);
                        _mm256_storeu_ps(h.as_mut_ptr().add(u), _mm256_mul_ps(go, tanh8(cn)));
                    }
                }
                out[d * H..(d + 1) * H].copy_from_slice(&h);
            }
        }
    }
}

fn main() -> TractResult<()> {
    let dir = std::env::args().nth(1).unwrap();
    let (w, r, b, x) = (
        read_f32(&format!("{dir}/W.f32")),
        read_f32(&format!("{dir}/R.f32")),
        read_f32(&format!("{dir}/B.f32")),
        read_f32(&format!("{dir}/X.f32")),
    );
    let l = Lstm::new(&w, &r, &b, &x);

    // tract, for both the reference values and the baseline timing.
    let plan = tract_onnx::onnx()
        .model_for_path(format!("{dir}/lstm.onnx"))?
        .with_input_fact(0, f32::fact([SEQ, 1, IN]).into())?
        .into_optimized()?
        .into_runnable()?;
    let xt = Tensor::from_shape(&[SEQ, 1, IN], &x)?;
    let y = plan.run(tvec!(xt.clone().into()))?;
    let yv = y[0].to_plain_array_view::<f32>()?;
    let flat = yv.as_slice().unwrap();
    // Y is [SEQ, D, 1, H]; the final state is t = SEQ-1 forward, t = 0 reverse.
    let at = |t: usize, d: usize, u: usize| flat[((t * D + d) * 1 + 0) * H + u];
    let mut want = vec![0.0f32; D * H];
    for u in 0..H {
        want[u] = at(SEQ - 1, 0, u);
        want[H + u] = at(0, 1, u);
    }

    let mut gates = vec![0.0f32; G];
    let (mut got_s, mut got_v) = (vec![0.0f32; D * H], vec![0.0f32; D * H]);
    scalar(&l, &mut gates, &mut got_s);
    #[cfg(target_arch = "x86_64")]
    let has_avx2 = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
    #[cfg(target_arch = "x86_64")]
    if has_avx2 {
        unsafe { avx2::run(&l, &mut gates, &mut got_v) };
    }
    let dev = |a: &[f32]| {
        a.iter()
            .zip(&want)
            .map(|(p, q)| (p - q).abs())
            .fold(0.0f32, f32::max)
    };
    println!("max |delta| vs tract:  scalar {:.3e}   avx2 {:.3e}", dev(&got_s), dev(&got_v));

    let bench = |name: &str, threads: usize, f: &(dyn Fn() + Sync)| {
        let calls = 200;
        let t0 = Instant::now();
        std::thread::scope(|s| {
            for _ in 0..threads {
                s.spawn(|| {
                    for _ in 0..calls {
                        f();
                    }
                });
            }
        });
        let wall = t0.elapsed().as_secs_f64();
        println!(
            "{name:<14} threads {threads:>3}   {:>7.1} us/read/core   {:>9.0} reads/s total",
            wall / calls as f64 * 1e6,
            (threads * calls) as f64 / wall
        );
    };

    for threads in [1usize, 16, 32, 48] {
        bench("tract", threads, &|| {
            let t = Tensor::from_shape(&[SEQ, 1, IN], &x).unwrap();
            std::hint::black_box(plan.run(tvec!(t.into())).unwrap());
        });
        bench("scalar", threads, &|| {
            let mut g = vec![0.0f32; G];
            let mut o = vec![0.0f32; D * H];
            scalar(&l, &mut g, &mut o);
            std::hint::black_box(o);
        });
        #[cfg(target_arch = "x86_64")]
        if has_avx2 {
            bench("avx2", threads, &|| {
                let mut g = vec![0.0f32; G];
                let mut o = vec![0.0f32; D * H];
                unsafe { avx2::run(&l, &mut g, &mut o) };
                std::hint::black_box(o);
            });
        }
        println!();
    }
    Ok(())
}
