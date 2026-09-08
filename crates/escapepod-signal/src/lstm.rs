// SPDX-License-Identifier: MIT

//! A native unidirectional LSTM recurrence, runtime-dispatched to AVX2/AVX-512.
//!
//! This is the axpy-form kernel `escapepod_classify::fnn_lstm` introduced for
//! the charging feature network's bidirectional LSTM (490 µs/read through
//! tract, ~97 µs/read here) and `escapepod_demux::crf::encoder_native` reuses
//! for the barcode CRF's five stacked *unidirectional* layers. Nothing about
//! the recurrence itself is specific to either caller — `H = 96` in both, ONNX
//! `W`/`R`/`B` layout, no peepholes, zero initial state — so it lives here,
//! one level below both consumers, and each caller supplies its own direction
//! and output layout:
//!
//! * classify's bidirectional net calls this once per direction (`reverse =
//!   false`, `true`) into one interleaved `[seq][2H]` buffer (`hs_stride =
//!   2H`, `hs_offset = d * H`), then reduces both halves together.
//! * the CRF's five unidirectional layers each call this once, into their own
//!   `[seq][H]` buffer (`hs_stride = H`, `hs_offset = 0`), and layer `l + 1`'s
//!   input is layer `l`'s output transposed back to channel-major (the ONNX
//!   graph avoids that transpose by keeping every LSTM in time-major layout
//!   throughout; this kernel keeps the channel-major *input* contract instead,
//!   because [`input_contribution`] wants one channel's whole timeseries
//!   contiguous, so the transpose happens once per layer boundary instead of
//!   once per gate element).
//!
//! For which layers reverse and why (bonito's "run backward without a
//! backward LSTM" trick — reverse the sequence, run an ONNX-forward LSTM,
//! reverse the output back), see `escapepod_demux::crf::encoder_native`.
//!
//! The kernel is the axpy form of the recurrence: for each hidden unit `j`,
//! the previous state `h[j]` scales row `j` of `Rᵀ` into the 4H gate
//! pre-activations, so every inner loop is a unit-stride fused multiply-add
//! and nothing is horizontally reduced. The input contribution does not
//! depend on the recurrence and is computed for all timesteps up front with
//! both ONNX bias halves folded in. AVX2+FMA is runtime-dispatched per the
//! build policy (never a baseline bump); the scalar path is the reference and
//! the fallback.
//!
//! Numerics: ONNX gate order is `i, o, f, c`; the default activations are
//! sigmoid, tanh, tanh. The vector path uses a Cephes-style `exp` — the same
//! construction `escapepod_demux::crf::avx2` uses — and `tanh(x) = 2σ(2x) −
//! 1`, so it is not bit-identical to the scalar path, or to tract, which uses
//! its own vector approximations. The contract is agreement on the output
//! within each caller's own tolerance, not bit-exactness.

use std::cell::RefCell;

/// Which kernel runs the recurrence. Ordered slowest to fastest, so a cap
/// from `ESCAPEPOD_LSTM_BACKEND` is a comparison. Named distinctly from
/// `escapepod_demux::crf::Backend` (the CRF *decode*'s kernel choice) since
/// `crf::encoder_native` has to name both in the same scope.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum LstmBackend {
    Scalar,
    #[cfg(target_arch = "x86_64")]
    Avx2,
    #[cfg(target_arch = "x86_64")]
    Avx512,
}

/// `ESCAPEPOD_LSTM_BACKEND=scalar|avx2|avx512` caps the kernel the dispatch
/// may pick — the A/B lever between the widths inside one binary. Never
/// raises: a cap the machine cannot run is simply not reached.
///
/// Public so a caller's own dispatch tests (`escapepod_classify::fnn_lstm`'s
/// among them) can tell whether the environment is already pinning a
/// backend before asserting what an unconstrained dispatch would pick.
pub fn backend_cap() -> Option<LstmBackend> {
    static CAP: std::sync::OnceLock<Option<LstmBackend>> = std::sync::OnceLock::new();
    *CAP.get_or_init(|| {
        let v = std::env::var("ESCAPEPOD_LSTM_BACKEND").ok()?;
        match v.as_str() {
            "scalar" => Some(LstmBackend::Scalar),
            #[cfg(target_arch = "x86_64")]
            "avx2" => Some(LstmBackend::Avx2),
            #[cfg(target_arch = "x86_64")]
            "avx512" => Some(LstmBackend::Avx512),
            other => {
                tracing::warn!("ESCAPEPOD_LSTM_BACKEND={other}: not a kernel name, ignored");
                None
            }
        }
    })
}

impl LstmBackend {
    /// Every kernel, slowest to fastest.
    #[cfg(target_arch = "x86_64")]
    const ALL: [LstmBackend; 3] = [LstmBackend::Scalar, LstmBackend::Avx2, LstmBackend::Avx512];
    #[cfg(not(target_arch = "x86_64"))]
    const ALL: [LstmBackend; 1] = [LstmBackend::Scalar];

    /// The vector width. A hidden size must be a multiple of it: the
    /// activations run `lanes` units at a time and there is no masked tail.
    pub fn lanes(self) -> usize {
        match self {
            LstmBackend::Scalar => 1,
            #[cfg(target_arch = "x86_64")]
            LstmBackend::Avx2 => 8,
            #[cfg(target_arch = "x86_64")]
            LstmBackend::Avx512 => 16,
        }
    }

    /// Whether this machine can run the kernel.
    ///
    /// The one place the `unsafe` calls into the `#[target_feature]` kernels
    /// rest on: every constructor that picks a backend consults it, so a
    /// backend a caller carries is one whose instructions the CPU can
    /// execute. AVX-512 asks for AVX2 + FMA as well, because its lone-read
    /// path is the AVX2 kernel.
    pub fn supported(self) -> bool {
        match self {
            LstmBackend::Scalar => true,
            #[cfg(target_arch = "x86_64")]
            LstmBackend::Avx2 => {
                is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
            }
            #[cfg(target_arch = "x86_64")]
            LstmBackend::Avx512 => {
                is_x86_feature_detected!("avx512f")
                    && is_x86_feature_detected!("avx2")
                    && is_x86_feature_detected!("fma")
            }
        }
    }

    /// Every kernel this machine can run, slowest to fastest — what an
    /// equivalence test sweeps, whichever one the dispatch prefers.
    pub fn available() -> Vec<LstmBackend> {
        Self::ALL.into_iter().filter(|b| b.supported()).collect()
    }

    /// The fastest kernel this machine can run for a hidden size.
    ///
    /// A hidden size that is not a multiple of a kernel's width falls
    /// through to the next kernel rather than growing a masked tail nobody
    /// ships. AVX-512 is runtime-detected, never a baseline bump: the release
    /// artifact is built for Haswell, and Broadwell login nodes and Alpine's
    /// Zen3 take the AVX2 kernel.
    pub fn best_for(h: usize) -> Self {
        let cap = backend_cap();
        Self::ALL
            .into_iter()
            .rev()
            .find(|&b| h.is_multiple_of(b.lanes()) && cap.is_none_or(|c| b <= c) && b.supported())
            .unwrap_or(LstmBackend::Scalar)
    }

    pub fn name(self) -> &'static str {
        match self {
            LstmBackend::Scalar => "scalar",
            #[cfg(target_arch = "x86_64")]
            LstmBackend::Avx2 => "avx2",
            #[cfg(target_arch = "x86_64")]
            LstmBackend::Avx512 => "avx512",
        }
    }

    /// How many reads a batched call scores per pass at full efficiency.
    ///
    /// Reads scored in lockstep share every weight-row load. Three is what
    /// the AVX2 register file holds — 4 accumulators × 3 reads, 3 broadcasts,
    /// one row load — and eight fills the AVX-512 one over the same 32-gate
    /// slices (2 accumulators × 8 reads, 8 broadcasts, 2 row loads). A caller
    /// batching in multiples of it loses nothing to the tail.
    ///
    /// `ESCAPEPOD_LSTM_BATCH` overrides the width, clamped to what this
    /// backend has kernels for — the sweep lever, and the way to measure the
    /// single-read kernel inside a binary that batches.
    pub fn preferred_batch(self) -> usize {
        static OVERRIDE: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
        let wanted = OVERRIDE.get_or_init(|| {
            std::env::var("ESCAPEPOD_LSTM_BATCH")
                .ok()
                .and_then(|s| s.parse().ok())
                .filter(|&n| n >= 1)
        });
        let max = match self {
            LstmBackend::Scalar => 1,
            #[cfg(target_arch = "x86_64")]
            LstmBackend::Avx2 => 3,
            #[cfg(target_arch = "x86_64")]
            LstmBackend::Avx512 => 8,
        };
        wanted.map_or(max, |n| n.min(max))
    }
}

/// One direction's recurrence weights, repacked for the step loop.
///
/// Pure data: which direction this is, and what feeds it (a bidirectional
/// net's other direction, a stacked net's neighbouring layer), is the
/// caller's business.
#[derive(Debug, Clone)]
pub struct LstmWeights {
    /// Input channels per timestep.
    pub n_in: usize,
    /// Hidden units.
    pub h: usize,
    /// `W` transposed to `[n_in][4H]`, so the input contribution for one
    /// timestep is `n_in` axpys over a 4H vector.
    wt: Vec<f32>,
    /// `[4H]`, both ONNX bias halves summed.
    bias: Vec<f32>,
    /// `R` transposed to `[H][4H]`.
    rt: Vec<f32>,
    /// The same `Rᵀ` packed block-major, `[4H/32][H][32]`, so one 32-gate
    /// pass over all `H` rows streams a contiguous 12 KB instead of 32 floats
    /// out of every 1.5 KB row — worth 14% of the single-read kernel and 11%
    /// of the batched one. Empty unless `4H` is a multiple of 32 (the AVX2
    /// backend's precondition, `H % 8 == 0`); the scalar path keeps reading
    /// `rt`. Only the x86_64 kernels read it, so on any other target it is
    /// built and never read.
    #[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
    rt_blocked: Vec<f32>,
}

impl LstmWeights {
    /// Repack one direction's ONNX-layout tensors.
    ///
    /// `w` is `[4H, n_in]` flat (one direction's slab of ONNX `W`), `r` is
    /// `[4H, H]` (one slab of `R`), `b` is `[8H]` (one slab of `B`, both
    /// halves summed here). The caller has already sliced the direction axis
    /// off; this function knows nothing about it.
    pub fn from_onnx(n_in: usize, h: usize, w: &[f32], r: &[f32], b: &[f32]) -> Self {
        let g = 4 * h;
        debug_assert_eq!(w.len(), g * n_in);
        debug_assert_eq!(r.len(), g * h);
        debug_assert_eq!(b.len(), 8 * h);
        let mut wt = vec![0.0f32; n_in * g];
        for gi in 0..g {
            for k in 0..n_in {
                wt[k * g + gi] = w[gi * n_in + k];
            }
        }
        let bias = (0..g).map(|gi| b[gi] + b[g + gi]).collect();
        let mut rt = vec![0.0f32; h * g];
        for gi in 0..g {
            for j in 0..h {
                rt[j * g + gi] = r[gi * h + j];
            }
        }
        let rt_blocked = block_rows(&rt, h, g, 32);
        Self {
            n_in,
            h,
            wt,
            bias,
            rt,
            rt_blocked,
        }
    }
}

thread_local! {
    /// Per-thread scratch shared by every caller in this crate family: one
    /// read's input contribution, per-timestep outputs and gate buffer.
    /// Sized on first use and kept, so the steady state allocates nothing.
    /// Exposed so a caller scoring many reads can reuse one buffer across
    /// calls instead of each allocating its own.
    pub static LSTM_SCRATCH: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

/// The input contribution for every timestep, biases folded in:
/// `xw[t][g] = bias[g] + Σ_k W[g][k] x[k][t]`.
///
/// `x` is channel-major, `[n_in][seq]` — one channel's whole timeseries
/// contiguous, so this reads unit-stride down `x` for a fixed `t` only in the
/// inner accumulation, not across the outer loop; `xw` is `[seq][4H]`.
pub fn input_contribution(w: &LstmWeights, seq: usize, x: &[f32], xw: &mut [f32]) {
    let g = 4 * w.h;
    debug_assert_eq!(x.len(), w.n_in * seq);
    debug_assert_eq!(xw.len(), seq * g);
    for t in 0..seq {
        let row = &mut xw[t * g..(t + 1) * g];
        row.copy_from_slice(&w.bias);
        for k in 0..w.n_in {
            let xk = x[k * seq + t];
            let wk = &w.wt[k * g..(k + 1) * g];
            for (r, wv) in row.iter_mut().zip(wk) {
                *r += xk * wv;
            }
        }
    }
}

/// One read, one direction, scalar reference kernel.
///
/// `x` is channel-major `[n_in][seq]`. `xw` is `seq * 4H` scratch; `gates` is
/// `4H` scratch. `hs` receives every hidden state: `hs[t * hs_stride +
/// hs_offset .. + H] = h_t`, so a caller writing several directions or layers
/// into one buffer chooses where each one lands. `reverse` walks `t` from
/// `seq - 1` down to `0` while still writing each `h_t` at its own `t` — the
/// "run backward, store forward" trick, applied without a reversal copy.
#[allow(clippy::too_many_arguments)]
pub fn run_scalar(
    w: &LstmWeights,
    seq: usize,
    reverse: bool,
    x: &[f32],
    xw: &mut [f32],
    hs: &mut [f32],
    hs_stride: usize,
    hs_offset: usize,
    gates: &mut [f32],
) {
    let (h, g) = (w.h, 4 * w.h);
    input_contribution(w, seq, x, xw);
    let mut hcur = vec![0.0f32; h];
    let mut c = vec![0.0f32; h];
    for step in 0..seq {
        let t = if reverse { seq - 1 - step } else { step };
        gates.copy_from_slice(&xw[t * g..(t + 1) * g]);
        if step > 0 {
            for (j, &hj) in hcur.iter().enumerate() {
                let row = &w.rt[j * g..(j + 1) * g];
                for (ga, r) in gates.iter_mut().zip(row) {
                    *ga += hj * r;
                }
            }
        }
        for u in 0..h {
            let i = sigmoid(gates[u]);
            let o = sigmoid(gates[h + u]);
            let f = sigmoid(gates[2 * h + u]);
            let ct = gates[3 * h + u].tanh();
            c[u] = f * c[u] + i * ct;
            hcur[u] = o * c[u].tanh();
        }
        let dst = t * hs_stride + hs_offset;
        hs[dst..dst + h].copy_from_slice(&hcur);
    }
}

/// [`run_scalar`], `N` reads in lockstep — trivially a loop, since the scalar
/// path has no shared weight-row load to amortise across reads.
#[allow(clippy::too_many_arguments)]
pub fn run_scalar_batch(
    w: &LstmWeights,
    seq: usize,
    reverse: bool,
    xs: &[&[f32]],
    xw: &mut [f32],
    hs: &mut [f32],
    hs_stride: usize,
    hs_offset: usize,
    gates: &mut [f32],
) {
    let g = 4 * w.h;
    let xw_len = seq * g;
    let hs_len = seq * hs_stride;
    for (r, x) in xs.iter().enumerate() {
        run_scalar(
            w,
            seq,
            reverse,
            x,
            &mut xw[r * xw_len..(r + 1) * xw_len],
            &mut hs[r * hs_len..r * hs_len + seq * hs_stride],
            hs_stride,
            hs_offset,
            &mut gates[..g],
        );
    }
}

#[inline(always)]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(target_arch = "x86_64")]
// `#[inline(never)]` on every kernel below, on evidence rather than taste:
// with a batched kernel inlined beside its single-read sibling (the crate
// baseline is x86-64-v3, so the AVX2 ones may be), the AVX2 width-2 batched
// kernel returned wrong values for its second read — wrong by the same bits
// on every run, on a node where the same kernel source had passed the day
// before. Bisected to the *form* of the caller's dispatch (an or-pattern over
// two lone-read arms failed, two arms with the same body passed) and cleared
// by this attribute with the or-pattern kept. The kernels' unsafe code was
// audited for aliasing and bounds and nothing was found; an LLVM miscompile is
// suspected and not proven. These are large leaf functions, so inlining them
// buys nothing, and callers pin every kernel on every backend the machine
// has, whichever the dispatch prefers, not only the one it reaches.
#[inline(never)]
#[target_feature(enable = "avx2,fma")]
/// # Safety
///
/// The caller must have verified `is_x86_feature_detected!("avx2")` and
/// `("fma")`, and that `w.h` is a multiple of 8 (`LstmBackend::Avx2` only
/// dispatches when both hold).
#[allow(clippy::too_many_arguments)]
pub unsafe fn run_avx2(
    w: &LstmWeights,
    seq: usize,
    reverse: bool,
    x: &[f32],
    xw: &mut [f32],
    hs: &mut [f32],
    hs_stride: usize,
    hs_offset: usize,
    gates: &mut [f32],
) {
    use std::arch::x86_64::*;
    let (h, g) = (w.h, 4 * w.h);
    debug_assert!(h.is_multiple_of(8));
    // Safety: every pointer below is derived from a slice whose length was
    // checked by the caller (`gates` is 4H, `w.rt` is H x 4H, and this
    // backend is only selected for H a multiple of 8), so each 8-wide load
    // and store stays inside its buffer; the target features are guaranteed
    // by the dispatch in `LstmBackend::best_for`.
    unsafe {
        input_contribution(w, seq, x, xw);
        let mut hcur = vec![0.0f32; h];
        let mut c = vec![0.0f32; h];
        let rt = w.rt.as_ptr();
        let rtb = w.rt_blocked.as_ptr();
        for step in 0..seq {
            let t = if reverse { seq - 1 - step } else { step };
            gates.copy_from_slice(&xw[t * g..(t + 1) * g]);
            if step > 0 {
                // gates += h[j] * Rᵀ[j][..]: an axpy per hidden unit, so every
                // load is unit-stride and nothing is horizontally reduced.
                // Accumulators stay in registers a chunk at a time; 64 gates =
                // 8 vectors held live.
                let gp = gates.as_mut_ptr();
                let mut base = 0usize;
                while base + 64 <= g {
                    // Eight named accumulators, not an array: an indexed
                    // `[__m256; 8]` walked by closures was measured at ~3x
                    // the arithmetic bound, the signature of accumulators
                    // living on the stack. Named, they are eight independent
                    // FMA chains in eight registers.
                    let p = gp.add(base);
                    let mut a0 = _mm256_loadu_ps(p);
                    let mut a1 = _mm256_loadu_ps(p.add(8));
                    let mut a2 = _mm256_loadu_ps(p.add(16));
                    let mut a3 = _mm256_loadu_ps(p.add(24));
                    let mut a4 = _mm256_loadu_ps(p.add(32));
                    let mut a5 = _mm256_loadu_ps(p.add(40));
                    let mut a6 = _mm256_loadu_ps(p.add(48));
                    let mut a7 = _mm256_loadu_ps(p.add(56));
                    // Two 32-gate blocks of `rt_blocked`, each a contiguous
                    // `H x 32` run: two sequential streams the prefetcher can
                    // follow.
                    let mut row0 = rtb.add((base / 32) * h * 32);
                    let mut row1 = row0.add(h * 32);
                    for &hj in hcur.iter() {
                        let hv = _mm256_set1_ps(hj);
                        a0 = _mm256_fmadd_ps(hv, _mm256_loadu_ps(row0), a0);
                        a1 = _mm256_fmadd_ps(hv, _mm256_loadu_ps(row0.add(8)), a1);
                        a2 = _mm256_fmadd_ps(hv, _mm256_loadu_ps(row0.add(16)), a2);
                        a3 = _mm256_fmadd_ps(hv, _mm256_loadu_ps(row0.add(24)), a3);
                        a4 = _mm256_fmadd_ps(hv, _mm256_loadu_ps(row1), a4);
                        a5 = _mm256_fmadd_ps(hv, _mm256_loadu_ps(row1.add(8)), a5);
                        a6 = _mm256_fmadd_ps(hv, _mm256_loadu_ps(row1.add(16)), a6);
                        a7 = _mm256_fmadd_ps(hv, _mm256_loadu_ps(row1.add(24)), a7);
                        row0 = row0.add(32);
                        row1 = row1.add(32);
                    }
                    _mm256_storeu_ps(p, a0);
                    _mm256_storeu_ps(p.add(8), a1);
                    _mm256_storeu_ps(p.add(16), a2);
                    _mm256_storeu_ps(p.add(24), a3);
                    _mm256_storeu_ps(p.add(32), a4);
                    _mm256_storeu_ps(p.add(40), a5);
                    _mm256_storeu_ps(p.add(48), a6);
                    _mm256_storeu_ps(p.add(56), a7);
                    base += 64;
                }
                // A hidden size that is a multiple of 8 but not of 16 leaves a
                // tail narrower than one chunk.
                while base < g {
                    let mut a = _mm256_loadu_ps(gp.add(base));
                    for (j, &hj) in hcur.iter().enumerate() {
                        a = _mm256_fmadd_ps(
                            _mm256_set1_ps(hj),
                            _mm256_loadu_ps(rt.add(j * g + base)),
                            a,
                        );
                    }
                    _mm256_storeu_ps(gp.add(base), a);
                    base += 8;
                }
            }
            let gp = gates.as_ptr();
            for u in (0..h).step_by(8) {
                let gi = sigmoid8(_mm256_loadu_ps(gp.add(u)));
                let go = sigmoid8(_mm256_loadu_ps(gp.add(h + u)));
                let gf = sigmoid8(_mm256_loadu_ps(gp.add(2 * h + u)));
                let gc = tanh8(_mm256_loadu_ps(gp.add(3 * h + u)));
                let cprev = _mm256_loadu_ps(c.as_ptr().add(u));
                let cn = _mm256_fmadd_ps(gf, cprev, _mm256_mul_ps(gi, gc));
                _mm256_storeu_ps(c.as_mut_ptr().add(u), cn);
                _mm256_storeu_ps(hcur.as_mut_ptr().add(u), _mm256_mul_ps(go, tanh8(cn)));
            }
            let dst = t * hs_stride + hs_offset;
            hs[dst..dst + h].copy_from_slice(&hcur);
        }
    }
}

#[cfg(target_arch = "x86_64")]
/// `N` reads in lockstep: one recurrence step advances every read, and each
/// 32-gate slice of a weight row is loaded once and applied to all `N`
/// accumulator sets. `xw` is `[N][seq][G]`, `hs` is the caller's buffer
/// (`hs_stride`/`hs_offset` as in [`run_avx2`]), `gates` is `[N][G]`.
///
/// Bit-identical to [`run_avx2`] called once per read: every gate lane still
/// accumulates `h[j] * Rᵀ[j]` over `j` in order, whatever the chunk width or
/// the batch. The register budget is `4 accumulators × N + N broadcasts + 1
/// row load`, which is why `N ≤ 3`.
///
/// # Safety
///
/// As [`run_avx2`]; additionally `xs.len() == N`.
#[inline(never)]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn run_avx2_batch<const N: usize>(
    w: &LstmWeights,
    seq: usize,
    reverse: bool,
    xs: &[&[f32]],
    xw: &mut [f32],
    hs: &mut [f32],
    hs_stride: usize,
    hs_offset: usize,
    gates: &mut [f32],
) {
    use std::arch::x86_64::*;
    debug_assert_eq!(xs.len(), N);
    let (h, g) = (w.h, 4 * w.h);
    debug_assert!(h.is_multiple_of(8));
    let xw_len = seq * g;
    let hs_len = seq * hs_stride;
    // Safety: as `run_avx2`. The caller sized `xw`/`hs`/`gates` for `N` reads.
    unsafe {
        for (r, x) in xs.iter().enumerate() {
            let off = r * xw_len;
            input_contribution(w, seq, x, &mut xw[off..off + xw_len]);
        }
        let mut hcur = vec![0.0f32; N * h];
        let mut c = vec![0.0f32; N * h];
        let rt = w.rt.as_ptr();
        let rtb = w.rt_blocked.as_ptr();
        for step in 0..seq {
            let t = if reverse { seq - 1 - step } else { step };
            for r in 0..N {
                let src = r * xw_len + t * g;
                gates[r * g..(r + 1) * g].copy_from_slice(&xw[src..src + g]);
            }
            if step > 0 {
                let gp = gates.as_mut_ptr();
                let mut base = 0usize;
                while base + 32 <= g {
                    // `acc[v][r]`: vector `v` of the 32-gate slice, for read
                    // `r` — so one row load serves the inner loop over reads.
                    let mut acc = [[_mm256_setzero_ps(); N]; 4];
                    for (v, accv) in acc.iter_mut().enumerate() {
                        for (r, a) in accv.iter_mut().enumerate() {
                            *a = _mm256_loadu_ps(gp.add(r * g + base + v * 8));
                        }
                    }
                    // One 32-gate block of `rt_blocked`: the `H` rows this
                    // pass reads are contiguous.
                    let mut row = rtb.add((base / 32) * h * 32);
                    // Raw reads of `hcur`: indexed, every `j` paid two bounds
                    // checks per read inside the loop that streams the
                    // recurrent weights.
                    let hc = hcur.as_ptr();
                    for j in 0..h {
                        let mut hv = [_mm256_setzero_ps(); N];
                        for (r, hvr) in hv.iter_mut().enumerate() {
                            *hvr = _mm256_set1_ps(*hc.add(r * h + j));
                        }
                        for (v, accv) in acc.iter_mut().enumerate() {
                            let wv = _mm256_loadu_ps(row.add(v * 8));
                            for (a, hvr) in accv.iter_mut().zip(hv.iter()) {
                                *a = _mm256_fmadd_ps(*hvr, wv, *a);
                            }
                        }
                        row = row.add(32);
                    }
                    for (v, accv) in acc.iter().enumerate() {
                        for (r, a) in accv.iter().enumerate() {
                            _mm256_storeu_ps(gp.add(r * g + base + v * 8), *a);
                        }
                    }
                    base += 32;
                }
                while base < g {
                    for r in 0..N {
                        let mut a = _mm256_loadu_ps(gp.add(r * g + base));
                        let hc = hcur.as_ptr().add(r * h);
                        for j in 0..h {
                            a = _mm256_fmadd_ps(
                                _mm256_set1_ps(*hc.add(j)),
                                _mm256_loadu_ps(rt.add(j * g + base)),
                                a,
                            );
                        }
                        _mm256_storeu_ps(gp.add(r * g + base), a);
                    }
                    base += 8;
                }
            }
            for r in 0..N {
                let gp = gates.as_ptr().add(r * g);
                let cp = c.as_mut_ptr().add(r * h);
                let hp = hcur.as_mut_ptr().add(r * h);
                for u in (0..h).step_by(8) {
                    let gi = sigmoid8(_mm256_loadu_ps(gp.add(u)));
                    let go = sigmoid8(_mm256_loadu_ps(gp.add(h + u)));
                    let gf = sigmoid8(_mm256_loadu_ps(gp.add(2 * h + u)));
                    let gc = tanh8(_mm256_loadu_ps(gp.add(3 * h + u)));
                    let cprev = _mm256_loadu_ps(cp.add(u));
                    let cn = _mm256_fmadd_ps(gf, cprev, _mm256_mul_ps(gi, gc));
                    _mm256_storeu_ps(cp.add(u), cn);
                    _mm256_storeu_ps(hp.add(u), _mm256_mul_ps(go, tanh8(cn)));
                }
                let dst = r * hs_len + t * hs_stride + hs_offset;
                hs[dst..dst + h].copy_from_slice(&hcur[r * h..(r + 1) * h]);
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
/// `N` reads in lockstep, sixteen lanes wide, over the same 32-gate slices as
/// [`run_avx2_batch`]: 2 accumulators × `N`, `N` broadcasts and 2 row loads,
/// which fits the 32 ZMM registers up to `N = 8`. The weights are streamed
/// once per eight reads instead of three.
///
/// Bit-identical to [`run_avx2_batch`] and [`run_avx2`]: each gate lane still
/// accumulates `h[j] * Rᵀ[j]` over `j` in order whatever the vector width, and
/// every activation is the same sequence of IEEE operations per lane (`vec16`
/// mirrors `vec8` op for op).
///
/// # Safety
///
/// As [`run_avx2_batch`]; additionally `w.h` must be a multiple of 16 and
/// `N <= 8`.
#[inline(never)]
#[target_feature(enable = "avx512f")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn run_avx512_batch<const N: usize>(
    w: &LstmWeights,
    seq: usize,
    reverse: bool,
    xs: &[&[f32]],
    xw: &mut [f32],
    hs: &mut [f32],
    hs_stride: usize,
    hs_offset: usize,
    gates: &mut [f32],
) {
    use std::arch::x86_64::*;
    use vec16::{sigmoid16, tanh16};
    debug_assert_eq!(xs.len(), N);
    let (h, g) = (w.h, 4 * w.h);
    // `h % 16 == 0` (the dispatch's precondition) makes `g` a multiple of 64,
    // so the slice loop has no tail and `rt_blocked` is populated.
    debug_assert!(h.is_multiple_of(16));
    debug_assert!(N <= 8);
    let xw_len = seq * g;
    let hs_len = seq * hs_stride;
    // Safety: as `run_avx2_batch`, at the wider feature and width.
    unsafe {
        for (r, x) in xs.iter().enumerate() {
            let off = r * xw_len;
            input_contribution(w, seq, x, &mut xw[off..off + xw_len]);
        }
        let mut hcur = vec![0.0f32; N * h];
        let mut c = vec![0.0f32; N * h];
        let rtb = w.rt_blocked.as_ptr();
        for step in 0..seq {
            let t = if reverse { seq - 1 - step } else { step };
            for r in 0..N {
                let src = r * xw_len + t * g;
                gates[r * g..(r + 1) * g].copy_from_slice(&xw[src..src + g]);
            }
            if step > 0 {
                let gp = gates.as_mut_ptr();
                let hc = hcur.as_ptr();
                let mut base = 0usize;
                while base < g {
                    // `acc[v][r]`: vector `v` of the 32-gate slice for read
                    // `r`; one block of `rt_blocked`, one contiguous stream.
                    let mut acc = [[_mm512_setzero_ps(); N]; 2];
                    for (v, accv) in acc.iter_mut().enumerate() {
                        for (r, a) in accv.iter_mut().enumerate() {
                            *a = _mm512_loadu_ps(gp.add(r * g + base + v * 16));
                        }
                    }
                    let mut row = rtb.add((base / 32) * h * 32);
                    for j in 0..h {
                        let mut hv = [_mm512_setzero_ps(); N];
                        for (r, hvr) in hv.iter_mut().enumerate() {
                            *hvr = _mm512_set1_ps(*hc.add(r * h + j));
                        }
                        let wv = [_mm512_loadu_ps(row), _mm512_loadu_ps(row.add(16))];
                        for (accv, wvv) in acc.iter_mut().zip(wv.iter()) {
                            for (a, hvr) in accv.iter_mut().zip(hv.iter()) {
                                *a = _mm512_fmadd_ps(*hvr, *wvv, *a);
                            }
                        }
                        row = row.add(32);
                    }
                    for (v, accv) in acc.iter().enumerate() {
                        for (r, a) in accv.iter().enumerate() {
                            _mm512_storeu_ps(gp.add(r * g + base + v * 16), *a);
                        }
                    }
                    base += 32;
                }
            }
            for r in 0..N {
                let gp = gates.as_ptr().add(r * g);
                let cp = c.as_mut_ptr().add(r * h);
                let hp = hcur.as_mut_ptr().add(r * h);
                for u in (0..h).step_by(16) {
                    let gi = sigmoid16(_mm512_loadu_ps(gp.add(u)));
                    let go = sigmoid16(_mm512_loadu_ps(gp.add(h + u)));
                    let gf = sigmoid16(_mm512_loadu_ps(gp.add(2 * h + u)));
                    let gc = tanh16(_mm512_loadu_ps(gp.add(3 * h + u)));
                    let cprev = _mm512_loadu_ps(cp.add(u));
                    let cn = _mm512_fmadd_ps(gf, cprev, _mm512_mul_ps(gi, gc));
                    _mm512_storeu_ps(cp.add(u), cn);
                    _mm512_storeu_ps(hp.add(u), _mm512_mul_ps(go, tanh16(cn)));
                }
                let dst = r * hs_len + t * hs_stride + hs_offset;
                hs[dst..dst + h].copy_from_slice(&hcur[r * h..(r + 1) * h]);
            }
        }
    }
}

/// `rows` as `[h][g]` repacked block-major, `[g / width][h][width]`, so a
/// pass over all `h` rows of one `width`-gate block is one contiguous run.
/// Empty when `g` is not a multiple of `width`; the caller keeps the row
/// layout for that case.
fn block_rows(rows: &[f32], h: usize, g: usize, width: usize) -> Vec<f32> {
    if !g.is_multiple_of(width) {
        return Vec::new();
    }
    let mut out = vec![0.0f32; h * g];
    for (blk, dst) in out.chunks_exact_mut(h * width).enumerate() {
        for (j, row) in dst.chunks_exact_mut(width).enumerate() {
            let src = j * g + blk * width;
            row.copy_from_slice(&rows[src..src + width]);
        }
    }
    out
}

/// `vec8`, sixteen wide: the same operations in the same order, so a lane
/// through here and a lane through `vec8` produce the same bits.
#[cfg(target_arch = "x86_64")]
mod vec16 {
    use std::arch::x86_64::*;

    #[inline]
    #[target_feature(enable = "avx512f")]
    pub fn exp16(x: __m512) -> __m512 {
        let x = _mm512_min_ps(_mm512_set1_ps(88.376_26), x);
        let x = _mm512_max_ps(_mm512_set1_ps(-88.376_26), x);
        let fx = _mm512_fmadd_ps(
            x,
            _mm512_set1_ps(std::f32::consts::LOG2_E),
            _mm512_set1_ps(0.5),
        );
        // Floor with exceptions suppressed: what `_mm256_floor_ps` is.
        let fx = _mm512_roundscale_ps::<0x09>(fx);
        let r = _mm512_fnmadd_ps(fx, _mm512_set1_ps(0.693_359_4), x);
        let r = _mm512_fnmadd_ps(fx, _mm512_set1_ps(-2.121_944_4e-4), r);
        let r2 = _mm512_mul_ps(r, r);
        let mut y = _mm512_set1_ps(1.987_569_1e-4);
        y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(1.398_199_9e-3));
        y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(8.333_452e-3));
        y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(4.166_579_6e-2));
        y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(1.666_666_6e-1));
        y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(5e-1));
        y = _mm512_fmadd_ps(y, r2, r);
        y = _mm512_add_ps(y, _mm512_set1_ps(1.0));
        let imm = _mm512_cvtps_epi32(fx);
        let pow2 = _mm512_castsi512_ps(_mm512_slli_epi32::<23>(_mm512_add_epi32(
            imm,
            _mm512_set1_epi32(0x7f),
        )));
        _mm512_mul_ps(y, pow2)
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    pub fn sigmoid16(x: __m512) -> __m512 {
        let e = exp16(_mm512_sub_ps(_mm512_setzero_ps(), x));
        _mm512_div_ps(_mm512_set1_ps(1.0), _mm512_add_ps(_mm512_set1_ps(1.0), e))
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    pub fn tanh16(x: __m512) -> __m512 {
        let s = sigmoid16(_mm512_add_ps(x, x));
        _mm512_fmsub_ps(_mm512_set1_ps(2.0), s, _mm512_set1_ps(1.0))
    }
}

#[cfg(target_arch = "x86_64")]
mod vec8 {
    use std::arch::x86_64::*;

    /// Cephes-style `exp`, the same construction `escapepod_demux::crf::avx2`
    /// uses: range-reduce by `log2 e`, a degree-5 polynomial on the
    /// remainder, and the power of two assembled straight into the exponent
    /// field.
    //
    // Safe functions with the target features enabled: the intrinsics they
    // use are safe under those features, so callers that share the features
    // (`run_avx2`) call them like any other function, and a caller that does
    // not is made to write the `unsafe` it owes.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub fn exp8(x: __m256) -> __m256 {
        let x = _mm256_min_ps(_mm256_set1_ps(88.376_26), x);
        let x = _mm256_max_ps(_mm256_set1_ps(-88.376_26), x);
        let fx = _mm256_fmadd_ps(
            x,
            _mm256_set1_ps(std::f32::consts::LOG2_E),
            _mm256_set1_ps(0.5),
        );
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
        let imm = _mm256_cvtps_epi32(fx);
        let pow2 = _mm256_castsi256_ps(_mm256_slli_epi32(
            _mm256_add_epi32(imm, _mm256_set1_epi32(0x7f)),
            23,
        ));
        _mm256_mul_ps(y, pow2)
    }

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub fn sigmoid8(x: __m256) -> __m256 {
        let e = exp8(_mm256_sub_ps(_mm256_setzero_ps(), x));
        _mm256_div_ps(_mm256_set1_ps(1.0), _mm256_add_ps(_mm256_set1_ps(1.0), e))
    }

    /// `tanh(x) = 2σ(2x) − 1`: one `exp`, like the sigmoid.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub fn tanh8(x: __m256) -> __m256 {
        let s = sigmoid8(_mm256_add_ps(x, x));
        _mm256_fmsub_ps(_mm256_set1_ps(2.0), s, _mm256_set1_ps(1.0))
    }
}
#[cfg(target_arch = "x86_64")]
use vec8::{sigmoid8, tanh8};

/// `tanh`, elementwise and in place, dispatched to the fastest backend this
/// machine has and the caller allows. Shared so a caller applying `tanh` to a
/// wide, non-recurrent vector — `escapepod_demux::crf::encoder_native`'s
/// linear head is the first, 1024 moves at every one of 300 timesteps — does
/// not pay scalar `f32::tanh` (a full libm call) per element: measured there
/// as ~10 ms of a read's ~40 ms total, the same order as the whole LSTM
/// stack, for what is otherwise a small GEMM. Uses the same Cephes-style
/// vector `tanh` the recurrence kernels use, so it is not bit-identical to
/// [`f32::tanh`] — the contract is the caller's own tolerance, as for the
/// recurrence.
pub fn tanh_slice(backend: LstmBackend, x: &mut [f32]) {
    match backend {
        LstmBackend::Scalar => {
            for v in x.iter_mut() {
                *v = v.tanh();
            }
        }
        #[cfg(target_arch = "x86_64")]
        // Safety: `backend` was constructed by `LstmBackend::best_for` or
        // `with_backend`, both of which only produce a variant this machine
        // supports.
        LstmBackend::Avx2 | LstmBackend::Avx512 => unsafe { tanh_slice_avx2(x) },
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn tanh_slice_avx2(x: &mut [f32]) {
    use std::arch::x86_64::*;
    let (chunks, remainder) = x.as_chunks_mut::<8>();
    for c in chunks {
        // Safety: `as_chunks_mut::<8>()` guarantees each `c` is exactly 8
        // `f32`s, so the load/store below stay in bounds; the target feature
        // is guaranteed by the caller.
        unsafe {
            let v = _mm256_loadu_ps(c.as_ptr());
            _mm256_storeu_ps(c.as_mut_ptr(), tanh8(v));
        }
    }
    for v in remainder {
        *v = v.tanh();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A deterministic xorshift stream in `[-scale, scale)`.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self, scale: f32) -> f32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            ((self.0 as u32 as f32) / (u32::MAX as f32) * 2.0 - 1.0) * scale
        }
    }

    fn weights(n_in: usize, h: usize, seed: u64) -> LstmWeights {
        let mut rng = Rng(seed | 1);
        let g = 4 * h;
        let w: Vec<f32> = (0..g * n_in).map(|_| rng.next(0.5)).collect();
        let r: Vec<f32> = (0..g * h).map(|_| rng.next(0.3)).collect();
        let b: Vec<f32> = (0..8 * h).map(|_| rng.next(0.2)).collect();
        LstmWeights::from_onnx(n_in, h, &w, &r, &b)
    }

    fn input(n_in: usize, seq: usize, seed: u64) -> Vec<f32> {
        let mut rng = Rng(seed | 1);
        (0..n_in * seq).map(|_| rng.next(1.5)).collect()
    }

    #[cfg(target_arch = "x86_64")]
    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(p, q)| (p - q).abs())
            .fold(0.0, f32::max)
    }

    /// A plain, allocation-heavy reimplementation of the recurrence used only
    /// as a reference for these tests — not the kernel under test.
    #[allow(clippy::needless_range_loop)]
    fn reference(w: &LstmWeights, seq: usize, reverse: bool, x: &[f32]) -> Vec<f32> {
        let (h, g) = (w.h, 4 * w.h);
        let mut hs = vec![0.0f32; seq * h];
        let mut hcur = vec![0.0f32; h];
        let mut c = vec![0.0f32; h];
        for step in 0..seq {
            let t = if reverse { seq - 1 - step } else { step };
            let mut gates = w.bias.clone();
            for k in 0..w.n_in {
                let xk = x[k * seq + t];
                for gi in 0..g {
                    gates[gi] += xk * w.wt[k * g + gi];
                }
            }
            if step > 0 {
                for j in 0..h {
                    for gi in 0..g {
                        gates[gi] += hcur[j] * w.rt[j * g + gi];
                    }
                }
            }
            for u in 0..h {
                let i = sigmoid(gates[u]);
                let o = sigmoid(gates[h + u]);
                let f = sigmoid(gates[2 * h + u]);
                let ct = gates[3 * h + u].tanh();
                c[u] = f * c[u] + i * ct;
                hcur[u] = o * c[u].tanh();
            }
            hs[t * h..(t + 1) * h].copy_from_slice(&hcur);
        }
        hs
    }

    #[test]
    fn scalar_matches_the_reference_both_directions() {
        let (n_in, seq, h) = (4, 33, 96);
        let w = weights(n_in, h, 7);
        for reverse in [false, true] {
            let x = input(n_in, seq, 3);
            let want = reference(&w, seq, reverse, &x);
            let mut xw = vec![0.0f32; seq * 4 * h];
            let mut hs = vec![0.0f32; seq * h];
            let mut gates = vec![0.0f32; 4 * h];
            run_scalar(&w, seq, reverse, &x, &mut xw, &mut hs, h, 0, &mut gates);
            assert_eq!(hs, want, "reverse={reverse}");
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_matches_scalar() {
        if !LstmBackend::Avx2.supported() {
            eprintln!("no AVX2 + FMA on this machine: avx2-vs-scalar pin not exercised");
            return;
        }
        let (n_in, seq, h) = (4, 33, 96);
        let w = weights(n_in, h, 11);
        for reverse in [false, true] {
            let x = input(n_in, seq, 5);
            let mut xw_s = vec![0.0f32; seq * 4 * h];
            let mut hs_s = vec![0.0f32; seq * h];
            let mut gates_s = vec![0.0f32; 4 * h];
            run_scalar(
                &w,
                seq,
                reverse,
                &x,
                &mut xw_s,
                &mut hs_s,
                h,
                0,
                &mut gates_s,
            );
            let mut xw_v = vec![0.0f32; seq * 4 * h];
            let mut hs_v = vec![0.0f32; seq * h];
            let mut gates_v = vec![0.0f32; 4 * h];
            unsafe {
                run_avx2(
                    &w,
                    seq,
                    reverse,
                    &x,
                    &mut xw_v,
                    &mut hs_v,
                    h,
                    0,
                    &mut gates_v,
                );
            }
            let d = max_abs_diff(&hs_s, &hs_v);
            assert!(d < 1e-4, "reverse={reverse}: {d:e}");
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn batched_matches_single_bit_for_bit() {
        let (n_in, seq, h) = (4, 33, 96);
        let w = weights(n_in, h, 21);
        for backend in LstmBackend::available() {
            for reverse in [false, true] {
                // Every width the dispatch has an explicit arm for, on this
                // backend — a caller batching past `preferred_batch()` opens
                // a new group instead of asking one kernel call to cover it,
                // so this only exercises widths a single call actually
                // supports (`run_scalar_batch` is exempt below: it has no
                // register-file limit, so it batches by looping).
                for n_reads in 1..=backend.preferred_batch() {
                    let inputs: Vec<Vec<f32>> = (1..=n_reads as u64)
                        .map(|s| input(n_in, seq, 40 + s))
                        .collect();
                    let refs: Vec<&[f32]> = inputs.iter().map(Vec::as_slice).collect();
                    let mut hs_single = vec![0.0f32; n_reads * seq * h];
                    for (r, x) in inputs.iter().enumerate() {
                        let mut xw = vec![0.0f32; seq * 4 * h];
                        let mut gates = vec![0.0f32; 4 * h];
                        let dst = &mut hs_single[r * seq * h..(r + 1) * seq * h];
                        match backend {
                            LstmBackend::Scalar => {
                                run_scalar(&w, seq, reverse, x, &mut xw, dst, h, 0, &mut gates)
                            }
                            LstmBackend::Avx2 => unsafe {
                                run_avx2(&w, seq, reverse, x, &mut xw, dst, h, 0, &mut gates)
                            },
                            LstmBackend::Avx512 => unsafe {
                                run_avx2(&w, seq, reverse, x, &mut xw, dst, h, 0, &mut gates)
                            },
                        }
                    }
                    let mut hs_batch = vec![0.0f32; n_reads * seq * h];
                    let mut xw = vec![0.0f32; n_reads * seq * 4 * h];
                    let mut gates = vec![0.0f32; n_reads * 4 * h];
                    match (backend, n_reads) {
                        (LstmBackend::Avx512, 8) => unsafe {
                            run_avx512_batch::<8>(
                                &w,
                                seq,
                                reverse,
                                &refs,
                                &mut xw,
                                &mut hs_batch,
                                h,
                                0,
                                &mut gates,
                            )
                        },
                        (LstmBackend::Avx512, 7) => unsafe {
                            run_avx512_batch::<7>(
                                &w,
                                seq,
                                reverse,
                                &refs,
                                &mut xw,
                                &mut hs_batch,
                                h,
                                0,
                                &mut gates,
                            )
                        },
                        (LstmBackend::Avx512, 6) => unsafe {
                            run_avx512_batch::<6>(
                                &w,
                                seq,
                                reverse,
                                &refs,
                                &mut xw,
                                &mut hs_batch,
                                h,
                                0,
                                &mut gates,
                            )
                        },
                        (LstmBackend::Avx512, 5) => unsafe {
                            run_avx512_batch::<5>(
                                &w,
                                seq,
                                reverse,
                                &refs,
                                &mut xw,
                                &mut hs_batch,
                                h,
                                0,
                                &mut gates,
                            )
                        },
                        (LstmBackend::Avx512, 4) => unsafe {
                            run_avx512_batch::<4>(
                                &w,
                                seq,
                                reverse,
                                &refs,
                                &mut xw,
                                &mut hs_batch,
                                h,
                                0,
                                &mut gates,
                            )
                        },
                        (LstmBackend::Avx512, 3) => unsafe {
                            run_avx512_batch::<3>(
                                &w,
                                seq,
                                reverse,
                                &refs,
                                &mut xw,
                                &mut hs_batch,
                                h,
                                0,
                                &mut gates,
                            )
                        },
                        (LstmBackend::Avx512, 2) => unsafe {
                            run_avx512_batch::<2>(
                                &w,
                                seq,
                                reverse,
                                &refs,
                                &mut xw,
                                &mut hs_batch,
                                h,
                                0,
                                &mut gates,
                            )
                        },
                        (LstmBackend::Avx2, 3) => unsafe {
                            run_avx2_batch::<3>(
                                &w,
                                seq,
                                reverse,
                                &refs,
                                &mut xw,
                                &mut hs_batch,
                                h,
                                0,
                                &mut gates,
                            )
                        },
                        (LstmBackend::Avx2, 2) => unsafe {
                            run_avx2_batch::<2>(
                                &w,
                                seq,
                                reverse,
                                &refs,
                                &mut xw,
                                &mut hs_batch,
                                h,
                                0,
                                &mut gates,
                            )
                        },
                        (LstmBackend::Avx2, _) | (LstmBackend::Avx512, _) => unsafe {
                            run_avx2(
                                &w,
                                seq,
                                reverse,
                                refs[0],
                                &mut xw[..seq * 4 * h],
                                &mut hs_batch[..seq * h],
                                h,
                                0,
                                &mut gates[..4 * h],
                            )
                        },
                        (LstmBackend::Scalar, _) => run_scalar_batch(
                            &w,
                            seq,
                            reverse,
                            &refs,
                            &mut xw,
                            &mut hs_batch,
                            h,
                            0,
                            &mut gates,
                        ),
                    }
                    assert_eq!(
                        hs_single.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        hs_batch.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        "{backend:?} reverse={reverse} n_reads={n_reads}"
                    );
                }
            }
        }
    }

    /// [`run_scalar_batch`] has no register-file width to cap at, so it is
    /// checked separately at a group size past every vector backend's
    /// `preferred_batch()`.
    #[test]
    fn scalar_batch_matches_single_past_every_vector_width() {
        let (n_in, seq, h) = (4, 33, 96);
        let w = weights(n_in, h, 22);
        for reverse in [false, true] {
            let inputs: Vec<Vec<f32>> = (1..=9u64).map(|s| input(n_in, seq, 80 + s)).collect();
            let refs: Vec<&[f32]> = inputs.iter().map(Vec::as_slice).collect();
            let mut xw = vec![0.0f32; refs.len() * seq * 4 * h];
            let mut hs_batch = vec![0.0f32; refs.len() * seq * h];
            let mut gates = vec![0.0f32; refs.len() * 4 * h];
            run_scalar_batch(
                &w,
                seq,
                reverse,
                &refs,
                &mut xw,
                &mut hs_batch,
                h,
                0,
                &mut gates,
            );
            for (r, x) in inputs.iter().enumerate() {
                let mut xw1 = vec![0.0f32; seq * 4 * h];
                let mut gates1 = vec![0.0f32; 4 * h];
                let mut hs1 = vec![0.0f32; seq * h];
                run_scalar(&w, seq, reverse, x, &mut xw1, &mut hs1, h, 0, &mut gates1);
                assert_eq!(hs1, hs_batch[r * seq * h..(r + 1) * seq * h], "read {r}");
            }
        }
    }
}
