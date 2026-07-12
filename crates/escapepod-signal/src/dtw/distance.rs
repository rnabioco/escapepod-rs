//! Core DTW distance computation with Sakoe-Chiba band constraint.

use ndarray::Array2;
use rayon::prelude::*;

/// Compute DTW distance between two sequences.
///
/// Uses the classic DTW recurrence relation:
/// `D[i,j] = dist(a[i], b[j]) + min(D[i-1,j], D[i,j-1], D[i-1,j-1])`
///
/// # Arguments
///
/// * `a` - First sequence
/// * `b` - Second sequence
/// * `window` - Optional Sakoe-Chiba band width. If `Some(w)`, only compute
///   distances where `|i - j| <= w`. This restricts the warping path
///   to a diagonal band and improves performance.
///
/// # Returns
///
/// The DTW distance between the two sequences.
///
/// # Example
///
/// ```
/// use escapepod_signal::dtw::dtw_distance;
///
/// let a = vec![1.0, 2.0, 3.0, 4.0];
/// let b = vec![1.0, 2.0, 3.0, 4.0];
/// let distance = dtw_distance(&a, &b, None);
/// assert_eq!(distance, 0.0);
/// ```
pub fn dtw_distance(a: &[f32], b: &[f32], window: Option<usize>) -> f32 {
    dtw_distance_bounded_penalty(a, b, window, f32::INFINITY, 0.0)
}

/// Compute DTW distance with a warping penalty.
///
/// Matches `dtaidistance`'s `penalty` parameter (used by WarpDemuX): a fixed
/// cost added to the two non-diagonal (expansion / compression) transitions of
/// the DP recurrence, so the alignment is biased toward the diagonal:
///
/// `D[i,j] = (a_i - b_j)^2 + min(D[i-1,j-1], D[i-1,j] + penalty^2, D[i,j-1] + penalty^2)`
///
/// `penalty` matches dtaidistance's parameter: it is given in the *non-squared*
/// distance space, so each warping step contributes `penalty^2` to this DP's
/// squared cumulative cost (the final distance is `sqrt(D[n,m])`). Verified
/// against `dtaidistance.dtw.distance(..., penalty=p, use_c=True)`. `penalty ==
/// 0.0` is bit-identical to [`dtw_distance`].
pub fn dtw_distance_penalty(a: &[f32], b: &[f32], window: Option<usize>, penalty: f32) -> f32 {
    dtw_distance_bounded_penalty(a, b, window, f32::INFINITY, penalty)
}

/// Compute DTW distance with early abandonment.
///
/// If the minimum distance in any row exceeds `upper_bound`, returns `f32::INFINITY`
/// early without completing the full computation. This is useful when searching for
/// the best match and we can skip candidates that can't beat the current best.
///
/// # Arguments
///
/// * `a` - First sequence
/// * `b` - Second sequence
/// * `window` - Optional Sakoe-Chiba band width
/// * `upper_bound` - If all values in a row exceed this, return early
///
/// # Returns
///
/// The DTW distance, or `f32::INFINITY` if early abandonment occurred.
pub fn dtw_distance_bounded(a: &[f32], b: &[f32], window: Option<usize>, upper_bound: f32) -> f32 {
    dtw_distance_bounded_penalty(a, b, window, upper_bound, 0.0)
}

/// Core DTW with both early abandonment and a warping `penalty`. See
/// [`dtw_distance_bounded`] and [`dtw_distance_penalty`]. The penalty is added
/// to the expansion (`prev[j]`) and compression (`curr[j-1]`) neighbors before
/// the `min`; with `penalty == 0.0` (`x + 0.0 == x` for finite and ±INF) the
/// arithmetic is identical to the no-penalty path.
pub fn dtw_distance_bounded_penalty(
    a: &[f32],
    b: &[f32],
    window: Option<usize>,
    upper_bound: f32,
    penalty: f32,
) -> f32 {
    let mut scratch = DtwScratch::new();
    dtw_distance_bounded_penalty_into(a, b, window, upper_bound, penalty, &mut scratch)
}

/// Reusable row buffers for [`dtw_distance_bounded_penalty_into`].
///
/// A hot loop computing many DTW distances against a fixed query (e.g. SVM
/// classify scoring a read against tens of thousands of training fingerprints)
/// otherwise heap-allocates four `Vec<f32>` *per call*. At fingerprint sizes
/// (length ~10–100) that per-call allocation + zero-fill rivals the actual DP
/// arithmetic. Hand one `DtwScratch` per worker and the buffers grow once to
/// their high-water mark and stay there.
#[derive(Default, Clone, Debug)]
pub struct DtwScratch {
    prev: Vec<f32>,
    curr: Vec<f32>,
    cost: Vec<f32>,
    prev_min: Vec<f32>,
}

impl DtwScratch {
    /// A fresh, empty scratch. Buffers grow lazily on first use.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Workspace-backed variant of [`dtw_distance_bounded_penalty`]. Reuses the
/// caller-owned row buffers in `scratch` instead of allocating per call.
/// Numerically bit-identical to the allocating version.
pub fn dtw_distance_bounded_penalty_into(
    a: &[f32],
    b: &[f32],
    window: Option<usize>,
    upper_bound: f32,
    penalty: f32,
    scratch: &mut DtwScratch,
) -> f32 {
    let n = a.len();
    let m = b.len();

    if n == 0 || m == 0 {
        return f32::INFINITY;
    }

    // Classical Sakoe-Chiba: the endpoint `(n, m)` itself has to lie in the
    // band. If `|n - m| > w` no alignment is possible and the DP would
    // otherwise return a stale value left over from an earlier in-band row.
    if let Some(w) = window
        && n.abs_diff(m) > w
    {
        return f32::INFINITY;
    }

    // Fast path for the dominant SVM-classify case: no Sakoe-Chiba band, no
    // warping penalty, no early-abandonment bound. A single fused pass needs
    // only the two row buffers (no `cost`/`prev_min` scratch) and drops all
    // per-row window/penalty branching. Bit-identical to the general path:
    // `min(min(prev[j-1], prev[j]), curr[j-1])` is the same as the two-pass
    // `min(prev_min, left)` with `pen_sq == 0`.
    if window.is_none() && penalty == 0.0 && upper_bound == f32::INFINITY {
        return dtw_fused_unconstrained(a, b, scratch);
    }

    // dtaidistance's `penalty` is expressed in the *non-squared* distance space,
    // but this DP accumulates squared local costs (`(a-b)^2`) and takes a single
    // `sqrt` at the end. To match, each warping step adds `penalty^2` to the
    // squared cumulative (verified: dtaidistance `distance([0,0,0],[0], penalty=p)`
    // == sqrt(2 * p^2)). `penalty == 0.0` keeps `pen_sq == 0.0`, a no-op.
    let pen_sq = penalty * penalty;

    // Two rows for memory efficiency (current and previous) plus two scratch
    // buffers that hold per-row precomputed values. The inner loop is split
    // into a vectorizable "precompute" pass (no loop-carried deps, LLVM
    // auto-vectorizes to AVX2) and a short serial "chain" pass that applies
    // the `curr[j-1]` left-neighbor dependency. Buffers are reused across
    // calls and fully (re)initialized here, so stale state never leaks in.
    let mut prev = std::mem::take(&mut scratch.prev);
    let mut curr = std::mem::take(&mut scratch.curr);
    reset_row(&mut prev, m + 1);
    reset_row(&mut curr, m + 1);
    scratch.cost.clear();
    scratch.cost.resize(m + 1, 0.0);
    scratch.prev_min.clear();
    scratch.prev_min.resize(m + 1, 0.0);
    let cost_buf = scratch.cost.as_mut_slice();
    let prev_min_buf = scratch.prev_min.as_mut_slice();
    prev[0] = 0.0;

    let result = (|| {
        for i in 1..=n {
            curr[0] = f32::INFINITY;

            // Determine the valid column range based on Sakoe-Chiba band
            let j_start = if let Some(w) = window {
                1.max(i.saturating_sub(w))
            } else {
                1
            };

            let j_end = if let Some(w) = window {
                m.min(i + w)
            } else {
                m
            };

            // Classical Sakoe-Chiba treats cells outside the band as unreachable
            // (INF). `curr[j_start - 1]` still holds whatever a prior row wrote,
            // which can be a finite value and would let the DP cheat by using an
            // out-of-band predecessor. Re-seed it to INF before we compute the
            // row so the `curr[j-1]` read below is correct for j == j_start.
            // Only needed when the inner loop actually runs and is reading from
            // a non-zero-indexed boundary.
            if j_start <= j_end && j_start > 1 {
                curr[j_start - 1] = f32::INFINITY;
            }

            if j_start > j_end {
                std::mem::swap(&mut prev, &mut curr);
                continue;
            }

            let ai = a[i - 1];
            let len = j_end - j_start + 1;

            // Pass 1 (vectorizable): cost[k] = (ai - b[j-1])^2,
            //                      prev_min[k] = min(prev[j-1], prev[j])
            // where k = j - j_start. No loop-carried deps on writes; LLVM
            // auto-vectorizes to AVX2 8-wide with -C target-cpu=x86-64-v3.
            {
                let b_slice = &b[j_start - 1..j_end];
                let prev_left = &prev[j_start - 1..j_end];
                let prev_right = &prev[j_start..=j_end];
                let cost = &mut cost_buf[..len];
                let pm = &mut prev_min_buf[..len];
                for k in 0..len {
                    let diff = ai - b_slice[k];
                    cost[k] = diff * diff;
                    // Diagonal (prev_left) takes no penalty; expansion (prev_right)
                    // does. `+ 0.0` is a no-op for the default penalty.
                    pm[k] = prev_left[k].min(prev_right[k] + pen_sq);
                }
            }

            // Pass 2 (serial): apply the `curr[j-1]` left-chain and update curr.
            //   curr[j] = cost[k] + min(prev_min[k], curr[j-1])
            // Only three fp ops per iteration: min, add, row_min update.
            let mut row_min = f32::INFINITY;
            let mut left = curr[j_start - 1];
            let cost = &cost_buf[..len];
            let pm = &prev_min_buf[..len];
            let out = &mut curr[j_start..=j_end];
            for k in 0..len {
                // Compression (left neighbor, `curr[j-1]`) takes the penalty too.
                let v = cost[k] + pm[k].min(left + pen_sq);
                out[k] = v;
                left = v;
                row_min = row_min.min(v);
            }

            // Early abandonment: if minimum in row exceeds bound, can't beat best
            if row_min > upper_bound {
                return f32::INFINITY;
            }

            std::mem::swap(&mut prev, &mut curr);
        }

        prev[m].sqrt()
    })();

    scratch.prev = prev;
    scratch.curr = curr;
    result
}

/// Fused single-pass DTW for the unconstrained case (`window == None`,
/// `penalty == 0`, no early-abandonment bound). Uses only the `prev`/`curr`
/// row buffers — no `cost`/`prev_min` scratch — and a single loop that folds
/// the three-neighbor `min` directly. This is the dominant SVM-classify path
/// (small fingerprints scored against a fixed query), where the two-pass
/// vectorized variant's extra buffer traffic and per-row setup outweigh its
/// SIMD benefit. Bit-identical to the general path for these parameters.
#[inline]
fn dtw_fused_unconstrained(a: &[f32], b: &[f32], scratch: &mut DtwScratch) -> f32 {
    let n = a.len();
    let m = b.len();

    let mut prev = std::mem::take(&mut scratch.prev);
    let mut curr = std::mem::take(&mut scratch.curr);
    if prev.len() < m + 1 {
        prev.resize(m + 1, f32::INFINITY);
    }
    if curr.len() < m + 1 {
        curr.resize(m + 1, f32::INFINITY);
    }

    // Seed the virtual row 0: prev[0] = 0, prev[1..=m] = INF. Only [0..=m] is
    // ever read, so leftover tail from a larger prior call is harmless.
    for x in prev[..=m].iter_mut() {
        *x = f32::INFINITY;
    }
    prev[0] = 0.0;

    for i in 1..=n {
        // Column-0 boundary D[i][0] = +INF for every row i >= 1. This must be
        // written every row (not just relied on from init): the buffer is
        // reused across calls, so a stale finite value here would become
        // `prev[0]` (the diagonal/up neighbor for j == 1) on the next row and
        // let the DP cheat. `left` carries the same INF in as curr[j-1] for
        // j == 1.
        curr[0] = f32::INFINITY;
        let ai = a[i - 1];
        let mut left = f32::INFINITY;
        for j in 1..=m {
            let diff = ai - b[j - 1];
            let best = prev[j - 1].min(prev[j]).min(left);
            let v = diff * diff + best;
            curr[j] = v;
            left = v;
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    let result = prev[m].sqrt();
    scratch.prev = prev;
    scratch.curr = curr;
    result
}

/// Reset a reused row buffer to `len` cells of `+INF`. Grows the buffer if it
/// is too short; never shrinks (keeps the high-water-mark capacity).
#[inline]
fn reset_row(buf: &mut Vec<f32>, len: usize) {
    if buf.len() < len {
        buf.resize(len, f32::INFINITY);
    }
    for x in buf[..len].iter_mut() {
        *x = f32::INFINITY;
    }
}

/// Number of training fingerprints scored per SIMD batch by
/// [`dtw_distances_batch_unconstrained`]. Sixteen `f32` lanes fill one AVX-512
/// `zmm` register; on the `-C target-cpu=x86-64-v3` (AVX2) baseline the same
/// block layout is consumed as two `ymm` registers per step.
pub const DTW_LANES: usize = 16;

/// Reusable row buffers for [`dtw_distances_batch_unconstrained`]. Each row
/// cell holds `DTW_LANES` independent DP states (one per training fingerprint
/// in the current batch).
#[derive(Default, Clone, Debug)]
pub struct DtwBatchScratch {
    prev: Vec<[f32; DTW_LANES]>,
    curr: Vec<[f32; DTW_LANES]>,
}

impl DtwBatchScratch {
    /// A fresh, empty scratch. Buffers grow lazily on first use.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Lane-parallel unconstrained DTW: compute `dtw_distance(query, training[k],
/// None)` for every `k` in `0..n_train`, evaluating `DTW_LANES` training
/// sequences at once down independent SIMD lanes. All training sequences must
/// have the same length `train_len`.
///
/// `train_blocks` is the training bank in a SIMD-friendly *structure of arrays*
/// layout: block `c` (training indices `c*DTW_LANES .. c*DTW_LANES+DTW_LANES`)
/// occupies `train_blocks[c*train_len*DTW_LANES ..]`, and within a block
/// element `[j*DTW_LANES + lane]` is `training[c*DTW_LANES + lane][j]`. The
/// trailing lanes of the final block (when `n_train` is not a multiple of
/// `DTW_LANES`) may hold arbitrary values — their distances are simply not
/// written.
///
/// `out` receives exactly `n_train` distances in training-index order, each
/// **bit-identical** to the scalar [`dtw_distance`] for the same pair: the
/// per-lane body is the same fused recurrence, and on the (square-distance +
/// `+INF`) value domain here the `<`-fold and `f32::min` select equal bits.
///
/// Dispatch: the default path is the AVX2 baseline — the 16-lane block feeds
/// two independent `ymm` chains per step, which (measured) beats a single
/// 512-bit `zmm` chain. The DP carries a serial `left` dependency along each
/// row, so throughput is bound by how many *independent* lane-chains stay in
/// flight; 2×`ymm` exposes more instruction-level parallelism than 1×`zmm`.
/// AVX-512 was ~22% slower on Cascade Lake (rna, where 512-bit ops also
/// downclock) and still ~11% slower on Emerald Rapids (gpu node, ~no
/// frequency penalty) — i.e. the loss is mostly ILP/port throughput, not just
/// downclocking. The AVX-512 kernel is therefore **opt-in** via
/// `ESCAPEPOD_DTW_AVX512=1` and off by default. The choice is a single cached
/// CPUID + env probe, so the same binary serves every node.
pub fn dtw_distances_batch_unconstrained(
    query: &[f32],
    train_blocks: &[f32],
    train_len: usize,
    n_train: usize,
    out: &mut Vec<f64>,
    scratch: &mut DtwBatchScratch,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if use_avx512() {
            // SAFETY: `use_avx512()` returned true, so `avx512f` is supported
            // on this CPU; that is the only requirement of the target-feature
            // function.
            unsafe {
                dtw_batch_avx512(query, train_blocks, train_len, n_train, out, scratch);
            }
            return;
        }
    }
    dtw_batch_kernel(query, train_blocks, train_len, n_train, out, scratch);
}

/// Cached dispatch decision: AVX-512 is used only when the CPU supports
/// `avx512f` *and* the caller opted in via `ESCAPEPOD_DTW_AVX512=1`. Off by
/// default because the 512-bit path measured slower than the AVX2 baseline on
/// every cluster CPU tested (Cascade Lake −22%, Emerald Rapids −11%).
/// `is_x86_feature_detected!` caches its own CPUID probe; the `OnceLock` also
/// folds in the env check so neither runs per call.
#[cfg(target_arch = "x86_64")]
fn use_avx512() -> bool {
    use std::sync::OnceLock;
    static USE: OnceLock<bool> = OnceLock::new();
    *USE.get_or_init(|| {
        std::env::var_os("ESCAPEPOD_DTW_AVX512").is_some_and(|v| v == "1")
            && std::arch::is_x86_feature_detected!("avx512f")
    })
}

/// AVX-512 entry point: identical body to the baseline, but compiled with
/// `avx512f` enabled so the inlined kernel autovectorizes to 512-bit `zmm`
/// (16 lanes per register). Only ever reached via [`use_avx512`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn dtw_batch_avx512(
    query: &[f32],
    train_blocks: &[f32],
    train_len: usize,
    n_train: usize,
    out: &mut Vec<f64>,
    scratch: &mut DtwBatchScratch,
) {
    dtw_batch_kernel(query, train_blocks, train_len, n_train, out, scratch);
}

/// The lane-parallel DTW body, shared by the baseline and AVX-512 entry points.
/// `#[inline(always)]` so that when it is inlined into the `avx512f` wrapper it
/// is recompiled with that feature and the `for lane in 0..DTW_LANES` loop
/// lowers to `zmm`; inlined into the baseline it stays at AVX2 (`ymm`).
#[inline(always)]
fn dtw_batch_kernel(
    query: &[f32],
    train_blocks: &[f32],
    train_len: usize,
    n_train: usize,
    out: &mut Vec<f64>,
    scratch: &mut DtwBatchScratch,
) {
    out.clear();
    if n_train == 0 {
        return;
    }
    out.reserve(n_train);

    let n = query.len();
    let m = train_len;
    // Match `dtw_distance`'s empty-input contract (returns +INF).
    if n == 0 || m == 0 {
        out.resize(n_train, f32::INFINITY as f64);
        return;
    }

    let n_blocks = n_train.div_ceil(DTW_LANES);
    if scratch.prev.len() < m + 1 {
        scratch.prev.resize(m + 1, [f32::INFINITY; DTW_LANES]);
    }
    if scratch.curr.len() < m + 1 {
        scratch.curr.resize(m + 1, [f32::INFINITY; DTW_LANES]);
    }
    let mut prev = std::mem::take(&mut scratch.prev);
    let mut curr = std::mem::take(&mut scratch.curr);

    for c in 0..n_blocks {
        let block = &train_blocks[c * m * DTW_LANES..(c + 1) * m * DTW_LANES];

        // Seed virtual row 0 for all lanes: prev[0] = 0, prev[1..=m] = +INF.
        prev[0] = [0.0; DTW_LANES];
        for p in prev[1..=m].iter_mut() {
            *p = [f32::INFINITY; DTW_LANES];
        }

        for i in 1..=n {
            let ai = query[i - 1];
            curr[0] = [f32::INFINITY; DTW_LANES];
            let mut left = [f32::INFINITY; DTW_LANES];
            for j in 1..=m {
                let pl = prev[j - 1];
                let pr = prev[j];
                let bj = &block[(j - 1) * DTW_LANES..j * DTW_LANES];
                let mut row = [0.0f32; DTW_LANES];
                // Independent across lanes → autovectorizes to one vector op per
                // arithmetic step (zmm under avx512f, ymm×2 at baseline). No NaN
                // can arise (squared diffs + ±INF), so the `<`-fold matches
                // `f32::min` bit-for-bit.
                for lane in 0..DTW_LANES {
                    let d = ai - bj[lane];
                    let mut best = pl[lane];
                    if pr[lane] < best {
                        best = pr[lane];
                    }
                    if left[lane] < best {
                        best = left[lane];
                    }
                    row[lane] = d * d + best;
                }
                curr[j] = row;
                left = row;
            }
            std::mem::swap(&mut prev, &mut curr);
        }

        let valid = if c + 1 < n_blocks {
            DTW_LANES
        } else {
            n_train - c * DTW_LANES
        };
        let last = prev[m];
        for &v in last.iter().take(valid) {
            out.push(v.sqrt() as f64);
        }
    }

    scratch.prev = prev;
    scratch.curr = curr;
}

/// Lane-parallel DTW for an arbitrary Sakoe-Chiba `window` and warping
/// `penalty` — the general form used by real WarpDemuX-config DTW-SVM models
/// (which carry e.g. `window=15, penalty=0.1`). Same SoA `train_blocks` layout
/// and bit-identical-to-scalar contract as
/// [`dtw_distances_batch_unconstrained`], to which it fast-paths when
/// `window == None && penalty == 0.0`.
///
/// All training sequences in a batch share the query length `n` and the
/// uniform `train_len = m`, so the band geometry (`[j_start, j_end]` per row)
/// and the feasibility test `|n - m| > w` are identical across lanes — only the
/// per-cell `(a_i - b_j)^2` and the three-way `min` differ between lanes.
#[allow(clippy::too_many_arguments)]
pub fn dtw_distances_batch(
    query: &[f32],
    train_blocks: &[f32],
    train_len: usize,
    n_train: usize,
    window: Option<usize>,
    penalty: f32,
    out: &mut Vec<f64>,
    scratch: &mut DtwBatchScratch,
) {
    if window.is_none() && penalty == 0.0 {
        dtw_distances_batch_unconstrained(query, train_blocks, train_len, n_train, out, scratch);
        return;
    }
    dtw_batch_kernel_windowed(
        query,
        train_blocks,
        train_len,
        n_train,
        window,
        penalty,
        out,
        scratch,
    );
}

/// Banded + penalized lane-parallel DTW body. Mirrors the scalar general path
/// in [`dtw_distance_bounded_penalty_into`] cell-for-cell (full `+INF` re-init
/// of both rows per block, `curr[0]` and the band's left boundary re-seeded to
/// `+INF` each row, diagonal step un-penalized, expansion/compression steps
/// charged `penalty^2`), so each lane is bit-identical to the scalar distance.
#[allow(clippy::too_many_arguments)]
fn dtw_batch_kernel_windowed(
    query: &[f32],
    train_blocks: &[f32],
    train_len: usize,
    n_train: usize,
    window: Option<usize>,
    penalty: f32,
    out: &mut Vec<f64>,
    scratch: &mut DtwBatchScratch,
) {
    out.clear();
    if n_train == 0 {
        return;
    }
    out.reserve(n_train);

    let n = query.len();
    let m = train_len;
    if n == 0 || m == 0 {
        out.resize(n_train, f32::INFINITY as f64);
        return;
    }

    let pen_sq = penalty * penalty;
    let inf = [f32::INFINITY; DTW_LANES];

    // The endpoint must lie in the band; if not, every lane's distance is +INF
    // (no feasible alignment) — identical across lanes since n, m, w are shared.
    let infeasible = matches!(window, Some(w) if n.abs_diff(m) > w);
    if infeasible {
        out.resize(n_train, f32::INFINITY as f64);
        return;
    }

    let n_blocks = n_train.div_ceil(DTW_LANES);
    if scratch.prev.len() < m + 1 {
        scratch.prev.resize(m + 1, inf);
    }
    if scratch.curr.len() < m + 1 {
        scratch.curr.resize(m + 1, inf);
    }
    let mut prev = std::mem::take(&mut scratch.prev);
    let mut curr = std::mem::take(&mut scratch.curr);

    for c in 0..n_blocks {
        let block = &train_blocks[c * m * DTW_LANES..(c + 1) * m * DTW_LANES];

        // Full re-init of both rows (the band leaves cells unwritten; they must
        // read as +INF, exactly as the scalar path's fresh buffers do).
        for cell in prev[..=m].iter_mut() {
            *cell = inf;
        }
        for cell in curr[..=m].iter_mut() {
            *cell = inf;
        }
        prev[0] = [0.0; DTW_LANES];

        for i in 1..=n {
            curr[0] = inf;
            let j_start = match window {
                Some(w) => 1.max(i.saturating_sub(w)),
                None => 1,
            };
            let j_end = match window {
                Some(w) => m.min(i + w),
                None => m,
            };
            if j_start <= j_end && j_start > 1 {
                // Re-seed the left boundary so j == j_start can't read a stale
                // in-band value from an earlier row.
                curr[j_start - 1] = inf;
            }
            if j_start > j_end {
                std::mem::swap(&mut prev, &mut curr);
                continue;
            }

            let ai = query[i - 1];
            // `left` carries curr[j-1]; at j == j_start it is the (re-seeded
            // or boundary) +INF cell.
            let mut left = curr[j_start - 1];
            for j in j_start..=j_end {
                let pl = prev[j - 1]; // diagonal (no penalty)
                let pr = prev[j]; // expansion (penalty)
                let bj = &block[(j - 1) * DTW_LANES..j * DTW_LANES];
                let mut row = [0.0f32; DTW_LANES];
                for lane in 0..DTW_LANES {
                    let d = ai - bj[lane];
                    let mut best = pl[lane];
                    let up = pr[lane] + pen_sq;
                    if up < best {
                        best = up;
                    }
                    let lf = left[lane] + pen_sq; // compression (penalty)
                    if lf < best {
                        best = lf;
                    }
                    row[lane] = d * d + best;
                }
                curr[j] = row;
                left = row;
            }
            std::mem::swap(&mut prev, &mut curr);
        }

        let valid = if c + 1 < n_blocks {
            DTW_LANES
        } else {
            n_train - c * DTW_LANES
        };
        let last = prev[m];
        for &v in last.iter().take(valid) {
            out.push(v.sqrt() as f64);
        }
    }

    scratch.prev = prev;
    scratch.curr = curr;
}

/// Pack a uniform-length training bank into the SoA block layout consumed by
/// [`dtw_distances_batch_unconstrained`]. Returns the flat buffer; padding
/// lanes in the final block are zero-filled (their distances go unused).
pub fn pack_training_blocks(training: &[Vec<f32>], train_len: usize) -> Vec<f32> {
    let n_train = training.len();
    let n_blocks = n_train.div_ceil(DTW_LANES);
    let mut blocks = vec![0.0f32; n_blocks * train_len * DTW_LANES];
    for (k, fp) in training.iter().enumerate() {
        let c = k / DTW_LANES;
        let lane = k % DTW_LANES;
        let base = c * train_len * DTW_LANES;
        for (j, &v) in fp.iter().take(train_len).enumerate() {
            blocks[base + j * DTW_LANES + lane] = v;
        }
    }
    blocks
}

/// Compute the full DTW distance matrix between query sequences and reference sequences.
///
/// This function computes pairwise DTW distances between all query sequences and all
/// reference sequences in parallel using rayon.
///
/// # Arguments
///
/// * `queries` - Slice of query sequences
/// * `references` - Slice of reference sequences
/// * `window` - Optional Sakoe-Chiba band width
///
/// # Returns
///
/// A 2D array where `result[i, j]` is the DTW distance between `queries[i]` and `references[j]`.
///
/// # Example
///
/// ```
/// use escapepod_signal::dtw::dtw_distance_matrix;
///
/// let queries = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
/// let references = vec![vec![1.0, 2.0, 3.0], vec![7.0, 8.0, 9.0]];
/// let matrix = dtw_distance_matrix(&queries, &references, None);
/// assert_eq!(matrix.shape(), &[2, 2]);
/// ```
pub fn dtw_distance_matrix<Q, R>(
    queries: &[Q],
    references: &[R],
    window: Option<usize>,
) -> Array2<f32>
where
    Q: AsRef<[f32]> + Sync,
    R: AsRef<[f32]> + Sync,
{
    let n_queries = queries.len();
    let n_refs = references.len();

    // Flat row-major buffer; rayon writes each row in parallel. The previous
    // `flat_map(... .collect::<Vec<_>>())` allocated n_queries temporary Vecs
    // just to be flattened and dropped.
    let mut distances = vec![0.0f32; n_queries * n_refs];
    distances
        .par_chunks_mut(n_refs)
        .enumerate()
        .for_each(|(i, row)| {
            let q = queries[i].as_ref();
            for (j, slot) in row.iter_mut().enumerate() {
                *slot = dtw_distance(q, references[j].as_ref(), window);
            }
        });

    Array2::from_shape_vec((n_queries, n_refs), distances)
        .expect("Failed to create distance matrix")
}

/// Compute DTW distance matrix with block-based parallelization.
///
/// This divides the distance matrix into blocks and computes them in parallel,
/// which can be more efficient for very large matrices.
///
/// # Arguments
///
/// * `queries` - Slice of query sequences
/// * `references` - Slice of reference sequences
/// * `window` - Optional Sakoe-Chiba band width
/// * `block_size` - Size of blocks for parallel computation
///
/// # Returns
///
/// A 2D array where `result[i, j]` is the DTW distance between `queries[i]` and `references[j]`.
pub fn dtw_distance_matrix_blocked(
    queries: &[Vec<f32>],
    references: &[Vec<f32>],
    window: Option<usize>,
    block_size: usize,
) -> Array2<f32> {
    let n_queries = queries.len();
    let n_refs = references.len();
    let bs = block_size.max(1);

    // One row-major result buffer written in place: no per-block `Array2` to
    // allocate and no separate reassembly pass. Rows are partitioned into
    // contiguous blocks of `bs` rows so each rayon task owns a disjoint slice
    // of `data`; within a task the columns are still walked in cache-friendly
    // blocks of `bs`.
    let stride = n_refs.max(1);
    let mut data = vec![0.0f32; n_queries * n_refs];
    data.par_chunks_mut(bs * stride)
        .enumerate()
        .for_each(|(blk, rows)| {
            let i_start = blk * bs;
            let n_rows = rows.len() / stride;
            for j_start in (0..n_refs).step_by(bs) {
                let j_end = (j_start + bs).min(n_refs);
                for r in 0..n_rows {
                    let q = queries[i_start + r].as_slice();
                    let row = &mut rows[r * n_refs..(r + 1) * n_refs];
                    for j in j_start..j_end {
                        row[j] = dtw_distance(q, references[j].as_slice(), window);
                    }
                }
            }
        });

    Array2::from_shape_vec((n_queries, n_refs), data).expect("Failed to create distance matrix")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtw_identical_sequences() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let distance = dtw_distance(&a, &b, None);
        assert_eq!(distance, 0.0);
    }

    #[test]
    fn test_dtw_symmetric() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![2.0, 3.0, 4.0];
        let d1 = dtw_distance(&a, &b, None);
        let d2 = dtw_distance(&b, &a, None);
        assert_eq!(d1, d2);
    }

    #[test]
    fn test_dtw_known_distance() {
        // Simple case: [0] vs [1] should give distance of 1
        let a = vec![0.0];
        let b = vec![1.0];
        let distance = dtw_distance(&a, &b, None);
        assert_eq!(distance, 1.0);

        // [0, 0] vs [1, 1]: accumulated squared cost = 1+1 = 2, sqrt(2)
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 1.0];
        let distance = dtw_distance(&a, &b, None);
        assert!((distance - 2.0_f32.sqrt()).abs() < 1e-6);
    }

    #[test]
    fn test_dtw_with_window() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        // With window constraint, identical sequences should still have distance 0
        let distance = dtw_distance(&a, &b, Some(2));
        assert_eq!(distance, 0.0);
    }

    #[test]
    fn test_dtw_empty_sequences() {
        let a: Vec<f32> = vec![];
        let b = vec![1.0, 2.0];
        let distance = dtw_distance(&a, &b, None);
        assert!(distance.is_infinite());

        let a = vec![1.0, 2.0];
        let b: Vec<f32> = vec![];
        let distance = dtw_distance(&a, &b, None);
        assert!(distance.is_infinite());
    }

    #[test]
    fn test_dtw_distance_matrix() {
        let queries = vec![vec![1.0, 2.0, 3.0], vec![2.0, 3.0, 4.0]];
        let references = vec![vec![1.0, 2.0, 3.0], vec![2.0, 3.0, 4.0]];

        let matrix = dtw_distance_matrix(&queries, &references, None);

        assert_eq!(matrix.shape(), &[2, 2]);
        // Diagonal should be zero (identical sequences)
        assert_eq!(matrix[[0, 0]], 0.0);
        assert_eq!(matrix[[1, 1]], 0.0);
        // Matrix should be symmetric
        assert_eq!(matrix[[0, 1]], matrix[[1, 0]]);
    }

    #[test]
    fn test_dtw_distance_matrix_blocked() {
        let queries = vec![
            vec![1.0, 2.0, 3.0],
            vec![2.0, 3.0, 4.0],
            vec![3.0, 4.0, 5.0],
        ];
        let references = vec![
            vec![1.0, 2.0, 3.0],
            vec![2.0, 3.0, 4.0],
            vec![3.0, 4.0, 5.0],
        ];

        let matrix1 = dtw_distance_matrix(&queries, &references, None);
        let matrix2 = dtw_distance_matrix_blocked(&queries, &references, None, 2);

        // Both methods should produce the same result
        assert_eq!(matrix1.shape(), matrix2.shape());
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(matrix1[[i, j]], matrix2[[i, j]]);
            }
        }
    }

    #[test]
    fn test_dtw_alignment_stretch() {
        // Test that DTW can handle sequences of different lengths
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 1.5, 2.0, 2.5, 3.0];

        let distance = dtw_distance(&a, &b, None);
        // Should be able to align despite length difference
        assert!(distance < f32::INFINITY);
        assert!(distance >= 0.0);
    }

    #[test]
    fn test_dtw_bounded_no_abandonment() {
        // With high upper bound, should return same result as unbounded
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 3.0, 4.0, 5.0];

        let unbounded = dtw_distance(&a, &b, None);
        let bounded = dtw_distance_bounded(&a, &b, None, 100.0);
        assert_eq!(unbounded, bounded);
    }

    #[test]
    fn test_dtw_bounded_early_abandonment() {
        // With very low upper bound, should abandon early
        let a = vec![0.0, 0.0, 0.0, 0.0];
        let b = vec![10.0, 10.0, 10.0, 10.0];

        // The actual distance would be 40, so bound of 5 should cause abandonment
        let bounded = dtw_distance_bounded(&a, &b, None, 5.0);
        assert!(bounded.is_infinite());
    }

    #[test]
    fn test_dtw_penalty_zero_matches_unpenalized() {
        // penalty == 0.0 must be bit-identical to the no-penalty path.
        let a = vec![1.0, 2.0, 3.0, 2.5, 4.0];
        let b = vec![1.0, 1.5, 3.0, 4.0];
        assert_eq!(
            dtw_distance(&a, &b, None),
            dtw_distance_penalty(&a, &b, None, 0.0)
        );
        assert_eq!(
            dtw_distance(&a, &b, Some(2)),
            dtw_distance_penalty(&a, &b, Some(2), 0.0)
        );
    }

    #[test]
    fn test_dtw_penalty_forced_warp() {
        // Aligning [0,0,0] to [0] forces two compression steps, each charged
        // `penalty^2` in the squared cumulative: D = 2*penalty^2, distance =
        // sqrt(2)*penalty. Matches dtaidistance: distance([0,0,0],[0],penalty=p).
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![0.0];
        assert!(dtw_distance_penalty(&a, &b, None, 0.0).abs() < 1e-6);
        // dtaidistance penalty=0.1 -> sqrt(2 * 0.1^2) = 0.14142...
        let d = dtw_distance_penalty(&a, &b, None, 0.1);
        let expected = (2.0f32 * 0.1 * 0.1).sqrt();
        assert!((d - expected).abs() < 1e-6, "expected {expected}, got {d}");
    }

    #[test]
    fn test_dtw_into_reused_scratch_matches_alloc() {
        // The workspace-backed path must be bit-identical to the allocating
        // path *and* a reused scratch must not leak state between calls. This
        // is the case the original unit tests missed: a fresh scratch hides a
        // stale-boundary bug because the first resize zero/INF-fills it.
        let cases: [(Vec<f32>, Vec<f32>); 5] = [
            (vec![1.0, 2.0, 3.0, 4.0], vec![1.0, 2.0, 3.0, 4.0]),
            (vec![0.0, 5.0, 2.0, 9.0, 1.0], vec![3.0, 3.0, 3.0]),
            (vec![2.0, 2.0], vec![9.0, 1.0, 4.0, 4.0, 7.0]),
            (vec![1.0], vec![1.0, 2.0, 3.0]),
            (vec![7.0, 7.0, 7.0, 7.0, 7.0], vec![7.0, 7.0, 7.0, 7.0, 7.0]),
        ];

        // One scratch reused across every (window, penalty, case) combination.
        let mut scratch = DtwScratch::new();
        for window in [None, Some(2), Some(4)] {
            for penalty in [0.0f32, 0.5, 2.0] {
                for (a, b) in &cases {
                    let want = dtw_distance_bounded_penalty(a, b, window, f32::INFINITY, penalty);
                    let got = dtw_distance_bounded_penalty_into(
                        a,
                        b,
                        window,
                        f32::INFINITY,
                        penalty,
                        &mut scratch,
                    );
                    assert_eq!(
                        want.to_bits(),
                        got.to_bits(),
                        "mismatch for window={window:?} penalty={penalty} a={a:?} b={b:?}",
                    );
                }
            }
        }
    }

    #[test]
    fn test_dtw_fused_path_reuse_no_leak() {
        // Specifically hammer the unconstrained fused path (window=None,
        // penalty=0) with a reused scratch interleaving different shapes.
        let mut scratch = DtwScratch::new();
        let pairs: [(Vec<f32>, Vec<f32>); 4] = [
            (vec![0.0, 0.0], vec![1.0, 1.0]),
            (vec![5.0, 4.0, 3.0, 2.0, 1.0], vec![1.0, 2.0, 3.0]),
            (vec![0.0, 0.0], vec![1.0, 1.0]),
            (vec![9.0], vec![9.0, 9.0, 9.0, 9.0]),
        ];
        for (a, b) in &pairs {
            let want = dtw_distance(a, b, None);
            let got =
                dtw_distance_bounded_penalty_into(a, b, None, f32::INFINITY, 0.0, &mut scratch);
            assert_eq!(want.to_bits(), got.to_bits(), "a={a:?} b={b:?}");
        }
    }

    #[test]
    fn test_dtw_batch_matches_scalar() {
        // The lane-parallel batch must be bit-identical to the scalar
        // `dtw_distance` for every training fingerprint, including the padded
        // final block (n_train not a multiple of DTW_LANES) and reused scratch.
        fn pseudo(seed: u64, i: usize) -> f32 {
            // Deterministic, no Date/rand: a cheap LCG-ish hash → small floats.
            let x = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(i as u64)
                .wrapping_mul(1442695040888963407);
            ((x >> 33) as f32 / u32::MAX as f32) * 4.0 - 2.0
        }

        let mut scratch = DtwBatchScratch::new();
        for &train_len in &[1usize, 4, 10, 17] {
            for &n_train in &[1usize, 7, 8, 9, 20, 41] {
                for &query_len in &[1usize, 6, 10, 25] {
                    let query: Vec<f32> = (0..query_len).map(|i| pseudo(11, i)).collect();
                    let training: Vec<Vec<f32>> = (0..n_train)
                        .map(|k| (0..train_len).map(|j| pseudo(100 + k as u64, j)).collect())
                        .collect();

                    let want: Vec<f32> = training
                        .iter()
                        .map(|t| dtw_distance(&query, t, None))
                        .collect();

                    let blocks = pack_training_blocks(&training, train_len);

                    // Check both the public dispatcher (which uses the AVX-512
                    // kernel on a capable node) and the baseline kernel directly
                    // (so the non-AVX-512 path is covered on the same machine).
                    let mut got_dispatch = Vec::new();
                    dtw_distances_batch_unconstrained(
                        &query,
                        &blocks,
                        train_len,
                        n_train,
                        &mut got_dispatch,
                        &mut scratch,
                    );
                    let mut got_baseline = Vec::new();
                    dtw_batch_kernel(
                        &query,
                        &blocks,
                        train_len,
                        n_train,
                        &mut got_baseline,
                        &mut scratch,
                    );

                    // Also exercise the AVX-512 kernel directly when the host
                    // supports it (the public dispatcher keeps it opt-in, so it
                    // wouldn't otherwise run in the default test config).
                    #[cfg(target_arch = "x86_64")]
                    let mut got_avx512 = Vec::new();
                    #[cfg(target_arch = "x86_64")]
                    if std::arch::is_x86_feature_detected!("avx512f") {
                        // SAFETY: gated by the runtime feature check above.
                        unsafe {
                            dtw_batch_avx512(
                                &query,
                                &blocks,
                                train_len,
                                n_train,
                                &mut got_avx512,
                                &mut scratch,
                            );
                        }
                    } else {
                        got_avx512 = got_baseline.clone();
                    }
                    #[cfg(not(target_arch = "x86_64"))]
                    let got_avx512 = got_baseline.clone();

                    assert_eq!(got_dispatch.len(), n_train);
                    assert_eq!(got_baseline.len(), n_train);
                    assert_eq!(got_avx512.len(), n_train);
                    for (k, &w) in want.iter().enumerate() {
                        for (path, &g) in [
                            ("dispatch", &got_dispatch[k]),
                            ("baseline", &got_baseline[k]),
                            ("avx512", &got_avx512[k]),
                        ] {
                            assert_eq!(
                                (w as f64).to_bits(),
                                g.to_bits(),
                                "{path} mismatch train_len={train_len} n_train={n_train} \
                                 query_len={query_len} k={k}: want {w} got {g}",
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_dtw_distances_batch_windowed_matches_scalar() {
        // The general (windowed + penalty) lane-parallel batch must be
        // bit-identical to the scalar `dtw_distance_bounded_penalty` for every
        // training fingerprint, across window/penalty settings, query/train
        // lengths, padded final blocks, and a reused scratch.
        fn pseudo(seed: u64, i: usize) -> f32 {
            let x = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(i as u64)
                .wrapping_mul(1442695040888963407);
            ((x >> 33) as f32 / u32::MAX as f32) * 4.0 - 2.0
        }

        let mut scratch = DtwBatchScratch::new();
        for &(window, penalty) in &[
            (None, 0.0f32),
            (None, 0.1),
            (Some(15usize), 0.1),
            (Some(3), 0.0),
            (Some(2), 0.5),
            (Some(0), 0.1),
        ] {
            for &train_len in &[1usize, 4, 10, 25] {
                for &n_train in &[1usize, 8, 9, 33] {
                    for &query_len in &[4usize, 10, 25] {
                        let query: Vec<f32> = (0..query_len).map(|i| pseudo(7, i)).collect();
                        let training: Vec<Vec<f32>> = (0..n_train)
                            .map(|k| (0..train_len).map(|j| pseudo(200 + k as u64, j)).collect())
                            .collect();

                        let want: Vec<f32> = training
                            .iter()
                            .map(|t| {
                                dtw_distance_bounded_penalty(
                                    &query,
                                    t,
                                    window,
                                    f32::INFINITY,
                                    penalty,
                                )
                            })
                            .collect();

                        let blocks = pack_training_blocks(&training, train_len);
                        let mut got = Vec::new();
                        dtw_distances_batch(
                            &query,
                            &blocks,
                            train_len,
                            n_train,
                            window,
                            penalty,
                            &mut got,
                            &mut scratch,
                        );

                        assert_eq!(got.len(), n_train);
                        for (k, (&w, &g)) in want.iter().zip(got.iter()).enumerate() {
                            assert_eq!(
                                (w as f64).to_bits(),
                                g.to_bits(),
                                "mismatch window={window:?} penalty={penalty} \
                                 train_len={train_len} n_train={n_train} \
                                 query_len={query_len} k={k}: want {w} got {g}",
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_dtw_bounded_exact_bound() {
        // Bound equal to actual distance should not abandon
        let a = vec![0.0];
        let b = vec![1.0];

        // Distance is exactly 1.0
        let bounded = dtw_distance_bounded(&a, &b, None, 1.0);
        assert_eq!(bounded, 1.0);

        // Just below should abandon
        let bounded_low = dtw_distance_bounded(&a, &b, None, 0.5);
        assert!(bounded_low.is_infinite());
    }
}
