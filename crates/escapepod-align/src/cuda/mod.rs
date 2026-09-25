// SPDX-License-Identifier: MIT

//! The CUDA score kernel (feature `gpu`): every read of a batch against every
//! reference of the panel, one GPU thread per pair, scores only.
//!
//! This replaces [`crate::Aligner::score_all`] for a batch and nothing else.
//! The [`ScoreMatrix`] it returns goes to [`crate::Aligner::map_reads_scored`],
//! which picks the tie set, traces the winners with the CPU pairs kernels and
//! computes `MD`/`NM` exactly as the CPU path does — so the output can only
//! differ if a score does, and the scores are pinned to [`crate::scalar::score`]
//! by equality (`tests/gpu_parity.rs`).
//!
//! Mirrors `escapepod_signal::dtw::cuda`: the kernel is a source string
//! compiled by NVRTC when the scorer is built, cudarc loads the driver and
//! NVRTC with `dlopen` (its `dynamic-loading` feature), so nothing CUDA is
//! needed at build time, and a machine without either gets an error from
//! [`GpuScorer::new`] rather than a panic.
//!
//! # What goes to the GPU, and what does not
//!
//! A query is scored here when it is non-empty, at most [`MAX_READ_LEN`]
//! bases, and inside the same [`fits_i16`] bound the SIMD kernels use (which
//! the kernel relies on for its packed `i16` boundary and result). Anything
//! else comes back as an unscored row and the caller's CPU kernel scores it,
//! in the same batch.
//!
//! The cap on read length is a *scheduling* bound, not a memory one: nothing
//! about a read is held in registers or shared memory beyond one 16-row
//! stripe, so a read of any length would fit. But one block owns a read for
//! its whole length, serially, and a batch finishes when its longest read
//! does; a real sample carries reads of 50–400 kb (0.5–6 s each on one block)
//! among ~10⁵ reads of ~130 nt. Past 4 kb — some 30× the median tRNA read —
//! a read costs one block longer than the whole rest of a 50 k-read batch is
//! likely to take, so it goes to the CPU instead, where `escpod align`
//! already gives such a read a chunk of its own.
//!
//! The cap on *reference* length is the shared-memory budget: the panel is
//! resident in shared memory at 4 bits a base, and a block of at least one
//! warp must fit in the 48 KB every CUDA device gives a block without opt-in —
//! 3,072 bases. A panel with a longer reference is refused by
//! [`GpuScorer::new`] (the tRNA panels this is built for are ~150 nt).

mod kernel;

use std::fmt;
use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, DriverError, LaunchConfig, PushKernelArg,
    sys::CUdevice_attribute,
};
use cudarc::nvrtc::{CompileError, compile_ptx};

use crate::mapper::ScoreMatrix;
use crate::panel::Panel;
use crate::scoring::{Mode, Scoring};
use crate::simd::fits_i16;

use kernel::STRIPE_ROWS;

/// Longest read the GPU scores; longer ones come back unscored (see the
/// module docs for why this is a scheduling bound).
pub const MAX_READ_LEN: usize = 4096;

/// Longest reference the kernel can hold in shared memory (see the module
/// docs): 48 KB for one warp at 4 bits a base.
pub const MAX_REF_LEN: usize = SMEM_BUDGET * 8 / 4 / 32;

/// Shared memory a block may use without opting in, on every CUDA device.
const SMEM_BUDGET: usize = 48 * 1024;

/// Most threads (references) per block.
const MAX_LANES: usize = 256;

/// Why the GPU scorer could not be built or run.
#[derive(Debug)]
pub enum GpuError {
    /// The CUDA driver or NVRTC library cannot be loaded on this machine.
    Unavailable(&'static str),
    /// A CUDA driver call failed: no device, out of memory, a launch failure.
    Driver(DriverError),
    /// NVRTC could not compile the kernel.
    Compile(CompileError),
    /// The panel has a reference the kernel cannot hold.
    ReferenceTooLong { len: usize, max: usize },
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unavailable(what) => write!(f, "{what} cannot be loaded"),
            Self::Driver(e) => write!(f, "CUDA driver error: {e}"),
            Self::Compile(e) => write!(f, "CUDA kernel compilation failed: {e}"),
            Self::ReferenceTooLong { len, max } => write!(
                f,
                "the panel's longest reference ({len} nt) exceeds the GPU kernel's \
                 {max} nt (the panel must fit in shared memory)"
            ),
        }
    }
}

impl std::error::Error for GpuError {}

impl From<DriverError> for GpuError {
    fn from(e: DriverError) -> Self {
        Self::Driver(e)
    }
}

impl From<CompileError> for GpuError {
    fn from(e: CompileError) -> Self {
        Self::Compile(e)
    }
}

/// 4-bit mask of an [`crate::alphabet`] code: A/C/G/T one bit each, a
/// wildcard all four, so "masks intersect" is `Scoring::substitution`'s
/// match rule.
#[inline]
fn mask(code: u8) -> u32 {
    if code < 4 { 1 << code } else { 0xF }
}

/// A panel resident on the device under one scoring and one mode, ready to
/// score batches of reads.
///
/// Build once per run: construction compiles the kernel (NVRTC) and uploads
/// the panel. [`GpuScorer::score_batch`] takes `&mut self` because every
/// launch reuses one scratch buffer; `escpod align` gives the scorer a
/// dedicated thread.
pub struct GpuScorer {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    func: CudaFunction,
    panel_dev: CudaSlice<u32>,
    ref_len_dev: CudaSlice<i32>,
    ref_idx_dev: CudaSlice<i32>,
    scratch_dev: CudaSlice<u32>,
    lanes: usize,
    n_groups: usize,
    grid_x: usize,
    words_per_lane: usize,
    n_refs: usize,
    max_ref_len: usize,
    scoring: Scoring,
    mode: Mode,
}

impl GpuScorer {
    /// Prepare `panel` on CUDA device 0.
    pub fn new(panel: &Panel, scoring: Scoring, mode: Mode) -> Result<Self, GpuError> {
        Self::new_on_device(0, panel, scoring, mode)
    }

    /// Prepare `panel` on CUDA device `ordinal`.
    pub fn new_on_device(
        ordinal: usize,
        panel: &Panel,
        scoring: Scoring,
        mode: Mode,
    ) -> Result<Self, GpuError> {
        // cudarc's dlopen wrappers panic when a library is missing, and the
        // release profile aborts on panic; ask first so the caller gets an
        // error it can report (or fall back from).
        // SAFETY: both only try to dlopen a library and report whether that
        // worked; no pointer from them is dereferenced.
        if !unsafe { cudarc::driver::sys::is_culib_present() } {
            return Err(GpuError::Unavailable("the CUDA driver library (libcuda)"));
        }
        if !unsafe { cudarc::nvrtc::sys::is_culib_present() } {
            return Err(GpuError::Unavailable("the CUDA NVRTC library (libnvrtc)"));
        }
        let max_ref_len = panel.max_len();
        if max_ref_len > MAX_REF_LEN {
            return Err(GpuError::ReferenceTooLong {
                len: max_ref_len,
                max: MAX_REF_LEN,
            });
        }
        let words_per_lane = max_ref_len.div_ceil(8);
        // One group if the panel fits a block, else as wide as shared memory allows.
        let mut lanes = panel.len().next_multiple_of(32).min(MAX_LANES);
        while lanes > 32 && words_per_lane * 4 * lanes > SMEM_BUDGET {
            lanes -= 32;
        }

        // Longest first, stable, as the SIMD profile does: a warp's threads
        // then have references of similar length and finish together.
        let mut order: Vec<usize> = (0..panel.len()).collect();
        order.sort_by_key(|&i| std::cmp::Reverse(panel.get(i).codes.len()));
        let n_groups = order.len().div_ceil(lanes);
        let mut words = vec![0u32; n_groups * words_per_lane * lanes];
        let mut ref_len = vec![0i32; n_groups * lanes];
        let mut ref_idx = vec![-1i32; n_groups * lanes];
        for (g, chunk) in order.chunks(lanes).enumerate() {
            let base = g * words_per_lane * lanes;
            for (lane, &ri) in chunk.iter().enumerate() {
                let codes = &panel.get(ri).codes;
                ref_len[g * lanes + lane] = codes.len() as i32;
                ref_idx[g * lanes + lane] = ri as i32;
                for (j, &c) in codes.iter().enumerate() {
                    words[base + (j / 8) * lanes + lane] |= mask(c) << (4 * (j % 8));
                }
            }
        }

        let ctx = CudaContext::new(ordinal)?;
        let stream = ctx.default_stream();
        let module = ctx.load_module(compile_ptx(kernel::KERNEL_SRC)?)?;
        let func = module.load_function(kernel::KERNEL_NAME)?;
        let smem = words_per_lane * 4 * lanes;
        let per_sm = func
            .occupancy_max_active_blocks_per_multiprocessor(lanes as u32, smem, None)?
            .max(1) as usize;
        let sms = ctx
            .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?
            .max(1) as usize;
        // Enough persistent blocks to fill every SM once, spread over the groups.
        let grid_x = (sms * per_sm).div_ceil(n_groups).max(1);
        let threads = grid_x * n_groups * lanes;

        let panel_dev = stream.clone_htod(&words)?;
        let ref_len_dev = stream.clone_htod(&ref_len)?;
        let ref_idx_dev = stream.clone_htod(&ref_idx)?;
        let scratch_dev = stream.alloc_zeros::<u32>(max_ref_len.max(1) * threads)?;

        Ok(Self {
            ctx,
            stream,
            func,
            panel_dev,
            ref_len_dev,
            ref_idx_dev,
            scratch_dev,
            lanes,
            n_groups,
            grid_x,
            words_per_lane,
            n_refs: panel.len(),
            max_ref_len,
            scoring,
            mode,
        })
    }

    /// The device's name, for the log.
    pub fn device_name(&self) -> String {
        self.ctx
            .name()
            .unwrap_or_else(|_| "unknown CUDA device".into())
    }

    /// How the kernel was laid out, for the log: `(threads per block,
    /// reference groups, persistent blocks per group)`.
    pub fn geometry(&self) -> (usize, usize, usize) {
        (self.lanes, self.n_groups, self.grid_x)
    }

    /// Whether [`GpuScorer::score_batch`] scores a query of `len` bases (the
    /// rest come back unscored, for the CPU).
    pub fn scores_len(&self, len: usize) -> bool {
        len > 0 && len <= MAX_READ_LEN && fits_i16(len, self.max_ref_len, &self.scoring)
    }

    /// Score every query (codes from [`crate::alphabet`]) against every
    /// reference: row `k` of the result is query `k`, in panel order, or
    /// unscored when [`GpuScorer::scores_len`] says no.
    pub fn score_batch(&mut self, queries: &[&[u8]]) -> Result<ScoreMatrix, GpuError> {
        let n = queries.len();
        let scored: Vec<bool> = queries.iter().map(|q| self.scores_len(q.len())).collect();
        let mut work: Vec<i32> = (0..n as i32).filter(|&k| scored[k as usize]).collect();
        if work.is_empty() {
            return Ok(ScoreMatrix::from_parts(
                self.n_refs,
                vec![0; n * self.n_refs],
                scored,
            ));
        }
        // Longest first: a long read starts early rather than finishing last.
        work.sort_by_key(|&k| std::cmp::Reverse(queries[k as usize].len()));

        // Every query gets whole stripes (`STRIPE_ROWS` codes, `STRIPE_ROWS / 8`
        // words), zero-padded: code 0 matches nothing, and the kernel masks
        // the rows anyway. Unscored queries get an empty slot.
        let stripe_words = STRIPE_ROWS / 8;
        let mut qoff = vec![0u64; n];
        let mut qlen = vec![0i32; n];
        let total_words: usize = queries
            .iter()
            .zip(&scored)
            .filter(|(_, s)| **s)
            .map(|(q, _)| q.len().div_ceil(STRIPE_ROWS) * stripe_words)
            .sum();
        let mut qwords = vec![0u32; total_words.max(1)];
        let mut at = 0usize;
        for (k, q) in queries.iter().enumerate() {
            if !scored[k] {
                continue;
            }
            qoff[k] = at as u64;
            qlen[k] = q.len() as i32;
            for (i, &c) in q.iter().enumerate() {
                qwords[at + i / 8] |= mask(c) << (4 * (i % 8));
            }
            at += q.len().div_ceil(STRIPE_ROWS) * stripe_words;
        }

        let s = &self.stream;
        let qwords_dev = s.clone_htod(&qwords)?;
        let qoff_dev = s.clone_htod(&qoff)?;
        let qlen_dev = s.clone_htod(&qlen)?;
        let order_dev = s.clone_htod(&work)?;
        let mut out_dev = s.alloc_zeros::<i16>(n * self.n_refs)?;
        let mut counters_dev = s.alloc_zeros::<i32>(self.n_groups)?;

        let n_work = work.len() as i32;
        let n_refs = self.n_refs as i32;
        let words_per_lane = self.words_per_lane as i32;
        let (ma, mi, o, e) = (
            self.scoring.match_score,
            self.scoring.mismatch,
            self.scoring.gap_open,
            self.scoring.gap_extend,
        );
        let local = i32::from(self.mode == Mode::Local);
        let cfg = LaunchConfig {
            grid_dim: (self.grid_x as u32, self.n_groups as u32, 1),
            block_dim: (self.lanes as u32, 1, 1),
            shared_mem_bytes: (self.words_per_lane * 4 * self.lanes) as u32,
        };
        let mut b = s.launch_builder(&self.func);
        b.arg(&qwords_dev)
            .arg(&qoff_dev)
            .arg(&qlen_dev)
            .arg(&order_dev)
            .arg(&self.panel_dev)
            .arg(&self.ref_len_dev)
            .arg(&self.ref_idx_dev)
            .arg(&mut out_dev)
            .arg(&mut self.scratch_dev)
            .arg(&mut counters_dev)
            .arg(&n_work)
            .arg(&n_refs)
            .arg(&words_per_lane)
            .arg(&ma)
            .arg(&mi)
            .arg(&o)
            .arg(&e)
            .arg(&local);
        // SAFETY: the argument list matches `align_score_kernel`'s signature
        // in order and type; every buffer is sized for the indices the kernel
        // forms from these same arguments (`qoff`/`qlen` per query, `order`
        // `n_work` long, the panel `n_groups * words_per_lane * lanes`, the
        // scratch `max_ref_len` words per launched thread, `out` a row per
        // query), and the shared memory is the panel slice it copies.
        unsafe { b.launch(cfg) }?;
        let scores = s.clone_dtoh(&out_dev)?;
        Ok(ScoreMatrix::from_parts(self.n_refs, scores, scored))
    }
}
