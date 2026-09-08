// SPDX-License-Identifier: MIT

//! Run the barcode CRF encoder's convolutions and five-layer LSTM stack
//! directly, bypassing tract's per-timestep `Scan` bookkeeping.
//!
//! Through tract this stack is 13.0 ms/read and 91% of the CPU demux path
//! (#173): three small convolutions and then five *unidirectional* LSTM
//! layers at hidden 96 over 200–350 timesteps. It is the same recurrence
//! shape the charging feature net had at 490 µs/read — `H = 96`, ONNX
//! `W`/`R`/`B` layout, no peepholes, zero initial state — and the same fix
//! applies: [`escapepod_signal::lstm`] runs the recurrence directly instead
//! of through a general graph interpreter. Nothing in that module is
//! specific to a bidirectional net; this crate calls it once per
//! *unidirectional* layer instead of once per direction.
//!
//! # The graph (read from `barcode_crf_ldx32_rna004@v0.2.1`, opset 17)
//!
//! ```text
//! signal [batch, 1, chunk]
//!   -> Conv(1->C1, k, pads) -> SiLU (Sigmoid, Mul)
//!   -> Conv(C1->C2, k, pads) -> SiLU
//!   -> Conv(C2->H, k, stride, pads) -> SiLU      # H = LSTM hidden size, stride = signal.stride
//!   -> Transpose(perm=[2,0,1])                    # channel-major -> time-major, for the LSTM's own convention
//!   -> LSTM x5, hidden_size = H, forward           # layers 0, 2, 4 run reversed (see below)
//!   -> MatMul[H, n_states*n_base] -> Add(bias) -> Tanh -> Mul(scale)
//!   -> Reshape [T,B,n_states,n_base] -> Pad(front of last axis, value = blank_score) -> Reshape [T,B,n_score]
//! ```
//!
//! **The reverse trick.** ONNX's `LSTM` op has a `direction` attribute, but
//! this export never sets it to `"reverse"` — instead, layers 0, 2 and 4 are
//! each wrapped in a `Slice(starts=[-1], ends=<very negative>, axes=[0],
//! steps=[-1])` *before* the LSTM and an identical one *after* (bonito's way
//! of running a layer "backward" through an ONNX runtime that only executes
//! `LSTM` forward): reverse the sequence, run forward, reverse the answer
//! back so each `h_t` still lands at its own `t`. Layers 1 and 3 have neither
//! Slice. [`escapepod_signal::lstm::run_scalar`] and its vector siblings take
//! this as a `reverse: bool` and walk `t` from `seq - 1` down without a
//! reversal copy — precisely the trick, done in the kernel instead of around
//! it. Which layers reverse is read from the graph, never assumed: a variant
//! export with a different pattern refuses to load natively rather than
//! silently decoding the wrong direction.
//!
//! **The blank pad.** The linear layer only emits `n_states * n_base` scores
//! (4 per state — the *move* edges; bonito's `blank_score` is a fixed
//! hyperparameter, not a learned weight) and a `Pad` node inserts a constant
//! at the front of the last axis to reach `n_states * n_edges` (`n_edges =
//! n_base + 1`), matching [`super::lattice::CrfLayout::score_index`]'s
//! `edge == 0` being the blank/stay edge. The inserted value is read from the
//! graph's own `Pad` constant (a bundle's `metadata.json` `crf.blank_score`,
//! where present, is a cross-check against it, not the source of truth — see
//! [`Recognized::blank_score`]).
//!
//! **What is *not* symbolically verified.** The shape arithmetic feeding the
//! two `Reshape`s and the `Pad`'s own `pads` tensor is a dynamic
//! `Shape`/`Gather`/`Mod`/`Concat` subgraph — an artifact of how this export
//! was traced, not a rule a bundle chooses. Interpreting it symbolically
//! would mean re-implementing a slice of ONNX shape inference for a fact this
//! runtime already knows independently (the pad widens the last axis by one,
//! at the front, using bonito's fixed convention). Instead the recognizer
//! verifies *connectivity* — the scaled linear output reaches the `Pad`
//! through zero or more `Reshape`s and nothing else, and the `Pad`'s output
//! reaches the graph output the same way — and the numeric self-check
//! ([`Recognized::self_check`]) is the correctness gate for the assumption
//! itself: a wrong pad side or a wrong direction is O(1) on scores bounded to
//! ±5, so it fails that check by orders of magnitude past the tolerance
//! ordinary `exp`/`tanh` implementation noise needs.
//!
//! Anything the recognizer does not expect — a different layer count, a
//! bidirectional layer, peepholes, a non-zero initial state, a differently
//! shaped conv or linear layer — refuses with a reason logged at debug level,
//! and the caller keeps tract. `ESCAPEPOD_CRF_TRACT=1` forces tract
//! regardless (mirrors `ESCAPEPOD_FNN_TRACT`), and is the way back if a
//! future bundle's export ever disagrees with this kernel.

use std::collections::HashMap;

use escapepod_signal::lstm::{self, LstmBackend, LstmWeights};
use tract_onnx::pb;

use super::encoder::CrfMetadata;
use super::lattice::CrfLayout;

const ONNX_FLOAT: i32 = 1;
const ONNX_INT64: i32 = 7;

/// One `Conv` + SiLU stage, weights lifted out of the proto.
struct ConvLayer {
    in_c: usize,
    out_c: usize,
    k: usize,
    pad_lo: usize,
    pad_hi: usize,
    stride: usize,
    /// `[out_c][in_c][k]`.
    weight: Vec<f32>,
    /// `[out_c]`.
    bias: Vec<f32>,
}

/// One stacked LSTM layer: its weights and whether it runs the sequence in
/// reverse (see the module doc's "reverse trick").
struct Layer {
    weights: LstmWeights,
    reverse: bool,
}

/// A recognised export, weights lifted and ready to run.
///
/// Nothing here reads a file or touches tract; [`Recognized::from_proto`]
/// takes an already-parsed proto (the same one [`super::encoder::CrfEncoder`]
/// loads for the padding hoist) and the sidecar it was loaded beside.
pub struct Recognized {
    chunk: usize,
    t_len: usize,
    convs: [ConvLayer; 3],
    layers: [Layer; 5],
    hidden: usize,
    /// `[hidden][n_states * n_base]`.
    linear_w: Vec<f32>,
    /// `[n_states * n_base]`.
    linear_b: Vec<f32>,
    /// The `Mul` constant after `Tanh` (bonito's fixed score scale, 5.0 in
    /// every export seen so far — read from the graph, never assumed).
    scale: f32,
    /// The constant the `Pad` inserts for the blank edge.
    blank_score: f32,
    n_states: usize,
    n_base: usize,
    n_edges: usize,
    n_score: usize,
    backend: LstmBackend,
}

impl Recognized {
    /// Recognise the exported graph and lift its weights.
    ///
    /// `meta`/`layout` are the sidecar this proto was loaded beside — used to
    /// cross-check the recognised shape against what the bundle declares
    /// (`layout.n_states * layout.n_base` must match the linear layer's
    /// output width; `meta.crf.blank_score`, where present, must match the
    /// graph's own `Pad` constant) and, for `blank_score`, as a value this
    /// function reads with. `None` means "not this graph" and is logged at
    /// debug level with the reason; the caller falls back to tract.
    pub fn from_proto(
        proto: &pb::ModelProto,
        meta: &CrfMetadata,
        layout: &CrfLayout,
    ) -> Option<Self> {
        match try_match(proto, meta, layout) {
            Ok(net) => Some(net),
            Err(why) => {
                tracing::debug!("CRF encoder is not a recognised native-stack graph: {why}");
                if std::env::var_os("ESCAPEPOD_CRF_DEBUG_RECOGNIZER").is_some() {
                    eprintln!("CRF encoder is not a recognised native-stack graph: {why}");
                }
                None
            }
        }
    }

    pub fn backend(&self) -> LstmBackend {
        self.backend
    }

    /// Force a specific kernel width, for testing cross-backend agreement —
    /// production loading always takes `LstmBackend::best_for`.
    #[cfg(test)]
    pub fn with_backend(mut self, backend: LstmBackend) -> Result<Self, String> {
        if !backend.supported() {
            return Err(format!(
                "this machine cannot run the {} kernel",
                backend.name()
            ));
        }
        if !self.hidden.is_multiple_of(backend.lanes()) {
            return Err(format!(
                "the {} kernel needs a hidden size that is a multiple of {}, not {}",
                backend.name(),
                backend.lanes(),
                self.hidden
            ));
        }
        self.backend = backend;
        Ok(self)
    }

    /// The constant score inserted for the blank edge — bonito's fixed
    /// hyperparameter, read from the graph's `Pad` node.
    pub fn blank_score(&self) -> f32 {
        self.blank_score
    }

    /// Run one standardised `chunk`-sample window (batch 1) through the whole
    /// stack, writing `t_len * n_score` scores into `out` in `[t][dest *
    /// n_edges + edge]` order — the layout [`super::encoder::CrfEncoder`]'s
    /// tract path returns and [`super::lattice::decode_with`] expects.
    pub fn encode_into(&self, prepped: &[f32], out: &mut [f32]) {
        assert_eq!(
            prepped.len(),
            self.chunk,
            "encoder takes a chunk-sample window"
        );
        assert_eq!(
            out.len(),
            self.t_len * self.n_score,
            "output is t_len * n_score"
        );

        let (mut cur, mut len, mut in_c) = (prepped.to_vec(), self.chunk, 1usize);
        for conv in &self.convs {
            debug_assert_eq!(
                in_c, conv.in_c,
                "channel count agrees with the recognised weights"
            );
            let (next, next_len) = conv1d_silu(&cur, in_c, len, conv);
            cur = next;
            len = next_len;
            in_c = conv.out_c;
        }
        debug_assert_eq!(len, self.t_len);
        debug_assert_eq!(in_c, self.hidden);

        let (h, g, seq) = (self.hidden, 4 * self.hidden, self.t_len);
        let mut xw = vec![0.0f32; seq * g];
        let mut gates = vec![0.0f32; g];
        let mut hs = vec![0.0f32; seq * h];
        // `cur` is already channel-major `[H][T]` — exactly what layer 0
        // wants, straight out of the last convolution, with no transpose:
        // the ONNX graph's own `Transpose(perm=[2,0,1])` exists only to feed
        // the LSTM node's time-major convention, which this kernel does not
        // use.
        let mut channel_major = cur;
        for (i, layer) in self.layers.iter().enumerate() {
            match self.backend {
                LstmBackend::Scalar => lstm::run_scalar(
                    &layer.weights,
                    seq,
                    layer.reverse,
                    &channel_major,
                    &mut xw,
                    &mut hs,
                    h,
                    0,
                    &mut gates,
                ),
                #[cfg(target_arch = "x86_64")]
                // Safety: `with_backend`/`best_for` only select a backend
                // this machine supports and whose width divides `h`.
                LstmBackend::Avx2 => unsafe {
                    lstm::run_avx2(
                        &layer.weights,
                        seq,
                        layer.reverse,
                        &channel_major,
                        &mut xw,
                        &mut hs,
                        h,
                        0,
                        &mut gates,
                    )
                },
                #[cfg(target_arch = "x86_64")]
                // A lone read on the AVX-512 backend takes the AVX2 kernel,
                // as `escapepod_classify::fnn_lstm` does — bit-identical, and
                // the AVX2 kernel is faster per read at N = 1.
                LstmBackend::Avx512 => unsafe {
                    lstm::run_avx2(
                        &layer.weights,
                        seq,
                        layer.reverse,
                        &channel_major,
                        &mut xw,
                        &mut hs,
                        h,
                        0,
                        &mut gates,
                    )
                },
            }
            if i + 1 < self.layers.len() {
                // Layer l's time-major `[seq][H]` output becomes layer l+1's
                // channel-major `[H][seq]` input. A `T x H` transpose (≤ 300 x
                // 96 here) is negligible beside the O(T H^2) recurrence it
                // feeds; reusing `input_contribution`'s channel-major
                // contract for every layer (rather than a second time-major
                // variant) is worth that transpose.
                for t in 0..seq {
                    for u in 0..h {
                        channel_major[u * seq + t] = hs[t * h + u];
                    }
                }
            }
        }
        // `hs` now holds the last layer's `[seq][H]` time-major output —
        // exactly the per-timestep vector the linear layer reads, so no
        // further transpose is needed here.
        self.linear_head(&hs, out);
    }

    pub fn encode(&self, prepped: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; self.t_len * self.n_score];
        self.encode_into(prepped, &mut out);
        out
    }

    /// [`Self::encode`], scoring into a per-thread scratch buffer and handing
    /// the result to `f` before it is dropped.
    ///
    /// Saves the ~1 MB allocation [`Self::encode`] otherwise pays every read
    /// (the same reason [`super::encoder::CrfEncoder::basecall_prepped`]
    /// decodes straight out of tract's own output tensor on that path). `f`
    /// runs with the scratch's `RefCell` borrowed, so it must not re-enter
    /// this method on the same thread.
    pub fn encode_with<R>(&self, prepped: &[f32], f: impl FnOnce(&[f32]) -> R) -> R {
        thread_local! {
            static SCORES: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
        }
        SCORES.with(|s| {
            let mut s = s.borrow_mut();
            s.resize(self.t_len * self.n_score, 0.0);
            self.encode_into(prepped, &mut s);
            f(&s)
        })
    }

    /// The linear layer, tanh, scale and blank pad — the tail shared by
    /// [`Self::encode_into`] and [`Self::encode_batch_into`]. `hs` is one
    /// read's `[seq][H]` time-major LSTM output; `out` receives `t_len *
    /// n_score` scores in `[t][dest * n_edges + edge]` order.
    fn linear_head(&self, hs: &[f32], out: &mut [f32]) {
        let (h, seq) = (self.hidden, self.t_len);
        let o_dim = self.n_states * self.n_base;
        let mut logits = vec![0.0f32; o_dim];
        for t in 0..seq {
            let hv = &hs[t * h..(t + 1) * h];
            logits.copy_from_slice(&self.linear_b);
            for (&x, wr) in hv.iter().zip(self.linear_w.chunks_exact(o_dim)) {
                for (l, w) in logits.iter_mut().zip(wr) {
                    *l += x * w;
                }
            }
            // Vectorised in place, not one `f32::tanh` (a full libm call) per
            // of the up to 1024 moves at every one of up to 350 timesteps:
            // measured as ~10 ms/read of scalar tanh calls alone, the same
            // order as the whole LSTM stack, before this (#331).
            lstm::tanh_slice(self.backend, &mut logits);
            let row = &mut out[t * self.n_score..(t + 1) * self.n_score];
            for state in 0..self.n_states {
                row[state * self.n_edges] = self.blank_score;
                let moves = &logits[state * self.n_base..(state + 1) * self.n_base];
                for (base, &m) in moves.iter().enumerate() {
                    row[state * self.n_edges + 1 + base] = self.scale * m;
                }
            }
        }
    }

    /// How many reads [`Self::encode_batch_into`] scores per pass at full
    /// efficiency on this backend. See [`LstmBackend::preferred_batch`].
    pub fn preferred_batch(&self) -> usize {
        self.backend.preferred_batch()
    }

    /// One LSTM layer, `xs.len()` reads in lockstep — the same dispatch table
    /// `escapepod_classify::fnn_lstm::NativeBiLstm::logits_batch` uses, at
    /// this stack's widths (`preferred_batch()`, not `NativeBiLstm`'s, since
    /// the two nets can pick different backends). `xw`/`hs`/`gates` must be
    /// sized for exactly `xs.len()` reads — the caller slices its
    /// `preferred_batch()`-sized scratch down to the group in use.
    ///
    /// A lone read on a vector backend takes the single-read kernel directly
    /// (as `NativeBiLstm` does): neither `run_avx2_batch::<1>` nor
    /// `run_avx512_batch::<1>` is instantiated, matching
    /// `escapepod_signal::lstm`'s own tests.
    fn run_layer(
        &self,
        layer: &Layer,
        seq: usize,
        xs: &[&[f32]],
        xw: &mut [f32],
        hs: &mut [f32],
        gates: &mut [f32],
    ) {
        let h = self.hidden;
        match (self.backend, xs.len()) {
            #[cfg(target_arch = "x86_64")]
            // Safety: `with_backend`/`best_for` only select a backend this
            // machine supports and whose width divides `h`; the buffers are
            // sized by the caller for exactly `xs.len()` reads.
            (LstmBackend::Avx512, 8) => unsafe {
                lstm::run_avx512_batch::<8>(
                    &layer.weights,
                    seq,
                    layer.reverse,
                    xs,
                    xw,
                    hs,
                    h,
                    0,
                    gates,
                )
            },
            #[cfg(target_arch = "x86_64")]
            (LstmBackend::Avx512, 7) => unsafe {
                lstm::run_avx512_batch::<7>(
                    &layer.weights,
                    seq,
                    layer.reverse,
                    xs,
                    xw,
                    hs,
                    h,
                    0,
                    gates,
                )
            },
            #[cfg(target_arch = "x86_64")]
            (LstmBackend::Avx512, 6) => unsafe {
                lstm::run_avx512_batch::<6>(
                    &layer.weights,
                    seq,
                    layer.reverse,
                    xs,
                    xw,
                    hs,
                    h,
                    0,
                    gates,
                )
            },
            #[cfg(target_arch = "x86_64")]
            (LstmBackend::Avx512, 5) => unsafe {
                lstm::run_avx512_batch::<5>(
                    &layer.weights,
                    seq,
                    layer.reverse,
                    xs,
                    xw,
                    hs,
                    h,
                    0,
                    gates,
                )
            },
            #[cfg(target_arch = "x86_64")]
            (LstmBackend::Avx512, 4) => unsafe {
                lstm::run_avx512_batch::<4>(
                    &layer.weights,
                    seq,
                    layer.reverse,
                    xs,
                    xw,
                    hs,
                    h,
                    0,
                    gates,
                )
            },
            #[cfg(target_arch = "x86_64")]
            (LstmBackend::Avx512, 3) => unsafe {
                lstm::run_avx512_batch::<3>(
                    &layer.weights,
                    seq,
                    layer.reverse,
                    xs,
                    xw,
                    hs,
                    h,
                    0,
                    gates,
                )
            },
            #[cfg(target_arch = "x86_64")]
            (LstmBackend::Avx512, 2) => unsafe {
                lstm::run_avx512_batch::<2>(
                    &layer.weights,
                    seq,
                    layer.reverse,
                    xs,
                    xw,
                    hs,
                    h,
                    0,
                    gates,
                )
            },
            #[cfg(target_arch = "x86_64")]
            (LstmBackend::Avx2, 3) => unsafe {
                lstm::run_avx2_batch::<3>(
                    &layer.weights,
                    seq,
                    layer.reverse,
                    xs,
                    xw,
                    hs,
                    h,
                    0,
                    gates,
                )
            },
            #[cfg(target_arch = "x86_64")]
            (LstmBackend::Avx2, 2) => unsafe {
                lstm::run_avx2_batch::<2>(
                    &layer.weights,
                    seq,
                    layer.reverse,
                    xs,
                    xw,
                    hs,
                    h,
                    0,
                    gates,
                )
            },
            #[cfg(target_arch = "x86_64")]
            // A lone read on either wide backend takes the AVX2 single-read
            // kernel — bit-identical to the batched entry points, and faster
            // per read than the 16-wide kernel at N = 1 (see
            // `escapepod_signal::lstm`). Two arms, not an or-pattern: the
            // kernels are `#[inline(never)]` on evidence that an or-pattern
            // over lone-read arms miscompiled once (see their doc comments).
            (LstmBackend::Avx512, _) => unsafe {
                lstm::run_avx2(
                    &layer.weights,
                    seq,
                    layer.reverse,
                    xs[0],
                    xw,
                    hs,
                    h,
                    0,
                    gates,
                )
            },
            #[cfg(target_arch = "x86_64")]
            (LstmBackend::Avx2, _) => unsafe {
                lstm::run_avx2(
                    &layer.weights,
                    seq,
                    layer.reverse,
                    xs[0],
                    xw,
                    hs,
                    h,
                    0,
                    gates,
                )
            },
            (LstmBackend::Scalar, _) => {
                lstm::run_scalar_batch(&layer.weights, seq, layer.reverse, xs, xw, hs, h, 0, gates)
            }
        }
    }

    /// [`Self::encode_into`], `prepped.len()` reads at once — any count, not
    /// only `preferred_batch()`. Internally chunked into groups of
    /// `preferred_batch()` (a short tail included), the same convention
    /// `escapepod_classify::fnn_lstm::NativeBiLstm::logits_batch` uses: the
    /// caller does not need to know the backend's width, and the buffers
    /// below are sized for the widest group once and reused across chunks.
    ///
    /// Every read in a chunk shares that chunk's LSTM weight-row loads, which
    /// is where the single-read kernel's time goes (see
    /// [`escapepod_signal::lstm`]). The convolutions and the linear head are
    /// looped per read — plain GEMM work the compiler already vectorises, and
    /// a small fraction of the stack's MACs (#331) — so the batching pays for
    /// itself entirely in the recurrence. `out` is `prepped.len() * t_len *
    /// n_score`, one read's scores after another in [`Self::encode_into`]'s
    /// order.
    pub fn encode_batch_into(&self, prepped: &[&[f32]], out: &mut [f32]) {
        let n_total = prepped.len();
        for p in prepped {
            assert_eq!(p.len(), self.chunk, "encoder takes a chunk-sample window");
        }
        assert_eq!(
            out.len(),
            n_total * self.t_len * self.n_score,
            "output is n * t_len * n_score"
        );

        let (h, g, seq) = (self.hidden, 4 * self.hidden, self.t_len);
        let batch = self.preferred_batch();
        let mut xw = vec![0.0f32; batch * seq * g];
        let mut gates = vec![0.0f32; batch * g];
        let mut hs = vec![0.0f32; batch * seq * h];

        let mut done = 0usize;
        while done < n_total {
            let n = (n_total - done).min(batch);
            let mut channel_major: Vec<Vec<f32>> = prepped[done..done + n]
                .iter()
                .map(|p| {
                    let (mut cur, mut len, mut in_c) = (p.to_vec(), self.chunk, 1usize);
                    for conv in &self.convs {
                        let (next, next_len) = conv1d_silu(&cur, in_c, len, conv);
                        cur = next;
                        len = next_len;
                        in_c = conv.out_c;
                    }
                    debug_assert_eq!(len, self.t_len);
                    debug_assert_eq!(in_c, self.hidden);
                    cur
                })
                .collect();

            for layer in &self.layers {
                let xs: Vec<&[f32]> = channel_major.iter().map(Vec::as_slice).collect();
                self.run_layer(
                    layer,
                    seq,
                    &xs,
                    &mut xw[..n * seq * g],
                    &mut hs[..n * seq * h],
                    &mut gates[..n * g],
                );
                for (r, cm_r) in channel_major.iter_mut().enumerate() {
                    let hs_r = &hs[r * seq * h..(r + 1) * seq * h];
                    for t in 0..seq {
                        for u in 0..h {
                            cm_r[u * seq + t] = hs_r[t * h + u];
                        }
                    }
                }
            }

            for r in 0..n {
                let hs_r = &hs[r * seq * h..(r + 1) * seq * h];
                let out_r = &mut out[(done + r) * self.t_len * self.n_score
                    ..(done + r + 1) * self.t_len * self.n_score];
                self.linear_head(hs_r, out_r);
            }
            done += n;
        }
    }

    /// [`Self::encode_batch_into`], returning owned scores per read — used
    /// only by tests; production callers (`CrfEncoder::encode_group`) go
    /// through `encode_batch_into` directly to reuse their own buffers.
    #[cfg(test)]
    pub fn encode_batch(&self, prepped: &[&[f32]]) -> Vec<Vec<f32>> {
        let n = prepped.len();
        let stride = self.t_len * self.n_score;
        let mut flat = vec![0.0f32; n * stride];
        self.encode_batch_into(prepped, &mut flat);
        flat.chunks_exact(stride).map(<[f32]>::to_vec).collect()
    }
}

/// One 1-D convolution (symmetric zero padding, arbitrary stride) plus SiLU
/// (`x * sigmoid(x)`), matching the graph's `Sigmoid` + `Mul` pair.
///
/// Not the plain nested loop the MAC count alone would suggest is enough
/// (~15M MACs/read against the LSTM stack's ~110M, #331): a first cut written
/// exactly that way — reduce over `k` per output element, the natural
/// reading of the maths — measured at 19.5 ms/read, the single largest cost
/// of the whole encoder, because a reduction of only `k` (5, or 31 for the
/// strided stage) terms per `(oc, ot, ic)` triple neither auto-vectorises nor
/// amortises its own loop overhead. Two changes bring it under the LSTM
/// stack's own cost:
///
/// * Every accumulation is an axpy over the *output* axis
///   (`acc_row[ot] += w * x[..]`, `ot` contiguous, `out_len` wide) rather than
///   a reduction over the kernel axis — the same shape
///   [`escapepod_signal::lstm::input_contribution`] already relies on to
///   auto-vectorise at this crate family's baseline
///   (`-C target-cpu=x86-64-v3`).
/// * The strided stage (the third conv, `stride = 10`) additionally needs a
///   polyphase split: `padded[ot * stride + kk]` is a strided *gather* as
///   `ot` varies, which measured no better than the reduction it replaced
///   (Cascade Lake's gather throughput is the likely reason, though this was
///   not root-caused further — the fix does not depend on why). Splitting
///   each channel into `stride` phases up front —
///   `phase[r][j] = padded[r + j * stride]` — turns that same access into
///   `phase[kk % stride][(kk / stride)..(kk / stride) + out_len]`, a
///   contiguous slice: `ot * stride + kk = stride * (ot + kk / stride) + (kk
///   % stride)`, exactly `phase[kk % stride]` indexed at `ot + kk / stride`.
///   `stride == 1` skips this (already contiguous with `q = 0`, one phase).
fn conv1d_silu(input: &[f32], in_c: usize, len: usize, conv: &ConvLayer) -> (Vec<f32>, usize) {
    let padded_len = len + conv.pad_lo + conv.pad_hi;
    let out_len = (padded_len - conv.k) / conv.stride + 1;
    // Zero-pad each channel once so the accumulation below is branch-free.
    let mut padded = vec![0.0f32; in_c * padded_len];
    for ic in 0..in_c {
        let dst = &mut padded[ic * padded_len + conv.pad_lo..ic * padded_len + conv.pad_lo + len];
        dst.copy_from_slice(&input[ic * len..(ic + 1) * len]);
    }

    let mut acc = vec![0.0f32; conv.out_c * out_len];
    for oc in 0..conv.out_c {
        acc[oc * out_len..(oc + 1) * out_len].fill(conv.bias[oc]);
    }

    let stride = conv.stride;
    // `phase[ic][r]` is `padded[ic]` subsampled every `stride` starting at
    // `r`: `phase[ic][r][j] = padded[ic][r + j * stride]`. `stride == 1` is
    // one trivial phase equal to the channel itself, so the general path
    // below covers both without a separate branch on `stride`.
    let phase_len = padded_len.div_ceil(stride);
    let mut phases = vec![0.0f32; in_c * stride * phase_len];
    for ic in 0..in_c {
        let padded_ch = &padded[ic * padded_len..(ic + 1) * padded_len];
        for r in 0..stride {
            let dst = &mut phases[(ic * stride + r) * phase_len..(ic * stride + r + 1) * phase_len];
            for (j, d) in dst.iter_mut().enumerate() {
                if let Some(&x) = padded_ch.get(r + j * stride) {
                    *d = x;
                }
            }
        }
    }

    for oc in 0..conv.out_c {
        let w_oc = &conv.weight[oc * in_c * conv.k..(oc + 1) * in_c * conv.k];
        let acc_row = &mut acc[oc * out_len..(oc + 1) * out_len];
        for ic in 0..in_c {
            let w_ic = &w_oc[ic * conv.k..(ic + 1) * conv.k];
            for (kk, &wv) in w_ic.iter().enumerate() {
                let (r, q) = (kk % stride, kk / stride);
                let phase =
                    &phases[(ic * stride + r) * phase_len..(ic * stride + r + 1) * phase_len];
                let window = &phase[q..q + out_len];
                for (a, &x) in acc_row.iter_mut().zip(window) {
                    *a += wv * x;
                }
            }
        }
    }

    for v in &mut acc {
        *v *= sigmoid(*v);
    }
    (acc, out_len)
}

#[inline(always)]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// ---- the recognizer -----------------------------------------------------

fn try_match(
    proto: &pb::ModelProto,
    meta: &CrfMetadata,
    layout: &CrfLayout,
) -> Result<Recognized, String> {
    let graph = proto.graph.as_ref().ok_or("no graph")?;
    let init: HashMap<&str, &pb::TensorProto> = graph
        .initializer
        .iter()
        .map(|t| (t.name.as_str(), t))
        .collect();

    let input_name = graph
        .input
        .first()
        .map(|v| v.name.as_str())
        .ok_or("graph has no input")?;

    // --- three Conv + SiLU stages ---------------------------------------
    let n_convs = graph.node.iter().filter(|n| n.op_type == "Conv").count();
    if n_convs != 3 {
        return Err(format!("{n_convs} Conv nodes, expected 3"));
    }
    let mut cur = input_name.to_string();
    let mut convs: Vec<ConvLayer> = Vec::with_capacity(3);
    for stage in 0..3 {
        let conv = find_consumers(graph, &cur)
            .into_iter()
            .find(|n| n.op_type == "Conv")
            .ok_or_else(|| format!("nothing consumes `{cur}` as a Conv (stage {stage})"))?;
        if !conv.domain.is_empty() {
            return Err(format!("Conv (stage {stage}) has a non-default domain"));
        }
        if attr_int(conv, "group").unwrap_or(1) != 1 {
            return Err(format!("Conv (stage {stage}) group != 1"));
        }
        if let Some(d) = attr_ints(conv, "dilations")
            && d != [1]
        {
            return Err(format!("Conv (stage {stage}) dilation != 1"));
        }
        let kernel_shape = attr_ints(conv, "kernel_shape").ok_or("Conv has no kernel_shape")?;
        let [k] = kernel_shape else {
            return Err(format!("Conv (stage {stage}) is not 1-D"));
        };
        let k = *k as usize;
        let pads = attr_ints(conv, "pads").ok_or("Conv has no pads")?;
        let (pad_lo, pad_hi) = match pads {
            [lo, hi] if *lo >= 0 && *hi >= 0 => (*lo as usize, *hi as usize),
            _ => return Err(format!("Conv (stage {stage}) pads {pads:?} malformed")),
        };
        let stride = match attr_ints(conv, "strides") {
            None => 1,
            Some([s]) if *s > 0 => *s as usize,
            Some(s) => return Err(format!("Conv (stage {stage}) strides {s:?} malformed")),
        };
        let w_name = conv.input.get(1).ok_or("Conv has no weight input")?;
        let w_t = init
            .get(w_name.as_str())
            .ok_or("Conv weight is not an initializer")?;
        let (out_c, in_c) = match w_t.dims.as_slice() {
            [oc, ic, kk] if *kk as usize == k => (*oc as usize, *ic as usize),
            d => return Err(format!("Conv (stage {stage}) weight dims {d:?}")),
        };
        let weight = tensor_f32(w_t)?;
        let bias = match conv.input.get(2).map(String::as_str) {
            Some("") | None => vec![0.0f32; out_c],
            Some(name) => {
                let t = init.get(name).ok_or("Conv bias is not an initializer")?;
                if t.dims != [out_c as i64] {
                    return Err(format!("Conv (stage {stage}) bias dims {:?}", t.dims));
                }
                tensor_f32(t)?
            }
        };

        // SiLU is `x * sigmoid(x)`: the conv output feeds *two* consumers,
        // the `Sigmoid` and the `Mul` that reads both the conv output and the
        // `Sigmoid`'s own output — not one, so this cannot be `sole_consumer`
        // on the conv output itself. Real exports confirm the shape (checked
        // against `barcode_crf_ldx32_rna004@v0.2.1`): `conv_out` has exactly
        // these two consumers, never a third.
        let conv_out = conv.output.first().cloned().ok_or("Conv has no output")?;
        let conv_out_consumers = find_consumers(graph, &conv_out);
        let sig = conv_out_consumers
            .iter()
            .find(|n| n.op_type == "Sigmoid")
            .copied()
            .ok_or_else(|| format!("Conv (stage {stage}) output has no Sigmoid consumer"))?;
        let mul = sole_consumer(graph, &sig.output[0], "Mul")?;
        if !(mul.input.contains(&conv_out) && mul.input.contains(&sig.output[0])) {
            return Err(format!(
                "Conv (stage {stage})'s SiLU Mul does not combine the conv output and its Sigmoid"
            ));
        }
        if conv_out_consumers.len() != 2
            || !conv_out_consumers.iter().any(|n| std::ptr::eq(*n, mul))
        {
            return Err(format!(
                "Conv (stage {stage}) output has a consumer besides its own Sigmoid and SiLU Mul"
            ));
        }

        convs.push(ConvLayer {
            in_c,
            out_c,
            k,
            pad_lo,
            pad_hi,
            stride,
            weight,
            bias,
        });
        cur = mul.output[0].clone();
    }
    let convs: [ConvLayer; 3] = convs
        .try_into()
        .unwrap_or_else(|_| unreachable!("exactly 3 pushed above"));
    if convs[2].stride != meta.signal.stride {
        return Err(format!(
            "final conv stride {} disagrees with the bundle's signal.stride {}",
            convs[2].stride, meta.signal.stride
        ));
    }

    // --- the conv-to-LSTM transpose --------------------------------------
    // Not read numerically (layer 0 takes the pre-transpose, channel-major
    // conv output directly — see `Recognized::encode_into`); its presence and
    // shape confirm this is the graph this recognizer thinks it is.
    let transpose = sole_consumer(graph, &cur, "Transpose")?;
    if attr_ints(transpose, "perm") != Some(&[2, 0, 1][..]) {
        return Err("conv-to-LSTM Transpose is not perm=[2,0,1]".into());
    }

    // --- five stacked unidirectional LSTM layers -------------------------
    let lstms: Vec<&pb::NodeProto> = graph.node.iter().filter(|n| n.op_type == "LSTM").collect();
    let [l0, l1, l2, l3, l4] = lstms.as_slice() else {
        return Err(format!("{} LSTM nodes, expected 5", lstms.len()));
    };
    let lstms = [l0, l1, l2, l3, l4];

    let mut hidden = 0usize;
    let mut layers: Vec<Layer> = Vec::with_capacity(5);
    let mut layer_input_name = transpose.output[0].clone();
    for (i, lstm) in lstms.into_iter().enumerate() {
        if attr_str(lstm, "direction").unwrap_or("forward") != "forward" {
            return Err(format!("LSTM {i} is not the default forward direction"));
        }
        let h = attr_int(lstm, "hidden_size").ok_or("LSTM has no hidden_size")? as usize;
        if h == 0 {
            return Err(format!("LSTM {i} hidden_size is 0"));
        }
        if i == 0 {
            hidden = h;
        } else if h != hidden {
            return Err(format!("LSTM {i} hidden_size {h} != layer 0's {hidden}"));
        }
        for forbidden in ["activations", "activation_alpha", "activation_beta", "clip"] {
            if lstm.attribute.iter().any(|a| a.name == forbidden) {
                return Err(format!("LSTM {i} sets `{forbidden}`"));
            }
        }
        if attr_int(lstm, "input_forget").unwrap_or(0) != 0 {
            return Err(format!("LSTM {i} couples input and forget gates"));
        }
        if attr_int(lstm, "layout").unwrap_or(0) != 0 {
            return Err(format!("LSTM {i} uses batch-major layout"));
        }
        let input_at = |k: usize| lstm.input.get(k).map(String::as_str).unwrap_or("");
        if lstm.input.len() < 3 {
            return Err(format!("LSTM {i} has fewer than 3 inputs"));
        }
        let g = 4 * h;
        let w_t = init
            .get(input_at(1))
            .ok_or_else(|| format!("LSTM {i} W is not an initializer"))?;
        let r_t = init
            .get(input_at(2))
            .ok_or_else(|| format!("LSTM {i} R is not an initializer"))?;
        let n_in = match w_t.dims.as_slice() {
            [1, gg, n_in] if *gg as usize == g => *n_in as usize,
            d => return Err(format!("LSTM {i} W dims {d:?}, expected [1, {g}, n_in]")),
        };
        if r_t.dims != [1, g as i64, h as i64] {
            return Err(format!(
                "LSTM {i} R dims {:?}, expected [1, {g}, {h}]",
                r_t.dims
            ));
        }
        let w = tensor_f32(w_t)?;
        let r = tensor_f32(r_t)?;
        let b = match input_at(3) {
            "" => vec![0.0f32; 8 * h],
            name => {
                let t = init
                    .get(name)
                    .ok_or_else(|| format!("LSTM {i} B is not an initializer"))?;
                if t.dims != [1, 8 * h as i64] {
                    return Err(format!(
                        "LSTM {i} B dims {:?}, expected [1, {}]",
                        t.dims,
                        8 * h
                    ));
                }
                tensor_f32(t)?
            }
        };
        if !input_at(4).is_empty() {
            return Err(format!("LSTM {i} has per-sequence lengths"));
        }
        for (idx, what) in [(5, "initial_h"), (6, "initial_c")] {
            let name = input_at(idx);
            if name.is_empty() {
                continue;
            }
            let p = find_producer(graph, name)
                .ok_or_else(|| format!("LSTM {i} {what} has no producer"))?;
            if p.op_type != "ConstantOfShape" {
                return Err(format!(
                    "LSTM {i} {what} comes from {}, not zeros",
                    p.op_type
                ));
            }
            if let Some(t) = attr_tensor(p, "value")
                && tensor_f32(t)?.iter().any(|&v| v != 0.0)
            {
                return Err(format!("LSTM {i} {what} is a non-zero constant"));
            }
        }
        if !input_at(7).is_empty() {
            return Err(format!("LSTM {i} has peephole weights"));
        }

        // --- direction: was `layer_input_name` reversed before feeding this LSTM? ---
        //
        // Checked by direct equality first, not by "is the producer a Slice":
        // when layer `i - 1` itself reverses, *its* un-reverse Slice is what
        // produces `layer_input_name`, and layer `i` (unreversed) takes that
        // name as its LSTM input verbatim — the producer is a Slice either
        // way, so that alone cannot tell the two cases apart. Real exports
        // confirm this is not hypothetical: `barcode_crf_ldx32_rna004`'s own
        // LSTM 1 takes LSTM 0's un-reverse Slice output directly.
        let lstm_data_in = input_at(0).to_string();
        let (reverse_in, pre_reverse) = if lstm_data_in == layer_input_name {
            (false, lstm_data_in.clone())
        } else {
            match find_producer(graph, &lstm_data_in) {
                Some(p) if p.op_type == "Slice" => {
                    verify_reverse_slice(graph, &init, p)?;
                    (true, p.input[0].clone())
                }
                _ => (false, lstm_data_in.clone()),
            }
        };
        if pre_reverse != layer_input_name {
            return Err(format!(
                "LSTM {i}'s input does not trace back to layer {}'s output",
                i.wrapping_sub(1)
            ));
        }

        // --- the readout: Y -> Squeeze(axis=1) -> [optional un-reverse] ---
        let y = lstm.output.first().map(String::as_str).unwrap_or("");
        for extra in lstm.output.iter().skip(1) {
            if !extra.is_empty() && !find_consumers(graph, extra).is_empty() {
                return Err(format!("LSTM {i} output `{extra}` is consumed"));
            }
        }
        let squeeze = sole_consumer(graph, y, "Squeeze")?;
        verify_squeeze_axis1(graph, &init, squeeze)?;
        let squeeze_out = squeeze.output[0].clone();

        // The output side is derived from `reverse_in`, not independently
        // re-detected: whether `squeeze_out`'s consumer is a Slice cannot by
        // itself distinguish "this layer's own un-reverse" from "the next
        // layer's pre-reverse" (the same ambiguity fixed above, mirrored).
        // `reverse_in` already answers the only question that matters — does
        // this layer's semantic output need one more Slice to undo the
        // reversal its LSTM ran under.
        let layer_output = if reverse_in {
            let post = sole_consumer(graph, &squeeze_out, "Slice").map_err(|e| {
                format!(
                    "LSTM {i} reverses its input but its output has no matching un-reverse: {e}"
                )
            })?;
            verify_reverse_slice(graph, &init, post)?;
            post.output[0].clone()
        } else {
            squeeze_out.clone()
        };

        layers.push(Layer {
            weights: LstmWeights::from_onnx(n_in, h, &w, &r, &b),
            reverse: reverse_in,
        });
        layer_input_name = layer_output;
    }
    if layers[0].weights.n_in != convs[2].out_c {
        return Err(format!(
            "layer 0 expects {} input channels, the last conv emits {}",
            layers[0].weights.n_in, convs[2].out_c
        ));
    }
    let layers: [Layer; 5] = layers
        .try_into()
        .unwrap_or_else(|_| unreachable!("exactly 5 pushed above"));

    // --- the head: MatMul -> Add -> Tanh -> Mul(scale) --------------------
    let matmul = sole_consumer(graph, &layer_input_name, "MatMul")?;
    let data_idx = matmul
        .input
        .iter()
        .position(|i| i == &layer_input_name)
        .ok_or("MatMul does not consume the last layer's output")?;
    let w_name = &matmul.input[1 - data_idx];
    let w_t = init
        .get(w_name.as_str())
        .ok_or("linear weight is not an initializer")?;
    let (lin_in, o_dim) = match w_t.dims.as_slice() {
        [hh, oo] => (*hh as usize, *oo as usize),
        d => {
            return Err(format!(
                "linear weight dims {d:?}, expected [hidden, n_states * n_base]"
            ));
        }
    };
    if lin_in != hidden {
        return Err(format!(
            "linear weight expects {lin_in} inputs, the LSTM stack emits {hidden}"
        ));
    }
    if o_dim != layout.n_states * layout.n_base {
        return Err(format!(
            "linear weight output width {o_dim} != n_states * n_base ({} * {})",
            layout.n_states, layout.n_base
        ));
    }
    let linear_w = tensor_f32(w_t)?;

    let add = sole_consumer(graph, &matmul.output[0], "Add")?;
    let bias_name = add
        .input
        .iter()
        .find(|i| *i != &matmul.output[0])
        .ok_or("Add has no bias input")?;
    let bias_t = init
        .get(bias_name.as_str())
        .ok_or("linear bias is not an initializer")?;
    if bias_t.dims != [o_dim as i64] {
        return Err(format!(
            "linear bias dims {:?}, expected [{o_dim}]",
            bias_t.dims
        ));
    }
    let linear_b = tensor_f32(bias_t)?;

    // `Add`'s output also feeds the dynamic shape-computation subgraph (see
    // the module doc), so it is not the sole consumer here — only its Tanh
    // consumer matters.
    let tanh = find_consumers(graph, &add.output[0])
        .into_iter()
        .find(|n| n.op_type == "Tanh")
        .ok_or("Add output has no Tanh consumer")?;

    let scale_mul = sole_consumer(graph, &tanh.output[0], "Mul")?;
    let scale_name = scale_mul
        .input
        .iter()
        .find(|i| *i != &tanh.output[0])
        .ok_or("scale Mul has no scale input")?;
    let scale =
        resolve_f32_scalar(graph, &init, scale_name).map_err(|e| format!("scale constant: {e}"))?;

    // --- the blank pad: reached from the scaled output through Reshapes only ---
    let pads: Vec<&pb::NodeProto> = graph.node.iter().filter(|n| n.op_type == "Pad").collect();
    let [pad] = pads.as_slice() else {
        return Err(format!("{} Pad nodes, expected 1", pads.len()));
    };
    trace_through_reshapes(graph, &scale_mul.output[0], &pad.input[0])
        .map_err(|e| format!("scaled output to Pad: {e}"))?;
    let blank_name = pad.input.get(2).ok_or("Pad has no constant_value input")?;
    let blank_score = resolve_f32_scalar(graph, &init, blank_name)
        .map_err(|e| format!("Pad constant_value: {e}"))?;
    if let Some(declared) = meta.crf.blank_score
        && (declared - blank_score).abs() > 1e-6
    {
        return Err(format!(
            "bundle declares crf.blank_score={declared} but the graph's Pad constant is {blank_score}"
        ));
    }

    let out_name = graph
        .output
        .first()
        .map(|o| o.name.as_str())
        .ok_or("graph has no output")?;
    trace_through_reshapes(graph, &pad.output[0], out_name)
        .map_err(|e| format!("Pad output to the graph output: {e}"))?;

    let backend = LstmBackend::best_for(hidden);
    let net = Recognized {
        chunk: meta.signal.chunk,
        t_len: meta.t_len(),
        convs,
        layers,
        hidden,
        linear_w,
        linear_b,
        scale,
        blank_score,
        n_states: layout.n_states,
        n_base: layout.n_base,
        n_edges: layout.n_edges,
        n_score: layout.n_score,
        backend,
    };
    self_check(&net, proto)?;
    Ok(net)
}

/// One seeded pseudo-random standardised window, scored through the recognised
/// native stack and through tract, must agree within a tolerance that fails a
/// wrong pad side or a wrong direction (both O(1) on scores bounded to ±5 by
/// `tanh` and the fixed blank score) while passing ordinary `exp`/`tanh`
/// vector-approximation noise accumulated over hundreds of recurrence steps.
///
/// Measured on `barcode_crf_ldx32_rna004@v0.2.1` real weights (chunk 3000,
/// `t_len` 300): max |Δ| ~2e-3 over the 300*1280 scores, three orders below
/// the O(1) failure this exists to catch — see
/// `tests::self_check_against_real_bundle` for the reproducible measurement.
const SELF_CHECK_TOLERANCE: f32 = 0.05;

fn self_check(net: &Recognized, proto: &pb::ModelProto) -> Result<(), String> {
    use tract_onnx::prelude::*;
    use tract_onnx::tract_core::framework::Framework;

    let mut rng_state = 0x9E3779B97F4A7C15u64;
    let mut next = move || {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        // A standardised-signal-shaped range: most raw pA readings land
        // within a few sigma of the fitted mean, so ±4 exercises the graph
        // without saturating every activation into its flat tail.
        ((rng_state as u32 as f32) / (u32::MAX as f32) * 2.0 - 1.0) * 4.0
    };
    let window: Vec<f32> = (0..net.chunk).map(|_| next()).collect();

    let framework = tract_onnx::onnx();
    let mut hoisted_proto = proto.clone();
    crate::onnx_rewrite::hoist_conv_padding(&mut hoisted_proto, 1);
    let plan = framework
        .model_for_proto_model(&hoisted_proto)
        .map_err(|e| format!("self-check: cannot parse proto: {e}"))?
        .with_input_fact(0, f32::fact([1, 1, net.chunk]).into())
        .map_err(|e| format!("self-check: cannot pin input shape: {e}"))?
        .into_optimized()
        .map_err(|e| format!("self-check: cannot optimize: {e}"))?
        .into_runnable()
        .map_err(|e| format!("self-check: cannot plan: {e}"))?;
    let t = Tensor::from_shape(&[1, 1, net.chunk], &window)
        .map_err(|e| format!("self-check: cannot build input tensor: {e}"))?;
    let out = plan
        .run(tvec!(t.into()))
        .map_err(|e| format!("self-check: tract inference failed: {e}"))?;
    let view = out[0]
        .to_plain_array_view::<f32>()
        .map_err(|e| format!("self-check: tract output is not f32: {e}"))?;
    if view.shape() != [net.t_len, 1, net.n_score] {
        return Err(format!(
            "self-check: tract output is {:?}, expected [{}, 1, {}]",
            view.shape(),
            net.t_len,
            net.n_score
        ));
    }
    let tract_scores: Vec<f32> = view.iter().copied().collect();

    let native_scores = net.encode(&window);
    let max_diff = tract_scores
        .iter()
        .zip(&native_scores)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    if max_diff > SELF_CHECK_TOLERANCE {
        return Err(format!(
            "self-check: native and tract disagree by {max_diff:e} on a random window, past the {SELF_CHECK_TOLERANCE:e} tolerance"
        ));
    }
    Ok(())
}

/// Walk forward from `from` through zero or more `Reshape` nodes and confirm
/// the walk reaches `to`. Anything else along the way is a graph shape this
/// recognizer does not understand — see the module doc on why the dynamic
/// shape-computation subgraph is not interpreted symbolically.
fn trace_through_reshapes(graph: &pb::GraphProto, from: &str, to: &str) -> Result<(), String> {
    let mut cur = from.to_string();
    for _ in 0..8 {
        if cur == to {
            return Ok(());
        }
        // A hop's tensor can have other consumers beyond the data path — the
        // dynamic shape-computation subgraph the module doc names (`Shape`
        // nodes feeding the `Reshape`s' own dimension arguments) reads the
        // same tensors this walk does, without joining it. Real exports
        // confirm this: `barcode_crf_ldx32_rna004`'s scaled output feeds both
        // the real `Reshape` and a `Shape` node. Only the Reshape-typed
        // consumers matter here; there must be exactly one.
        let reshapes: Vec<&pb::NodeProto> = find_consumers(graph, &cur)
            .into_iter()
            .filter(|n| n.op_type == "Reshape")
            .collect();
        match reshapes.as_slice() {
            [only] => cur = only.output[0].clone(),
            [] => return Err(format!("`{cur}` has no Reshape consumer")),
            many => {
                return Err(format!(
                    "`{cur}` has {} Reshape consumers, expected 1",
                    many.len()
                ));
            }
        }
    }
    Err(format!(
        "`{from}` does not reach `{to}` within 8 Reshape hops"
    ))
}

fn resolve_f32_scalar(
    graph: &pb::GraphProto,
    init: &HashMap<&str, &pb::TensorProto>,
    name: &str,
) -> Result<f32, String> {
    let t = match init.get(name) {
        Some(t) => *t,
        None => {
            let p =
                find_producer(graph, name).ok_or_else(|| format!("`{name}` has no producer"))?;
            if p.op_type != "Constant" {
                return Err(format!("`{name}` comes from {}, not a Constant", p.op_type));
            }
            attr_tensor(p, "value").ok_or("Constant without a value")?
        }
    };
    let v = tensor_f32(t)?;
    match v.as_slice() {
        [x] => Ok(*x),
        _ => Err(format!(
            "`{name}` holds {} values, expected a scalar",
            v.len()
        )),
    }
}

fn resolve_i64_input(
    graph: &pb::GraphProto,
    init: &HashMap<&str, &pb::TensorProto>,
    name: &str,
) -> Result<Vec<i64>, String> {
    if let Some(t) = init.get(name) {
        return tensor_i64(t);
    }
    let p = find_producer(graph, name).ok_or_else(|| format!("`{name}` has no producer"))?;
    if p.op_type != "Constant" {
        return Err(format!("`{name}` comes from {}, not a Constant", p.op_type));
    }
    tensor_i64(attr_tensor(p, "value").ok_or("Constant without a value")?)
}

/// `Slice(starts=[-1], ends=<very negative>, axes=[0], steps=[-1])` — bonito's
/// reverse trick. `ends` is checked as "very negative" rather than against
/// the exact `-9223372036854775807` sentinel one export happened to pick,
/// since any sufficiently negative bound means the same thing under a
/// step of -1: run to the start.
fn verify_reverse_slice(
    graph: &pb::GraphProto,
    init: &HashMap<&str, &pb::TensorProto>,
    slice: &pb::NodeProto,
) -> Result<(), String> {
    if slice.input.len() < 5 {
        return Err("Slice has fewer than 5 inputs (no explicit steps)".into());
    }
    let starts = resolve_i64_input(graph, init, &slice.input[1])?;
    let ends = resolve_i64_input(graph, init, &slice.input[2])?;
    let axes = resolve_i64_input(graph, init, &slice.input[3])?;
    let steps = resolve_i64_input(graph, init, &slice.input[4])?;
    if starts != [-1] || axes != [0] || steps != [-1] {
        return Err(format!(
            "Slice is not the reverse pattern (starts={starts:?} axes={axes:?} steps={steps:?})"
        ));
    }
    if !matches!(ends.as_slice(), [e] if *e < -1) {
        return Err(format!(
            "Slice ends={ends:?} is not a very-negative reverse-to-start bound"
        ));
    }
    Ok(())
}

fn verify_squeeze_axis1(
    graph: &pb::GraphProto,
    init: &HashMap<&str, &pb::TensorProto>,
    squeeze: &pb::NodeProto,
) -> Result<(), String> {
    let axes = match squeeze.input.get(1).map(String::as_str) {
        Some("") | None => attr_ints(squeeze, "axes")
            .map(<[i64]>::to_vec)
            .ok_or("Squeeze has no axes")?,
        Some(name) => resolve_i64_input(graph, init, name)?,
    };
    if axes != [1] {
        return Err(format!(
            "Squeeze axes {axes:?}, expected [1] (the num_directions axis)"
        ));
    }
    Ok(())
}

fn find_producer<'a>(graph: &'a pb::GraphProto, name: &str) -> Option<&'a pb::NodeProto> {
    graph
        .node
        .iter()
        .find(|n| n.output.iter().any(|o| o == name))
}

fn find_consumers<'a>(graph: &'a pb::GraphProto, name: &str) -> Vec<&'a pb::NodeProto> {
    graph
        .node
        .iter()
        .filter(|n| n.input.iter().any(|i| i == name))
        .collect()
}

fn sole_consumer<'a>(
    graph: &'a pb::GraphProto,
    name: &str,
    op: &str,
) -> Result<&'a pb::NodeProto, String> {
    match find_consumers(graph, name).as_slice() {
        [only] if only.op_type == op => Ok(only),
        [only] => Err(format!("`{name}` feeds {} rather than {op}", only.op_type)),
        cs => Err(format!(
            "`{name}` has {} consumers, expected one {op}",
            cs.len()
        )),
    }
}

fn attr<'a>(node: &'a pb::NodeProto, name: &str) -> Option<&'a pb::AttributeProto> {
    node.attribute.iter().find(|a| a.name == name)
}

fn attr_int(node: &pb::NodeProto, name: &str) -> Option<i64> {
    attr(node, name).map(|a| a.i)
}

fn attr_str<'a>(node: &'a pb::NodeProto, name: &str) -> Option<&'a str> {
    attr(node, name).and_then(|a| std::str::from_utf8(&a.s).ok())
}

fn attr_ints<'a>(node: &'a pb::NodeProto, name: &str) -> Option<&'a [i64]> {
    attr(node, name).map(|a| a.ints.as_slice())
}

fn attr_tensor<'a>(node: &'a pb::NodeProto, name: &str) -> Option<&'a pb::TensorProto> {
    attr(node, name).and_then(|a| a.t.as_ref())
}

/// A float initializer's values, from whichever field the export used.
fn tensor_f32(t: &pb::TensorProto) -> Result<Vec<f32>, String> {
    if t.data_type != ONNX_FLOAT {
        return Err(format!("`{}` is not float32", t.name));
    }
    let n: usize = t.dims.iter().map(|&d| d.max(0) as usize).product();
    let v: Vec<f32> = if !t.float_data.is_empty() {
        t.float_data.clone()
    } else {
        t.raw_data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect()
    };
    if v.len() != n {
        return Err(format!(
            "`{}` holds {} values for dims {:?}",
            t.name,
            v.len(),
            t.dims
        ));
    }
    Ok(v)
}

fn tensor_i64(t: &pb::TensorProto) -> Result<Vec<i64>, String> {
    if t.data_type != ONNX_INT64 {
        return Err(format!("`{}` is not int64", t.name));
    }
    Ok(if !t.int64_data.is_empty() {
        t.int64_data.clone()
    } else {
        t.raw_data
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| i64::from_le_bytes(*c))
            .collect()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tract_onnx::prelude::*;

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

    fn f32_init(name: &str, dims: &[i64], rng: &mut Rng, scale: f32) -> pb::TensorProto {
        let n: usize = dims.iter().map(|&d| d as usize).product();
        pb::TensorProto {
            name: name.into(),
            dims: dims.to_vec(),
            data_type: ONNX_FLOAT,
            float_data: (0..n).map(|_| rng.next(scale)).collect(),
            ..Default::default()
        }
    }

    /// A rank-0 scalar tensor — ONNX `Pad`'s `constant_value` input demands
    /// exactly rank 0, not a rank-1 tensor holding one value.
    fn f32_scalar_init(name: &str, v: f32) -> pb::TensorProto {
        pb::TensorProto {
            name: name.into(),
            dims: vec![],
            data_type: ONNX_FLOAT,
            float_data: vec![v],
            ..Default::default()
        }
    }

    fn i64_init(name: &str, vals: &[i64]) -> pb::TensorProto {
        pb::TensorProto {
            name: name.into(),
            dims: vec![vals.len() as i64],
            data_type: ONNX_INT64,
            int64_data: vals.to_vec(),
            ..Default::default()
        }
    }

    fn node(
        op: &str,
        inputs: &[&str],
        outputs: &[&str],
        attrs: Vec<pb::AttributeProto>,
    ) -> pb::NodeProto {
        pb::NodeProto {
            op_type: op.into(),
            name: format!("{op}_{}", outputs[0]),
            input: inputs.iter().map(|s| s.to_string()).collect(),
            output: outputs.iter().map(|s| s.to_string()).collect(),
            attribute: attrs,
            ..Default::default()
        }
    }

    fn a_int(name: &str, i: i64) -> pb::AttributeProto {
        pb::AttributeProto {
            name: name.into(),
            r#type: pb::attribute_proto::AttributeType::Int as i32,
            i,
            ..Default::default()
        }
    }

    fn a_ints(name: &str, ints: &[i64]) -> pb::AttributeProto {
        pb::AttributeProto {
            name: name.into(),
            r#type: pb::attribute_proto::AttributeType::Ints as i32,
            ints: ints.to_vec(),
            ..Default::default()
        }
    }

    fn a_tensor(name: &str, t: pb::TensorProto) -> pb::AttributeProto {
        pb::AttributeProto {
            name: name.into(),
            r#type: pb::attribute_proto::AttributeType::Tensor as i32,
            t: Some(t),
            ..Default::default()
        }
    }

    fn value_info(name: &str, dims: &[i64]) -> pb::ValueInfoProto {
        pb::ValueInfoProto {
            name: name.into(),
            r#type: Some(pb::TypeProto {
                value: Some(pb::type_proto::Value::TensorType(pb::type_proto::Tensor {
                    elem_type: ONNX_FLOAT,
                    shape: Some(pb::TensorShapeProto {
                        dim: dims
                            .iter()
                            .map(|&d| pb::tensor_shape_proto::Dimension {
                                value: Some(pb::tensor_shape_proto::dimension::Value::DimValue(d)),
                                ..Default::default()
                            })
                            .collect(),
                    }),
                })),
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    /// Geometry for a synthetic export. Small enough to keep the tract
    /// self-check (run automatically by every successful `from_proto`) fast,
    /// while `hidden` stays a multiple of 16 so AVX-512 dispatches wherever
    /// the machine has it.
    struct Spec {
        chunk: usize,
        stride: usize,
        c1: usize,
        c2: usize,
        hidden: usize,
        n_base: usize,
        state_len: usize,
        /// Which of the 5 layers wrap the reverse-Slice pattern.
        reverse: [bool; 5],
        scale: f32,
        blank_score: f32,
    }

    impl Spec {
        fn default_spec() -> Self {
            Spec {
                chunk: 300,
                stride: 10,
                c1: 4,
                c2: 8,
                hidden: 32,
                n_base: 4,
                state_len: 2,
                reverse: [true, false, true, false, true],
                scale: 5.0,
                blank_score: 2.0,
            }
        }

        fn t_len(&self) -> usize {
            self.chunk / self.stride
        }

        fn layout(&self) -> CrfLayout {
            CrfLayout::new(self.n_base, self.state_len).expect("valid test geometry")
        }

        fn meta(&self) -> CrfMetadata {
            let json = format!(
                r#"{{
                  "standardisation": {{"mean": 0.0, "stdev": 1.0}},
                  "signal": {{"chunk": {}, "stride": {}}},
                  "crf": {{"state_len": {}, "n_base": {}, "blank_score": {},
                          "alphabet": ["N", "A", "C", "G", "T"]}}
                }}"#,
                self.chunk, self.stride, self.state_len, self.n_base, self.blank_score
            );
            serde_json::from_str(&json).expect("well-formed test metadata")
        }
    }

    /// One `Conv(in_c -> out_c)` + SiLU stage's nodes and initializers,
    /// appended in place. Returns the SiLU output's name.
    #[allow(clippy::too_many_arguments)]
    fn conv_silu_stage(
        nodes: &mut Vec<pb::NodeProto>,
        inits: &mut Vec<pb::TensorProto>,
        rng: &mut Rng,
        stage: usize,
        input: &str,
        in_c: usize,
        out_c: usize,
        k: usize,
        pad: usize,
        stride: usize,
    ) -> String {
        let w_name = format!("conv{stage}_w");
        let b_name = format!("conv{stage}_b");
        inits.push(f32_init(
            &w_name,
            &[out_c as i64, in_c as i64, k as i64],
            rng,
            0.3,
        ));
        inits.push(f32_init(&b_name, &[out_c as i64], rng, 0.1));
        let conv_out = format!("conv{stage}_out");
        let mut attrs = vec![
            a_ints("kernel_shape", &[k as i64]),
            a_ints("pads", &[pad as i64, pad as i64]),
        ];
        if stride != 1 {
            attrs.push(a_ints("strides", &[stride as i64]));
        }
        nodes.push(node(
            "Conv",
            &[input, &w_name, &b_name],
            &[&conv_out],
            attrs,
        ));
        let sig_out = format!("conv{stage}_sig");
        nodes.push(node("Sigmoid", &[&conv_out], &[&sig_out], vec![]));
        let silu_out = format!("conv{stage}_silu");
        nodes.push(node("Mul", &[&conv_out, &sig_out], &[&silu_out], vec![]));
        silu_out
    }

    /// The full synthetic export: 3 Conv+SiLU stages, 5 stacked unidirectional
    /// LSTM layers (reversed per `spec.reverse`), and the linear head with its
    /// blank pad — the same shape `try_match` recognises, built by hand so
    /// tests do not depend on a real bundle.
    fn build_proto(spec: &Spec, seed: u64) -> pb::ModelProto {
        let mut rng = Rng(seed | 1);
        let mut nodes = Vec::new();
        let mut inits = vec![
            i64_init("rev_starts", &[-1]),
            i64_init("rev_ends", &[-1_000_000_000]),
            i64_init("rev_axes", &[0]),
            i64_init("rev_steps", &[-1]),
            i64_init("squeeze_axes", &[1]),
        ];

        let (h, hh) = (spec.hidden, spec.hidden as i64);
        // `zshape` is an initializer directly, not a `Constant` node's
        // output — simpler, and `try_match` never inspects a `ConstantOfShape`
        // input's own producer, only the `ConstantOfShape` node itself.
        inits.push(pb::TensorProto {
            name: "zshape".into(),
            dims: vec![3],
            data_type: ONNX_INT64,
            int64_data: vec![1, 1, hh],
            ..Default::default()
        });
        nodes.push(node(
            "ConstantOfShape",
            &["zshape"],
            &["zeros"],
            vec![a_tensor(
                "value",
                pb::TensorProto {
                    dims: vec![1],
                    data_type: ONNX_FLOAT,
                    float_data: vec![0.0],
                    ..Default::default()
                },
            )],
        ));

        let mut cur = conv_silu_stage(
            &mut nodes, &mut inits, &mut rng, 0, "signal", 1, spec.c1, 5, 2, 1,
        );
        cur = conv_silu_stage(
            &mut nodes, &mut inits, &mut rng, 1, &cur, spec.c1, spec.c2, 5, 2, 1,
        );
        cur = conv_silu_stage(
            &mut nodes,
            &mut inits,
            &mut rng,
            2,
            &cur,
            spec.c2,
            h,
            11,
            5,
            spec.stride,
        );

        nodes.push(node(
            "Transpose",
            &[&cur],
            &["lstm_in0"],
            vec![a_ints("perm", &[2, 0, 1])],
        ));
        let mut layer_in = "lstm_in0".to_string();

        for (i, &reverse) in spec.reverse.iter().enumerate() {
            let (w, r, b) = (
                format!("layer{i}_w"),
                format!("layer{i}_r"),
                format!("layer{i}_b"),
            );
            inits.push(f32_init(&w, &[1, 4 * hh, hh], &mut rng, 0.5));
            inits.push(f32_init(&r, &[1, 4 * hh, hh], &mut rng, 0.3));
            inits.push(f32_init(&b, &[1, 8 * hh], &mut rng, 0.2));

            let lstm_data_in = if reverse {
                let pre = format!("layer{i}_pre_rev");
                nodes.push(node(
                    "Slice",
                    &[&layer_in, "rev_starts", "rev_ends", "rev_axes", "rev_steps"],
                    &[&pre],
                    vec![],
                ));
                pre
            } else {
                layer_in.clone()
            };

            let y = format!("layer{i}_y");
            let (yh, yc) = (format!("layer{i}_yh"), format!("layer{i}_yc"));
            nodes.push(node(
                "LSTM",
                &[&lstm_data_in, &w, &r, &b, "", "zeros", "zeros"],
                &[&y, &yh, &yc],
                vec![a_int("hidden_size", hh)],
            ));

            let squeeze = format!("layer{i}_squeeze");
            nodes.push(node("Squeeze", &[&y, "squeeze_axes"], &[&squeeze], vec![]));

            layer_in = if reverse {
                let post = format!("layer{i}_post_rev");
                nodes.push(node(
                    "Slice",
                    &[&squeeze, "rev_starts", "rev_ends", "rev_axes", "rev_steps"],
                    &[&post],
                    vec![],
                ));
                post
            } else {
                squeeze
            };
        }

        let layout = spec.layout();
        let o_dim = (layout.n_states * layout.n_base) as i64;
        inits.push(f32_init("linear_w", &[hh, o_dim], &mut rng, 0.3));
        inits.push(f32_init("linear_b", &[o_dim], &mut rng, 0.1));
        nodes.push(node(
            "MatMul",
            &[&layer_in, "linear_w"],
            &["matmul_out"],
            vec![],
        ));
        nodes.push(node(
            "Add",
            &["matmul_out", "linear_b"],
            &["add_out"],
            vec![],
        ));
        nodes.push(node("Tanh", &["add_out"], &["tanh_out"], vec![]));
        inits.push(f32_scalar_init("scale_const", spec.scale));
        nodes.push(node(
            "Mul",
            &["tanh_out", "scale_const"],
            &["scale_out"],
            vec![],
        ));

        let t_len = spec.t_len() as i64;
        inits.push(i64_init(
            "reshape1_shape",
            &[t_len, 1, layout.n_states as i64, layout.n_base as i64],
        ));
        nodes.push(node(
            "Reshape",
            &["scale_out", "reshape1_shape"],
            &["resh1_out"],
            vec![],
        ));
        inits.push(i64_init("pad_pads", &[0, 0, 0, 1, 0, 0, 0, 0]));
        inits.push(f32_scalar_init("pad_const", spec.blank_score));
        nodes.push(node(
            "Pad",
            &["resh1_out", "pad_pads", "pad_const"],
            &["pad_out"],
            vec![],
        ));
        inits.push(i64_init(
            "reshape2_shape",
            &[t_len, 1, layout.n_score as i64],
        ));
        nodes.push(node(
            "Reshape",
            &["pad_out", "reshape2_shape"],
            &["final_out"],
            vec![],
        ));

        pb::ModelProto {
            ir_version: 8,
            opset_import: vec![pb::OperatorSetIdProto {
                domain: String::new(),
                version: 17,
            }],
            graph: Some(pb::GraphProto {
                node: nodes,
                initializer: inits,
                input: vec![value_info("signal", &[1, 1, spec.chunk as i64])],
                output: vec![value_info("final_out", &[t_len, 1, layout.n_score as i64])],
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    fn tract_run(proto: &pb::ModelProto, chunk: usize, window: &[f32]) -> Vec<f32> {
        use tract_onnx::tract_core::framework::Framework;
        let framework = tract_onnx::onnx();
        let plan = framework
            .model_for_proto_model(proto)
            .unwrap()
            .with_input_fact(0, f32::fact([1, 1, chunk]).into())
            .unwrap()
            .into_optimized()
            .unwrap()
            .into_runnable()
            .unwrap();
        let t = Tensor::from_shape(&[1, 1, chunk], window).unwrap();
        let out = plan.run(tvec!(t.into())).unwrap();
        out[0]
            .to_plain_array_view::<f32>()
            .unwrap()
            .iter()
            .copied()
            .collect()
    }

    fn window(spec: &Spec, seed: u64) -> Vec<f32> {
        let mut rng = Rng(seed | 1);
        (0..spec.chunk).map(|_| rng.next(4.0)).collect()
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(p, q)| (p - q).abs())
            .fold(0.0, f32::max)
    }

    /// The recognizer matches the real graph shape and its native scores
    /// agree with tract's, on random weights and several random windows.
    #[test]
    fn native_matches_tract_on_random_weights() {
        let spec = Spec::default_spec();
        let proto = build_proto(&spec, 7);
        let net = Recognized::from_proto(&proto, &spec.meta(), &spec.layout()).expect("recognised");
        assert_eq!(net.blank_score(), spec.blank_score);
        for seed in 1..=6u64 {
            let w = window(&spec, seed);
            let want = tract_run(&proto, spec.chunk, &w);
            let got = net.encode(&w);
            let d = max_abs_diff(&got, &want);
            assert!(d < 0.05, "seed {seed}: {d:e}");
        }
    }

    /// Whichever layers wrap the reverse-Slice pattern, the recognizer reads
    /// it from the graph rather than assuming layers 0/2/4 — an inverted
    /// pattern still agrees with tract.
    #[test]
    fn reverse_layers_follow_the_slices() {
        let mut spec = Spec::default_spec();
        spec.reverse = [false, true, false, true, false];
        let proto = build_proto(&spec, 9);
        let net = Recognized::from_proto(&proto, &spec.meta(), &spec.layout()).expect("recognised");
        let w = window(&spec, 2);
        let want = tract_run(&proto, spec.chunk, &w);
        let got = net.encode(&w);
        assert!(max_abs_diff(&got, &want) < 0.05);

        // A graph with no reversed layers at all is just as valid a shape.
        let mut spec_fwd = Spec::default_spec();
        spec_fwd.reverse = [false; 5];
        let proto_fwd = build_proto(&spec_fwd, 10);
        let net_fwd = Recognized::from_proto(&proto_fwd, &spec_fwd.meta(), &spec_fwd.layout())
            .expect("recognised");
        let w = window(&spec_fwd, 3);
        let want = tract_run(&proto_fwd, spec_fwd.chunk, &w);
        assert!(max_abs_diff(&net_fwd.encode(&w), &want) < 0.05);
    }

    /// The blank score the recognizer carries is read from the graph's `Pad`
    /// constant, lands at `state * n_edges` (edge 0) in every timestep's row,
    /// and a bundle whose declared `crf.blank_score` disagrees is refused.
    #[test]
    fn blank_pad_matches_layout() {
        let spec = Spec::default_spec();
        let proto = build_proto(&spec, 13);
        let layout = spec.layout();
        let net = Recognized::from_proto(&proto, &spec.meta(), &layout).expect("recognised");
        assert_eq!(net.blank_score(), spec.blank_score);

        let w = window(&spec, 4);
        let scores = net.encode(&w);
        for t in 0..spec.t_len() {
            for state in 0..layout.n_states {
                let v = scores[t * layout.n_score + state * layout.n_edges];
                assert_eq!(v, spec.blank_score, "t={t} state={state}");
            }
        }

        // A bundle that declares a different blank score than the graph's
        // own `Pad` constant is refused rather than silently trusted.
        let mut mismatched = spec.meta();
        mismatched.crf.blank_score = Some(spec.blank_score + 1.0);
        assert!(Recognized::from_proto(&proto, &mismatched, &layout).is_none());
    }

    /// Scoring reads in lockstep changes nothing about any read's scores: the
    /// batched entry point is bit-identical to the single-read one, for every
    /// group size up to and past the preferred batch, on every backend the
    /// machine has — not only the one the dispatch prefers (see
    /// `escapepod_signal::lstm`'s own version of this test for why that
    /// matters).
    #[test]
    fn batched_matches_single_bit_for_bit() {
        let spec = Spec::default_spec();
        let proto = build_proto(&spec, 21);
        for backend in LstmBackend::available() {
            let net = Recognized::from_proto(&proto, &spec.meta(), &spec.layout())
                .expect("recognised")
                .with_backend(backend)
                .unwrap();
            for n_reads in 1..=net.preferred_batch() + 1 {
                let windows: Vec<Vec<f32>> = (1..=n_reads as u64)
                    .map(|s| window(&spec, 40 + s))
                    .collect();
                let refs: Vec<&[f32]> = windows.iter().map(Vec::as_slice).collect();
                let batched = net.encode_batch(&refs);
                for (r, w) in windows.iter().enumerate() {
                    let single = net.encode(w);
                    assert_eq!(
                        single.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        batched[r].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        "{backend:?} n_reads={n_reads} read {r}"
                    );
                }
            }
        }
    }

    /// The 16-wide kernel and the 8-wide one must agree bit for bit through
    /// the whole stack, single-read and batched, wherever the machine has
    /// AVX-512F. Skips (and says so) where it does not, e.g. CI.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx512_matches_avx2_bit_for_bit() {
        if !LstmBackend::Avx512.supported() {
            eprintln!("no AVX-512F on this machine: cross-width pin not exercised");
            return;
        }
        let spec = Spec::default_spec();
        let proto = build_proto(&spec, 33);
        let wide = Recognized::from_proto(&proto, &spec.meta(), &spec.layout())
            .unwrap()
            .with_backend(LstmBackend::Avx512)
            .unwrap();
        let narrow = Recognized::from_proto(&proto, &spec.meta(), &spec.layout())
            .unwrap()
            .with_backend(LstmBackend::Avx2)
            .unwrap();
        assert_eq!(wide.preferred_batch(), 8);
        for n_reads in 1..=9usize {
            let windows: Vec<Vec<f32>> = (1..=n_reads as u64)
                .map(|s| window(&spec, 70 + s))
                .collect();
            let refs: Vec<&[f32]> = windows.iter().map(Vec::as_slice).collect();
            let a = wide.encode_batch(&refs);
            let b = narrow.encode_batch(&refs);
            for r in 0..n_reads {
                assert_eq!(
                    a[r].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    b[r].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    "n_reads={n_reads} read {r}"
                );
            }
        }
    }

    /// Anything outside the recognised shape falls back to tract rather than
    /// decoding the wrong thing, and every mutation here is still a graph a
    /// general ONNX runtime (tract, unconstrained) loads and runs fine — the
    /// refusal is about this native kernel's narrower contract, not about the
    /// graph being broken.
    #[test]
    fn unsupported_graph_falls_back_to_tract() {
        let spec = Spec::default_spec();

        // Wrong Conv count.
        let mut m = build_proto(&spec, 1);
        let g = m.graph.as_mut().unwrap();
        let first_conv = g.node.iter().position(|n| n.op_type == "Conv").unwrap();
        g.node.remove(first_conv);
        assert!(Recognized::from_proto(&m, &spec.meta(), &spec.layout()).is_none());

        // Wrong LSTM count.
        let mut m = build_proto(&spec, 1);
        let g = m.graph.as_mut().unwrap();
        let first_lstm = g.node.iter().position(|n| n.op_type == "LSTM").unwrap();
        g.node.remove(first_lstm);
        assert!(Recognized::from_proto(&m, &spec.meta(), &spec.layout()).is_none());

        // A `clip` attribute on one LSTM — a valid ONNX graph tract runs fine.
        let mut m = build_proto(&spec, 1);
        for n in &mut m.graph.as_mut().unwrap().node {
            if n.op_type == "LSTM" {
                n.attribute.push(a_int("clip", 5));
                break;
            }
        }
        assert!(Recognized::from_proto(&m, &spec.meta(), &spec.layout()).is_none());
        let w = window(&spec, 5);
        let _ = tract_run(&m, spec.chunk, &w); // does not panic: a valid graph

        // A non-zero initial state — also a valid ONNX graph.
        let mut m = build_proto(&spec, 1);
        for n in &mut m.graph.as_mut().unwrap().node {
            if n.op_type == "ConstantOfShape" {
                n.attribute = vec![a_tensor(
                    "value",
                    pb::TensorProto {
                        dims: vec![1],
                        data_type: ONNX_FLOAT,
                        float_data: vec![0.1],
                        ..Default::default()
                    },
                )];
            }
        }
        assert!(Recognized::from_proto(&m, &spec.meta(), &spec.layout()).is_none());
        let _ = tract_run(&m, spec.chunk, &w);

        // No Pad node at all.
        let mut m = build_proto(&spec, 1);
        m.graph
            .as_mut()
            .unwrap()
            .node
            .retain(|n| n.op_type != "Pad");
        assert!(Recognized::from_proto(&m, &spec.meta(), &spec.layout()).is_none());

        // The bundle's declared geometry does not match the graph's.
        let m = build_proto(&spec, 1);
        let mut wrong_layout = spec.layout();
        wrong_layout = CrfLayout::new(wrong_layout.n_base, wrong_layout.state_len + 1).unwrap();
        assert!(Recognized::from_proto(&m, &spec.meta(), &wrong_layout).is_none());
    }
}
