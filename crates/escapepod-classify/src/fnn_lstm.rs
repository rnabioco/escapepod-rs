// SPDX-License-Identifier: MIT

//! A native bidirectional LSTM for the `feature_model` variant.
//!
//! The shipped charging network (`charging_feature_nn_rna004`, `arch: lstm`)
//! is a bidirectional LSTM with 96 hidden units over 33 timesteps of 4
//! channels, read out as the mean and the max over time of the two
//! directions' outputs, then one linear layer to two logits. Through tract
//! that graph costs ~490 µs per read, and a flat profile of `escpod classify`
//! puts ~34% of the whole command's CPU in it — of which only ~14% is the
//! matmul, sigmoid and tanh kernels. The rest is per-timestep bookkeeping:
//! `Scan` state evaluation, plan execution, tensor allocation, dynamic-rank
//! `ndarray` iteration, symbolic-dimension evaluation. An unrolled graph was
//! measured 3× *worse*, so there is no structural fix on tract's side; the
//! lever is to run the recurrence directly.
//!
//! [`NativeBiLstm::from_proto`] recognises exactly that graph in the ONNX
//! proto — one bidirectional `LSTM`, fed by a transpose of the `[batch,
//! channel, offset]` input, read out by `ReduceMean` + `ReduceMax` over time
//! into a `Concat` and a `Gemm` — and lifts its weights out. Anything else
//! returns `None` and the caller keeps tract, so a bundle whose export
//! differs in any way is scored the slow, general way rather than wrongly.
//! Every constraint the match applies is one ONNX's LSTM semantics let an
//! export vary (peepholes, custom activations, clipping, `input_forget`, a
//! non-zero initial state, per-sequence lengths, the `layout` flag); each is
//! refused explicitly rather than assumed absent.
//!
//! The kernel is the axpy form of the recurrence: for each hidden unit `j`,
//! the previous state `h[j]` scales row `j` of `Rᵀ` into the 4H gate
//! pre-activations, so every inner loop is a unit-stride fused multiply-add
//! and nothing is horizontally reduced. The input contribution does not
//! depend on the recurrence and is computed for all timesteps up front with
//! both bias halves folded in. AVX2+FMA is runtime-dispatched per the build
//! policy (never a baseline bump); the scalar path is the reference and the
//! fallback.
//!
//! Numerics: ONNX gate order is `i, o, f, c`; the default activations are
//! sigmoid, tanh, tanh. The vector path uses a Cephes-style `exp` — the same
//! construction `escapepod_demux::crf::avx2` uses — and `tanh(x) = 2σ(2x) −
//! 1`, so it is not bit-identical to the scalar path, or to tract, which uses
//! its own vector approximations. The contract is agreement on the
//! probability within the parity tolerance the feature grid already carries
//! (1e-4), pinned by the tests below against tract on the real graph shape.

use anyhow::{Result, bail};
use std::cell::RefCell;
use std::collections::HashMap;
use tract_onnx::pb;

const ONNX_FLOAT: i32 = 1;
const ONNX_INT64: i32 = 7;

/// Which kernel scores a read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    Scalar,
    #[cfg(target_arch = "x86_64")]
    Avx2,
}

impl Backend {
    /// The fastest kernel this machine can run for a hidden size.
    ///
    /// The AVX2 activations are 8-wide, so a hidden size that is not a
    /// multiple of 8 stays scalar rather than growing a masked tail nobody
    /// ships.
    pub fn best_for(h: usize) -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            if h.is_multiple_of(8)
                && is_x86_feature_detected!("avx2")
                && is_x86_feature_detected!("fma")
            {
                return Backend::Avx2;
            }
        }
        let _ = h;
        Backend::Scalar
    }

    pub fn name(self) -> &'static str {
        match self {
            Backend::Scalar => "scalar",
            #[cfg(target_arch = "x86_64")]
            Backend::Avx2 => "avx2",
        }
    }
}

/// The recognised graph's weights, repacked for the step loop.
#[derive(Debug)]
pub struct NativeBiLstm {
    /// Timesteps (the offset axis).
    seq: usize,
    /// Input channels per timestep.
    n_in: usize,
    /// Hidden units per direction.
    h: usize,
    /// Per direction: `W` transposed to `[n_in][4H]`, so the input
    /// contribution for one timestep is `n_in` axpys over a 4H vector.
    wt: [Vec<f32>; 2],
    /// Per direction: `[4H]`, both ONNX bias halves summed.
    bias: [Vec<f32>; 2],
    /// Per direction: `R` transposed to `[H][4H]`.
    rt: [Vec<f32>; 2],
    /// Per direction: the same `Rᵀ` packed block-major, `[4H/32][H][32]`,
    /// so one 32-gate pass over all `H` rows streams a contiguous 12 KB
    /// instead of 32 floats out of every 1.5 KB row — worth 14% of the
    /// single-read kernel and 11% of the batched one. Empty unless `4H` is
    /// a multiple of 32 (the AVX2 backend's precondition, `H % 8 == 0`); the
    /// scalar path keeps reading `rt`.
    rt_blocked: [Vec<f32>; 2],
    /// `[n_cls][4H]` over the readout in the order the graph concatenates.
    head_w: Vec<f32>,
    head_b: Vec<f32>,
    n_cls: usize,
    /// Whether the `Concat` puts the mean block before the max block.
    mean_first: bool,
    backend: Backend,
}

// Per-thread scratch: one read's input contribution, per-timestep outputs
// and gate buffer. Sized on first use and kept, so the steady state
// allocates nothing.
thread_local! {
    static SCRATCH: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

impl NativeBiLstm {
    /// Lift the weights out of a proto whose graph is the recognised shape.
    ///
    /// `n_ch` and `n_off` are the input the bundle declares (channels,
    /// offsets); the graph input, when its dimensions are static, must agree.
    /// `None` means "not this graph" and is logged at debug level with the
    /// reason; the caller falls back to tract.
    pub fn from_proto(proto: &pb::ModelProto, n_ch: usize, n_off: usize) -> Option<Self> {
        match Self::try_match(proto, n_ch, n_off) {
            Ok(net) => Some(net),
            Err(why) => {
                tracing::debug!("feature model is not a native-BiLSTM graph: {why}");
                None
            }
        }
    }

    pub fn backend(&self) -> Backend {
        self.backend
    }

    /// Choose the kernel explicitly (tests; equivalence checks).
    pub fn with_backend(mut self, backend: Backend) -> Self {
        #[cfg(target_arch = "x86_64")]
        if backend == Backend::Avx2 {
            assert!(
                self.h.is_multiple_of(8),
                "the AVX2 kernel needs a hidden size that is a multiple of 8"
            );
        }
        self.backend = backend;
        self
    }

    pub fn hidden(&self) -> usize {
        self.h
    }

    pub fn n_classes(&self) -> usize {
        self.n_cls
    }

    fn try_match(
        proto: &pb::ModelProto,
        n_ch: usize,
        n_off: usize,
    ) -> std::result::Result<Self, String> {
        let graph = proto.graph.as_ref().ok_or("no graph")?;
        let init: HashMap<&str, &pb::TensorProto> = graph
            .initializer
            .iter()
            .map(|t| (t.name.as_str(), t))
            .collect();
        let producer = |name: &str| {
            graph
                .node
                .iter()
                .find(|n| n.output.iter().any(|o| o == name))
        };
        let consumers = |name: &str| -> Vec<&pb::NodeProto> {
            graph
                .node
                .iter()
                .filter(|n| n.input.iter().any(|i| i == name))
                .collect()
        };
        let sole_consumer = |name: &str, op: &str| -> std::result::Result<&pb::NodeProto, String> {
            let c = consumers(name);
            match c.as_slice() {
                [only] if only.op_type == op => Ok(only),
                [only] => Err(format!("{name} feeds {} rather than {op}", only.op_type)),
                _ => Err(format!(
                    "{name} has {} consumers, expected one {op}",
                    c.len()
                )),
            }
        };

        // --- the LSTM itself ------------------------------------------------
        let lstms: Vec<&pb::NodeProto> =
            graph.node.iter().filter(|n| n.op_type == "LSTM").collect();
        let lstm = match lstms.as_slice() {
            [one] => *one,
            _ => return Err(format!("{} LSTM nodes", lstms.len())),
        };
        if attr_str(lstm, "direction").unwrap_or("forward") != "bidirectional" {
            return Err("LSTM is not bidirectional".into());
        }
        let h = attr_int(lstm, "hidden_size").ok_or("LSTM has no hidden_size")? as usize;
        if h == 0 {
            return Err("hidden_size is 0".into());
        }
        for forbidden in ["activations", "activation_alpha", "activation_beta", "clip"] {
            if lstm.attribute.iter().any(|a| a.name == forbidden) {
                return Err(format!("LSTM sets `{forbidden}`"));
            }
        }
        if attr_int(lstm, "input_forget").unwrap_or(0) != 0 {
            return Err("LSTM couples input and forget gates".into());
        }
        if attr_int(lstm, "layout").unwrap_or(0) != 0 {
            return Err("LSTM uses batch-major layout".into());
        }
        let input_at = |i: usize| lstm.input.get(i).map(String::as_str).unwrap_or("");
        if lstm.input.len() < 3 {
            return Err("LSTM has fewer than 3 inputs".into());
        }
        let g = 4 * h;
        let w_t = init.get(input_at(1)).ok_or("W is not an initializer")?;
        let r_t = init.get(input_at(2)).ok_or("R is not an initializer")?;
        let n_in = match w_t.dims.as_slice() {
            [2, gg, n_in] if *gg as usize == g => *n_in as usize,
            d => return Err(format!("W dims {d:?}, expected [2, {g}, n_in]")),
        };
        if r_t.dims != [2, g as i64, h as i64] {
            return Err(format!("R dims {:?}, expected [2, {g}, {h}]", r_t.dims));
        }
        let w = tensor_f32(w_t)?;
        let r = tensor_f32(r_t)?;
        let b = match input_at(3) {
            "" => vec![0.0f32; 2 * 8 * h],
            name => {
                let t = init.get(name).ok_or("B is not an initializer")?;
                if t.dims != [2, 8 * h as i64] {
                    return Err(format!("B dims {:?}, expected [2, {}]", t.dims, 8 * h));
                }
                tensor_f32(t)?
            }
        };
        if !input_at(4).is_empty() {
            return Err("LSTM has per-sequence lengths".into());
        }
        for (i, what) in [(5, "initial_h"), (6, "initial_c")] {
            let name = input_at(i);
            if name.is_empty() {
                continue;
            }
            let p = producer(name).ok_or(format!("{what} has no producer"))?;
            if p.op_type != "ConstantOfShape" {
                return Err(format!("{what} comes from {}, not zeros", p.op_type));
            }
            if let Some(t) = attr_tensor(p, "value")
                && tensor_f32(t)?.iter().any(|&v| v != 0.0)
            {
                return Err(format!("{what} is a non-zero constant"));
            }
        }
        if !input_at(7).is_empty() {
            return Err("LSTM has peephole weights".into());
        }

        // --- the input: features[b, ch, off] -> X[off, b, ch] -------------
        let x_src = producer(input_at(0)).ok_or("LSTM input has no producer")?;
        if x_src.op_type != "Transpose" || attr_ints(x_src, "perm") != Some(&[2, 0, 1]) {
            return Err("LSTM input is not Transpose(perm=[2,0,1]) of the graph input".into());
        }
        let feat = x_src.input.first().map(String::as_str).unwrap_or("");
        let gin = graph
            .input
            .iter()
            .find(|v| v.name == feat)
            .ok_or("the transposed tensor is not a graph input")?;
        if let Some(dims) = value_dims(gin) {
            if dims.len() != 3 {
                return Err(format!("graph input has rank {}", dims.len()));
            }
            for (axis, want) in [(1usize, n_ch), (2, n_off)] {
                if let Some(d) = dims[axis]
                    && d != want
                {
                    return Err(format!(
                        "graph input axis {axis} is {d}, bundle declares {want}"
                    ));
                }
            }
        }
        if n_in != n_ch {
            return Err(format!("W expects {n_in} channels, bundle declares {n_ch}"));
        }

        // --- the readout: Y -> [seq, b, 2H] -> mean & max over seq --------
        let y = lstm.output.first().map(String::as_str).unwrap_or("");
        for extra in lstm.output.iter().skip(1) {
            if !extra.is_empty() && !consumers(extra).is_empty() {
                return Err(format!("LSTM output `{extra}` is consumed"));
            }
        }
        let t1 = sole_consumer(y, "Transpose")?;
        if attr_ints(t1, "perm") != Some(&[0, 2, 1, 3]) {
            return Err("first readout Transpose is not perm=[0,2,1,3]".into());
        }
        let rs = sole_consumer(&t1.output[0], "Reshape")?;
        let shape_name = rs.input.get(1).map(String::as_str).unwrap_or("");
        let shape = match init.get(shape_name) {
            Some(t) => tensor_i64(t)?,
            None => {
                let c = producer(shape_name).ok_or("Reshape shape has no producer")?;
                if c.op_type != "Constant" {
                    return Err(format!("Reshape shape comes from {}", c.op_type));
                }
                tensor_i64(attr_tensor(c, "value").ok_or("Constant without a value")?)?
            }
        };
        if shape != [0, 0, -1] {
            return Err(format!("Reshape shape is {shape:?}, expected [0, 0, -1]"));
        }
        let t2 = sole_consumer(&rs.output[0], "Transpose")?;
        if attr_ints(t2, "perm") != Some(&[1, 0, 2]) {
            return Err("second readout Transpose is not perm=[1,0,2]".into());
        }
        let pooled = consumers(&t2.output[0]);
        let (mut mean, mut max) = (None, None);
        for n in &pooled {
            let slot = match n.op_type.as_str() {
                "ReduceMean" => &mut mean,
                "ReduceMax" => &mut max,
                other => return Err(format!("readout feeds {other}")),
            };
            if attr_ints(n, "axes") != Some(&[1]) || attr_int(n, "keepdims").unwrap_or(1) != 0 {
                return Err(format!("{} is not axes=[1], keepdims=0", n.op_type));
            }
            if slot.replace(*n).is_some() {
                return Err(format!("two {} nodes", n.op_type));
            }
        }
        let (mean, max) = (mean.ok_or("no ReduceMean")?, max.ok_or("no ReduceMax")?);
        let cat = sole_consumer(&mean.output[0], "Concat")?;
        if !std::ptr::eq(cat, sole_consumer(&max.output[0], "Concat")?) {
            return Err("mean and max feed different Concats".into());
        }
        if attr_int(cat, "axis") != Some(1) || cat.input.len() != 2 {
            return Err("Concat is not a two-input axis=1 concat".into());
        }
        let mean_first = cat.input[0] == mean.output[0];
        if !mean_first && cat.input[0] != max.output[0] {
            return Err("Concat inputs are not the two poolings".into());
        }

        // --- the head -------------------------------------------------------
        let gemm = sole_consumer(&cat.output[0], "Gemm")?;
        if attr_f32(gemm, "alpha").unwrap_or(1.0) != 1.0
            || attr_f32(gemm, "beta").unwrap_or(1.0) != 1.0
            || attr_int(gemm, "transA").unwrap_or(0) != 0
            || attr_int(gemm, "transB").unwrap_or(0) != 1
        {
            return Err("Gemm is not alpha=1, beta=1, transB=1".into());
        }
        let hw_t = init
            .get(gemm.input.get(1).map(String::as_str).unwrap_or(""))
            .ok_or("head weight is not an initializer")?;
        let n_cls = match hw_t.dims.as_slice() {
            [n, k] if *k as usize == 4 * h => *n as usize,
            d => {
                return Err(format!(
                    "head weight dims {d:?}, expected [n_cls, {}]",
                    4 * h
                ));
            }
        };
        let head_w = tensor_f32(hw_t)?;
        let head_b = match gemm.input.get(2).map(String::as_str).unwrap_or("") {
            "" => vec![0.0; n_cls],
            name => {
                let t = init.get(name).ok_or("head bias is not an initializer")?;
                if t.dims != [n_cls as i64] {
                    return Err(format!("head bias dims {:?}, expected [{n_cls}]", t.dims));
                }
                tensor_f32(t)?
            }
        };
        if !graph.output.iter().any(|o| o.name == gemm.output[0]) {
            return Err("Gemm does not produce the graph output".into());
        }

        // --- repack ---------------------------------------------------------
        let mut wt = [Vec::new(), Vec::new()];
        let mut bias = [Vec::new(), Vec::new()];
        let mut rt = [Vec::new(), Vec::new()];
        let mut rt_blocked = [Vec::new(), Vec::new()];
        for d in 0..2 {
            let mut wtd = vec![0.0f32; n_in * g];
            for gi in 0..g {
                for k in 0..n_in {
                    wtd[k * g + gi] = w[(d * g + gi) * n_in + k];
                }
            }
            wt[d] = wtd;
            bias[d] = (0..g)
                .map(|gi| b[d * 8 * h + gi] + b[d * 8 * h + g + gi])
                .collect();
            let mut rtd = vec![0.0f32; h * g];
            for gi in 0..g {
                for j in 0..h {
                    rtd[j * g + gi] = r[(d * g + gi) * h + j];
                }
            }
            rt_blocked[d] = block_rows(&rtd, h, g, 32);
            rt[d] = rtd;
        }
        Ok(Self {
            seq: n_off,
            n_in,
            h,
            wt,
            bias,
            rt,
            rt_blocked,
            head_w,
            head_b,
            n_cls,
            mean_first,
            backend: Backend::best_for(h),
        })
    }

    /// How many reads [`Self::logits_batch`] scores per pass at full
    /// efficiency on this backend.
    ///
    /// The kernel is bound by streaming `Rᵀ` from L2 once per timestep, so
    /// reads scored in lockstep share every weight-row load. Three is what the
    /// AVX2 register file holds — 4 accumulators × 3 reads, 3 broadcasts, one
    /// row load — and a caller batching in multiples of it loses nothing to
    /// the tail.
    pub fn preferred_batch(&self) -> usize {
        // `ESCAPEPOD_LSTM_BATCH` overrides the width, clamped to what the
        // backend has kernels for — the sweep lever, and the way to measure
        // the single-read kernel inside a binary that batches.
        static OVERRIDE: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
        let wanted = OVERRIDE.get_or_init(|| {
            std::env::var("ESCAPEPOD_LSTM_BATCH")
                .ok()
                .and_then(|s| s.parse().ok())
                .filter(|&n| n >= 1)
        });
        let max = match self.backend {
            Backend::Scalar => 1,
            #[cfg(target_arch = "x86_64")]
            Backend::Avx2 => 3,
        };
        wanted.map_or(max, |n| n.min(max))
    }

    /// Logits for several reads, scored in lockstep.
    ///
    /// `xs` are per-read inputs as [`Self::logits`] takes them; `out` is
    /// `[n_reads][n_cls]`, flat. Identical to calling [`Self::logits`] per
    /// read — the per-lane accumulation order does not change with the
    /// batch — but each `Rᵀ` row streamed from L2 serves every read in the
    /// group rather than one, which is where the single-read kernel's time
    /// goes.
    pub fn logits_batch(&self, xs: &[&[f32]], out: &mut [f32]) -> Result<()> {
        let (seq, n_in, h, g) = (self.seq, self.n_in, self.h, 4 * self.h);
        if out.len() != xs.len() * self.n_cls {
            bail!(
                "{} logit slots for {} reads of {} classes",
                out.len(),
                xs.len(),
                self.n_cls
            );
        }
        for x in xs {
            if x.len() != n_in * seq {
                bail!(
                    "input has {} values, the graph takes {n_in} x {seq}",
                    x.len()
                );
            }
        }
        let batch = self.preferred_batch();
        let (xw_len, hs_len) = (2 * seq * g, seq * 2 * h);
        SCRATCH.with(|s| {
            let mut s = s.borrow_mut();
            s.resize(batch * (xw_len + hs_len + g), 0.0);
            let (xw, rest) = s.split_at_mut(batch * xw_len);
            let (hs, gates) = rest.split_at_mut(batch * hs_len);
            let mut done = 0usize;
            while done < xs.len() {
                let n = (xs.len() - done).min(batch);
                let group = &xs[done..done + n];
                match (self.backend, n) {
                    #[cfg(target_arch = "x86_64")]
                    // Safety: as for `run_avx2` — buffer sizes are checked
                    // above and `preferred_batch` sized the scratch; the
                    // features are guaranteed by the dispatch.
                    (Backend::Avx2, 3) => unsafe { self.run_avx2_batch::<3>(group, xw, hs, gates) },
                    #[cfg(target_arch = "x86_64")]
                    (Backend::Avx2, 2) => unsafe { self.run_avx2_batch::<2>(group, xw, hs, gates) },
                    #[cfg(target_arch = "x86_64")]
                    // A lone read takes the single-read kernel, on exactly
                    // one read's worth of the scratch: it copies gates by
                    // exact length.
                    (Backend::Avx2, _) => unsafe {
                        self.run_avx2(
                            group[0],
                            &mut xw[..xw_len],
                            &mut hs[..hs_len],
                            &mut gates[..g],
                        )
                    },
                    (Backend::Scalar, _) => {
                        for (r, x) in group.iter().enumerate() {
                            let (xw_r, hs_r) = (
                                &mut xw[r * xw_len..(r + 1) * xw_len],
                                &mut hs[r * hs_len..(r + 1) * hs_len],
                            );
                            self.run_scalar(x, xw_r, hs_r, &mut gates[..g]);
                        }
                    }
                }
                for r in 0..n {
                    let o = &mut out[(done + r) * self.n_cls..(done + r + 1) * self.n_cls];
                    self.readout(&hs[r * hs_len..(r + 1) * hs_len], o);
                }
                done += n;
            }
        });
        Ok(())
    }

    /// Logits for one read.
    ///
    /// `x` is the standardised input, channel-major `[n_in][seq]` — exactly
    /// what [`crate::fnn::fold_standardise`] produces. `out` receives one
    /// logit per class.
    pub fn logits(&self, x: &[f32], out: &mut [f32]) -> Result<()> {
        let (seq, n_in, h, g) = (self.seq, self.n_in, self.h, 4 * self.h);
        if x.len() != n_in * seq {
            bail!(
                "input has {} values, the graph takes {n_in} x {seq}",
                x.len()
            );
        }
        if out.len() != self.n_cls {
            bail!("{} logit slots for {} classes", out.len(), self.n_cls);
        }
        let xw_len = 2 * seq * g;
        let hs_len = seq * 2 * h;
        SCRATCH.with(|s| {
            let mut s = s.borrow_mut();
            s.resize(xw_len + hs_len + g, 0.0);
            let (xw, rest) = s.split_at_mut(xw_len);
            let (hs, gates) = rest.split_at_mut(hs_len);
            match self.backend {
                Backend::Scalar => self.run_scalar(x, xw, hs, gates),
                #[cfg(target_arch = "x86_64")]
                // Safety: `best_for` / `with_backend` only select this when
                // the CPU has AVX2 + FMA and `h` is a multiple of 8.
                Backend::Avx2 => unsafe { self.run_avx2(x, xw, hs, gates) },
            }
            self.readout(hs, out);
        });
        Ok(())
    }

    /// Mean and max over time of `[seq][2H]`, in the graph's concat order,
    /// then the linear head.
    fn readout(&self, hs: &[f32], out: &mut [f32]) {
        let (seq, w) = (self.seq, 2 * self.h);
        let mut mean = vec![0.0f32; w];
        let mut max = vec![f32::NEG_INFINITY; w];
        for t in 0..seq {
            let row = &hs[t * w..(t + 1) * w];
            for u in 0..w {
                mean[u] += row[u];
                max[u] = max[u].max(row[u]);
            }
        }
        let inv = 1.0 / seq as f32;
        for m in &mut mean {
            *m *= inv;
        }
        let (first, second) = if self.mean_first {
            (&mean, &max)
        } else {
            (&max, &mean)
        };
        for (c, o) in out.iter_mut().enumerate() {
            let wr = &self.head_w[c * 2 * w..(c + 1) * 2 * w];
            let mut acc = self.head_b[c];
            for u in 0..w {
                acc += wr[u] * first[u];
            }
            for u in 0..w {
                acc += wr[w + u] * second[u];
            }
            *o = acc;
        }
    }

    /// The input contribution for every timestep of one direction, biases
    /// folded in: `xw[t][g] = b[g] + Σ_k W[g][k] x[k][t]`.
    fn input_contribution(&self, d: usize, x: &[f32], xw: &mut [f32]) {
        let (seq, n_in, g) = (self.seq, self.n_in, 4 * self.h);
        for t in 0..seq {
            let row = &mut xw[t * g..(t + 1) * g];
            row.copy_from_slice(&self.bias[d]);
            for k in 0..n_in {
                let xk = x[k * seq + t];
                let wk = &self.wt[d][k * g..(k + 1) * g];
                for (r, w) in row.iter_mut().zip(wk) {
                    *r += xk * w;
                }
            }
        }
    }

    fn run_scalar(&self, x: &[f32], xw: &mut [f32], hs: &mut [f32], gates: &mut [f32]) {
        let (seq, h, g) = (self.seq, self.h, 4 * self.h);
        let mut hcur = vec![0.0f32; h];
        let mut c = vec![0.0f32; h];
        for d in 0..2 {
            let xwd = &mut xw[d * seq * g..(d + 1) * seq * g];
            self.input_contribution(d, x, xwd);
            hcur.iter_mut().for_each(|v| *v = 0.0);
            c.iter_mut().for_each(|v| *v = 0.0);
            for step in 0..seq {
                let t = if d == 0 { step } else { seq - 1 - step };
                gates.copy_from_slice(&xwd[t * g..(t + 1) * g]);
                if step > 0 {
                    for (j, &hj) in hcur.iter().enumerate() {
                        let row = &self.rt[d][j * g..(j + 1) * g];
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
                hs[t * 2 * h + d * h..t * 2 * h + (d + 1) * h].copy_from_slice(&hcur);
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn run_avx2(&self, x: &[f32], xw: &mut [f32], hs: &mut [f32], gates: &mut [f32]) {
        use std::arch::x86_64::*;
        let (seq, h, g) = (self.seq, self.h, 4 * self.h);
        debug_assert!(h.is_multiple_of(8));
        // Safety: every pointer below is derived from a slice whose length
        // was checked by `logits` (`gates` is 4H, `rt[d]` is H x 4H, and the
        // AVX2 backend is only selected for H a multiple of 8), so each
        // 8-wide load and store stays inside its buffer; the target features
        // are guaranteed by the dispatch in `Backend::best_for`.
        unsafe {
            let mut hcur = vec![0.0f32; h];
            let mut c = vec![0.0f32; h];
            for d in 0..2 {
                let xwd = &mut xw[d * seq * g..(d + 1) * seq * g];
                self.input_contribution(d, x, xwd);
                hcur.iter_mut().for_each(|v| *v = 0.0);
                c.iter_mut().for_each(|v| *v = 0.0);
                let rt = self.rt[d].as_ptr();
                let rtb = self.rt_blocked[d].as_ptr();
                for step in 0..seq {
                    let t = if d == 0 { step } else { seq - 1 - step };
                    gates.copy_from_slice(&xwd[t * g..(t + 1) * g]);
                    if step > 0 {
                        // gates += h[j] * Rᵀ[j][..]: an axpy per hidden unit, so
                        // every load is unit-stride and nothing is horizontally
                        // reduced. Accumulators stay in registers a chunk at a
                        // time; 64 gates = 8 vectors held live.
                        let gp = gates.as_mut_ptr();
                        let mut base = 0usize;
                        while base + 64 <= g {
                            // Eight named accumulators, not an array: an
                            // indexed `[__m256; 8]` walked by closures was
                            // measured at ~3x the arithmetic bound, which is
                            // the signature of the accumulators living on the
                            // stack. Named, they are eight independent FMA
                            // chains in eight registers.
                            let p = gp.add(base);
                            let mut a0 = _mm256_loadu_ps(p);
                            let mut a1 = _mm256_loadu_ps(p.add(8));
                            let mut a2 = _mm256_loadu_ps(p.add(16));
                            let mut a3 = _mm256_loadu_ps(p.add(24));
                            let mut a4 = _mm256_loadu_ps(p.add(32));
                            let mut a5 = _mm256_loadu_ps(p.add(40));
                            let mut a6 = _mm256_loadu_ps(p.add(48));
                            let mut a7 = _mm256_loadu_ps(p.add(56));
                            // Two 32-gate blocks of `rt_blocked`, each a
                            // contiguous `H x 32` run: two sequential
                            // streams the prefetcher can follow.
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
                        // A hidden size that is a multiple of 8 but not of 16
                        // leaves a tail narrower than one chunk.
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
                    hs[t * 2 * h + d * h..t * 2 * h + (d + 1) * h].copy_from_slice(&hcur);
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
impl NativeBiLstm {
    /// `N` reads in lockstep: one recurrence step advances every read, and
    /// each 32-gate slice of a weight row is loaded once and applied to all
    /// `N` accumulator sets. Layout of the scratch: `xw` is `[N][2][seq][G]`,
    /// `hs` is `[N][seq][2H]`, `gates` is `[N][G]`.
    ///
    /// Bit-identical to [`Self::run_avx2`]: every gate lane still accumulates
    /// `h[j] * Rᵀ[j]` over `j` in order, whatever the chunk width or the
    /// batch. The register budget is `4 accumulators × N + N broadcasts + 1
    /// row load`, which is why `N ≤ 3`.
    #[target_feature(enable = "avx2,fma")]
    unsafe fn run_avx2_batch<const N: usize>(
        &self,
        xs: &[&[f32]],
        xw: &mut [f32],
        hs: &mut [f32],
        gates: &mut [f32],
    ) {
        use std::arch::x86_64::*;
        debug_assert_eq!(xs.len(), N);
        let (seq, h, g) = (self.seq, self.h, 4 * self.h);
        debug_assert!(h.is_multiple_of(8));
        let (xw_len, hs_len) = (2 * seq * g, seq * 2 * h);
        // Safety: as for `run_avx2`. `logits_batch` checked every input
        // length and sized the scratch for `preferred_batch() >= N` reads.
        unsafe {
            let mut hcur = vec![0.0f32; N * h];
            let mut c = vec![0.0f32; N * h];
            for d in 0..2 {
                for (r, x) in xs.iter().enumerate() {
                    let off = r * xw_len + d * seq * g;
                    self.input_contribution(d, x, &mut xw[off..off + seq * g]);
                }
                hcur.iter_mut().for_each(|v| *v = 0.0);
                c.iter_mut().for_each(|v| *v = 0.0);
                let rt = self.rt[d].as_ptr();
                let rtb = self.rt_blocked[d].as_ptr();
                for step in 0..seq {
                    let t = if d == 0 { step } else { seq - 1 - step };
                    for r in 0..N {
                        let src = r * xw_len + d * seq * g + t * g;
                        gates[r * g..(r + 1) * g].copy_from_slice(&xw[src..src + g]);
                    }
                    if step > 0 {
                        let gp = gates.as_mut_ptr();
                        let mut base = 0usize;
                        while base + 32 <= g {
                            // `acc[v][r]`: vector `v` of the 32-gate slice,
                            // for read `r` — so one row load serves the
                            // inner loop over reads.
                            let mut acc = [[_mm256_setzero_ps(); N]; 4];
                            for (v, accv) in acc.iter_mut().enumerate() {
                                for (r, a) in accv.iter_mut().enumerate() {
                                    *a = _mm256_loadu_ps(gp.add(r * g + base + v * 8));
                                }
                            }
                            // One 32-gate block of `rt_blocked`: the `H`
                            // rows this pass reads are contiguous.
                            let mut row = rtb.add((base / 32) * h * 32);
                            // Raw reads of `hcur`: indexed, every `j` paid
                            // two bounds checks per read inside the loop
                            // that streams the recurrent weights.
                            let hc = hcur.as_ptr();
                            for j in 0..h {
                                let mut hv = [_mm256_setzero_ps(); N];
                                for (r, hvr) in hv.iter_mut().enumerate() {
                                    *hvr = _mm256_set1_ps(*hc.add(r * h + j));
                                }
                                for (v, accv) in acc.iter_mut().enumerate() {
                                    let w = _mm256_loadu_ps(row.add(v * 8));
                                    for (a, hvr) in accv.iter_mut().zip(hv.iter()) {
                                        *a = _mm256_fmadd_ps(*hvr, w, *a);
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
                        let dst = r * hs_len + t * 2 * h + d * h;
                        hs[dst..dst + h].copy_from_slice(&hcur[r * h..(r + 1) * h]);
                    }
                }
            }
        }
    }
}

#[inline(always)]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(target_arch = "x86_64")]
mod vec8 {
    use std::arch::x86_64::*;

    /// Cephes-style `exp`, the same construction `escapepod_demux::crf::avx2`
    /// uses: range-reduce by `log2 e`, a degree-5 polynomial on the
    /// remainder, and the power of two assembled straight into the exponent
    /// field.
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

// ---- proto helpers ------------------------------------------------------------

fn attr<'a>(node: &'a pb::NodeProto, name: &str) -> Option<&'a pb::AttributeProto> {
    node.attribute.iter().find(|a| a.name == name)
}

fn attr_int(node: &pb::NodeProto, name: &str) -> Option<i64> {
    attr(node, name).map(|a| a.i)
}

fn attr_f32(node: &pb::NodeProto, name: &str) -> Option<f32> {
    attr(node, name).map(|a| a.f)
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

/// A float initializer's values, from whichever field the export used.
fn tensor_f32(t: &pb::TensorProto) -> std::result::Result<Vec<f32>, String> {
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

fn tensor_i64(t: &pb::TensorProto) -> std::result::Result<Vec<i64>, String> {
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

/// A graph input's static dimensions (`None` per symbolic axis), or `None`
/// when it declares no shape at all.
fn value_dims(v: &pb::ValueInfoProto) -> Option<Vec<Option<usize>>> {
    let ty = v.r#type.as_ref()?;
    // tract's vendored proto carries only the tensor variant of `TypeProto`,
    // so this is irrefutable there; a fuller proto would make it a `let else`.
    let pb::type_proto::Value::TensorType(t) = ty.value.as_ref()?;
    let shape = t.shape.as_ref()?;
    Some(
        shape
            .dim
            .iter()
            .map(|d| match d.value.as_ref() {
                Some(pb::tensor_shape_proto::dimension::Value::DimValue(n)) if *n > 0 => {
                    Some(*n as usize)
                }
                _ => None,
            })
            .collect(),
    )
}

#[cfg(test)]
pub(crate) mod tests {
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

    fn a_str(name: &str, s: &str) -> pb::AttributeProto {
        pb::AttributeProto {
            name: name.into(),
            r#type: pb::attribute_proto::AttributeType::String as i32,
            s: s.as_bytes().to_vec(),
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

    /// The shipped export's graph, with random weights, at any size.
    ///
    /// Mirrors what torch.onnx wrote for `charging_feature_nn_rna004`: the
    /// zero initial state comes from a `ConstantOfShape`, the reshape from a
    /// `Constant`, the poolings carry `axes` as an attribute (opset 17).
    pub(crate) fn lstm_model(n_ch: usize, n_off: usize, h: usize, seed: u64) -> pb::ModelProto {
        let mut rng = Rng(seed | 1);
        let (g, hh) = (4 * h as i64, h as i64);
        let zeros_shape = pb::TensorProto {
            dims: vec![3],
            data_type: ONNX_INT64,
            int64_data: vec![2, 1, hh],
            ..Default::default()
        };
        let reshape_shape = pb::TensorProto {
            dims: vec![3],
            data_type: ONNX_INT64,
            int64_data: vec![0, 0, -1],
            ..Default::default()
        };
        let nodes = vec![
            node(
                "Transpose",
                &["features"],
                &["x"],
                vec![a_ints("perm", &[2, 0, 1])],
            ),
            node(
                "Constant",
                &[],
                &["zshape"],
                vec![a_tensor("value", zeros_shape)],
            ),
            node(
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
            ),
            node(
                "LSTM",
                &["x", "W", "R", "B", "", "zeros", "zeros"],
                &["Y", "Y_h", "Y_c"],
                vec![
                    a_str("direction", "bidirectional"),
                    a_int("hidden_size", hh),
                ],
            ),
            node(
                "Transpose",
                &["Y"],
                &["t1"],
                vec![a_ints("perm", &[0, 2, 1, 3])],
            ),
            node(
                "Constant",
                &[],
                &["rshape"],
                vec![a_tensor("value", reshape_shape)],
            ),
            node(
                "Reshape",
                &["t1", "rshape"],
                &["rs"],
                vec![a_int("allowzero", 0)],
            ),
            node(
                "Transpose",
                &["rs"],
                &["t2"],
                vec![a_ints("perm", &[1, 0, 2])],
            ),
            node(
                "ReduceMean",
                &["t2"],
                &["mean"],
                vec![a_ints("axes", &[1]), a_int("keepdims", 0)],
            ),
            node(
                "ReduceMax",
                &["t2"],
                &["max"],
                vec![a_ints("axes", &[1]), a_int("keepdims", 0)],
            ),
            node("Concat", &["mean", "max"], &["cat"], vec![a_int("axis", 1)]),
            node(
                "Gemm",
                &["cat", "hw", "hb"],
                &["logits"],
                vec![a_int("transB", 1)],
            ),
        ];
        pb::ModelProto {
            ir_version: 8,
            opset_import: vec![pb::OperatorSetIdProto {
                domain: String::new(),
                version: 17,
            }],
            graph: Some(pb::GraphProto {
                node: nodes,
                initializer: vec![
                    f32_init("W", &[2, g, n_ch as i64], &mut rng, 0.5),
                    f32_init("R", &[2, g, hh], &mut rng, 0.3),
                    f32_init("B", &[2, 8 * hh], &mut rng, 0.2),
                    f32_init("hw", &[2, 2 * 2 * hh], &mut rng, 0.3),
                    f32_init("hb", &[2], &mut rng, 0.1),
                ],
                input: vec![value_info("features", &[1, n_ch as i64, n_off as i64])],
                output: vec![value_info("logits", &[1, 2])],
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    fn tract_logits(proto: &pb::ModelProto, n_ch: usize, n_off: usize, x: &[f32]) -> [f32; 2] {
        let plan = tract_onnx::onnx()
            .model_for_proto_model(proto)
            .unwrap()
            .with_input_fact(0, f32::fact([1, n_ch, n_off]).into())
            .unwrap()
            .into_optimized()
            .unwrap()
            .into_runnable()
            .unwrap();
        let t = Tensor::from_shape(&[1, n_ch, n_off], x).unwrap();
        let out = plan.run(tvec!(t.into())).unwrap();
        let v = out[0].to_plain_array_view::<f32>().unwrap();
        let s: Vec<f32> = v.iter().copied().collect();
        [s[0], s[1]]
    }

    fn input(n_ch: usize, n_off: usize, seed: u64) -> Vec<f32> {
        let mut rng = Rng(seed | 1);
        (0..n_ch * n_off).map(|_| rng.next(1.5)).collect()
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(p, q)| (p - q).abs())
            .fold(0.0, f32::max)
    }

    /// The native kernel agrees with tract on the real graph shape, on
    /// random weights and random inputs — both backends.
    #[test]
    fn matches_tract_on_the_shipped_shape() {
        let (n_ch, n_off, h) = (4, 33, 96);
        let proto = lstm_model(n_ch, n_off, h, 7);
        let net = NativeBiLstm::from_proto(&proto, n_ch, n_off).expect("recognised");
        assert_eq!(net.hidden(), 96);
        for seed in 1..=6u64 {
            let x = input(n_ch, n_off, seed);
            let want = tract_logits(&proto, n_ch, n_off, &x);
            let mut got = [0.0f32; 2];
            let scalar = NativeBiLstm::from_proto(&proto, n_ch, n_off)
                .unwrap()
                .with_backend(Backend::Scalar);
            scalar.logits(&x, &mut got).unwrap();
            let d = max_abs_diff(&got, &want);
            assert!(
                d < 2e-4,
                "scalar vs tract, seed {seed}: {got:?} vs {want:?} ({d:e})"
            );
            net.logits(&x, &mut got).unwrap();
            let d = max_abs_diff(&got, &want);
            assert!(
                d < 2e-4,
                "{} vs tract, seed {seed}: {got:?} vs {want:?} ({d:e})",
                net.backend().name()
            );
        }
    }

    /// A hidden size that exercises the AVX2 tail (a multiple of 8 but not of
    /// 16) and a non-square input.
    #[test]
    fn odd_sizes_match_tract() {
        for (n_ch, n_off, h) in [(4usize, 33usize, 24usize), (6, 17, 8), (2, 9, 40)] {
            let proto = lstm_model(n_ch, n_off, h, 11);
            let net = NativeBiLstm::from_proto(&proto, n_ch, n_off).expect("recognised");
            let x = input(n_ch, n_off, 3);
            let want = tract_logits(&proto, n_ch, n_off, &x);
            let mut got = [0.0f32; 2];
            net.logits(&x, &mut got).unwrap();
            let d = max_abs_diff(&got, &want);
            assert!(d < 2e-4, "h={h}: {got:?} vs {want:?} ({d:e})");
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_matches_scalar() {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
            return;
        }
        let (n_ch, n_off, h) = (4, 33, 96);
        let proto = lstm_model(n_ch, n_off, h, 5);
        let s = NativeBiLstm::from_proto(&proto, n_ch, n_off)
            .unwrap()
            .with_backend(Backend::Scalar);
        let v = NativeBiLstm::from_proto(&proto, n_ch, n_off)
            .unwrap()
            .with_backend(Backend::Avx2);
        for seed in 1..=4u64 {
            let x = input(n_ch, n_off, seed);
            let (mut a, mut b) = ([0.0f32; 2], [0.0f32; 2]);
            s.logits(&x, &mut a).unwrap();
            v.logits(&x, &mut b).unwrap();
            let d = max_abs_diff(&a, &b);
            assert!(d < 1e-4, "seed {seed}: scalar {a:?} vs avx2 {b:?} ({d:e})");
        }
    }

    /// Anything that is not this graph falls back, and says why.
    #[test]
    fn a_different_graph_is_not_matched() {
        let (n_ch, n_off, h) = (4, 33, 16);
        // No LSTM at all.
        let mut m = lstm_model(n_ch, n_off, h, 1);
        m.graph
            .as_mut()
            .unwrap()
            .node
            .retain(|n| n.op_type != "LSTM");
        assert!(NativeBiLstm::from_proto(&m, n_ch, n_off).is_none());
        // A unidirectional one.
        let mut m = lstm_model(n_ch, n_off, h, 1);
        for n in &mut m.graph.as_mut().unwrap().node {
            if n.op_type == "LSTM" {
                n.attribute = vec![
                    a_str("direction", "forward"),
                    a_int("hidden_size", h as i64),
                ];
            }
        }
        assert!(NativeBiLstm::from_proto(&m, n_ch, n_off).is_none());
        // Peepholes.
        let mut m = lstm_model(n_ch, n_off, h, 1);
        for n in &mut m.graph.as_mut().unwrap().node {
            if n.op_type == "LSTM" {
                n.input.push("P".into());
            }
        }
        assert!(NativeBiLstm::from_proto(&m, n_ch, n_off).is_none());
        // A readout without the max pooling.
        let mut m = lstm_model(n_ch, n_off, h, 1);
        m.graph
            .as_mut()
            .unwrap()
            .node
            .retain(|n| n.op_type != "ReduceMax");
        assert!(NativeBiLstm::from_proto(&m, n_ch, n_off).is_none());
        // The bundle declares a different input than the graph carries.
        let m = lstm_model(n_ch, n_off, h, 1);
        assert!(NativeBiLstm::from_proto(&m, n_ch + 2, n_off).is_none());
        assert!(NativeBiLstm::from_proto(&m, n_ch, n_off + 1).is_none());
    }

    /// Scoring reads in lockstep changes nothing about any read's answer:
    /// the batched kernel is bit-identical to the single-read one, for every
    /// group size up to and past the preferred batch, including the tails.
    #[test]
    fn batched_matches_single_bit_for_bit() {
        let (n_ch, n_off, h) = (4, 33, 96);
        let proto = lstm_model(n_ch, n_off, h, 21);
        let net = NativeBiLstm::from_proto(&proto, n_ch, n_off).expect("recognised");
        for n_reads in 1..=8usize {
            let inputs: Vec<Vec<f32>> = (1..=n_reads as u64)
                .map(|s| input(n_ch, n_off, 40 + s))
                .collect();
            let refs: Vec<&[f32]> = inputs.iter().map(Vec::as_slice).collect();
            let mut batched = vec![0.0f32; 2 * n_reads];
            net.logits_batch(&refs, &mut batched).unwrap();
            for (r, x) in inputs.iter().enumerate() {
                let mut single = [0.0f32; 2];
                net.logits(x, &mut single).unwrap();
                assert_eq!(
                    single.map(f32::to_bits),
                    [batched[2 * r].to_bits(), batched[2 * r + 1].to_bits()],
                    "n_reads={n_reads} read {r}: {single:?} vs {:?}",
                    &batched[2 * r..2 * r + 2]
                );
            }
        }
        // The scalar backend batches by looping, which is trivially the same;
        // it goes through the same entry point and says so.
        let scalar = NativeBiLstm::from_proto(&proto, n_ch, n_off)
            .unwrap()
            .with_backend(Backend::Scalar);
        assert_eq!(scalar.preferred_batch(), 1);
        let inputs: Vec<Vec<f32>> = (1..=4u64).map(|s| input(n_ch, n_off, 60 + s)).collect();
        let refs: Vec<&[f32]> = inputs.iter().map(Vec::as_slice).collect();
        let mut batched = vec![0.0f32; 8];
        scalar.logits_batch(&refs, &mut batched).unwrap();
        for (r, x) in inputs.iter().enumerate() {
            let mut single = [0.0f32; 2];
            scalar.logits(x, &mut single).unwrap();
            assert_eq!(
                single.map(f32::to_bits),
                [batched[2 * r].to_bits(), batched[2 * r + 1].to_bits()]
            );
        }
    }

    /// The concat order is read from the graph, not assumed.
    #[test]
    fn a_max_first_concat_is_honoured() {
        let (n_ch, n_off, h) = (4, 33, 16);
        let mut m = lstm_model(n_ch, n_off, h, 9);
        for n in &mut m.graph.as_mut().unwrap().node {
            if n.op_type == "Concat" {
                n.input.reverse();
            }
        }
        let net = NativeBiLstm::from_proto(&m, n_ch, n_off).expect("recognised");
        assert!(!net.mean_first);
        let x = input(n_ch, n_off, 2);
        let want = tract_logits(&m, n_ch, n_off, &x);
        let mut got = [0.0f32; 2];
        net.logits(&x, &mut got).unwrap();
        assert!(max_abs_diff(&got, &want) < 2e-4, "{got:?} vs {want:?}");
    }
}
