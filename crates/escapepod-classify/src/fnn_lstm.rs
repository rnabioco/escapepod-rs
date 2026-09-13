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
//! The recurrence itself — the axpy-form kernel, its AVX2/AVX-512 dispatch,
//! and the batched "N reads in lockstep" entry points — lives in
//! [`escapepod_signal::lstm`], run once per direction into one interleaved
//! `[seq][2H]` buffer. `escapepod_demux::crf::encoder_native` is the other
//! caller, running the same primitive once per (unidirectional) layer of the
//! barcode CRF's five-layer stack; nothing about the recurrence is specific
//! to either shape. What stays here is the parts that are: recognising
//! *this* graph, and the mean/max readout into a linear head.
//!
//! Numerics: ONNX gate order is `i, o, f, c`; the default activations are
//! sigmoid, tanh, tanh. The vector path uses a Cephes-style `exp` and
//! `tanh(x) = 2σ(2x) − 1`, so it is not bit-identical to the scalar path, or
//! to tract, which uses its own vector approximations. The contract is
//! agreement on the probability within the parity tolerance the feature grid
//! already carries (1e-4), pinned by the tests below against tract on the
//! real graph shape.

use anyhow::{Result, bail};
use escapepod_signal::lstm::{self, LstmBackend, LstmWeights};
use std::collections::HashMap;
use tract_onnx::pb;

const ONNX_FLOAT: i32 = 1;
const ONNX_INT64: i32 = 7;

/// Which kernel scores a read. Re-exported so callers that logged
/// `NativeBiLstm::backend()` before the recurrence moved need not learn a new
/// path.
pub type Backend = LstmBackend;

/// The recognised graph's weights, repacked for the step loop.
#[derive(Debug)]
pub struct NativeBiLstm {
    /// Timesteps (the offset axis).
    seq: usize,
    /// Hidden units per direction.
    h: usize,
    /// Per direction's recurrence weights — see [`escapepod_signal::lstm`].
    weights: [LstmWeights; 2],
    /// `[n_cls][4H]` over the readout in the order the graph concatenates.
    head_w: Vec<f32>,
    head_b: Vec<f32>,
    n_cls: usize,
    /// Whether the `Concat` puts the mean block before the max block.
    mean_first: bool,
    backend: Backend,
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
    ///
    /// Refused when this machine cannot run it or the hidden size is not a
    /// multiple of its width — the two rules `best_for` applies — so a
    /// `NativeBiLstm` never carries a backend its kernels cannot execute.
    /// It used to assert the width and check nothing about the CPU, which
    /// made it a safe function through which safe code could reach an
    /// AVX-512 instruction on a machine without one.
    pub fn with_backend(mut self, backend: Backend) -> Result<Self> {
        if !backend.supported() {
            bail!("this machine cannot run the {} kernel", backend.name());
        }
        if !self.h.is_multiple_of(backend.lanes()) {
            bail!(
                "the {} kernel needs a hidden size that is a multiple of {}, not {}",
                backend.name(),
                backend.lanes(),
                self.h
            );
        }
        self.backend = backend;
        Ok(self)
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

        // --- repack -----------------------------------------------------
        // `w`/`r`/`b` hold both directions back to back
        // (`[2, ...]` in ONNX); slice the direction axis off before handing
        // each half to the shared packer.
        let weights = [0, 1].map(|d| {
            LstmWeights::from_onnx(
                n_in,
                h,
                &w[d * g * n_in..(d + 1) * g * n_in],
                &r[d * g * h..(d + 1) * g * h],
                &b[d * 8 * h..(d + 1) * 8 * h],
            )
        });
        let net = Self {
            seq: n_off,
            h,
            weights,
            head_w,
            head_b,
            n_cls,
            mean_first,
            backend: Backend::best_for(h),
        };
        self_check(&net, proto)?;
        Ok(net)
    }

    /// How many reads [`Self::logits_batch`] scores per pass at full
    /// efficiency on this backend. See [`LstmBackend::preferred_batch`].
    pub fn preferred_batch(&self) -> usize {
        self.backend.preferred_batch()
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
        let (seq, n_in, h, g) = (self.seq, self.weights[0].n_in, self.h, 4 * self.h);
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
        lstm::LSTM_SCRATCH.with(|s| {
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
                    // Safety: as for `run_both_avx2` — buffer sizes are
                    // checked above and `preferred_batch` sized the scratch;
                    // the features are guaranteed by the dispatch. The
                    // 16-wide kernel takes every width up to eight, a lone
                    // read included.
                    (Backend::Avx512, 8) => unsafe {
                        self.run_both_avx512_batch::<8>(group, xw, hs, gates)
                    },
                    #[cfg(target_arch = "x86_64")]
                    (Backend::Avx512, 7) => unsafe {
                        self.run_both_avx512_batch::<7>(group, xw, hs, gates)
                    },
                    #[cfg(target_arch = "x86_64")]
                    (Backend::Avx512, 6) => unsafe {
                        self.run_both_avx512_batch::<6>(group, xw, hs, gates)
                    },
                    #[cfg(target_arch = "x86_64")]
                    (Backend::Avx512, 5) => unsafe {
                        self.run_both_avx512_batch::<5>(group, xw, hs, gates)
                    },
                    #[cfg(target_arch = "x86_64")]
                    (Backend::Avx512, 4) => unsafe {
                        self.run_both_avx512_batch::<4>(group, xw, hs, gates)
                    },
                    #[cfg(target_arch = "x86_64")]
                    (Backend::Avx512, 3) => unsafe {
                        self.run_both_avx512_batch::<3>(group, xw, hs, gates)
                    },
                    #[cfg(target_arch = "x86_64")]
                    (Backend::Avx512, 2) => unsafe {
                        self.run_both_avx512_batch::<2>(group, xw, hs, gates)
                    },
                    #[cfg(target_arch = "x86_64")]
                    (Backend::Avx2, 3) => unsafe {
                        self.run_both_avx2_batch::<3>(group, xw, hs, gates)
                    },
                    #[cfg(target_arch = "x86_64")]
                    (Backend::Avx2, 2) => unsafe {
                        self.run_both_avx2_batch::<2>(group, xw, hs, gates)
                    },
                    #[cfg(target_arch = "x86_64")]
                    // A lone read on either wide backend takes the AVX2
                    // single-read kernel: eight accumulators over a 64-gate
                    // slice, against two per 32-gate slice through the
                    // 16-wide kernel at `N = 1` (208 against 168 µs/read).
                    // Every AVX-512F machine has AVX2 + FMA, and the two are
                    // bit-identical. Two arms, not an or-pattern: see the
                    // `#[inline(never)]` on the kernels.
                    (Backend::Avx512, _) => unsafe {
                        self.run_both_avx2(
                            group[0],
                            &mut xw[..xw_len],
                            &mut hs[..hs_len],
                            &mut gates[..g],
                        )
                    },
                    #[cfg(target_arch = "x86_64")]
                    (Backend::Avx2, _) => unsafe {
                        self.run_both_avx2(
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
                            self.run_both_scalar(x, xw_r, hs_r, &mut gates[..g]);
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
        let (seq, n_in, h, g) = (self.seq, self.weights[0].n_in, self.h, 4 * self.h);
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
        lstm::LSTM_SCRATCH.with(|s| {
            let mut s = s.borrow_mut();
            s.resize(xw_len + hs_len + g, 0.0);
            let (xw, rest) = s.split_at_mut(xw_len);
            let (hs, gates) = rest.split_at_mut(hs_len);
            match self.backend {
                Backend::Scalar => self.run_both_scalar(x, xw, hs, gates),
                #[cfg(target_arch = "x86_64")]
                // Safety: `best_for` / `with_backend` only select this when
                // the CPU has AVX2 + FMA and `h` is a multiple of 8.
                Backend::Avx2 => unsafe { self.run_both_avx2(x, xw, hs, gates) },
                #[cfg(target_arch = "x86_64")]
                // A lone read takes the AVX2 kernel on an AVX-512 machine
                // too (see `logits_batch`); it is bit-identical.
                Backend::Avx512 => unsafe { self.run_both_avx2(x, xw, hs, gates) },
            }
            self.readout(hs, out);
        });
        Ok(())
    }

    /// Both directions, scalar reference kernel, into one interleaved
    /// `[seq][2H]` buffer — `xw` is `[2][seq][4H]`, sliced per direction here
    /// since the shared kernel no longer knows about a direction count.
    fn run_both_scalar(&self, x: &[f32], xw: &mut [f32], hs: &mut [f32], gates: &mut [f32]) {
        let (seq, g, h) = (self.seq, 4 * self.h, self.h);
        for d in 0..2 {
            let xwd = &mut xw[d * seq * g..(d + 1) * seq * g];
            lstm::run_scalar(
                &self.weights[d],
                seq,
                d == 1,
                x,
                xwd,
                hs,
                2 * h,
                d * h,
                gates,
            );
        }
    }

    #[cfg(target_arch = "x86_64")]
    /// # Safety
    ///
    /// As [`escapepod_signal::lstm::run_avx2`]: the caller must have AVX2 +
    /// FMA and `self.h` a multiple of 8 (guaranteed by `Backend::best_for` /
    /// `with_backend`).
    unsafe fn run_both_avx2(&self, x: &[f32], xw: &mut [f32], hs: &mut [f32], gates: &mut [f32]) {
        let (seq, g, h) = (self.seq, 4 * self.h, self.h);
        for d in 0..2 {
            let xwd = &mut xw[d * seq * g..(d + 1) * seq * g];
            // Safety: forwarded from the caller.
            unsafe {
                lstm::run_avx2(
                    &self.weights[d],
                    seq,
                    d == 1,
                    x,
                    xwd,
                    hs,
                    2 * h,
                    d * h,
                    gates,
                );
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    /// # Safety
    ///
    /// As [`escapepod_signal::lstm::run_avx2_batch`]; additionally
    /// `xs.len() == N`.
    unsafe fn run_both_avx2_batch<const N: usize>(
        &self,
        xs: &[&[f32]],
        xw: &mut [f32],
        hs: &mut [f32],
        gates: &mut [f32],
    ) {
        let (seq, g, h) = (self.seq, 4 * self.h, self.h);
        let per_dir = N * seq * g;
        let (xw0, xw1) = xw[..2 * per_dir].split_at_mut(per_dir);
        // Safety: forwarded from the caller; `xw` was sized for at least
        // `preferred_batch()` reads by `logits_batch`.
        unsafe {
            lstm::run_avx2_batch::<N>(&self.weights[0], seq, false, xs, xw0, hs, 2 * h, 0, gates);
            lstm::run_avx2_batch::<N>(&self.weights[1], seq, true, xs, xw1, hs, 2 * h, h, gates);
        }
    }

    #[cfg(target_arch = "x86_64")]
    /// # Safety
    ///
    /// As [`escapepod_signal::lstm::run_avx512_batch`]; additionally
    /// `xs.len() == N`.
    unsafe fn run_both_avx512_batch<const N: usize>(
        &self,
        xs: &[&[f32]],
        xw: &mut [f32],
        hs: &mut [f32],
        gates: &mut [f32],
    ) {
        let (seq, g, h) = (self.seq, 4 * self.h, self.h);
        let per_dir = N * seq * g;
        let (xw0, xw1) = xw[..2 * per_dir].split_at_mut(per_dir);
        // Safety: forwarded from the caller.
        unsafe {
            lstm::run_avx512_batch::<N>(&self.weights[0], seq, false, xs, xw0, hs, 2 * h, 0, gates);
            lstm::run_avx512_batch::<N>(&self.weights[1], seq, true, xs, xw1, hs, 2 * h, h, gates);
        }
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
}

/// One seeded pseudo-random input of the graph's own `[n_ch, n_off]` shape,
/// scored through the lifted native weights and through a fresh tract plan of
/// the same proto; the contract is agreement on the class *probability*
/// (softmax of the logits), not the raw logit, within
/// [`SELF_CHECK_TOLERANCE`] — the guard against a recognizer that matches a
/// graph's shape but not its semantics, mirroring
/// `escapepod_demux::crf::encoder_native::self_check`.
///
/// The same 1e-4 the module doc already declares as the contract between the
/// native kernel and tract; every `from_proto` call in `tests` now runs this
/// check, so the width-pin tests (`matches_tract_on_the_shipped_shape` and
/// friends, agreeing with tract to within 2e-4 on the *logit*) are also its
/// coverage on the real graph shape and pass at this tolerance on probability.
const SELF_CHECK_TOLERANCE: f32 = 1e-4;

fn self_check(net: &NativeBiLstm, proto: &pb::ModelProto) -> std::result::Result<(), String> {
    use tract_onnx::prelude::*;
    use tract_onnx::tract_core::framework::Framework;

    let n_ch = net.weights[0].n_in;
    let n_off = net.seq;

    let mut rng_state = 0x9E3779B97F4A7C15u64;
    let mut next = move || {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        ((rng_state as u32 as f32) / (u32::MAX as f32) * 2.0 - 1.0) * 1.5
    };
    let x: Vec<f32> = (0..n_ch * n_off).map(|_| next()).collect();

    let plan = tract_onnx::onnx()
        .model_for_proto_model(proto)
        .map_err(|e| format!("self-check: cannot parse proto: {e}"))?
        .with_input_fact(0, f32::fact([1, n_ch, n_off]).into())
        .map_err(|e| format!("self-check: cannot pin input shape: {e}"))?
        .into_optimized()
        .map_err(|e| format!("self-check: cannot optimize: {e}"))?
        .into_runnable()
        .map_err(|e| format!("self-check: cannot plan: {e}"))?;
    let t = Tensor::from_shape(&[1, n_ch, n_off], &x)
        .map_err(|e| format!("self-check: cannot build input tensor: {e}"))?;
    let out = plan
        .run(tvec!(t.into()))
        .map_err(|e| format!("self-check: tract inference failed: {e}"))?;
    let view = out[0]
        .to_plain_array_view::<f32>()
        .map_err(|e| format!("self-check: tract output is not f32: {e}"))?;
    let tract_logits: Vec<f32> = view.iter().copied().collect();
    if tract_logits.len() != net.n_cls {
        return Err(format!(
            "self-check: tract emitted {} logits, expected {}",
            tract_logits.len(),
            net.n_cls
        ));
    }

    let mut native_logits = vec![0.0f32; net.n_cls];
    net.logits(&x, &mut native_logits)
        .map_err(|e| format!("self-check: native inference failed: {e}"))?;

    let max_diff = softmax(&tract_logits)
        .iter()
        .zip(softmax(&native_logits))
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    if max_diff > SELF_CHECK_TOLERANCE {
        return Err(format!(
            "self-check: native and tract disagree on P(class) by {max_diff:e}, past the {SELF_CHECK_TOLERANCE:e} tolerance"
        ));
    }
    Ok(())
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&v| (v - m).exp()).collect();
    let s: f32 = exps.iter().sum();
    exps.iter().map(|&v| v / s).collect()
}

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
                .with_backend(Backend::Scalar)
                .unwrap();
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

    /// The 16-wide kernel is the 8-wide one op for op, so it is pinned bit
    /// for bit — batched at every width it takes, and single — against the
    /// AVX2 kernels rather than to a tolerance. Skips on a machine without
    /// AVX-512F (CI), and says so.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx512_matches_avx2_bit_for_bit() {
        if !Backend::Avx512.supported() {
            eprintln!("no AVX-512F on this machine: cross-width pin not exercised");
            return;
        }
        let (n_ch, n_off, h) = (4, 33, 96);
        let proto = lstm_model(n_ch, n_off, h, 33);
        let wide = NativeBiLstm::from_proto(&proto, n_ch, n_off)
            .unwrap()
            .with_backend(Backend::Avx512)
            .unwrap();
        let narrow = NativeBiLstm::from_proto(&proto, n_ch, n_off)
            .unwrap()
            .with_backend(Backend::Avx2)
            .unwrap();
        assert_eq!(wide.preferred_batch(), 8);
        // 1..=17 covers every group width and every tail after a full group.
        for n_reads in 1..=17usize {
            let inputs: Vec<Vec<f32>> = (1..=n_reads as u64)
                .map(|s| input(n_ch, n_off, 70 + s))
                .collect();
            let refs: Vec<&[f32]> = inputs.iter().map(Vec::as_slice).collect();
            let (mut a, mut b) = (vec![0.0f32; 2 * n_reads], vec![0.0f32; 2 * n_reads]);
            wide.logits_batch(&refs, &mut a).unwrap();
            narrow.logits_batch(&refs, &mut b).unwrap();
            assert_eq!(
                a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                b.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "n_reads={n_reads}: avx512 {a:?} vs avx2 {b:?}"
            );
            for x in &inputs {
                let (mut sa, mut sb) = ([0.0f32; 2], [0.0f32; 2]);
                wide.logits(x, &mut sa).unwrap();
                narrow.logits(x, &mut sb).unwrap();
                assert_eq!(
                    sa.map(f32::to_bits),
                    sb.map(f32::to_bits),
                    "single: {sa:?} vs {sb:?}"
                );
            }
        }
        // A hidden size that is a multiple of 8 but not 16 is the AVX2
        // kernel's, whatever the machine has — unless `ESCAPEPOD_LSTM_BACKEND`
        // caps the dispatch, which the closing round sets to run these
        // tests under every backend.
        if lstm::backend_cap().is_none() {
            assert_eq!(Backend::best_for(88), Backend::Avx2);
            assert_eq!(Backend::best_for(96), Backend::Avx512);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_matches_scalar() {
        if !Backend::Avx2.supported() {
            eprintln!("no AVX2 + FMA on this machine: avx2-vs-scalar pin not exercised");
            return;
        }
        let (n_ch, n_off, h) = (4, 33, 96);
        let proto = lstm_model(n_ch, n_off, h, 5);
        let s = NativeBiLstm::from_proto(&proto, n_ch, n_off)
            .unwrap()
            .with_backend(Backend::Scalar)
            .unwrap();
        let v = NativeBiLstm::from_proto(&proto, n_ch, n_off)
            .unwrap()
            .with_backend(Backend::Avx2)
            .unwrap();
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
    /// Every kernel the machine has, not only the one the dispatch prefers:
    /// on an AVX-512 node the dispatch never reaches the AVX2 batched
    /// kernels, and a run that only followed the dispatch stayed green
    /// while the AVX2 width-2 kernel was returning wrong logits (see the
    /// note on `run_avx2`).
    #[test]
    fn batched_matches_single_bit_for_bit() {
        let (n_ch, n_off, h) = (4, 33, 96);
        let proto = lstm_model(n_ch, n_off, h, 21);
        for backend in Backend::available() {
            let net = NativeBiLstm::from_proto(&proto, n_ch, n_off)
                .expect("recognised")
                .with_backend(backend)
                .unwrap();
            // Up to one more than the widest group, so every width and
            // every tail after a full group is scored.
            for n_reads in 1..=net.preferred_batch() + 1 {
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
                        "{backend:?} n_reads={n_reads} read {r}: {single:?} vs {:?}",
                        &batched[2 * r..2 * r + 2]
                    );
                }
            }
        }
        // The scalar backend batches by looping, which is trivially the same;
        // it goes through the same entry point and says so.
        let scalar = NativeBiLstm::from_proto(&proto, n_ch, n_off)
            .unwrap()
            .with_backend(Backend::Scalar)
            .unwrap();
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

    /// A recognizer that matches a graph's shape but not its semantics is
    /// exactly what the self-check guards against: perturb one lifted weight
    /// after recognition succeeded, and it must refuse.
    #[test]
    fn self_check_refuses_a_perturbed_weight() {
        let (n_ch, n_off, h) = (4, 33, 16);
        let proto = lstm_model(n_ch, n_off, h, 42);
        let mut net = NativeBiLstm::from_proto(&proto, n_ch, n_off).expect("recognised");
        net.head_w[0] += 50.0;
        let err = self_check(&net, &proto).expect_err("perturbed weight must fail self-check");
        assert!(err.contains("self-check"), "{err}");
    }
}
