// SPDX-License-Identifier: MIT

//! Can tract run the windowed charging TCN on a GPU, and is it worth it?
//!
//! The measurement that decides the runtime question, kept re-runnable in the
//! spirit of `escapepod-demux/examples/tract_dynamo_probe.rs`. Nothing here is
//! wired into `escpod`; this exists to produce numbers before anything is.
//!
//! # Why the question is open at all
//!
//! The column (`feature_model`) scorer was paid down on the CPU to 97 us/read
//! and a GPU would be pointless for it — ~5 MFLOP/read, latency-bound. The
//! windowed variant is a different animal: two dilated TCN stacks, 22
//! convolutions of `[64,64,3]` over 390 samples, about **219 MFLOP/read**, and
//! `benches/charging.rs`'s `waveform` group measures ~5.96 ms/read. That is
//! ~37 GFLOP/s, within about 1.5x of single-core AVX2 f32 peak — so there is
//! no CPU micro-optimisation left that matters, and the *only* remaining lever
//! is a device.
//!
//! Two facts make the GPU case stronger than it looks:
//!
//! * **CPU batching is provably useless here** (`onnx_rewrite.rs` records that
//!   the cost is per row, not per call, and batch-64 cost the FNN CNN 12% at 48
//!   threads by blowing L2). That objection *inverts* on a device, which wants
//!   exactly the batch the CPU cannot use. So this probe sweeps batch.
//! * `tract-cuda` ships a `Conv` kernel, and this graph is almost entirely
//!   convolution. The reason the CRF encoder had to stay on `ort` was a missing
//!   LSTM kernel; there is no LSTM here.
//!
//! # What it must not gloss over
//!
//! `CudaTransform` translates the ops it knows and leaves the rest on the CPU,
//! inserting device/host syncs around them. This export carries a lot of
//! LayerNorm scaffolding (`InstanceNormalization`, `Reshape`, `Shape`), and if
//! those stay on the host the graph ping-pongs across PCIe per read and can
//! easily come out *slower* than pure CPU. So the probe reports the node
//! split, not just a time — a speedup with 200 syncs in it is a different
//! finding from a speedup without.
//!
//! Parity is checked, not assumed: the CUDA kernels are not the CPU ones, so
//! agreement is a tolerance, and the number is printed rather than asserted.
//!
//! ```text
//! srun -p gpu -A gpu_rbi -c 16 --gres=gpu:1 -- \
//!   pixi run -e gpu ./target/release/examples/tcn_cuda_probe \
//!     <bundle dir> --batches 1,8,32,128 --iters 20 [--dump] [--fix-norm] [--hoist]
//! ```
//!
//! `--dump` reports where each node ran and which convolution kernel it got;
//! `--fix-norm` and `--hoist` apply the two graph rewrites, so either can be
//! A/B'd against the graph as tract lowers it.
//!
//! Run it inside the `gpu` pixi env. tract-cuda JIT-compiles its kernels
//! through NVRTC on the machine that runs them, so it needs CUDA **headers**
//! (`cuda_fp16.h`, CCCL) present at RUN time — `cuda-cudart-dev` and
//! `cuda-cccl` are in `[feature.gpu.dependencies]` for that reason alone. That
//! is the one way this differs from every other GPU path in the workspace:
//! cudarc's driver API and `ort` dlopen a prebuilt library and are done.
//!
//! # Batching this graph through tract is not sound without `--fix-norm`
//!
//! The first thing this probe measured was a speedup. The second thing it
//! measured is that the speedup was on a different function, and that is the
//! more important result.
//!
//! `--dump` reports the placement, and every host node in the CUDA-transformed
//! graph was the same op: **29 `Reduce<MeanOfSquares>`, one per
//! `InstanceNormalization`**, each an island of its own with a device sync on
//! either side. The cause is one line of tract's ONNX lowering
//! (`tract-onnx/src/ops/nn/instance_norm.rs`):
//!
//! ```text
//! let axes: Vec<_> = (0..rank as i64).filter(|&axis| axis != 1).collect();
//! ```
//!
//! Two axes, so `GpuReduce::new` (single axis only) refuses — and the rewrite
//! that would split a multi-axis reduce covers `Sum | Prod | Min | Max | Any |
//! All`, correctly not `MeanOfSquares`, which does not chain. The 31
//! `Reduce<Sum>` in the same normalisations *are* split and do stay on the
//! device, which is why the count is 29 and not 60.
//!
//! But those two axes include the **batch**. This probe feeds read 0 the same
//! input at every batch size (one RNG stream per input role) precisely so that
//! its logit can be compared down the sweep, and it moves:
//!
//! ```text
//!   batch     read 0 logit
//!       1       +1.474677
//!       2       +0.263454
//!       8       +1.320732
//!     128       -0.284231
//! ```
//!
//! Every read is normalised against every other read it was batched with. That
//! is not a tolerance question and it is not fixable with a bigger epsilon —
//! it is a different function, and the sign flips. Nothing shipped is affected:
//! `WaveformNet::load` pins the batch to 1, where reducing over an axis of
//! length 1 is a no-op and tract's lowering is exactly the spec.
//!
//! [`escapepod_demux::onnx_rewrite::expand_instance_norm`] rewrites each
//! `InstanceNormalization` into the spec's own per-instance formula over axis 2
//! alone. One axis fixes both halves at once: the score stops depending on a
//! read's neighbours, and single-axis reductions are what the device accepts.
//!
//! # What it measured (2026-09-07, A30, charging_tcn_rna004@v0.1.2)
//!
//! Both arms in one job, interleaved, two reps that agree within noise:
//!
//! ```text
//!   batch   CPU ms/read   CUDA, tract's norm   CUDA, --fix-norm   vs CPU
//!       1      4.95              6.67                5.01          0.99x
//!       8      5.91              2.62                1.13          5.2x
//!      32      6.25              1.32                0.62         10.0x
//!     128      6.56              1.26                0.59         11.1x
//!     256      6.63              1.40                0.57         11.7x
//! ```
//!
//! ```text
//!             host nodes   device syncs   graph nodes
//!   as-is         29            62            687
//!   --fix-norm     0             4            484
//! ```
//!
//! Four syncs is the floor: three inputs in, one output back. The whole graph
//! runs on the device.
//!
//! **A single read is still no faster on the device than on a core** — 5.01 vs
//! 4.95 ms. The win is entirely the batch, which is the mirror image of the CPU
//! finding that batching buys nothing there, so a device path is only worth
//! building if reads are grouped and `waveform::classify_reads` scores one per
//! `par_iter` element today.
//!
//! Parity against the CPU arm holds and, with the norm fixed, stops widening
//! with the batch: max |dlogit| 1.9e-6 at 1, 2.5e-5 at 32, 3.7e-5 at 256,
//! against 9.97e-5 at 256 before. Some of the old drift was the cross-batch
//! reduction rather than kernel arithmetic. What remains is past the export's
//! own 1.3e-5 residual against torch, so a device path still needs a decision
//! about the operating point rather than an assumption — which is why this
//! prints the number instead of asserting a tolerance.
//!
//! **It costs ~5% on the CPU** (4.95 -> 5.15 ms/read at batch 1): tract fuses
//! its own two-axis lowering better than this explicit graph. The CPU path
//! scores batch 1, where tract's lowering is already the spec, so it buys
//! nothing there — which is why `waveform_net.rs` does not turn it on.
//!
//! # The convolution kernel is the next 4x, and it is not free
//!
//! With the graph resident, **24 of the 27 convolutions still run on
//! `conv1d_f32_generic`** — a naive direct kernel — rather than cuDNN.
//! `wire_cuda_conv` takes the cuDNN path only when every spatial dimension
//! pads symmetrically (`pad_before == pad_after`), and this export's causal
//! convolutions do not. Both spellings report as `CudaConv`, so the choice is
//! invisible in a node count; `--dump` prints it for that reason.
//!
//! `hoist_conv_padding` — the rewrite `waveform_net.rs` measured as a *loss*
//! on the CPU, and which was worth almost nothing on the device before the
//! norm was fixed — moves all 27 onto cuDNN, and on the resident graph it is
//! worth **4.1x**. Three reps, batch 128, A/B interleaved in one job:
//!
//! ```text
//!                        CUDA ms/read        max |dlogit|
//!   --fix-norm            0.593 0.596 0.596      3.1e-5
//!   --fix-norm --hoist    0.138 0.153 0.145      1.7e-2   <- 500x worse
//! ```
//!
//! So the two rewrites together are ~45x a CPU core rather than ~11x. The
//! catch is in the second column and it is not noise — the figure is identical
//! across reps, so it is the algorithm, not the scheduling. Both arms run the
//! same graph on both runtimes, and zero padding *is* a concatenation of zeros,
//! so the CPU arm barely moves; what changed is that cuDNN is now free to pick
//! a Winograd-class algorithm for a k=3 convolution, and twenty-two layers
//! compound its error. 1.7e-2 on a logit whose useful range is a few units is
//! about 1%, which is well past anything this family calls agreement.
//!
//! That is a question for whoever builds the device path — pin a cuDNN
//! algorithm, or take the 11x and keep the numbers — and not one to settle by
//! leaving a flag on. It is off, and the number is printed rather than
//! asserted, for the same reason everything else here is.

use escapepod_classify::{ChargingBundle, WaveformTensor};
use std::path::{Path, PathBuf};
use std::time::Instant;

use tract_onnx::prelude::*;

/// Deterministic filler, so both runtimes see identical inputs.
struct Rng(u64);

impl Rng {
    fn float(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 as u32 as f32) / (u32::MAX as f32) * 2.0 - 1.0
    }
}

/// Time `iters` runs after one warm-up, and return the warm-up's logits.
///
/// A macro rather than a function: `into_runnable()` on the CPU model and on
/// the CUDA-transformed one hand back plans whose concrete types differ, and
/// naming them here would buy nothing but a pair of turbofish.
macro_rules! run_timed {
    ($label:expr, $plan:expr, $inputs:expr, $iters:expr, $batch:expr, $nodes:expr, $syncs:expr) => {{
        let plan = &$plan;
        let warm = plan.run($inputs.clone())?;
        let logits: Vec<f32> = warm[0]
            .to_plain_array_view::<f32>()?
            .iter()
            .copied()
            .collect();

        let t0 = Instant::now();
        for _ in 0..$iters {
            let _ = plan.run($inputs.clone())?;
        }
        let per_call = t0.elapsed() / $iters as u32;
        let syncs: usize = $syncs;
        println!(
            "{}:   {:>9.3?}/call  {:>8.3?}/read  ({} nodes{})",
            $label,
            per_call,
            per_call / $batch as u32,
            $nodes,
            if syncs > 0 {
                format!(", {syncs} device syncs")
            } else {
                String::new()
            }
        );
        logits
    }};
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let Some(dir) = args.next() else {
        eprintln!("usage: tcn_cuda_probe <waveform bundle dir> [--batches 1,4,8] [--iters N]");
        std::process::exit(2);
    };
    let mut batches = vec![1usize, 4, 8, 16, 32];
    let mut iters = 30usize;
    let mut dump = false;
    let mut hoist = false;
    let mut fix_norm = false;
    // Read 0's logit at the first batch in the sweep, to compare the rest to.
    let mut solo: Option<f32> = None;
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--batches" => {
                batches = args
                    .next()
                    .expect("--batches needs a value")
                    .split(',')
                    .map(|s| s.parse().expect("batch must be a number"))
                    .collect()
            }
            "--iters" => iters = args.next().expect("--iters needs a value").parse().unwrap(),
            "--dump" => dump = true,
            "--hoist" => hoist = true,
            "--fix-norm" => fix_norm = true,
            other => panic!("unknown flag {other}"),
        }
    }

    // The bundle is the source of truth for the tensor shapes; the graph is
    // whatever `.onnx` sits beside its metadata.
    let dir = PathBuf::from(dir);
    let bundle = ChargingBundle::load(&dir)?;
    let spec = bundle.waveform_spec()?;
    let onnx_path = find_onnx(&dir)?;
    println!("graph:  {}", onnx_path.display());

    // Which tensors this graph takes, in the graph's own input order — the
    // same resolution `WaveformNet::load` does, because feeding three
    // different-shaped tensors positionally is not a shape error on two of the
    // six orderings.
    let onnx = tract_onnx::onnx();
    let proto = onnx.proto_model_for_path(&onnx_path)?;
    let base = onnx.model_for_proto_model(&proto)?;
    let roles: Vec<WaveformTensor> = base
        .input_outlets()?
        .iter()
        .map(|o| {
            let name = base.node(o.node).name.as_str();
            WaveformTensor::from_name(name)
                .unwrap_or_else(|| panic!("graph input {name:?} is not one this runtime assembles"))
        })
        .collect();
    for role in &roles {
        let [r, c] = spec.tensor_shape(*role);
        println!("input:  {:<9} [batch, {r}, {c}]", role.name());
    }

    for &batch in &batches {
        println!("\n=== batch {batch} ===");
        // `hoist_conv_padding` rewrites the proto, so each batch starts from a
        // clean copy — the zero block it splices in carries a concrete batch
        // dimension.
        let mut batch_proto = proto.clone();
        if fix_norm {
            let n = escapepod_demux::onnx_rewrite::expand_instance_norm(&mut batch_proto, 3);
            println!("norm:   {n} InstanceNormalization(s) rewritten to reduce over axis 2 alone");
        }
        if hoist {
            let n = escapepod_demux::onnx_rewrite::hoist_conv_padding(&mut batch_proto, batch);
            println!("hoist:  {n} convolution(s) rewritten to Concat(zeros) + Conv(pads=0)");
        }
        let mut model = onnx.model_for_proto_model(&batch_proto)?;
        for (i, role) in roles.iter().enumerate() {
            let [rows, cols] = spec.tensor_shape(*role);
            model = model.with_input_fact(i, f32::fact([batch, rows, cols]).into())?;
        }
        let typed = model.into_typed()?.into_decluttered()?;
        println!("nodes:  {} decluttered", typed.nodes().len());
        if dump {
            report_ops("declut", &typed);
        }

        // Identical inputs for both runtimes — and, deliberately, an identical
        // *read 0* at every batch size. Each role gets its own stream, so the
        // first `rows * cols` draws are the same whether the batch is 1 or 256.
        // That makes `logit[0]` comparable down the sweep, which is the only
        // way to see whether batching this graph is even sound: ONNX
        // `InstanceNormalization` is per-instance, so read 0's score must not
        // depend on who it was batched with.
        let inputs: TVec<TValue> = roles
            .iter()
            .enumerate()
            .map(|(i, role)| {
                let mut rng = Rng(0xC0FFEE + i as u64);
                let [rows, cols] = spec.tensor_shape(*role);
                let data: Vec<f32> = (0..batch * rows * cols).map(|_| rng.float()).collect();
                Tensor::from_shape(&[batch, rows, cols], &data).map(IntoTValue::into_tvalue)
            })
            .collect::<TractResult<_>>()?;

        // --- CPU -----------------------------------------------------------
        let cpu = typed.clone().into_optimized()?;
        let cpu_nodes = cpu.nodes().len();
        let cpu_plan = cpu.into_runnable()?;
        let cpu_logits = run_timed!("cpu ", cpu_plan, inputs, iters, batch, cpu_nodes, 0);

        // Read 0's input is the same at every batch size, so its logit must be
        // too. If it drifts, some reduction in the graph is running across the
        // batch axis and a batched score is not the per-read score.
        match solo {
            None => solo = Some(cpu_logits[0]),
            Some(reference) => {
                let d = (cpu_logits[0] - reference).abs();
                println!(
                    "batch:  read 0 logit {:+.6} vs {:+.6} at batch 1  (|d| {:.3e}){}",
                    cpu_logits[0],
                    reference,
                    d,
                    if d > 1e-4 {
                        "   <-- NOT BATCH-INVARIANT"
                    } else {
                        ""
                    }
                );
            }
        }

        // --- CUDA ----------------------------------------------------------
        #[cfg(feature = "cuda")]
        {
            use tract_core::transform::ModelTransform;
            let mut gpu = typed.clone();
            match tract_cuda::CudaTransform.transform(&mut gpu) {
                Ok(()) => {
                    match gpu.into_optimized().and_then(|m| {
                        // How much of the graph actually landed on the device.
                        // A translation that leaves scaffolding on the host
                        // pays a PCIe round trip per sync, and the node split
                        // is the only way to see that in the number below.
                        // Counted AFTER optimisation, because `rewire_syncs`
                        // and the optimiser both delete syncs and a count
                        // taken before them describes a graph nothing runs.
                        let n = m.nodes().len();
                        let syncs = count_syncs(&m);
                        if dump {
                            report_placement(&m);
                        }
                        m.into_runnable().map(|p| (n, syncs, p))
                    }) {
                        Ok((gpu_nodes, syncs, gpu_plan)) => {
                            let gpu_logits = tract_cuda::with_cuda_stream(|_| {
                                Ok(run_timed!(
                                    "cuda", gpu_plan, inputs, iters, batch, gpu_nodes, syncs
                                ))
                            })?;
                            report_parity(&cpu_logits, &gpu_logits);
                        }
                        Err(e) => println!("cuda:   optimize/plan failed: {e}"),
                    }
                }
                Err(e) => println!("cuda:   transform failed: {e}"),
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (&cpu_logits, dump);
            let _ = fix_norm;
            println!("cuda:   not built (add --features cuda)");
        }
    }
    Ok(())
}

/// The op histogram of a model, in evaluation order.
///
/// Cheap, needs no device, and answers the first half of the placement
/// question on a CPU node: every op here that `tract-cuda` has no translator
/// for is a host island waiting to happen. Cross-reference against
/// `tract-cuda/src/transform.rs` (`register_cuda_op!` plus the `Conv` /
/// `PrefixMatMul` / `Sdpa` / `Const` special cases) and
/// `tract-gpu/src/ops/copy_based.rs`.
fn report_ops(tag: &str, model: &TypedModel) {
    use std::collections::BTreeMap;
    let mut by_op: BTreeMap<String, usize> = BTreeMap::new();
    for node in model.nodes() {
        *by_op.entry(node.op.name().to_string()).or_default() += 1;
    }
    let mut rows: Vec<(&String, &usize)> = by_op.iter().collect();
    rows.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
    println!(
        "{tag}: {} ops over {} nodes",
        rows.len(),
        model.nodes().len()
    );
    for (op, n) in rows {
        println!("{tag}:   {n:>4}x {op}");
    }

    // One full `Debug` per distinct reduce, because the reduce is where the
    // CUDA transform gives up. `GpuReduce::new` refuses anything but a single
    // axis, and `split_multi_axis_reduce` — the rewrite that would split one —
    // covers `Sum | Prod | Min | Max | Any | All` and deliberately not
    // `MeanOfSquares`, which does not chain. So the axis count is the whole
    // question, and it is not in the op's display name.
    let mut seen: std::collections::BTreeSet<String> = Default::default();
    for node in model.nodes() {
        if !node.op.name().starts_with("Reduce") {
            continue;
        }
        let d = format!("{:?}", node.op);
        if seen.insert(d.clone()) {
            let fact = node
                .inputs
                .first()
                .and_then(|o| model.outlet_fact(*o).ok())
                .map(|f| format!("{:?} {:?}", f.datum_type, f.shape))
                .unwrap_or_else(|| "?".into());
            println!("{tag}:   reduce {d}  <- {fact}");
        }
    }
}

/// A node runs on the device iff its op carries a backend prefix.
///
/// `tract-gpu` names every translated op `Gpu<something>` and `tract-cuda`
/// names its own `Cuda<something>` (`ops/*.rs`: `format!("{}{}",
/// self.backend_name, ...)`), so the prefix *is* the placement — there is no
/// per-node flag to read, and the alternative (probing each outlet for a
/// `DeviceFact`) needs a `tract-gpu` dependency this crate does not otherwise
/// have. `Source` and `Const` are neither: they are inputs, and a `Const` the
/// transform uploaded still reports itself as `Const`.
#[cfg(feature = "cuda")]
fn is_device_op(op_name: &str) -> bool {
    op_name.starts_with("Cuda") || op_name.starts_with("Gpu")
}

/// Host/device transitions in the final, optimised graph.
#[cfg(feature = "cuda")]
fn count_syncs(model: &TypedModel) -> usize {
    model
        .nodes()
        .iter()
        .filter(|n| n.op.name().starts_with("DeviceSync"))
        .count()
}

/// Which nodes did NOT make it onto the device, and what they cost.
///
/// The sync count alone says a graph ping-pongs; it does not say what to do
/// about it. Every host node between two syncs is a candidate for either a
/// tract-cuda kernel or a graph rewrite in `waveform_net.rs`, and the only way
/// to choose is to see the ops and their shapes. Printed as islands — maximal
/// runs of host nodes in evaluation order — because the unit of cost is the
/// round trip, not the node: ten host ops in one island cost one pair of
/// syncs, and ten scattered across the graph cost ten.
#[cfg(feature = "cuda")]
fn report_placement(model: &TypedModel) {
    use std::collections::BTreeMap;

    let order = match model.eval_order() {
        Ok(o) => o,
        Err(e) => {
            println!("dump:   no eval order: {e}");
            return;
        }
    };

    let mut host_by_op: BTreeMap<String, usize> = BTreeMap::new();
    let mut islands: Vec<Vec<usize>> = Vec::new();
    let mut current: Vec<usize> = Vec::new();

    for &id in &order {
        let node = model.node(id);
        let op = node.op.name().to_string();

        let structural = op == "Source" || op == "Const" || op.starts_with("DeviceSync");
        if is_device_op(&op) || structural {
            if !current.is_empty() {
                islands.push(std::mem::take(&mut current));
            }
        } else {
            *host_by_op.entry(op).or_default() += 1;
            current.push(id);
        }
    }
    if !current.is_empty() {
        islands.push(current);
    }

    println!(
        "dump:   {} nodes, {} on the host in {} island(s)",
        order.len(),
        host_by_op.values().sum::<usize>(),
        islands.len()
    );

    // Which convolution kernel each conv landed on. `wire_cuda_conv` only
    // takes the cuDNN path when every spatial dimension pads symmetrically
    // (`pad_before == pad_after`) and falls back to `conv1d_*_generic` — a
    // naive direct kernel — otherwise. For a graph that is 22 convolutions
    // that is the difference between a GPU port and a GPU-shaped disappointment,
    // and it is invisible in the node count because both are `CudaConv`.
    let mut kernels: BTreeMap<String, usize> = BTreeMap::new();
    for node in model.nodes() {
        if node.op.name() != "CudaConv" {
            continue;
        }
        let kind = node
            .op
            .info()
            .ok()
            .and_then(|lines| lines.into_iter().find(|l| l.starts_with("kernel:")))
            .unwrap_or_else(|| "kernel: ?".into());
        *kernels.entry(kind).or_default() += 1;
    }
    for (kind, n) in &kernels {
        println!("dump:   {n:>4}x CudaConv {kind}");
    }

    if host_by_op.is_empty() {
        println!("dump:   every op is on the device");
    } else {
        println!("dump:   host ops by kind:");
        for (op, n) in &host_by_op {
            println!("dump:     {n:>4}x {op}");
        }
    }

    // One line per island: the ops in it, and the shape crossing into it.
    // That shape is the transfer the island costs, which is the number that
    // decides whether it is worth removing.
    for (i, island) in islands.iter().enumerate().take(24) {
        let ops: Vec<String> = island
            .iter()
            .map(|&id| model.node(id).op.name().to_string())
            .collect();
        let first = model.node(island[0]);
        let shape = first
            .inputs
            .first()
            .and_then(|o| model.outlet_fact(*o).ok())
            .map(|f| format!("{:?}", f.shape))
            .unwrap_or_else(|| "?".into());
        println!("dump:   island {i:>2} in {shape}: {}", ops.join(" -> "));
        println!("dump:            first node: {}", first.name);
    }
    if islands.len() > 24 {
        println!("dump:   ... and {} more island(s)", islands.len() - 24);
    }
}

/// The CUDA kernels are not the CPU ones, so agreement is a tolerance. Print
/// it rather than assert it — a probe that dies on 1e-6 tells you nothing
/// about whether the answer was usable.
///
/// Only the CUDA arm calls this, so without that feature it is dead code
/// rather than missing code: the comparison is the point of the probe, and a
/// `cfg` on the function keeps that visible instead of deleting it.
#[cfg(feature = "cuda")]
fn report_parity(cpu: &[f32], gpu: &[f32]) {
    if cpu.len() != gpu.len() {
        println!("parity: SHAPE MISMATCH {} vs {}", cpu.len(), gpu.len());
        return;
    }
    let max = cpu
        .iter()
        .zip(gpu)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!(
        "parity: max |dlogit| {max:.3e} over {} value(s){}",
        cpu.len(),
        if max > 1e-3 { "   <-- LOOK" } else { "" }
    );
}

fn find_onnx(dir: &Path) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let mut found: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "onnx"))
        .collect();
    found.sort();
    match found.len() {
        1 => Ok(found.remove(0)),
        0 => Err(format!("no .onnx in {}", dir.display()).into()),
        _ => Err(format!(
            "{} holds {} .onnx files; this probe wants a single-graph bundle",
            dir.display(),
            found.len()
        )
        .into()),
    }
}
