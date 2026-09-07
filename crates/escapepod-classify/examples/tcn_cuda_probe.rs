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
//!     <bundle dir> --batches 1,8,32,128 --iters 20
//! ```
//!
//! Run it inside the `gpu` pixi env. tract-cuda JIT-compiles its kernels
//! through NVRTC on the machine that runs them, so it needs CUDA **headers**
//! (`cuda_fp16.h`, CCCL) present at RUN time — `cuda-cudart-dev` and
//! `cuda-cccl` are in `[feature.gpu.dependencies]` for that reason alone. That
//! is the one way this differs from every other GPU path in the workspace:
//! cudarc's driver API and `ort` dlopen a prebuilt library and are done.
//!
//! # What it measured (2026-09-07, A30, charging_tcn_rna004@v0.1.2)
//!
//! ```text
//!   batch    CPU ms/read   CUDA ms/read   speedup
//!       1        4.99          7.87        0.63x   <- slower
//!       8        5.38          2.12        2.5x
//!      32        5.60          1.31        4.3x
//!      64        6.45          1.24        5.2x
//!     128        6.47          1.21        5.3x
//!     256        6.66          1.30        5.1x
//! ```
//!
//! **A single read is slower on the device than on a core.** The whole win is
//! the batch, which is the mirror image of the CPU finding that batching buys
//! nothing there — so a device path is only worth building if reads are
//! grouped, and `waveform::classify_reads` currently scores one per
//! `par_iter` element.
//!
//! Parity holds but widens with the batch: max |dlogit| 2.9e-6 at batch 1,
//! 4.1e-5 at 128, 9.8e-5 at 256. That is a max over more values rather than
//! drift, but the largest of them is past the export's own 1.3e-5 residual
//! against torch, so a device path needs a decision about the operating point
//! rather than an assumption.
//!
//! **64 device syncs at every batch**: the graph does not run entirely on the
//! device, and the LayerNorm scaffolding this export carries is the obvious
//! suspect. That is headroom on the 5.3x, not a reason to discount it.

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
        let mut model = onnx.model_for_proto_model(&proto)?;
        for (i, role) in roles.iter().enumerate() {
            let [rows, cols] = spec.tensor_shape(*role);
            model = model.with_input_fact(i, f32::fact([batch, rows, cols]).into())?;
        }
        let typed = model.into_typed()?.into_decluttered()?;
        println!("nodes:  {} decluttered", typed.nodes().len());

        // Identical inputs for both runtimes.
        let mut rng = Rng(0xC0FFEE);
        let inputs: TVec<TValue> = roles
            .iter()
            .map(|role| {
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

        // --- CUDA ----------------------------------------------------------
        #[cfg(feature = "cuda")]
        {
            use tract_core::transform::ModelTransform;
            let mut gpu = typed.clone();
            match tract_cuda::CudaTransform.transform(&mut gpu) {
                Ok(()) => {
                    // How much of the graph actually landed on the device.
                    // A translation that leaves the LayerNorm scaffolding on
                    // the host pays a PCIe round trip per sync, and the node
                    // split is the only way to see that in the number below.
                    let syncs = gpu
                        .nodes()
                        .iter()
                        .filter(|n| n.op.name().contains("Sync") || n.op.name().contains("sync"))
                        .count();
                    match gpu.into_optimized().and_then(|m| {
                        let n = m.nodes().len();
                        m.into_runnable().map(|p| (n, p))
                    }) {
                        Ok((gpu_nodes, gpu_plan)) => {
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
            let _ = &cpu_logits;
            println!("cuda:   not built (add --features cuda)");
        }
    }
    Ok(())
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
