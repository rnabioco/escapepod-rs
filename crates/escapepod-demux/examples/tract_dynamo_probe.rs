//! Does tract load this ONNX graph, and if not, where does it give up?
//!
//! Written for escapepod-rs#306, where the answer decided a runtime, and then
//! decided it again. The leech waveform TCN shipped first as
//! `charging_tcn_rna004@v0.1.0`, a PyTorch *dynamo* export that tract 0.23
//! parses whole and then fails shape analysis on, in every configuration
//! below. Two independent causes, and fixing either alone leaves it unloadable:
//! `value_info` written for all 667 intermediates with the batch axis as the
//! symbol `batch` (so a consumer that pins the batch dies at the first
//! convolution), and `adaptive_avg_pool1d(390 -> 11)` open-coded into a rank-8
//! `GatherND` because the output size does not divide the input.
//!
//! Not `nn.MultiheadAttention`, which escapepod-models#96 and escapepod-rs's
//! `waveform_net.rs` both named before anyone read the graph: the `GatherND`
//! consumes the last block of `signal_tcn` and carries that pool's bin mask and
//! bin widths, while `cross_attn` exports as plain
//! `Mul`/`MatMul`/`Softmax`/`MatMul`/`Gemm`.
//!
//! `@v0.1.1` (leech 0.10.0) is the re-export that fixes both, and this probe is
//! how it was checked from this side:
//!
//! ```text
//! v0.1.0   669 nodes   FAILED at node_GatherND_329 / node_index, all 3 modes
//! v0.1.1   471 nodes   optimized to 655, ran, output [1, 1], all 3 modes
//! ```
//!
//! (Counts are tract's after parsing, so larger than the 479 -> 319 the ONNX
//! graph itself reports.)
//!
//! So `escapepod_classify::waveform_net` is plain tract, and a bundle tract
//! cannot load is now a bundle-side problem with a build-time gate on it
//! (escapepod-models#97, `scripts/release/check_onnx_loadable.py`, which gates
//! against exactly the load path used here). Kept so both halves can be re-run
//! against a later tract or a later export. Takes any `.onnx` path.
use tract_onnx::prelude::*;

/// Replace every symbolic dimension with a concrete 1, everywhere the proto
/// declares one — inputs, outputs and value_info alike.
fn fix_batch(proto: &mut tract_onnx::pb::ModelProto) -> usize {
    let mut n = 0;
    let Some(g) = proto.graph.as_mut() else {
        return 0;
    };
    for vi in g
        .input
        .iter_mut()
        .chain(g.output.iter_mut())
        .chain(g.value_info.iter_mut())
    {
        let Some(t) = vi.r#type.as_mut() else {
            continue;
        };
        use tract_onnx::pb::type_proto::Value;
        if let Some(Value::TensorType(tt)) = t.value.as_mut()
            && let Some(shape) = tt.shape.as_mut()
        {
            for d in shape.dim.iter_mut() {
                use tract_onnx::pb::tensor_shape_proto::dimension::Value as DV;
                if matches!(d.value, Some(DV::DimParam(_))) {
                    d.value = Some(DV::DimValue(1));
                    n += 1;
                }
            }
        }
    }
    n
}

fn try_mode(path: &str, mode: &str) -> anyhow::Result<()> {
    let onnx = tract_onnx::onnx();
    let mut proto = onnx.proto_model_for_path(path)?;
    if mode.contains("fixbatch") {
        println!("  rewrote {} symbolic dims to 1", fix_batch(&mut proto));
    }
    if mode.contains("clear")
        && let Some(g) = proto.graph.as_mut()
    {
        g.value_info.clear();
    }
    let mut m = onnx.model_for_proto_model(&proto)?;
    if mode.contains("pin") {
        m = m
            .with_input_fact(0, f32::fact([1, 2, 390]).into())?
            .with_input_fact(1, f32::fact([1, 36, 390]).into())?
            .with_input_fact(2, f32::fact([1, 12, 21]).into())?;
    }
    println!("  parsed ok, {} nodes", m.nodes().len());
    let opt = m.into_optimized()?;
    println!("  optimized ok, {} nodes", opt.nodes().len());
    let plan = opt.into_runnable()?;
    let out = plan.run(tvec!(
        Tensor::zero::<f32>(&[1, 2, 390])?.into(),
        Tensor::zero::<f32>(&[1, 36, 390])?.into(),
        Tensor::zero::<f32>(&[1, 12, 21])?.into(),
    ))?;
    println!(
        "  ran ok: {} outputs, shape {:?}, value {:?}",
        out.len(),
        out[0].shape(),
        out[0].to_plain_array_view::<f32>()?
    );
    Ok(())
}

fn main() {
    let path = std::env::args().nth(1).expect("usage: probe_tcn <onnx>");
    for mode in ["fixbatch", "fixbatch+pin", "fixbatch+clear+pin"] {
        println!("== mode {mode} ==");
        match try_mode(&path, mode) {
            Ok(()) => println!("  OK"),
            Err(e) => println!("  FAILED: {e:#}"),
        }
    }
}
