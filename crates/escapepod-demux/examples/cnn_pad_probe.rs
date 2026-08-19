//! Does zero-padding a short input actually change the model's output at
//! positions before the padding starts?
//!
//! A length-bucketing experiment showed short reads changing their call when
//! padded, and I first attributed that to the padding landing inside the
//! receptive field. That reasoning is suspect: a `Conv1d` with zero-padding
//! already reads zeros past the tensor edge, so for positions before the pad,
//! explicit zeros and the implicit edge ought to agree exactly. If they do, the
//! divergence is a wiring bug, not a property of the model.
//!
//! This compares raw channel-0 scores position by position, which distinguishes
//! the two:
//!
//! * identical over `0..valid_len` → the model is length-agnostic there, and the
//!   parity failure is ours to fix;
//! * divergent → the model genuinely depends on total input length (a
//!   length-dependent normalisation, or transposed-conv alignment), and no
//!   `valid_len` clamp can recover it.
//!
//! **It answered: divergent, from k=0.** Which is why batching must never
//! introduce padding of its own, and why the fix was to make prep emit the
//! model's own fixed length with the model's own pad value (#187) rather than
//! to bucket ragged ones.
//!
//! Since that fix `prep` always returns `input_len`, so there is no naturally
//! short input to probe with; the short case is synthesised here by truncating
//! a prepped window, which is exactly the tensor a read ending early used to
//! produce.
//!
//! ```text
//! cargo run --release --features cnn-gpu --example cnn_pad_probe -- <adapter.onnx>
//! ```

use escapepod_demux::{AdapterCnnConfig, AdapterCnnGpu};

fn synth(seed: u64, len: usize) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    (0..len)
        .map(|_| {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let u = ((s >> 33) as f32) / (1u64 << 31) as f32;
            400.0 + 200.0 * (u - 0.5)
        })
        .collect()
}

fn main() -> anyhow::Result<()> {
    let onnx = std::env::args()
        .nth(1)
        .ok_or_else(|| anyhow::anyhow!("usage: cnn_pad_probe <adapter.onnx>"))?;
    let cfg = AdapterCnnConfig::default();
    let det = AdapterCnnGpu::load(&onnx).map_err(|e| anyhow::anyhow!("load: {e}"))?;

    let full_len = cfg.input_len();
    // A few short reads. `prep` pads every read to `input_len` now, so truncate
    // to recover the ragged tensor the old prep produced — the input whose
    // batching behaviour is in question.
    for raw_len in [1374usize, 2196, 3500, 6000] {
        let sig = synth(raw_len as u64, raw_len);
        let prepped = match cfg.prep(&sig) {
            Some(p) => p,
            None => {
                println!("raw_len={raw_len}: too short to prep");
                continue;
            }
        };
        // Downscaled length this read would have occupied before padding.
        let valid = ((raw_len.saturating_sub(cfg.min_obs_adapter)).div_ceil(cfg.downscale_factor))
            .min(full_len);
        if valid == 0 {
            println!("raw_len={raw_len}: no window");
            continue;
        }

        // Exact: the tensor is exactly `valid` long — the conv sees the edge.
        let exact = det
            .scores_for_probe(&prepped.data[..valid], valid)
            .map_err(|e| anyhow::anyhow!("exact: {e}"))?;
        // Padded: the same leading `valid` samples in a full-length tensor.
        // Zeros, deliberately: the question is what *batch* padding would do,
        // and `pack_batch` zero-fills. (Prep itself pads with SCORE_EXCL.)
        let mut padded_in = vec![0f32; full_len];
        padded_in[..valid].copy_from_slice(&prepped.data[..valid]);
        let padded = det
            .scores_for_probe(&padded_in, full_len)
            .map_err(|e| anyhow::anyhow!("padded: {e}"))?;

        // Compare only where both are defined and the decode would actually look.
        let n = valid.min(exact.len()).min(padded.len());
        let mut max_abs = 0f32;
        let mut first_bad = None;
        for k in 0..n {
            let d = (exact[k] - padded[k]).abs();
            if d > max_abs {
                max_abs = d;
            }
            if d > 1e-4 && first_bad.is_none() {
                first_bad = Some((k, exact[k], padded[k]));
            }
        }
        let am = |v: &[f32], n: usize| (0..n).fold(0usize, |b, k| if v[k] > v[b] { k } else { b });
        println!(
            "raw_len={raw_len:<6} prepped={valid:<4} out_exact={:<4} out_padded={:<4} \
             max|Δ| over 0..{n} = {max_abs:.6}   argmax exact={} padded={}",
            exact.len(),
            padded.len(),
            am(&exact, n),
            am(&padded, n),
        );
        if let Some((k, a, b)) = first_bad {
            println!("    first divergence at k={k}: exact={a:.6} padded={b:.6}");
        }
    }

    println!(
        "\nIf max|Δ| is ~0 the model is length-agnostic before the padding and the\n\
         bucketing divergence is a wiring bug. If it is large, the model depends on\n\
         total input length and padding cannot be made exact."
    );
    Ok(())
}
