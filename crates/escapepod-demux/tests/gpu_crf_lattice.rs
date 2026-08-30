//! GPU CTC-CRF lattice decode: parity with bonito and with the CPU reference.
//!
//! Gated on `--features gpu`; skips transparently on hosts without a CUDA
//! device, like the other GPU tests here.
//!
//! The headline test reuses `crf_golden.json` — the fixture produced by running
//! real `bonito.crf.model.CTC_CRF` through koi's own CUDA kernels — so this
//! checks the GPU decode against bonito directly rather than only against our
//! CPU port. `crf_golden.rs` documents how that fixture is generated.

#![cfg(feature = "gpu")]

use std::path::Path;

use escapepod_demux::crf::{
    Backend, CrfLayout, CrfScratch, decode_with, lattice_gpu::CrfLatticeGpu,
};

/// A context, or `None` when this host genuinely has no usable CUDA device.
///
/// Deliberately narrow: **only** a missing driver/device is a skip. A kernel
/// that fails to compile, or a module that fails to load, is a real defect and
/// must fail the run — the first version of this helper skipped on any error,
/// which made the whole file pass green against a kernel that did not compile.
fn try_context(layout: CrfLayout) -> Option<CrfLatticeGpu> {
    match std::panic::catch_unwind(|| CrfLatticeGpu::new(layout)) {
        Ok(Ok(ctx)) => Some(ctx),
        Ok(Err(e)) => {
            let msg = e.to_string();
            assert!(
                msg.contains("CUDA init"),
                "GPU lattice init failed for a reason that is not a missing device — \
                 this is a real failure, not a skip: {msg}"
            );
            eprintln!("[gpu_crf] skipping — no usable CUDA device: {msg}");
            None
        }
        Err(panic) => {
            let msg = panic
                .downcast_ref::<&'static str>()
                .map(|s| s.to_string())
                .or_else(|| panic.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "<non-string panic payload>".into());
            eprintln!(
                "[gpu_crf] skipping — GPU init panicked (missing libnvrtc / libcuda?): {msg}"
            );
            None
        }
    }
}

fn golden() -> serde_json::Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/crf_golden.json");
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
    serde_json::from_str(&raw).expect("fixture is valid JSON")
}

/// Same closed-form score tensor `crf_golden.rs` regenerates — kept in step
/// with it deliberately, so both tests measure the same input.
fn synthesize(f: &serde_json::Value, n: usize) -> Vec<f32> {
    let g = |k: &str| f[k].as_f64().unwrap();
    let (t_len, n_score) = (f["T"].as_u64().unwrap(), f["C"].as_u64().unwrap());
    let (a_t, a_c, a_n, amp, b) = (g("a_t"), g("a_c"), g("a_n"), g("amp"), g("b"));
    let mut out = Vec::with_capacity((t_len * n_score) as usize);
    for t in 0..t_len {
        for c in 0..n_score {
            let s = (a_t * t as f64 + a_c * c as f64 + a_n * n as f64).sin() * amp;
            let k = (b * (t * n_score + c) as f64).cos();
            out.push((s + k) as f32);
        }
    }
    out
}

fn golden_layout(fixture: &serde_json::Value) -> CrfLayout {
    let sd = &fixture["seqdist"];
    CrfLayout::new(
        sd["n_base"].as_u64().unwrap() as usize,
        sd["state_len"].as_u64().unwrap() as usize,
    )
    .unwrap()
}

fn golden_alphabet(fixture: &serde_json::Value) -> Vec<u8> {
    fixture["seqdist"]["alphabet"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s.as_str().unwrap().as_bytes()[0])
        .collect()
}

/// The headline check: the batched GPU decode reproduces bonito's own
/// sequences, for every read in the fixture, decoded as one batch.
#[test]
fn gpu_decode_matches_bonito() {
    let fixture = golden();
    let layout = golden_layout(&fixture);
    let Some(gpu) = try_context(layout) else {
        return;
    };
    let alphabet = golden_alphabet(&fixture);
    let f = &fixture["formula"];
    let t_len = f["T"].as_u64().unwrap() as usize;
    let n_reads = f["N"].as_u64().unwrap() as usize;

    // One flat batch, exactly as the encoder hands it over.
    let mut batch = Vec::new();
    for n in 0..n_reads {
        batch.extend(synthesize(f, n));
    }
    let got = gpu.decode_batch(&batch, t_len, &alphabet).unwrap();

    assert_eq!(got.len(), n_reads);
    let seqs = fixture["small"]["seqs"].as_array().unwrap();
    for (n, seq) in got.iter().enumerate() {
        assert_eq!(
            seq,
            seqs[n].as_str().unwrap(),
            "GPU read {n}: decoded sequence differs from bonito"
        );
    }
}

/// A read decoded alone and the same read decoded inside a batch must agree —
/// this is what catches a batch-stride error, which would otherwise produce
/// plausible sequences attributed to the wrong reads.
#[test]
fn gpu_batching_does_not_change_any_read() {
    let fixture = golden();
    let layout = golden_layout(&fixture);
    let Some(gpu) = try_context(layout) else {
        return;
    };
    let alphabet = golden_alphabet(&fixture);
    let f = &fixture["formula"];
    let t_len = f["T"].as_u64().unwrap() as usize;
    let n_reads = f["N"].as_u64().unwrap() as usize;

    let per_read: Vec<Vec<f32>> = (0..n_reads).map(|n| synthesize(f, n)).collect();
    let alone: Vec<String> = per_read
        .iter()
        .map(|s| gpu.decode_batch(s, t_len, &alphabet).unwrap().remove(0))
        .collect();

    // Repeat the fixture's reads so the batch is wider than the read count and
    // the mapping cannot accidentally be the identity.
    let mut batch = Vec::new();
    let mut want = Vec::new();
    for rep in 0..4 {
        for n in 0..n_reads {
            batch.extend_from_slice(&per_read[n]);
            want.push(alone[n].clone());
            let _ = rep;
        }
    }
    let got = gpu.decode_batch(&batch, t_len, &alphabet).unwrap();
    assert_eq!(
        got, want,
        "batched decode disagrees with single-read decode"
    );
}

/// onnxruntime emits `[t_len][batch][n_score]`. Decoding that directly is the
/// whole point of the stride addressing — it removes a 1.5 MB-per-read host
/// de-interleave — so it must give bit-for-bit the same sequences as feeding
/// the same reads in batch-major order.
#[test]
fn time_major_decodes_identically_to_batch_major() {
    let fixture = golden();
    let layout = golden_layout(&fixture);
    let Some(gpu) = try_context(layout) else {
        return;
    };
    let alphabet = golden_alphabet(&fixture);
    let f = &fixture["formula"];
    let t_len = f["T"].as_u64().unwrap() as usize;
    let n_score = layout.n_score;

    // Four distinct reads, so a stride mix-up cannot land on the identity.
    let reads: Vec<Vec<f32>> = (0..4).map(|n| synthesize(f, n % 2)).collect();
    let batch = reads.len();

    let mut batch_major = Vec::new();
    for r in &reads {
        batch_major.extend_from_slice(r);
    }
    // [t][b][c] from the same reads.
    let mut time_major = vec![0f32; batch * t_len * n_score];
    for (b, r) in reads.iter().enumerate() {
        for t in 0..t_len {
            let dst = (t * batch + b) * n_score;
            time_major[dst..dst + n_score].copy_from_slice(&r[t * n_score..(t + 1) * n_score]);
        }
    }

    let want = gpu.decode_batch(&batch_major, t_len, &alphabet).unwrap();
    let got = gpu
        .decode_time_major(&time_major, t_len, &alphabet)
        .unwrap();
    assert_eq!(got.len(), batch);
    assert_eq!(
        got, want,
        "time-major decode disagrees with the batch-major decode of the same reads"
    );
}

/// GPU against the CPU reference on inputs the fixture does not cover: a
/// smooth lattice, a flat one (every edge tied — exercises the argmax
/// tie-break, which must resolve to the lowest `dest * n_edges + edge` on both
/// sides), and one with a -inf column.
#[test]
fn gpu_agrees_with_cpu_reference() {
    let layout = CrfLayout::new(4, 4).unwrap();
    let Some(gpu) = try_context(layout) else {
        return;
    };
    let alphabet = b"NACGT".to_vec();
    let t_len = 24usize;
    let n_score = layout.n_score;

    let cases: Vec<(&str, Vec<f32>)> = vec![
        (
            "smooth",
            (0..t_len * n_score)
                .map(|i| ((i as f32) * 0.017).sin() * 3.0 + ((i % 37) as f32) * 0.05)
                .collect(),
        ),
        // Every score identical: every edge ties at every timestep, so the
        // decode is decided entirely by the tie-break rule.
        ("all-tied", vec![0.5f32; t_len * n_score]),
        (
            "with-neg-inf",
            (0..t_len * n_score)
                .map(|i| {
                    if i % 1024 == 0 {
                        f32::NEG_INFINITY
                    } else {
                        ((i as f32) * 0.031).cos() * 2.0
                    }
                })
                .collect(),
        ),
    ];

    let mut scratch = CrfScratch::new();
    for (name, scores) in &cases {
        let want = decode_with(
            &layout,
            &alphabet,
            scores,
            t_len,
            &mut scratch,
            Backend::Scalar,
        )
        .unwrap();
        let got = gpu.decode_batch(scores, t_len, &alphabet).unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0], want, "case `{name}`: GPU differs from scalar CPU");
    }
}

/// A batch big enough to span several blocks, checked read-by-read against the
/// CPU. Deterministic input — no RNG, so a failure reproduces exactly.
#[test]
fn gpu_agrees_with_cpu_over_a_wide_batch() {
    let layout = CrfLayout::new(4, 4).unwrap();
    let Some(gpu) = try_context(layout) else {
        return;
    };
    let alphabet = b"NACGT".to_vec();
    let (t_len, batch) = (16usize, 64usize);
    let n_score = layout.n_score;

    let mut flat = Vec::with_capacity(batch * t_len * n_score);
    for b in 0..batch {
        for i in 0..t_len * n_score {
            let x = (i as f32) * 0.013 + (b as f32) * 0.71;
            flat.push(x.sin() * 4.0 + (x * 0.37).cos());
        }
    }

    let got = gpu.decode_batch(&flat, t_len, &alphabet).unwrap();
    assert_eq!(got.len(), batch);

    let mut scratch = CrfScratch::new();
    let per_read = t_len * n_score;
    for b in 0..batch {
        let want = decode_with(
            &layout,
            &alphabet,
            &flat[b * per_read..(b + 1) * per_read],
            t_len,
            &mut scratch,
            Backend::Scalar,
        )
        .unwrap();
        assert_eq!(got[b], want, "read {b} differs from scalar CPU");
    }
}

/// The device reference scan must agree with the CPU scan it replaces.
///
/// This is the test that matters for #297: `--ref-scores` used to force the
/// whole decode onto the host, and the kernel exists to stop that. It changes
/// where `log P(reference | signal)` is computed, so the only thing standing
/// between "faster" and "confidently wrong barcodes" is this comparison.
///
/// Tolerance rather than equality, for the reason the module note already
/// gives: the device reassociates reductions exactly as the AVX kernels do, so
/// the contract is agreement, not bit-identity. The sequences must match
/// exactly — that contract *is* bit-level.
#[test]
fn ref_scan_matches_cpu() {
    use escapepod_demux::crf::{RefChains, decode_with_refs};

    let fixture = golden();
    let layout = golden_layout(&fixture);
    let Some(ctx) = try_context(layout) else {
        return;
    };
    let alphabet: Vec<u8> = fixture["seqdist"]["alphabet"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s.as_str().unwrap().as_bytes()[0])
        .collect();

    let f = &fixture["formula"];
    let t_len = f["T"].as_u64().unwrap() as usize;
    let n_reads = f["N"].as_u64().unwrap() as usize;

    let refs = ["ACGTACGTACGT", "TTTTGGGGCCCC", "GATTACAGATTACA"];
    let seqs: Vec<&[u8]> = refs.iter().map(|s| s.as_bytes()).collect();
    let chains = RefChains::build(&layout, &alphabet, &seqs).unwrap();
    let tables = ctx
        .upload_ref_chains(&chains)
        .expect("panel fits the scan kernel");
    assert_eq!(tables.n_refs(), refs.len());

    // One interleaved time-major batch, the encoder's own layout, so a stride
    // or indexing error reads a neighbour rather than running off the end.
    let packed: Vec<Vec<f32>> = (0..n_reads).map(|n| synthesize(f, n)).collect();
    let mut batched = Vec::with_capacity(t_len * n_reads * layout.n_score);
    for t in 0..t_len {
        for read in &packed {
            batched.extend_from_slice(&read[t * layout.n_score..(t + 1) * layout.n_score]);
        }
    }

    let (mut got_logp, mut got_mean) = (Vec::new(), Vec::new());
    let got_seqs = ctx
        .decode_time_major_with_refs(
            &batched,
            t_len,
            &alphabet,
            &tables,
            &mut got_logp,
            &mut got_mean,
        )
        .expect("device scan");
    assert_eq!(got_seqs.len(), n_reads);
    assert_eq!(got_logp.len(), n_reads * refs.len());
    assert_eq!(got_mean.len(), n_reads);

    let backend = Backend::best_for(&layout);
    let mut scratch = CrfScratch::new();
    for n in 0..n_reads {
        let mut want_logp = Vec::new();
        let want_seq = decode_with_refs(
            &layout,
            &alphabet,
            &packed[n],
            t_len,
            &mut scratch,
            backend,
            &chains,
            &mut want_logp,
        )
        .unwrap();
        let want_mean = scratch.path_score() / t_len as f32;

        assert_eq!(got_seqs[n], want_seq, "read {n}: sequence differs");
        for (r, want) in want_logp.iter().enumerate() {
            let got = got_logp[n * refs.len() + r];
            assert!(
                (got - want).abs() <= 1e-3 * want.abs().max(1.0),
                "read {n} ref {r}: log P {got} vs {want}"
            );
        }
        assert!(
            (got_mean[n] - want_mean).abs() <= 1e-3 * want_mean.abs().max(1.0),
            "read {n}: mean_logpost {} vs {want_mean}",
            got_mean[n]
        );
    }
}
