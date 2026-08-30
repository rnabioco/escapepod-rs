// SPDX-License-Identifier: MIT

//! Check a real `waveform_model` charging bundle against the corpus it was
//! trained on — on real weights, real reads, and the corpus's own tensors.
//!
//! Two claims, checked separately, because a single end-to-end number cannot
//! tell them apart (rnabioco/escapepod-rs#306, acceptance 1):
//!
//! 1. **The graph.** Feed escpod's runtime the tensors the corpus *stores* and
//!    compare its logit to onnxruntime's on the same rows. The export already
//!    round-trips against torch at 1.4e-05 with zero decision disagreements
//!    over 4,096 real chunks, so a residue here is this runtime.
//! 2. **The assembly.** Build the three tensors from POD5 + BAM and compare
//!    them *element by element* to the corpus's. This is the half with no
//!    reference implementation in Rust, and every one of its failure modes is
//!    silent: a window justified to the wrong side of a base, a permuted
//!    channel list, a k-mer context split the other way round — each yields a
//!    correctly shaped tensor of plausible numbers.
//!
//! Reference side: `scripts/dump_waveform_reference.py`.
//!
//! ```bash
//! python scripts/dump_waveform_reference.py \
//!     --bundle <bundle dir> --corpus <corpus>.npz --n 512 --out /tmp/tcn_ref
//!
//! # (1) the graph alone
//! cargo run --release --example verify_waveform_model \
//!     --features waveform-onnx -- <bundle dir> /tmp/tcn_ref
//!
//! # (2) and the assembly, against the reads the corpus was built from
//! cargo run --release --example verify_waveform_model \
//!     --features waveform-onnx -- <bundle dir> /tmp/tcn_ref \
//!     --pod5 <dir> --bam <aln.bam> --reference <ref.fa>
//! ```
//!
//! Exits non-zero if either comparison exceeds its tolerance. It prints the
//! distribution either way: a bare "N differ" cannot distinguish a wrong rule
//! from a numerical tie.

use escapepod_classify::waveform::{self, WaveformRead};
use escapepod_classify::{ChargingBundle, Pod5Index, WaveformTensor, junction_positions};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

fn read_f32(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assert_eq!(
        bytes.len() % 4,
        0,
        "{}: not a whole f32 array",
        path.display()
    );
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&c| f32::from_le_bytes(c))
        .collect()
}

/// Worst and median absolute difference over two equal-length slices.
fn residuals(a: &[f32], b: &[f32]) -> (f64, f64) {
    let mut d: Vec<f64> = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| (x as f64 - y as f64).abs())
        .collect();
    d.sort_by(f64::total_cmp);
    let max = d.last().copied().unwrap_or(0.0);
    let med = d.get(d.len() / 2).copied().unwrap_or(0.0);
    (max, med)
}

struct Args {
    bundle: PathBuf,
    prefix: PathBuf,
    pod5: Option<PathBuf>,
    bam: Option<PathBuf>,
    reference: Option<PathBuf>,
    tol_logit: f64,
    tol_tensor: f64,
}

fn parse_args() -> Args {
    let mut positional = Vec::new();
    let mut args = Args {
        bundle: PathBuf::new(),
        prefix: PathBuf::new(),
        pod5: None,
        bam: None,
        reference: None,
        tol_logit: 1e-3,
        tol_tensor: 0.0,
    };
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--pod5" => args.pod5 = it.next().map(PathBuf::from),
            "--bam" => args.bam = it.next().map(PathBuf::from),
            "--reference" => args.reference = it.next().map(PathBuf::from),
            "--tol-logit" => args.tol_logit = it.next().unwrap().parse().unwrap(),
            "--tol-tensor" => args.tol_tensor = it.next().unwrap().parse().unwrap(),
            _ => positional.push(a),
        }
    }
    assert_eq!(
        positional.len(),
        2,
        "usage: verify_waveform_model <bundle dir> <ref prefix> \
         [--pod5 D --bam B --reference F]"
    );
    args.bundle = PathBuf::from(&positional[0]);
    args.prefix = PathBuf::from(&positional[1]);
    args
}

fn main() {
    let args = parse_args();
    let meta: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(format!("{}.json", args.prefix.display())).unwrap(),
    )
    .unwrap();
    let n = meta["n"].as_u64().unwrap() as usize;
    let read_ids: Vec<String> = meta["read_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    let base_indices: Vec<i64> = meta["base_indices"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap())
        .collect();

    let bundle = ChargingBundle::load(&args.bundle).expect("bundle loads");
    let spec = bundle.waveform_spec().expect("a windowed bundle").clone();
    let net = bundle.waveform_net().expect("a linked graph");
    println!(
        "bundle {} [{}], {} reference chunks",
        bundle.model_id,
        bundle.scorer.kind(),
        n
    );

    let shape = |r| {
        let s = spec.tensor_shape(r);
        s[0] * s[1]
    };
    let ref_signal = read_f32(Path::new(&format!("{}.signal.f32", args.prefix.display())));
    let ref_seq = read_f32(Path::new(&format!(
        "{}.sequence.f32",
        args.prefix.display()
    )));
    let ref_feat = read_f32(Path::new(&format!(
        "{}.features.f32",
        args.prefix.display()
    )));
    let ref_logit = read_f32(Path::new(&format!("{}.logit.f32", args.prefix.display())));
    assert_eq!(ref_signal.len(), n * shape(WaveformTensor::Signal));
    assert_eq!(ref_seq.len(), n * shape(WaveformTensor::Sequence));
    assert_eq!(ref_feat.len(), n * shape(WaveformTensor::Features));
    assert_eq!(ref_logit.len(), n);

    let chunk_at = |i: usize| escapepod_signal::chunk::Chunk {
        signal: ref_signal
            [i * shape(WaveformTensor::Signal)..(i + 1) * shape(WaveformTensor::Signal)]
            .to_vec(),
        sequence: ref_seq
            [i * shape(WaveformTensor::Sequence)..(i + 1) * shape(WaveformTensor::Sequence)]
            .to_vec(),
        sequence_rows: spec.tensor_shape(WaveformTensor::Sequence)[0],
        sequence_cols: spec.tensor_shape(WaveformTensor::Sequence)[1],
        features: ref_feat
            [i * shape(WaveformTensor::Features)..(i + 1) * shape(WaveformTensor::Features)]
            .to_vec(),
        base_index: base_indices[i],
        focus_signal_pos: 0,
    };

    // --- (1) the graph, on the corpus's own tensors ---------------------
    //
    // Timed as well as compared: the per-chunk cost is what says whether this
    // model wants a device at all, and one read at a time is the shape the
    // pipeline actually runs.
    let chunks: Vec<_> = (0..n).map(chunk_at).collect();
    let t0 = std::time::Instant::now();
    let ours: Vec<f32> = chunks
        .iter()
        .map(|c| net.logit(c, &spec).expect("inference") as f32)
        .collect();
    let per_chunk = t0.elapsed().as_secs_f64() / n as f64;
    let (max, med) = residuals(&ours, &ref_logit);
    println!("graph:    max |dlogit| = {max:.3e}, median {med:.3e}  (over {n})");
    println!(
        "graph:    {:.0} us/chunk, one session, single-threaded",
        per_chunk * 1e6
    );
    let mut failed = max > args.tol_logit;
    if failed {
        println!("  FAIL: above --tol-logit {:.1e}", args.tol_logit);
    }

    // --- (2) the assembly, from POD5 + BAM ------------------------------
    if let (Some(pod5_dir), Some(bam), Some(reference)) = (&args.pod5, &args.bam, &args.reference) {
        let geometry = junction_positions(
            reference,
            &bundle.anchor.motif,
            bundle.anchor.motif_offset,
            &bundle.anchor.common_arm,
        )
        .expect("reference geometry");
        let refs = escapepod_classify::reference_sequences(reference).expect("reference FASTA");
        let scan =
            waveform::scan_bam(bam, &geometry, bundle.anchor.motif_offset, 1).expect("BAM scan");
        println!(
            "assembly: {} records scanned, {} reads anchored",
            scan.records_scanned,
            scan.anchored.len()
        );

        let wanted: HashSet<uuid::Uuid> = read_ids
            .iter()
            .filter_map(|s| escapepod_signal::parse_uuid_flexible(s).ok())
            .collect();
        let pod5_files = pod5_inputs(pod5_dir);
        let index = Pod5Index::build(&pod5_files, &wanted).expect("POD5 index");
        let extractors = index.extractors().expect("extractors");

        let by_id: HashMap<uuid::Uuid, &WaveformRead> =
            scan.anchored.iter().map(|(k, v)| (*k, v)).collect();

        let mut compared = 0usize;
        let mut worst = [0.0f64; 3];
        let mut worst_logit = 0.0f64;
        let mut missing = 0usize;
        let mut wrong_anchor = 0usize;
        for i in 0..n {
            let Ok(id) = escapepod_signal::parse_uuid_flexible(&read_ids[i]) else {
                missing += 1;
                continue;
            };
            let (Some(read), Some(info)) = (by_id.get(&id), index.reads().get(&id)) else {
                missing += 1;
                continue;
            };
            if read.base_index != base_indices[i] {
                wrong_anchor += 1;
                continue;
            }
            let raw = escapepod_classify::pipeline::signal_adc(info, &extractors).expect("signal");
            let Some(reference_seq) = refs.get(&read.reference) else {
                missing += 1;
                continue;
            };
            let Some(chunk) = waveform::assemble_chunk(
                bundle.kmer.as_ref(),
                &spec,
                read,
                reference_seq.as_bytes(),
                &raw,
            ) else {
                missing += 1;
                continue;
            };
            let want = chunk_at(i);
            for (k, (a, b)) in [
                (&chunk.signal, &want.signal),
                (&chunk.sequence, &want.sequence),
                (&chunk.features, &want.features),
            ]
            .iter()
            .enumerate()
            {
                let (m, _) = residuals(a, b);
                worst[k] = worst[k].max(m);
            }
            let ours = net.logit(&chunk, &spec).expect("inference");
            worst_logit = worst_logit.max((ours - ref_logit[i] as f64).abs());
            compared += 1;
        }
        println!(
            "assembly: compared {compared} chunks ({missing} unmatched, \
             {wrong_anchor} at a different anchor)"
        );
        println!(
            "assembly: max |d| signal {:.3e}, sequence {:.3e}, features {:.3e}",
            worst[0], worst[1], worst[2]
        );
        println!("assembly: max |dlogit| end to end = {worst_logit:.3e}");
        if compared == 0 {
            println!("  FAIL: no chunk could be compared");
            failed = true;
        }
        let tensor_max = worst.iter().cloned().fold(0.0, f64::max);
        if tensor_max > args.tol_tensor {
            println!(
                "  FAIL: tensors differ above --tol-tensor {:.1e}",
                args.tol_tensor
            );
            failed = true;
        }
        if worst_logit > args.tol_logit {
            println!("  FAIL: end-to-end logit above --tol-logit");
            failed = true;
        }
    }

    if failed {
        std::process::exit(1);
    }
    println!("OK");
}

/// Every `.pod5` under `dir` (or `dir` itself if it is a file).
fn pod5_inputs(dir: &Path) -> Vec<PathBuf> {
    if dir.is_file() {
        return vec![dir.to_path_buf()];
    }
    let mut out: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "pod5"))
        .collect();
    out.sort();
    out
}
