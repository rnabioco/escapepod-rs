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
    /// Where to cache the reads' raw signal, so a diagnosis does not cost a
    /// full POD5 index each time. Indexing a 173 GB production file took 31
    /// minutes of wall for 23 seconds of CPU, which makes every hypothesis a
    /// half-hour round trip; that is how a wrong guess becomes the cheapest
    /// option, which is the opposite of what you want while debugging parity.
    cache: Option<PathBuf>,
    tol_logit: f64,
    tol_tensor: f64,
    /// Write our own per-base dwell rows (refined and unrefined) so they can be
    /// compared against a corpus prepared either way. The dwell row is the one
    /// place a boundary disagreement shows up as an integer.
    dump_dwells: Option<PathBuf>,
}

fn parse_args() -> Args {
    let mut positional = Vec::new();
    let mut args = Args {
        bundle: PathBuf::new(),
        prefix: PathBuf::new(),
        pod5: None,
        bam: None,
        reference: None,
        cache: None,
        tol_logit: 1e-3,
        tol_tensor: 0.0,
        dump_dwells: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--pod5" => args.pod5 = it.next().map(PathBuf::from),
            "--bam" => args.bam = it.next().map(PathBuf::from),
            "--reference" => args.reference = it.next().map(PathBuf::from),
            "--cache" => args.cache = it.next().map(PathBuf::from),
            "--tol-logit" => args.tol_logit = it.next().unwrap().parse().unwrap(),
            "--tol-tensor" => args.tol_tensor = it.next().unwrap().parse().unwrap(),
            "--dump-dwells" => args.dump_dwells = it.next().map(PathBuf::from),
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
    let ref_focus = read_i64(Path::new(&format!("{}.focus.i64", args.prefix.display())));
    let ref_dwell = read_f32(Path::new(&format!("{}.dwell.f32", args.prefix.display())));
    // The corpus's own k-mer window per chunk. Comparing our reference slice
    // at the same base index against it tests the sequence side with the
    // signal held out entirely.
    let ref_seq_win: Vec<String> =
        std::fs::read_to_string(format!("{}.seq.txt", args.prefix.display()))
            .map(|t| t.lines().map(str::to_string).collect())
            .unwrap_or_default();
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

        // Raw signal for exactly the reads under test, from the cache if one
        // exists. Building it means indexing the whole POD5, which on a
        // production file is minutes of I/O for seconds of CPU.
        let signals: HashMap<uuid::Uuid, Vec<i16>> =
            match args.cache.as_ref().filter(|p| p.exists()) {
                Some(path) => {
                    let m = load_signal_cache(path);
                    println!(
                        "assembly: {} signals from cache {}",
                        m.len(),
                        path.display()
                    );
                    m
                }
                None => {
                    let pod5_files = pod5_inputs(pod5_dir);
                    let index = Pod5Index::build(&pod5_files, &wanted).expect("POD5 index");
                    let extractors = index.extractors().expect("extractors");
                    let mut m = HashMap::new();
                    for id in &wanted {
                        if let Some(info) = index.reads().get(id) {
                            m.insert(
                                *id,
                                escapepod_classify::pipeline::signal_adc(info, &extractors)
                                    .expect("signal"),
                            );
                        }
                    }
                    if let Some(path) = &args.cache {
                        save_signal_cache(path, &m);
                        println!("assembly: cached {} signals to {}", m.len(), path.display());
                    }
                    m
                }
            };

        let by_id: HashMap<uuid::Uuid, &WaveformRead> =
            scan.anchored.iter().map(|(k, v)| (*k, v)).collect();

        let mut compared = 0usize;
        let mut worst = [0.0f64; 3];
        let mut worst_logit = 0.0f64;
        let mut missing = 0usize;
        let mut wrong_anchor = 0usize;
        // Diagnostics that separate *where* the window was cut from *what* was
        // in it. A displaced window makes all three tensors differ in a way
        // that looks like noise, so the aggregate residual cannot tell the two
        // apart and the focus sample can.
        let n_feat = spec.tensor_shape(WaveformTensor::Features)[0];
        let feat_w = spec.tensor_shape(WaveformTensor::Features)[1];
        let mut per_row = vec![0.0f64; n_feat];
        let mut focus_delta: Vec<i64> = Vec::new();
        let mut exact = 0usize;
        let mut best_shifts: Vec<i64> = Vec::new();
        let mut shown = 0usize;
        // (refined matches, unrefined matches, neither)
        let mut refine_tally = (0usize, 0usize, 0usize);
        // Our dwell row per chunk, with the DP on and off. `f32::NAN` marks a
        // chunk we could not assemble, so the row index still lines up with
        // the corpus's.
        let w_dwell = shape(WaveformTensor::Features) / 12;
        let mut per_sig_ch = vec![0f64; spec.tensor_shape(WaveformTensor::Signal)[0]];
        let mut ours_sig: Vec<f32> = vec![f32::NAN; n * shape(WaveformTensor::Signal)];
        let mut ours_on: Vec<f32> = vec![f32::NAN; n * w_dwell];
        let mut ours_off: Vec<f32> = vec![f32::NAN; n * w_dwell];
        let mut seq_ok = 0usize;
        let mut seq_case_only = 0usize;
        // (clean+match, clean+miss, lowercase+match, lowercase+miss)
        let mut cross = (0usize, 0usize, 0usize, 0usize);
        let mut seq_real = 0usize;
        let mut seq_shown = 0usize;
        let mut matched_bases: Vec<i64> = Vec::new();
        let mut missed_bases: Vec<i64> = Vec::new();
        for i in 0..n {
            let Ok(id) = escapepod_signal::parse_uuid_flexible(&read_ids[i]) else {
                missing += 1;
                continue;
            };
            let (Some(read), Some(raw)) = (by_id.get(&id), signals.get(&id)) else {
                missing += 1;
                continue;
            };
            if read.base_index != base_indices[i] {
                wrong_anchor += 1;
                continue;
            }
            let Some(reference_seq) = refs.get(&read.reference) else {
                missing += 1;
                continue;
            };
            let Some(chunk) = waveform::assemble_chunk(bundle.kmer.as_ref(), &spec, read, raw)
            else {
                missing += 1;
                continue;
            };
            let want = chunk_at(i);
            // Is the banded DP the thing we disagree about? Assemble the same
            // read with refinement off and see which reproduces the corpus.
            // One run answers what a stack of hypotheses about the DP cannot:
            // whether the corpus was refined at all.
            {
                let mut plain = spec.clone();
                plain.refine = None;
                let off = waveform::assemble_chunk(bundle.kmer.as_ref(), &plain, read, raw);
                let on_ok = chunk.features == want.features;
                let off_ok = off.as_ref().is_some_and(|c| c.features == want.features);
                match (on_ok, off_ok) {
                    (true, _) => refine_tally.0 += 1,
                    (false, true) => refine_tally.1 += 1,
                    (false, false) => refine_tally.2 += 1,
                }
                // Row 0 of `features` is the raw per-base dwell, which is what
                // the corpus stores as `dwells_flat`.
                ours_on[i * w_dwell..(i + 1) * w_dwell].copy_from_slice(&chunk.features[..w_dwell]);
                let sw = shape(WaveformTensor::Signal);
                ours_sig[i * sw..(i + 1) * sw].copy_from_slice(&chunk.signal);
                if let Some(c) = off.as_ref() {
                    ours_off[i * w_dwell..(i + 1) * w_dwell]
                        .copy_from_slice(&c.features[..w_dwell]);
                }
                // The preset's Theil-Sen rescale subsamples only above
                // `max_points` (200) bases. If agreement tracks that
                // threshold, the disagreement is the subsample, not the DP.
                // Our 11-mer at the anchor, from the aligned reference slice.
                if let Some(theirs) = ref_seq_win.get(i) {
                    let half = theirs.len() / 2;
                    let lo = base_indices[i] - half as i64;
                    let ours: String = (0..theirs.len() as i64)
                        .map(|k| {
                            usize::try_from(lo + k)
                                .ok()
                                .and_then(|u| reference_seq.as_bytes().get(read.ref_start + u))
                                .map(|&b| b as char)
                                .unwrap_or('N')
                        })
                        .collect();
                    // Case is not the question: `extract_levels` uppercases and
                    // `base_to_int` is case-insensitive, so a purely lowercase
                    // difference cannot move a boundary. Different LETTERS can.
                    // Does a lowercase (read-vs-reference mismatch) position
                    // PREDICT a feature mismatch? Case cannot move a boundary
                    // by itself, so a strong association would mean it leaks
                    // somewhere, and independence would exonerate it.
                    let lower = theirs.bytes().any(|b| b.is_ascii_lowercase());
                    match (lower, on_ok) {
                        (false, true) => cross.0 += 1,
                        (false, false) => cross.1 += 1,
                        (true, true) => cross.2 += 1,
                        (true, false) => cross.3 += 1,
                    }
                    if &ours == theirs {
                        seq_ok += 1;
                    } else if ours.eq_ignore_ascii_case(theirs) {
                        seq_case_only += 1;
                    } else {
                        seq_real += 1;
                        if seq_shown < 5 {
                            seq_shown += 1;
                            println!("  seq DIFFERS chunk {i}: ours {ours} theirs {theirs}");
                        }
                    }
                }
                let n_bases = read.ref_end - read.ref_start;
                if on_ok {
                    matched_bases.push(n_bases as i64);
                } else {
                    missed_bases.push(n_bases as i64);
                }
            }
            {
                let cols = spec.tensor_shape(WaveformTensor::Signal)[1];
                for (c, acc) in per_sig_ch.iter_mut().enumerate() {
                    let lo = c * cols;
                    for j in lo..lo + cols {
                        let d = (chunk.signal[j] as f64 - want.signal[j] as f64).abs();
                        if d > *acc {
                            *acc = d;
                        }
                    }
                }
            }
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
            // Per-feature-row residual: names which channel moved.
            for (r, worst_row) in per_row.iter_mut().enumerate() {
                let (m, _) = residuals(
                    &chunk.features[r * feat_w..(r + 1) * feat_w],
                    &want.features[r * feat_w..(r + 1) * feat_w],
                );
                *worst_row = worst_row.max(m);
            }
            // Where each side centred the window, in its own frame. The corpus
            // records this, so it is a direct check rather than an inference.
            focus_delta.push(chunk.focus_signal_pos - ref_focus[i]);
            // Per-base dwells over the feature window, straight from the map:
            // if these agree the refinement matched and only the window moved.
            let ours_dwell = &chunk.features[..feat_w];
            let their_dwell = &ref_dwell[i * feat_w..(i + 1) * feat_w];
            let dwell_ok = ours_dwell.iter().zip(their_dwell).all(|(a, b)| a == b);
            if chunk.signal == want.signal && chunk.features == want.features {
                exact += 1;
            }
            // The dwell row IS the map, one number per base, so printing a few
            // mismatching pairs says where the boundaries diverge far more
            // directly than any aggregate. Which columns move — the edges, or
            // the middle — is the whole question.
            if chunk.features != want.features && shown < 3 {
                shown += 1;
                let bad = (0..n_feat * feat_w)
                    .find(|&k| chunk.features[k] != want.features[k])
                    .unwrap_or(0);
                println!(
                    "  chunk {i} read {} base {} | dwell rows equal to dwells_flat: {dwell_ok}",
                    read_ids[i], base_indices[i]
                );
                println!(
                    "    first diff at row {} ({}) col {}: ours {} theirs {}",
                    bad / feat_w,
                    spec.chunk.feature_channels[bad / feat_w].name(),
                    bad % feat_w,
                    chunk.features[bad],
                    want.features[bad]
                );
                println!("    ours   row0 {:?}", &ours_dwell[..8]);
                println!("    want   row0 {:?}", &want.features[..8]);
                println!("    corpus dwell {:?}", &their_dwell[..8]);
            }
            // Is our signal window the corpus's, translated? Search a small
            // shift; a clean minimum at a non-zero offset says the values are
            // right and the placement is not.
            best_shifts.push(best_shift(
                &chunk.signal[..spec.chunk.signal_len],
                &want.signal[..spec.chunk.signal_len],
            ));
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
        // Per signal CHANNEL, not pooled. Channel 0 is the plain normalised
        // current in the window; it depends on the trim, the reverse, the
        // normalisation and the window placement, but NOT on the base
        // boundaries. Channel 1 (the k-mer residual) does depend on them. So
        // pooling the two hides which half of the pipeline disagrees.
        let sig_rows = spec.tensor_shape(WaveformTensor::Signal)[0];
        let sig_cols = spec.tensor_shape(WaveformTensor::Signal)[1];
        println!(
            "assembly: max |d| per signal channel {:?}",
            (0..sig_rows)
                .map(|c| format!("ch{c}={:.3e}", per_sig_ch[c]))
                .collect::<Vec<_>>()
        );
        let _ = sig_cols;
        println!("assembly: {exact} of {compared} chunks bit-identical");
        println!(
            "assembly: features reproduced with refinement ON {}, OFF {}, neither {}",
            refine_tally.0, refine_tally.1, refine_tally.2
        );
        let mut rows: Vec<(usize, f64)> = per_row.iter().copied().enumerate().collect();
        rows.sort_by(|a, b| b.1.total_cmp(&a.1));
        println!(
            "assembly: worst feature rows {:?}",
            rows.iter()
                .take(4)
                .map(|(r, m)| (
                    spec.chunk
                        .feature_channels
                        .get(*r)
                        .map(|c| c.name())
                        .unwrap_or("?"),
                    format!("{m:.3e}")
                ))
                .collect::<Vec<_>>()
        );
        let span = |v: &[i64]| {
            if v.is_empty() {
                return "-".to_string();
            }
            let (mut lo, mut hi, mut over) = (i64::MAX, i64::MIN, 0);
            for &x in v {
                lo = lo.min(x);
                hi = hi.max(x);
                if x > 200 {
                    over += 1;
                }
            }
            format!("n={} range {lo}..{hi}, {over} over 200 bases", v.len())
        };
        println!(
            "assembly: anchor 11-mer exact {seq_ok}, case-only {seq_case_only}, \
             different letters {seq_real} (of {compared})"
        );
        println!(
            "assembly: uppercase-only ref: {} match / {} miss | has lowercase: {} match / {} miss",
            cross.0, cross.1, cross.2, cross.3
        );
        println!("assembly: matched  {}", span(&matched_bases));
        println!("assembly: mismatch {}", span(&missed_bases));
        if let Some(dp) = args.dump_dwells.as_ref() {
            // The read's reference span, as WE compute it from the CIGAR. The
            // level array is zero-padded at both ends where no full k-mer
            // window fits, so a span that is off by one moves those zeros --
            // outside the feature window, and therefore invisible to every
            // comparison above, but seen by the DP over the whole read.
            let mut spans = String::new();
            for rid in read_ids.iter().take(n) {
                if let Some(r) = escapepod_signal::parse_uuid_flexible(rid)
                    .ok()
                    .and_then(|id| by_id.get(&id))
                {
                    spans.push_str(&format!("{}\t{}\t{}\n", rid, r.ref_start, r.ref_end));
                }
            }
            std::fs::write(format!("{}.spans.tsv", dp.display()), spans).unwrap();
            let bytes = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
            std::fs::write(
                format!("{}.ours_signal.f32", dp.display()),
                bytes(&ours_sig),
            )
            .unwrap();
            std::fs::write(
                format!("{}.ours_refined.f32", dp.display()),
                bytes(&ours_on),
            )
            .unwrap();
            std::fs::write(
                format!("{}.ours_unrefined.f32", dp.display()),
                bytes(&ours_off),
            )
            .unwrap();
            println!(
                "assembly: wrote our dwell rows ({n} x {w_dwell}) to {}.ours_{{refined,unrefined}}.f32",
                dp.display()
            );
        }
        println!("assembly: focus delta {}", histogram(&focus_delta));
        println!("assembly: best signal shift {}", histogram(&best_shifts));
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

/// `{u128 id, u32 len, len x i16}*`, length-prefixed. A throwaway format for a
/// throwaway file: it exists so a parity hypothesis costs a rebuild rather than
/// a re-index, and nothing but this example ever reads it.
fn save_signal_cache(path: &Path, m: &HashMap<uuid::Uuid, Vec<i16>>) {
    let mut buf = Vec::new();
    buf.extend_from_slice(&(m.len() as u64).to_le_bytes());
    for (id, sig) in m {
        buf.extend_from_slice(id.as_bytes());
        buf.extend_from_slice(&(sig.len() as u32).to_le_bytes());
        for &v in sig {
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }
    std::fs::write(path, buf).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
}

fn load_signal_cache(path: &Path) -> HashMap<uuid::Uuid, Vec<i16>> {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut out = HashMap::new();
    let n = u64::from_le_bytes(b[..8].try_into().unwrap()) as usize;
    let mut off = 8usize;
    for _ in 0..n {
        let id = uuid::Uuid::from_bytes(b[off..off + 16].try_into().unwrap());
        off += 16;
        let len = u32::from_le_bytes(b[off..off + 4].try_into().unwrap()) as usize;
        off += 4;
        let sig: Vec<i16> = b[off..off + 2 * len]
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&c| i16::from_le_bytes(c))
            .collect();
        off += 2 * len;
        out.insert(id, sig);
    }
    out
}

/// The integer shift in `[-32, 32]` that best aligns `ours` onto `theirs`.
///
/// Answers the one question an aggregate residual cannot: are these the same
/// samples in the wrong place, or different samples? A clean minimum away from
/// zero is a displaced window; no minimum at all is a value difference.
fn best_shift(ours: &[f32], theirs: &[f32]) -> i64 {
    let n = ours.len() as i64;
    let mut best = (f64::INFINITY, 0i64);
    for s in -32..=32i64 {
        let (mut sum, mut count) = (0.0f64, 0usize);
        for i in 0..n {
            let j = i + s;
            if j < 0 || j >= n {
                continue;
            }
            sum += (ours[i as usize] as f64 - theirs[j as usize] as f64).abs();
            count += 1;
        }
        if count > 0 {
            let mean = sum / count as f64;
            if mean < best.0 {
                best = (mean, s);
            }
        }
    }
    best.1
}

/// `value: count` for the commonest few values, so a systematic offset is
/// visible as one bucket rather than averaged into a summary statistic.
fn histogram(v: &[i64]) -> String {
    let mut counts: HashMap<i64, usize> = HashMap::new();
    for &x in v {
        *counts.entry(x).or_default() += 1;
    }
    let mut pairs: Vec<(i64, usize)> = counts.into_iter().collect();
    pairs.sort_by_key(|&(_, n)| std::cmp::Reverse(n));
    pairs
        .iter()
        .take(6)
        .map(|(val, n)| format!("{val}:{n}"))
        .collect::<Vec<_>>()
        .join(" ")
}

fn read_i64(path: &Path) -> Vec<i64> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    bytes
        .as_chunks::<8>()
        .0
        .iter()
        .map(|&c| i64::from_le_bytes(c))
        .collect()
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
