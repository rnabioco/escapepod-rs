// SPDX-License-Identifier: MIT

//! The windowed variant's scan and chunk assembly, on the fixture reads.
//!
//! No graph and no weights: this is the half of the windowed path that has to
//! match the corpus builder bit for bit, and it is testable without an ONNX
//! runtime because [`waveform::assemble_chunk`] takes the k-mer levels rather
//! than a bundle. `examples/verify_waveform_model.rs` is the other half — the
//! same assembly against the corpus's own stored tensors, on real weights.
//!
//! # Both branches, deliberately
//!
//! rnabioco/escapepod-rs#306 asks for goldens that exercise more than the
//! majority path, and names the reason: `escapepod-classify` reproduced a
//! superseded feature definition for two months and its counted golden missed
//! it, because all 19 fixture reads took the other branch. Windowing has two
//! branches and they are easy to leave untested, because on a normal read the
//! window sits comfortably inside the signal and the feature offsets sit
//! comfortably inside the map:
//!
//! * the window **fits** — every sample is copied, nothing is padded;
//! * the window **overhangs** — the samples that exist keep their alignment
//!   and the rest is zero, which is not the same as sliding the window inwards.
//!
//! The second is forced here by asking for a window wider than any fixture
//! read, so the branch cannot go unexercised because the fixtures happened to
//! be long enough. The same is done for the feature offsets, and for a read
//! whose anchor the aligner never reached.

use escapepod_classify::recipe::KmerLevels;
use escapepod_classify::{Pod5Index, WaveformSpec, WaveformTensor, junction_positions, waveform};
use escapepod_signal::chunk::{
    BaseJustify, ChunkSpec, FeatureChannel, SeqEncoding, SignalChannel, SignalNorm,
};
use escapepod_signal::seq_encoding::KmerContext;
use std::collections::HashSet;
use std::path::{Path, PathBuf};

const MOTIF: &str = "CCAGGC";
const ARM: &str = "GGCTTCTTCTTGCTCTT";
/// The shipped windowed bundle's anchor: the A of CCA, one base *before* the
/// arm the motif is identified by.
const MOTIF_OFFSET: usize = 2;

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

/// The shipped geometry, with the channel lists a bundle would declare.
fn spec() -> WaveformSpec {
    WaveformSpec {
        chunk: ChunkSpec {
            signal_context: (90, 300),
            signal_len: 390,
            base_justify: BaseJustify::End,
            signal_channels: vec![SignalChannel::Current, SignalChannel::KmerResidual],
            seq_encoding: SeqEncoding::SignalKmer {
                ctx: KmerContext::new(4, 4),
            },
            feature_offsets: (0, 20),
            feature_channels: vec![
                FeatureChannel::Dwell,
                FeatureChannel::DwellLog,
                FeatureChannel::DwellMean,
                FeatureChannel::DwellStd,
                FeatureChannel::DwellRatio,
                FeatureChannel::LevelMean,
                FeatureChannel::LevelMedian,
                FeatureChannel::LevelStd,
                FeatureChannel::LevelRange,
                FeatureChannel::KmerExpected,
                FeatureChannel::KmerResidual,
                FeatureChannel::KmerResidualAbs,
            ],
            dwell_window: escapepod_signal::chunk::DEFAULT_DWELL_WINDOW,
        },
        reverse_signal: true,
        normalization: SignalNorm::MedianMad,
        refine: Some(escapepod_signal::chunk::RefineParams::default()),
        positive_class: 0,
    }
}

/// The fixture k-mer table (5-mers), as the levels the residual channels and
/// the refinement are defined against.
fn kmer() -> KmerLevels {
    let (map, k) =
        escapepod_signal::resquiggle::load_kmer_table(&fixtures().join("bundle/kmer_levels.tsv"))
            .expect("the fixture k-mer table loads");
    KmerLevels {
        map,
        k,
        center_idx: k / 2,
    }
}

/// Every fixture read that anchors, with its raw signal.
fn anchored_reads() -> (
    waveform::WaveformScan,
    Vec<(waveform::WaveformRead, Vec<i16>)>,
) {
    let geometry = junction_positions(
        &fixtures().join("trna_reference.fa"),
        MOTIF,
        MOTIF_OFFSET,
        ARM,
    )
    .expect("fixture reference geometry");
    let scan = waveform::scan_bam(
        &fixtures().join("trna_mappings_padded.bam"),
        &geometry,
        MOTIF_OFFSET,
        1,
    )
    .expect("fixture BAM scan");

    let wanted: HashSet<uuid::Uuid> = scan.anchored.keys().copied().collect();
    let index = Pod5Index::build(&[fixtures().join("trna_reads.pod5")], &wanted)
        .expect("fixture POD5 index");
    let extractors = index.extractors().unwrap();
    let mut out = Vec::new();
    for (id, read) in &scan.anchored {
        if let Some(info) = index.reads().get(id) {
            let raw = escapepod_classify::pipeline::signal_adc(info, &extractors).unwrap();
            if raw.len() as i64 == read.ns {
                out.push((read.clone(), raw));
            }
        }
    }
    out.sort_by_key(|(r, _)| r.read_id);
    (scan, out)
}

fn references() -> std::collections::HashMap<String, String> {
    escapepod_classify::reference_sequences(&fixtures().join("trna_reference.fa")).unwrap()
}

/// The anchor is the motif's own origin plus the bundle's offset, which is a
/// *different base* from the junction the column variants use. Getting this
/// from the junction instead would displace every window by one base and
/// validate cleanly — which is why the two are separate fields.
#[test]
fn the_anchor_is_the_motif_offset_not_the_arm() {
    let geometry = junction_positions(
        &fixtures().join("trna_reference.fa"),
        MOTIF,
        MOTIF_OFFSET,
        ARM,
    )
    .unwrap();
    assert!(!geometry.is_empty());
    for g in geometry.values() {
        assert_eq!(g.junction, g.motif_start + MOTIF_OFFSET);
        // The arm still starts at +3, so the anchor is one base earlier.
        assert_eq!(g.cca_a, g.motif_start + 2);
        assert_eq!(g.divergent, g.motif_start + 3 + ARM.len());
    }
}

/// The scan and the assembly on the shipped geometry: reads anchor, chunks
/// come out at the declared shapes, and nothing is `NaN` (these tensors feed a
/// network, where one `NaN` poisons the forward pass).
#[test]
fn the_shipped_geometry_assembles_from_pod5_and_bam() {
    let spec = spec();
    let kmer = kmer();
    let refs = references();
    let (scan, reads) = anchored_reads();
    assert!(
        !reads.is_empty(),
        "no fixture read anchored at {MOTIF}+{MOTIF_OFFSET}"
    );

    let mut assembled = 0usize;
    for (read, raw) in &reads {
        let reference = refs.get(&read.reference).expect("a known reference");
        let Some(chunk) =
            waveform::assemble_chunk(Some(&kmer), &spec, read, reference.as_bytes(), raw)
        else {
            continue;
        };
        assembled += 1;
        let [c, l] = spec.tensor_shape(WaveformTensor::Signal);
        assert_eq!(chunk.signal.len(), c * l);
        let [r, sl] = spec.tensor_shape(WaveformTensor::Sequence);
        assert_eq!(chunk.sequence.len(), r * sl);
        assert_eq!((chunk.sequence_rows, chunk.sequence_cols), (r, sl));
        let [f, w] = spec.tensor_shape(WaveformTensor::Features);
        assert_eq!(chunk.features.len(), f * w);

        for (name, t) in [
            ("signal", &chunk.signal),
            ("sequence", &chunk.sequence),
            ("features", &chunk.features),
        ] {
            assert!(
                t.iter().all(|v| v.is_finite()),
                "read {}: the {name} tensor carries a non-finite value",
                read.read_id
            );
        }
        // The sequence encoding is one-hot: a column is either cold or has
        // exactly one hot row per k-mer position.
        assert!(chunk.sequence.iter().all(|&v| v == 0.0 || v == 1.0));
    }
    assert!(
        assembled > 0,
        "no chunk assembled from {} reads",
        reads.len()
    );
    // Both outcomes of the scan are present in the fixture, so the skip path
    // is exercised rather than assumed.
    assert!(scan.records_scanned as usize > scan.anchored.len());
    assert!(!scan.skips.is_empty());
}

/// The overhang branch, forced rather than hoped for.
///
/// A window wider than any read must pad, and the padding rule is the part
/// worth pinning: the samples that exist keep their **alignment** — offset by
/// how far the window underflows — rather than sliding to the left edge. A
/// slid window is a correctly shaped tensor of real samples, at the wrong
/// phase.
#[test]
fn a_window_wider_than_the_read_pads_without_sliding() {
    let kmer = kmer();
    let refs = references();
    let (_, reads) = anchored_reads();

    // Sized from the fixtures rather than guessed, so the branch cannot go
    // untested because a read happened to be longer than a hard-coded window.
    let half = reads.iter().map(|(_, raw)| raw.len()).max().unwrap() + 100;
    let mut spec = spec();
    spec.chunk.signal_context = (half as i64, half as i64);
    spec.chunk.signal_len = 2 * half;
    // One channel keeps the assertion about *where* the samples land simple.
    spec.chunk.signal_channels = vec![SignalChannel::Current];

    let mut padded = 0usize;
    for (read, raw) in &reads {
        let reference = refs.get(&read.reference).unwrap();
        let Some(chunk) =
            waveform::assemble_chunk(Some(&kmer), &spec, read, reference.as_bytes(), raw)
        else {
            continue;
        };
        assert_eq!(chunk.signal.len(), 2 * half);
        let hot = chunk.signal.iter().filter(|v| **v != 0.0).count();
        assert!(hot > 0, "read {}: the window copied nothing", read.read_id);
        assert!(
            hot < 2 * half,
            "read {}: a window this wide must be padded",
            read.read_id
        );
        // The window starts `half` samples before the focus, and the read
        // starts at sample 0, so the copy lands at `half - focus` and
        // everything before it is padding.
        let first_hot = chunk.signal.iter().position(|v| *v != 0.0).unwrap();
        assert!(
            first_hot > 0,
            "read {}: the window underflows the signal, so it must pad on the LEFT",
            read.read_id
        );
        padded += 1;
    }
    assert!(padded > 0, "no read exercised the padding branch");
}

/// The same for the feature tensor: offsets that run past the end of the map
/// pad on the right, and offsets before its start pad on the left, so offset
/// zero keeps landing on the anchor base either way.
#[test]
fn feature_offsets_past_the_map_pad_rather_than_shift() {
    let kmer = kmer();
    let refs = references();
    let (_, reads) = anchored_reads();

    let mut narrow = spec();
    narrow.chunk.feature_offsets = (0, 20);
    let mut wide = spec();
    // Wider than any tRNA reference, so the right-hand side must pad.
    wide.chunk.feature_offsets = (-400, 400);

    let mut checked = 0usize;
    for (read, raw) in &reads {
        let reference = refs.get(&read.reference).unwrap();
        let (Some(a), Some(b)) = (
            waveform::assemble_chunk(Some(&kmer), &narrow, read, reference.as_bytes(), raw),
            waveform::assemble_chunk(Some(&kmer), &wide, read, reference.as_bytes(), raw),
        ) else {
            continue;
        };
        let (nw, ww) = (narrow.chunk.feature_width(), wide.chunk.feature_width());
        assert_eq!(b.features.len(), 12 * ww);
        // Offset 0 sits at column 0 of the narrow tensor and at column 400 of
        // the wide one. Same base, same numbers, whatever the padding around
        // it.
        for row in 0..12 {
            for k in 0..nw {
                let lhs = a.features[row * nw + k];
                let rhs = b.features[row * ww + 400 + k];
                assert_eq!(
                    lhs.to_bits(),
                    rhs.to_bits(),
                    "read {}: row {row}, offset {k} moved when the window widened",
                    read.read_id
                );
            }
        }
        // ...and the far edges, which no base can reach, are padding.
        assert!(b.features[..50].iter().all(|&v| v == 0.0));
        checked += 1;
    }
    assert!(checked > 0, "no read exercised the feature padding branch");
}

/// A base the map does not resolve yields no chunk at all — the "no chunk, no
/// call" the bundle declares, rather than a window of zeros that would score
/// like a real one.
#[test]
fn an_anchor_outside_the_map_yields_no_chunk() {
    let spec = spec();
    let kmer = kmer();
    let refs = references();
    let (_, reads) = anchored_reads();
    let (read, raw) = reads.first().expect("at least one anchored read");
    let reference = refs.get(&read.reference).unwrap();

    let mut off_the_end = read.clone();
    off_the_end.base_index = 1_000_000;
    assert!(
        waveform::assemble_chunk(Some(&kmer), &spec, &off_the_end, reference.as_bytes(), raw)
            .is_none()
    );

    let mut negative = read.clone();
    negative.base_index = -1;
    assert!(
        waveform::assemble_chunk(Some(&kmer), &spec, &negative, reference.as_bytes(), raw)
            .is_none()
    );
}

/// Refinement moves the boundaries every feature is measured over, so a corpus
/// built with it and scored without reads a different stretch of signal for
/// every base. It must therefore *change something* — a run where it silently
/// no-ops would produce exactly the output shape it should.
#[test]
fn refinement_moves_the_spans_it_is_asked_to_move() {
    let kmer = kmer();
    let refs = references();
    let (_, reads) = anchored_reads();

    let refined = spec();
    let mut plain = spec();
    plain.refine = None;

    let mut differed = 0usize;
    let mut compared = 0usize;
    for (read, raw) in &reads {
        let reference = refs.get(&read.reference).unwrap();
        let (Some(a), Some(b)) = (
            waveform::assemble_chunk(Some(&kmer), &refined, read, reference.as_bytes(), raw),
            waveform::assemble_chunk(Some(&kmer), &plain, read, reference.as_bytes(), raw),
        ) else {
            continue;
        };
        compared += 1;
        if a.features != b.features || a.signal != b.signal {
            differed += 1;
        }
    }
    assert!(compared > 0);
    assert!(
        differed > 0,
        "refinement changed nothing on any of {compared} reads, so the flag is not \
         reaching the DP"
    );
}
