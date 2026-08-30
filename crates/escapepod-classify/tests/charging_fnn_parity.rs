// SPDX-License-Identifier: MIT

//! Parity of the `feature_model` scorer against the reference implementation
//! (`escapepod_models.charging`'s `feature_nn_fold` / `feature_nn_input`, plus
//! the exported graph). Golden vectors come from
//! `tests/fixtures/gen_charging_fnn_golden.py`; floats travel as IEEE-754 bit
//! patterns.
//!
//! `charging_parity.rs` pins the feature chain and the GBM. This pins the
//! three rules between the flat feature vector and the graph — the fold, the
//! per-channel standardisation, and the explicit encoding of missingness —
//! each of which fails *silently* if reproduced wrong: fold the other way and
//! every read still scores, on a transposed input.
//!
//! The check is deliberately in two parts, for the reason
//! `scripts/ldx/analysis/verify_crf_decode.py` gives about running our pass on
//! the reference's own input:
//!
//! 1. **The tensor, exactly.** Both sides fold the *identical* reference
//!    feature vector, so the input tensors must agree bit for bit. A residue
//!    here is a wrong rule, not arithmetic.
//! 2. **The probability, end to end.** Through escpod's own feature grid,
//!    which differs from NumPy's in the last bits (`charging_parity.rs` bounds
//!    that at 1e-4). Any residue here is therefore attributable to the
//!    features, which part 1 has already excluded as a rule difference.

// The scorer under test only exists with the ONNX runtime linked. `escpod`
// always has it (the CLI's `classify` feature enables it); a bare
// `cargo test -p escapepod-classify` does not.
#![cfg(feature = "fnn-onnx")]

use escapepod_classify::{
    ChargingBundle, ChargingScorer, Pod5Index, classify_reads, junction_positions,
    resolve_orientation, scan_bam,
};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn golden() -> Value {
    let text = std::fs::read_to_string(fixtures().join("charging_fnn_golden.json")).unwrap();
    serde_json::from_str(&text).unwrap()
}

fn f32_vec(v: &Value) -> Vec<f32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| f32::from_bits(x.as_u64().unwrap() as u32))
        .collect()
}

fn net(bundle: &ChargingBundle) -> &escapepod_classify::FeatureNet {
    match &bundle.scorer {
        ChargingScorer::FeatureNn(n) => n,
        other => panic!("fixture bundle is {}, not a feature network", other.kind()),
    }
}

/// The fold, the gauge and the missingness encoding, on the reference's own
/// feature vectors — so the comparison is exact.
#[test]
fn feature_tensor_matches_reference_bit_for_bit() {
    let bundle = ChargingBundle::load(&fixtures().join("bundle_fnn")).unwrap();
    let g = golden();
    let net = net(&bundle);
    assert_eq!(net.n_value_channels(), 4);
    assert_eq!(net.n_offsets(), 25);
    assert_eq!(
        net.n_features(),
        bundle.feature_space().unwrap().columns.len()
    );

    let mut n_masked = 0usize;
    for gr in g["reads"].as_array().unwrap() {
        let id = gr["read_id"].as_str().unwrap();
        // The reference's features, selected through the bundle's own column
        // order — the vector `pipeline::classify_reads` hands the scorer.
        let grid = f32_vec(&gr["features_bits"]);
        let cols = bundle.select_columns(&grid).unwrap();
        let got = net.input_tensor(&cols).unwrap();
        let want = f32_vec(&gr["input_bits"]);
        assert_eq!(got.len(), want.len(), "read {id}: tensor length");
        for (i, (&a, &b)) in got.iter().zip(&want).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "read {id} tensor element {i}: {a} vs {b}"
            );
        }
        n_masked += want[want.len() / 2..].iter().filter(|&&m| m == 0.0).count();
    }
    // If nothing were unresolved, the mask channels and the zeroing rule
    // would be untested and this file would pass against a model fed NaN.
    assert!(
        n_masked > 0,
        "no unresolved bases in the fixture: the missingness rule is untested"
    );
}

/// End to end through the real pipeline: the same code `escpod signal
/// classify` runs, against a bundle whose scorer is the network.
#[test]
fn fnn_bundle_classifies_like_the_reference() {
    let bundle = ChargingBundle::load(&fixtures().join("bundle_fnn")).unwrap();
    assert_eq!(bundle.scorer.kind(), "feature-nn (onnx)");
    assert!(bundle.scorer.as_gbm().is_none());
    // The feature half is shared verbatim with the GBM variant.
    assert_eq!(bundle.feature_space().unwrap().columns.len(), 100);
    assert_eq!(bundle.feature_space().unwrap().offsets.len(), 25);
    assert!(bundle.kmer.is_some(), "fixture recipe uses residuals");

    let g = golden();
    let geometry = junction_positions(
        &fixtures().join("trna_reference.fa"),
        &bundle.anchor.motif,
        bundle.anchor.motif_offset,
        &bundle.anchor.common_arm,
    )
    .unwrap();
    let scan = scan_bam(
        &fixtures().join("trna_mappings_padded.bam"),
        &geometry,
        &bundle.feature_space().unwrap().offsets,
        1,
    )
    .unwrap();
    let orientation = resolve_orientation(&scan.votes, 50).unwrap();
    let wanted: HashSet<uuid::Uuid> = scan.anchored.keys().copied().collect();
    let pod5 = Pod5Index::build(&[fixtures().join("trna_reads.pod5")], &wanted).unwrap();
    let (calls, _stats) = classify_reads(&bundle, &scan.anchored, &pod5, orientation).unwrap();

    let reads = g["reads"].as_array().unwrap();
    assert_eq!(calls.len(), reads.len(), "classified read count");
    let by_id: HashMap<uuid::Uuid, _> = calls.iter().map(|c| (c.read_id, c)).collect();

    let mut max_dev = 0.0f64;
    let mut spread = (1.0f64, 0.0f64);
    for gr in reads {
        let id: uuid::Uuid = gr["read_id"].as_str().unwrap().parse().unwrap();
        let call = by_id
            .get(&id)
            .unwrap_or_else(|| panic!("golden read {id} not classified"));
        let p_ref = f64::from_bits(gr["p_bits"].as_u64().unwrap());
        let dev = (call.p - p_ref).abs();
        max_dev = max_dev.max(dev);
        // Tolerance covers exactly one thing: the feature grid's own 1e-4
        // reduction-rounding headroom propagated through the graph. The
        // tensor test above has already excluded a rule difference.
        assert!(
            dev <= 1e-4,
            "read {id}: P = {} vs reference {p_ref} (dev {dev:e})",
            call.p
        );
        spread = (spread.0.min(p_ref), spread.1.max(p_ref));
    }
    // A golden whose probabilities all sat at 0.5 would pass against almost
    // anything; assert the fixture actually separates reads.
    assert!(
        spread.1 - spread.0 > 0.25,
        "golden probabilities span only {spread:?} — the fixture discriminates nothing"
    );
    eprintln!(
        "fnn end-to-end: max |dP| = {max_dev:e} over {} reads",
        reads.len()
    );
}

/// A bundle naming both scorers, or neither, is refused by name.
///
/// One format tag covers three models, which is how an unloadable bundle sat
/// in the registry for three days. "Both" has no defensible resolution and
/// "neither" is the raw-signal CNN variant, so each gets its own message
/// rather than whichever check happens to run first.
#[test]
fn exactly_one_scorer_is_required() {
    let read_meta = |dir: &Path| -> Value {
        serde_json::from_str(&std::fs::read_to_string(dir.join("metadata.json")).unwrap()).unwrap()
    };

    // Both: graft the GBM bundle's `gbm` block onto the fnn bundle.
    let dir = tempfile::tempdir().unwrap();
    for f in [
        "metadata.json",
        "charging_fnn_fixture.onnx",
        "kmer_levels.tsv",
    ] {
        std::fs::copy(fixtures().join("bundle_fnn").join(f), dir.path().join(f)).unwrap();
    }
    std::fs::copy(
        fixtures().join("bundle/model.gbm.json"),
        dir.path().join("model.gbm.json"),
    )
    .unwrap();
    let mut meta = read_meta(dir.path());
    meta["gbm"] = read_meta(&fixtures().join("bundle"))["gbm"].clone();
    std::fs::write(
        dir.path().join("metadata.json"),
        serde_json::to_string_pretty(&meta).unwrap(),
    )
    .unwrap();
    let err = ChargingBundle::load(dir.path()).unwrap_err().to_string();
    assert!(err.contains("both `gbm` and `feature_model`"), "got: {err}");

    // Neither: the shape a raw-signal CNN bundle has.
    let dir = tempfile::tempdir().unwrap();
    for f in ["metadata.json", "kmer_levels.tsv"] {
        std::fs::copy(fixtures().join("bundle_fnn").join(f), dir.path().join(f)).unwrap();
    }
    let mut meta = read_meta(dir.path());
    meta.as_object_mut().unwrap().remove("feature_model");
    std::fs::write(
        dir.path().join("metadata.json"),
        serde_json::to_string_pretty(&meta).unwrap(),
    )
    .unwrap();
    let err = ChargingBundle::load(dir.path()).unwrap_err().to_string();
    assert!(
        err.contains("none of `gbm`, `feature_model` or `waveform_model`"),
        "got: {err}"
    );

    // The raw-signal CNN variant: the same tag, a top-level `onnx`, and a
    // different input space. It is named, because "no `gbm` field" is what
    // sent someone looking for a corrupt file for three days.
    let mut meta = read_meta(dir.path());
    meta["onnx"] = serde_json::json!("charging_cnn_rna004.onnx");
    std::fs::write(
        dir.path().join("metadata.json"),
        serde_json::to_string_pretty(&meta).unwrap(),
    )
    .unwrap();
    let err = ChargingBundle::load(dir.path()).unwrap_err().to_string();
    assert!(err.contains("raw-signal CNN variant"), "got: {err}");
}

/// The ONNX is pinned by sha256 like every other bundle dependency: a swapped
/// graph is a different model, not a differently-calibrated one.
#[test]
fn fnn_bundle_rejects_a_tampered_graph() {
    let dir = tempfile::tempdir().unwrap();
    for f in [
        "metadata.json",
        "charging_fnn_fixture.onnx",
        "kmer_levels.tsv",
    ] {
        std::fs::copy(fixtures().join("bundle_fnn").join(f), dir.path().join(f)).unwrap();
    }
    let onnx = dir.path().join("charging_fnn_fixture.onnx");
    let mut bytes = std::fs::read(&onnx).unwrap();
    let last = bytes.len() - 2;
    bytes[last] ^= 0x01;
    std::fs::write(&onnx, bytes).unwrap();
    let err = ChargingBundle::load(dir.path()).unwrap_err().to_string();
    assert!(err.contains("checksum mismatch"), "got: {err}");
}

/// The declared channels must reproduce `features.order`. This is the check
/// that stands between a transposed input and a confident wrong answer, and
/// unlike the others it cannot be caught by a shape.
#[test]
fn a_channel_order_the_columns_contradict_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    for f in [
        "metadata.json",
        "charging_fnn_fixture.onnx",
        "kmer_levels.tsv",
    ] {
        std::fs::copy(fixtures().join("bundle_fnn").join(f), dir.path().join(f)).unwrap();
    }
    let mp = dir.path().join("metadata.json");
    let mut meta: Value = serde_json::from_str(&std::fs::read_to_string(&mp).unwrap()).unwrap();
    // Swap two value channels (and their masks): same shape, same count,
    // wrong assignment of columns to channels.
    meta["feature_model"]["input"]["channels"] = serde_json::json!([
        "mean",
        "dwell",
        "std",
        "resid",
        "mean_observed",
        "dwell_observed",
        "std_observed",
        "resid_observed"
    ]);
    std::fs::write(&mp, serde_json::to_string_pretty(&meta).unwrap()).unwrap();
    let err = ChargingBundle::load(dir.path()).unwrap_err().to_string();
    assert!(err.contains("declared fold"), "got: {err}");
}
