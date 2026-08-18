// SPDX-License-Identifier: MIT

//! The bundle's abstain rule, applied.
//!
//! A charging bundle can name reads the model must not be asked about
//! (`aligner_arm_depth == 0`: the aligner placed no common-arm base). The block
//! was parsed and then ignored, so those reads got a confident `cl` like any
//! other (rnabioco/escapepod-rs#230).
//!
//! **The bundle's own rationale for the rule is stale.** It cites 23-34% of
//! charged-library reads, measured under the *aligner*-derived span rule on the
//! yeast/v2 adapters. Measured here on 1.06M reads of an edx07 corpus with the
//! counting anchor, the rule fires on **0.85%** — the geometry it was written
//! for has largely been fixed upstream, which is the outcome
//! `rnabioco/aa-tRNA-seq-pipeline#110` is after. What it still catches is a
//! distinct population rather than a scoring failure; see
//! [`NoCallReason::Abstained`](escapepod_classify::NoCallReason).
//!
//! What needs pinning is therefore not that a flag exists but that the rule
//! *removes reads and loses none*: every read that used to be called is now
//! either called or explicitly no-called, with a reason.

use escapepod_classify::{
    ChargingBundle, Pod5Index, classify_reads, junction_positions, resolve_orientation, scan_bam,
};
use std::collections::HashSet;
use std::path::{Path, PathBuf};

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

/// Copy the fixture GBM bundle into `dir`, splicing `abstain` into its
/// metadata (the fixture predates the block).
fn bundle_with_abstain(dir: &Path, rule: Option<&str>) -> PathBuf {
    let src = fixtures().join("bundle");
    let dst = dir.join("bundle");
    std::fs::create_dir_all(&dst).unwrap();
    for entry in std::fs::read_dir(&src).unwrap() {
        let entry = entry.unwrap();
        std::fs::copy(entry.path(), dst.join(entry.file_name())).unwrap();
    }
    if let Some(rule) = rule {
        let path = dst.join("metadata.json");
        let mut meta: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        meta["abstain"] = serde_json::json!({
            "rule": rule,
            "emit": "no-call (absent cl tag / null score), NOT a default class",
        });
        std::fs::write(&path, serde_json::to_string_pretty(&meta).unwrap()).unwrap();
    }
    dst
}

/// Run the fixture corpus through a bundle, returning `(n_called, n_abstained)`.
fn run(bundle_dir: &Path) -> (usize, u64) {
    let bundle = ChargingBundle::load(bundle_dir).unwrap();
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
        &bundle.offsets,
        1,
    )
    .unwrap();
    let orientation = resolve_orientation(&scan.votes, 50).unwrap();
    let wanted: HashSet<uuid::Uuid> = scan.anchored.keys().copied().collect();
    let pod5 = Pod5Index::build(&[fixtures().join("trna_reads.pod5")], &wanted).unwrap();
    let (calls, stats) = classify_reads(&bundle, &scan.anchored, &pod5, orientation).unwrap();
    (calls.len(), stats.abstained)
}

/// The rule removes reads, and every read it removes is accounted for.
#[test]
fn abstaining_removes_reads_and_loses_none() {
    let tmp = tempfile::tempdir().unwrap();
    let (baseline, none_abstained) = run(&bundle_with_abstain(tmp.path(), None));
    assert_eq!(none_abstained, 0, "no rule, no abstentions");

    let tmp2 = tempfile::tempdir().unwrap();
    let with = bundle_with_abstain(tmp2.path(), Some("aligner_arm_depth == 0"));
    let (called, abstained) = run(&with);

    assert_eq!(
        called as u64 + abstained,
        baseline as u64,
        "every read that was called before is now called or no-called"
    );
    assert!(called > 0, "the rule excluded the entire corpus");
    // The 19 fixture reads all align through the common arm, so none is
    // excluded here — which is why the rule *firing* is pinned on coords
    // directly (`abstains_on_an_unreached_arm`), not left to a corpus that
    // happens not to contain the population. On a real edx07 run it is 0.85%.
    assert_eq!(
        abstained, 0,
        "fixture expectation changed — revisit this test"
    );
}

/// The rule itself, on coords rather than a corpus.
#[test]
fn abstains_on_an_unreached_arm() {
    use escapepod_classify::{Abstain, AbstainRule, JunctionCoords, MaskSource, abstained_by};

    let coords = |aligner_arm_depth: i32| JunctionCoords {
        feat_spans: Vec::new(),
        common_start_sig: 0,
        junction_sig: 0,
        mask_source: MaskSource::Exact,
        cca_a_sig: -1,
        cca_a_dwell: 0,
        junction_dwell: 0,
        arm_resolved_depth: 8,
        aligner_arm_depth,
        polya_mid_sig: -1,
        body_mid_sig: -1,
    };
    let rule = Abstain {
        rule: "aligner_arm_depth == 0".into(),
        kind: AbstainRule::NoAlignedArm,
    };

    assert_eq!(
        abstained_by(Some(&rule), &coords(0)),
        Some(AbstainRule::NoAlignedArm),
        "0 is the excluded case"
    );
    assert_eq!(
        abstained_by(Some(&rule), &coords(1)),
        None,
        "one arm base is enough"
    );
    assert_eq!(abstained_by(Some(&rule), &coords(17)), None);
    // No rule means score everything, which is what every bundle without an
    // `abstain` block asks for.
    assert_eq!(abstained_by(None, &coords(0)), None);
    // `arm_resolved_depth` is the COUNTED depth and is 8 in all of the above:
    // keying on it instead would abstain on nothing, since counting always
    // succeeds. That confusion is the whole reason the rule names the aligner.
    assert!(abstained_by(Some(&rule), &coords(0)).is_some());
}

/// Whitespace is presentation, not meaning.
#[test]
fn the_rule_is_read_regardless_of_spacing() {
    for rule in [
        "aligner_arm_depth == 0",
        "aligner_arm_depth==0",
        "  aligner_arm_depth   ==   0  ",
    ] {
        let tmp = tempfile::tempdir().unwrap();
        let dir = bundle_with_abstain(tmp.path(), Some(rule));
        let bundle = ChargingBundle::load(&dir).unwrap_or_else(|e| panic!("{rule:?}: {e}"));
        assert!(bundle.abstain.is_some(), "{rule:?}");
    }
}

/// A rule this runtime cannot evaluate is a load error, not a silent pass.
/// Accepting it would score exactly the reads the bundle excludes — the
/// failure #230 was.
#[test]
fn an_unevaluable_rule_is_refused() {
    for rule in [
        "aligner_arm_depth < 2",
        "arm_resolved_depth == 0",
        "p < 0.5",
        "",
    ] {
        let tmp = tempfile::tempdir().unwrap();
        let dir = bundle_with_abstain(tmp.path(), Some(rule));
        let err = ChargingBundle::load(&dir)
            .expect_err(&format!("{rule:?} should be refused"))
            .to_string();
        assert!(
            err.contains("cannot evaluate"),
            "{rule:?} gave the wrong error: {err}"
        );
    }
}
