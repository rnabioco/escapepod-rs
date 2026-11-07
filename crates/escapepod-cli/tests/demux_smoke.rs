// SPDX-License-Identifier: MIT

//! End-to-end smoke tests for the `escpod demux` stage chain (#503).
//!
//! `dir_inputs` only checks that each stage accepts a directory. These run the
//! stages *into* one another on the tracked 20-read fixture and check what
//! comes out the far end:
//!
//! - `detect -> fingerprint -> classify` with a synthetic `DtwSvmModel` (the
//!   kernel-weighted builder from `escapepod-demux/benches/classify.rs`), whose
//!   references are two of the run's own fingerprints. A read that sits at DTW
//!   distance 0 from a reference must come back with that reference's label,
//!   so a stage that mis-orders, drops or relabels reads fails here.
//! - `classify -> split`, writing one POD5 per barcode with every read routed.
//!
//! What is **not** covered: `demux basecall --barcodes`. It needs a CRF encoder
//! ONNX bundle, and there is no model-free path through it (the lattice decode
//! has no CLI entry without an encoder), so `basecall`'s CSV-in half is only
//! reachable with a real bundle; `split` is instead driven from a
//! `read_id,barcode` CSV, which is the shape `basecall --barcodes` emits.

#![cfg(feature = "cli")]

mod common;

use common::{Workspace, assert_has_rows, assert_ok};
use escapepod_demux::{DtwSvmModel, KernelParams};
use std::collections::{BTreeMap, HashMap};
use std::path::Path;

/// Label of the class whose reference is the first read's fingerprint.
const LABEL_A: i32 = 7;
/// Label of the class whose reference is the last read's fingerprint.
const LABEL_B: i32 = 9;

/// The classifications CSV spells a label `BC{label:02}`.
fn bc(label: i32) -> String {
    format!("BC{label:02}")
}

/// `read_id -> fingerprint` from the `fingerprint` CSV.
fn read_fingerprints(path: &Path) -> Vec<(String, Vec<f64>)> {
    let text = std::fs::read_to_string(path).unwrap();
    text.lines()
        .skip(1)
        .filter(|l| !l.is_empty())
        .map(|l| {
            let mut it = l.split(',');
            let id = it.next().unwrap().to_string();
            let fp = it.map(|v| v.parse::<f64>().unwrap()).collect();
            (id, fp)
        })
        .collect()
}

/// Two-class kernel-weighted `DtwSvmModel` with one reference per class.
fn two_class_model(ref_a: &[f64], ref_b: &[f64]) -> DtwSvmModel {
    let n_classes = 2;
    DtwSvmModel {
        version: "1.0".to_string(),
        training_fingerprints: vec![ref_a.to_vec(), ref_b.to_vec()],
        training_labels: vec![LABEL_A, LABEL_B],
        support_indices: vec![0, 1],
        dual_coef: vec![vec![0.0; 2]; n_classes - 1],
        intercept: vec![0.0; 1],
        classes: vec![LABEL_A, LABEL_B],
        kernel_params: KernelParams {
            gamma: 1.0,
            power: 2.0,
        },
        window: None,
        penalty: 0.0,
        label_mapper: HashMap::from([(0, LABEL_A), (1, LABEL_B)]),
        thresholds: None,
        prob_a: None,
        prob_b: None,
        n_classes,
        noise_class: false,
        use_kernel_weighted: true,
    }
}

/// `read_id -> barcode` from a classifications CSV (sorted: row order is
/// nondeterministic).
fn read_calls(path: &Path) -> BTreeMap<String, String> {
    let text = std::fs::read_to_string(path).unwrap();
    text.lines()
        .skip(1)
        .filter(|l| !l.is_empty())
        .map(|l| {
            let mut it = l.split(',');
            (
                it.next().unwrap().to_string(),
                it.next().unwrap().to_string(),
            )
        })
        .collect()
}

fn count_reads(pod5s: &[std::path::PathBuf]) -> usize {
    pod5s
        .iter()
        .map(|p| {
            escapepod_signal::pod5::Reader::open(p)
                .unwrap()
                .read_count()
                .unwrap()
        })
        .sum()
}

#[test]
fn detect_fingerprint_classify_labels_reads_by_nearest_reference() {
    let ws = Workspace::new();
    let boundaries = ws.boundaries();
    let fp_csv = ws.out("fingerprints.csv");
    assert_ok(&[
        "demux",
        "fingerprint",
        ws.dir(),
        "--boundaries",
        boundaries.to_str().unwrap(),
        "-o",
        fp_csv.to_str().unwrap(),
    ]);
    assert_has_rows(&fp_csv);
    let fps = read_fingerprints(&fp_csv);
    assert!(fps.len() >= 2, "need two reads to build two classes");
    // Two references that are not each other's neighbours: the two reads that
    // are furthest apart in the first feature.
    let (ia, ib) = {
        let mut order: Vec<usize> = (0..fps.len()).collect();
        order.sort_by(|&a, &b| fps[a].1[0].total_cmp(&fps[b].1[0]));
        (order[0], *order.last().unwrap())
    };
    assert_ne!(fps[ia].1, fps[ib].1);

    let model_path = ws.out("model.json");
    std::fs::write(
        &model_path,
        serde_json::to_string(&two_class_model(&fps[ia].1, &fps[ib].1)).unwrap(),
    )
    .unwrap();

    let calls_csv = ws.out("classifications.csv");
    assert_ok(&[
        "demux",
        "classify",
        fp_csv.to_str().unwrap(),
        "--model",
        model_path.to_str().unwrap(),
        "-o",
        calls_csv.to_str().unwrap(),
    ]);
    assert_has_rows(&calls_csv);
    let calls = read_calls(&calls_csv);

    // Every fingerprinted read is classified exactly once.
    assert_eq!(calls.len(), fps.len(), "one call per fingerprinted read");
    for (id, _) in &fps {
        assert!(calls.contains_key(id), "read {id} missing from output");
    }
    // The model's labels, never class indices or placeholders.
    for (id, bc_) in &calls {
        assert!(
            *bc_ == bc(LABEL_A) || *bc_ == bc(LABEL_B) || bc_.contains("-1"),
            "read {id}: unexpected barcode {bc_:?}"
        );
    }
    // A read identical to a reference is that reference's class.
    assert_eq!(calls[&fps[ia].0], bc(LABEL_A), "read {}", fps[ia].0);
    assert_eq!(calls[&fps[ib].0], bc(LABEL_B), "read {}", fps[ib].0);
}

#[test]
fn classify_into_split_routes_every_read_to_its_barcode() {
    let ws = Workspace::new();
    let boundaries = ws.boundaries();
    let fp_csv = ws.out("fingerprints.csv");
    assert_ok(&[
        "demux",
        "fingerprint",
        ws.dir(),
        "--boundaries",
        boundaries.to_str().unwrap(),
        "-o",
        fp_csv.to_str().unwrap(),
    ]);
    let fps = read_fingerprints(&fp_csv);
    let model_path = ws.out("model.json");
    std::fs::write(
        &model_path,
        serde_json::to_string(&two_class_model(&fps[0].1, &fps[fps.len() - 1].1)).unwrap(),
    )
    .unwrap();
    let calls_csv = ws.out("classifications.csv");
    assert_ok(&[
        "demux",
        "classify",
        fp_csv.to_str().unwrap(),
        "--model",
        model_path.to_str().unwrap(),
        "-o",
        calls_csv.to_str().unwrap(),
    ]);
    let calls = read_calls(&calls_csv);

    let split_dir = ws.out("split");
    assert_ok(&[
        "demux",
        "split",
        ws.dir(),
        "--classifications",
        calls_csv.to_str().unwrap(),
        "-d",
        split_dir.to_str().unwrap(),
    ]);
    let written: Vec<_> = std::fs::read_dir(&split_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "pod5"))
        .collect();
    assert!(!written.is_empty(), "no POD5 written");
    // One file per distinct barcode actually called.
    let distinct: std::collections::BTreeSet<_> = calls.values().collect();
    assert_eq!(written.len(), distinct.len(), "{written:?} vs {distinct:?}");
    // And no read lost or duplicated on the way through.
    assert_eq!(count_reads(&written), calls.len());
}

#[test]
fn split_from_a_read_id_barcode_csv_routes_by_column() {
    // The shape `demux basecall --barcodes` emits; basecall itself needs a CRF
    // encoder bundle and is not coverable here.
    let ws = Workspace::new();
    let boundaries = ws.boundaries();
    let text = std::fs::read_to_string(&boundaries).unwrap();
    let ids: Vec<&str> = text
        .lines()
        .skip(1)
        .filter(|l| !l.is_empty())
        .map(|l| l.split(',').next().unwrap())
        .collect();
    let mut csv = String::from("read_id,barcode\n");
    for (i, id) in ids.iter().enumerate() {
        csv.push_str(&format!(
            "{id},{}\n",
            if i % 2 == 0 { "bc01" } else { "bc02" }
        ));
    }
    let map = ws.out("assign.csv");
    std::fs::write(&map, csv).unwrap();
    let split_dir = ws.out("split");
    assert_ok(&[
        "demux",
        "split",
        ws.dir(),
        "--classifications",
        map.to_str().unwrap(),
        "-d",
        split_dir.to_str().unwrap(),
    ]);
    let mut names: Vec<String> = std::fs::read_dir(&split_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.ends_with(".pod5"))
        .collect();
    names.sort();
    assert_eq!(names.len(), 2, "{names:?}");
    assert!(names[0].contains("bc01") && names[1].contains("bc02"));
    let paths: Vec<_> = names.iter().map(|n| split_dir.join(n)).collect();
    assert_eq!(count_reads(&paths), ids.len());
}
