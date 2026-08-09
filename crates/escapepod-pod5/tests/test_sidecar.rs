//! Integration tests for the `.p5s` sidecar: annotations and the read index
//! must coexist and preserve each other across writes, the POD5 file must
//! never change, and the identity binding must fail loudly on stale or
//! misplaced sidecars.

mod common;

use std::collections::HashMap;

use escapepod_pod5::operations::{
    AnnotateOptions, DesignOptions, read_annotation, read_design, write_annotation, write_design,
};
use escapepod_pod5::sidecar::{AnnotationSection, sidecar_path};
use escapepod_pod5::{Reader, Uuid};
use tempfile::TempDir;

use common::write_fixture;

const N_READS: usize = 25;

/// Build a fixture, return (tempdir, path, read ids, original pod5 bytes).
fn fixture() -> (TempDir, std::path::PathBuf, Vec<Uuid>, Vec<u8>) {
    let tmp = TempDir::new().expect("tempdir");
    let path = tmp.path().join("reads.pod5");
    let fx = write_fixture(&path, "sidecar_acq", N_READS, 700);
    let original = std::fs::read(&path).expect("read original bytes");
    (tmp, path, fx.read_ids, original)
}

/// Assign BC01/BC02 alternately to the first `n` reads.
fn make_assignments(ids: &[Uuid], n: usize) -> HashMap<Uuid, String> {
    ids.iter()
        .take(n)
        .enumerate()
        .map(|(i, id)| (*id, format!("BC{:02}", i % 2 + 1)))
        .collect()
}

#[test]
fn annotate_writes_sidecar_and_pod5_is_untouched() {
    let (_tmp, path, ids, original) = fixture();
    let mut assignments = make_assignments(&ids, 10);
    // An assignment for a read not in this file must be dropped silently.
    assignments.insert(Uuid::new_v4(), "BC99".to_string());

    let result = write_annotation(&path, &assignments, &AnnotateOptions::default()).unwrap();
    assert_eq!(result.total_reads, N_READS);
    assert_eq!(result.assigned_reads, 10, "foreign read must not count");
    assert_eq!(result.labels, 2);
    assert!(sidecar_path(&path).exists());

    assert_eq!(
        std::fs::read(&path).unwrap(),
        original,
        "annotation must never modify the POD5 file"
    );

    let annotation = read_annotation(&path, Some("barcode")).unwrap();
    assert_eq!(annotation.len(), 10);
    for (i, id) in ids.iter().take(10).enumerate() {
        assert_eq!(
            annotation.get(id),
            Some(format!("BC{:02}", i % 2 + 1)).as_deref()
        );
    }
    for id in ids.iter().skip(10) {
        assert_eq!(annotation.get(id), None, "unassigned reads must be absent");
    }
}

#[test]
fn annotation_and_index_preserve_each_other() {
    let (_tmp, path, ids, _original) = fixture();
    let assignments = make_assignments(&ids, ids.len());

    // annotate → index: the index rebuild must keep the annotation.
    write_annotation(&path, &assignments, &AnnotateOptions::default()).unwrap();
    let reader = Reader::open(&path).unwrap();
    let count = reader.build_and_write_index(sidecar_path(&path)).unwrap();
    assert_eq!(count, N_READS);
    let annotation = read_annotation(&path, Some("barcode")).unwrap();
    assert_eq!(annotation.len(), ids.len());

    // The sidecar-loaded index must resolve every read.
    let reader = Reader::open(&path).unwrap();
    let index = reader.read_index().unwrap();
    assert_eq!(index.len(), N_READS);
    for id in &ids {
        assert!(index.get(id).is_some(), "index lost read {id}");
    }
}

#[test]
fn annotate_after_index_preserves_index() {
    let (_tmp, path, ids, _original) = fixture();

    // index → annotate: record the index entries, annotate, compare.
    let reader = Reader::open(&path).unwrap();
    reader.build_and_write_index(sidecar_path(&path)).unwrap();
    let before: Vec<(Uuid, (usize, usize))> = {
        let index = reader.read_index().unwrap();
        ids.iter().map(|id| (*id, index.get(id).unwrap())).collect()
    };
    drop(reader);

    write_annotation(
        &path,
        &make_assignments(&ids, 5),
        &AnnotateOptions::default(),
    )
    .unwrap();

    let reader = Reader::open(&path).unwrap();
    let index = reader.read_index().unwrap();
    for (id, location) in before {
        assert_eq!(index.get(&id), Some(location), "index changed for {id}");
    }
}

#[test]
fn multiple_annotations_coexist() {
    let (_tmp, path, ids, _original) = fixture();
    write_annotation(
        &path,
        &make_assignments(&ids, 10),
        &AnnotateOptions::default(),
    )
    .unwrap();

    let samples: HashMap<Uuid, String> =
        ids.iter().map(|id| (*id, "sampleA".to_string())).collect();
    let result = write_annotation(
        &path,
        &samples,
        &AnnotateOptions {
            name: "sample".to_string(),
            ..AnnotateOptions::default()
        },
    )
    .unwrap();
    assert_eq!(result.annotations_in_sidecar, 2);

    assert_eq!(read_annotation(&path, Some("barcode")).unwrap().len(), 10);
    let sample = read_annotation(&path, Some("sample")).unwrap();
    assert_eq!(sample.len(), N_READS);
    assert_eq!(sample.labels(), ["sampleA"]);

    // Ambiguous unnamed read must list what's available.
    let err = read_annotation(&path, None).unwrap_err().to_string();
    assert!(
        err.contains("barcode") && err.contains("sample"),
        "unexpected error: {err}"
    );
}

#[test]
fn unnamed_read_works_with_single_annotation() {
    let (_tmp, path, ids, _original) = fixture();
    write_annotation(
        &path,
        &make_assignments(&ids, 7),
        &AnnotateOptions::default(),
    )
    .unwrap();
    let annotation = read_annotation(&path, None).unwrap();
    assert_eq!(annotation.name(), "barcode");
    assert_eq!(annotation.len(), 7);
}

#[test]
fn rewriting_annotation_replaces_it() {
    let (_tmp, path, ids, _original) = fixture();
    write_annotation(
        &path,
        &make_assignments(&ids, 10),
        &AnnotateOptions::default(),
    )
    .unwrap();

    let second: HashMap<Uuid, String> = ids
        .iter()
        .take(3)
        .map(|id| (*id, "BC07".to_string()))
        .collect();
    let result = write_annotation(&path, &second, &AnnotateOptions::default()).unwrap();
    assert_eq!(result.annotations_in_sidecar, 1, "same name must replace");

    let annotation = read_annotation(&path, Some("barcode")).unwrap();
    assert_eq!(annotation.len(), 3);
    assert_eq!(annotation.labels(), ["BC07"]);
}

#[test]
fn stale_sidecar_is_rejected_until_forced() {
    let (_tmp, path, ids, _original) = fixture();
    write_annotation(
        &path,
        &make_assignments(&ids, 5),
        &AnnotateOptions::default(),
    )
    .unwrap();

    // Replace the POD5 with a different file (new identifier + size); the
    // sidecar next to it is now stale.
    let fx = write_fixture(&path, "replacement_acq", N_READS + 3, 500);

    let err = read_annotation(&path, Some("barcode"))
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("does not match"),
        "staleness guard did not fire: {err}"
    );

    let new_assignments = make_assignments(&fx.read_ids, 4);
    let err = write_annotation(&path, &new_assignments, &AnnotateOptions::default())
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("does not match"),
        "annotate over stale sidecar must fail without overwrite: {err}"
    );

    // overwrite=true replaces the stale sidecar with a fresh one.
    let result = write_annotation(
        &path,
        &new_assignments,
        &AnnotateOptions {
            overwrite: true,
            ..AnnotateOptions::default()
        },
    )
    .unwrap();
    assert_eq!(result.total_reads, N_READS + 3);
    assert_eq!(read_annotation(&path, Some("barcode")).unwrap().len(), 4);
}

#[test]
fn sidecar_copied_to_another_file_is_rejected() {
    let (tmp, path_a, ids, _original) = fixture();
    let path_b = tmp.path().join("other.pod5");
    write_fixture(&path_b, "other_acq", N_READS, 700);

    write_annotation(
        &path_a,
        &make_assignments(&ids, 5),
        &AnnotateOptions::default(),
    )
    .unwrap();
    std::fs::copy(sidecar_path(&path_a), sidecar_path(&path_b)).unwrap();

    let err = read_annotation(&path_b, Some("barcode"))
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("does not match"),
        "identity guard did not fire: {err}"
    );
}

#[test]
fn missing_sidecar_and_missing_annotation_error_clearly() {
    let (_tmp, path, ids, _original) = fixture();

    let err = read_annotation(&path, Some("barcode"))
        .unwrap_err()
        .to_string();
    assert!(err.contains("no sidecar"), "unexpected error: {err}");

    write_annotation(
        &path,
        &make_assignments(&ids, 5),
        &AnnotateOptions::default(),
    )
    .unwrap();
    let err = read_annotation(&path, Some("sample"))
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("no annotation 'sample'") && err.contains("barcode"),
        "unexpected error: {err}"
    );
}

#[test]
fn reserved_names_and_empty_labels_rejected() {
    let (_tmp, path, ids, _original) = fixture();
    for reserved in ["read_id", "batch_idx", "row_idx", ""] {
        let err = write_annotation(
            &path,
            &make_assignments(&ids, 2),
            &AnnotateOptions {
                name: reserved.to_string(),
                ..AnnotateOptions::default()
            },
        )
        .unwrap_err()
        .to_string();
        assert!(
            err.contains("not a valid annotation name"),
            "'{reserved}' accepted: {err}"
        );
    }

    let err = AnnotationSection::from_pairs("barcode", [(ids[0], "")])
        .unwrap_err()
        .to_string();
    assert!(err.contains("empty label"), "unexpected error: {err}");
}

/// Annotate + write a design CSV mapping BC01→condA, BC02→condB.
fn setup_with_design(
    path: &std::path::Path,
    tmp: &TempDir,
    ids: &[Uuid],
    n_assigned: usize,
) -> std::path::PathBuf {
    write_annotation(
        path,
        &make_assignments(ids, n_assigned),
        &AnnotateOptions::default(),
    )
    .unwrap();
    let csv = tmp.path().join("design.csv");
    std::fs::write(&csv, "barcode,condition\nBC01,condA\nBC02,condB\n").unwrap();
    csv
}

#[test]
fn design_derives_condition_column() {
    let (tmp, path, ids, original) = fixture();
    let csv = setup_with_design(&path, &tmp, &ids, 10);

    let result = write_design(&path, &csv, &DesignOptions::default()).unwrap();
    assert_eq!(result.key_columns, ["barcode"]);
    assert_eq!(result.value_columns, ["condition"]);
    assert_eq!(result.design_rows, 2);
    assert_eq!(result.derived, [("condition".to_string(), 10)]);
    assert_eq!(
        std::fs::read(&path).unwrap(),
        original,
        "design write must never modify the POD5"
    );

    // Each read's condition follows its barcode through the design.
    let condition = read_annotation(&path, Some("condition")).unwrap();
    assert_eq!(condition.len(), 10);
    for (i, id) in ids.iter().take(10).enumerate() {
        let expected = if i % 2 == 0 { "condA" } else { "condB" };
        assert_eq!(condition.get(id), Some(expected), "read {i}");
    }
    for id in ids.iter().skip(10) {
        assert_eq!(condition.get(id), None, "unassigned reads get no condition");
    }

    // The design table itself round-trips.
    let design = read_design(&path).unwrap();
    assert_eq!(design.key_columns, ["barcode"]);
    assert_eq!(design.value_columns, ["condition"]);
    assert_eq!(design.rows.len(), 2);
}

#[test]
fn design_multi_key_combinations() {
    let (tmp, path, ids, _original) = fixture();
    // Two independent annotations: ldx alternates L1/L2, edx is E1 for the
    // first half of reads and E2 for the rest.
    let ldx: HashMap<Uuid, String> = ids
        .iter()
        .enumerate()
        .map(|(i, id)| (*id, format!("L{}", i % 2 + 1)))
        .collect();
    let edx: HashMap<Uuid, String> = ids
        .iter()
        .enumerate()
        .map(|(i, id)| (*id, format!("E{}", if i < ids.len() / 2 { 1 } else { 2 })))
        .collect();
    for (name, mapping) in [("ldx", &ldx), ("edx", &edx)] {
        write_annotation(
            &path,
            mapping,
            &AnnotateOptions {
                name: name.to_string(),
                ..AnnotateOptions::default()
            },
        )
        .unwrap();
    }

    // Only two of the four combinations are in the design.
    let csv = tmp.path().join("design.csv");
    std::fs::write(
        &csv,
        "ldx,edx,condition,replicate\nL1,E1,fresh,r1\nL2,E2,frozen,r2\n",
    )
    .unwrap();
    let result = write_design(&path, &csv, &DesignOptions::default()).unwrap();
    assert_eq!(result.key_columns, ["ldx", "edx"]);
    assert_eq!(result.value_columns, ["condition", "replicate"]);

    let condition = read_annotation(&path, Some("condition")).unwrap();
    let replicate = read_annotation(&path, Some("replicate")).unwrap();
    for (i, id) in ids.iter().enumerate() {
        let expected = match (i % 2, i < ids.len() / 2) {
            (0, true) => Some(("fresh", "r1")),   // L1+E1
            (1, false) => Some(("frozen", "r2")), // L2+E2
            _ => None,                            // L1+E2, L2+E1: not in design
        };
        assert_eq!(condition.get(id), expected.map(|(c, _)| c), "read {i}");
        assert_eq!(replicate.get(id), expected.map(|(_, r)| r), "read {i}");
    }
}

#[test]
fn design_rederives_when_key_annotation_rewritten() {
    let (tmp, path, ids, _original) = fixture();
    let csv = setup_with_design(&path, &tmp, &ids, 10);
    write_design(&path, &csv, &DesignOptions::default()).unwrap();
    assert_eq!(
        read_annotation(&path, Some("condition"))
            .unwrap()
            .get(&ids[0]),
        Some("condA")
    );

    // Rewrite the barcode annotation flipping read 0 to BC02 — its derived
    // condition must follow without touching the design.
    let mut flipped = make_assignments(&ids, 10);
    flipped.insert(ids[0], "BC02".to_string());
    write_annotation(&path, &flipped, &AnnotateOptions::default()).unwrap();

    let condition = read_annotation(&path, Some("condition")).unwrap();
    assert_eq!(
        condition.get(&ids[0]),
        Some("condB"),
        "stale derived column"
    );
    assert_eq!(
        condition.get(&ids[1]),
        Some("condB"),
        "unrelated read changed"
    );
}

#[test]
fn design_value_column_rejects_direct_writes() {
    let (tmp, path, ids, _original) = fixture();
    let csv = setup_with_design(&path, &tmp, &ids, 10);
    write_design(&path, &csv, &DesignOptions::default()).unwrap();

    let err = write_annotation(
        &path,
        &make_assignments(&ids, 3),
        &AnnotateOptions {
            name: "condition".to_string(),
            ..AnnotateOptions::default()
        },
    )
    .unwrap_err()
    .to_string();
    assert!(
        err.contains("derived from the experimental design"),
        "unexpected error: {err}"
    );
}

#[test]
fn design_validation_errors() {
    let (tmp, path, ids, _original) = fixture();
    write_annotation(
        &path,
        &make_assignments(&ids, 10),
        &AnnotateOptions::default(),
    )
    .unwrap();

    // Duplicate key combination.
    let csv = tmp.path().join("dup.csv");
    std::fs::write(&csv, "barcode,condition\nBC01,condA\nBC01,condB\n").unwrap();
    let err = write_design(&path, &csv, &DesignOptions::default())
        .unwrap_err()
        .to_string();
    assert!(err.contains("duplicate design key"), "got: {err}");

    // No column matches an annotation.
    let csv = tmp.path().join("nokey.csv");
    std::fs::write(&csv, "sample,condition\ns1,condA\n").unwrap();
    let err = write_design(&path, &csv, &DesignOptions::default())
        .unwrap_err()
        .to_string();
    assert!(err.contains("no design CSV column matches"), "got: {err}");

    // --keys names a column that references a missing annotation.
    let err = write_design(
        &path,
        &csv,
        &DesignOptions {
            keys: Some(vec!["sample".to_string()]),
            ..DesignOptions::default()
        },
    )
    .unwrap_err()
    .to_string();
    assert!(err.contains("'sample' is not an annotation"), "got: {err}");

    // Empty key cell.
    let csv = tmp.path().join("emptykey.csv");
    std::fs::write(&csv, "barcode,condition\n,condA\n").unwrap();
    let err = write_design(&path, &csv, &DesignOptions::default())
        .unwrap_err()
        .to_string();
    assert!(err.contains("empty key cell"), "got: {err}");
}

#[test]
fn design_survives_index_rebuild() {
    let (tmp, path, ids, _original) = fixture();
    let csv = setup_with_design(&path, &tmp, &ids, 10);
    write_design(&path, &csv, &DesignOptions::default()).unwrap();

    let reader = Reader::open(&path).unwrap();
    reader.build_and_write_index(sidecar_path(&path)).unwrap();

    assert_eq!(read_design(&path).unwrap().rows.len(), 2);
    assert_eq!(read_annotation(&path, Some("condition")).unwrap().len(), 10);
}

#[test]
fn index_via_sidecar_matches_scan() {
    let (tmp, path, ids, _original) = fixture();

    // Copy without sidecar → scan path; original with sidecar → sidecar path.
    let bare = tmp.path().join("bare.pod5");
    std::fs::copy(&path, &bare).unwrap();
    write_annotation(
        &path,
        &make_assignments(&ids, 5),
        &AnnotateOptions::default(),
    )
    .unwrap();

    let scan_reader = Reader::open(&bare).unwrap();
    let sidecar_reader = Reader::open(&path).unwrap();
    let scan_index = scan_reader.read_index().unwrap();
    let sidecar_index = sidecar_reader.read_index().unwrap();
    assert_eq!(scan_index.len(), sidecar_index.len());
    for id in &ids {
        assert_eq!(
            scan_index.get(id),
            sidecar_index.get(id),
            "sidecar index diverges from scan for {id}"
        );
    }
}
