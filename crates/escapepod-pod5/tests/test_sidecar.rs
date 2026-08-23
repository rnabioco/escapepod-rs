//! Integration tests for the `.p5s` sidecar: annotations and the read index
//! must coexist and preserve each other across writes, the POD5 file must
//! never change, and the identity binding must fail loudly on stale or
//! misplaced sidecars.

mod common;

use std::collections::{HashMap, HashSet};

use escapepod_pod5::operations::{
    AnnotateOptions, ColumnValues, ColumnWrite, DesignOptions, read_annotation, read_columns,
    read_design, read_score, remove_annotation, remove_design, write_annotation, write_columns,
    write_design,
};
use escapepod_pod5::sidecar::{AnnotationSection, P5S_VERSION, P5S_VERSION_SCORES, sidecar_path};
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
fn remove_annotation_and_design_guards() {
    let (tmp, path, ids, _original) = fixture();
    let csv = setup_with_design(&path, &tmp, &ids, 10);
    write_design(&path, &csv, &DesignOptions::default()).unwrap();
    write_annotation(
        &path,
        &ids.iter()
            .map(|id| (*id, "sampleA".to_string()))
            .collect::<HashMap<_, _>>(),
        &AnnotateOptions {
            name: "sample".to_string(),
            ..AnnotateOptions::default()
        },
    )
    .unwrap();

    // Design key and value columns are protected.
    let err = remove_annotation(&path, "barcode").unwrap_err().to_string();
    assert!(err.contains("design key column"), "got: {err}");
    let err = remove_annotation(&path, "condition")
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("derived from the experimental design"),
        "got: {err}"
    );

    // A free-standing annotation removes fine; absent names return false.
    assert!(remove_annotation(&path, "sample").unwrap());
    assert!(!remove_annotation(&path, "sample").unwrap());
    assert!(read_annotation(&path, Some("sample")).is_err());

    // Removing the design drops it AND its derived columns, then frees the
    // key column for removal too.
    assert!(remove_design(&path).unwrap());
    assert!(!remove_design(&path).unwrap());
    assert!(read_design(&path).is_err());
    assert!(read_annotation(&path, Some("condition")).is_err());
    assert!(remove_annotation(&path, "barcode").unwrap());
    assert!(
        read_annotation(&path, None)
            .unwrap_err()
            .to_string()
            .contains("no annotations")
    );
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

/// Assign a float to the first `n` reads.
fn make_scores(ids: &[Uuid], n: usize) -> HashMap<Uuid, f32> {
    ids.iter()
        .take(n)
        .enumerate()
        .map(|(i, id)| (*id, -0.25 * i as f32))
        .collect()
}

#[test]
fn scores_round_trip_alongside_labels() {
    let (_tmp, path, ids, original) = fixture();
    let mut scores = make_scores(&ids, 8);
    // A score for a read not in this file must be dropped, like a label.
    scores.insert(Uuid::new_v4(), 1.0);

    let result = write_columns(
        &path,
        &[
            ColumnWrite {
                name: "barcode".into(),
                values: ColumnValues::Labels(make_assignments(&ids, 10)),
            },
            ColumnWrite {
                name: "crf_logp".into(),
                values: ColumnValues::Scores(scores),
            },
        ],
        false,
    )
    .unwrap();
    assert_eq!(result.total_reads, N_READS);
    assert_eq!(result.columns.len(), 2);
    assert_eq!(result.columns[0].assigned, 10);
    assert_eq!(result.columns[0].labels, 2);
    assert_eq!(result.columns[1].assigned, 8, "foreign read must not count");
    assert_eq!(result.columns[1].labels, 0, "a score has no dictionary");
    assert_eq!(result.columns_in_sidecar, 2);

    // The POD5 is still byte-identical — the whole point of the sidecar.
    assert_eq!(std::fs::read(&path).unwrap(), original);

    let back = read_score(&path, "crf_logp").unwrap();
    assert_eq!(back.len(), 8);
    for (i, id) in ids.iter().take(8).enumerate() {
        assert_eq!(back.get(id), Some(-0.25 * i as f32));
    }
    // Reads past the eighth are unscored, which is absence rather than 0.0.
    assert_eq!(back.get(&ids[8]), None);
    // And the label column is untouched by the score column beside it.
    assert_eq!(read_annotation(&path, Some("barcode")).unwrap().len(), 10);
}

/// The version is gated on content: a barcode-only sidecar keeps declaring v1,
/// so an escpod that predates score columns can still read it. Only a file that
/// actually carries a numeric column declares v2.
#[test]
fn version_is_bumped_only_when_a_score_column_exists() {
    let (_tmp, path, ids, _original) = fixture();

    let version = |p: &std::path::Path| -> String {
        let file = std::fs::File::open(sidecar_path(p)).unwrap();
        let reader = arrow::ipc::reader::FileReader::try_new(file, None).unwrap();
        reader.schema().metadata()["escapepod:p5s_version"].clone()
    };

    write_annotation(
        &path,
        &make_assignments(&ids, 5),
        &AnnotateOptions::default(),
    )
    .unwrap();
    assert_eq!(version(&path), P5S_VERSION, "labels only");

    write_columns(
        &path,
        &[ColumnWrite {
            name: "crf_logp".into(),
            values: ColumnValues::Scores(make_scores(&ids, 5)),
        }],
        false,
    )
    .unwrap();
    assert_eq!(version(&path), P5S_VERSION_SCORES, "a score column exists");
}

/// `NaN` is the one float that already means "no value", so it is refused
/// rather than written — otherwise absence and NaN would be the same thing on
/// the way back in.
#[test]
fn nan_scores_are_refused() {
    let (_tmp, path, ids, _original) = fixture();
    let scores: HashMap<Uuid, f32> = [(ids[0], f32::NAN)].into_iter().collect();
    let err = write_columns(
        &path,
        &[ColumnWrite {
            name: "crf_logp".into(),
            values: ColumnValues::Scores(scores),
        }],
        false,
    )
    .unwrap_err()
    .to_string();
    assert!(err.contains("NaN"), "unexpected error: {err}");

    // Infinities are real answers, not missing ones, and do round-trip.
    let scores: HashMap<Uuid, f32> = [(ids[0], f32::NEG_INFINITY)].into_iter().collect();
    write_columns(
        &path,
        &[ColumnWrite {
            name: "crf_logp".into(),
            values: ColumnValues::Scores(scores),
        }],
        false,
    )
    .unwrap();
    assert_eq!(
        read_score(&path, "crf_logp").unwrap().get(&ids[0]),
        Some(f32::NEG_INFINITY)
    );
}

/// Asking for a score column that is really a label says so, rather than
/// reporting it missing next to a list that plainly contains it.
#[test]
fn reading_a_label_as_a_score_says_which_it_is() {
    let (_tmp, path, ids, _original) = fixture();
    write_annotation(
        &path,
        &make_assignments(&ids, 5),
        &AnnotateOptions::default(),
    )
    .unwrap();
    let err = read_score(&path, "barcode").unwrap_err().to_string();
    assert!(err.contains("is a label column"), "unexpected error: {err}");
}

/// Requesting no columns must not require a sidecar.
///
/// `escpod view` resolves its `--include` names into built-in read fields and
/// sidecar columns, and the overwhelmingly common case is that the sidecar list
/// is empty. Reading the sidecar before looking at the names turned every plain
/// `view` on a file without a `.p5s` into an error — caught by the POD5 compat
/// suite, not by anything in this file, which is why it is now here.
#[test]
fn reading_no_columns_needs_no_sidecar() {
    let (_tmp, path, _ids, _original) = fixture();
    assert!(!sidecar_path(&path).exists());
    let empty: [&str; 0] = [];
    assert!(read_columns(&path, &empty).unwrap().is_empty());

    // And a named column against a file with no sidecar is still an error.
    let err = read_columns(&path, &["barcode"]).unwrap_err().to_string();
    assert!(err.contains("no sidecar"), "unexpected error: {err}");
}

/// A name is one column, so a score and a label cannot both claim it: writing
/// one kind over the other replaces it rather than producing a sidecar that
/// cannot be written.
#[test]
fn a_score_and_a_label_cannot_share_a_name() {
    let (_tmp, path, ids, _original) = fixture();
    write_annotation(
        &path,
        &make_assignments(&ids, 5),
        &AnnotateOptions::default(),
    )
    .unwrap();

    write_columns(
        &path,
        &[ColumnWrite {
            name: "barcode".into(),
            values: ColumnValues::Scores(make_scores(&ids, 3)),
        }],
        false,
    )
    .unwrap();

    assert_eq!(read_score(&path, "barcode").unwrap().len(), 3);
    let err = read_annotation(&path, Some("barcode"))
        .unwrap_err()
        .to_string();
    assert!(err.contains("no annotation"), "unexpected error: {err}");
}

// ---------------------------------------------------------------------------
// Locator trust: identity says the sidecar belongs to this POD5, and nothing
// says its `(batch_idx, row_idx)` locators are right. Following one blind
// returns a *different real read, correctly self-labelled* — which no caller
// can detect — so every indexed lookup confirms the row it landed on.
//
// These forge a sidecar that passes the identity gate and has wrong locators,
// which is the only way to reach the check: a sidecar this crate wrote for
// this file is correct by construction, so a round-trip test is blind here.
// ---------------------------------------------------------------------------

/// Rewrite the sidecar's locators with `f`, keeping its identity valid.
fn forge_locators(path: &std::path::Path, f: impl Fn(&mut Vec<([u8; 16], u32, u32)>)) -> Vec<Uuid> {
    use escapepod_pod5::sidecar::{Sidecar, read_sidecar_file, write_sidecar_file};

    let reader = Reader::open(path).unwrap();
    let identity = reader.sidecar_identity().unwrap();
    let p5s = sidecar_path(path);
    let existing = read_sidecar_file(&p5s, &identity).unwrap().unwrap();

    let mut entries = existing.entries().to_vec();
    f(&mut entries);
    let ids: Vec<Uuid> = entries.iter().map(|e| Uuid::from_bytes(e.0)).collect();

    write_sidecar_file(&p5s, &identity, &Sidecar::new(entries)).unwrap();
    ids
}

/// Build a fixture whose sidecar exists, and return its first two read ids.
fn fixture_with_index() -> (TempDir, std::path::PathBuf, Vec<Uuid>) {
    let (tmp, path, ids, _) = fixture();
    Reader::open(&path)
        .unwrap()
        .build_and_write_index(sidecar_path(&path))
        .unwrap();
    (tmp, path, ids)
}

#[test]
fn indexed_reads_reject_a_locator_pointing_at_another_read() {
    let (_tmp, path, _ids) = fixture_with_index();
    // Swap two entries' locators: each UUID now points at the other's row.
    // Both rows exist, so only a read-ID confirmation can catch this.
    let ids = forge_locators(&path, |entries| {
        let (b0, r0) = (entries[0].1, entries[0].2);
        entries[0].1 = entries[1].1;
        entries[0].2 = entries[1].2;
        entries[1].1 = b0;
        entries[1].2 = r0;
    });

    let reader = Reader::open(&path).unwrap();
    let targets: HashSet<Uuid> = [ids[0]].into_iter().collect();

    let err = reader.reads_by_ids(&targets).unwrap_err().to_string();
    assert!(
        err.contains("read index for") && err.contains("which holds"),
        "swapped locator returned another read instead of erroring: {err}"
    );

    // Same guard on the signal path, which is worse without it: it labels the
    // signal with the *queried* UUID whatever row it came from.
    let err = reader
        .find_signal_rows_by_ids(&targets)
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("read index for") && err.contains("which holds"),
        "swapped locator returned another read's signal: {err}"
    );
}

#[test]
fn indexed_reads_reject_an_out_of_range_locator() {
    let (_tmp, path, _ids) = fixture_with_index();
    // Past the end of the batch. Without a bounds check this reaches an Arrow
    // accessor and panics rather than erroring.
    let ids = forge_locators(&path, |entries| {
        entries[0].2 = 10_000;
    });

    let reader = Reader::open(&path).unwrap();
    let targets: HashSet<Uuid> = [ids[0]].into_iter().collect();

    let err = reader.reads_by_ids(&targets).unwrap_err().to_string();
    assert!(
        err.contains("which has") && err.contains("rows"),
        "out-of-range row was not reported as an error: {err}"
    );

    let err = reader
        .find_signal_rows_by_ids(&targets)
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("which has") && err.contains("rows"),
        "out-of-range row was not reported on the signal path: {err}"
    );
}

#[test]
fn a_correct_sidecar_still_reads_through_the_confirmation() {
    // The guard must not cost correctness on the happy path. Both readers are
    // now indexed — the bare one builds its index by projected scan instead of
    // loading it — so this also pins that a built index and a sidecar-loaded
    // one resolve the same reads to the same rows.
    let (tmp, path, ids) = fixture_with_index();
    let bare = tmp.path().join("bare.pod5");
    std::fs::copy(&path, &bare).unwrap();

    let targets: HashSet<Uuid> = ids.iter().take(5).copied().collect();
    let indexed = Reader::open(&path).unwrap().reads_by_ids(&targets).unwrap();
    let scanned = Reader::open(&bare).unwrap().reads_by_ids(&targets).unwrap();

    assert_eq!(indexed.len(), 5);
    let mut a: Vec<Uuid> = indexed.iter().map(|r| r.read_id).collect();
    let mut b: Vec<Uuid> = scanned.iter().map(|r| r.read_id).collect();
    a.sort();
    b.sort();
    assert_eq!(a, b);
}

/// escapepod-rs#251. A lookup on a POD5 with no sidecar must **build** the
/// index it already knows how to build — once — and then seek, rather than
/// rescanning the whole reads table on every call.
///
/// The two strategies return identical results, so nothing observable in the
/// return value can tell them apart; what differs is the work done. This pins
/// it through the crate's own tracing output, which is also why that output
/// exists: without it the difference is a day of profiling.
#[test]
fn a_lookup_without_a_sidecar_builds_the_index_once() {
    use std::sync::{Arc, Mutex};

    struct Capture(Arc<Mutex<Vec<u8>>>);
    impl std::io::Write for Capture {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    let (_tmp, path, ids, _) = fixture();
    assert!(
        !sidecar_path(&path).exists(),
        "this test is only meaningful without a sidecar"
    );

    let logs = Arc::new(Mutex::new(Vec::new()));
    let sink = logs.clone();
    let subscriber = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .with_ansi(false)
        .with_writer(move || Capture(sink.clone()))
        .finish();

    let targets: HashSet<Uuid> = ids.iter().take(3).copied().collect();
    let (first, second) = tracing::subscriber::with_default(subscriber, || {
        let reader = Reader::open(&path).unwrap();
        // Two lookups on ONE reader: the second must ride the cached index.
        let a = reader.reads_by_ids(&targets).unwrap();
        let b = reader.reads_by_ids(&targets).unwrap();
        (a, b)
    });

    assert_eq!(first.len(), 3, "first lookup lost reads");
    assert_eq!(second.len(), 3, "second lookup lost reads");

    let text = String::from_utf8(logs.lock().unwrap().clone()).unwrap();
    assert_eq!(
        text.matches("read index built from a projected scan")
            .count(),
        1,
        "index must be built exactly once for two lookups on one reader:\n{text}"
    );
    assert!(
        text.contains("no .p5s sidecar"),
        "a build should say why it happened:\n{text}"
    );
}

// ---------------------------------------------------------------------------
// Provenance: descriptive, optional, never a gate.
// ---------------------------------------------------------------------------

#[test]
fn provenance_names_the_source_in_a_mismatch_error() {
    let (_tmp, path, ids, _) = fixture();
    write_annotation(
        &path,
        &make_assignments(&ids, 5),
        &AnnotateOptions::default(),
    )
    .unwrap();

    let sidecar = escapepod_pod5::sidecar::read_sidecar_file(
        sidecar_path(&path),
        &Reader::open(&path).unwrap().sidecar_identity().unwrap(),
    )
    .unwrap()
    .unwrap();
    let prov = sidecar.provenance();
    assert_eq!(prov.source_name.as_deref(), Some("reads.pod5"));
    assert_eq!(prov.read_count, Some(N_READS as u64));
    assert!(
        prov.writer
            .as_deref()
            .unwrap()
            .starts_with("escapepod-pod5 ")
    );

    // Replace the POD5; the mismatch error must now say what it came from.
    write_fixture(&path, "replacement_acq", N_READS + 3, 500);
    let err = read_annotation(&path, Some("barcode"))
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("does not match") && err.contains("reads.pod5") && err.contains("25 reads"),
        "mismatch error does not identify the sidecar's origin: {err}"
    );
}

#[test]
fn a_sidecar_without_provenance_keys_still_loads() {
    // The keys are additive, not a version bump: strip them and the sidecar
    // must still be accepted, with empty provenance and no error.
    use escapepod_pod5::sidecar::{
        P5S_READ_COUNT_KEY, P5S_SOURCE_NAME_KEY, P5S_WRITER_KEY, read_sidecar_file,
    };

    let (_tmp, path, ids, _) = fixture();
    write_annotation(
        &path,
        &make_assignments(&ids, 5),
        &AnnotateOptions::default(),
    )
    .unwrap();

    let p5s = sidecar_path(&path);
    let identity = Reader::open(&path).unwrap().sidecar_identity().unwrap();

    // Rewrite the IPC file with the provenance keys removed from the schema.
    let table = {
        let f = std::fs::File::open(&p5s).unwrap();
        let mut r = arrow::ipc::reader::FileReader::try_new(f, None).unwrap();
        let schema = r.schema();
        let batch = r.next().unwrap().unwrap();
        let mut md = schema.metadata().clone();
        md.remove(P5S_SOURCE_NAME_KEY);
        md.remove(P5S_READ_COUNT_KEY);
        md.remove(P5S_WRITER_KEY);
        let stripped = std::sync::Arc::new(
            arrow::datatypes::Schema::new(schema.fields().clone()).with_metadata(md),
        );
        (stripped, batch)
    };
    {
        let f = std::fs::File::create(&p5s).unwrap();
        let mut w = arrow::ipc::writer::FileWriter::try_new(f, &table.0).unwrap();
        w.write(&table.1).unwrap();
        w.finish().unwrap();
    }

    let sidecar = read_sidecar_file(&p5s, &identity).unwrap().unwrap();
    assert_eq!(sidecar.len(), N_READS, "stripped sidecar must still load");
    assert_eq!(sidecar.provenance().describe(), None);
    assert_eq!(read_annotation(&path, Some("barcode")).unwrap().len(), 5);
}
