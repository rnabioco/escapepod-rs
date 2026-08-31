//! Integration tests for the **collection** `.p5s`: one sidecar covering a
//! directory of POD5 files.
//!
//! What matters here is not that a file round-trips — it is that the two
//! sidecar shapes can never be mistaken for one another. A collection has no
//! single POD5 to bind to, so the per-file identity gate cannot protect it;
//! what protects it instead is that each reader recognises the other's file
//! and says so. Both directions are tested.

mod common;

use std::collections::HashMap;
use std::path::PathBuf;

use escapepod_pod5::operations::{
    ColumnValues, ColumnWrite, read_annotation, read_columns, read_score, write_collection_columns,
};
use escapepod_pod5::sidecar::{
    P5S_VERSION_COLLECTION, collection_sidecar_path, read_collection_file, read_sidecar_file,
    sidecar_path,
};
use escapepod_pod5::{Reader, Uuid};
use tempfile::TempDir;

use common::write_fixture;

const N_READS: usize = 12;
const N_FILES: usize = 3;

/// A directory of `N_FILES` POD5s, and every read id in them, in file order.
fn fixture_dir() -> (TempDir, PathBuf, Vec<Vec<Uuid>>) {
    let tmp = TempDir::new().expect("tempdir");
    let dir = tmp.path().join("pod5");
    std::fs::create_dir(&dir).expect("mkdir");
    let ids = (0..N_FILES)
        .map(|i| {
            let path = dir.join(format!("reads_{i}.pod5"));
            write_fixture(&path, &format!("acq_{i}"), N_READS, 700).read_ids
        })
        .collect();
    (tmp, dir, ids)
}

fn members(dir: &std::path::Path) -> Vec<PathBuf> {
    (0..N_FILES)
        .map(|i| dir.join(format!("reads_{i}.pod5")))
        .collect()
}

/// Alternate BC01/BC02 across every read in every file, plus a score column.
fn columns(ids: &[Vec<Uuid>]) -> Vec<ColumnWrite> {
    let mut labels: HashMap<Uuid, String> = HashMap::new();
    let mut scores: HashMap<Uuid, f32> = HashMap::new();
    for (f, file_ids) in ids.iter().enumerate() {
        for (i, id) in file_ids.iter().enumerate() {
            labels.insert(*id, format!("BC{:02}", i % 2 + 1));
            scores.insert(*id, f as f32 + i as f32 / 100.0);
        }
    }
    vec![
        ColumnWrite {
            name: "barcode".to_string(),
            values: ColumnValues::Labels(labels),
        },
        ColumnWrite {
            name: "crf_logp".to_string(),
            values: ColumnValues::Scores(scores),
        },
    ]
}

#[test]
fn one_collection_covers_every_file_in_the_directory() {
    let (_tmp, dir, ids) = fixture_dir();
    let p5s = collection_sidecar_path(&dir);

    let result = write_collection_columns(&p5s, &members(&dir), &columns(&ids), false).unwrap();
    assert_eq!(result.members, N_FILES);
    assert_eq!(result.total_reads, N_FILES * N_READS);
    assert_eq!(result.columns_in_sidecar, 2);

    // Beside the directory, not inside it, and not colliding with any
    // member's own sidecar.
    assert_eq!(p5s, dir.with_extension("p5s"));
    assert!(p5s.exists());
    assert_eq!(
        std::fs::read_dir(&dir).unwrap().count(),
        N_FILES,
        "the collection must not land among the POD5 files"
    );

    let collection = read_collection_file(&p5s).unwrap().unwrap();
    assert_eq!(collection.len(), N_FILES * N_READS);
    assert_eq!(collection.members().len(), N_FILES);
    for (i, member) in collection.members().iter().enumerate() {
        assert_eq!(member.name, format!("reads_{i}.pod5"));
        assert_eq!(member.reads, N_READS as u64);
    }

    // Every read resolves to the file it actually came from, at a locator
    // that file's own reader agrees with.
    let barcode = collection.annotation("barcode").expect("barcode column");
    let logp = collection.score("crf_logp").expect("score column");
    for (f, file_ids) in ids.iter().enumerate() {
        for (i, id) in file_ids.iter().enumerate() {
            let (member, _, _) = collection.locate(id).expect("read is indexed");
            assert_eq!(member.name, format!("reads_{f}.pod5"));
            assert_eq!(barcode.get(id).unwrap(), format!("BC{:02}", i % 2 + 1));
            assert_eq!(logp.get(id).unwrap(), f as f32 + i as f32 / 100.0);
        }
    }
}

#[test]
fn the_two_sidecar_shapes_refuse_each_other() {
    let (_tmp, dir, ids) = fixture_dir();
    let collection_p5s = collection_sidecar_path(&dir);
    write_collection_columns(&collection_p5s, &members(&dir), &columns(&ids), false).unwrap();

    // A per-file reader pointed at a collection must name the shape, not
    // report a version it cannot read or a column it cannot find: this build
    // *does* read the file, just not through that reader.
    let member = members(&dir).into_iter().next().unwrap();
    let identity = Reader::open(&member).unwrap().sidecar_identity().unwrap();
    let err = read_sidecar_file(&collection_p5s, &identity).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("collection sidecar"),
        "per-file reader must name the shape, got: {msg}"
    );

    // And the mirror: a collection reader pointed at a per-file sidecar. The
    // members key is what distinguishes them, so this is not merely a version
    // check — a v1 sidecar has no member table at all.
    escapepod_pod5::operations::write_columns(&member, &columns(&ids), false).unwrap();
    let err = read_collection_file(sidecar_path(&member)).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("per-file sidecar"),
        "collection reader must name the shape, got: {msg}"
    );
}

#[test]
fn a_collection_declares_version_3() {
    let (_tmp, dir, ids) = fixture_dir();
    let p5s = collection_sidecar_path(&dir);
    write_collection_columns(&p5s, &members(&dir), &columns(&ids), false).unwrap();

    // Read the version straight out of the Arrow footer rather than through
    // our own reader: the point of the bump is what an *older* escpod sees.
    let file = std::fs::File::open(&p5s).unwrap();
    let reader = arrow::ipc::reader::FileReader::try_new(file, None).unwrap();
    let metadata = reader.schema().metadata().clone();
    assert_eq!(
        metadata.get("escapepod:p5s_version").map(String::as_str),
        Some(P5S_VERSION_COLLECTION)
    );
    assert!(metadata.contains_key("escapepod:members"));
    // Bound to no single POD5 — the member table replaces those keys, and
    // leaving them out is what makes an old escpod's failure unambiguous.
    assert!(!metadata.contains_key("escapepod:file_identifier"));
}

#[test]
fn rewriting_a_collection_keeps_columns_it_was_not_given() {
    let (_tmp, dir, ids) = fixture_dir();
    let p5s = collection_sidecar_path(&dir);
    write_collection_columns(&p5s, &members(&dir), &columns(&ids), false).unwrap();

    // A second run writing only one column must merge, not replace: the
    // barcodes of a run that took hours live here and nowhere else.
    let only_scores = vec![columns(&ids).pop().unwrap()];
    let result = write_collection_columns(&p5s, &members(&dir), &only_scores, false).unwrap();
    assert_eq!(result.columns_in_sidecar, 2, "barcode column must survive");

    let collection = read_collection_file(&p5s).unwrap().unwrap();
    assert_eq!(collection.annotations().len(), 1);
    assert_eq!(collection.scores().len(), 1);
    assert_eq!(
        collection.annotation("barcode").unwrap().len(),
        N_FILES * N_READS
    );
}

#[test]
fn a_dropped_member_takes_its_reads_out_of_every_column() {
    let (_tmp, dir, ids) = fixture_dir();
    let p5s = collection_sidecar_path(&dir);
    write_collection_columns(&p5s, &members(&dir), &columns(&ids), false).unwrap();

    // Re-run over a subset. The file set is authoritative — a read no longer
    // indexed must not linger in a column, or the collection would keep
    // reporting barcodes for files it no longer covers.
    let fewer: Vec<PathBuf> = members(&dir).into_iter().take(1).collect();
    let result = write_collection_columns(&p5s, &fewer, &columns(&ids), false).unwrap();
    assert_eq!(result.members, 1);
    assert_eq!(result.total_reads, N_READS);

    let collection = read_collection_file(&p5s).unwrap().unwrap();
    assert_eq!(collection.len(), N_READS);
    assert_eq!(collection.annotation("barcode").unwrap().len(), N_READS);
    for id in &ids[1] {
        assert!(collection.locate(id).is_none(), "dropped member's read");
        assert!(collection.annotation("barcode").unwrap().get(id).is_none());
    }
}

#[test]
fn the_collection_path_does_not_depend_on_a_trailing_slash() {
    let (_tmp, dir, _) = fixture_dir();
    let with_slash = PathBuf::from(format!("{}/", dir.display()));
    assert_eq!(
        collection_sidecar_path(&with_slash),
        collection_sidecar_path(&dir),
        "a trailing separator would otherwise put the collection *inside* \
         the directory, as the hidden file .p5s"
    );
    // And in particular it is not that hidden file.
    assert!(!collection_sidecar_path(&with_slash).ends_with(".p5s/.p5s"));
    assert_ne!(collection_sidecar_path(&dir), dir.join(".p5s"));
}

#[test]
fn a_member_sidecar_and_the_collection_agree_on_where_a_read_is() {
    let (_tmp, dir, ids) = fixture_dir();
    let p5s = collection_sidecar_path(&dir);
    // Deliberately with no per-file sidecars on disk: the assembly must fall
    // back to scanning each member's reads table and reach the same answer.
    write_collection_columns(&p5s, &members(&dir), &columns(&ids), false).unwrap();
    let collection = read_collection_file(&p5s).unwrap().unwrap();

    for (f, member_path) in members(&dir).iter().enumerate() {
        let reader = Reader::open(member_path).unwrap();
        let index = reader.read_index().unwrap();
        for id in &ids[f] {
            let (member, batch, row) = collection.locate(id).expect("indexed");
            assert_eq!(member.name, format!("reads_{f}.pod5"));
            let (b, r) = index.get(id).expect("member index has it");
            assert_eq!(
                (batch as usize, row as usize),
                (b, r),
                "locators must agree"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Reading a collection through the per-POD5 surface
//
// The collection is only worth writing if the tools that consume a sidecar can
// read it without being told one exists. `demux split --sidecar`, `filter
// --annotation`, `view --include` and the Python `Reader` all go through
// `read_annotation` / `read_columns` / `read_score`, so these are the tests
// that say those commands work on a demultiplexed directory.
// ---------------------------------------------------------------------------

#[test]
fn a_member_with_no_sidecar_of_its_own_reads_from_the_collection() {
    let (_tmp, dir, ids) = fixture_dir();
    write_collection_columns(
        collection_sidecar_path(&dir),
        &members(&dir),
        &columns(&ids),
        false,
    )
    .unwrap();

    for (f, member_path) in members(&dir).iter().enumerate() {
        assert!(
            !sidecar_path(member_path).exists(),
            "the point of a collection is that this file does not exist"
        );

        let barcode = read_annotation(member_path, Some("barcode")).unwrap();
        // Restricted to this member. A column carried across whole would make
        // `escpod view` on one POD5 report labels for reads it does not hold.
        assert_eq!(barcode.len(), N_READS);
        for (i, id) in ids[f].iter().enumerate() {
            assert_eq!(barcode.get(id).unwrap(), format!("BC{:02}", i % 2 + 1));
        }
        for other in ids.iter().enumerate().filter(|&(g, _)| g != f) {
            for id in other.1 {
                assert!(barcode.get(id).is_none(), "another member's read");
            }
        }

        let logp = read_score(member_path, "crf_logp").unwrap();
        assert_eq!(logp.len(), N_READS);
        assert_eq!(logp.get(&ids[f][0]).unwrap(), f as f32);
    }
}

#[test]
fn a_member_sidecar_and_the_collection_are_merged_column_by_column() {
    let (_tmp, dir, ids) = fixture_dir();
    let member = members(&dir).into_iter().next().unwrap();

    // The file's own sidecar carries one column; the directory's collection
    // carries the other two. This is what `escpod index` (index only) beside a
    // demultiplexed directory looks like, and neither may hide the other.
    let own = vec![ColumnWrite {
        name: "flowcell".to_string(),
        values: ColumnValues::Labels(
            ids[0]
                .iter()
                .map(|id| (*id, "FAX00001".to_string()))
                .collect(),
        ),
    }];
    escapepod_pod5::operations::write_columns(&member, &own, false).unwrap();
    write_collection_columns(
        collection_sidecar_path(&dir),
        &members(&dir),
        &columns(&ids),
        false,
    )
    .unwrap();

    let got = read_columns(&member, &["flowcell", "barcode", "crf_logp"]).unwrap();
    let names: Vec<&str> = got.iter().map(|c| c.name()).collect();
    assert_eq!(names, ["flowcell", "barcode", "crf_logp"]);
}

#[test]
fn a_collection_never_answers_for_a_pod5_it_does_not_cover() {
    let (_tmp, dir, ids) = fixture_dir();
    write_collection_columns(
        collection_sidecar_path(&dir),
        &members(&dir),
        &columns(&ids),
        false,
    )
    .unwrap();

    // A POD5 that arrived in the directory after the collection was written.
    // It is found by the same directory rule and it is not in the member
    // table, so it must get nothing — the member `file_id` + `size` are the
    // whole identity gate a collection has, standing in for the file-level
    // binding a per-file sidecar carries.
    let newcomer = dir.join("reads_later.pod5");
    let later_ids = write_fixture(&newcomer, "acq_later", N_READS, 700).read_ids;
    let err = read_annotation(&newcomer, Some("barcode")).unwrap_err();
    assert!(
        err.to_string().contains("no sidecar"),
        "unexpected error: {err}"
    );
    // And in particular it did not inherit a neighbour's labels.
    for id in &later_ids {
        assert!(
            read_collection_file(collection_sidecar_path(&dir))
                .unwrap()
                .unwrap()
                .locate(id)
                .is_none()
        );
    }
}
