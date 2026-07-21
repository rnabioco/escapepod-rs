//! Atomicity of POD5 output: a write that does not complete must leave the
//! destination untouched, and must not litter staging files.

mod common;

use common::{make_read, make_run_info, synth_signal};
use escapepod_pod5::operations::{FilterOptions, subset_files};
use escapepod_pod5::{MergeOptions, Reader, Writer, WriterOptions, merge_files};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Staging files are named with a fixed prefix precisely so strays are easy to
/// spot; every test asserts none survive.
fn assert_no_strays(dir: &Path) {
    let strays: Vec<_> = fs::read_dir(dir)
        .unwrap()
        .filter_map(|e| {
            let name = e.ok()?.file_name().to_string_lossy().into_owned();
            name.starts_with(".escpod-tmp-").then_some(name)
        })
        .collect();
    assert!(strays.is_empty(), "staging files left behind: {strays:?}");
}

/// Write a valid POD5 with `n` reads and return their IDs.
fn write_valid(path: &Path, acq: &str, n: usize) -> Vec<escapepod_pod5::Uuid> {
    let mut writer = Writer::create(path, WriterOptions::default()).unwrap();
    writer.add_run_info(make_run_info(acq)).unwrap();
    let mut ids = Vec::new();
    for i in 0..n {
        let read = make_read(0, i as u32 + 1, 200);
        ids.push(read.read_id);
        writer.add_read(read, &synth_signal(200, i as u64)).unwrap();
    }
    writer.finish().unwrap();
    ids
}

#[test]
fn dropping_a_writer_creates_no_destination() {
    let tmp = tempfile::tempdir().unwrap();
    let dest = tmp.path().join("out.pod5");

    {
        let mut writer = Writer::create(&dest, WriterOptions::default()).unwrap();
        writer.add_run_info(make_run_info("acq_drop")).unwrap();
        writer
            .add_read(make_read(0, 1, 200), &synth_signal(200, 0))
            .unwrap();
        // Dropped here without finish().
    }

    assert!(
        !dest.exists(),
        "an unfinished writer must not leave a file at the destination"
    );
    assert_no_strays(tmp.path());
}

/// The data-loss regression: an interrupted overwrite must not damage the
/// archive that was already there.
#[test]
fn an_unfinished_overwrite_preserves_the_existing_archive() {
    let tmp = tempfile::tempdir().unwrap();
    let dest = tmp.path().join("out.pod5");

    let original_ids = write_valid(&dest, "acq_original", 5);
    let original_bytes = fs::read(&dest).unwrap();

    {
        let mut writer = Writer::create(&dest, WriterOptions::default()).unwrap();
        writer
            .add_run_info(make_run_info("acq_replacement"))
            .unwrap();
        for i in 0..3u32 {
            writer
                .add_read(make_read(0, i + 1, 200), &synth_signal(200, i as u64))
                .unwrap();
        }
        // Dropped without finish() — the replacement never lands.
    }

    assert_eq!(
        fs::read(&dest).unwrap(),
        original_bytes,
        "the original archive was modified"
    );

    let reader = Reader::open(&dest).unwrap();
    assert_eq!(reader.read_count().unwrap(), original_ids.len());
    assert_no_strays(tmp.path());
}

#[test]
fn abort_discards_the_output() {
    let tmp = tempfile::tempdir().unwrap();
    let dest = tmp.path().join("out.pod5");

    let mut writer = Writer::create(&dest, WriterOptions::default()).unwrap();
    writer.add_run_info(make_run_info("acq_abort")).unwrap();
    writer
        .add_read(make_read(0, 1, 200), &synth_signal(200, 0))
        .unwrap();
    writer.abort().unwrap();

    assert!(!dest.exists());
    assert_no_strays(tmp.path());
}

/// Reaching a deterministic error mid-write must not leave a stump behind.
/// A predefined pore-type dictionary rejects any value outside it.
#[test]
fn an_error_mid_write_leaves_nothing_at_the_destination() {
    use escapepod_pod5::PredefinedDictionaries;

    let tmp = tempfile::tempdir().unwrap();
    let dest = tmp.path().join("out.pod5");

    let options = WriterOptions {
        predefined_dictionaries: Some(PredefinedDictionaries {
            pore_types: Some(vec!["known_pore".to_string()]),
            end_reasons: None,
        }),
        ..Default::default()
    };

    let mut writer = Writer::create(&dest, options).unwrap();
    writer.add_run_info(make_run_info("acq_err")).unwrap();

    let mut read = make_read(0, 1, 200);
    read.pore_type = "an_unknown_pore".into();
    let err = writer.add_read(read, &synth_signal(200, 0));
    assert!(err.is_err(), "expected the dictionary check to reject this");
    drop(writer);

    assert!(!dest.exists());
    assert_no_strays(tmp.path());
}

/// Merging over one of the inputs used to truncate a live mmap and SIGBUS.
/// Staging makes it safe: the inputs stay on their original inode and only a
/// directory entry is swapped at the end.
#[test]
fn merge_in_place_over_an_input_is_safe() {
    let tmp = tempfile::tempdir().unwrap();
    let a = tmp.path().join("a.pod5");
    let b = tmp.path().join("b.pod5");

    let a_ids = write_valid(&a, "acq_a", 4);
    let b_ids = write_valid(&b, "acq_b", 6);

    // Output IS one of the inputs.
    let result = merge_files(&[a.clone(), b.clone()], &a, &MergeOptions::default(), None).unwrap();
    assert_eq!(result.reads_written, (a_ids.len() + b_ids.len()) as u64);

    let reader = Reader::open(&a).unwrap();
    assert_eq!(reader.read_count().unwrap(), a_ids.len() + b_ids.len());
    assert_no_strays(tmp.path());
}

/// One failing group must not take down the others, and must leave no file.
/// Pre-creating a *directory* where a group's file belongs makes the rename
/// fail deterministically for exactly that group.
#[test]
fn a_failing_subset_group_leaves_the_others_intact() {
    let tmp = tempfile::tempdir().unwrap();
    let input = tmp.path().join("in.pod5");
    let out_dir = tmp.path().join("out");
    fs::create_dir_all(&out_dir).unwrap();

    let ids = write_valid(&input, "acq_subset", 6);

    let mut read_to_group = HashMap::new();
    for (i, id) in ids.iter().enumerate() {
        let group = if i % 2 == 0 { "good.pod5" } else { "bad.pod5" };
        read_to_group.insert(*id, group.to_string());
    }

    // `bad.pod5` is a directory, so persisting over it fails.
    fs::create_dir(out_dir.join("bad.pod5")).unwrap();

    let outcome = subset_files(
        &[&input],
        &read_to_group,
        &out_dir,
        FilterOptions::default(),
    )
    .unwrap();

    assert_eq!(
        outcome.failures.len(),
        1,
        "expected exactly one failing group, got {:?}",
        outcome.failures
    );
    assert_eq!(outcome.failures[0].0, "bad.pod5");

    assert_eq!(outcome.groups.len(), 1, "the good group should still land");
    assert_eq!(outcome.groups[0].0, "good.pod5");

    let reader = Reader::open(out_dir.join("good.pod5")).unwrap();
    assert_eq!(reader.read_count().unwrap(), 3);

    assert_no_strays(&out_dir);
}
