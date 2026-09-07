// SPDX-License-Identifier: MIT

//! The two things escapepod-rs#334 asks the POD5 side of `classify` to change
//! — take the `.p5s` index when there is one, and read in storage order
//! rather than BAM order — must change *when* a read is read and nothing else.
//!
//! Both are invisible in the output by construction, which is exactly how they
//! went unnoticed: an unindexed, BAM-ordered run finishes successfully with
//! correct calls, ~60x slower. So the guard has to be the equality itself —
//! same reads selected, same probabilities, bit for bit, with and without a
//! sidecar.

use escapepod_classify::{
    ChargingBundle, Pod5Index, classify_reads, junction_positions, resolve_orientation, scan_bam,
};
use std::collections::HashSet;
use std::path::{Path, PathBuf};

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

/// A copy of the fixture POD5 in a scratch directory, so the sidecar this
/// test writes never lands in the repository tree.
fn copy_pod5(dir: &Path, name: &str) -> PathBuf {
    let dst = dir.join(name);
    std::fs::copy(fixtures().join("trna_reads.pod5"), &dst).unwrap();
    dst
}

struct Run {
    calls: Vec<escapepod_classify::ReadCall>,
    no_call_ids: Vec<uuid::Uuid>,
    no_signal: u64,
    ns_mismatch: u64,
}

/// The pipeline the `escpod classify` command runs, over one POD5 path.
fn classify(pod5_path: &Path) -> (Run, Pod5Index, escapepod_classify::BamScan) {
    let bundle = ChargingBundle::load(&fixtures().join("bundle")).unwrap();
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
    let pod5 = Pod5Index::build(&[pod5_path.to_path_buf()], &wanted).unwrap();
    let (calls, stats) = classify_reads(&bundle, &scan.anchored, &pod5, orientation).unwrap();
    let run = Run {
        calls,
        no_call_ids: stats.no_calls.iter().map(|n| n.read_id).collect(),
        no_signal: stats.no_signal,
        ns_mismatch: stats.ns_mismatch,
    };
    (run, pod5, scan)
}

/// A `.p5s` changes where the read index comes from, never what it says. Both
/// arms are the same POD5 bytes, so any difference is the lookup's own.
#[test]
fn a_sidecar_does_not_change_a_single_call() {
    let tmp = tempfile::tempdir().unwrap();
    let bare = copy_pod5(tmp.path(), "bare.pod5");
    let indexed = copy_pod5(tmp.path(), "indexed.pod5");

    let sidecar = escapepod_signal::pod5::sidecar::sidecar_path(&indexed);
    let written = escapepod_signal::Reader::open(&indexed)
        .unwrap()
        .build_and_write_index(&sidecar)
        .unwrap();
    assert!(written > 0, "the fixture indexed to nothing");
    assert!(
        escapepod_signal::Reader::open(&indexed)
            .unwrap()
            .has_sidecar_index()
    );
    assert!(
        !escapepod_signal::Reader::open(&bare)
            .unwrap()
            .has_sidecar_index()
    );

    let (from_scan, _, _) = classify(&bare);
    let (from_sidecar, _, _) = classify(&indexed);

    assert_eq!(from_scan.no_signal, from_sidecar.no_signal);
    assert_eq!(from_scan.ns_mismatch, from_sidecar.ns_mismatch);
    assert_eq!(from_scan.no_call_ids, from_sidecar.no_call_ids);
    assert_eq!(
        from_scan.calls.len(),
        from_sidecar.calls.len(),
        "the two arms scored different numbers of reads"
    );
    assert!(!from_scan.calls.is_empty(), "the fixture scored nothing");
    for (a, b) in from_scan.calls.iter().zip(&from_sidecar.calls) {
        assert_eq!(a.read_id, b.read_id, "call order diverged");
        assert_eq!(
            a.p.to_bits(),
            b.p.to_bits(),
            "read {}: P differs between the scanned and sidecar index",
            a.read_id
        );
        assert_eq!(a.cl, b.cl, "read {}: cl byte differs", a.read_id);
    }
}

/// The tripwire: `Pod5Index::build` must resolve reads **through the read
/// index**, never by scanning the reads table and keeping the rows it wanted.
///
/// A scan and an index return the same reads, which is precisely why a scan
/// survived fourteen slow production runs unnoticed — no assertion about the
/// *output* can tell them apart. So this makes them disagree: it forges a
/// sidecar that passes the identity gate and whose locators are all rotated
/// one entry along. An indexed lookup follows a locator, lands on a row
/// holding another read, and refuses. A scan never looks at a locator, so it
/// would sail through — and this test would fail, which is the whole point.
///
/// If you are here because this test broke: you have replaced an index lookup
/// with a table scan. That is the escapepod-rs#334 regression, and it is
/// invisible in every other test in this repository.
#[test]
fn pod5_index_resolves_reads_through_the_index_not_a_scan() {
    use escapepod_signal::pod5::sidecar::{
        Sidecar, read_sidecar_file, sidecar_path, write_sidecar_file,
    };

    let tmp = tempfile::tempdir().unwrap();
    let pod5 = copy_pod5(tmp.path(), "forged.pod5");
    let p5s = sidecar_path(&pod5);

    let reader = escapepod_signal::Reader::open(&pod5).unwrap();
    let identity = reader.sidecar_identity().unwrap();
    reader.build_and_write_index(&p5s).unwrap();

    // Rotate every locator by one. The UUIDs stay put and stay sorted, so the
    // sidecar loads; each now points at the row of its neighbour.
    let existing = read_sidecar_file(&p5s, &identity).unwrap().unwrap();
    let mut entries = existing.entries().to_vec();
    assert!(entries.len() > 1, "fixture cannot be rotated");
    let locators: Vec<(u32, u32)> = entries.iter().map(|e| (e.1, e.2)).collect();
    for (i, entry) in entries.iter_mut().enumerate() {
        let (batch, row) = locators[(i + 1) % locators.len()];
        entry.1 = batch;
        entry.2 = row;
    }
    write_sidecar_file(&p5s, &identity, &Sidecar::new(entries)).unwrap();

    let bundle = ChargingBundle::load(&fixtures().join("bundle")).unwrap();
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
    let wanted: HashSet<uuid::Uuid> = scan.anchored.keys().copied().collect();

    let err = match Pod5Index::build(std::slice::from_ref(&pod5), &wanted) {
        Ok(_) => panic!("a rotated read index must be refused, not scanned around"),
        Err(e) => e.to_string(),
    };
    assert!(
        err.contains("read index for") && err.contains("which holds"),
        "the build failed for some other reason: {err}"
    );
}

/// Storage order is an order, not a filter: sorting the selection on
/// [`Pod5Index::storage_key`] must leave exactly the same reads in it, and
/// leave them monotone in file position.
#[test]
fn storage_order_is_forward_and_keeps_every_read() {
    let tmp = tempfile::tempdir().unwrap();
    let bare = copy_pod5(tmp.path(), "bare.pod5");
    let (_run, pod5, scan) = classify(&bare);

    // As `classify_reads` sees them: `HashMap` order, which is the BAM's
    // filtered through a hash.
    let mut reads: Vec<uuid::Uuid> = scan.anchored.keys().copied().collect();
    let before: HashSet<uuid::Uuid> = reads.iter().copied().collect();
    reads.sort_by_cached_key(|id| pod5.storage_key(id));
    let after: HashSet<uuid::Uuid> = reads.iter().copied().collect();
    assert_eq!(before, after, "ordering dropped or invented a read");

    let keys: Vec<_> = reads.iter().map(|id| pod5.storage_key(id)).collect();
    assert!(
        keys.windows(2).all(|w| w[0] <= w[1]),
        "storage order is not monotone in (file, first signal row)"
    );
    // The fixture holds reads the BAM anchors but the POD5 does not (renamed
    // UUIDs), so both halves of the key are exercised: `None` sorts first and
    // costs nothing to visit.
    assert!(
        keys.first().is_some_and(|k| k.is_none()),
        "fixture no longer exercises the unindexed-read case"
    );
    assert!(keys.last().is_some_and(|k| k.is_some()));
}
