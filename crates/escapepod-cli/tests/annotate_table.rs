// SPDX-License-Identifier: MIT

//! `escpod sidecar annotate --from-table` (#486): label reads in the `.p5s` sidecar
//! from another tool's read-to-label table.
//!
//! These run the real binary, because what matters is what lands in the
//! sidecar and what the rest of the toolchain then reads back out of it — and,
//! for the refusal test, the *absence* of a write, which no unit test of the
//! parser can see. The table is shaped like WarpDemuX's: gzipped, a
//! `#read_id` header, bare-integer labels including `-1`, and a share of reads
//! that are not in it at all.

#![cfg(feature = "cli")]

mod common;

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};

use common::{Workspace, assert_ok, run};
use flate2::Compression;
use flate2::write::GzEncoder;

/// Labels cycled through the labelled reads, WarpDemuX-style: bare integers,
/// one of them the `-1` it uses for "no call".
const LABELS: [&str; 5] = ["7", "4", "5", "-1", "3"];

/// How many of the fixture's 20 reads the table labels. The rest are absent.
const LABELLED: usize = 14;

fn pod5(ws: &Workspace) -> PathBuf {
    ws.dir.join("reads.pod5")
}

fn per_file_sidecar(ws: &Workspace) -> PathBuf {
    ws.dir.join("reads.pod5.p5s")
}

/// Where a directory argument's collection sidecar lands: beside the directory.
fn collection_sidecar(ws: &Workspace) -> PathBuf {
    ws.dir.with_extension("p5s")
}

fn s(path: &Path) -> &str {
    path.to_str().unwrap()
}

/// The fixture's read ids, from the binary rather than a hard-coded list.
fn read_ids(ws: &Workspace) -> Vec<String> {
    let out = assert_ok(&["view", s(&pod5(ws)), "--ids"]);
    let ids: Vec<String> = String::from_utf8(out.stdout)
        .unwrap()
        .lines()
        .map(str::to_string)
        .collect();
    assert_eq!(ids.len(), 20, "the fixture is expected to hold 20 reads");
    ids
}

/// The table's rows for the first [`LABELLED`] reads, without a header.
fn table_rows(ids: &[String], delimiter: char) -> Vec<String> {
    ids.iter()
        .take(LABELLED)
        .enumerate()
        .map(|(i, id)| {
            [id.as_str(), LABELS[i % LABELS.len()], "0.97", "0.01"].join(&delimiter.to_string())
        })
        .collect()
}

const HEADER: &str = "#read_id,predicted_barcode,confidence_score,p03";

/// What the sidecar should say per read once the table is applied with the
/// default fill.
fn expected(ids: &[String]) -> BTreeMap<String, String> {
    ids.iter()
        .enumerate()
        .map(|(i, id)| {
            let label = if i < LABELLED {
                LABELS[i % LABELS.len()]
            } else {
                "unclassified"
            };
            (id.clone(), label.to_string())
        })
        .collect()
}

fn gzip_members(members: &[String]) -> Vec<u8> {
    let mut out = Vec::new();
    for member in members {
        let mut enc = GzEncoder::new(&mut out, Compression::default());
        enc.write_all(member.as_bytes()).unwrap();
        enc.finish().unwrap();
    }
    out
}

fn write_gz_table(ws: &Workspace, ids: &[String]) -> PathBuf {
    let text = format!("{HEADER}\n{}\n", table_rows(ids, ',').join("\n"));
    let path = ws.out("calls.csv.gz");
    std::fs::write(&path, gzip_members(&[text])).unwrap();
    path
}

/// `view --include <column>` as read → label, `""` for an unlabelled read.
fn viewed(file: &Path, column: &str) -> BTreeMap<String, String> {
    let out = assert_ok(&["view", s(file), "--include", column]);
    String::from_utf8(out.stdout)
        .unwrap()
        .lines()
        .skip(1)
        .map(|line| {
            let (id, label) = line.split_once('\t').unwrap_or((line, ""));
            (id.to_string(), label.to_string())
        })
        .collect()
}

fn annotate_table(input: &Path, table: &Path, extra: &[&str]) {
    let mut args = vec![
        "sidecar",
        "annotate",
        s(input),
        "--from-table",
        s(table),
        "--column",
        "barcode=predicted_barcode",
    ];
    args.extend_from_slice(extra);
    assert_ok(&args);
}

#[test]
fn from_table_reads_warpdemux_shaped_gz() {
    let ws = Workspace::new();
    let ids = read_ids(&ws);
    let table = write_gz_table(&ws, &ids);

    annotate_table(&pod5(&ws), &table, &[]);

    let labels = viewed(&pod5(&ws), "barcode");
    assert_eq!(labels, expected(&ids));
    // The calls are carried verbatim: `-1` is a label here, not "unassigned",
    // and `4` is not renamed to `BC04`.
    assert!(labels.values().any(|l| l == "-1"), "{labels:?}");
    assert!(labels.values().any(|l| l == "4"), "{labels:?}");
    assert!(!labels.values().any(|l| l.starts_with("BC")), "{labels:?}");
}

/// A `.gz` table is not necessarily one gzip member — bgzipped output and
/// `cat a.gz b.gz` are several. `GzDecoder` stops after the first without an
/// error, and every call in the rest of the file becomes "not in the table",
/// which the fill then labels `unclassified`: wrong answers, no complaint.
#[test]
fn a_multi_member_gzip_is_read_whole() {
    let ws = Workspace::new();
    let ids = read_ids(&ws);
    let rows = table_rows(&ids, ',');
    let (first, second) = rows.split_at(LABELLED / 2);
    let members = [
        format!("{HEADER}\n{}\n", first.join("\n")),
        format!("{}\n", second.join("\n")),
    ];
    let table = ws.out("calls.csv.gz");
    std::fs::write(&table, gzip_members(&members)).unwrap();

    annotate_table(&pod5(&ws), &table, &[]);

    assert_eq!(
        viewed(&pod5(&ws), "barcode"),
        expected(&ids),
        "reads in the second gzip member were not labelled"
    );
}

#[test]
fn from_table_tsv_matches_csv() {
    let csv_ws = Workspace::new();
    let tsv_ws = Workspace::new();
    let ids = read_ids(&csv_ws);

    let csv = csv_ws.out("calls.csv");
    std::fs::write(
        &csv,
        format!("{HEADER}\n{}\n", table_rows(&ids, ',').join("\n")),
    )
    .unwrap();
    let tsv = tsv_ws.out("calls.tsv");
    std::fs::write(
        &tsv,
        format!(
            "{}\n{}\n",
            HEADER.replace(',', "\t"),
            table_rows(&ids, '\t').join("\n")
        ),
    )
    .unwrap();

    annotate_table(&pod5(&csv_ws), &csv, &[]);
    annotate_table(&pod5(&tsv_ws), &tsv, &[]);

    let from_csv = viewed(&pod5(&csv_ws), "barcode");
    assert_eq!(from_csv, viewed(&pod5(&tsv_ws), "barcode"));
    assert_eq!(from_csv, expected(&ids));
}

#[test]
fn absent_reads_are_labelled_unclassified_and_empty_label_disables_it() {
    let ws = Workspace::new();
    let ids = read_ids(&ws);
    // Read 0 is in the table with an empty cell; reads 1.. as usual, and the
    // reads past LABELLED are not in it at all.
    let mut rows = table_rows(&ids, ',');
    rows[0] = format!("{},,0.5,0.5", ids[0]);
    let table = ws.out("calls.csv");
    std::fs::write(&table, format!("{HEADER}\n{}\n", rows.join("\n"))).unwrap();

    // Default: both kinds of unlabelled read get `unclassified`.
    annotate_table(&pod5(&ws), &table, &[]);
    let labels = viewed(&pod5(&ws), "barcode");
    assert_eq!(labels[&ids[0]], "unclassified", "empty cell");
    assert_eq!(labels[&ids[LABELLED]], "unclassified", "absent from table");
    assert_eq!(labels[&ids[1]], LABELS[1]);
    assert!(
        labels.values().all(|l| !l.is_empty()),
        "every read is labelled with the fill: {labels:?}"
    );

    // `--unassigned-label ''` is today's `-a` behaviour: those reads are
    // absent from the column, replacing the filled one in place.
    annotate_table(&pod5(&ws), &table, &["--unassigned-label", ""]);
    let labels = viewed(&pod5(&ws), "barcode");
    assert_eq!(labels[&ids[0]], "", "empty cell stays unlabelled");
    assert_eq!(labels[&ids[LABELLED]], "", "absent read stays unlabelled");
    assert_eq!(labels[&ids[1]], LABELS[1]);
    assert!(!labels.values().any(|l| l == "unclassified"), "{labels:?}");

    // And any other word is taken as given.
    annotate_table(&pod5(&ws), &table, &["--unassigned-label", "none"]);
    let labels = viewed(&pod5(&ws), "barcode");
    assert_eq!(labels[&ids[0]], "none");
    assert_eq!(labels[&ids[LABELLED]], "none");
}

/// A directory gets one sidecar, as with `demux --annotate` — the same writer
/// serves both, so the behaviour is theirs and this only checks it is reached.
#[test]
fn from_table_on_a_directory_writes_one_collection_sidecar() {
    let ws = Workspace::new();
    let ids = read_ids(&ws);
    let table = write_gz_table(&ws, &ids);

    annotate_table(&ws.dir, &table, &[]);

    assert!(
        collection_sidecar(&ws).exists(),
        "no collection sidecar at {}",
        collection_sidecar(&ws).display()
    );
    assert!(
        !per_file_sidecar(&ws).exists(),
        "a directory input must not also write a per-file sidecar"
    );
    // Reading goes through the POD5, and finds the collection.
    assert_eq!(viewed(&pod5(&ws), "barcode"), expected(&ids));

    let out = assert_ok(&["sidecar", "annotate", "--list", ws.dir()]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("covered by") && !stdout.contains("no sidecar"),
        "--list must call the member covered:\n{stdout}"
    );
}

#[test]
fn from_table_replaces_the_named_column_and_keeps_the_rest() {
    let ws = Workspace::new();
    let ids = read_ids(&ws);

    // A second column, written by `-a` (which reads the `barcode` column of its
    // CSV whatever the sidecar column is called), that the table run must not
    // disturb.
    let samples = ws.out("samples.csv");
    let text: String = std::iter::once("read_id,barcode".to_string())
        .chain(ids.iter().map(|id| format!("{id},s1")))
        .collect::<Vec<_>>()
        .join("\n");
    std::fs::write(&samples, text + "\n").unwrap();
    assert_ok(&[
        "sidecar",
        "annotate",
        s(&pod5(&ws)),
        "-a",
        s(&samples),
        "--name",
        "sample",
    ]);

    let table = write_gz_table(&ws, &ids);
    annotate_table(&pod5(&ws), &table, &[]);
    assert_eq!(viewed(&pod5(&ws), "barcode"), expected(&ids));

    // A different table for the same column replaces it in place.
    let swapped = ws.out("swapped.csv");
    let rows: Vec<String> = ids.iter().map(|id| format!("{id},9,0.5,0.5")).collect();
    std::fs::write(&swapped, format!("{HEADER}\n{}\n", rows.join("\n"))).unwrap();
    annotate_table(&pod5(&ws), &swapped, &[]);

    let barcodes = viewed(&pod5(&ws), "barcode");
    assert!(
        barcodes.values().all(|l| l == "9"),
        "barcode was not replaced: {barcodes:?}"
    );
    let samples = viewed(&pod5(&ws), "sample");
    assert!(
        samples.values().all(|l| l == "s1") && samples.len() == 20,
        "the other column did not survive: {samples:?}"
    );

    let out = assert_ok(&["sidecar", "annotate", "--list", s(&pod5(&ws))]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("20 reads indexed")
            && stdout.contains("barcode: 1 labels")
            && stdout.contains("sample: 1 labels"),
        "{stdout}"
    );
}

/// The refusal has to come *before* the write. With a fill label every read
/// counts as assigned, so nothing the writers report can say "this table is
/// for another run" — and an in-place replace would overwrite a good barcode
/// column with all-`unclassified`.
#[test]
fn a_table_for_another_run_is_refused_before_anything_is_written() {
    let ws = Workspace::new();
    let ids = read_ids(&ws);

    let stranger = ws.out("other_run.csv");
    std::fs::write(
        &stranger,
        format!(
            "{HEADER}\n00000000-0000-4000-8000-000000000001,4,0.9,0.1\n\
             00000000-0000-4000-8000-000000000002,7,0.9,0.1\n"
        ),
    )
    .unwrap();

    // Nothing exists yet: refusing must not leave a sidecar behind either.
    let out = run(&[
        "sidecar",
        "annotate",
        s(&pod5(&ws)),
        "--from-table",
        s(&stranger),
        "--column",
        "barcode=predicted_barcode",
    ]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "wrong table accepted:\n{stderr}");
    assert!(
        stderr.contains("none of the 2 reads") && stderr.contains("Nothing was written"),
        "{stderr}"
    );
    assert!(!per_file_sidecar(&ws).exists(), "a sidecar was written");
    let out = run(&[
        "sidecar",
        "annotate",
        ws.dir(),
        "--from-table",
        s(&stranger),
        "--column",
        "barcode=predicted_barcode",
    ]);
    assert!(!out.status.success());
    assert!(
        !collection_sidecar(&ws).exists(),
        "a collection sidecar was written"
    );

    // And with a good column already in place, it must survive byte for byte.
    let table = write_gz_table(&ws, &ids);
    annotate_table(&pod5(&ws), &table, &[]);
    let before = std::fs::read(per_file_sidecar(&ws)).unwrap();
    let out = run(&[
        "sidecar",
        "annotate",
        s(&pod5(&ws)),
        "--from-table",
        s(&stranger),
        "--column",
        "barcode=predicted_barcode",
    ]);
    assert!(!out.status.success());
    assert_eq!(
        std::fs::read(per_file_sidecar(&ws)).unwrap(),
        before,
        "the existing sidecar was modified by a refused run"
    );
    assert_eq!(viewed(&pod5(&ws), "barcode"), expected(&ids));

    // `-a` has the same guard: it used to write an empty column over the old
    // one and only then report the mismatch.
    let wrong_csv = ws.out("wrong.csv");
    std::fs::write(
        &wrong_csv,
        "read_id,barcode\n00000000-0000-4000-8000-000000000001,BC01\n",
    )
    .unwrap();
    let out = run(&["sidecar", "annotate", s(&pod5(&ws)), "-a", s(&wrong_csv)]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "wrong CSV accepted:\n{stderr}");
    assert!(stderr.contains("Nothing was written"), "{stderr}");
    assert_eq!(
        std::fs::read(per_file_sidecar(&ws)).unwrap(),
        before,
        "`-a` modified the sidecar before refusing"
    );
}

/// With the fill off, `--from-table` is `-a` with a different front end: same
/// reads labelled, same bytes on disk.
#[test]
fn from_table_without_fill_matches_dash_a() {
    let table_ws = Workspace::new();
    let csv_ws = Workspace::new();
    let ids = read_ids(&table_ws);

    // A demux-classify-shaped CSV, which `-a` reads.
    let rows: Vec<String> = ids
        .iter()
        .take(LABELLED)
        .enumerate()
        .map(|(i, id)| format!("{id},BC0{}", i % 3 + 1))
        .collect();
    let csv = csv_ws.out("assign.csv");
    std::fs::write(&csv, format!("read_id,barcode\n{}\n", rows.join("\n"))).unwrap();
    let table = table_ws.out("assign.csv");
    std::fs::write(&table, format!("read_id,barcode\n{}\n", rows.join("\n"))).unwrap();

    assert_ok(&["sidecar", "annotate", s(&pod5(&csv_ws)), "-a", s(&csv)]);
    assert_ok(&[
        "sidecar",
        "annotate",
        s(&pod5(&table_ws)),
        "--from-table",
        s(&table),
        "--column",
        "barcode=barcode",
        "--unassigned-label",
        "",
    ]);

    let via_a = viewed(&pod5(&csv_ws), "barcode");
    assert_eq!(via_a, viewed(&pod5(&table_ws), "barcode"));
    assert_eq!(
        via_a.values().filter(|l| l.is_empty()).count(),
        20 - LABELLED
    );
    assert_eq!(
        std::fs::read(per_file_sidecar(&csv_ws)).unwrap(),
        std::fs::read(per_file_sidecar(&table_ws)).unwrap(),
        "the two routes wrote different sidecars"
    );
}

/// The point of the exercise: the sidecar a table produced drives the rest of
/// the toolchain, and the POD5 it describes is never touched.
#[test]
fn filter_annotation_and_split_sidecar_work_on_the_result() {
    let ws = Workspace::new();
    let ids = read_ids(&ws);
    let table = write_gz_table(&ws, &ids);
    let pod5_before = std::fs::read(pod5(&ws)).unwrap();

    annotate_table(&pod5(&ws), &table, &[]);

    // `filter --annotation` selects by the verbatim label.
    let sevens = ws.out("sevens.pod5");
    assert_ok(&[
        "filter",
        s(&pod5(&ws)),
        "--annotation",
        "barcode=7",
        "-o",
        s(&sevens),
    ]);
    let out = assert_ok(&["view", s(&sevens), "--ids"]);
    let mut got: Vec<String> = String::from_utf8(out.stdout)
        .unwrap()
        .lines()
        .map(str::to_string)
        .collect();
    got.sort();
    let mut want: Vec<String> = expected(&ids)
        .into_iter()
        .filter(|(_, label)| label == "7")
        .map(|(id, _)| id)
        .collect();
    want.sort();
    assert_eq!(got, want);
    assert!(!want.is_empty());

    // A label escpod would have written (`BC04`) is not what a WarpDemuX
    // table says (`4`), and nothing is silently renamed to make it so.
    let none = ws.out("none.pod5");
    assert_ok(&[
        "filter",
        s(&pod5(&ws)),
        "--annotation",
        "barcode=BC04",
        "-o",
        s(&none),
    ]);
    let out = assert_ok(&["view", s(&none), "--ids"]);
    assert!(
        out.stdout.is_empty(),
        "BC04 selected reads from a table of bare ints"
    );

    // `demux split --sidecar` makes one file per label, `unclassified` included.
    let split = ws.out("split");
    assert_ok(&[
        "demux",
        "split",
        s(&pod5(&ws)),
        "--sidecar",
        "-d",
        s(&split),
    ]);
    let mut names: Vec<String> = std::fs::read_dir(&split)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    assert_eq!(
        names,
        [
            "barcode_-1.pod5",
            "barcode_3.pod5",
            "barcode_4.pod5",
            "barcode_5.pod5",
            "barcode_7.pod5",
            "barcode_unclassified.pod5",
        ]
    );

    assert_eq!(
        std::fs::read(pod5(&ws)).unwrap(),
        pod5_before,
        "the POD5 was modified"
    );
}
