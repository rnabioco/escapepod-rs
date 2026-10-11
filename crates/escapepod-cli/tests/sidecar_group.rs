// SPDX-License-Identifier: MIT

//! `escpod sidecar` (#488): `index` and `annotate` grouped under the noun that
//! names the file they both write, with the old top-level spellings kept as
//! hidden, deprecated forwarders.
//!
//! These run the real binary on a tempdir copy of the tracked fixture. The
//! claim under test is "no behaviour change", so each pair of spellings runs
//! on two identical copies and the sidecar *bytes* are compared — the sidecar
//! is a pure function of the POD5 and the labels, so any difference is the
//! forwarder doing something the real command does not.

#![cfg(feature = "cli")]

mod common;

use std::path::{Path, PathBuf};
use std::process::Output;

use common::{Workspace, assert_ok, run};

fn s(path: &Path) -> &str {
    path.to_str().unwrap()
}

fn sidecar(ws: &Workspace) -> PathBuf {
    ws.dir.join("reads.pod5.p5s")
}

fn stderr(out: &Output) -> String {
    String::from_utf8_lossy(&out.stderr).into_owned()
}

/// Lines of stderr that announce a deprecation.
fn deprecation_lines(out: &Output) -> Vec<String> {
    stderr(out)
        .lines()
        .filter(|l| l.contains("deprecated"))
        .map(str::to_string)
        .collect()
}

/// The fixture's read ids, from the binary rather than a hard-coded list.
fn read_ids(ws: &Workspace) -> Vec<String> {
    let out = assert_ok(&["view", s(&ws.dir.join("reads.pod5")), "--ids"]);
    let ids: Vec<String> = String::from_utf8(out.stdout)
        .unwrap()
        .lines()
        .map(str::to_string)
        .collect();
    assert_eq!(ids.len(), 20, "the fixture is expected to hold 20 reads");
    ids
}

/// A read → label table over the first twelve reads, `read_id,predicted_barcode`.
fn write_table(ws: &Workspace, ids: &[String]) -> PathBuf {
    let rows: Vec<String> = ids
        .iter()
        .take(12)
        .enumerate()
        .map(|(i, id)| format!("{id},BC{:02}", i % 3 + 1))
        .collect();
    let path = ws.out("calls.csv");
    std::fs::write(
        &path,
        format!("read_id,predicted_barcode\n{}\n", rows.join("\n")),
    )
    .unwrap();
    path
}

/// The first whitespace-delimited word of every top-level command line in
/// `escpod --help`: the two-space-indented lines under `Commands:`. Wrapped
/// descriptions are indented further, so they do not count.
fn listed_commands(help: &str) -> Vec<String> {
    help.lines()
        .filter(|l| l.starts_with("  ") && !l.starts_with("   "))
        .filter_map(|l| l.split_whitespace().next())
        .map(str::to_string)
        .collect()
}

#[test]
fn sidecar_index_matches_legacy_index() {
    let new = Workspace::new();
    let old = Workspace::new();

    let a = run(&["sidecar", "index", new.dir()]);
    let b = run(&["index", old.dir()]);

    assert!(a.status.success(), "sidecar index failed:\n{}", stderr(&a));
    assert!(b.status.success(), "legacy index failed:\n{}", stderr(&b));
    assert_eq!(a.status.code(), b.status.code());
    assert_eq!(
        std::fs::read(sidecar(&new)).unwrap(),
        std::fs::read(sidecar(&old)).unwrap(),
        "the two spellings wrote different sidecars"
    );

    // Same failure, same exit code: a path that does not exist.
    let missing = new.out("nope.pod5");
    let a = run(&["sidecar", "index", s(&missing)]);
    let b = run(&["index", s(&missing)]);
    assert!(!a.status.success() && !b.status.success());
    assert_eq!(a.status.code(), b.status.code());
}

#[test]
fn sidecar_annotate_from_table_matches_legacy_annotate() {
    let new = Workspace::new();
    let old = Workspace::new();
    let ids = read_ids(&new);
    let table_new = write_table(&new, &ids);
    let table_old = write_table(&old, &ids);

    let a = run(&[
        "sidecar",
        "annotate",
        new.dir(),
        "--from-table",
        s(&table_new),
        "--column",
        "barcode=predicted_barcode",
    ]);
    let b = run(&[
        "annotate",
        old.dir(),
        "--from-table",
        s(&table_old),
        "--column",
        "barcode=predicted_barcode",
    ]);

    assert!(
        a.status.success(),
        "sidecar annotate failed:\n{}",
        stderr(&a)
    );
    assert!(
        b.status.success(),
        "legacy annotate failed:\n{}",
        stderr(&b)
    );
    assert_eq!(a.status.code(), b.status.code());

    // A directory argument writes one collection sidecar beside the directory.
    let (collection_new, collection_old) =
        (new.dir.with_extension("p5s"), old.dir.with_extension("p5s"));
    assert!(collection_new.exists(), "no sidecar for `sidecar annotate`");
    assert_eq!(
        std::fs::read(&collection_new).unwrap(),
        std::fs::read(&collection_old).unwrap(),
        "the two spellings wrote different sidecars"
    );

    // And what a reader gets back out is the labels the table gave.
    let viewed = |ws: &Workspace| {
        let out = assert_ok(&[
            "view",
            s(&ws.dir.join("reads.pod5")),
            "--include",
            "barcode",
        ]);
        String::from_utf8(out.stdout).unwrap()
    };
    let labelled = viewed(&new);
    assert!(
        labelled.contains("BC01"),
        "no labels read back:\n{labelled}"
    );
    assert_eq!(labelled, viewed(&old));
}

/// The forwarders are not in `--help`, still run, and say once on stderr what
/// to type instead. The real spelling does not say it.
#[test]
fn legacy_index_and_annotate_are_hidden_but_work() {
    let help = String::from_utf8(assert_ok(&["--help"]).stdout).unwrap();
    let listed = listed_commands(&help);
    assert!(
        listed.iter().any(|c| c == "sidecar"),
        "`sidecar` is missing from --help:\n{help}"
    );
    for hidden in ["index", "annotate"] {
        assert!(
            !listed.iter().any(|c| c == hidden),
            "`{hidden}` must be hidden from --help, listed: {listed:?}\n{help}"
        );
    }

    let ws = Workspace::new();
    let ids = read_ids(&ws);

    // `escpod index`.
    let out = run(&["index", ws.dir()]);
    assert!(
        out.status.success(),
        "legacy index failed:\n{}",
        stderr(&out)
    );
    assert!(sidecar(&ws).exists(), "legacy index wrote no sidecar");
    let warned = deprecation_lines(&out);
    assert_eq!(warned.len(), 1, "expected one deprecation line: {warned:?}");
    assert!(
        warned[0].contains("escpod sidecar index"),
        "the warning must name the new spelling: {}",
        warned[0]
    );

    // `escpod annotate`.
    let table = write_table(&ws, &ids);
    let out = run(&[
        "annotate",
        ws.dir(),
        "--from-table",
        s(&table),
        "--column",
        "barcode=predicted_barcode",
    ]);
    assert!(
        out.status.success(),
        "legacy annotate failed:\n{}",
        stderr(&out)
    );
    let warned = deprecation_lines(&out);
    assert_eq!(warned.len(), 1, "expected one deprecation line: {warned:?}");
    assert!(
        warned[0].contains("escpod sidecar annotate"),
        "the warning must name the new spelling: {}",
        warned[0]
    );

    // Hidden is not gone: `--help` on a hidden command still works.
    assert_ok(&["index", "--help"]);
    assert_ok(&["annotate", "--help"]);

    // The real spellings stay quiet.
    let out = run(&["sidecar", "index", ws.dir(), "--force"]);
    assert!(out.status.success(), "{}", stderr(&out));
    assert!(
        deprecation_lines(&out).is_empty(),
        "`sidecar index` must not warn:\n{}",
        stderr(&out)
    );
    let out = run(&["sidecar", "annotate", "--list", ws.dir()]);
    assert!(out.status.success(), "{}", stderr(&out));
    assert!(
        deprecation_lines(&out).is_empty(),
        "`sidecar annotate` must not warn:\n{}",
        stderr(&out)
    );
}

/// The group's own help carries the overview, and each subcommand's help keeps
/// the examples it had as a top-level command — spelled the new way.
#[test]
fn sidecar_help_explains_the_group() {
    let out = assert_ok(&["sidecar", "--help"]);
    let help = String::from_utf8(out.stdout).unwrap();
    let commands = listed_commands(&help);
    for sub in ["index", "annotate"] {
        assert!(
            commands.iter().any(|c| c == sub),
            "`sidecar --help` does not list `{sub}`:\n{help}"
        );
    }
    // The overview: both write the .p5s, annotate builds the index when it is
    // missing, index is the labels-free form, the POD5 is never written.
    for phrase in [
        ".p5s sidecar",
        "builds the read index",
        "labels-free",
        "never written",
    ] {
        assert!(
            help.contains(phrase),
            "`sidecar --help` overview lacks {phrase:?}:\n{help}"
        );
    }

    let index = String::from_utf8(assert_ok(&["sidecar", "index", "--help"]).stdout).unwrap();
    assert!(index.contains("escpod sidecar index input.pod5"), "{index}");
    let annotate = String::from_utf8(assert_ok(&["sidecar", "annotate", "--help"]).stdout).unwrap();
    assert!(annotate.contains("--from-table"), "{annotate}");
    assert!(
        annotate.contains("escpod sidecar annotate -a demux.csv"),
        "{annotate}"
    );

    // A group is a namespace, not a command: bare `escpod sidecar` is a usage
    // error rather than a silent no-op.
    let out = run(&["sidecar"]);
    assert!(!out.status.success());
}
