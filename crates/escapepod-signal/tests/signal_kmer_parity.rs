// SPDX-License-Identifier: MIT

//! Parity of `escapepod_signal::seq_encoding` against golden vectors generated
//! from leech's NumPy reference implementation
//! (`tests/fixtures/gen_signal_kmer_golden.py`, rnabioco/escapepod-rs#271).
//!
//! The encoding is exactly zeros and ones, so every comparison here is exact --
//! there is no tolerance to argue about, and a mismatch is always a rule that
//! was ported wrong rather than arithmetic that drifted.
//!
//! The golden is generated from the NumPy path deliberately. leech dispatches
//! the same function to its `leech_core` extension when that is importable, and
//! the two disagree on a span with a negative start: NumPy keeps the surviving
//! tail, the extension drops the base. This crate follows NumPy, and the
//! `negative_start` case below is what says so out loud.

use escapepod_signal::seq_encoding::{
    KmerContext, encode_signal_kmer, encode_signal_kmer_into, sequence_to_int,
};
use serde_json::Value;
use std::path::{Path, PathBuf};

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn golden() -> Value {
    let text = std::fs::read_to_string(fixture("signal_kmer_golden.json")).unwrap();
    serde_json::from_str(&text).unwrap()
}

/// The golden's `'0'`/`'1'` string as the floats the encoder emits.
fn expected_encoding(case: &Value) -> Vec<f32> {
    case["encoding"]
        .as_str()
        .unwrap()
        .chars()
        .map(|c| match c {
            '0' => 0.0,
            '1' => 1.0,
            other => panic!("the encoding is one-hot; found {other:?}"),
        })
        .collect()
}

/// Where the two arrays first differ, as `(row, sample)` -- a flat index is
/// unreadable at 36 channels, and which row it is says which k-mer position
/// and which base got it wrong.
fn first_difference(got: &[f32], want: &[f32], signal_len: usize) -> Option<String> {
    got.iter().zip(want).position(|(a, b)| a != b).map(|i| {
        let (row, sample) = (i / signal_len, i % signal_len);
        let (kmer_pos, base) = (row / 4, "ACGT".as_bytes()[row % 4] as char);
        format!(
            "row {row} (k-mer position {kmer_pos}, base {base}), sample {sample}: \
                 got {}, want {}",
            got[i], want[i]
        )
    })
}

#[test]
fn sequence_to_int_matches_leech() {
    for entry in golden()["sequence_to_int"].as_array().unwrap() {
        let seq = entry["sequence"].as_str().unwrap();
        let want: Vec<i8> = entry["ints"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap() as i8)
            .collect();
        assert_eq!(sequence_to_int(seq.as_bytes()), want, "sequence {seq:?}");
    }
}

#[test]
fn encode_signal_kmer_matches_leech() {
    let g = golden();
    let cases = g["cases"].as_array().unwrap();
    assert!(
        cases.len() >= 30,
        "the golden should not have shrunk silently"
    );

    for case in cases {
        let name = case["name"].as_str().unwrap();
        let seq_ints = sequence_to_int(case["sequence"].as_str().unwrap().as_bytes());
        let map: Vec<i64> = case["seq_to_sig_map"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap())
            .collect();
        let signal_len = case["signal_len"].as_u64().unwrap() as usize;
        let ctx = KmerContext::new(
            case["kmer_before"].as_u64().unwrap() as usize,
            case["kmer_after"].as_u64().unwrap() as usize,
        );

        // The shape leech reports is the shape this crate computes from the
        // context alone -- a caller sizing its model input never sees the
        // array first.
        let shape = case["shape"].as_array().unwrap();
        assert_eq!(
            shape[0].as_u64().unwrap() as usize,
            ctx.channels(),
            "{name}: rows"
        );
        assert_eq!(
            shape[1].as_u64().unwrap() as usize,
            signal_len,
            "{name}: columns"
        );

        let want = expected_encoding(case);
        let got = encode_signal_kmer(&seq_ints, &map, signal_len, ctx);
        assert_eq!(got.len(), want.len(), "{name}: length");
        if let Some(diff) = first_difference(&got, &want, signal_len.max(1)) {
            let note = case["note"].as_str().unwrap_or("");
            panic!("{name}: {diff}\n{note}");
        }

        // The buffer-reusing form is the same function, and a hot loop's
        // leftovers must not leak into it.
        let mut buf = vec![1.0f32; ctx.channels() * signal_len];
        encode_signal_kmer_into(&seq_ints, &map, signal_len, ctx, &mut buf);
        assert_eq!(
            buf, got,
            "{name}: `_into` disagrees with the allocating form"
        );
    }
}
