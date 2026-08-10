// SPDX-License-Identifier: MIT

//! Bit-level parity of the k-mer level primitives against golden vectors
//! generated from leech's NumPy reference implementations
//! (`tests/fixtures/gen_kmer_levels_golden.py`, rnabioco/escapepod-rs#204).
//!
//! Floats travel as IEEE-754 bit patterns, so every comparison here is
//! exact — except `rough_rescale_quantile`'s final signal, where the
//! closed-form least squares replaces NumPy's SVD `lstsq` and up to 1 ulp
//! of `f32` slack is allowed (the fit parameters agree to ~1e-13 relative).

use escapepod_signal::resquiggle::{extract_levels, load_kmer_table, rough_rescale_quantile};
use serde_json::Value;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn golden() -> Value {
    let text = std::fs::read_to_string(fixture("kmer_levels_golden.json")).unwrap();
    serde_json::from_str(&text).unwrap()
}

fn f32s(v: &Value) -> Vec<f32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| f32::from_bits(x.as_u64().unwrap() as u32))
        .collect()
}

fn i64s(v: &Value) -> Vec<i64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_i64().unwrap())
        .collect()
}

/// ULP distance between two f32 of the same sign; `u32::MAX` across signs.
fn ulp_diff(a: f32, b: f32) -> u32 {
    if a == b {
        return 0; // covers +0.0 vs -0.0
    }
    if a.is_sign_positive() != b.is_sign_positive() {
        return u32::MAX;
    }
    a.to_bits().abs_diff(b.to_bits())
}

fn load_golden_table() -> (HashMap<String, f64>, usize, Value) {
    let g = golden();
    let (map, k) = load_kmer_table(&fixture(g["table_file"].as_str().unwrap())).unwrap();
    (map, k, g)
}

#[test]
fn parity_load_kmer_table() {
    let (map, k, g) = load_golden_table();
    let want = g["load_kmer_table"]["levels"].as_object().unwrap();
    assert_eq!(k as u64, g["load_kmer_table"]["k"].as_u64().unwrap());
    assert_eq!(map.len(), want.len(), "kmer set size");
    for (kmer, bits) in want {
        let want_level = f64::from_bits(bits.as_u64().unwrap());
        let got = map
            .get(kmer)
            .unwrap_or_else(|| panic!("kmer {kmer} missing"));
        assert_eq!(got.to_bits(), want_level.to_bits(), "level for {kmer}");
    }
}

#[test]
fn parity_extract_levels() {
    // leech's Python extract_levels stores levels as float32; the f64 port
    // must match it exactly after an `as f32` cast (both are the same
    // correctly-rounded conversion of the same parsed f64).
    let (map, k, g) = load_golden_table();
    for case in g["extract_levels"].as_array().unwrap() {
        let seq = case["seq"].as_str().unwrap();
        let center = case["center"].as_u64().map(|c| c as usize);
        let want = f32s(&case["out_f32_bits"]);
        let got = extract_levels(seq, &map, k, center);
        assert_eq!(got.len(), want.len(), "length for {seq:?}");
        for (i, (&g64, &w)) in got.iter().zip(want.iter()).enumerate() {
            assert_eq!(
                (g64 as f32).to_bits(),
                w.to_bits(),
                "seq {seq:?} center {center:?} position {i}: {g64} vs {w}"
            );
        }
    }
}

#[test]
fn parity_rough_rescale_quantile() {
    let g = golden();
    for case in g["rough_rescale_quantile"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let signal = f32s(&case["signal_bits"]);
        let levels = f32s(&case["levels_bits"]);
        let map = i64s(&case["map"]);
        let clip = case["clip_bases"].as_u64().unwrap() as usize;
        let got = rough_rescale_quantile(&signal, &levels, &map, clip).unwrap();
        assert_eq!(got.len(), signal.len(), "{name}: length");

        if case["degenerate"].as_bool().unwrap() {
            // Singular post-clip fit: NumPy returns a minimum-norm lstsq
            // fit, the port documents returning the signal unchanged.
            assert_eq!(got, signal, "{name}: degenerate case must be a no-op");
            continue;
        }

        let want = f32s(&case["out_bits"]);
        let mut inexact = 0usize;
        for (i, (&a, &b)) in got.iter().zip(want.iter()).enumerate() {
            let d = ulp_diff(a, b);
            assert!(d <= 1, "{name}: position {i}: {a} vs {b} ({d} ulps apart)");
            inexact += (d != 0) as usize;
        }
        // The closed-form fit tracks NumPy's lstsq to ~1e-13 relative;
        // rounding differences should be vanishingly rare, not systematic.
        assert!(
            inexact * 100 <= want.len(),
            "{name}: {inexact}/{} samples differ from NumPy by 1 ulp — systematic drift",
            want.len()
        );
    }
}
