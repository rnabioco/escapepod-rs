// SPDX-License-Identifier: MIT
// Moved from rnabioco/leech (MIT) per rnabioco/escapepod-rs#204.

//! Remora-convention k-mer level primitives shared with leech.
//!
//! Canonical home of the signal-refinement primitives both leech and the
//! charging classifier consume (#204): a lenient level-table parser,
//! expected-level extraction with an explicit center index, and the
//! quantile-based rough rescale that puts observed signal into k-mer level
//! units. leech's `leech_core` delegates here, so the two consumers cannot
//! drift numerically — the k-mer *residual* (observed minus expected level)
//! is defined relative to these exact functions.
//!
//! Distinct from [`KmerTable`](super::KmerTable), which serves the
//! `resquiggle` command with fishnet conventions: a strict complete-table
//! parse, `f32` levels, and a Kruskal-Wallis-chosen dominant base. The
//! functions here follow the Remora conventions leech uses instead: levels
//! stay `f64` exactly as parsed, the center index is explicit (it comes from
//! model metadata), and unknown k-mers or too-short sequences yield zero
//! levels rather than errors.
//!
//! Numerics contract (pinned by `tests/kmer_levels_parity.rs` against golden
//! vectors generated from leech's NumPy implementations):
//! - [`load_kmer_table`] parses levels as `f64`; Python's `float()` and
//!   Rust's `f64::from_str` are both correctly rounded, so values agree
//!   bit-for-bit. Casting a level to `f32` reproduces leech's Python
//!   `extract_levels` output exactly (same correctly-rounded `f64 → f32`).
//! - [`extract_levels`] is a literal port of leech's Rust
//!   `extract_levels_inner` (bit-identical).
//! - [`rough_rescale_quantile`] mirrors NumPy op-for-op — `int64` floor-div
//!   centers, `arange` quantile recurrence, and NumPy's quantile lerp
//!   (`f64` interpolation with the `t >= 0.5` variant; for `f32` data the
//!   bracketing values stay `f32`, so their difference rounds to `f32`
//!   before the interpolation — all verified at the bit level) — except
//!   the 2-parameter fit, where NumPy's
//!   SVD `lstsq` is replaced by the closed-form least squares. On the
//!   quantile fit's well-conditioned 19 points the two agree to ~1e-13
//!   relative, below the final `f32` rounding for all but ~1-ulp cases; on a
//!   *singular* fit (all post-clip signal quantiles equal, only possible for
//!   degenerate maps of ≤ 24 bases) NumPy would return a meaningless
//!   minimum-norm fit while this returns the signal unchanged.

use anyhow::{Result, bail};
use flate2::read::GzDecoder;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Bases clipped from each end of the quantile rough rescale (leech default).
pub const ROUGH_RESCALE_CLIP_BASES: usize = 10;

/// Load a k-mer level table from a tab-separated file (`kmer<TAB>level`),
/// transparently gunzipping `.gz` paths. Returns `(kmer → level, kmer_len)`.
///
/// Parsing is lenient, matching leech's `load_kmer_table`: blank lines and
/// `#` comments are skipped, whitespace splitting is the fallback when a
/// line has no tab, header rows (unparseable level) are skipped, k-mers are
/// uppercased as stored, duplicates last-win, and `kmer_len` is taken from
/// the last parsed row. The one tightening over leech: a file that yields
/// no levels at all is an error rather than an empty map.
pub fn load_kmer_table(path: &Path) -> Result<(HashMap<String, f64>, usize)> {
    let file = File::open(path)
        .map_err(|e| anyhow::anyhow!("cannot open kmer table {}: {}", path.display(), e))?;
    let is_gz = path.to_string_lossy().ends_with(".gz");
    let reader: Box<dyn BufRead> = if is_gz {
        Box::new(BufReader::new(GzDecoder::new(file)))
    } else {
        Box::new(BufReader::new(file))
    };

    let mut kmer_to_level = HashMap::new();
    let mut kmer_len = 0usize;

    for line in reader.lines() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut parts: Vec<&str> = line.split('\t').collect();
        if parts.len() < 2 {
            parts = line.split_whitespace().collect();
        }
        if parts.len() < 2 {
            continue;
        }
        // Header rows fail the level parse and are skipped.
        let Ok(level) = parts[1].parse::<f64>() else {
            continue;
        };
        let kmer = parts[0].to_uppercase();
        kmer_len = kmer.chars().count();
        kmer_to_level.insert(kmer, level);
    }

    if kmer_to_level.is_empty() {
        bail!("no parseable kmer levels in {}", path.display());
    }
    Ok((kmer_to_level, kmer_len))
}

/// Expected signal level per base of `sequence` from a k-mer level map.
///
/// For each full k-mer window the level is assigned to the window's
/// `center_idx`-th base (`None` → `kmer_len / 2`). The sequence is
/// uppercased with `U → T` before lookup. Positions with no full window or
/// an unknown k-mer stay at 0. A sequence shorter than `kmer_len` returns
/// all zeros.
///
/// # Panics
/// If `center_idx >= kmer_len` (the center must lie inside the k-mer).
pub fn extract_levels(
    sequence: &str,
    kmer_to_level: &HashMap<String, f64>,
    kmer_len: usize,
    center_idx: Option<usize>,
) -> Vec<f64> {
    let seq_bytes = sequence.as_bytes();
    let seq_len = seq_bytes.len();
    let mut levels = vec![0.0f64; seq_len];
    let cidx = center_idx.unwrap_or(kmer_len / 2);

    if seq_len < kmer_len {
        return levels;
    }

    let seq_upper: Vec<u8> = seq_bytes
        .iter()
        .map(|&b| {
            let c = b.to_ascii_uppercase();
            if c == b'U' { b'T' } else { c }
        })
        .collect();

    for pos in 0..=(seq_len - kmer_len) {
        let kmer = std::str::from_utf8(&seq_upper[pos..pos + kmer_len]).unwrap_or("");
        if let Some(&level) = kmer_to_level.get(kmer) {
            levels[pos + cidx] = level;
        }
    }

    levels
}

/// Quantile-based rough rescale (Remora convention): fit observed signal
/// into k-mer level units and return the rescaled signal.
///
/// Takes the signal value at each base's center (`(map[i] + map[i+1]) / 2`,
/// floor division), clips `clip_bases` bases from each end (only when more
/// than `2 * clip_bases` bases exist), matches the 0.05..0.95 quantiles of
/// those center values against the same quantiles of `expected_levels` with
/// a 2-parameter least-squares fit, and applies
/// `scale * signal + shift` per sample (computed in `f64`, rounded to
/// `f32`). Returns the signal unchanged when the fit is singular or its
/// slope's magnitude is below 1e-10.
///
/// `expected_levels` is `f32` deliberately: that is the dtype leech's
/// pipeline feeds (its `extract_levels` output), and the fit is sensitive
/// to the levels being f32-quantized before the quantiles are taken.
/// Levels from [`extract_levels`] cast with `as f32` match leech
/// bit-for-bit.
///
/// Errors on a map/levels length mismatch or an out-of-range center index
/// (NumPy would wrap negative indexes or raise; neither is meaningful
/// here).
pub fn rough_rescale_quantile(
    signal: &[f32],
    expected_levels: &[f32],
    seq_to_sig_map: &[i64],
    clip_bases: usize,
) -> Result<Vec<f32>> {
    let n_bases = seq_to_sig_map.len().saturating_sub(1);
    if n_bases == 0 {
        bail!("seq_to_sig_map needs at least 2 entries");
    }
    if expected_levels.len() != n_bases {
        bail!(
            "expected_levels length {} != seq_to_sig_map length - 1 ({})",
            expected_levels.len(),
            n_bases
        );
    }

    let mut center_signal = Vec::with_capacity(n_bases);
    for w in seq_to_sig_map.windows(2) {
        let center = (w[0] + w[1]).div_euclid(2);
        if center < 0 || center as usize >= signal.len() {
            bail!(
                "base center {} out of range for signal of length {}",
                center,
                signal.len()
            );
        }
        center_signal.push(signal[center as usize] as f64);
    }

    let (center_signal, levels): (&[f64], &[f32]) = if clip_bases > 0 && n_bases > clip_bases * 2 {
        (
            &center_signal[clip_bases..n_bases - clip_bases],
            &expected_levels[clip_bases..n_bases - clip_bases],
        )
    } else {
        (&center_signal, expected_levels)
    };

    // NumPy `arange(0.05, 1, 0.05)`: element i is `0.05 + i * 0.05` in f64.
    let quants: Vec<f64> = (0..19).map(|i| 0.05 + i as f64 * 0.05).collect();

    let sig_qs = quantiles_f64(center_signal, &quants);
    let level_qs = quantiles_f32_data(levels, &quants);

    // Closed-form 2-parameter least squares: level = shift + scale * sig.
    let n = quants.len() as f64;
    let x_mean = sig_qs.iter().sum::<f64>() / n;
    let y_mean = level_qs.iter().sum::<f64>() / n;
    let mut sxx = 0.0f64;
    let mut sxy = 0.0f64;
    for (&x, &y) in sig_qs.iter().zip(level_qs.iter()) {
        let dx = x - x_mean;
        sxx += dx * dx;
        sxy += dx * (y - y_mean);
    }
    if sxx == 0.0 {
        // Singular fit (all signal quantiles equal). NumPy's lstsq would
        // return a minimum-norm fit here; a quantile fit with no spread is
        // meaningless either way, so leave the signal unchanged.
        return Ok(signal.to_vec());
    }
    let scale_est = sxy / sxx;
    let shift_est = y_mean - scale_est * x_mean;

    if scale_est.abs() < 1e-10 {
        return Ok(signal.to_vec());
    }

    Ok(signal
        .iter()
        .map(|&s| (scale_est * s as f64 + shift_est) as f32)
        .collect())
}

/// NumPy-default (`linear`) quantiles of `f64` data.
fn quantiles_f64(data: &[f64], quants: &[f64]) -> Vec<f64> {
    let mut sorted = data.to_vec();
    sorted.sort_unstable_by(|a, b| a.total_cmp(b));
    quants
        .iter()
        .map(|&q| {
            let (lo, hi, gamma) = quantile_pos(sorted.len(), q);
            let (a, b) = (sorted[lo], sorted[hi]);
            // NumPy's _lerp: `a + t*(b-a)`, switching to `b - (b-a)*(1-t)`
            // for t >= 0.5 (better behaved when a and b differ in sign).
            if gamma < 0.5 {
                a + (b - a) * gamma
            } else {
                b - (b - a) * (1.0 - gamma)
            }
        })
        .collect()
}

/// NumPy-default (`linear`) quantiles of `f32` data. The output promotes to
/// `f64` (the quantile grid is `f64`), but NumPy takes the bracketing values
/// in the array's dtype, so their difference `b - a` rounds to `f32` before
/// entering the `f64` interpolation. That rounding is observable whenever a
/// sorted-adjacent pair spans binades, and reproducing it is required for
/// bit parity with leech (`tests/fixtures/` probes pin it).
fn quantiles_f32_data(data: &[f32], quants: &[f64]) -> Vec<f64> {
    let mut sorted = data.to_vec();
    sorted.sort_unstable_by(|a, b| a.total_cmp(b));
    quants
        .iter()
        .map(|&q| {
            let (lo, hi, gamma) = quantile_pos(sorted.len(), q);
            let (a, b) = (sorted[lo], sorted[hi]);
            let diff = (b - a) as f64; // f32 subtract, then promote
            if gamma < 0.5 {
                a as f64 + diff * gamma
            } else {
                b as f64 - diff * (1.0 - gamma)
            }
        })
        .collect()
}

/// Bracketing indexes and interpolant for NumPy's `linear` quantile method:
/// virtual index `q * (n - 1)` split into floor, floor+1 (clamped), and the
/// fractional part.
fn quantile_pos(n: usize, q: f64) -> (usize, usize, f64) {
    debug_assert!(n > 0, "quantile of empty data");
    let virtual_idx = q * (n - 1) as f64;
    let lo = (virtual_idx.floor() as usize).min(n - 1);
    let hi = (lo + 1).min(n - 1);
    (lo, hi, virtual_idx - lo as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table() -> HashMap<String, f64> {
        let mut m = HashMap::new();
        m.insert("AAAAA".to_string(), 1.5);
        m.insert("AAAAC".to_string(), -0.25);
        m.insert("AAACG".to_string(), 0.75);
        m
    }

    #[test]
    fn test_extract_levels_center_default() {
        // k=5, default center = 2: window at pos 0 writes levels[2].
        let levels = extract_levels("AAAAACG", &table(), 5, None);
        assert_eq!(levels.len(), 7);
        assert_eq!(levels[2], 1.5); // AAAAA
        assert_eq!(levels[3], -0.25); // AAAAC
        assert_eq!(levels[4], 0.75); // AAACG
        assert_eq!(levels[0], 0.0);
        assert_eq!(levels[1], 0.0);
    }

    #[test]
    fn test_extract_levels_explicit_center() {
        let levels = extract_levels("AAAAACG", &table(), 5, Some(0));
        assert_eq!(levels[0], 1.5);
        assert_eq!(levels[1], -0.25);
        assert_eq!(levels[2], 0.75);
    }

    #[test]
    fn test_extract_levels_case_and_u() {
        // Lowercase and U→T both resolve; RNA 'u' maps to the T-form kmer.
        let mut m = table();
        m.insert("AAAAT".to_string(), 9.0);
        let levels = extract_levels("aaaau", &m, 5, None);
        assert_eq!(levels[2], 9.0);
    }

    #[test]
    fn test_extract_levels_unknown_kmer_and_short_seq() {
        // N-containing window misses the map → stays 0.
        let levels = extract_levels("AANAACG", &table(), 5, None);
        assert!(levels.iter().all(|&l| l == 0.0));
        // Shorter than k → all zeros, not an error.
        let levels = extract_levels("ACG", &table(), 5, None);
        assert_eq!(levels, vec![0.0; 3]);
    }

    #[test]
    fn test_load_kmer_table_lenient() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("levels.tsv");
        std::fs::write(
            &path,
            "# comment\nkmer\tlevel_mean\nAAAAA\t0.95838\nacgta\t2.5\nAAAAA\t1.5\nGGGGG 0.125\n\nCCCCC\t-0.5\n",
        )
        .unwrap();
        let (map, k) = load_kmer_table(&path).unwrap();
        assert_eq!(k, 5);
        assert_eq!(map.len(), 4);
        assert_eq!(map["AAAAA"], 1.5); // duplicate: last wins
        assert_eq!(map["ACGTA"], 2.5); // uppercased
        assert_eq!(map["GGGGG"], 0.125); // whitespace fallback
        assert_eq!(map["CCCCC"], -0.5);
    }

    #[test]
    fn test_load_kmer_table_empty_is_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.tsv");
        std::fs::write(&path, "# only a comment\nkmer\tlevel_mean\n").unwrap();
        assert!(load_kmer_table(&path).is_err());
    }

    #[test]
    fn test_rough_rescale_quantile_recovers_linear_transform() {
        // levels = 2 * signal + 1 at every base center → the fit must
        // recover exactly that transform (all quantile pairs are colinear).
        let n_bases = 40;
        let map: Vec<i64> = (0..=n_bases as i64).map(|i| i * 10).collect();
        let signal: Vec<f32> = (0..n_bases * 10)
            .map(|i| ((i % 17) as f32) * 0.1 - 0.8)
            .collect();
        let levels: Vec<f32> = map
            .windows(2)
            .map(|w| {
                let c = ((w[0] + w[1]) / 2) as usize;
                2.0 * signal[c] + 1.0
            })
            .collect();
        let out = rough_rescale_quantile(&signal, &levels, &map, ROUGH_RESCALE_CLIP_BASES).unwrap();
        for (s, o) in signal.iter().zip(out.iter()) {
            assert!((o - (2.0 * s + 1.0)).abs() < 1e-4, "{} -> {}", s, o);
        }
    }

    #[test]
    fn test_rough_rescale_quantile_flat_levels_unchanged() {
        // Constant levels → slope 0 → signal returned unchanged.
        let map: Vec<i64> = (0..=30i64).map(|i| i * 5).collect();
        let signal: Vec<f32> = (0..150).map(|i| (i as f32).sin()).collect();
        let levels = vec![0.5f32; 30];
        let out = rough_rescale_quantile(&signal, &levels, &map, ROUGH_RESCALE_CLIP_BASES).unwrap();
        assert_eq!(out, signal);
    }

    #[test]
    fn test_rough_rescale_quantile_rejects_bad_inputs() {
        let signal = vec![0.0f32; 10];
        // length mismatch
        assert!(rough_rescale_quantile(&signal, &[0.0; 3], &[0, 5, 10], 10).is_err());
        // center out of range (map beyond signal)
        assert!(rough_rescale_quantile(&signal, &[0.0; 2], &[0, 10, 30], 10).is_err());
        // too-short map
        assert!(rough_rescale_quantile(&signal, &[], &[0], 10).is_err());
    }

    #[test]
    fn test_quantiles_match_numpy_convention() {
        // np.quantile([1,2,3,4], [0.25, 0.5, 0.75]) == [1.75, 2.5, 3.25]
        let q = quantiles_f64(&[4.0, 2.0, 1.0, 3.0], &[0.25, 0.5, 0.75]);
        assert!((q[0] - 1.75).abs() < 1e-12);
        assert!((q[1] - 2.5).abs() < 1e-12);
        assert!((q[2] - 3.25).abs() < 1e-12);
    }

    /// Bit-exact NumPy parity for the quantile internals, pinned by the
    /// golden fixture (`tests/fixtures/gen_kmer_levels_golden.py`). The
    /// function-level goldens live in `tests/kmer_levels_parity.rs`; this
    /// covers the two pieces they exercise only indirectly: the
    /// `arange(0.05, 1, 0.05)` recurrence and the promote-f32-then-lerp
    /// quantile semantics.
    #[test]
    fn test_numpy_quantile_probe_parity() {
        let path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/kmer_levels_golden.json");
        let golden: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();

        let quants: Vec<f64> = (0..19).map(|i| 0.05 + i as f64 * 0.05).collect();
        let want_quants: Vec<u64> = golden["quants_arange_bits"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap())
            .collect();
        assert_eq!(
            quants.iter().map(|q| q.to_bits()).collect::<Vec<_>>(),
            want_quants,
            "np.arange(0.05, 1, 0.05) recurrence"
        );

        for probe in golden["quantile_probes"].as_array().unwrap() {
            let name = probe["name"].as_str().unwrap();
            assert_eq!(
                probe["out_dtype"].as_str().unwrap(),
                "float64",
                "{name}: NumPy quantile output dtype changed — revisit the port"
            );
            let bits = probe["data_bits"].as_array().unwrap();
            let got: Vec<u64> = if probe["kind"] == "f64" {
                let data: Vec<f64> = bits
                    .iter()
                    .map(|v| f64::from_bits(v.as_u64().unwrap()))
                    .collect();
                quantiles_f64(&data, &quants)
            } else {
                let data: Vec<f32> = bits
                    .iter()
                    .map(|v| f32::from_bits(v.as_u64().unwrap() as u32))
                    .collect();
                quantiles_f32_data(&data, &quants)
            }
            .iter()
            .map(|q| q.to_bits())
            .collect();
            let want: Vec<u64> = probe["out_bits"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap())
                .collect();
            assert_eq!(got, want, "{name} quantile probe");
        }
    }
}
