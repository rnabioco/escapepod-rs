//! Small shared statistics helpers.
//!
//! A single O(n)-expected median (via `select_nth_unstable`) plus a median/MAD
//! pair, shared across the signal and demux layers. Several modules previously
//! carried their own byte-for-byte copies of this logic; consolidating them
//! here keeps the semantics bit-identical — the same `f32::total_cmp` ordering
//! and the same even/odd midpoint convention — which demux/classify parity and
//! resquiggle output both depend on.

/// Median of a slice via `select_nth_unstable` (O(n) expected).
///
/// Partially reorders `data` in place. Uses `f32::total_cmp` for a total order
/// (NaN sorts deterministically to the high end, so there is no panic). For an
/// even length the two central elements are averaged; for an odd length the
/// central element is returned. Returns `0.0` for an empty slice.
pub fn median_via_select(data: &mut [f32]) -> f32 {
    let n = data.len();
    if n == 0 {
        return 0.0;
    }
    let mid = n / 2;
    let (lo_part, pivot, _) = data.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
    let hi = *pivot;
    if n.is_multiple_of(2) {
        // data[..mid] is an unsorted partition of the values <= hi; its max is
        // the lower-median element.
        let lo = lo_part.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        (lo + hi) / 2.0
    } else {
        hi
    }
}

/// Median and MAD (median absolute deviation) of `data`, each O(n) expected.
///
/// Returns `(median, mad)`. A single scratch copy is made and reused across
/// both passes. Returns `(0.0, 0.0)` for an empty slice.
pub fn median_and_mad(data: &[f32]) -> (f32, f32) {
    // select_nth partitions in place; clone once and reuse buf for the MAD pass.
    let mut buf = data.to_vec();
    let med = median_via_select(&mut buf);

    // Overwrite buf with absolute deviations from the median (keyed off the
    // original data, whose order buf no longer shares).
    for (v, slot) in data.iter().zip(buf.iter_mut()) {
        *slot = (*v - med).abs();
    }
    let mad = median_via_select(&mut buf);

    (med, mad)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_odd_even_single_empty() {
        let mut a = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(median_via_select(&mut a), 3.0);
        let mut b = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(median_via_select(&mut b), 2.5);
        let mut c = [42.0];
        assert_eq!(median_via_select(&mut c), 42.0);
        assert_eq!(median_via_select(&mut []), 0.0);
    }

    #[test]
    fn median_order_independent() {
        let mut odd = [5.0, 3.0, 1.0, 4.0, 2.0];
        assert_eq!(median_via_select(&mut odd), 3.0);
        let mut even = [4.0, 1.0, 3.0, 2.0];
        assert_eq!(median_via_select(&mut even), 2.5);
    }

    #[test]
    fn median_and_mad_basic() {
        // data [1,2,3,4,5]: median 3; |dev| [2,1,0,1,2] → MAD 1.
        let (m, mad) = median_and_mad(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(m, 3.0);
        assert_eq!(mad, 1.0);
        assert_eq!(median_and_mad(&[]), (0.0, 0.0));
    }

    /// Cross-check the select-based median against a straightforward sort-based
    /// reference across odd/even/duplicate/negative inputs — this is the
    /// equivalence the six migrated call sites rely on.
    #[test]
    fn median_matches_sort_reference() {
        fn ref_median(v: &[f32]) -> f32 {
            if v.is_empty() {
                return 0.0;
            }
            let mut s = v.to_vec();
            s.sort_unstable_by(|a, b| a.total_cmp(b));
            let n = s.len();
            if n % 2 == 1 {
                s[n / 2]
            } else {
                (s[n / 2 - 1] + s[n / 2]) / 2.0
            }
        }
        let cases: &[&[f32]] = &[
            &[3.0, 1.0, 2.0],
            &[1.0, 2.0, 3.0, 4.0],
            &[5.0],
            &[2.0, 2.0, 2.0, 2.0],
            &[-1.0, 0.5, 100.0, -3.0, 7.0, 7.0],
        ];
        for c in cases {
            let mut buf = c.to_vec();
            assert_eq!(median_via_select(&mut buf), ref_median(c));
        }
    }
}
