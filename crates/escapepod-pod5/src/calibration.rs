//! ADC → picoampere calibration: the one definition (rnabioco/escapepod-rs#501).
//!
//! A POD5 read stores raw ADC counts plus `calibration_offset` and
//! `calibration_scale`; the current in pA is `(adc + offset) * scale`. That
//! expression has two `f32` roundings as written. Every caller here uses the
//! **fused** form instead — `adc.mul_add(scale, offset * scale)`, one rounding
//! for the multiply-add plus one for the precomputed bias — and the two differ
//! by about an ulp on some samples. Four sites used to spell this out by hand,
//! in both forms; a model fitted on one and scored on the other inherits that
//! ulp silently. So there is one function, and every site calls it.

/// Calibrate one raw ADC sample to picoamperes: `(adc + offset) * scale`,
/// evaluated as the fused `adc.mul_add(scale, offset * scale)`.
///
/// `#[inline]` so a caller's loop hoists `offset * scale` and keeps the FMA
/// in the loop body — the result is bit-identical whether or not it does.
#[inline]
pub fn calibrate_pa(adc: i16, offset: f32, scale: f32) -> f32 {
    f32::from(adc).mul_add(scale, offset * scale)
}

/// Calibrate a slice of raw ADC samples to picoamperes, sample by sample
/// through [`calibrate_pa`].
pub fn calibrate_pa_slice(adc: &[i16], offset: f32, scale: f32) -> Vec<f32> {
    adc.iter()
        .map(|&v| calibrate_pa(v, offset, scale))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const OFFSET: f32 = 7.3;
    const SCALE: f32 = 0.183_452_1;

    fn adc() -> Vec<i16> {
        let mut v: Vec<i16> = vec![i16::MIN, -1, 0, 1, i16::MAX];
        v.extend((0..2000).map(|i| ((i * 37) % 4096 - 2048) as i16));
        v
    }

    #[test]
    fn calibrate_pa_is_the_fused_form() {
        let bias = OFFSET * SCALE;
        for a in adc() {
            let want = f32::from(a).mul_add(SCALE, bias);
            assert_eq!(calibrate_pa(a, OFFSET, SCALE).to_bits(), want.to_bits());
        }
    }

    /// The fused and unfused forms must actually disagree on this input, or
    /// the fused-form pin above (and the cross-language golden) would pass
    /// for either definition.
    #[test]
    fn fused_differs_from_unfused_somewhere() {
        let diffs = adc()
            .into_iter()
            .filter(|&a| {
                calibrate_pa(a, OFFSET, SCALE).to_bits()
                    != ((f32::from(a) + OFFSET) * SCALE).to_bits()
            })
            .count();
        assert!(diffs > 0, "fixture cannot tell fused from unfused");
    }

    #[test]
    fn slice_matches_scalar() {
        let a = adc();
        let got = calibrate_pa_slice(&a, OFFSET, SCALE);
        assert_eq!(got.len(), a.len());
        for (&x, &p) in a.iter().zip(&got) {
            assert_eq!(p.to_bits(), calibrate_pa(x, OFFSET, SCALE).to_bits());
        }
    }
}
