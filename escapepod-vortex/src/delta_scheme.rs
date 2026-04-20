//! Custom `DeltaScheme` for `vortex-btrblocks` (vortex 0.68 API).
//!
//! Selection trail for nanopore i16 signal:
//!     i16 → ZigZag → u16 → Delta → (bases, deltas) → BitPacking
//! mirroring VBZ's delta+zigzag+SVB16 pipeline.
//!
//! This scheme deliberately accepts SIGNED integer inputs in addition to
//! unsigned ones. The cascade selector won't usually invoke ZigZagScheme on a
//! positive-valued nanopore signal column (because `min_is_negative()` is
//! false), so we do the ZigZag step ourselves before delta-encoding. The
//! resulting array is wrapped as `ZigZag(Delta(unsigned bases, deltas))` so
//! the canonical decode path naturally reconstructs the original signed
//! values.

use vortex::array::dtype::PType;
use vortex::array::{ArrayRef, Canonical, IntoArray, ToCanonical};
use vortex_btrblocks::{Scheme, SchemeExt};
use vortex_compressor::CascadingCompressor;
use vortex_compressor::ctx::CompressorContext;
use vortex_compressor::estimate::CompressionEstimate;
use vortex_compressor::stats::ArrayAndStats;
use vortex_error::VortexResult;
use vortex_fastlanes::{Delta, delta_compress};
use vortex_zigzag::{ZigZag, ZigZagArrayExt, zigzag_encode};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeltaScheme;

impl DeltaScheme {
    fn ptype_supported(p: PType) -> bool {
        matches!(
            p,
            PType::I8 | PType::I16 | PType::I32 | PType::I64
                | PType::U8 | PType::U16 | PType::U32 | PType::U64
        )
    }
}

impl Scheme for DeltaScheme {
    fn scheme_name(&self) -> &'static str {
        "escapepod.delta"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        let Canonical::Primitive(p) = canonical else {
            return false;
        };
        Self::ptype_supported(p.ptype())
    }

    fn num_children(&self) -> usize {
        2 // bases + deltas
    }

    fn expected_compression_ratio(
        &self,
        _data: &mut ArrayAndStats,
        ctx: CompressorContext,
    ) -> CompressionEstimate {
        if ctx.finished_cascading() {
            return CompressionEstimate::Skip;
        }
        // Defer to a sample compression — the cascade will measure us against
        // FoR/BitPacking/pco and pick the winner.
        CompressionEstimate::Sample
    }

    fn compress(
        &self,
        compressor: &CascadingCompressor,
        data: &mut ArrayAndStats,
        ctx: CompressorContext,
    ) -> VortexResult<ArrayRef> {
        let prim = data.array().to_primitive();
        let logical_len = prim.len();
        let signed = prim.ptype().is_signed_int();

        // If signed, zigzag to unsigned first.
        let (unsigned_prim, zigzag_wrap) = if signed {
            let zag = zigzag_encode(prim)?;
            let inner = zag.encoded().to_primitive();
            (inner, true)
        } else {
            (prim, false)
        };

        // delta_compress takes unsigned, returns (bases, deltas) PrimitiveArrays.
        let (bases, deltas) = {
            let mut exec = compressor.execution_ctx();
            delta_compress(&unsigned_prim, &mut exec)?
        };

        // Recursively compress each child via the cascade. BitPacking is the
        // expected pick for the deltas (small magnitudes); bases are tiny.
        let bases_compressed =
            compressor.compress_child(&bases.into_array(), &ctx, self.id(), 0)?;
        let deltas_compressed =
            compressor.compress_child(&deltas.into_array(), &ctx, self.id(), 1)?;

        let delta_arr =
            Delta::try_new(bases_compressed, deltas_compressed, 0, logical_len)?.into_array();

        if zigzag_wrap {
            Ok(ZigZag::try_new(delta_arr)?.into_array())
        } else {
            Ok(delta_arr)
        }
    }
}
