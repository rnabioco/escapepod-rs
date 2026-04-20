//! Signal-column conversion: POD5 i16 signal → Vortex file with default encodings.

use std::path::Path;
use std::sync::Arc;

use escapepod::Reader;
use vortex::array::ArrayRef;
use vortex::array::IntoArray;
use vortex::array::arrays::{ChunkedArray, ListArray, PrimitiveArray};
use vortex::array::dtype::{DType, Nullability, PType};
use vortex::array::validity::Validity;
use vortex_btrblocks::{BtrBlocksCompressor, BtrBlocksCompressorBuilder};
use vortex_buffer::{Buffer, ByteBuffer};
use vortex_file::{OpenOptionsSessionExt, VortexWriteOptions, WriteStrategyBuilder};
use vortex_session::VortexSession;

use crate::Result;

/// Selects which Vortex encoding pipeline to use for the converter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Codec {
    /// `BtrBlocksCompressorBuilder::default()` — all standard schemes, no pco/zstd.
    Default,
    /// Default + `with_compact()` — adds Pco for ints/floats and Zstd for strings.
    Pco,
    /// Pre-delta the i16 signal in-process, then encode with default cascade.
    DeltaDefault,
    /// Delta pre-pass then with_compact (so pco can also try).
    DeltaPco,
    /// Default cascade + our custom DeltaScheme (lets the cascade use Delta→BitPacking
    /// natively, no in-process pre-pass). This is the in-cascade equivalent of VBZ.
    DeltaSchemeOnly,
    /// DeltaScheme + with_compact (so pco is also a candidate).
    DeltaSchemePco,
}

impl Codec {
    pub fn label(self) -> &'static str {
        match self {
            Codec::Default => "vortex-default",
            Codec::Pco => "vortex-pco",
            Codec::DeltaDefault => "vortex-delta",
            Codec::DeltaPco => "vortex-delta+pco",
            Codec::DeltaSchemeOnly => "vortex-delta-scheme",
            Codec::DeltaSchemePco => "vortex-delta-scheme+pco",
        }
    }

    fn delta_pre_pass(self) -> bool {
        matches!(self, Codec::DeltaDefault | Codec::DeltaPco)
    }

    fn use_compact(self) -> bool {
        matches!(self, Codec::Pco | Codec::DeltaPco | Codec::DeltaSchemePco)
    }

    fn use_delta_scheme(self) -> bool {
        matches!(self, Codec::DeltaSchemeOnly | Codec::DeltaSchemePco)
    }
}

/// In-place delta transform: x[i] -> x[i] - x[i-1]. x[0] preserved.
fn delta_encode(samples: &mut [i16]) {
    for i in (1..samples.len()).rev() {
        samples[i] = samples[i].wrapping_sub(samples[i - 1]);
    }
}

/// Inverse of `delta_encode`. x[i] += x[i-1].
fn delta_decode(samples: &mut [i16]) {
    for i in 1..samples.len() {
        samples[i] = samples[i].wrapping_add(samples[i - 1]);
    }
}

/// Build a fresh session with the default Vortex encodings registered.
fn session() -> VortexSession {
    use vortex::array::scalar_fn::session::ScalarFnSession;
    use vortex::array::session::ArraySession;
    use vortex_io::session::RuntimeSession;
    use vortex_layout::session::LayoutSession;

    let session = VortexSession::empty()
        .with::<ArraySession>()
        .with::<LayoutSession>()
        .with::<ScalarFnSession>()
        .with::<RuntimeSession>();

    vortex_file::register_default_encodings(&session);
    session
}

fn btrblocks_for(codec: Codec) -> BtrBlocksCompressorBuilder {
    let mut b = BtrBlocksCompressorBuilder::default();
    if codec.use_compact() {
        b = b.with_compact();
    }
    if codec.use_delta_scheme() {
        b = b.with_new_scheme(&crate::delta_scheme::DeltaScheme);
    }
    b
}

/// Build the LayoutStrategy for the chosen codec.
fn write_strategy(codec: Codec) -> Arc<dyn vortex_layout::LayoutStrategy> {
    WriteStrategyBuilder::default()
        .with_btrblocks_builder(btrblocks_for(codec))
        .build()
}

/// EXPERIMENT A: report what encoding a single sample chunk produces.
/// Returns (encoding_id, compressed_bytes) for the configured codec.
pub fn inspect_encoding(samples: &[i16], codec: Codec) -> Result<(String, u64)> {
    let mut owned = samples.to_vec();
    if codec.delta_pre_pass() {
        delta_encode(&mut owned);
    }
    let buf: Buffer<i16> = Buffer::copy_from(owned.as_slice());
    let prim = PrimitiveArray::new(buf, Validity::NonNullable).into_array();

    let compressor: BtrBlocksCompressor = btrblocks_for(codec).build();
    let compressed = compressor.compress(&prim)?;
    Ok((compressed.encoding_id().to_string(), compressed.nbytes()))
}

/// Convert signals in a POD5 file to a single Vortex file using the given codec.
///
/// `max_reads = None` converts the whole file; `Some(N)` stops after N reads.
/// `concat_into_single_chunk = true` concatenates all reads into one chunk
/// (Experiment B: gives the cascade more data to choose a scheme from).
pub async fn convert_signal_only(
    input: impl AsRef<Path>,
    output: impl AsRef<Path>,
    codec: Codec,
    max_reads: Option<usize>,
    concat_into_single_chunk: bool,
) -> Result<(u64, usize, u64)> {
    let reader = Reader::open(input.as_ref())?;

    let mut chunks: Vec<ArrayRef> = Vec::new();
    let mut concat: Vec<i16> = Vec::new();
    let mut read_count = 0usize;
    let mut total_samples: u64 = 0;

    for read_result in reader.reads()? {
        if let Some(limit) = max_reads {
            if read_count >= limit {
                break;
            }
        }
        let read = read_result?;
        let mut samples: Vec<i16> = reader.get_signal(&read.signal_rows)?;
        total_samples += samples.len() as u64;

        if codec.delta_pre_pass() {
            delta_encode(&mut samples);
        }

        if concat_into_single_chunk {
            concat.extend_from_slice(&samples);
        } else {
            let buf: Buffer<i16> = Buffer::copy_from(samples.as_slice());
            chunks.push(PrimitiveArray::new(buf, Validity::NonNullable).into_array());
        }
        read_count += 1;
    }

    if concat_into_single_chunk {
        if concat.is_empty() {
            return Err(crate::Error::Other("no reads".into()));
        }
        let buf: Buffer<i16> = Buffer::copy_from(concat.as_slice());
        chunks.push(PrimitiveArray::new(buf, Validity::NonNullable).into_array());
    }

    if chunks.is_empty() {
        return Err(crate::Error::Other("no reads".into()));
    }

    let chunked = ChunkedArray::try_new(
        chunks,
        DType::Primitive(PType::I16, Nullability::NonNullable),
    )?
    .into_array();

    let session = session();
    let file = tokio::fs::File::create(output.as_ref()).await?;
    let _summary = VortexWriteOptions::new(session.clone())
        .with_strategy(write_strategy(codec))
        .write(file, chunked.to_array_stream())
        .await?;

    let size = tokio::fs::metadata(output.as_ref()).await?.len();
    Ok((size, read_count, total_samples))
}

/// Decode every i16 sample from a Vortex signal file. Forces canonical materialization.
/// If `delta_decode_after = true`, undoes the in-process delta pre-pass — required for
/// any codec where samples were delta-encoded before writing.
pub async fn decode_all(input: impl AsRef<Path>, delta_decode_after: bool) -> Result<u64> {
    use futures::StreamExt;
    use vortex::array::ToCanonical;

    let session = session();
    let bytes = tokio::fs::read(input.as_ref()).await?;
    let buf = ByteBuffer::copy_from(bytes.as_slice());
    let file = session.open_options().open_buffer(buf)?;
    let stream = file.scan()?.into_array_stream()?;
    futures::pin_mut!(stream);

    let mut total: u64 = 0;
    let mut checksum: i64 = 0;
    while let Some(chunk) = stream.next().await {
        let array = chunk?;
        #[allow(deprecated)]
        let prim = array.to_primitive();
        let slice = prim.as_slice::<i16>();
        total += slice.len() as u64;
        if delta_decode_after {
            // Have to copy to undo delta. Models the real cost of the pre-pass.
            let mut owned = slice.to_vec();
            delta_decode(&mut owned);
            for &v in owned.iter() {
                checksum = checksum.wrapping_add(v as i64);
            }
        } else {
            for &v in slice.iter() {
                checksum = checksum.wrapping_add(v as i64);
            }
        }
    }
    std::hint::black_box(checksum);
    Ok(total)
}

/// Convert signal to a Vortex file as `List<Int16>` — one row per read.
/// Enables row-range random access. Returns (size, read_count, total_samples).
pub async fn convert_signal_as_list(
    input: impl AsRef<Path>,
    output: impl AsRef<Path>,
    codec: Codec,
    max_reads: Option<usize>,
) -> Result<(u64, usize, u64)> {
    let reader = Reader::open(input.as_ref())?;

    let mut all_samples: Vec<i16> = Vec::new();
    let mut offsets: Vec<i64> = vec![0];
    let mut read_count = 0usize;

    for read_result in reader.reads()? {
        if let Some(limit) = max_reads {
            if read_count >= limit {
                break;
            }
        }
        let read = read_result?;
        let mut samples: Vec<i16> = reader.get_signal(&read.signal_rows)?;
        if codec.delta_pre_pass() {
            delta_encode(&mut samples);
        }
        all_samples.extend_from_slice(&samples);
        offsets.push(all_samples.len() as i64);
        read_count += 1;
    }

    let total_samples = all_samples.len() as u64;
    if read_count == 0 {
        return Err(crate::Error::Other("no reads".into()));
    }

    let elements = PrimitiveArray::new(
        Buffer::<i16>::copy_from(all_samples.as_slice()),
        Validity::NonNullable,
    )
    .into_array();
    let offsets_arr = PrimitiveArray::new(
        Buffer::<i64>::copy_from(offsets.as_slice()),
        Validity::NonNullable,
    )
    .into_array();
    let list = ListArray::try_new(elements, offsets_arr, Validity::NonNullable)?.into_array();

    let session = session();
    let file = tokio::fs::File::create(output.as_ref()).await?;
    let _summary = VortexWriteOptions::new(session.clone())
        .with_strategy(write_strategy(codec))
        .write(file, list.to_array_stream())
        .await?;

    let size = tokio::fs::metadata(output.as_ref()).await?.len();
    Ok((size, read_count, total_samples))
}

fn primitive_first_as_usize(prim: &PrimitiveArray) -> usize {
    use vortex::array::dtype::PType;
    match prim.ptype() {
        PType::U32 => prim.as_slice::<u32>()[0] as usize,
        PType::I32 => prim.as_slice::<i32>()[0] as usize,
        PType::U64 => prim.as_slice::<u64>()[0] as usize,
        PType::I64 => prim.as_slice::<i64>()[0] as usize,
        other => panic!("unsupported list offset/size ptype: {:?}", other),
    }
}

/// Random access on a List<i16> Vortex file: fetch the signal for each row index
/// using `with_row_range(i..i+1)`. Returns (samples_decoded, also_undelta_if_needed).
pub async fn random_access_list(
    input: impl AsRef<Path>,
    indices: &[u64],
    delta_decode_after: bool,
) -> Result<u64> {
    use futures::StreamExt;
    use vortex::array::ToCanonical;

    let session = session();
    let bytes = tokio::fs::read(input.as_ref()).await?;
    let buf = ByteBuffer::copy_from(bytes.as_slice());
    let file = session.open_options().open_buffer(buf)?;

    let mut total: u64 = 0;
    let mut checksum: i64 = 0;
    for &i in indices {
        let stream = file
            .scan()?
            .with_row_range(i..i + 1)
            .into_array_stream()?;
        futures::pin_mut!(stream);
        while let Some(chunk) = stream.next().await {
            let array = chunk?;
            // The chunk is List<Int16> with one row. Canonicalize to ListView,
            // then take the elements buffer slice for our single list row.
            #[allow(deprecated)]
            let listview = array.to_listview();
            let parts = listview.into_data_parts();
            #[allow(deprecated)]
            let off_prim = parts.offsets.to_primitive();
            #[allow(deprecated)]
            let sz_prim = parts.sizes.to_primitive();
            // Offsets/sizes carry their own primitive type; cast at runtime.
            let off = primitive_first_as_usize(&off_prim);
            let sz = primitive_first_as_usize(&sz_prim);
            #[allow(deprecated)]
            let prim = parts.elements.to_primitive();
            let full = prim.as_slice::<i16>();
            let slice = &full[off..off + sz];
            total += slice.len() as u64;
            if delta_decode_after {
                let mut owned = slice.to_vec();
                delta_decode(&mut owned);
                for &v in owned.iter() {
                    checksum = checksum.wrapping_add(v as i64);
                }
            } else {
                for &v in slice.iter() {
                    checksum = checksum.wrapping_add(v as i64);
                }
            }
        }
    }
    std::hint::black_box(checksum);
    Ok(total)
}

pub fn pod5_total_samples(input: impl AsRef<Path>) -> Result<u64> {
    let reader = Reader::open(input.as_ref())?;
    let mut total: u64 = 0;
    for read_result in reader.reads()? {
        let read = read_result?;
        total += read.num_samples;
    }
    Ok(total)
}

pub fn pod5_read_count(input: impl AsRef<Path>) -> Result<usize> {
    Ok(Reader::open(input.as_ref())?.read_count()?)
}
