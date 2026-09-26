//! `escpod align` — all-vs-all alignment of reads to a small reference panel
//! (tRNA), writing an input-ordered BAM with every input tag carried through.
//!
//! The DP, the kernels and the winner/tie rules live in `escapepod_align`;
//! this command is I/O and orchestration:
//!
//! ```text
//! reader thread ──batches──▶ dispatcher ──chunks──▶ rayon pool ──numbered chunks──▶ writer thread
//!   (BAM / FASTA / FASTQ)    (cuts ≤ 64 reads /    (score panel, tie set,       (reorders, BGZF,
//!                             ≤ 16 kb per chunk)    trace winners together,      input order)
//!                                                   build + encode records)
//! ```
//!
//! Memory is O(`--batch-size`): two batches may wait for the dispatcher, and
//! chunks worth two more may be in flight (spawned, not yet written). There
//! is no barrier between batches — see [`dispatch`] for the measurement that
//! removed it.
//!
//! # With the scoring on a GPU
//!
//! When `--device` places [`Stage::Align`](crate::device::Stage::Align) on the
//! GPU, one more thread sits between the reader and the dispatcher:
//!
//! ```text
//! reader ──batches──▶ GPU thread ──(batch, score matrix)──▶ dispatcher ──▶ rayon pool ──▶ writer
//!                     (encode every read, one kernel
//!                      launch per batch: all reads ×
//!                      all references)
//! ```
//!
//! It takes the reader's batches off the same bounded channel, scores a whole
//! batch in one launch, and hands the batch on with its `[reads × references]`
//! matrix behind an `Arc`. Each chunk carries the `Arc` and its offset into
//! the batch, and its worker passes its rows to
//! `Aligner::map_reads_scored` instead of scoring — everything after the
//! scores (the tie set, the pairs-kernel tracebacks, `MD`/`NM`, the records)
//! is the CPU path's own code, so the output is byte-identical. A read the
//! kernel does not take (over `escapepod_align::cuda::MAX_READ_LEN`, or
//! outside the `i16` bound) comes back unscored, and its worker scores it on
//! the CPU. The GPU thread works on batch *k + 1* while the pool finishes
//! batch *k*; the channel after it holds two, so memory stays O(`--batch-size`).
//!
//! # Long reads
//!
//! A tRNA sample's median read is ~130 nt, but real ones carry a few reads of
//! 50–400 kb. Two things keep them from stalling the pipeline. The aligner
//! scores any read of `simd::TRANSPOSED_MIN_READ_LEN` or more with its
//! reference-major kernel, whose state is one reference row per lane group
//! rather than one read column, so it stays in cache at any read length. And
//! a read of [`SPLIT_MIN_BASES`] or more — one that [`cut_chunks`] already
//! gives a chunk of its own — has its panel scored across the pool, one lane
//! group per rayon task ([`score_across_pool`]), and the row handed to
//! `Aligner::map_reads_scored` exactly as a GPU row is. Without that, the
//! chunk holding such a read ran for seconds on one thread while the ordered
//! writer waited for it and, once the permit window filled, the whole pool
//! with it. [`ROW_MAJOR_ONLY_ENV`] turns both off, for A/B measurement.
//!
//! # Where the per-read work runs
//!
//! Everything that scales with a read's tag payload runs inside the rayon
//! pool, not on the two I/O threads: the reader hands over raw
//! `bam::Record`s (just the bytes), the workers decode them to `RecordBuf`,
//! align, rewrite and *encode* the output record (through a `bam::io::Writer`
//! over a `Vec<u8>`, which writes the uncompressed BAM record), and the writer
//! thread only appends those bytes to the BGZF stream. A dorado uBAM record
//! carries ~5 kB of `mv`/`MM`/`ML` for a ~130 nt read, so decoding and
//! re-encoding it on one thread would cap the whole command well below the
//! kernels' rate.
//!
//! # Output rules
//!
//! * Records in input order (`@HD SO:unsorted`); `@SQ` from the reference in
//!   file order; the input's `@RG`, `@PG`, `@CO` copied; one `@PG` for this run
//!   chained to the last.
//! * Every input tag is copied byte for byte. `NM`, `MD`, `AS`, `XS`, `XA` are
//!   the aligner's and replace any the input carried.
//! * MAPQ is 60 for a unique best hit and 0 for a tie. `XA` lists the other tie
//!   members (`ref,±pos,CIGAR,NM;`, bwa's format); `--secondary` also writes
//!   them as 0x100 records, which carry `SEQ`/`QUAL` as `*` and only the
//!   alignment tags plus `RG` (minimap2's convention), so the move table and
//!   modification calls are not duplicated per tie.
//! * A read below `--min-score` is written unmapped (flag 4) with its tags,
//!   never dropped. Of the input flags only QC-fail (0x200) and duplicate
//!   (0x400) survive; everything else is the aligner's to set.

use anyhow::{Context, bail};
use clap::Args;
use escapepod_signal::parse_uuid_flexible;
use std::collections::BTreeMap;
use std::collections::HashSet;
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, Sender, SyncSender, channel, sync_channel};
use std::time::Instant;

use tracing::{debug, info, warn};

use noodles_bam as bam;
use noodles_bgzf as bgzf;
use noodles_sam as sam;
use sam::alignment::RecordBuf;
use sam::alignment::io::Write as _;
use sam::alignment::record::cigar::Op;
use sam::alignment::record::cigar::op::Kind;
use sam::alignment::record::data::field::Tag;
use sam::alignment::record::{Flags, MappingQuality};
use sam::alignment::record_buf::data::field::Value;
use sam::header::record::value::Map;
use sam::header::record::value::map::header::{Version, tag as hd_tag};
use sam::header::record::value::map::{self, Program, ReferenceSequence, program::tag as pg_tag};

use std::sync::Arc;

use escapepod_align::{
    Aligner, Backend, CigarOp, Hit, MapOptions, Mode, Panel, ReadMapping, ScoreMatrix, Scoring,
    sam as align_sam,
};
use rayon::prelude::*;

use crate::progress::create_spinner;
use crate::style;

#[derive(Args)]
pub struct AlignArgs {
    /// Reads: an unaligned BAM (dorado's uBAM), or FASTA/FASTQ, plain or gzip.
    /// Forward-strand aligned BAM records are accepted too; their alignment is
    /// discarded and they are aligned afresh
    #[arg(value_name = "READS")]
    pub reads: PathBuf,

    /// Reference panel FASTA, plain or gzip
    #[arg(short, long, value_name = "FASTA")]
    pub reference: PathBuf,

    /// Output BAM (`-` for stdout), records in input order
    #[arg(short, long, value_name = "BAM")]
    pub output: PathBuf,

    /// `local` (Smith-Waterman, overhangs soft-clipped) or `semiglobal`
    /// (overlap: leading/trailing gaps of either sequence free)
    #[arg(long, default_value = "local", value_parser = parse_mode, value_name = "MODE")]
    pub mode: Mode,

    /// MATCH,MISMATCH,GAP_OPEN,GAP_EXTEND. A gap of length k scores
    /// GAP_OPEN + (k-1)*GAP_EXTEND. bwa's `-A1 -B1 -O1 -E1` is `1,-1,-2,-1`
    #[arg(
        long,
        default_value = "2,-1,-10,-1",
        value_parser = parse_scoring,
        allow_hyphen_values = true,
        value_name = "M,X,O,E"
    )]
    pub scoring: Scoring,

    /// A read whose best score is below this is written unmapped
    #[arg(
        long,
        default_value_t = 0,
        allow_hyphen_values = true,
        value_name = "N"
    )]
    pub min_score: i32,

    /// Reads longer than this (nt) are not aligned: they are written unmapped,
    /// in order, with their tags. `0` = no limit
    #[arg(long, default_value_t = 1000, value_name = "N")]
    pub max_read_len: usize,

    /// At most this many tied references besides the primary go into `XA`
    /// (and `--secondary`); default all. MAPQ is 0 for any tie regardless
    #[arg(long, value_name = "N")]
    pub max_ties: Option<usize>,

    /// Also write each tied reference as a secondary (0x100) record
    #[arg(long)]
    pub secondary: bool,

    /// `forward` (direct RNA reads are sense) or `both` (also score the
    /// reverse complement; a reverse winner gets flag 16 with SEQ/QUAL
    /// reversed, per-base tags copied verbatim)
    #[arg(long, value_enum, default_value_t = StrandArg::Forward)]
    pub strand: StrandArg,

    /// Align only the reads named in FILE (one name per line), like
    /// `samtools view -N`; the rest are not written
    #[arg(long, value_name = "FILE")]
    pub read_ids: Option<PathBuf>,

    /// Reads per batch; memory is proportional to it
    #[arg(long, default_value_t = 50_000, value_name = "N")]
    pub batch_size: usize,

    /// Number of threads for parallel processing
    #[arg(short = 't', long, visible_short_alias = 'j', value_name = "N")]
    pub threads: Option<usize>,

    /// Panel scoring runs on a CUDA GPU under `auto` when a `gpu` build sees
    /// one; tracebacks and output stay on the CPU, and the output is identical
    #[command(flatten)]
    pub device: crate::device::DeviceArgs,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum StrandArg {
    Forward,
    Both,
}

fn parse_mode(s: &str) -> Result<Mode, String> {
    s.parse()
}

fn parse_scoring(s: &str) -> Result<Scoring, String> {
    s.parse()
        .map_err(|e: escapepod_align::AlignError| e.to_string())
}

/// Tags this command owns: any the input carried are replaced (or, on an
/// unmapped record, removed).
const OWNED_TAGS: [Tag; 5] = [
    Tag::EDIT_DISTANCE,
    Tag::MISMATCHED_POSITIONS,
    Tag::ALIGNMENT_SCORE,
    Tag::new(b'X', b'S'),
    Tag::new(b'X', b'A'),
];

// ---------------------------------------------------------------------------
// Input
// ---------------------------------------------------------------------------

/// One input read, as the reader thread hands it over.
enum InRecord {
    /// Raw BAM bytes, decoded by the worker.
    Bam(bam::Record),
    /// A FASTA/FASTQ read already built as an unmapped record.
    Buf(RecordBuf),
}

/// What the reader thread skipped rather than batched.
#[derive(Default, Debug)]
struct ReadCounts {
    /// Secondary/supplementary input records (not reads).
    not_primary: u64,
    /// Not in `--read-ids`.
    filtered: u64,
}

enum InputFormat {
    Bam,
    Fasta,
    Fastq,
}

/// Sniff the input: BGZF/gzip whose payload starts with `BAM\1`, or a
/// FASTA/FASTQ (plain or gzip) by its first character.
fn sniff(path: &Path) -> anyhow::Result<InputFormat> {
    let mut head = [0u8; 2];
    let mut f = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let n = f.read(&mut head)?;
    let gz = n == 2 && head == [0x1f, 0x8b];
    let mut reader: Box<dyn BufRead> = if gz {
        Box::new(BufReader::new(flate2::read::MultiGzDecoder::new(
            File::open(path)?,
        )))
    } else {
        Box::new(BufReader::new(File::open(path)?))
    };
    let buf = reader.fill_buf()?;
    if buf.starts_with(b"BAM\x01") {
        if !gz {
            bail!("{}: uncompressed BAM is not supported", path.display());
        }
        return Ok(InputFormat::Bam);
    }
    match buf.iter().find(|b| !b.is_ascii_whitespace()) {
        Some(b'>') => Ok(InputFormat::Fasta),
        Some(b'@') => Ok(InputFormat::Fastq),
        _ => bail!(
            "{}: not a BAM, FASTA or FASTQ file (plain or gzip)",
            path.display()
        ),
    }
}

/// Open a FASTA/FASTQ, transparently gunzipping.
fn open_text(path: &Path) -> anyhow::Result<Box<dyn BufRead + Send>> {
    let mut head = [0u8; 2];
    let n = File::open(path)
        .with_context(|| format!("opening {}", path.display()))?
        .read(&mut head)?;
    let f = File::open(path)?;
    Ok(if n == 2 && head == [0x1f, 0x8b] {
        Box::new(BufReader::with_capacity(
            1 << 20,
            flate2::read::MultiGzDecoder::new(f),
        ))
    } else {
        Box::new(BufReader::with_capacity(1 << 20, f))
    })
}

/// A minimal FASTA/FASTQ record reader: multi-line FASTA, four-line FASTQ.
struct FastxReader {
    inner: Box<dyn BufRead + Send>,
    fastq: bool,
    line: Vec<u8>,
    /// A FASTA header read while finishing the previous record.
    pending_header: Option<Vec<u8>>,
    line_no: u64,
}

/// `(name, sequence, qualities)` — qualities as raw Phred (not +33).
type FastxRecord = (Vec<u8>, Vec<u8>, Option<Vec<u8>>);

impl FastxReader {
    fn new(inner: Box<dyn BufRead + Send>, fastq: bool) -> Self {
        Self {
            inner,
            fastq,
            line: Vec::new(),
            pending_header: None,
            line_no: 0,
        }
    }

    /// Next line without its terminator; `None` at EOF.
    fn next_line(&mut self) -> std::io::Result<Option<&[u8]>> {
        self.line.clear();
        if self.inner.read_until(b'\n', &mut self.line)? == 0 {
            return Ok(None);
        }
        self.line_no += 1;
        while matches!(self.line.last(), Some(b'\n' | b'\r')) {
            self.line.pop();
        }
        Ok(Some(&self.line))
    }

    fn name_of(header: &[u8]) -> Vec<u8> {
        header
            .split(|b| b.is_ascii_whitespace())
            .next()
            .unwrap_or_default()
            .to_vec()
    }

    fn next_record(&mut self) -> anyhow::Result<Option<FastxRecord>> {
        if self.fastq {
            let header = loop {
                match self.next_line()? {
                    None => return Ok(None),
                    Some([]) => continue,
                    Some(l) => break l.to_vec(),
                }
            };
            let Some(h) = header.strip_prefix(b"@") else {
                bail!("FASTQ line {}: expected '@'", self.line_no);
            };
            let name = Self::name_of(h);
            let seq = self
                .next_line()?
                .context("FASTQ truncated after header")?
                .to_vec();
            match self.next_line()? {
                Some(l) if l.starts_with(b"+") => {}
                _ => bail!("FASTQ line {}: expected '+'", self.line_no),
            }
            let qual = self
                .next_line()?
                .context("FASTQ truncated before quality")?
                .to_vec();
            if qual.len() != seq.len() {
                bail!(
                    "FASTQ line {}: {} quality values for {} bases",
                    self.line_no,
                    qual.len(),
                    seq.len()
                );
            }
            let qual = qual.iter().map(|q| q.saturating_sub(33)).collect();
            Ok(Some((name, seq, Some(qual))))
        } else {
            let header = match self.pending_header.take() {
                Some(h) => h,
                None => loop {
                    match self.next_line()? {
                        None => return Ok(None),
                        Some([]) => continue,
                        Some(l) => break l.to_vec(),
                    }
                },
            };
            let Some(h) = header.strip_prefix(b">") else {
                bail!("FASTA line {}: expected '>'", self.line_no);
            };
            let name = Self::name_of(h);
            let mut seq = Vec::new();
            while let Some(l) = self.next_line()? {
                if l.starts_with(b">") {
                    self.pending_header = Some(l.to_vec());
                    break;
                }
                seq.extend(l.iter().filter(|b| !b.is_ascii_whitespace()));
            }
            Ok(Some((name, seq, None)))
        }
    }
}

/// A FASTX sequence as BAM can hold it: upper case, U as T, anything outside
/// the BAM alphabet as N.
fn normalise_seq(seq: &[u8]) -> Vec<u8> {
    seq.iter()
        .map(|&b| match b.to_ascii_uppercase() {
            b'U' => b'T',
            u if b"=ACMGRSVTWYHKDBN".contains(&u) => u,
            _ => b'N',
        })
        .collect()
}

fn fastx_to_record(name: Vec<u8>, seq: Vec<u8>, qual: Option<Vec<u8>>) -> RecordBuf {
    let mut b = RecordBuf::builder()
        .set_name(name)
        .set_flags(Flags::UNMAPPED)
        .set_sequence(normalise_seq(&seq).into());
    if let Some(q) = qual {
        b = b.set_quality_scores(q.into());
    }
    b.build()
}

/// Load the reference panel (FASTA, plain or gzip), in file order.
fn load_reference(path: &Path) -> anyhow::Result<Panel> {
    let refs = escapepod_align::fasta::read_fasta(open_text(path)?)
        .with_context(|| format!("reading reference {}", path.display()))?;
    Panel::new(refs).with_context(|| format!("reading reference {}", path.display()))
}

/// Canonicalise a read name for `--read-ids` matching, the way `escpod
/// filter`'s ID list does: a name that parses as a UUID — dashed or the
/// dashless 32-hex-char form `parse_uuid_flexible` also accepts — compares by
/// its canonical dashed lowercase string, so either spelling names the same
/// read. A name that is not a UUID (a FASTQ read name) compares as raw bytes.
fn canonical_read_name(name: &[u8]) -> Vec<u8> {
    std::str::from_utf8(name)
        .ok()
        .and_then(|s| parse_uuid_flexible(s).ok())
        .map(|uuid| uuid.to_string().into_bytes())
        .unwrap_or_else(|| name.to_vec())
}

fn load_read_ids(path: &Path) -> anyhow::Result<HashSet<Vec<u8>>> {
    let text = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let ids: HashSet<Vec<u8>> = text
        .split(|&b| b == b'\n')
        .filter_map(|l| {
            let l = l.trim_ascii();
            (!l.is_empty() && !l.starts_with(b"#")).then(|| {
                let name = l.split(|b| b.is_ascii_whitespace()).next().unwrap();
                canonical_read_name(name)
            })
        })
        .collect();
    if ids.is_empty() {
        bail!("no read names in {}", path.display());
    }
    Ok(ids)
}

/// The reader thread's body: batch input records until EOF.
fn read_batches(
    mut source: Source,
    wanted: Option<HashSet<Vec<u8>>>,
    batch_size: usize,
    tx: SyncSender<Vec<InRecord>>,
) -> anyhow::Result<ReadCounts> {
    let mut counts = ReadCounts::default();
    let mut batch = Vec::with_capacity(batch_size);
    let keep = |name: Option<&[u8]>, counts: &mut ReadCounts| match &wanted {
        None => true,
        Some(w) => {
            let ok = name.is_some_and(|n| w.contains(&canonical_read_name(n)));
            if !ok {
                counts.filtered += 1;
            }
            ok
        }
    };
    loop {
        let rec = match &mut source {
            Source::Bam(r) => {
                let mut rec = bam::Record::default();
                if r.read_record(&mut rec)? == 0 {
                    break;
                }
                let flags = rec.flags();
                if flags.is_secondary() || flags.is_supplementary() {
                    counts.not_primary += 1;
                    continue;
                }
                if !flags.is_unmapped() && flags.is_reverse_complemented() {
                    bail!(
                        "input record {} is aligned to the reverse strand; `escpod align` \
                         takes unaligned reads (or forward-strand alignments, whose SEQ is \
                         the read as sequenced)",
                        rec.name()
                            .map(|n| n.to_string())
                            .unwrap_or_else(|| "*".into())
                    );
                }
                if !keep(rec.name().map(|n| n.as_ref()), &mut counts) {
                    continue;
                }
                InRecord::Bam(rec)
            }
            Source::Fastx(r) => {
                let Some((name, seq, qual)) = r.next_record()? else {
                    break;
                };
                if !keep(Some(&name), &mut counts) {
                    continue;
                }
                InRecord::Buf(fastx_to_record(name, seq, qual))
            }
        };
        batch.push(rec);
        if batch.len() >= batch_size {
            let full = std::mem::replace(&mut batch, Vec::with_capacity(batch_size));
            if tx.send(full).is_err() {
                // The consumer stopped (it failed); its error is the one to report.
                return Ok(counts);
            }
        }
    }
    if !batch.is_empty() {
        let _ = tx.send(batch);
    }
    Ok(counts)
}

enum Source {
    Bam(bam::io::Reader<bgzf::io::MultithreadedReader<File>>),
    Fastx(FastxReader),
}

// ---------------------------------------------------------------------------
// Output records
// ---------------------------------------------------------------------------

/// An integer tag in the smallest BAM type that holds it, as htslib writes.
fn int_value(v: i64) -> Value {
    if let Ok(x) = u8::try_from(v) {
        Value::UInt8(x)
    } else if let Ok(x) = i8::try_from(v) {
        Value::Int8(x)
    } else if let Ok(x) = u16::try_from(v) {
        Value::UInt16(x)
    } else if let Ok(x) = i16::try_from(v) {
        Value::Int16(x)
    } else if let Ok(x) = u32::try_from(v) {
        Value::UInt32(x)
    } else {
        Value::Int32(v as i32)
    }
}

fn sam_cigar(hit: &Hit, read_len: usize) -> sam::alignment::record_buf::Cigar {
    let a = &hit.alignment;
    let mut ops = Vec::with_capacity(a.ops.len() + 2);
    if a.query_start > 0 {
        ops.push(Op::new(Kind::SoftClip, a.query_start));
    }
    for &(op, len) in &a.ops {
        let kind = match op {
            CigarOp::Match => Kind::Match,
            CigarOp::Ins => Kind::Insertion,
            CigarOp::Del => Kind::Deletion,
        };
        ops.push(Op::new(kind, len as usize));
    }
    if read_len > a.query_end {
        ops.push(Op::new(Kind::SoftClip, read_len - a.query_end));
    }
    ops.into()
}

/// bwa's `XA:Z:` entry: `ref,±pos,CIGAR,NM;`.
fn xa_entry(panel: &Panel, hit: &Hit, read_len: usize) -> String {
    let a = &hit.alignment;
    format!(
        "{},{}{},{},{};",
        panel.get(hit.reference).name,
        if hit.reverse { '-' } else { '+' },
        a.ref_start + 1,
        align_sam::cigar_string(&a.ops, read_len, a.query_start, a.query_end),
        hit.nm
    )
}

fn position(p0: usize) -> noodles_core::Position {
    noodles_core::Position::try_from(p0 + 1).expect("1-based position is non-zero")
}

/// Which way a read went, for the summary.
#[derive(Clone, Copy)]
enum Outcome {
    Unique,
    Tied,
    Unmapped,
}

/// Rewrite one input record into its output record(s).
fn build_records(
    mut rec: RecordBuf,
    m: &ReadMapping,
    panel: &Panel,
    secondary: bool,
) -> (Vec<RecordBuf>, Outcome) {
    let kept_flags = rec.flags() & (Flags::QC_FAIL | Flags::DUPLICATE);
    for t in &OWNED_TAGS {
        rec.data_mut().remove(t);
    }
    *rec.mate_reference_sequence_id_mut() = None;
    *rec.mate_alignment_start_mut() = None;
    *rec.template_length_mut() = 0;

    let Some(primary) = m.primary() else {
        *rec.flags_mut() = kept_flags | Flags::UNMAPPED;
        *rec.reference_sequence_id_mut() = None;
        *rec.alignment_start_mut() = None;
        *rec.mapping_quality_mut() = MappingQuality::new(0);
        *rec.cigar_mut() = Default::default();
        return (vec![rec], Outcome::Unmapped);
    };

    let read_len = rec.sequence().len();
    let score = m.best_score.expect("mapped reads have a score");
    let unique = m.n_tied == 1;
    let mut flags = kept_flags;
    if primary.reverse {
        flags |= Flags::REVERSE_COMPLEMENTED;
        let rc = escapepod_align::alphabet::reverse_complement(rec.sequence().as_ref());
        *rec.sequence_mut() = rc.into();
        let mut q: Vec<u8> = rec.quality_scores().as_ref().to_vec();
        q.reverse();
        *rec.quality_scores_mut() = q.into();
    }
    *rec.flags_mut() = flags;
    *rec.reference_sequence_id_mut() = Some(primary.reference);
    *rec.alignment_start_mut() = Some(position(primary.alignment.ref_start));
    *rec.mapping_quality_mut() = MappingQuality::new(if unique { 60 } else { 0 });
    *rec.cigar_mut() = sam_cigar(primary, read_len);

    let data = rec.data_mut();
    data.insert(Tag::EDIT_DISTANCE, int_value(primary.nm as i64));
    data.insert(
        Tag::MISMATCHED_POSITIONS,
        Value::String(primary.md.clone().into()),
    );
    data.insert(Tag::ALIGNMENT_SCORE, int_value(score as i64));
    if let Some(xs) = m.suboptimal {
        data.insert(Tag::new(b'X', b'S'), int_value(xs as i64));
    }
    let others = &m.hits[1..];
    if !others.is_empty() {
        let xa: String = others
            .iter()
            .map(|h| xa_entry(panel, h, read_len))
            .collect();
        data.insert(Tag::new(b'X', b'A'), Value::String(xa.into()));
    }

    let mut out = Vec::with_capacity(1 + if secondary { others.len() } else { 0 });
    if secondary {
        let rg = rec.data().get(&Tag::READ_GROUP).cloned();
        for h in others {
            let mut sec_flags = Flags::SECONDARY | kept_flags;
            if h.reverse {
                sec_flags |= Flags::REVERSE_COMPLEMENTED;
            }
            let mut d = sam::alignment::record_buf::Data::default();
            d.insert(Tag::EDIT_DISTANCE, int_value(h.nm as i64));
            d.insert(
                Tag::MISMATCHED_POSITIONS,
                Value::String(h.md.clone().into()),
            );
            d.insert(Tag::ALIGNMENT_SCORE, int_value(score as i64));
            if let Some(rg) = &rg {
                d.insert(Tag::READ_GROUP, rg.clone());
            }
            let mut b = RecordBuf::builder()
                .set_flags(sec_flags)
                .set_reference_sequence_id(h.reference)
                .set_alignment_start(position(h.alignment.ref_start))
                .set_mapping_quality(MappingQuality::new(0).expect("0 is a valid MAPQ"))
                .set_cigar(sam_cigar(h, read_len))
                .set_data(d);
            if let Some(n) = rec.name() {
                b = b.set_name(n.to_vec());
            }
            out.push(b.build());
        }
    }
    out.insert(0, rec);
    (
        out,
        if unique {
            Outcome::Unique
        } else {
            Outcome::Tied
        },
    )
}

/// Everything a worker needs, shared read-only across the pool.
struct Job<'a> {
    aligner: &'a Aligner,
    opts: MapOptions,
    in_header: &'a sam::Header,
    out_header: &'a sam::Header,
    secondary: bool,
    /// Score a read of [`SPLIT_MIN_BASES`] or more across the pool.
    split_long: bool,
    /// Read strands scored that way, for `-v`.
    split_strands: AtomicU64,
    /// `--max-read-len`: a longer read is written unmapped, never scored
    /// (`0` = no limit).
    max_read_len: usize,
    /// Reads written unmapped for `--max-read-len`.
    skipped_long: AtomicU64,
}

/// Whether `--max-read-len` keeps a read of `len` bases out of alignment.
fn over_limit(max_read_len: usize, len: usize) -> bool {
    max_read_len != 0 && len > max_read_len
}

/// `=1` puts every read back on the row-major score kernel and on one
/// thread per chunk — `escpod align` as it was before the long-read path
/// (#415). An A/B lever: the output is the same either way.
const ROW_MAJOR_ONLY_ENV: &str = "ESCAPEPOD_ALIGN_ROW_MAJOR_ONLY";

/// Reads per unit of rayon work. The winners of a chunk's reads are traced
/// together, so this is also how full the traceback kernels' lanes get.
const CHUNK: usize = 64;

/// Bases per unit of rayon work, so one very long read makes its own chunk.
const CHUNK_BASES: usize = 64 * 256;

/// A read at least this long is always a chunk of its own (see
/// [`cut_chunks`]), and has its panel scored across the pool.
const SPLIT_MIN_BASES: usize = CHUNK_BASES;

/// Each read's output records, BAM-encoded, and how it went.
type Encoded = Vec<(Vec<u8>, Outcome)>;

/// One batch's panel scores from the GPU: the forward strand of every read,
/// then (with `--strand both`) every reverse complement.
struct BatchScores {
    matrix: ScoreMatrix,
    reads: usize,
}

impl BatchScores {
    /// Read `k`'s row for one strand, `None` if the GPU left it to the CPU.
    fn row(&self, k: usize, reverse: bool) -> Option<&[i16]> {
        self.matrix.row(if reverse { self.reads + k } else { k })
    }
}

/// A batch as the dispatcher receives it: scored on the GPU, or not.
type Batch = (Vec<InRecord>, Option<Arc<BatchScores>>);

/// A chunk's share of its batch's scores: the matrix and the chunk's first read.
type ChunkScores = Option<(Arc<BatchScores>, usize)>;

/// A chunk of reads, start to finish on a worker: decode, align, rewrite,
/// encode.
fn process_chunk(
    ctx: &Job<'_>,
    chunk: Vec<InRecord>,
    scores: ChunkScores,
) -> anyhow::Result<Encoded> {
    let records = chunk
        .into_iter()
        .map(|rec| match rec {
            InRecord::Bam(raw) => Ok(RecordBuf::try_from_alignment_record(ctx.in_header, &raw)?),
            InRecord::Buf(r) => Ok(r),
        })
        .collect::<anyhow::Result<Vec<RecordBuf>>>()?;
    // Reads over --max-read-len are not scored or traced anywhere; the rest
    // are aligned as one batch, `kept[k]` being read `k`'s place in the chunk.
    let kept: Vec<usize> = (0..records.len())
        .filter(|&k| !over_limit(ctx.max_read_len, records[k].sequence().len()))
        .collect();
    let skipped = (records.len() - kept.len()) as u64;
    if skipped > 0 {
        ctx.skipped_long.fetch_add(skipped, Ordering::Relaxed);
    }
    let seqs: Vec<&[u8]> = kept
        .iter()
        .map(|&k| records[k].sequence().as_ref())
        .collect();
    let gpu_row = |k: usize, reverse: bool| {
        let k = kept[k];
        scores
            .as_ref()
            .and_then(|(batch, first)| batch.row(first + k, reverse))
    };
    // A long read's strands the GPU did not score, scored here across the
    // pool: [forward, reverse] per read.
    let pooled: Vec<[Option<Vec<i16>>; 2]> = seqs
        .iter()
        .enumerate()
        .map(|(k, seq)| {
            if !ctx.split_long || seq.len() < SPLIT_MIN_BASES {
                return [None, None];
            }
            let fwd = escapepod_align::alphabet::encode(seq);
            let mut out = [None, None];
            if gpu_row(k, false).is_none() {
                out[0] = score_across_pool(ctx.aligner, &fwd);
            }
            if ctx.opts.both_strands && gpu_row(k, true).is_none() {
                let rev = escapepod_align::alphabet::reverse_complement_codes(&fwd);
                out[1] = score_across_pool(ctx.aligner, &rev);
            }
            let n = out.iter().filter(|r| r.is_some()).count() as u64;
            ctx.split_strands.fetch_add(n, Ordering::Relaxed);
            out
        })
        .collect();
    let aligned = ctx
        .aligner
        .map_reads_scored(&seqs, &ctx.opts, |k, reverse| {
            pooled[k][usize::from(reverse)]
                .as_deref()
                .or_else(|| gpu_row(k, reverse))
        });
    // Back into chunk order, with a skipped read as an unmapped mapping —
    // written exactly as a read below --min-score is.
    let mut mappings: Vec<ReadMapping> = (0..records.len())
        .map(|_| ReadMapping {
            best_score: None,
            n_tied: 0,
            suboptimal: None,
            hits: Vec::new(),
        })
        .collect();
    for (&k, m) in kept.iter().zip(aligned) {
        mappings[k] = m;
    }
    records
        .into_iter()
        .zip(&mappings)
        .map(|(rec, mapping)| {
            let (out, outcome) = build_records(rec, mapping, ctx.aligner.panel(), ctx.secondary);
            let mut w = bam::io::Writer::from(Vec::with_capacity(8 << 10));
            for r in &out {
                w.write_alignment_record(ctx.out_header, r)?;
            }
            Ok((w.into_inner(), outcome))
        })
        .collect()
}

/// One read strand's panel scores (codes in), computed one lane group per
/// rayon task — the row `Aligner::score_all` would give, as `i16` for
/// `map_reads_scored`. `None` outside the `i16` bound, where the aligner's
/// own scalar fallback must score it.
fn score_across_pool(aligner: &Aligner, codes: &[u8]) -> Option<Vec<i16>> {
    let panel = aligner.panel();
    if !escapepod_align::simd::fits_i16(codes.len(), panel.max_len(), aligner.scoring()) {
        return None;
    }
    // Known hazard, left as is: while this chunk's task waits here for its
    // groups, rayon's work-stealing may run an unrelated, later chunk on the
    // waiting thread — possibly another long read's — which delays this
    // chunk (and so the ordered writer) until that one finishes, and counts
    // the stolen chunk's time twice in `-v`'s workers-busy total.
    let units: Vec<Vec<(usize, i32)>> = (0..aligner.score_groups(codes.len()))
        .into_par_iter()
        .map(|g| {
            let mut unit = Vec::new();
            aligner.score_group(codes, g, &mut unit);
            unit
        })
        .collect();
    let mut row = vec![0i16; panel.len()];
    for (r, s) in units.into_iter().flatten() {
        row[r] = i16::try_from(s).expect("fits_i16 bounds every score");
    }
    Some(row)
}

// ---------------------------------------------------------------------------
// Header
// ---------------------------------------------------------------------------

fn build_header(panel: &Panel, input: &sam::Header) -> anyhow::Result<sam::Header> {
    let mut hd = Map::<map::Header>::new(Version::new(1, 6));
    hd.other_fields_mut()
        .insert(hd_tag::SORT_ORDER, "unsorted".into());
    let mut b = sam::Header::builder().set_header(hd);
    for r in panel.references() {
        let len = std::num::NonZero::new(r.seq.len()).expect("panel refuses empty references");
        b = b.add_reference_sequence(r.name.clone(), Map::<ReferenceSequence>::new(len));
    }
    let mut header = b.build();
    *header.read_groups_mut() = input.read_groups().clone();
    *header.programs_mut() = input.programs().clone();
    *header.comments_mut() = input.comments().to_vec();
    let cl: Vec<String> = std::env::args().collect();
    let pg = Map::<Program>::builder()
        .insert(pg_tag::NAME, "escpod")
        .insert(pg_tag::VERSION, env!("CARGO_PKG_VERSION"))
        .insert(pg_tag::COMMAND_LINE, cl.join(" "))
        .build()?;
    header.programs_mut().add("escpod-align", pg)?;
    Ok(header)
}

// ---------------------------------------------------------------------------
// Run
// ---------------------------------------------------------------------------

#[derive(Default)]
struct Summary {
    reads: u64,
    unique: u64,
    tied: u64,
    unmapped: u64,
}

pub fn run(args: AlignArgs) -> anyhow::Result<()> {
    let started = Instant::now();
    let device = args.device.resolve();
    // Decided and reported before anything is read, so a missing feature or
    // device is the first line of the run, not a surprise at the end of it.
    let placement = crate::device::place_and_report(device, crate::device::Stage::Align)?;
    if args.batch_size == 0 {
        bail!("--batch-size must be at least 1");
    }

    // --- Reference panel + kernel ------------------------------------------
    let panel = load_reference(&args.reference)?;
    info!(
        "reference: {} sequences, {} bases (longest {}) from {}",
        style::count(panel.len()),
        style::count(panel.total_len()),
        panel.max_len(),
        style::path(args.reference.display()),
    );
    if panel.len() > 10_000 || panel.max_len() > 1_000 {
        warn!(
            "`escpod align` scores every read against every reference with no seeding; \
             it is built for small panels (<= ~10k references of <= ~1 kb) and runs \
             proportionally slower beyond that"
        );
    }
    if let Err(v) = Backend::cap_from_env() {
        warn!(
            "{}={v}: not a kernel name (scalar|avx2|avx512), ignored",
            escapepod_align::simd::BACKEND_ENV
        );
    }
    let row_major_only = escapepod_signal::pod5::env::flag(ROW_MAJOR_ONLY_ENV);
    let mut aligner = Aligner::new(panel, args.scoring, args.mode)?;
    if row_major_only {
        aligner = aligner.with_transposed_min_len(None);
        info!(
            "{ROW_MAJOR_ONLY_ENV} is set: every read on the row-major kernel, long reads on \
             one thread each"
        );
    }
    info!(
        "{} mode, scoring {} (match,mismatch,gap_open,gap_extend), {} strand{}, kernel {} ({} lanes)",
        args.mode.name(),
        args.scoring,
        match args.strand {
            StrandArg::Forward => "forward",
            StrandArg::Both => "both",
        },
        if args.strand == StrandArg::Both {
            "s"
        } else {
            ""
        },
        aligner.backend().name(),
        aligner.backend().lanes(),
    );
    let opts = MapOptions {
        min_score: args.min_score,
        max_ties: args.max_ties,
        both_strands: args.strand == StrandArg::Both,
    };
    let gpu_scorer = if placement.is_gpu() {
        gpu::start(&aligner, device)?
    } else {
        None
    };

    // --- Input -------------------------------------------------------------
    let wanted = args.read_ids.as_deref().map(load_read_ids).transpose()?;
    if let Some(w) = &wanted {
        info!(
            "aligning only the {} reads named in --read-ids",
            style::count(w.len())
        );
    }
    let threads = rayon::current_num_threads().max(1);
    let (source, in_header) = match sniff(&args.reads)? {
        InputFormat::Bam => {
            let workers = std::num::NonZero::new(threads.div_ceil(4)).expect("at least one");
            let file = File::open(&args.reads)?;
            let mut r = bam::io::Reader::from(bgzf::io::MultithreadedReader::with_worker_count(
                workers, file,
            ));
            let h = r
                .read_header()
                .with_context(|| format!("reading BAM header of {}", args.reads.display()))?;
            (Source::Bam(r), h)
        }
        InputFormat::Fasta => (
            Source::Fastx(FastxReader::new(open_text(&args.reads)?, false)),
            sam::Header::default(),
        ),
        InputFormat::Fastq => (
            Source::Fastx(FastxReader::new(open_text(&args.reads)?, true)),
            sam::Header::default(),
        ),
    };
    let out_header = build_header(aligner.panel(), &in_header)?;

    // --- Output ------------------------------------------------------------
    let sink: Box<dyn Write + Send> = if args.output.as_os_str() == "-" {
        Box::new(std::io::stdout())
    } else {
        Box::new(
            File::create(&args.output)
                .with_context(|| format!("creating {}", args.output.display()))?,
        )
    };
    let writer_workers = std::num::NonZero::new(threads).expect("at least one");
    let mut writer = bam::io::Writer::from(bgzf::io::MultithreadedWriter::with_worker_count(
        writer_workers,
        sink,
    ));
    writer.write_header(&out_header)?;

    // --- Pipeline ----------------------------------------------------------
    let (in_tx, in_rx) = sync_channel::<Vec<InRecord>>(2);
    let (res_tx, res_rx) = channel::<(u64, anyhow::Result<Encoded>)>();
    // Chunks in flight — spawned and not yet written — are capped at two
    // batches' worth of reads: the memory bound, and the distance a slow
    // chunk can fall behind before the dispatcher waits for it.
    let max_in_flight = (2 * args.batch_size).div_ceil(CHUNK).max(2);
    let (permit_tx, permit_rx) = sync_channel::<()>(max_in_flight);
    for _ in 0..max_in_flight {
        permit_tx.send(()).expect("receiver is alive");
    }
    let batch_size = args.batch_size;
    let reader = std::thread::Builder::new()
        .name("escpod-align-reader".into())
        .spawn(move || read_batches(source, wanted, batch_size, in_tx))?;
    // With a GPU, its thread sits between the reader and the dispatcher; the
    // channel after it is as shallow as the one before, for the same memory
    // bound.
    let (batches, gpu_thread): (Box<dyn Iterator<Item = Batch>>, _) = match gpu_scorer {
        Some(scorer) => {
            let (tx, rx) = sync_channel::<Batch>(2);
            let both = opts.both_strands;
            let max_read_len = args.max_read_len;
            let t = std::thread::Builder::new()
                .name("escpod-align-gpu".into())
                .spawn(move || gpu::score_batches(scorer, in_rx, tx, both, max_read_len))?;
            (Box::new(rx.into_iter()), Some(t))
        }
        None => (Box::new(in_rx.into_iter().map(|b| (b, None))), None),
    };
    let spinner = create_spinner("aligning")?;
    let writer_spinner = spinner.clone();
    let writer_thread = std::thread::Builder::new()
        .name("escpod-align-writer".into())
        .spawn(move || write_in_order(writer, res_rx, permit_tx, &writer_spinner, started))?;

    let ctx = Job {
        aligner: &aligner,
        opts,
        in_header: &in_header,
        out_header: &out_header,
        secondary: args.secondary,
        split_long: !row_major_only,
        split_strands: AtomicU64::new(0),
        max_read_len: args.max_read_len,
        skipped_long: AtomicU64::new(0),
    };
    let busy = dispatch(&ctx, batches, res_tx, permit_rx);
    // Join every thread before deciding which error to report: a failed stage
    // drops its channel ends, which unblocks the others.
    let gpu_result = gpu_thread
        .map(|t| t.join().map_err(|_| anyhow::anyhow!("GPU thread panicked")))
        .transpose()?;
    let read_result = reader
        .join()
        .map_err(|_| anyhow::anyhow!("reader thread panicked"))?;
    let write_result = writer_thread
        .join()
        .map_err(|_| anyhow::anyhow!("writer thread panicked"))?;
    spinner.finish_and_clear();
    // A GPU failure ends the input early, and the writer then finishes a
    // truncated file without complaint: report it first.
    if let Some(r) = gpu_result {
        let stats = r.context("scoring on the GPU")?;
        info!(
            "GPU scored {} of {} {} ({:.1} s waiting on the device); the other {} (longer \
             than {} nt, or outside the i16 bound) were scored on the CPU",
            style::count(stats.scored),
            style::count(stats.queries),
            if opts.both_strands {
                "read strands"
            } else {
                "reads"
            },
            stats.device_secs,
            style::count(stats.queries - stats.scored),
            gpu::MAX_READ_LEN,
        );
    }
    let summary = write_result.with_context(|| format!("writing {}", args.output.display()))?;
    let counts = read_result.with_context(|| format!("reading {}", args.reads.display()))?;
    let skipped_long = ctx.skipped_long.load(Ordering::Relaxed);
    debug!(
        "workers were busy {:.1} s of {:.1} s x {} threads; {} read strands of {}+ nt \
         scored across the pool; {} reads over --max-read-len not aligned",
        busy.as_secs_f64(),
        started.elapsed().as_secs_f64(),
        threads,
        ctx.split_strands.load(Ordering::Relaxed),
        SPLIT_MIN_BASES,
        skipped_long,
    );

    // --- Summary -----------------------------------------------------------
    if counts.not_primary > 0 {
        info!(
            "skipped {} secondary/supplementary input records (not reads)",
            style::count(counts.not_primary)
        );
    }
    if counts.filtered > 0 {
        info!(
            "skipped {} reads not named in --read-ids",
            style::count(counts.filtered)
        );
    }
    if skipped_long > 0 {
        info!(
            "{} reads longer than --max-read-len {} nt were not aligned (written unmapped)",
            style::count(skipped_long),
            args.max_read_len
        );
    }
    let wall = started.elapsed().as_secs_f64();
    let mapped = summary.unique + summary.tied;
    let pct = |n: u64| 100.0 * n as f64 / summary.reads.max(1) as f64;
    info!(
        "{} reads: {} mapped ({:.1}%) — {} unique, {} tied — {} unmapped; {:.1} s ({:.0} reads/s)",
        style::count(summary.reads),
        style::count(mapped),
        pct(mapped),
        style::count(summary.unique),
        style::count(summary.tied),
        style::count(summary.unmapped),
        wall,
        summary.reads as f64 / wall.max(1e-9),
    );
    if args.output.as_os_str() != "-" {
        info!("wrote {}", style::path(args.output.display()));
    }
    Ok(())
}

/// Cut a batch into chunks of at most [`CHUNK`] reads or [`CHUNK_BASES`]
/// bases, whichever comes first — so a read of hundreds of kilobases (they
/// occur: up to ~400 kb in a real tRNA sample) is a chunk of its own rather
/// than the tail that sixty-three short reads wait behind.
fn cut_chunks(batch: Vec<InRecord>) -> Vec<Vec<InRecord>> {
    let mut chunks = Vec::new();
    let mut cur = Vec::with_capacity(CHUNK);
    let mut bases = 0usize;
    for rec in batch {
        let len = match &rec {
            InRecord::Bam(r) => r.sequence().len(),
            InRecord::Buf(r) => r.sequence().len(),
        };
        if !cur.is_empty() && (cur.len() >= CHUNK || bases + len > CHUNK_BASES) {
            chunks.push(std::mem::replace(&mut cur, Vec::with_capacity(CHUNK)));
            bases = 0;
        }
        bases += len;
        cur.push(rec);
    }
    if !cur.is_empty() {
        chunks.push(cur);
    }
    chunks
}

/// The dispatcher, on the calling thread: cut each batch into chunks and
/// spawn every chunk onto the rayon pool as soon as a permit allows, numbering
/// them so the writer can restore input order. Returns the workers' total busy
/// time.
///
/// There is deliberately no barrier between batches. Read lengths in a real
/// sample run to hundreds of kilobases, and a chunk holding one costs seconds
/// where a typical chunk costs milliseconds; waiting for each batch to finish
/// before starting the next left half the pool idle behind such stragglers
/// (measured: 16 of 32 threads busy on average). Now a slow chunk only holds
/// back the *writing* of what follows it, and only once two batches' worth
/// of later chunks are done does the dispatcher wait for it.
fn dispatch(
    ctx: &Job<'_>,
    batches: impl Iterator<Item = Batch>,
    res_tx: Sender<(u64, anyhow::Result<Encoded>)>,
    permit_rx: Receiver<()>,
) -> std::time::Duration {
    let busy_ns = AtomicU64::new(0);
    rayon::in_place_scope(|scope| {
        let mut seq = 0u64;
        'batches: for (batch, scores) in batches {
            let mut first = 0usize;
            for chunk in cut_chunks(batch) {
                if permit_rx.recv().is_err() {
                    // The writer stopped; its error is reported after the join.
                    break 'batches;
                }
                let chunk_scores = scores.as_ref().map(|s| (Arc::clone(s), first));
                first += chunk.len();
                let tx = res_tx.clone();
                let busy_ns = &busy_ns;
                let this = seq;
                scope.spawn(move |_| {
                    let t0 = Instant::now();
                    let r = process_chunk(ctx, chunk, chunk_scores);
                    busy_ns.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    let _ = tx.send((this, r));
                });
                seq += 1;
            }
        }
        drop(res_tx);
    });
    std::time::Duration::from_nanos(busy_ns.load(Ordering::Relaxed))
}

/// The writer thread's body: take encoded chunks as workers finish them,
/// append them to the BGZF stream in input order, and return a permit for
/// each chunk written.
fn write_in_order(
    writer: bam::io::Writer<bgzf::io::MultithreadedWriter<Box<dyn Write + Send>>>,
    rx: Receiver<(u64, anyhow::Result<Encoded>)>,
    permits: SyncSender<()>,
    spinner: &indicatif::ProgressBar,
    started: Instant,
) -> anyhow::Result<Summary> {
    let mut bgzf = writer.into_inner();
    let mut summary = Summary::default();
    let mut pending: BTreeMap<u64, Encoded> = BTreeMap::new();
    let mut next = 0u64;
    let mut last_report = Instant::now();
    for (seq, result) in rx {
        pending.insert(seq, result?);
        while let Some(encoded) = pending.remove(&next) {
            for (bytes, outcome) in encoded {
                bgzf.write_all(&bytes)?;
                summary.reads += 1;
                match outcome {
                    Outcome::Unique => summary.unique += 1,
                    Outcome::Tied => summary.tied += 1,
                    Outcome::Unmapped => summary.unmapped += 1,
                }
            }
            next += 1;
            // The dispatcher may be gone already (it stops at end of input);
            // a permit nobody will take is not an error.
            let _ = permits.try_send(());
        }
        if last_report.elapsed().as_millis() >= 500 {
            last_report = Instant::now();
            let secs = started.elapsed().as_secs_f64().max(1e-9);
            spinner.set_message(format!(
                "{} reads ({:.0}/s)",
                style::count(summary.reads),
                summary.reads as f64 / secs
            ));
        }
    }
    if !pending.is_empty() {
        bail!("{} chunks were aligned but never written", pending.len());
    }
    bgzf.finish()?;
    Ok(summary)
}

// ---------------------------------------------------------------------------
// GPU scoring
// ---------------------------------------------------------------------------

/// The GPU stage: building the scorer, and the thread that feeds it.
///
/// Without the `gpu` feature `place_and_report` never returns a GPU placement
/// (it errors under `--device gpu` and reports the CPU under `auto`), so
/// [`gpu::start`] is unreachable there and its stand-in only has to exist.
mod gpu {
    use super::*;

    #[cfg(feature = "gpu")]
    pub(super) use escapepod_align::cuda::{GpuScorer, MAX_READ_LEN};

    /// Stand-in so the pipeline compiles identically without the feature;
    /// uninhabited, so nothing can construct one.
    #[cfg(not(feature = "gpu"))]
    pub(super) enum GpuScorer {}

    #[cfg(not(feature = "gpu"))]
    pub(super) const MAX_READ_LEN: usize = 0;

    /// What the GPU thread did, for the summary.
    #[cfg_attr(not(feature = "gpu"), allow(dead_code))]
    pub(super) struct GpuStats {
        /// Queries sent (reads, times two with `--strand both`).
        pub(super) queries: u64,
        /// Of those, scored on the device.
        pub(super) scored: u64,
        /// Seconds the thread spent blocked on the device (uploads, and
        /// whatever of a kernel and its download was not hidden behind the
        /// next batch's packing).
        pub(super) device_secs: f64,
    }

    /// Build the scorer for a GPU placement. `Ok(None)` is a fall-back to the
    /// CPU under `--device auto` (said in the log); under `--device gpu` the
    /// same failure is an error.
    #[cfg(feature = "gpu")]
    pub(super) fn start(
        aligner: &Aligner,
        device: crate::device::Device,
    ) -> anyhow::Result<Option<GpuScorer>> {
        let t0 = Instant::now();
        match GpuScorer::new(aligner.panel(), *aligner.scoring(), aligner.mode()) {
            Ok(s) => {
                let (lanes, groups, blocks) = s.geometry();
                info!(
                    "GPU: {} — {groups} reference group{} x {blocks} blocks of {lanes} threads, \
                     kernel ready in {:.1} s; reads over {MAX_READ_LEN} nt are scored on the CPU",
                    s.device_name(),
                    if groups == 1 { "" } else { "s" },
                    t0.elapsed().as_secs_f64(),
                );
                Ok(Some(s))
            }
            Err(e) if device == crate::device::Device::Gpu => Err(anyhow::anyhow!(
                "--device gpu cannot run read alignment scoring: {e}"
            )),
            Err(e) => {
                warn!(
                    "read alignment scoring cannot use the GPU ({e}); scoring on the CPU instead"
                );
                Ok(None)
            }
        }
    }

    #[cfg(not(feature = "gpu"))]
    pub(super) fn start(
        _aligner: &Aligner,
        _device: crate::device::Device,
    ) -> anyhow::Result<Option<GpuScorer>> {
        unreachable!("a build without `gpu` never places a stage on the GPU")
    }

    /// The letters of one input read, as the worker will see them.
    #[cfg(feature = "gpu")]
    fn letters(rec: &InRecord, out: &mut Vec<u8>) {
        out.clear();
        match rec {
            InRecord::Bam(r) => out.extend(r.sequence().iter()),
            InRecord::Buf(r) => out.extend_from_slice(r.sequence().as_ref()),
        }
    }

    /// The GPU thread's body: score each batch whole, pass it on with its
    /// matrix. Ends when the reader does, or when the dispatcher stops
    /// taking batches (its error is the one reported).
    #[cfg(feature = "gpu")]
    pub(super) fn score_batches(
        mut scorer: GpuScorer,
        rx: Receiver<Vec<InRecord>>,
        tx: SyncSender<Batch>,
        both_strands: bool,
        max_read_len: usize,
    ) -> anyhow::Result<GpuStats> {
        use escapepod_align::alphabet::{encode_base, reverse_complement_codes};
        let mut stats = GpuStats {
            queries: 0,
            scored: 0,
            device_secs: 0.0,
        };
        let mut buf = Vec::new();
        let (mut wait_in, mut prep, mut wait_gpu, mut wait_out) = (0.0, 0.0, 0.0, 0.0);
        // One batch in flight on the device while the next is encoded and
        // packed here, so the kernel is not idle between batches.
        let mut in_flight: Option<(Vec<InRecord>, escapepod_align::cuda::PendingBatch)> = None;
        let mut t = Instant::now();
        let mut rx = rx.into_iter();
        loop {
            let next = rx.next();
            wait_in += t.elapsed().as_secs_f64();
            t = Instant::now();
            let prepared = next.as_ref().map(|batch| {
                // A read over --max-read-len goes as an empty query, which
                // the kernel leaves unscored; the worker never asks for it.
                let mut skipped = 0u64;
                let mut codes: Vec<Vec<u8>> = batch
                    .iter()
                    .map(|rec| {
                        letters(rec, &mut buf);
                        if over_limit(max_read_len, buf.len()) {
                            skipped += 1;
                            return Vec::new();
                        }
                        buf.iter().map(|&b| encode_base(b)).collect()
                    })
                    .collect();
                if both_strands {
                    skipped *= 2;
                }
                if both_strands {
                    let rev: Vec<Vec<u8>> =
                        codes.iter().map(|c| reverse_complement_codes(c)).collect();
                    codes.extend(rev);
                }
                let queries: Vec<&[u8]> = codes.iter().map(Vec::as_slice).collect();
                stats.queries += queries.len() as u64 - skipped;
                scorer.prepare(&queries)
            });
            prep += t.elapsed().as_secs_f64();
            t = Instant::now();
            // Collect the batch before this one, then queue this one: the
            // download waits for its own kernel only.
            let done = match in_flight.take() {
                Some((batch, pending)) => Some((batch, scorer.finish(pending)?)),
                None => None,
            };
            if let (Some(batch), Some(prepared)) = (next, prepared) {
                in_flight = Some((batch, scorer.submit(prepared)?));
            }
            wait_gpu += t.elapsed().as_secs_f64();
            t = Instant::now();
            let Some((batch, matrix)) = done else {
                if in_flight.is_none() {
                    break;
                }
                continue;
            };
            stats.scored += matrix.n_scored() as u64;
            let scores = BatchScores {
                matrix,
                reads: batch.len(),
            };
            if tx.send((batch, Some(Arc::new(scores)))).is_err() {
                break;
            }
            wait_out += t.elapsed().as_secs_f64();
            t = Instant::now();
            if in_flight.is_none() {
                break;
            }
        }
        stats.device_secs = wait_gpu;
        debug!(
            "GPU thread: {wait_in:.1} s waiting for the reader, {prep:.1} s encoding and \
             packing, {wait_gpu:.1} s waiting on the device, {wait_out:.1} s waiting for the \
             workers"
        );
        Ok(stats)
    }

    #[cfg(not(feature = "gpu"))]
    pub(super) fn score_batches(
        scorer: GpuScorer,
        _rx: Receiver<Vec<InRecord>>,
        _tx: SyncSender<Batch>,
        _both_strands: bool,
        _max_read_len: usize,
    ) -> anyhow::Result<GpuStats> {
        match scorer {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn int_tags_take_the_smallest_type() {
        assert_eq!(int_value(3), Value::UInt8(3));
        assert_eq!(int_value(-3), Value::Int8(-3));
        assert_eq!(int_value(300), Value::UInt16(300));
        assert_eq!(int_value(-300), Value::Int16(-300));
        assert_eq!(int_value(70_000), Value::UInt32(70_000));
    }

    #[test]
    fn long_reads_get_chunks_of_their_own() {
        let read =
            |len: usize| InRecord::Buf(fastx_to_record(b"r".to_vec(), vec![b'A'; len], None));
        let mut batch: Vec<InRecord> = (0..100).map(|_| read(100)).collect();
        batch.insert(10, read(300_000));
        let sizes: Vec<usize> = cut_chunks(batch).iter().map(Vec::len).collect();
        // 10 short, the giant alone, then the other 90 in chunks of <= CHUNK.
        assert_eq!(sizes, vec![10, 1, CHUNK, 90 - CHUNK]);
    }

    #[test]
    fn fastx_sequences_fit_bam() {
        assert_eq!(normalise_seq(b"acguNx*"), b"ACGTNNN".to_vec());
    }

    #[test]
    fn fasta_reader_joins_lines() {
        let text = b">a desc\nACG\nTT\n\n>b\nGG\n".to_vec();
        let mut r = FastxReader::new(Box::new(std::io::Cursor::new(text)), false);
        let (n, s, q) = r.next_record().unwrap().unwrap();
        assert_eq!(
            (n.as_slice(), s.as_slice(), q),
            (&b"a"[..], &b"ACGTT"[..], None)
        );
        let (n, s, _) = r.next_record().unwrap().unwrap();
        assert_eq!((n.as_slice(), s.as_slice()), (&b"b"[..], &b"GG"[..]));
        assert!(r.next_record().unwrap().is_none());
    }

    #[test]
    fn fastq_reader_decodes_phred() {
        let text = b"@r1 x=1\nACGT\n+\nII#!\n".to_vec();
        let mut r = FastxReader::new(Box::new(std::io::Cursor::new(text)), true);
        let (n, s, q) = r.next_record().unwrap().unwrap();
        assert_eq!(n, b"r1");
        assert_eq!(s, b"ACGT");
        assert_eq!(q.unwrap(), vec![40, 40, 2, 0]);
    }
}
