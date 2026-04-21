// SPDX-License-Identifier: GPL-3.0-or-later
// Inspired by fishnet, licensed under the GNU General Public License v3.0.

//! Kmer table loading and level extraction.

use anyhow::{Result, bail};
use flate2::read::GzDecoder;
use std::collections::HashSet;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Encode a nucleotide base to a 2-bit value (A=0, C=1, G=2, T/U=3).
#[inline]
fn encode_base(b: u8) -> Option<usize> {
    match b {
        b'A' | b'a' => Some(0),
        b'C' | b'c' => Some(1),
        b'G' | b'g' => Some(2),
        b'T' | b't' | b'U' | b'u' => Some(3),
        _ => None,
    }
}

/// Encode a kmer as an integer index for flat array lookup.
#[inline]
fn encode_kmer(kmer: &[u8]) -> Option<usize> {
    let mut idx = 0usize;
    for &b in kmer {
        idx = (idx << 2) | encode_base(b)?;
    }
    Some(idx)
}

/// A table mapping k-mers to their expected signal levels.
///
/// Uses a flat array indexed by 2-bit-per-base encoding for O(1) lookup
/// (4^k entries for k-mers over {A,C,G,T/U}).
#[derive(Debug)]
pub struct KmerTable {
    levels: Vec<f32>,
    k: usize,
    dominant_base: usize,
}

impl KmerTable {
    /// Load a kmer table from a tab-delimited file (kmer\tlevel).
    ///
    /// Transparently handles gzip-compressed files (`.gz` extension).
    pub fn from_file(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        let is_gz = path
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("gz"));
        let reader: Box<dyn BufRead> = if is_gz {
            Box::new(BufReader::new(GzDecoder::new(file)))
        } else {
            Box::new(BufReader::new(file))
        };

        let mut unique_kmers = HashSet::new();
        let mut kmer_levels: Vec<(Vec<u8>, f32)> = Vec::new();
        let mut prev_k: Option<usize> = None;

        for line in reader.lines() {
            let line = line?;
            if line.is_empty() {
                continue;
            }
            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() != 2 {
                bail!("expected 2 tab-separated columns, got {}", parts.len());
            }

            let kmer = parts[0].as_bytes().to_vec();
            if kmer.is_empty() {
                bail!("empty kmer");
            }
            if kmer.len().is_multiple_of(2) {
                bail!("even kmer length {} (odd expected)", kmer.len());
            }

            let k = kmer.len();
            match prev_k {
                Some(pk) if pk != k => bail!("non-uniform kmer length: {} vs {}", k, pk),
                None => prev_k = Some(k),
                _ => {}
            }

            if !unique_kmers.insert(kmer.clone()) {
                bail!("duplicate kmer: {}", parts[0]);
            }

            let level: f32 = parts[1]
                .parse()
                .map_err(|e| anyhow::anyhow!("cannot parse level '{}': {}", parts[1], e))?;

            kmer_levels.push((kmer, level));
        }

        if kmer_levels.is_empty() {
            bail!("empty kmer table file");
        }

        let k = kmer_levels[0].0.len();
        let table_size = 4usize.pow(k as u32);
        if kmer_levels.len() < table_size {
            bail!(
                "kmer table has {} entries, expected at least {} (4^{})",
                kmer_levels.len(),
                table_size,
                k
            );
        }

        // Build flat array indexed by encoded kmer
        let mut levels = vec![0.0f32; table_size];
        for (kmer, level) in &kmer_levels {
            let idx = encode_kmer(kmer)
                .ok_or_else(|| anyhow::anyhow!("invalid base in kmer: {:?}", kmer))?;
            levels[idx] = *level;
        }

        let dominant_base = determine_dominant_base(&levels, k);

        Ok(KmerTable {
            levels,
            k,
            dominant_base,
        })
    }

    /// Normalize levels using MAD: (level - median) / (MAD * 1.4826).
    pub fn fix_gauge(&mut self) -> Result<()> {
        let median = median_f32(&self.levels).ok_or_else(|| anyhow::anyhow!("empty levels"))?;

        let deviations: Vec<f32> = self.levels.iter().map(|el| (el - median).abs()).collect();
        let mad = median_f32(&deviations).ok_or_else(|| anyhow::anyhow!("cannot compute MAD"))?;

        let scaled_mad = mad * 1.4826;
        if scaled_mad == 0.0 {
            bail!("MAD is zero, cannot normalize");
        }

        for level in &mut self.levels {
            *level = (*level - median) / scaled_mad;
        }

        Ok(())
    }

    /// Look up the level for a kmer (as bytes).
    #[inline]
    pub fn get(&self, kmer: &[u8]) -> Result<f32> {
        if kmer.len() != self.k {
            bail!("kmer length {} != expected {}", kmer.len(), self.k);
        }
        let idx = encode_kmer(kmer).ok_or_else(|| anyhow::anyhow!("kmer contains invalid base"))?;
        Ok(self.levels[idx])
    }

    /// Extract expected levels for each position in a sequence.
    ///
    /// Uses rolling 2-bit encoding for O(1) per-position lookup.
    pub fn extract_levels(&self, seq: &[u8]) -> Result<Vec<f32>> {
        if seq.len() < self.k {
            bail!("sequence length {} < kmer size {}", seq.len(), self.k);
        }

        let mut levels = vec![0.0f32; seq.len()];
        let mask = (1usize << (2 * self.k)) - 1;

        // Encode first kmer
        let mut idx = 0usize;
        for (i, &base) in seq[..self.k].iter().enumerate() {
            let b = encode_base(base)
                .ok_or_else(|| anyhow::anyhow!("invalid base at position {}", i))?;
            idx = (idx << 2) | b;
        }
        levels[self.dominant_base] = self.levels[idx];

        // Rolling encode for remaining positions
        for pos in 1..=(seq.len() - self.k) {
            let new_base = encode_base(seq[pos + self.k - 1])
                .ok_or_else(|| anyhow::anyhow!("invalid base at position {}", pos + self.k - 1))?;
            idx = ((idx << 2) | new_base) & mask;
            levels[pos + self.dominant_base] = self.levels[idx];
        }

        Ok(levels)
    }

    /// The kmer size.
    pub fn k(&self) -> usize {
        self.k
    }
}

/// Determine which position in the kmer has the most influence on levels
/// using the Kruskal-Wallis H test.
fn determine_dominant_base(levels: &[f32], k: usize) -> usize {
    let n_kmers = levels.len();

    // Sort kmer indices by level to get rank ordering
    let mut sorted_indices: Vec<usize> = (0..n_kmers).collect();
    sorted_indices.sort_by(|&a, &b| {
        levels[a]
            .partial_cmp(&levels[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Build rank array: rank[encoded_kmer] = position in sorted order
    let mut rank = vec![0usize; n_kmers];
    for (sort_pos, &encoded_idx) in sorted_indices.iter().enumerate() {
        rank[encoded_idx] = sort_pos;
    }

    let mut best_pos = k / 2; // default to center
    let mut best_h = f64::NEG_INFINITY;

    for base_idx in 0..k {
        let shift = 2 * (k - 1 - base_idx);
        let mut indices_a = Vec::with_capacity(n_kmers / 4);
        let mut indices_c = Vec::with_capacity(n_kmers / 4);
        let mut indices_g = Vec::with_capacity(n_kmers / 4);
        let mut indices_t = Vec::with_capacity(n_kmers / 4);

        for (encoded_idx, &r) in rank.iter().enumerate() {
            // Extract the base at position base_idx from the encoded kmer
            let base = (encoded_idx >> shift) & 0x3;
            match base {
                0 => indices_a.push(r),
                1 => indices_c.push(r),
                2 => indices_g.push(r),
                3 => indices_t.push(r),
                _ => unreachable!(),
            }
        }

        let h = kruskal_h(&[&indices_a, &indices_c, &indices_g, &indices_t]);
        if h > best_h {
            best_h = h;
            best_pos = base_idx;
        }
    }

    best_pos
}

/// Kruskal-Wallis H statistic.
fn kruskal_h(samples: &[&[usize]]) -> f64 {
    let total: f64 = samples.iter().map(|s| s.len() as f64).sum();
    if total == 0.0 {
        return 0.0;
    }

    let sum: f64 = samples
        .iter()
        .filter(|g| !g.is_empty())
        .map(|group| {
            let rank_sum: f64 = group.iter().map(|&el| el as f64).sum();
            rank_sum.powi(2) / (group.len() as f64)
        })
        .sum();

    (12.0 / (total * (total + 1.0))) * sum - 3.0 * (total + 1.0)
}

/// Median of an f32 slice (makes a sorted copy).
fn median_f32(data: &[f32]) -> Option<f32> {
    if data.is_empty() {
        return None;
    }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let len = sorted.len();
    Some(if len % 2 == 1 {
        sorted[len / 2]
    } else {
        (sorted[len / 2 - 1] + sorted[len / 2]) / 2.0
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_kmer() {
        assert_eq!(encode_kmer(b"A"), Some(0));
        assert_eq!(encode_kmer(b"C"), Some(1));
        assert_eq!(encode_kmer(b"G"), Some(2));
        assert_eq!(encode_kmer(b"T"), Some(3));
        assert_eq!(encode_kmer(b"AC"), Some(0b0001));
        assert_eq!(encode_kmer(b"GT"), Some(0b1011));
        assert_eq!(encode_kmer(b"AAAAAAAAA"), Some(0));
        assert_eq!(encode_kmer(b"N"), None);
    }

    #[test]
    fn test_kruskal_h() {
        let x = vec![1, 3, 5, 7, 9];
        let y = vec![2, 4, 6, 8, 10];
        let h = kruskal_h(&[&x, &y]);
        assert!((h - 0.2727272727272734).abs() < 1e-5);
    }

    #[test]
    fn test_median() {
        assert_eq!(median_f32(&[1.0, 2.0, 3.0]), Some(2.0));
        assert_eq!(median_f32(&[1.0, 2.0, 3.0, 4.0]), Some(2.5));
        assert_eq!(median_f32(&[]), None);
    }

    #[test]
    fn test_load_rna004_kmer_table() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("data/kmer_models/rna004_9mer_levels_v1.txt.gz");
        if !path.exists() {
            eprintln!("skipping test: {:?} not found", path);
            return;
        }
        let table = KmerTable::from_file(&path).unwrap();
        assert_eq!(table.k(), 9);

        // Spot-check a known kmer
        let level = table.get(b"AAAAAAAAA").unwrap();
        assert!((level - 0.95838).abs() < 1e-4);

        // extract_levels on a short sequence
        let seq = b"AAAAAAAAAAAA"; // 12 bases, should produce 12 levels
        let levels = table.extract_levels(seq).unwrap();
        assert_eq!(levels.len(), 12);

        // Sequence too short should fail
        assert!(table.extract_levels(b"ACGT").is_err());
    }

    #[test]
    fn test_fix_gauge_rna004() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("data/kmer_models/rna004_9mer_levels_v1.txt.gz");
        if !path.exists() {
            eprintln!("skipping test: {:?} not found", path);
            return;
        }
        let mut table = KmerTable::from_file(&path).unwrap();
        table.fix_gauge().unwrap();

        // After MAD normalization, median should be ~0 and MAD ~1
        let median = median_f32(&table.levels).unwrap();
        assert!(median.abs() < 1e-5, "median after fix_gauge: {}", median);
    }
}
