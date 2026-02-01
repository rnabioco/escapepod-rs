//! Kmer table loading and level extraction.

use anyhow::{bail, Result};
use flate2::read::GzDecoder;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// A table mapping k-mers to their expected signal levels.
#[derive(Debug)]
pub struct KmerTable {
    index: HashMap<Vec<u8>, usize>,
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
        let mut kmers_unsorted: Vec<Vec<u8>> = Vec::new();
        let mut levels_unsorted: Vec<f32> = Vec::new();
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
            if kmer.len() % 2 == 0 {
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

            kmers_unsorted.push(kmer);
            levels_unsorted.push(level);
        }

        if kmers_unsorted.is_empty() {
            bail!("empty kmer table file");
        }

        let k = kmers_unsorted[0].len();
        let exp_len = 4usize.pow(k as u32);
        if kmers_unsorted.len() < exp_len {
            bail!(
                "kmer table has {} entries, expected at least {} (4^{})",
                kmers_unsorted.len(),
                exp_len,
                k
            );
        }

        // Sort by level and build index
        let mut indices: Vec<usize> = (0..levels_unsorted.len()).collect();
        indices.sort_by(|&i, &j| {
            levels_unsorted[i]
                .partial_cmp(&levels_unsorted[j])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut index = HashMap::new();
        let mut kmers_sorted = Vec::with_capacity(kmers_unsorted.len());
        let mut levels_sorted = Vec::with_capacity(levels_unsorted.len());

        for (new_idx, &old_idx) in indices.iter().enumerate() {
            kmers_sorted.push(kmers_unsorted[old_idx].clone());
            levels_sorted.push(levels_unsorted[old_idx]);
            index.insert(kmers_unsorted[old_idx].clone(), new_idx);
        }

        let dominant_base = determine_dominant_base(&kmers_sorted, k);

        Ok(KmerTable {
            index,
            levels: levels_sorted,
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

        self.levels = self
            .levels
            .iter()
            .map(|el| (el - median) / scaled_mad)
            .collect();

        Ok(())
    }

    /// Look up the level for a kmer (as bytes).
    pub fn get(&self, kmer: &[u8]) -> Result<f32> {
        if kmer.len() != self.k {
            bail!("kmer length {} != expected {}", kmer.len(), self.k);
        }
        let idx = self
            .index
            .get(kmer)
            .ok_or_else(|| anyhow::anyhow!("kmer not found in table"))?;
        Ok(self.levels[*idx])
    }

    /// Extract expected levels for each position in a sequence.
    ///
    /// Slides a kmer window across the sequence and assigns each kmer's level
    /// to the dominant base position within that kmer.
    pub fn extract_levels(&self, seq: &[u8]) -> Result<Vec<f32>> {
        if seq.len() < self.k {
            bail!("sequence length {} < kmer size {}", seq.len(), self.k);
        }

        let mut levels = vec![0.0f32; seq.len()];

        for pos in 0..(seq.len() - self.k + 1) {
            let center_pos = pos + self.dominant_base;
            let kmer = &seq[pos..(pos + self.k)];
            let level = self.get(kmer)?;
            levels[center_pos] = level;
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
fn determine_dominant_base(kmers_sorted: &[Vec<u8>], k: usize) -> usize {
    let n_kmers = kmers_sorted.len();
    let mut best_pos = k / 2; // default to center
    let mut best_h = f64::NEG_INFINITY;

    for base_idx in 0..k {
        let mut indices_a = Vec::with_capacity(n_kmers / 4);
        let mut indices_c = Vec::with_capacity(n_kmers / 4);
        let mut indices_g = Vec::with_capacity(n_kmers / 4);
        let mut indices_t = Vec::with_capacity(n_kmers / 4);

        for (kmer_idx, kmer) in kmers_sorted.iter().enumerate() {
            match kmer[base_idx] {
                b'A' | b'a' => indices_a.push(kmer_idx),
                b'C' | b'c' => indices_c.push(kmer_idx),
                b'G' | b'g' => indices_g.push(kmer_idx),
                b'T' | b't' => indices_t.push(kmer_idx),
                _ => {}
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
