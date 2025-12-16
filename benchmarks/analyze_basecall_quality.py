#!/usr/bin/env python3
"""
Analyze basecalling quality from benchmark BAM files.

This script compares basecalling results from original vs downsampled POD5 files,
extracting quality metrics and generating summary statistics.

Usage:
    python analyze_basecall_quality.py <output_dir> <factors> <models>

Example:
    python analyze_basecall_quality.py results/ "2 4" "fast hac sup"
"""

import sys
import json
import re
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

try:
    import pysam
except ImportError:
    print("Error: pysam is required. Install with: pip install pysam", file=sys.stderr)
    sys.exit(1)

try:
    import pandas as pd
except ImportError:
    pd = None  # Optional, for nicer output


@dataclass
class QualityMetrics:
    """Quality metrics extracted from a BAM file."""
    total_reads: int = 0
    mapped_reads: int = 0
    total_bases: int = 0
    aligned_bases: int = 0
    qscore_sum: float = 0.0
    matches: int = 0
    mismatches: int = 0
    insertions: int = 0
    deletions: int = 0
    mapq_sum: int = 0
    mapq_ge60: int = 0

    @property
    def mean_qscore(self) -> float:
        """Calculate mean Q-score across all reads."""
        if self.total_reads == 0:
            return 0.0
        return self.qscore_sum / self.total_reads

    @property
    def mapped_pct(self) -> float:
        """Percentage of reads that mapped."""
        if self.total_reads == 0:
            return 0.0
        return 100.0 * self.mapped_reads / self.total_reads

    @property
    def identity(self) -> float:
        """Alignment identity (matches / aligned length)."""
        total = self.matches + self.mismatches + self.insertions + self.deletions
        if total == 0:
            return 0.0
        return 100.0 * self.matches / total

    @property
    def substitution_rate(self) -> float:
        """Substitution (mismatch) rate."""
        if self.aligned_bases == 0:
            return 0.0
        return 100.0 * self.mismatches / self.aligned_bases

    @property
    def insertion_rate(self) -> float:
        """Insertion rate."""
        if self.aligned_bases == 0:
            return 0.0
        return 100.0 * self.insertions / self.aligned_bases

    @property
    def deletion_rate(self) -> float:
        """Deletion rate."""
        if self.aligned_bases == 0:
            return 0.0
        return 100.0 * self.deletions / self.aligned_bases

    @property
    def mean_mapq(self) -> float:
        """Mean mapping quality."""
        if self.mapped_reads == 0:
            return 0.0
        return self.mapq_sum / self.mapped_reads

    @property
    def mapq_ge60_pct(self) -> float:
        """Percentage of reads with MAPQ >= 60."""
        if self.mapped_reads == 0:
            return 0.0
        return 100.0 * self.mapq_ge60 / self.mapped_reads


def parse_cigar(cigar_tuples) -> Dict[str, int]:
    """Parse CIGAR tuples and count operations.

    CIGAR operations:
        M (0): alignment match (can be match or mismatch)
        I (1): insertion to reference
        D (2): deletion from reference
        N (3): skipped region from reference
        S (4): soft clipping
        H (5): hard clipping
        P (6): padding
        = (7): sequence match
        X (8): sequence mismatch
    """
    counts = {"M": 0, "I": 0, "D": 0, "=": 0, "X": 0}
    op_map = {0: "M", 1: "I", 2: "D", 7: "=", 8: "X"}

    for op, length in cigar_tuples:
        if op in op_map:
            counts[op_map[op]] += length

    return counts


def calculate_read_qscore(qualities) -> float:
    """Calculate mean Q-score from quality array."""
    if qualities is None or len(qualities) == 0:
        return 0.0
    return sum(qualities) / len(qualities)


def extract_metrics(bam_path: Path) -> QualityMetrics:
    """Extract quality metrics from a BAM file."""
    metrics = QualityMetrics()

    try:
        with pysam.AlignmentFile(str(bam_path), "rb") as bam:
            for read in bam:
                metrics.total_reads += 1

                # Q-score from quality string
                if read.query_qualities is not None:
                    metrics.qscore_sum += calculate_read_qscore(read.query_qualities)
                    metrics.total_bases += len(read.query_qualities)

                if not read.is_unmapped:
                    metrics.mapped_reads += 1
                    metrics.mapq_sum += read.mapping_quality

                    if read.mapping_quality >= 60:
                        metrics.mapq_ge60 += 1

                    # Parse CIGAR for error analysis
                    if read.cigartuples:
                        cigar_counts = parse_cigar(read.cigartuples)

                        # If CIGAR uses = and X, we have exact match/mismatch info
                        if cigar_counts["="] > 0 or cigar_counts["X"] > 0:
                            metrics.matches += cigar_counts["="]
                            metrics.mismatches += cigar_counts["X"]
                        else:
                            # M could be match or mismatch, count as match
                            # (actual mismatches would need MD tag parsing)
                            metrics.matches += cigar_counts["M"]

                        metrics.insertions += cigar_counts["I"]
                        metrics.deletions += cigar_counts["D"]
                        metrics.aligned_bases += sum(cigar_counts.values())

    except Exception as e:
        print(f"Warning: Error reading {bam_path}: {e}", file=sys.stderr)

    return metrics


def format_table(data: List[Dict], columns: List[str]) -> str:
    """Format data as a simple ASCII table."""
    if pd is not None:
        df = pd.DataFrame(data)
        return df[columns].to_string(index=False)

    # Manual formatting
    widths = {col: len(col) for col in columns}
    for row in data:
        for col in columns:
            widths[col] = max(widths[col], len(str(row.get(col, ""))))

    header = " | ".join(col.ljust(widths[col]) for col in columns)
    separator = "-+-".join("-" * widths[col] for col in columns)
    rows = [
        " | ".join(str(row.get(col, "")).ljust(widths[col]) for col in columns)
        for row in data
    ]

    return "\n".join([header, separator] + rows)


def main(output_dir: str, factors: str, models: str):
    output_path = Path(output_dir)
    factor_list = factors.split()
    model_list = models.split()

    results = []

    for model in model_list:
        # Original basecalls
        original_bam = output_path / f"original_{model}.bam"
        if original_bam.exists():
            metrics = extract_metrics(original_bam)
            results.append({
                "model": model,
                "condition": "original",
                "factor": 1,
                "reads": metrics.total_reads,
                "mapped_pct": round(metrics.mapped_pct, 1),
                "mean_qscore": round(metrics.mean_qscore, 2),
                "identity": round(metrics.identity, 2),
                "sub_rate": round(metrics.substitution_rate, 3),
                "ins_rate": round(metrics.insertion_rate, 3),
                "del_rate": round(metrics.deletion_rate, 3),
                "mean_mapq": round(metrics.mean_mapq, 1),
                "mapq_ge60_pct": round(metrics.mapq_ge60_pct, 1),
                "_metrics": asdict(metrics),
            })
        else:
            print(f"Warning: Not found: {original_bam}", file=sys.stderr)

        # Downsampled basecalls
        for factor in factor_list:
            archived_bam = output_path / f"archived_{factor}x_{model}.bam"
            if archived_bam.exists():
                metrics = extract_metrics(archived_bam)
                results.append({
                    "model": model,
                    "condition": f"{factor}x DS",
                    "factor": int(factor),
                    "reads": metrics.total_reads,
                    "mapped_pct": round(metrics.mapped_pct, 1),
                    "mean_qscore": round(metrics.mean_qscore, 2),
                    "identity": round(metrics.identity, 2),
                    "sub_rate": round(metrics.substitution_rate, 3),
                    "ins_rate": round(metrics.insertion_rate, 3),
                    "del_rate": round(metrics.deletion_rate, 3),
                    "mean_mapq": round(metrics.mean_mapq, 1),
                    "mapq_ge60_pct": round(metrics.mapq_ge60_pct, 1),
                    "_metrics": asdict(metrics),
                })
            else:
                print(f"Warning: Not found: {archived_bam}", file=sys.stderr)

    if not results:
        print("Error: No BAM files found to analyze", file=sys.stderr)
        sys.exit(1)

    # Calculate deltas from original
    for model in model_list:
        model_results = [r for r in results if r["model"] == model]
        original = next((r for r in model_results if r["condition"] == "original"), None)

        if original:
            for r in model_results:
                if r["condition"] != "original":
                    r["delta_qscore"] = round(r["mean_qscore"] - original["mean_qscore"], 2)
                    r["delta_identity"] = round(r["identity"] - original["identity"], 2)
                else:
                    r["delta_qscore"] = 0
                    r["delta_identity"] = 0

    # Output summary table
    print("\n" + "=" * 80)
    print("BASECALL QUALITY COMPARISON: Original vs Downsampled")
    print("=" * 80 + "\n")

    columns = ["model", "condition", "reads", "mapped_pct", "mean_qscore",
               "identity", "sub_rate", "ins_rate", "del_rate"]
    print(format_table(results, columns))

    # Output delta summary
    print("\n" + "-" * 80)
    print("QUALITY CHANGE FROM ORIGINAL (negative = worse)")
    print("-" * 80 + "\n")

    delta_columns = ["model", "condition", "delta_qscore", "delta_identity"]
    delta_results = [r for r in results if r["condition"] != "original"]
    if delta_results:
        print(format_table(delta_results, delta_columns))

    # Save detailed results
    summary_path = output_path / "quality_summary.tsv"
    if pd is not None:
        df = pd.DataFrame(results)
        df = df.drop(columns=["_metrics"], errors="ignore")
        df.to_csv(summary_path, sep="\t", index=False)
    else:
        with open(summary_path, "w") as f:
            cols = [c for c in columns + ["delta_qscore", "delta_identity"]]
            f.write("\t".join(cols) + "\n")
            for r in results:
                f.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")

    # Save full metrics as JSON
    json_path = output_path / "quality_metrics.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to:")
    print(f"  Summary: {summary_path}")
    print(f"  Details: {json_path}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)

    main(sys.argv[1], sys.argv[2], sys.argv[3])
