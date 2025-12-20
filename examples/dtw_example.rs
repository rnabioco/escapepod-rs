//! Example demonstrating DTW distance computation for barcode fingerprint comparison.
//!
//! This example shows how to:
//! 1. Create and normalize barcode fingerprints
//! 2. Compute DTW distances between fingerprints
//! 3. Build a distance matrix for multiple fingerprints
//! 4. Convert distances to a kernel matrix for classification

use escapepod::dtw::{
    distance_to_kernel, distance_to_kernel_auto, dtw_distance, dtw_distance_matrix, Fingerprint,
    NormMethod,
};
use uuid::Uuid;

fn main() {
    println!("DTW Barcode Fingerprint Example\n");

    // Example 1: Simple DTW distance between two sequences
    println!("=== Example 1: Basic DTW Distance ===");
    let seq_a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let seq_b = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let distance = dtw_distance(&seq_a, &seq_b, None);
    println!("Distance between identical sequences: {}", distance);

    let seq_c = vec![1.5, 2.5, 3.5, 4.5, 5.5];
    let distance = dtw_distance(&seq_a, &seq_c, None);
    println!("Distance between similar sequences: {:.2}", distance);
    println!();

    // Example 2: DTW with Sakoe-Chiba window constraint
    println!("=== Example 2: DTW with Window Constraint ===");
    let seq_long = vec![1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0];
    let seq_short = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let dist_no_window = dtw_distance(&seq_long, &seq_short, None);
    let dist_with_window = dtw_distance(&seq_long, &seq_short, Some(2));

    println!("Distance without window: {:.2}", dist_no_window);
    println!("Distance with window=2: {:.2}", dist_with_window);
    println!();

    // Example 3: Fingerprint normalization
    println!("=== Example 3: Fingerprint Normalization ===");

    // Raw signal means from barcode region
    let raw_values = vec![100.5, 105.2, 98.3, 102.7, 110.1];
    let mut fp = Fingerprint::new(raw_values.clone(), Uuid::new_v4());

    println!("Original values: {:?}", fp.values);

    fp.normalize(NormMethod::ZScore);
    println!("After Z-score normalization: {:?}", fp.values);

    // Reset and try min-max normalization
    let mut fp = Fingerprint::new(raw_values.clone(), Uuid::new_v4());
    fp.normalize(NormMethod::MinMax);
    println!("After Min-Max normalization: {:?}", fp.values);
    println!();

    // Example 4: Distance matrix for multiple fingerprints
    println!("=== Example 4: Distance Matrix ===");

    // Simulate barcode fingerprints for different reads
    let barcode_a_fingerprints = vec![
        vec![1.0, 2.0, 3.0, 2.0, 1.0], // Read 1
        vec![1.1, 2.1, 3.1, 2.1, 1.1], // Read 2 (similar to barcode A)
    ];

    let barcode_b_fingerprints = vec![
        vec![5.0, 4.0, 3.0, 4.0, 5.0], // Reference for barcode B
        vec![5.1, 4.1, 3.1, 4.1, 5.1], // Another barcode B reference
    ];

    let distance_matrix = dtw_distance_matrix(&barcode_a_fingerprints, &barcode_b_fingerprints, None);

    println!("Distance matrix shape: {:?}", distance_matrix.shape());
    println!("Distance matrix:");
    println!("{:.2}", distance_matrix);
    println!();

    // Example 5: Kernel conversion for classification
    println!("=== Example 5: Kernel Conversion ===");

    // Convert distances to RBF kernel
    let gamma = 0.1;
    let power = 1.0;
    let kernel = distance_to_kernel(&distance_matrix, gamma, power);

    println!("RBF Kernel with gamma={}, power={}:", gamma, power);
    println!("{:.4}", kernel);
    println!();

    // Auto-estimate gamma using median heuristic
    let (kernel_auto, estimated_gamma) = distance_to_kernel_auto(&distance_matrix, 1.0);
    println!("Auto-estimated gamma: {:.4}", estimated_gamma);
    println!("Kernel with auto gamma:");
    println!("{:.4}", kernel_auto);
    println!();

    // Example 6: Practical barcode classification scenario
    println!("=== Example 6: Barcode Classification Scenario ===");

    // Reference fingerprints for known barcodes
    let reference_barcodes = vec![
        vec![1.0, 2.0, 3.0, 2.0, 1.0],  // Barcode 1
        vec![5.0, 4.0, 3.0, 4.0, 5.0],  // Barcode 2
        vec![2.0, 3.0, 4.0, 5.0, 6.0],  // Barcode 3
    ];

    // Unknown read fingerprints to classify
    let unknown_reads = vec![
        vec![1.1, 2.1, 3.0, 2.0, 1.1],  // Should match Barcode 1
        vec![5.0, 4.1, 3.1, 4.0, 5.1],  // Should match Barcode 2
        vec![2.0, 3.1, 4.1, 5.1, 5.9],  // Should match Barcode 3
    ];

    let distances = dtw_distance_matrix(&unknown_reads, &reference_barcodes, Some(2));

    println!("Distance matrix (unknown reads x reference barcodes):");
    println!("{:.2}", distances);
    println!();

    // Find closest match for each unknown read
    for (i, row) in distances.outer_iter().enumerate() {
        let (min_idx, min_dist) = row
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        println!(
            "Read {} -> Barcode {} (distance: {:.2})",
            i + 1,
            min_idx + 1,
            min_dist
        );
    }
}
