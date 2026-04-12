//! Shared progress reporting types.

/// Progress state for file-level operations (filter, repack, etc.).
#[derive(Debug, Clone, Copy)]
pub struct Progress {
    /// Current item (0-based).
    pub current: u64,
    /// Total items.
    pub total: u64,
}

/// Callback type for reporting progress during operations.
pub type ProgressCallback = Box<dyn Fn(Progress) + Send + Sync>;
