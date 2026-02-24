//! Search API

use std::sync::Arc;

/// Search request parameters
#[derive(Clone)]
pub struct SearchRequest {
    /// Number of nearest neighbors to return
    pub top_k: usize,
    /// Number of probes for IVF indices
    pub nprobe: usize,
    /// Optional filter predicate
    pub filter: Option<Arc<dyn Predicate>>,
    /// Search params (JSON string)
    pub params: Option<String>,
}

/// Search result
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// IDs of the nearest neighbors
    pub ids: Vec<i64>,
    /// Distances to the nearest neighbors
    pub distances: Vec<f32>,
    /// Elapsed time in milliseconds
    pub elapsed_ms: f64,
    /// Number of vectors searched
    pub num_visited: usize,
}

impl SearchResult {
    pub fn new(ids: Vec<i64>, distances: Vec<f32>, elapsed_ms: f64) -> Self {
        let num_visited = ids.len();
        Self {
            ids,
            distances,
            elapsed_ms,
            num_visited,
        }
    }
}

/// Predicate for filtering
pub trait Predicate: Send + Sync {
    fn evaluate(&self, id: i64) -> bool;
}

/// Range predicate
#[derive(Debug, Clone)]
pub struct RangePredicate {
    pub field: String,
    pub min: f64,
    pub max: f64,
}

impl Predicate for RangePredicate {
    fn evaluate(&self, id: i64) -> bool {
        // Placeholder - actual implementation depends on storage
        true
    }
}

/// IDs predicate
#[derive(Debug, Clone)]
pub struct IdsPredicate {
    pub ids: Vec<i64>,
}

impl Predicate for IdsPredicate {
    fn evaluate(&self, id: i64) -> bool {
        self.ids.contains(&id)
    }
}
