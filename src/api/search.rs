//! Search API

use std::sync::Arc;

/// Search request parameters
#[derive(Clone, Default)]
pub struct SearchRequest {
    /// Number of nearest neighbors to return
    pub top_k: usize,
    /// Number of probes for IVF indices
    pub nprobe: usize,
    /// Optional filter predicate
    pub filter: Option<Arc<dyn Predicate>>,
    /// Search params (JSON string)
    pub params: Option<String>,
    /// Radius for range search (if set, performs range search)
    pub radius: Option<f32>,
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

/// AnnIterator - 迭代器风格的近似最近邻搜索
/// 
/// 对齐 C++ Knowhere 的 AnnIterator 接口，支持流式结果迭代
/// 
/// # Example
/// ```ignore
/// let mut iter = index.iter_search(query, top_k);
/// while let Some(result) = iter.next() {
///     println!("id: {}, distance: {}", result.id, result.distance);
/// }
/// ```
#[derive(Debug, Clone)]
pub struct AnnIterator {
    /// Current results buffer
    results: Vec<(i64, f32)>,
    /// Current position
    pos: usize,
    /// Total results returned
    returned: usize,
}

impl AnnIterator {
    /// Create a new iterator with pre-filled results
    pub fn new(results: Vec<(i64, f32)>) -> Self {
        Self {
            pos: 0,
            returned: 0,
            results,
        }
    }

    /// Get next result
    pub fn next(&mut self) -> Option<IterResult> {
        if self.pos >= self.results.len() {
            return None;
        }
        let result = IterResult {
            id: self.results[self.pos].0,
            distance: self.results[self.pos].1,
            index: self.returned,
        };
        self.pos += 1;
        self.returned += 1;
        Some(result)
    }

    /// Peek at next result without consuming
    pub fn peek(&self) -> Option<IterResult> {
        if self.pos >= self.results.len() {
            return None;
        }
        Some(IterResult {
            id: self.results[self.pos].0,
            distance: self.results[self.pos].1,
            index: self.pos,
        })
    }

    /// Check if iterator is exhausted
    pub fn is_exhausted(&self) -> bool {
        self.pos >= self.results.len()
    }

    /// Get total results returned so far
    pub fn count(&self) -> usize {
        self.returned
    }
}

/// Single iterator result
#[derive(Debug, Clone)]
pub struct IterResult {
    /// Vector ID
    pub id: i64,
    /// Distance to query
    pub distance: f32,
    /// Position in result list (0-based)
    pub index: usize,
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
        let id_f = id as f64;
        id_f >= self.min && id_f <= self.max
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
