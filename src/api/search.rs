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

/// Range search result - 返回满足半径阈值的所有向量
#[derive(Debug, Clone)]
pub struct RangeSearchResult {
    /// IDs of vectors within radius
    pub ids: Vec<i64>,
    /// Distances to the query vector
    pub distances: Vec<f32>,
    /// Elapsed time in milliseconds
    pub elapsed_ms: f64,
    /// Number of vectors within radius
    pub result_count: usize,
    /// Total number of vectors searched
    pub num_visited: usize,
}

impl RangeSearchResult {
    pub fn new(ids: Vec<i64>, distances: Vec<f32>, elapsed_ms: f64, num_visited: usize) -> Self {
        let result_count = ids.len();
        Self {
            ids,
            distances,
            elapsed_ms,
            result_count,
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

/// Bitset predicate for filtering using BitsetView
/// 
/// This predicate uses a bitset to filter vectors by their index position.
/// Bit value 1 = filtered out (excluded), 0 = kept (included).
/// 
/// # Example
/// ```ignore
/// let bitset = BitsetView::new(1000);
/// bitset.set(5, true);  // Filter out vector at position 5
/// let predicate = BitsetPredicate::new(bitset);
/// ```
pub struct BitsetPredicate {
    bitset: crate::bitset::BitsetView,
}

impl BitsetPredicate {
    pub fn new(bitset: crate::bitset::BitsetView) -> Self {
        Self { bitset }
    }
    
    pub fn from_raw(data: Vec<u64>, len: usize) -> Self {
        Self {
            bitset: crate::bitset::BitsetView::from_vec(data, len),
        }
    }
}

impl Predicate for BitsetPredicate {
    fn evaluate(&self, id: i64) -> bool {
        // id 是向量 ID，需要转换为位置索引
        // 对于连续 ID (0, 1, 2, ...)，id 本身就是位置
        // bitset 中 1 表示过滤，0 表示保留
        // Predicate::evaluate 返回 true 表示保留
        if id < 0 || id as usize >= self.bitset.len() {
            return true; // 超出范围，保留
        }
        !self.bitset.get(id as usize) // 0=保留 (true), 1=过滤 (false)
    }
}
