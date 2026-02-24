//! HNSW 搜索优化 - 优先队列版本
//!
//! **已废弃**: 请使用 `hnsw.rs`

#![deprecated(since = "0.1.0", note = "Use hnsw.rs instead")]

use std::collections::{BinaryHeap, HashSet};
use std::cmp::Reverse;

/// 搜索结果项
#[derive(Clone, Debug)]
pub struct SearchItem {
    pub id: usize,
    pub dist: f32,
}

/// 优先队列项（使用原始 f32 比较）
struct QueueItem {
    dist: f32,
    id: usize,
}

impl PartialEq for QueueItem {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist
    }
}

impl Eq for QueueItem {}

impl PartialOrd for QueueItem {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.dist.partial_cmp(&other.dist)
    }
}

impl Ord for QueueItem {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.dist.partial_cmp(&other.dist).unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// HNSW 贪婪搜索器
pub struct HnswSearcher {
    ef_search: usize,
}

impl HnswSearcher {
    pub fn new(ef: usize) -> Self {
        Self { ef_search: ef }
    }
    
    /// 搜索最近邻（图遍历 + 优先队列）
    pub fn search(
        &self,
        query: &[f32],
        vectors: &[f32],
        graph: &[Vec<(usize, f32)>],
        dim: usize,
        top_k: usize,
    ) -> Vec<SearchItem> {
        if graph.is_empty() || vectors.is_empty() {
            return vec![];
        }
        
        let n = vectors.len() / dim;
        if n == 0 { return vec![]; }
        
        // 入口点
        let mut visited = HashSet::new();
        let mut candidates: BinaryHeap<QueueItem> = BinaryHeap::new();
        let mut results: Vec<SearchItem> = Vec::new();
        
        // 从节点 0 开始
        let entry = 0;
        visited.insert(entry);
        let entry_dist = self.l2_distance(query, entry, vectors, dim);
        candidates.push(QueueItem { dist: entry_dist, id: entry });
        
        // 贪婪搜索
        while let Some(QueueItem { dist, id: node }) = candidates.pop() {
            // 提前退出
            if results.len() >= self.ef_search && dist > results.last().map(|r| r.dist).unwrap_or(f32::MAX) {
                break;
            }
            
            // 跳过已添加
            if results.iter().any(|r| r.id == node) {
                continue;
            }
            
            results.push(SearchItem { id: node, dist });
            
            // 遍历邻居
            for &(neighbor, _) in &graph[node] {
                if !visited.contains(&neighbor) {
                    visited.insert(neighbor);
                    let n_dist = self.l2_distance(query, neighbor, vectors, dim);
                    candidates.push(QueueItem { dist: n_dist, id: neighbor });
                }
            }
        }
        
        results.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap());
        results.truncate(top_k);
        results
    }
    
    #[inline]
    fn l2_distance(&self, query: &[f32], idx: usize, vectors: &[f32], dim: usize) -> f32 {
        let start = idx * dim;
        let mut sum = 0.0f32;
        for i in 0..dim {
            let diff = query[i] - vectors[start + i];
            sum += diff * diff;
        }
        sum
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_searcher_creation() {
        let s = HnswSearcher::new(50);
        assert_eq!(s.ef_search, 50);
    }
    
    #[test]
    fn test_search_empty() {
        let s = HnswSearcher::new(10);
        let result = s.search(&[0.0; 4], &[], &[], 4, 3);
        assert!(result.is_empty());
    }
    
    #[test]
    fn test_search_single_node() {
        let s = HnswSearcher::new(10);
        let vectors = vec![1.0, 2.0, 3.0, 4.0];
        let graph = vec![vec![]; 1];
        
        let result = s.search(&vectors, &vectors, &graph, 4, 1);
        assert!(!result.is_empty());
    }
}
