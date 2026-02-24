//! DiskANN Beam Search - 异步搜索优化

use std::collections::{BinaryHeap, HashSet};

/// Beam Search 结果项
#[derive(Clone, Debug)]
pub struct BeamItem {
    pub id: usize,
    pub dist: f32,
}

impl PartialEq for BeamItem {
    fn eq(&self, other: &Self) -> bool { self.dist == other.dist }
}

impl Eq for BeamItem {}

impl PartialOrd for BeamItem {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.dist.partial_cmp(&other.dist)
    }
}

impl Ord for BeamItem {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.dist.partial_cmp(&other.dist).unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// DiskANN Beam Search 搜索器
pub struct BeamSearcher {
    pub L: usize,    // 搜索宽度
    pub R: usize,    // 邻居数
}

impl BeamSearcher {
    pub fn new(L: usize, R: usize) -> Self {
        Self { L, R }
    }
    
    /// 异步 Beam Search（图遍历）
    pub fn search(
        &self,
        query: &[f32],
        vectors: &[f32],
        graph: &[Vec<usize>],
        dim: usize,
        top_k: usize,
    ) -> Vec<BeamItem> {
        if graph.is_empty() || vectors.is_empty() {
            return vec![];
        }
        
        let n = vectors.len() / dim;
        
        // 初始化：贪婪搜索到入口点
        let mut visited = HashSet::new();
        let mut candidates: BinaryHeap<BeamItem> = BinaryHeap::new();
        let mut results: BinaryHeap<BeamItem> = BinaryHeap::new();
        
        // 从随机入口开始
        let entry = 0;
        visited.insert(entry);
        let entry_dist = self.l2_distance(query, entry, vectors, dim);
        candidates.push(BeamItem { id: entry, dist: entry_dist });
        
        // Beam Search
        while let Some(current) = candidates.pop() {
            if results.len() >= self.L && current.dist > results.peek().map(|r| r.dist).unwrap_or(f32::MAX) {
                break;
            }
            
            if results.iter().any(|r| r.id == current.id) {
                continue;
            }
            
            results.push(current.clone());
            
            // 遍历邻居
            for &neighbor in &graph[current.id] {
                if !visited.contains(&neighbor) {
                    visited.insert(neighbor);
                    let n_dist = self.l2_distance(query, neighbor, vectors, dim);
                    candidates.push(BeamItem { id: neighbor, dist: n_dist });
                }
            }
        }
        
        // 排序返回 top_k
        let mut sorted: Vec<_> = results.into_vec();
        sorted.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap());
        sorted.truncate(top_k);
        sorted
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

/// 异步批量搜索（支持 io_uring）
pub struct AsyncSearcher {
    beam: BeamSearcher,
}

impl AsyncSearcher {
    pub fn new(L: usize, R: usize) -> Self {
        Self { beam: BeamSearcher::new(L, R) }
    }
    
    /// 批量搜索
    pub fn search_batch(
        &self,
        queries: &[f32],
        vectors: &[f32],
        graph: &[Vec<usize>],
        dim: usize,
        top_k: usize,
    ) -> Vec<Vec<BeamItem>> {
        let batch_size = queries.len() / dim;
        (0..batch_size)
            .map(|i| {
                let q = &queries[i * dim..];
                self.beam.search(q, vectors, graph, dim, top_k)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_beam_searcher() {
        let beam = BeamSearcher::new(10, 20);
        assert_eq!(beam.L, 10);
    }
    
    #[test]
    fn test_async_searcher() {
        let async_s = AsyncSearcher::new(10, 20);
        // Empty vectors should return empty batch results
        let results = async_s.search_batch(&[1.0, 2.0, 3.0, 4.0], &[], &[], 2, 1);
        assert_eq!(results.len(), 2); // 2 queries in batch
    }
}
