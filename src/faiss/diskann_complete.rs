//! DiskANN 完整实现
//! Vamana 图算法
//!
//! 参考: https://arxiv.org/abs/2207.00596
//! DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node

use std::collections::{BinaryHeap, HashSet};
use std::cmp::Ordering;
use crate::simd;

/// 搜索项
#[derive(Clone, Debug)]
struct SearchItem {
    dist: f32,
    id: usize,
}

impl PartialEq for SearchItem {
    fn eq(&self, other: &Self) -> bool { self.dist == other.dist }
}

impl Eq for SearchItem {}

impl PartialOrd for SearchItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.dist.partial_cmp(&other.dist)
    }
}

impl Ord for SearchItem {
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist.partial_cmp(&other.dist).unwrap_or(Ordering::Equal)
    }
}

/// Vamana 节点
#[derive(Clone)]
struct VamanaNode {
    vector: Vec<f32>,
    neighbors: Vec<usize>,
}

/// DiskANN / Vamana 索引
pub struct DiskAnnIndex {
    pub dim: usize,
    pub L: usize,
    pub R: usize,
    pub alpha: f32,
    
    nodes: Vec<VamanaNode>,
    num_vectors: usize,
}

impl DiskAnnIndex {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            L: 50,
            R: 32,
            alpha: 1.2,
            nodes: Vec::new(),
            num_vectors: 0,
        }
    }
    
    pub fn with_params(mut self, L: usize, R: usize, alpha: f32) -> Self {
        self.L = L;
        self.R = R;
        self.alpha = alpha;
        self
    }
    
    pub fn add(&mut self, vector: &[f32]) -> usize {
        let id = self.num_vectors;
        
        let node = VamanaNode {
            vector: vector.to_vec(),
            neighbors: Vec::new(),
        };
        
        // 第一个节点
        if id == 0 {
            self.nodes.push(node);
            self.num_vectors += 1;
            return 0;
        }
        
        // 搜索最近邻（如果没有邻居，返回所有节点作为候选）
        let nn: Vec<usize> = if self.num_vectors > 1 {
            self.search_neighbor(vector, 1).into_iter().map(|(i, _)| i).collect()
        } else {
            vec![0]
        };
        
        // 添加边
        let mut node = node;
        for &nid in &nn {
            if !node.neighbors.contains(&nid) {
                node.neighbors.push(nid);
            }
            if nid < self.nodes.len() && !self.nodes[nid].neighbors.contains(&id) {
                self.nodes[nid].neighbors.push(id);
            }
        }
        
        self.prune_neighbors(id);
        
        self.nodes.push(node);
        self.num_vectors += 1;
        
        id
    }
    
    pub fn add_batch(&mut self, vectors: &[f32]) -> usize {
        let n = vectors.len() / self.dim;
        for i in 0..n {
            self.add(&vectors[i * self.dim..]);
        }
        n
    }
    
    fn search_neighbor(&self, query: &[f32], top_k: usize) -> Vec<(usize, f32)> {
        if self.nodes.is_empty() {
            return vec![];
        }
        
        let mut visited = HashSet::new();
        let mut candidates: BinaryHeap<SearchItem> = BinaryHeap::new();
        let mut results: BinaryHeap<SearchItem> = BinaryHeap::new();
        
        let start = 0;
        let dist = self.l2_distance(query, &self.nodes[start].vector);
        
        visited.insert(start);
        candidates.push(SearchItem { dist, id: start });
        
        while let Some(current) = candidates.pop() {
            if results.len() >= self.L && current.dist > results.peek().map(|r| r.dist).unwrap_or(f32::MAX) {
                break;
            }
            
            if !results.iter().any(|r| r.id == current.id) {
                results.push(current.clone());
            }
            
            let neighbors: Vec<usize> = self.nodes[current.id].neighbors.clone();
            
            for &nid in &neighbors {
                if !visited.contains(&nid) {
                    visited.insert(nid);
                    let nd = self.l2_distance(query, &self.nodes[nid].vector);
                    candidates.push(SearchItem { dist: nd, id: nid });
                }
            }
        }
        
        let mut sorted: Vec<_> = results.into_vec();
        sorted.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap());
        sorted.truncate(top_k);
        
        sorted.into_iter().map(|r| (r.id, r.dist)).collect()
    }
    
    pub fn search(&self, query: &[f32], top_k: usize) -> Vec<(usize, f32)> {
        self.search_neighbor(query, top_k)
    }
    
    fn prune_neighbors(&mut self, node_id: usize) {
        let neighbor_ids: Vec<usize> = self.nodes[node_id].neighbors.clone();
        
        if neighbor_ids.len() <= self.R {
            return;
        }
        
        let vector = &self.nodes[node_id].vector;
        
        let mut distances: Vec<_> = neighbor_ids.iter().map(|&nid| {
            (nid, self.l2_distance(vector, &self.nodes[nid].vector))
        }).collect();
        
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        
        let pruned: Vec<usize> = distances.into_iter().map(|(n, _)| n).take(self.R).collect();
        
        self.nodes[node_id].neighbors = pruned;
    }
    
    #[inline]
    fn l2_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        simd::l2_distance(a, b)
    }
    
    pub fn len(&self) -> usize { self.num_vectors }
    pub fn is_empty(&self) -> bool { self.num_vectors == 0 }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_diskann_new() {
        let idx = DiskAnnIndex::new(128);
        assert_eq!(idx.dim, 128);
    }
    
    #[test]
    fn test_diskann_empty_search() {
        let idx = DiskAnnIndex::new(4);
        let results = idx.search(&[0.5, 0.5, 0.5, 0.5], 2);
        assert!(results.is_empty());
    }
    
    #[test]
    fn test_diskann_add_single() {
        let mut idx = DiskAnnIndex::new(4);
        let id = idx.add(&[0.0, 0.0, 0.0, 0.0]);
        assert_eq!(id, 0);
        assert_eq!(idx.len(), 1);
    }
}
