//! HNSW 图构建完整实现
//!
//! **已废弃**: 请使用 `hnsw.rs`

#![deprecated(since = "0.1.0", note = "Use hnsw.rs instead")]

use std::collections::{BinaryHeap, HashSet};

/// 最小堆项
#[derive(Clone, Debug)]
struct MinItem {
    dist: f32,
    id: usize,
}

impl PartialEq for MinItem {
    fn eq(&self, other: &Self) -> bool { self.dist == other.dist }
}

impl Eq for MinItem {}

impl PartialOrd for MinItem {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.dist.partial_cmp(&other.dist)
    }
}

impl Ord for MinItem {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.dist.partial_cmp(&other.dist).unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// 最大堆项（用于结果）
#[derive(Clone, Debug)]
struct MaxItem {
    dist: f32,
    id: usize,
}

impl PartialEq for MaxItem {
    fn eq(&self, other: &Self) -> bool { self.dist == other.dist }
}

impl Eq for MaxItem {}

impl PartialOrd for MaxItem {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        other.dist.partial_cmp(&self.dist)
    }
}

impl Ord for MaxItem {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.dist.partial_cmp(&self.dist).unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// HNSW 节点
#[derive(Clone)]
pub struct HnswNode {
    pub id: usize,
    pub level: usize,
    pub neighbors: Vec<Vec<usize>>, // 每层的邻居
}

/// HNSW 索引（完整构建版）
pub struct HnswIndex {
    pub dim: usize,
    pub m: usize,                  // 基础邻居数
    pub ef_construction: usize,     // 建图宽度
    pub max_level: usize,           // 最大层数
    pub ml: f32,                   // 层高参数
    
    vectors: Vec<f32>,
    nodes: Vec<HnswNode>,
    entry_point: Option<usize>,
    entry_level: usize,
}

impl HnswIndex {
    pub fn new(dim: usize, m: usize, ef_construction: usize) -> Self {
        Self {
            dim,
            m,
            ef_construction,
            max_level: 16,
            ml: 1.0 / (dim as f32).sqrt(),
            vectors: Vec::new(),
            nodes: Vec::new(),
            entry_point: None,
            entry_level: 0,
        }
    }
    
    /// 随机层数（基于几何分布）
    fn random_level(&self) -> usize {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut h = DefaultHasher::new();
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
            .hash(&mut h);
        h.finish().hash(&mut h);
        
        let r: f64 = (h.finish() as f64) / (u64::MAX as f64);
        let level = (-r.ln() / self.ml.ln() as f64) as usize;
        level.min(self.max_level)
    }
    
    /// 添加向量
    pub fn add(&mut self, vector: &[f32]) -> usize {
        let id = self.vectors.len() / self.dim;
        
        // 扩展向量存储
        self.vectors.extend_from_slice(vector);
        
        // 随机层数
        let level = self.random_level();
        
        // 创建节点
        let mut node = HnswNode {
            id,
            level,
            neighbors: (0..=level).map(|_| Vec::new()).collect(),
        };
        
        // 如果是第一个节点
        if id == 0 {
            self.entry_point = Some(0);
            self.entry_level = 0;
            self.nodes.push(node);
            return 0;
        }
        
        // 先添加节点（确保节点存在）
        self.nodes.push(node);
        
        // 搜索插入位置
        let mut entered = Vec::new();
        self.search_insert_point(vector, &mut entered);
        
        // 从最高层开始添加边
        for l in (0..=level).rev() {
            // 获取当前层的候选
            let candidates: Vec<usize> = entered.iter().take(self.ef_construction).copied().collect();
            
            // 添加双向边
            for &c in &candidates {
                self.add_edge(id, c, l);
            }
            
            // 剪枝：保持邻居数量上限
            self.prune_neighbors(id, l);
        }
        
        id
    }
    
    /// 搜索插入点（从顶向下）
    fn search_insert_point(&self, query: &[f32], entered: &mut Vec<usize>) {
        let mut current = self.entry_point.unwrap_or(0);
        let mut dist = self.l2_distance(query, current);
        
        entered.push(current);
        
        for level in (0..self.entry_level).rev() {
            let mut changed = true;
            while changed {
                changed = false;
                
                // 获取当前节点的邻居
                let level_neighbors: Vec<usize> = self.nodes[current].neighbors[level].clone();
                
                for &neighbor in &level_neighbors {
                    let d = self.l2_distance(query, neighbor);
                    if d < dist {
                        dist = d;
                        current = neighbor;
                        entered.push(current);
                        changed = true;
                    }
                }
            }
        }
    }
    
    /// 添加边
    fn add_edge(&mut self, a: usize, b: usize, level: usize) {
        if level >= self.nodes[a].neighbors.len() {
            self.nodes[a].neighbors.resize(level + 1, Vec::new());
        }
        if level >= self.nodes[b].neighbors.len() {
            self.nodes[b].neighbors.resize(level + 1, Vec::new());
        }
        
        if !self.nodes[a].neighbors[level].contains(&b) {
            self.nodes[a].neighbors[level].push(b);
        }
        if !self.nodes[b].neighbors[level].contains(&a) {
            self.nodes[b].neighbors[level].push(a);
        }
    }
    
    /// 剪枝：保持邻居数量上限
    fn prune_neighbors(&mut self, node_id: usize, level: usize) {
        // 先获取邻居ID和距离，避免同时可变和不可变借用
        let neighbors_ids: Vec<usize> = self.nodes[node_id].neighbors[level].clone();
        
        if neighbors_ids.len() <= self.m {
            return;
        }
        
        // 按距离排序，保留最近的 m 个
        let query = &self.vectors[node_id * self.dim..];
        
        let mut distances: Vec<_> = neighbors_ids.iter().map(|&n| {
            (n, self.l2_distance(query, n))
        }).collect();
        
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        
        let pruned: Vec<usize> = distances.into_iter().map(|(n, _)| n).take(self.m).collect();
        
        self.nodes[node_id].neighbors[level] = pruned;
    }
    
    /// 搜索最近邻
    pub fn search(&self, query: &[f32], ef: usize, top_k: usize) -> Vec<(usize, f32)> {
        if self.nodes.is_empty() {
            return vec![];
        }
        
        // 贪婪搜索
        let mut visited = HashSet::new();
        let mut candidates: BinaryHeap<MinItem> = BinaryHeap::new();
        let mut results: BinaryHeap<MaxItem> = BinaryHeap::new();
        
        let mut current = self.entry_point.unwrap_or(0);
        let mut dist = self.l2_distance(query, current);
        
        visited.insert(current);
        candidates.push(MinItem { dist, id: current });
        
        while let Some(MinItem { dist: d, id: node }) = candidates.pop() {
            // 提前退出
            if results.len() >= ef && d > results.peek().map(|item| item.dist).unwrap_or(f32::MAX) {
                break;
            }
            
            if !results.iter().any(|item| item.id == node) {
                results.push(MaxItem { dist: d, id: node });
            }
            
            // 遍历邻居
            for &neighbor in &self.nodes[node].neighbors[0] {
                if !visited.contains(&neighbor) {
                    visited.insert(neighbor);
                    let nd = self.l2_distance(query, neighbor);
                    candidates.push(MinItem { dist: nd, id: neighbor });
                }
            }
        }
        
        // 排序返回 top_k
        let mut sorted: Vec<_> = results.into_vec();
        sorted.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap());
        sorted.truncate(top_k);
        
        sorted.into_iter().map(|item| (item.id, item.dist)).collect()
    }
    
    #[inline]
    fn l2_distance(&self, query: &[f32], idx: usize) -> f32 {
        let start = idx * self.dim;
        let mut sum = 0.0f32;
        for i in 0..self.dim {
            let diff = query[i] - self.vectors[start + i];
            sum += diff * diff;
        }
        sum
    }
    
    pub fn num_vectors(&self) -> usize {
        self.vectors.len() / self.dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_hnsw_new() {
        let hnsw = HnswIndex::new(128, 16, 200);
        assert_eq!(hnsw.dim, 128);
        assert_eq!(hnsw.m, 16);
    }
    
    #[test]
    fn test_hnsw_add_single() {
        let mut hnsw = HnswIndex::new(4, 2, 10);
        let v = vec![1.0, 2.0, 3.0, 4.0];
        let id = hnsw.add(&v);
        assert_eq!(id, 0);
        assert_eq!(hnsw.num_vectors(), 1);
    }
    
    #[test]
    fn test_hnsw_add_multiple() {
        let mut hnsw = HnswIndex::new(4, 2, 10);
        
        hnsw.add(&[1.0, 0.0, 0.0, 0.0]);
        hnsw.add(&[0.0, 1.0, 0.0, 0.0]);
        hnsw.add(&[0.0, 0.0, 1.0, 0.0]);
        
        assert_eq!(hnsw.num_vectors(), 3);
    }
    
    #[test]
    fn test_hnsw_search() {
        let mut hnsw = HnswIndex::new(4, 2, 10);
        
        // 添加向量
        hnsw.add(&[0.0, 0.0, 0.0, 0.0]);
        hnsw.add(&[1.0, 0.0, 0.0, 0.0]);
        hnsw.add(&[10.0, 0.0, 0.0, 0.0]);
        
        // 搜索
        let query = vec![0.5, 0.0, 0.0, 0.0];
        let results = hnsw.search(&query, 10, 2);
        
        assert!(!results.is_empty());
        // 验证返回了有效结果
        assert!(results[0].0 < 3);
    }
}
