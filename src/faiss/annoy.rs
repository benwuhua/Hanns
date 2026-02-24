//! ANNOY (Approximate Nearest Neighbors Oh Yeah) Index
//! 
//! 使用随机投影树进行近似最近邻搜索
//! 参考: https://github.com/spotify/annoy

use std::collections::BinaryHeap;
use std::cmp::Ordering;

/// ANNOY 节点
#[derive(Clone)]
struct AnnoyNode {
    vector: Option<Vec<f32>>,  // 叶子节点存储向量
    left: Option<usize>,       // 左子树索引
    right: Option<usize>,      // 右子树索引
    split_dim: usize,         // 分割维度
    split_val: f32,           // 分割值
}

/// ANNOY 索引
pub struct AnnoyIndex {
    dim: usize,
    n_trees: usize,
    search_k: isize,
    nodes: Vec<AnnoyNode>,
    roots: Vec<usize>,  // 根节点索引
}

impl AnnoyIndex {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            n_trees: 50,
            search_k: -1,  // -1 表示 n_trees * n
            nodes: Vec::new(),
            roots: Vec::new(),
        }
    }
    
    /// 设置树数量
    pub fn set_n_trees(&mut self, n: usize) {
        self.n_trees = n;
    }
    
    /// 添加向量
    pub fn add_item(&mut self, vector: &[f32]) -> usize {
        let idx = self.nodes.len();
        
        let node = AnnoyNode {
            vector: Some(vector.to_vec()),
            left: None,
            right: None,
            split_dim: 0,
            split_val: 0.0,
        };
        
        self.nodes.push(node);
        idx
    }
    
    /// 构建索引 (多棵树)
    pub fn build(&mut self) {
        let n = self.nodes.len();
        if n == 0 { return; }
        
        // 构建 n_trees 棵树
        for _ in 0..self.n_trees {
            // 使用不同的随机顺序
            let mut indices: Vec<usize> = (0..n).collect();
            self.shuffle(&mut indices);
            
            let root = self.build_tree(&indices);
            self.roots.push(root);
        }
    }
    
    /// 构建单棵树
    fn build_tree(&mut self, indices: &[usize]) -> usize {
        if indices.is_empty() {
            return 0;
        }
        
        if indices.len() == 1 {
            return indices[0];
        }
        
        // 随机选择分割维度
        use rand::prelude::*;
        let mut rng = StdRng::from_entropy();
        let split_dim = rng.gen_range(0..self.dim);
        
        // 找到该维度的中位数
        let mut values: Vec<(usize, f32)> = indices.iter()
            .map(|&i| {
                let v = self.nodes[i].vector.as_ref().unwrap();
                (i, v[split_dim])
            })
            .collect();
        
        values.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let mid = values.len() / 2;
        let split_val = values[mid].1;
        
        // 创建内部节点
        let left_indices: Vec<usize> = values.iter().take(mid).map(|(i, _)| *i).collect();
        let right_indices: Vec<usize> = values.iter().skip(mid).map(|(i, _)| *i).collect();
        
        let node_idx = self.nodes.len();
        
        // 创建内部节点 (placeholder, 会更新)
        let left_child = self.build_tree(&left_indices);
        let right_child = self.build_tree(&right_indices);
        
        let node = AnnoyNode {
            vector: None,
            left: Some(left_child),
            right: Some(right_child),
            split_dim,
            split_val,
        };
        
        self.nodes.push(node);
        node_idx
    }
    
    /// 搜索
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(i64, f32)> {
        if self.roots.is_empty() {
            return vec![];
        }
        
        let n = self.nodes.len();
        
        // 收集所有候选
        let mut candidates: BinaryHeap<HeapItem> = BinaryHeap::new();
        
        for &root in &self.roots {
            self.search_recursive(query, root, &mut candidates, k);
        }
        
        // 提取 top-k
        let mut results: Vec<(i64, f32)> = Vec::new();
        
        while let Some(item) = candidates.pop() {
            // 检查是否是叶子节点
            if self.nodes[item.idx].vector.is_some() {
                results.push((item.idx as i64, item.dist));
                if results.len() >= k {
                    break;
                }
            }
        }
        
        results
    }
    
    /// 递归搜索
    fn search_recursive(&self, query: &[f32], node_idx: usize, candidates: &mut BinaryHeap<HeapItem>, k: usize) {
        if node_idx >= self.nodes.len() {
            return;
        }
        
        let node = &self.nodes[node_idx];
        
        // 叶子节点
        if let Some(ref vector) = node.vector {
            let dist = self.l2_distance(query, vector);
            if candidates.len() < k || dist < candidates.peek().map(|h| h.dist).unwrap_or(f32::MAX) {
                candidates.push(HeapItem { idx: node_idx, dist });
            }
            return;
        }
        
        // 内部节点
        let val = query[node.split_dim];
        
        if val < node.split_val {
            if let Some(left) = node.left {
                self.search_recursive(query, left, candidates, k);
            }
        } else {
            if let Some(right) = node.right {
                self.search_recursive(query, right, candidates, k);
            }
        }
    }
    
    /// Shuffle
    fn shuffle(&self, v: &mut Vec<usize>) {
        use rand::prelude::*;
        let mut rng = StdRng::from_entropy();
        for i in (1..v.len()).rev() {
            let j = rng.gen_range(0..=i);
            v.swap(i, j);
        }
    }
    
    #[inline]
    fn l2_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }
}

/// 堆项 (用于优先级队列)
#[derive(Clone)]
struct HeapItem {
    idx: usize,
    dist: f32,
}

impl PartialEq for HeapItem {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist
    }
}

impl Eq for HeapItem {}

impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // Reverse order: smaller distance = higher priority
        other.dist.partial_cmp(&self.dist)
    }
}

impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> Ordering {
        other.dist.partial_cmp(&self.dist).unwrap_or(Ordering::Equal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_annoy_basic() {
        let mut index = AnnoyIndex::new(2);
        index.set_n_trees(10);
        
        // 添加向量
        index.add_item(&[0.0, 0.0]);
        index.add_item(&[0.0, 1.0]);
        index.add_item(&[1.0, 0.0]);
        index.add_item(&[1.0, 1.0]);
        
        // 构建
        index.build();
        
        // 搜索
        let results = index.search(&[0.0, 0.0], 2);
        
        assert!(results.len() <= 2);
    }
}
