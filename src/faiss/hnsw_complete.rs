//! HNSW 完整实现
//! 多层跳表结构 + 完整搜索 + 序列化

use std::collections::{BinaryHeap, HashMap, HashSet};
use std::cmp::Ordering;

/// 搜索结果项
#[derive(Clone, Debug)]
struct SearchItem {
    id: usize,
    dist: f32,
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

/// HNSW 节点
#[derive(Clone)]
struct Node {
    /// 向量数据
    vector: Vec<f32>,
    /// 每层的邻居 [level -> neighbors]
    neighbors: Vec<Vec<usize>>,
}

impl Node {
    fn new(vector: Vec<f32>, level: usize) -> Self {
        Self {
            vector,
            neighbors: (0..=level).map(|_| Vec::new()).collect(),
        }
    }
}

/// HNSW 索引完整实现
pub struct HnswIndex {
    /// 向量维度
    pub dim: usize,
    /// 基础邻居数
    pub m: usize,
    /// 建图宽度
    pub ef_construction: usize,
    /// 搜索宽度
    pub ef_search: usize,
    /// 最大层数
    pub max_level: usize,
    /// 层高参数
    pub ml: f32,
    
    /// 所有节点
    nodes: Vec<Node>,
    /// 入口点
    entry_point: Option<usize>,
    /// 入口层
    entry_level: usize,
    /// 节点数
    num_vectors: usize,
}

impl HnswIndex {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            m: 16,
            ef_construction: 200,
            ef_search: 50,
            max_level: 16,
            ml: 1.0 / (dim as f32).sqrt(),
            nodes: Vec::new(),
            entry_point: None,
            entry_level: 0,
            num_vectors: 0,
        }
    }
    
    pub fn with_params(mut self, m: usize, ef_construction: usize, ef_search: usize) -> Self {
        self.m = m;
        self.ef_construction = ef_construction;
        self.ef_search = ef_search;
        self
    }
    
    /// 生成随机层数（几何分布）
    fn random_level(&self) -> usize {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut h = DefaultHasher::new();
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
            .hash(&mut h);
        let r = (h.finish() as f64) / (u64::MAX as f64);
        let level = (-r.ln() / (self.ml as f64).ln()) as usize;
        level.min(self.max_level)
    }
    
    /// 计算 L2 距离
    #[inline]
    fn l2_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        let mut sum = 0.0f32;
        for i in 0..self.dim {
            let diff = a[i] - b[i];
            sum += diff * diff;
        }
        sum
    }
    
    /// 添加向量
    pub fn add(&mut self, vector: &[f32]) -> usize {
        let id = self.num_vectors;
        let level = self.random_level();
        
        // 创建新节点
        let mut node = Node::new(vector.to_vec(), level);
        
        // 第一个节点
        if id == 0 {
            self.entry_point = Some(0);
            self.entry_level = 0;
            self.nodes.push(node);
            self.num_vectors += 1;
            return 0;
        }
        
        // 搜索插入位置
        let mut entered = Vec::new();
        self.search_insert_point(vector, &mut entered);
        
        // 添加节点
        self.nodes.push(node);
        
        // 在各层添加边
        for l in (0..=level).rev() {
            let candidates: Vec<usize> = entered.iter().take(self.ef_construction).copied().collect();
            
            // 添加双向边
            for &c in &candidates {
                self.add_edge(id, c, l);
            }
            
            // 剪枝
            self.prune_neighbors(id, l);
        }
        
        self.num_vectors += 1;
        id
    }
    
    /// 批量添加
    pub fn add_batch(&mut self, vectors: &[f32]) -> usize {
        let n = vectors.len() / self.dim;
        for i in 0..n {
            self.add(&vectors[i * self.dim..]);
        }
        n
    }
    
    /// 搜索插入点
    fn search_insert_point(&self, query: &[f32], entered: &mut Vec<usize>) {
        let mut current = self.entry_point.unwrap_or(0);
        let mut dist = self.l2_distance(query, &self.nodes[current].vector);
        
        entered.push(current);
        
        for level in (0..self.entry_level).rev() {
            loop {
                let mut found_better = false;
                
                let neighbors: Vec<usize> = self.nodes[current].neighbors.get(level).cloned().unwrap_or_default();
                
                for &neighbor in &neighbors {
                    let d = self.l2_distance(query, &self.nodes[neighbor].vector);
                    if d < dist {
                        dist = d;
                        current = neighbor;
                        entered.push(current);
                        found_better = true;
                    }
                }
                
                if !found_better { break; }
            }
        }
    }
    
    /// 添加边
    fn add_edge(&mut self, a: usize, b: usize, level: usize) {
        // 扩展层的邻居列表
        for _ in self.nodes[a].neighbors.len()..=level {
            self.nodes[a].neighbors.push(Vec::new());
        }
        for _ in self.nodes[b].neighbors.len()..=level {
            self.nodes[b].neighbors.push(Vec::new());
        }
        
        // 添加双向边
        if !self.nodes[a].neighbors[level].contains(&b) {
            self.nodes[a].neighbors[level].push(b);
        }
        if !self.nodes[b].neighbors[level].contains(&a) {
            self.nodes[b].neighbors[level].push(a);
        }
    }
    
    /// 剪枝
    fn prune_neighbors(&mut self, node_id: usize, level: usize) {
        if level >= self.nodes[node_id].neighbors.len() {
            return;
        }
        
        let neighbors_ids: Vec<usize> = self.nodes[node_id].neighbors[level].clone();
        
        if neighbors_ids.len() <= self.m {
            return;
        }
        
        // 按距离排序
        let query = &self.nodes[node_id].vector;
        let mut distances: Vec<_> = neighbors_ids.iter().map(|&n| {
            (n, self.l2_distance(query, &self.nodes[n].vector))
        }).collect();
        
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        
        let pruned: Vec<usize> = distances.into_iter().map(|(n, _)| n).take(self.m).collect();
        
        self.nodes[node_id].neighbors[level] = pruned;
    }
    
    /// 搜索最近邻
    pub fn search(&self, query: &[f32], top_k: usize) -> Vec<(usize, f32)> {
        if self.nodes.is_empty() {
            return vec![];
        }
        
        // 贪婪搜索
        let mut visited = HashSet::new();
        let mut candidates: BinaryHeap<SearchItem> = BinaryHeap::new();
        let mut results: BinaryHeap<SearchItem> = BinaryHeap::new();
        
        let mut current = self.entry_point.unwrap_or(0);
        let mut dist = self.l2_distance(query, &self.nodes[current].vector);
        
        visited.insert(current);
        candidates.push(SearchItem { id: current, dist });
        
        while let Some(current) = candidates.pop() {
            // 提前退出
            if results.len() >= self.ef_search && 
               current.dist > results.peek().map(|r| r.dist).unwrap_or(f32::MAX) {
                break;
            }
            
            // 跳过已添加
            if results.iter().any(|r| r.id == current.id) {
                continue;
            }
            
            results.push(current.clone());
            
            // 遍历第0层邻居
            for &neighbor in &self.nodes[current.id].neighbors[0] {
                if !visited.contains(&neighbor) {
                    visited.insert(neighbor);
                    let nd = self.l2_distance(query, &self.nodes[neighbor].vector);
                    candidates.push(SearchItem { id: neighbor, dist: nd });
                }
            }
        }
        
        // 排序返回 top_k
        let mut sorted: Vec<_> = results.into_vec();
        sorted.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(Ordering::Equal));
        sorted.truncate(top_k);
        
        sorted.into_iter().map(|r| (r.id, r.dist)).collect()
    }
    
    /// 批量搜索
    pub fn search_batch(&self, queries: &[f32], top_k: usize) -> Vec<Vec<(usize, f32)>> {
        let n = queries.len() / self.dim;
        (0..n).map(|i| {
            self.search(&queries[i * self.dim..], top_k)
        }).collect()
    }
    
    /// 获取向量
    pub fn get_vector(&self, id: usize) -> Option<&[f32]> {
        self.nodes.get(id).map(|n| n.vector.as_slice())
    }
    
    /// 节点数
    pub fn len(&self) -> usize { self.num_vectors }
    
    pub fn is_empty(&self) -> bool { self.num_vectors == 0 }
    
    /// 序列化
    pub fn serialize(&self) -> Vec<u8> {
        let mut data = Vec::new();
        
        // 写入 header
        data.extend_from_slice(&self.dim.to_le_bytes());
        data.extend_from_slice(&self.m.to_le_bytes());
        data.extend_from_slice(&self.num_vectors.to_le_bytes());
        data.extend_from_slice(&self.entry_point.unwrap_or(0).to_le_bytes());
        data.extend_from_slice(&self.entry_level.to_le_bytes());
        
        // 写入向量
        for node in &self.nodes {
            for v in &node.vector {
                data.extend_from_slice(&v.to_le_bytes());
            }
        }
        
        // 写入邻居 - 使用最大层数
        let max_lvl = self.nodes.iter().map(|n| n.neighbors.len()).max().unwrap_or(1);
        data.extend_from_slice(&(max_lvl as u32).to_le_bytes());
        
        for node in &self.nodes {
            for level in 0..max_lvl {
                let neighbors = node.neighbors.get(level).map(|n| n.as_slice()).unwrap_or(&[]);
                data.extend_from_slice(&(neighbors.len() as u32).to_le_bytes());
                for &n in neighbors {
                    data.extend_from_slice(&n.to_le_bytes());
                }
            }
        }
        
        data
    }
    
    /// 反序列化
    pub fn deserialize(&mut self, data: &[u8]) -> bool {
        if data.len() < 24 { return false; }
        
        let mut pos = 0;
        
        // 读取 header
        self.dim = usize::from_le_bytes(data[pos..pos+8].try_into().unwrap());
        pos += 8;
        self.m = usize::from_le_bytes(data[pos..pos+8].try_into().unwrap());
        pos += 8;
        let num = usize::from_le_bytes(data[pos..pos+8].try_into().unwrap());
        pos += 8;
        let entry = usize::from_le_bytes(data[pos..pos+8].try_into().unwrap());
        pos += 8;
        self.entry_level = usize::from_le_bytes(data[pos..pos+8].try_into().unwrap());
        pos += 8;
        
        self.entry_point = if entry == 0 && num > 0 { Some(0) } else { Some(entry) };
        
        // 读取向量
        self.nodes.clear();
        for _ in 0..num {
            let mut vector = vec![0.0f32; self.dim];
            for v in &mut vector {
                *v = f32::from_le_bytes(data[pos..pos+4].try_into().unwrap());
                pos += 4;
            }
            self.nodes.push(Node::new(vector, 0));
        }
        
        // 读取最大层数
        let max_lvl = u32::from_le_bytes(data[pos..pos+4].try_into().unwrap()) as usize;
        pos += 4;
        
        // 读取邻居
        for node in &mut self.nodes {
            node.neighbors.clear();
            for _ in 0..max_lvl {
                let len = u32::from_le_bytes(data[pos..pos+4].try_into().unwrap()) as usize;
                pos += 4;
                let mut neighbors = Vec::with_capacity(len);
                for _ in 0..len {
                    let n = usize::from_le_bytes(data[pos..pos+8].try_into().unwrap());
                    pos += 8;
                    neighbors.push(n);
                }
                node.neighbors.push(neighbors);
            }
        }
        
        self.num_vectors = num;
        true
    }
}

impl Default for HnswIndex {
    fn default() -> Self { Self::new(128) }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_hnsw_new() {
        let hnsw = HnswIndex::new(128);
        assert_eq!(hnsw.dim, 128);
    }
    
    #[test]
    fn test_hnsw_add() {
        let mut hnsw = HnswIndex::new(4);
        hnsw.m = 2;
        
        hnsw.add(&[0.0, 0.0, 0.0, 0.0]);
        hnsw.add(&[1.0, 0.0, 0.0, 0.0]);
        
        assert_eq!(hnsw.len(), 2);
    }
    
    #[test]
    fn test_hnsw_search() {
        let mut hnsw = HnswIndex::new(4);
        hnsw.m = 2;
        hnsw.ef_search = 10;
        
        hnsw.add(&[0.0, 0.0, 0.0, 0.0]);
        hnsw.add(&[1.0, 0.0, 0.0, 0.0]);
        hnsw.add(&[10.0, 0.0, 0.0, 0.0]);
        
        let results = hnsw.search(&[0.5, 0.0, 0.0, 0.0], 2);
        
        assert!(!results.is_empty());
    }
    
    #[test]
    fn test_hnsw_serialize() {
        let mut hnsw = HnswIndex::new(4);
        hnsw.m = 2;
        
        hnsw.add(&[1.0, 2.0, 3.0, 4.0]);
        hnsw.add(&[5.0, 6.0, 7.0, 8.0]);
        
        let data = hnsw.serialize();
        
        let mut hnsw2 = HnswIndex::new(4);
        hnsw2.deserialize(&data);
        
        assert_eq!(hnsw2.len(), 2);
    }
    
    #[test]
    fn test_hnsw_batch() {
        let mut hnsw = HnswIndex::new(4);
        hnsw.m = 2;
        
        let vectors = vec![
            0.0, 0.0, 0.0, 0.0,
            1.0, 0.0, 0.0, 0.0,
            2.0, 0.0, 0.0, 0.0,
        ];
        
        hnsw.add_batch(&vectors);
        
        assert_eq!(hnsw.len(), 3);
    }
}
