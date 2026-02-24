//! 内存优化 - 连续内存布局工具

use std::slice;

/// 连续内存向量存储
pub struct VectorStore {
    data: Vec<f32>,
    dim: usize,
}

impl VectorStore {
    pub fn new(dim: usize) -> Self {
        Self { data: Vec::new(), dim }
    }
    
    pub fn with_capacity(dim: usize, capacity: usize) -> Self {
        Self { 
            data: Vec::with_capacity(capacity * dim), 
            dim 
        }
    }
    
    pub fn push(&mut self, vector: &[f32]) {
        assert_eq!(vector.len(), self.dim);
        self.data.extend_from_slice(vector);
    }
    
    pub fn get(&self, idx: usize) -> Option<&[f32]> {
        if idx * self.dim >= self.data.len() {
            None
        } else {
            Some(&self.data[idx * self.dim..idx * self.dim + self.dim])
        }
    }
    
    pub fn len(&self) -> usize {
        self.data.len() / self.dim
    }
    
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
    
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }
    
    /// 安全转换为连续数组
    pub fn as_contiguous(&self) -> &[f32] {
        unsafe {
            slice::from_raw_parts(self.data.as_ptr(), self.data.len())
        }
    }
}

/// 扁平化邻接表（图存储）
pub struct GraphStore {
    /// 扁平化边数据
    edges: Vec<usize>,
    /// 每个节点的起始偏移
    offsets: Vec<usize>,
    /// 每个节点的度数
    degrees: Vec<usize>,
}

impl GraphStore {
    pub fn new(capacity: usize) -> Self {
        Self {
            edges: Vec::with_capacity(capacity * 16), // 假设平均度16
            offsets: Vec::with_capacity(capacity + 1),
            degrees: Vec::with_capacity(capacity),
        }
    }
    
    pub fn add_node(&mut self, neighbors: &[usize]) {
        self.offsets.push(self.edges.len());
        self.degrees.push(neighbors.len());
        self.edges.extend_from_slice(neighbors);
    }
    
    pub fn finish(&mut self) {
        self.offsets.push(self.edges.len());
    }
    
    pub fn neighbors(&self, node: usize) -> &[usize] {
        if node + 1 >= self.offsets.len() {
            return &[];
        }
        let start = self.offsets[node];
        let end = self.offsets[node + 1];
        &self.edges[start..end]
    }
    
    pub fn degree(&self, node: usize) -> usize {
        self.degrees.get(node).copied().unwrap_or(0)
    }
    
    pub fn num_nodes(&self) -> usize {
        self.degrees.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_vector_store() {
        let mut store = VectorStore::new(4);
        
        store.push(&[1.0, 2.0, 3.0, 4.0]);
        store.push(&[5.0, 6.0, 7.0, 8.0]);
        
        assert_eq!(store.len(), 2);
        assert_eq!(store.get(0), Some(&[1.0, 2.0, 3.0, 4.0][..]));
    }
    
    #[test]
    fn test_graph_store() {
        let mut graph = GraphStore::new(10);
        
        graph.add_node(&[1, 2, 3]);
        graph.add_node(&[0, 2]);
        graph.finish();
        
        assert_eq!(graph.num_nodes(), 2);
        assert_eq!(graph.neighbors(0), &[1, 2, 3]);
        assert_eq!(graph.degree(0), 3);
    }
}
