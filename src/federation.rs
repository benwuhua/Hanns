//! Federation 信息结构
//! 
//! 用于返回搜索结果的详细信息

use serde::{Deserialize, Serialize};

/// 搜索联邦信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederationInfo {
    /// 访问的节点数
    pub num_visited: usize,
    /// 过滤掉的节点数
    pub num_filtered: usize,
    /// 搜索范围
    pub search_range: Option<(f32, f32)>,
    /// 粗筛结果数
    pub coarse_results: usize,
    /// 细筛结果数
    pub fine_results: usize,
}

impl FederationInfo {
    pub fn new() -> Self {
        Self {
            num_visited: 0,
            num_filtered: 0,
            search_range: None,
            coarse_results: 0,
            fine_results: 0,
        }
    }
    
    pub fn with_visited(mut self, n: usize) -> Self {
        self.num_visited = n;
        self
    }
    
    pub fn with_filtered(mut self, n: usize) -> Self {
        self.num_filtered = n;
        self
    }
}

impl Default for FederationInfo {
    fn default() -> Self {
        Self::new()
    }
}

/// 统计信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Statistics {
    /// 索引向量数
    pub num_vectors: usize,
    /// 维度
    pub dim: usize,
    /// 索引类型
    pub index_type: String,
    /// 原始文件大小 (bytes)
    pub raw_file_size: Option<u64>,
    /// 索引文件大小 (bytes)
    pub index_file_size: Option<u64>,
    /// 内存使用 (bytes)
    pub memory_usage: Option<u64>,
}

impl Statistics {
    pub fn new(num_vectors: usize, dim: usize, index_type: &str) -> Self {
        Self {
            num_vectors,
            dim,
            index_type: index_type.to_string(),
            raw_file_size: None,
            index_file_size: None,
            memory_usage: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_federation_info() {
        let info = FederationInfo::new()
            .with_visited(100)
            .with_filtered(50);
        
        assert_eq!(info.num_visited, 100);
        assert_eq!(info.num_filtered, 50);
    }
    
    #[test]
    fn test_statistics() {
        let stats = Statistics::new(1000, 128, "HNSW");
        assert_eq!(stats.num_vectors, 1000);
        assert_eq!(stats.dim, 128);
    }
}
