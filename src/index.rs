//! Index Trait 定义与测试
//! 
//! 统一的索引接口

use crate::api::KnowhereError;
use crate::dataset::Dataset;
use crate::bitset::BitsetView;

/// 索引错误
#[derive(Debug)]
pub enum IndexError {
    /// 未训练
    NotTrained,
    /// 索引为空
    Empty,
    /// 维度不匹配
    DimMismatch,
    /// 不支持的操作
    Unsupported(String),
}

impl std::fmt::Display for IndexError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IndexError::NotTrained => write!(f, "Index not trained"),
            IndexError::Empty => write!(f, "Index is empty"),
            IndexError::DimMismatch => write!(f, "Dimension mismatch"),
            IndexError::Unsupported(msg) => write!(f, "Unsupported: {}", msg),
        }
    }
}

impl std::error::Error for IndexError {}

/// 索引 trait（统一接口）
pub trait Index: Send + Sync {
    /// 获取索引类型名称
    fn index_type(&self) -> &str;
    
    /// 获取维度
    fn dim(&self) -> usize;
    
    /// 获取向量数量
    fn count(&self) -> usize;
    
    /// 是否已训练
    fn is_trained(&self) -> bool;
    
    /// 训练索引
    fn train(&mut self, dataset: &Dataset) -> Result<(), IndexError>;
    
    /// 添加向量
    fn add(&mut self, dataset: &Dataset) -> Result<usize, IndexError>;
    
    /// 搜索
    fn search(&self, query: &Dataset, top_k: usize) -> Result<SearchResult, IndexError>;
    
    /// 范围搜索 (Range search)
    fn range_search(&self, query: &Dataset, radius: f32) -> Result<SearchResult, IndexError> {
        // 默认实现：使用 K=所有向量，然后过滤
        Err(IndexError::Unsupported("range_search not implemented".into()))
    }

    /// 搜索时使用 Bitset 过滤
    /// 
    /// Bitset 用于过滤掉某些向量（例如已删除的向量）。
    /// Bitset 参数是一个位图，每个 bit 代表一个向量是否被过滤（1=过滤，0=保留）。
    /// 
    /// # Arguments
    /// * `query` - 查询向量
    /// * `top_k` - 返回的最近邻数量
    /// * `bitset` - BitsetView，用于过滤向量
    /// 
    /// # Returns
    /// 返回搜索结果，不包含被过滤的向量
    fn search_with_bitset(&self, query: &Dataset, top_k: usize, bitset: &crate::bitset::BitsetView) -> Result<SearchResult, IndexError> {
        // 默认实现：先搜索，然后过滤
        let mut result = self.search(query, top_k)?;
        
        // 过滤掉 bitset 中标记为 1 的向量
        let mut filtered_ids = Vec::new();
        let mut filtered_distances = Vec::new();
        
        for (id, dist) in result.ids.iter().zip(result.distances.iter()) {
            // 检查 ID 是否在 bitset 范围内
            let idx = *id as usize;
            if idx < bitset.len() && !bitset.get(idx) {
                // 未被过滤，保留
                filtered_ids.push(*id);
                filtered_distances.push(*dist);
            }
        }
        
        result.ids = filtered_ids;
        result.distances = filtered_distances;
        Ok(result)
    }

    /// 按 ID 获取向量 (GetVectorByIds)
    fn get_vector_by_ids(&self, ids: &[i64]) -> Result<Vec<f32>, IndexError> {
        Err(IndexError::Unsupported("get_vector_by_ids not implemented".into()))
    }

    /// 序列化到内存 (BinarySet)
    fn serialize_to_memory(&self) -> Result<Vec<u8>, IndexError> {
        Err(IndexError::Unsupported("serialize_to_memory not implemented".into()))
    }

    /// 从内存反序列化 (BinarySet)
    fn deserialize_from_memory(&mut self, data: &[u8]) -> Result<(), IndexError> {
        Err(IndexError::Unsupported("deserialize_from_memory not implemented".into()))
    }
    
    /// 保存到文件
    fn save(&self, path: &str) -> Result<(), IndexError>;
    
    /// 从文件加载
    fn load(&mut self, path: &str) -> Result<(), IndexError>;
}

/// 搜索结果
#[derive(Debug)]
pub struct SearchResult {
    /// 最近的邻居 ID
    pub ids: Vec<i64>,
    /// 距离
    pub distances: Vec<f32>,
    /// 搜索耗时（毫秒）
    pub elapsed_ms: f64,
}

impl SearchResult {
    pub fn new(ids: Vec<i64>, distances: Vec<f32>, elapsed_ms: f64) -> Self {
        Self { ids, distances, elapsed_ms }
    }
    
    /// 获取结果数量
    pub fn len(&self) -> usize {
        self.ids.len()
    }
    
    /// 是否为空
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::Dataset;
    
    #[test]
    fn test_search_result() {
        let result = SearchResult::new(
            vec![1, 2, 3],
            vec![0.1, 0.2, 0.3],
            1.5,
        );
        
        assert_eq!(result.len(), 3);
        assert!(!result.is_empty());
        assert_eq!(result.ids[0], 1);
    }
    
    #[test]
    fn test_index_error_display() {
        let err = IndexError::NotTrained;
        assert!(err.to_string().contains("trained"));
    }
}
