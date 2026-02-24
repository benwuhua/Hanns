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
