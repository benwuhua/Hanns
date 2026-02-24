//! Dataset 定义与测试
//! 
//! 对齐 C++ 的 Dataset 结构

use crate::bitset::BitsetView;

/// 数据类型枚举
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    /// 32位浮点数
    Float32,
    /// 8位整数
    Int8,
    /// 8位无符号整数
    UInt8,
    /// 16位整数
    Int16,
    /// 32位整数
    Int32,
    /// 64位整数
    Int64,
}

impl DataType {
    /// 获取数据类型的大小（字节）
    pub fn size(&self) -> usize {
        match self {
            DataType::Float32 => 4,
            DataType::Int8 | DataType::UInt8 => 1,
            DataType::Int16 => 2,
            DataType::Int32 => 4,
            DataType::Int64 => 8,
        }
    }
}

/// 向量数据集
/// 
/// # 设计目标
/// - 支持多种数据类型
/// - 支持带 ID 的向量
/// - 与 C++ Dataset 对齐
#[derive(Clone)]
pub struct Dataset {
    /// 向量数据（扁平存储）
    vectors: Vec<f32>,
    /// 向量维度
    dim: usize,
    /// 向量数量
    num_vectors: usize,
    /// 向量 ID（可选）
    ids: Option<Vec<i64>>,
    /// 数据类型
    data_type: DataType,
    /// 软删除位图
    deleted: Option<BitsetView>,
}

impl Dataset {
    /// 从向量数组创建数据集
    pub fn from_vectors(vectors: Vec<f32>, dim: usize) -> Self {
        let num_vectors = vectors.len() / dim;
        Self {
            vectors,
            dim,
            num_vectors,
            ids: None,
            data_type: DataType::Float32,
            deleted: None,
        }
    }
    
    /// 创建带 ID 的数据集
    pub fn from_vectors_with_ids(vectors: Vec<f32>, dim: usize, ids: Vec<i64>) -> Self {
        let num_vectors = vectors.len() / dim;
        assert_eq!(ids.len(), num_vectors, "IDs count must match vector count");
        
        Self {
            vectors,
            dim,
            num_vectors,
            ids: Some(ids),
            data_type: DataType::Float32,
            deleted: None,
        }
    }
    
    /// 获取向量数量
    pub fn num_vectors(&self) -> usize {
        self.num_vectors
    }
    
    /// 获取维度
    pub fn dim(&self) -> usize {
        self.dim
    }
    
    /// 获取向量数据
    pub fn vectors(&self) -> &[f32] {
        &self.vectors
    }
    
    /// 获取向量数据（可变）
    pub fn vectors_mut(&mut self) -> &mut [f32] {
        &mut self.vectors
    }
    
    /// 获取指定索引的向量
    pub fn get_vector(&self, idx: usize) -> Option<&[f32]> {
        if idx >= self.num_vectors {
            return None;
        }
        
        let start = idx * self.dim;
        let end = start + self.dim;
        Some(&self.vectors[start..end])
    }
    
    /// 获取 IDs
    pub fn ids(&self) -> Option<&[i64]> {
        self.ids.as_deref()
    }
    
    /// 设置 IDs
    pub fn set_ids(&mut self, ids: Vec<i64>) {
        assert_eq!(ids.len(), self.num_vectors);
        self.ids = Some(ids);
    }
    
    /// 获取数据大小（字节）
    pub fn data_size(&self) -> usize {
        self.vectors.len() * std::mem::size_of::<f32>()
    }
    
    /// 设置软删除
    pub fn set_deleted(&mut self, deleted: BitsetView) {
        self.deleted = Some(deleted);
    }
    
    /// 获取软删除位图
    pub fn deleted(&self) -> Option<&BitsetView> {
        self.deleted.as_ref()
    }
    
    /// 检查向量是否已删除
    pub fn is_deleted(&self, idx: usize) -> bool {
        self.deleted
            .as_ref()
            .map(|d| d.get(idx))
            .unwrap_or(false)
    }
    
    /// 获取有效向量数量
    pub fn num_valid_vectors(&self) -> usize {
        if let Some(deleted) = &self.deleted {
            self.num_vectors - deleted.count()
        } else {
            self.num_vectors
        }
    }
}

impl Default for Dataset {
    fn default() -> Self {
        Self {
            vectors: Vec::new(),
            dim: 0,
            num_vectors: 0,
            ids: None,
            data_type: DataType::Float32,
            deleted: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_dataset_basic() {
        let vectors = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let dataset = Dataset::from_vectors(vectors, 2);
        
        assert_eq!(dataset.num_vectors(), 3);
        assert_eq!(dataset.dim(), 2);
        assert_eq!(dataset.data_size(), 24);
    }
    
    #[test]
    fn test_dataset_with_ids() {
        let vectors = vec![1.0, 2.0, 3.0, 4.0];
        let ids = vec![10, 20];
        let dataset = Dataset::from_vectors_with_ids(vectors, 2, ids.clone());
        
        let stored_ids = dataset.ids().unwrap();
        assert_eq!(stored_ids, &ids);
    }
    
    #[test]
    fn test_dataset_get_vector() {
        let vectors = vec![
            1.0, 2.0, 3.0,  // vector 0
            4.0, 5.0, 6.0,  // vector 1
        ];
        let dataset = Dataset::from_vectors(vectors, 3);
        
        let v0 = dataset.get_vector(0).unwrap();
        assert_eq!(v0, &[1.0, 2.0, 3.0]);
        
        let v1 = dataset.get_vector(1).unwrap();
        assert_eq!(v1, &[4.0, 5.0, 6.0]);
        
        assert!(dataset.get_vector(2).is_none());
    }
    
    #[test]
    fn test_dataset_deleted() {
        let vectors = vec![1.0, 2.0, 3.0, 4.0];
        let mut dataset = Dataset::from_vectors(vectors, 2);
        
        // 默认没有删除
        assert_eq!(dataset.num_valid_vectors(), 2);
        assert!(!dataset.is_deleted(0));
        
        // 添加删除位图
        let mut deleted = BitsetView::new(2);
        deleted.set_bit(0);
        dataset.set_deleted(deleted);
        
        assert!(dataset.is_deleted(0));
        assert!(!dataset.is_deleted(1));
        assert_eq!(dataset.num_valid_vectors(), 1);
    }
    
    #[test]
    fn test_datatype_size() {
        assert_eq!(DataType::Float32.size(), 4);
        assert_eq!(DataType::Int8.size(), 1);
        assert_eq!(DataType::Int16.size(), 2);
        assert_eq!(DataType::Int32.size(), 4);
        assert_eq!(DataType::Int64.size(), 8);
        assert_eq!(DataType::UInt8.size(), 1);
    }
}
