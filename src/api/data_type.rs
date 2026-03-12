//! Data type enumeration for vectors
//!
//! Corresponds to C++ VecType in knowhere/comp/index_param.h

use serde::{Deserialize, Serialize};

/// Vector data type
///
/// Values aligned with Milvus proto definitions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
#[repr(i32)]
pub enum DataType {
    /// Binary vectors (100)
    Binary = 100,
    /// Float32 vectors (101)
    #[default]
    Float = 101,
    /// Float16 vectors (102)
    Float16 = 102,
    /// BFloat16 vectors (103)
    BFloat16 = 103,
    /// Sparse float vectors (104)
    SparseFloat = 104,
    /// Int8 vectors (105)
    Int8 = 105,
}

impl DataType {
    /// Convert from i32 value (Milvus proto)
    pub fn from_i32(value: i32) -> Option<Self> {
        match value {
            100 => Some(DataType::Binary),
            101 => Some(DataType::Float),
            102 => Some(DataType::Float16),
            103 => Some(DataType::BFloat16),
            104 => Some(DataType::SparseFloat),
            105 => Some(DataType::Int8),
            _ => None,
        }
    }

    /// Convert to i32 value (Milvus proto)
    pub fn to_i32(self) -> i32 {
        self as i32
    }

    /// Get string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            DataType::Binary => "binary",
            DataType::Float => "float",
            DataType::Float16 => "float16",
            DataType::BFloat16 => "bfloat16",
            DataType::SparseFloat => "sparse_float",
            DataType::Int8 => "int8",
        }
    }

    /// Check if this is a dense floating-point type
    pub fn is_dense_float(&self) -> bool {
        matches!(
            self,
            DataType::Float | DataType::Float16 | DataType::BFloat16
        )
    }

    /// Check if this is a binary type
    pub fn is_binary(&self) -> bool {
        matches!(self, DataType::Binary)
    }

    /// Check if this is a sparse type
    pub fn is_sparse(&self) -> bool {
        matches!(self, DataType::SparseFloat)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_type_from_i32() {
        assert_eq!(DataType::from_i32(100), Some(DataType::Binary));
        assert_eq!(DataType::from_i32(101), Some(DataType::Float));
        assert_eq!(DataType::from_i32(102), Some(DataType::Float16));
        assert_eq!(DataType::from_i32(103), Some(DataType::BFloat16));
        assert_eq!(DataType::from_i32(104), Some(DataType::SparseFloat));
        assert_eq!(DataType::from_i32(105), Some(DataType::Int8));
        assert_eq!(DataType::from_i32(999), None);
    }

    #[test]
    fn test_data_type_to_i32() {
        assert_eq!(DataType::Binary.to_i32(), 100);
        assert_eq!(DataType::Float.to_i32(), 101);
    }

    #[test]
    fn test_data_type_is_dense_float() {
        assert!(DataType::Float.is_dense_float());
        assert!(DataType::Float16.is_dense_float());
        assert!(DataType::BFloat16.is_dense_float());
        assert!(!DataType::Binary.is_dense_float());
        assert!(!DataType::Int8.is_dense_float());
    }
}
