//! Sparse WAND Index - 稀疏 WAND (Weak AND) 索引
//!
//! 基于 SparseInvertedIndex 的薄封装，使用 WAND 算法作为默认搜索策略
//! WAND (Document-At-A-Time Weak AND) 是一种高效的稀疏向量搜索算法

use crate::dataset::Dataset;
use crate::faiss::sparse_inverted::{
    InvertedIndexAlgo, SparseInvertedIndex, SparseInvertedSearcher, SparseMetricType, SparseVector,
};
use crate::index::{Index, IndexError, SearchResult};

/// 稀疏 WAND 索引
pub struct SparseWandIndex {
    inner: SparseInvertedIndex,
}

impl SparseWandIndex {
    /// 创建新的 WAND 索引
    pub fn new(metric_type: SparseMetricType) -> Self {
        Self {
            inner: SparseInvertedIndex::new(metric_type),
        }
    }

    /// 添加单个向量
    pub fn add(&mut self, vector: &SparseVector, doc_id: i64) -> Result<(), String> {
        self.inner.add(vector, doc_id)
    }

    /// 使用 WAND 算法搜索
    pub fn search(
        &self,
        query: &SparseVector,
        k: usize,
        bitset: Option<&[bool]>,
    ) -> Vec<(i64, f32)> {
        let searcher = SparseInvertedSearcher::new(&self.inner, InvertedIndexAlgo::DaatWand);
        searcher.search(query, k, bitset)
    }

    /// 按 ID 获取向量
    pub fn get_vector_by_id(&self, doc_id: i64) -> Option<SparseVector> {
        self.inner.get_vector_by_id(doc_id)
    }

    /// 获取行数
    pub fn n_rows(&self) -> usize {
        self.inner.n_rows()
    }

    /// 获取内存大小 (字节)
    pub fn size(&self) -> usize {
        self.inner.size()
    }
}

impl Index for SparseWandIndex {
    fn index_type(&self) -> &str {
        "SparseWand"
    }

    fn dim(&self) -> usize {
        self.inner.n_cols()
    }

    fn count(&self) -> usize {
        self.n_rows()
    }

    fn is_trained(&self) -> bool {
        true
    }

    fn train(&mut self, _dataset: &Dataset) -> Result<(), IndexError> {
        Ok(())
    }

    fn add(&mut self, dataset: &Dataset) -> Result<usize, IndexError> {
        let base = self.count() as i64;
        for row in 0..dataset.num_vectors() {
            let dense = dataset.get_vector(row).ok_or(IndexError::DimMismatch)?;
            let sparse = SparseVector::from_dense(dense, 0.0);
            let doc_id = dataset
                .ids()
                .and_then(|ids| ids.get(row).copied())
                .unwrap_or(base + row as i64);
            SparseWandIndex::add(self, &sparse, doc_id).map_err(IndexError::Unsupported)?;
        }
        Ok(dataset.num_vectors())
    }

    fn search(&self, query: &Dataset, top_k: usize) -> Result<SearchResult, IndexError> {
        if query.num_vectors() == 0 {
            return Ok(SearchResult::new(Vec::new(), Vec::new(), 0.0));
        }
        let dense = query.get_vector(0).ok_or(IndexError::DimMismatch)?;
        let sparse = SparseVector::from_dense(dense, 0.0);
        let result = SparseWandIndex::search(self, &sparse, top_k, None);
        let (ids, distances): (Vec<i64>, Vec<f32>) = result.into_iter().unzip();
        Ok(SearchResult::new(ids, distances, 0.0))
    }

    fn search_with_bitset(
        &self,
        query: &Dataset,
        top_k: usize,
        bitset: &crate::bitset::BitsetView,
    ) -> Result<SearchResult, IndexError> {
        if query.num_vectors() == 0 {
            return Ok(SearchResult::new(Vec::new(), Vec::new(), 0.0));
        }
        let dense = query.get_vector(0).ok_or(IndexError::DimMismatch)?;
        let sparse = SparseVector::from_dense(dense, 0.0);
        let bools: Vec<bool> = (0..bitset.len()).map(|i| bitset.get(i)).collect();
        let result = SparseWandIndex::search(self, &sparse, top_k, Some(&bools));
        let (ids, distances): (Vec<i64>, Vec<f32>) = result.into_iter().unzip();
        Ok(SearchResult::new(ids, distances, 0.0))
    }

    fn get_vector_by_ids(&self, ids: &[i64]) -> Result<Vec<f32>, IndexError> {
        let dim = self.dim();
        let mut out = Vec::with_capacity(ids.len() * dim);
        for &id in ids {
            let Some(sparse) = self.get_vector_by_id(id) else {
                continue;
            };
            let mut dense = vec![0.0f32; dim];
            for elem in sparse.elements {
                if (elem.dim as usize) < dim {
                    dense[elem.dim as usize] = elem.val;
                }
            }
            out.extend_from_slice(&dense);
        }
        if out.is_empty() {
            return Err(IndexError::Empty);
        }
        Ok(out)
    }

    fn has_raw_data(&self) -> bool {
        true
    }

    fn save(&self, _path: &str) -> Result<(), IndexError> {
        Err(IndexError::Unsupported("save not implemented".into()))
    }

    fn load(&mut self, _path: &str) -> Result<(), IndexError> {
        Err(IndexError::Unsupported("load not implemented".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_wand_new() {
        let index = SparseWandIndex::new(SparseMetricType::Ip);
        assert_eq!(index.n_rows(), 0);
    }

    #[test]
    fn test_sparse_wand_add_search() {
        let mut index = SparseWandIndex::new(SparseMetricType::Ip);

        let v1 = SparseVector::from_pairs(&[(0, 1.0), (1, 1.0)]);
        let v2 = SparseVector::from_pairs(&[(1, 1.0), (2, 1.0)]);

        index.add(&v1, 0).unwrap();
        index.add(&v2, 1).unwrap();

        let query = SparseVector::from_pairs(&[(0, 1.0)]);
        let results = index.search(&query, 2, None);

        assert!(!results.is_empty());
    }

    #[test]
    fn test_sparse_wand_index_trait_path() {
        let mut index = SparseWandIndex::new(SparseMetricType::Ip);

        let base = Dataset::from_vectors(
            vec![
                1.0, 0.0, 1.0, 0.0, // id 0
                0.0, 1.0, 1.0, 0.0, // id 1
            ],
            4,
        );
        Index::add(&mut index, &base).unwrap();

        let query = Dataset::from_vectors(vec![1.0, 0.0, 0.0, 0.0], 4);
        let result = Index::search(&index, &query, 1).unwrap();
        assert_eq!(result.ids.len(), 1);
    }
}
