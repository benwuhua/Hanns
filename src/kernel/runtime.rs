use crate::api::{Result, SearchRequest};

use super::filter::{NoFilter, RowFilter};

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IndexFamily {
    Flat,
    Hnsw,
    Ivf,
    Quantized,
    DiskAnn,
    Sparse,
}

pub trait AnnRuntime: Send + Sync {
    fn family(&self) -> IndexFamily;
    fn dim(&self) -> usize;
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn search_into(
        &self,
        query: &[f32],
        req: &SearchRequest,
        ids: &mut [i64],
        dists: &mut [f32],
    ) -> Result<usize>;

    fn search_with_filter_into(
        &self,
        query: &[f32],
        req: &SearchRequest,
        _filter: &dyn RowFilter,
        ids: &mut [i64],
        dists: &mut [f32],
    ) -> Result<usize> {
        self.search_into(query, req, ids, dists)
    }

    fn search_without_filter_into(
        &self,
        query: &[f32],
        req: &SearchRequest,
        ids: &mut [i64],
        dists: &mut [f32],
    ) -> Result<usize> {
        self.search_with_filter_into(query, req, &NoFilter, ids, dists)
    }
}
