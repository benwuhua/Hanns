use crate::api::{Result, SearchRequest};
use crate::index::Index;
use crate::kernel::{AnnRuntime, IndexFamily, RowFilter};

use super::HnswIndex;

pub struct HnswRuntime {
    inner: HnswIndex,
}

impl HnswRuntime {
    pub fn new(inner: HnswIndex) -> Self {
        Self { inner }
    }

    pub fn into_inner(self) -> HnswIndex {
        self.inner
    }
}

impl AnnRuntime for HnswRuntime {
    fn family(&self) -> IndexFamily {
        IndexFamily::Hnsw
    }

    fn dim(&self) -> usize {
        self.inner.dim()
    }

    fn len(&self) -> usize {
        self.inner.ntotal()
    }

    fn search_into(
        &self,
        query: &[f32],
        req: &SearchRequest,
        ids: &mut [i64],
        dists: &mut [f32],
    ) -> Result<usize> {
        self.inner.search_into(query, req, ids, dists)
    }

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
}
