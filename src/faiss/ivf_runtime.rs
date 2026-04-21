use crate::api::{Result, SearchRequest};
use crate::index::Index as IndexTrait;
use crate::kernel::{AnnRuntime, IndexFamily};

use super::{IvfFlatIndex, IvfPqIndex, IvfSq8Index, IvfUsqIndex};

pub enum IvfRuntime {
    Flat(IvfFlatIndex),
    Sq8(IvfSq8Index),
    Pq(IvfPqIndex),
    Usq(IvfUsqIndex),
}

impl IvfRuntime {
    pub fn flat(index: IvfFlatIndex) -> Self {
        Self::Flat(index)
    }

    pub fn sq8(index: IvfSq8Index) -> Self {
        Self::Sq8(index)
    }

    pub fn pq(index: IvfPqIndex) -> Self {
        Self::Pq(index)
    }

    pub fn usq(index: IvfUsqIndex) -> Self {
        Self::Usq(index)
    }

    fn search(&self, query: &[f32], req: &SearchRequest) -> Result<crate::api::SearchResult> {
        match self {
            Self::Flat(index) => index.search(query, req),
            Self::Sq8(index) => index.search(query, req),
            Self::Pq(index) => index.search(query, req),
            Self::Usq(index) => index.search(query, req),
        }
    }
}

impl AnnRuntime for IvfRuntime {
    fn family(&self) -> IndexFamily {
        IndexFamily::Ivf
    }

    fn dim(&self) -> usize {
        match self {
            Self::Flat(index) => index.dim(),
            Self::Sq8(index) => IndexTrait::dim(index),
            Self::Pq(index) => index.dim(),
            Self::Usq(index) => index.config().dim,
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::Flat(index) => index.ntotal(),
            Self::Sq8(index) => index.ntotal(),
            Self::Pq(index) => index.ntotal(),
            Self::Usq(index) => index.ntotal(),
        }
    }

    fn search_into(
        &self,
        query: &[f32],
        req: &SearchRequest,
        ids: &mut [i64],
        dists: &mut [f32],
    ) -> Result<usize> {
        let result = self.search(query, req)?;
        let count = result.ids.len().min(ids.len()).min(dists.len());
        ids[..count].copy_from_slice(&result.ids[..count]);
        dists[..count].copy_from_slice(&result.distances[..count]);
        Ok(count)
    }
}
