//! **LEGACY/SCAFFOLD MODULE** — simplified IVF container, not native-compatible.
//! Lacks: metric_type dispatch, standard k-means training, range search, serialization.
//! For parity with native knowhere, see ivf_sq8.rs / ivfpq.rs / ivf_flat.rs.
#![allow(deprecated)] // Legacy IVF scaffold remains exported for compatibility tests and wrappers.

use std::io::{Read, Write};
use std::path::Path;

use crate::api::{MetricType, Result};
use crate::bitset::BitsetView;
use crate::dataset::Dataset;
use crate::index::{
    AnnIterator, Index as IndexTrait, IndexError, SearchResult as IndexSearchResult,
};
use crate::quantization::KMeans;
use crate::simd::{dot_product_f32, l2_distance_sq};

/// IVF-Flat index.
///
/// Stores full-precision vectors in inverted lists (no quantization).
#[deprecated(note = "use IvfFlatIndex (ivf_flat.rs) or IvfSq8Index (ivf_sq8.rs) for native parity")]
pub struct IvfIndex {
    pub dim: usize,
    pub nlist: usize,
    pub nprobe: usize,
    pub centroids: Vec<f32>,    // [nlist * dim]
    pub lists: Vec<Vec<usize>>, // row-id lists per centroid
    pub vectors: Vec<f32>,      // [N * dim]
    pub metric_type: MetricType,
    pub trained: bool,
    pub ids: Vec<i64>, // external ids by row-id
}

impl IvfIndex {
    pub fn new(dim: usize, nlist: usize) -> Self {
        Self {
            dim,
            nlist: nlist.max(1),
            nprobe: 1,
            centroids: vec![0.0; nlist.max(1) * dim],
            lists: (0..nlist.max(1)).map(|_| Vec::new()).collect(),
            vectors: Vec::new(),
            metric_type: MetricType::L2,
            trained: false,
            ids: Vec::new(),
        }
    }

    fn with_metric(dim: usize, nlist: usize, nprobe: usize, metric_type: MetricType) -> Self {
        Self {
            dim,
            nlist: nlist.max(1),
            nprobe: nprobe.max(1).min(nlist.max(1)),
            centroids: vec![0.0; nlist.max(1) * dim],
            lists: (0..nlist.max(1)).map(|_| Vec::new()).collect(),
            vectors: Vec::new(),
            metric_type,
            trained: false,
            ids: Vec::new(),
        }
    }

    fn score(&self, a: &[f32], b: &[f32]) -> f32 {
        match self.metric_type {
            MetricType::L2 | MetricType::Hamming => l2_distance_sq(a, b),
            MetricType::Ip | MetricType::Cosine => -dot_product_f32(a, b),
        }
    }

    fn train_impl(&mut self, data: &[f32]) -> Result<()> {
        if self.dim == 0 || self.nlist == 0 {
            return Err(crate::api::KnowhereError::InvalidArg(
                "invalid ivf config".to_string(),
            ));
        }
        if data.is_empty() || !data.len().is_multiple_of(self.dim) {
            return Err(crate::api::KnowhereError::InvalidArg(
                "training data dimension mismatch".to_string(),
            ));
        }
        let n = data.len() / self.dim;
        if n < self.nlist {
            return Err(crate::api::KnowhereError::InvalidArg(format!(
                "training vectors {} < nlist {}",
                n, self.nlist
            )));
        }

        // Keep KMeans centroid training aligned with existing IVF implementations.
        let mut km = KMeans::new(self.nlist, self.dim);
        km.train(data);
        self.centroids = km.centroids().to_vec();
        self.lists = (0..self.nlist).map(|_| Vec::new()).collect();
        self.trained = true;
        Ok(())
    }

    /// Compatibility API kept for existing call sites/tests.
    pub fn train(&mut self, data: &[f32]) {
        let _ = self.train_impl(data);
    }

    fn add_impl(&mut self, data: &[f32], ext_ids: Option<&[i64]>) -> Result<usize> {
        if !self.trained {
            return Err(crate::api::KnowhereError::InvalidArg(
                "index not trained".to_string(),
            ));
        }
        if data.is_empty() {
            return Ok(0);
        }
        if !data.len().is_multiple_of(self.dim) {
            return Err(crate::api::KnowhereError::InvalidArg(
                "add data dimension mismatch".to_string(),
            ));
        }
        let n = data.len() / self.dim;
        if let Some(ids) = ext_ids {
            if ids.len() != n {
                return Err(crate::api::KnowhereError::InvalidArg(
                    "ids length mismatch".to_string(),
                ));
            }
        }

        let base_row = self.vectors.len() / self.dim;
        self.vectors.extend_from_slice(data);

        for i in 0..n {
            let row = base_row + i;
            let start = row * self.dim;
            let vector = &self.vectors[start..start + self.dim];
            let list_id = self.find_nearest_centroid(vector);
            self.lists[list_id].push(row);
            let id = ext_ids.map(|ids| ids[i]).unwrap_or(row as i64);
            self.ids.push(id);
        }
        Ok(n)
    }

    /// Compatibility API kept for existing call sites/tests.
    pub fn add(&mut self, data: &[f32]) -> usize {
        self.add_impl(data, None).unwrap_or(0)
    }

    pub fn add_with_ids(&mut self, data: &[f32], ids: Option<&[i64]>) -> Result<usize> {
        self.add_impl(data, ids)
    }

    fn find_nearest_centroid(&self, vector: &[f32]) -> usize {
        let mut best = 0usize;
        let mut best_score = f32::INFINITY;
        for c in 0..self.nlist {
            let centroid = &self.centroids[c * self.dim..(c + 1) * self.dim];
            let score = self.score(vector, centroid);
            if score < best_score {
                best_score = score;
                best = c;
            }
        }
        best
    }

    fn probe_clusters(&self, query: &[f32], nprobe: usize) -> Vec<usize> {
        let mut dists: Vec<(usize, f32)> = (0..self.nlist)
            .map(|c| {
                let centroid = &self.centroids[c * self.dim..(c + 1) * self.dim];
                (c, self.score(query, centroid))
            })
            .collect();
        dists.sort_by(|a, b| a.1.total_cmp(&b.1));
        dists
            .into_iter()
            .take(nprobe.min(self.nlist).max(1))
            .map(|(c, _)| c)
            .collect()
    }

    fn search_rows(
        &self,
        query: &[f32],
        top_k: usize,
        bitset: Option<&BitsetView>,
    ) -> Vec<(usize, f32)> {
        if !self.trained || top_k == 0 || self.vectors.is_empty() {
            return Vec::new();
        }
        let clusters = self.probe_clusters(query, self.nprobe);
        let mut candidates: Vec<(usize, f32)> = Vec::new();
        for c in clusters {
            for &row in &self.lists[c] {
                if let Some(bs) = bitset {
                    if row < bs.len() && bs.get(row) {
                        continue;
                    }
                }
                let vec_ref = &self.vectors[row * self.dim..(row + 1) * self.dim];
                let dist = self.score(query, vec_ref);
                candidates.push((row, dist));
            }
        }
        candidates.sort_by(|a, b| a.1.total_cmp(&b.1));
        candidates.truncate(top_k);
        candidates
    }

    /// Compatibility API kept for existing call sites/tests.
    pub fn search_raw(&self, query: &[f32], top_k: usize) -> Vec<(usize, f32)> {
        self.search_rows(query, top_k, None)
    }

    /// Compatibility API kept for existing call sites/tests.
    pub fn search(&self, query: &[f32], top_k: usize) -> Vec<(usize, f32)> {
        self.search_raw(query, top_k)
    }

    pub fn search_with_bitset(
        &self,
        query: &[f32],
        top_k: usize,
        bitset: &BitsetView,
    ) -> Vec<(usize, f32)> {
        self.search_rows(query, top_k, Some(bitset))
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let mut f = std::fs::File::create(path)?;
        f.write_all(b"IVFF")?;
        f.write_all(&1u32.to_le_bytes())?;
        f.write_all(&(self.dim as u32).to_le_bytes())?;
        f.write_all(&(self.nlist as u32).to_le_bytes())?;
        f.write_all(&(self.nprobe as u32).to_le_bytes())?;
        let metric_byte = match self.metric_type {
            MetricType::L2 => 0u8,
            MetricType::Ip => 1u8,
            MetricType::Cosine => 2u8,
            MetricType::Hamming => 3u8,
        };
        f.write_all(&[metric_byte])?;
        let n = self.vectors.len() / self.dim;
        f.write_all(&(n as u64).to_le_bytes())?;
        for &v in &self.centroids {
            f.write_all(&v.to_le_bytes())?;
        }
        for &id in &self.ids {
            f.write_all(&id.to_le_bytes())?;
        }
        for &v in &self.vectors {
            f.write_all(&v.to_le_bytes())?;
        }
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self> {
        let mut f = std::fs::File::open(path)?;
        let mut magic = [0u8; 4];
        f.read_exact(&mut magic)?;
        if &magic != b"IVFF" {
            return Err(crate::api::KnowhereError::Codec(
                "invalid IVFF magic".to_string(),
            ));
        }
        let mut u32b = [0u8; 4];
        f.read_exact(&mut u32b)?; // version
        let version = u32::from_le_bytes(u32b);
        if version != 1 {
            return Err(crate::api::KnowhereError::Codec(format!(
                "unsupported IVFF version {version}"
            )));
        }
        f.read_exact(&mut u32b)?;
        let dim = u32::from_le_bytes(u32b) as usize;
        f.read_exact(&mut u32b)?;
        let nlist = u32::from_le_bytes(u32b) as usize;
        f.read_exact(&mut u32b)?;
        let nprobe = u32::from_le_bytes(u32b) as usize;
        let mut metricb = [0u8; 1];
        f.read_exact(&mut metricb)?;
        let metric_type = match metricb[0] {
            1 => MetricType::Ip,
            2 => MetricType::Cosine,
            3 => MetricType::Hamming,
            _ => MetricType::L2,
        };
        let mut u64b = [0u8; 8];
        f.read_exact(&mut u64b)?;
        let n = u64::from_le_bytes(u64b) as usize;

        let mut index = Self::with_metric(dim, nlist, nprobe, metric_type);
        index.centroids = vec![0.0; nlist * dim];
        for v in &mut index.centroids {
            let mut b = [0u8; 4];
            f.read_exact(&mut b)?;
            *v = f32::from_le_bytes(b);
        }

        index.ids = vec![0i64; n];
        for id in &mut index.ids {
            let mut b = [0u8; 8];
            f.read_exact(&mut b)?;
            *id = i64::from_le_bytes(b);
        }

        index.vectors = vec![0.0; n * dim];
        for v in &mut index.vectors {
            let mut b = [0u8; 4];
            f.read_exact(&mut b)?;
            *v = f32::from_le_bytes(b);
        }

        index.lists = (0..index.nlist).map(|_| Vec::new()).collect();
        for row in 0..n {
            let vector = &index.vectors[row * dim..(row + 1) * dim];
            let c = index.find_nearest_centroid(vector);
            index.lists[c].push(row);
        }
        index.trained = true;
        Ok(index)
    }
}

impl IndexTrait for IvfIndex {
    fn index_type(&self) -> &str {
        "IVF-FLAT"
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn count(&self) -> usize {
        self.vectors.len() / self.dim
    }

    fn is_trained(&self) -> bool {
        self.trained
    }

    fn train(&mut self, dataset: &Dataset) -> std::result::Result<(), IndexError> {
        self.train_impl(dataset.vectors())
            .map_err(|e| IndexError::Unsupported(e.to_string()))
    }

    fn add(&mut self, dataset: &Dataset) -> std::result::Result<usize, IndexError> {
        self.add_with_ids(dataset.vectors(), dataset.ids())
            .map_err(|e| IndexError::Unsupported(e.to_string()))
    }

    fn search(
        &self,
        query: &Dataset,
        top_k: usize,
    ) -> std::result::Result<IndexSearchResult, IndexError> {
        let q = query.vectors();
        if !q.len().is_multiple_of(self.dim) {
            return Err(IndexError::DimMismatch);
        }
        let nq = q.len() / self.dim;
        let mut ids = vec![-1; nq * top_k];
        let mut dists = vec![f32::MAX; nq * top_k];
        for qi in 0..nq {
            let query_vec = &q[qi * self.dim..(qi + 1) * self.dim];
            let hits = self.search_rows(query_vec, top_k, None);
            for (rank, (row, dist)) in hits.into_iter().enumerate() {
                ids[qi * top_k + rank] = self.ids.get(row).copied().unwrap_or(row as i64);
                dists[qi * top_k + rank] = dist;
            }
        }
        Ok(IndexSearchResult::new(ids, dists, 0.0))
    }

    fn range_search(
        &self,
        query: &Dataset,
        radius: f32,
    ) -> std::result::Result<IndexSearchResult, IndexError> {
        let q = query.vectors();
        if !q.len().is_multiple_of(self.dim) {
            return Err(IndexError::DimMismatch);
        }
        let nq = q.len() / self.dim;
        let mut out_ids = Vec::new();
        let mut out_dists = Vec::new();
        for qi in 0..nq {
            let query_vec = &q[qi * self.dim..(qi + 1) * self.dim];
            let clusters = self.probe_clusters(query_vec, self.nprobe);
            for c in clusters {
                for &row in &self.lists[c] {
                    let vec_ref = &self.vectors[row * self.dim..(row + 1) * self.dim];
                    let dist = self.score(query_vec, vec_ref);
                    if dist <= radius {
                        out_ids.push(self.ids[row]);
                        out_dists.push(dist);
                    }
                }
            }
        }
        Ok(IndexSearchResult::new(out_ids, out_dists, 0.0))
    }

    fn save(&self, path: &str) -> std::result::Result<(), IndexError> {
        self.save(Path::new(path))
            .map_err(|e| IndexError::Unsupported(e.to_string()))
    }

    fn load(&mut self, path: &str) -> std::result::Result<(), IndexError> {
        let loaded =
            IvfIndex::load(Path::new(path)).map_err(|e| IndexError::Unsupported(e.to_string()))?;
        *self = loaded;
        Ok(())
    }

    fn has_raw_data(&self) -> bool {
        true
    }

    fn get_vector_by_ids(&self, ids: &[i64]) -> std::result::Result<Vec<f32>, IndexError> {
        let mut out = Vec::with_capacity(ids.len() * self.dim);
        for &id in ids {
            let Some(row) = self.ids.iter().position(|&x| x == id) else {
                return Err(IndexError::Unsupported(format!("id {id} not found")));
            };
            out.extend_from_slice(&self.vectors[row * self.dim..(row + 1) * self.dim]);
        }
        Ok(out)
    }

    fn create_ann_iterator(
        &self,
        _query: &Dataset,
        _bitset: Option<&BitsetView>,
    ) -> std::result::Result<Box<dyn AnnIterator>, IndexError> {
        Err(IndexError::Unsupported(
            "IVF-FLAT iterator is not implemented".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::MetricType;

    #[test]
    fn test_ivf_new() {
        let ivf = IvfIndex::new(128, 10);
        assert_eq!(ivf.dim, 128);
        assert_eq!(ivf.nlist, 10);
    }

    #[test]
    fn test_ivf_train() {
        let mut ivf = IvfIndex::new(4, 2);
        let data = vec![0.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 10.0];
        ivf.train(&data);
        assert!(ivf.trained);
        assert!(!ivf.centroids.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_ivf_add_search() {
        let mut ivf = IvfIndex::new(4, 2);
        let data = vec![0.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 10.0];
        ivf.train(&data);
        ivf.add(&data);
        let query = vec![1.0, 1.0, 1.0, 1.0];
        let results = ivf.search(&query, 2);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_ivf_metric_ip_search_raw() {
        let mut ivf = IvfIndex::with_metric(2, 1, 1, MetricType::Ip);
        let train = vec![1.0, 0.0, 0.0, 1.0];
        ivf.train_impl(&train).unwrap();
        ivf.add_with_ids(&train, Some(&[10, 11])).unwrap();
        let hits = ivf.search_raw(&[1.0, 0.0], 1);
        assert_eq!(hits.len(), 1);
    }
}
