use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

use crate::api::{IndexConfig, KnowhereError, MetricType, Result, SearchRequest, SearchResult};
use crate::bitset::BitsetView;
use crate::dataset::Dataset;
use crate::index::{
    AnnIterator, Index as IndexTrait, IndexError, SearchResult as IndexSearchResult,
};
use crate::quantization::exrabitq::{
    scan_and_rerank, EncodedVector, ExFactor, ExRaBitQConfig, ExRaBitQFastScanState,
    ExRaBitQLayout, ExRaBitQQuantizer, ExShortFactors,
};
use crate::quantization::KMeans;

const EXRABITQ_MAGIC: &[u8; 8] = b"IVFXRBTQ";
const EXRABITQ_VERSION: u32 = 1;

#[derive(Clone, Debug)]
pub struct IvfExRaBitqConfig {
    pub dim: usize,
    pub nlist: usize,
    pub nprobe: usize,
    pub metric_type: MetricType,
    pub bits_per_dim: usize,
    pub use_high_accuracy_scan: bool,
    pub rotation_seed: u64,
    pub rerank_k: usize,
}

impl IvfExRaBitqConfig {
    pub fn new(dim: usize, nlist: usize, bits_per_dim: usize) -> Self {
        Self {
            dim,
            nlist,
            nprobe: 1,
            metric_type: MetricType::L2,
            bits_per_dim,
            use_high_accuracy_scan: false,
            rotation_seed: 42,
            rerank_k: 64,
        }
    }

    pub fn with_nprobe(mut self, nprobe: usize) -> Self {
        self.nprobe = nprobe;
        self
    }

    pub fn with_metric(mut self, metric_type: MetricType) -> Self {
        self.metric_type = metric_type;
        self
    }

    pub fn with_rotation_seed(mut self, rotation_seed: u64) -> Self {
        self.rotation_seed = rotation_seed;
        self
    }

    pub fn with_high_accuracy_scan(mut self, use_high_accuracy_scan: bool) -> Self {
        self.use_high_accuracy_scan = use_high_accuracy_scan;
        self
    }

    pub fn with_rerank_k(mut self, rerank_k: usize) -> Self {
        self.rerank_k = rerank_k;
        self
    }

    pub fn from_index_config(config: &IndexConfig) -> Result<Self> {
        if config.dim == 0 {
            return Err(KnowhereError::InvalidArg(
                "dimension must be > 0".to_string(),
            ));
        }
        if config.index_type != crate::api::IndexType::IvfExRaBitq {
            return Err(KnowhereError::InvalidArg(format!(
                "expected IvfExRaBitq config, got {:?}",
                config.index_type
            )));
        }

        let nlist = config.params.nlist.unwrap_or(100);
        let nprobe = config.params.nprobe.unwrap_or(8);
        let bits_per_dim = config.params.exrabitq_bits_per_dim.unwrap_or(4);
        let rotation_seed = config.params.exrabitq_rotation_seed.unwrap_or(42);
        let rerank_k = config.params.exrabitq_rerank_k.unwrap_or(64);

        Ok(Self {
            dim: config.dim,
            nlist,
            nprobe,
            metric_type: config.metric_type,
            bits_per_dim,
            use_high_accuracy_scan: config
                .params
                .exrabitq_use_high_accuracy_scan
                .unwrap_or(false),
            rotation_seed,
            rerank_k,
        })
    }
}

#[derive(Clone)]
struct ClusterState {
    ids: Vec<i64>,
    encoded: Vec<EncodedVector>,
    layout: Option<ExRaBitQLayout>,
}

impl ClusterState {
    fn new() -> Self {
        Self {
            ids: Vec::new(),
            encoded: Vec::new(),
            layout: None,
        }
    }
}

pub struct IvfExRaBitqIndex {
    config: IvfExRaBitqConfig,
    centroids: Vec<f32>,
    rotated_centroids: Vec<f32>,
    clusters: Vec<ClusterState>,
    quantizer: ExRaBitQQuantizer,
    trained: bool,
    ntotal: usize,
}

#[derive(Serialize, Deserialize)]
struct StoredEncodedVector {
    short_code: Vec<u8>,
    long_code: Vec<u8>,
    xipnorm: f32,
    short_ip: f32,
    short_sum_xb: f32,
    short_err: f32,
    x_norm: f32,
    x2: f32,
}

#[derive(Serialize, Deserialize)]
struct StoredCluster {
    ids: Vec<i64>,
    encoded: Vec<StoredEncodedVector>,
}

#[derive(Serialize, Deserialize)]
struct StoredSnapshot {
    config: StoredConfig,
    centroids: Vec<f32>,
    clusters: Vec<StoredCluster>,
    trained: bool,
    ntotal: usize,
}

#[derive(Serialize, Deserialize)]
struct StoredConfig {
    dim: usize,
    nlist: usize,
    nprobe: usize,
    metric_type: MetricType,
    bits_per_dim: usize,
    use_high_accuracy_scan: bool,
    rotation_seed: u64,
    rerank_k: usize,
}

impl IvfExRaBitqIndex {
    #[inline]
    fn compare_centroid_score(a: &(usize, f32), b: &(usize, f32)) -> std::cmp::Ordering {
        a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0))
    }

    #[inline]
    fn compare_candidate_score(a: &(i64, f32), b: &(i64, f32)) -> std::cmp::Ordering {
        a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0))
    }

    pub fn from_index_config(config: &IndexConfig) -> Result<Self> {
        Ok(Self::new(IvfExRaBitqConfig::from_index_config(config)?))
    }

    pub fn new(config: IvfExRaBitqConfig) -> Self {
        let quantizer = ExRaBitQQuantizer::new(
            ExRaBitQConfig::new(config.dim, config.bits_per_dim)
                .expect("valid config")
                .with_rotation_seed(config.rotation_seed),
        )
        .expect("quantizer");

        Self {
            centroids: Vec::new(),
            rotated_centroids: Vec::new(),
            clusters: (0..config.nlist).map(|_| ClusterState::new()).collect(),
            quantizer,
            trained: false,
            ntotal: 0,
            config,
        }
    }

    pub fn config(&self) -> &IvfExRaBitqConfig {
        &self.config
    }

    pub fn ntotal(&self) -> usize {
        self.ntotal
    }

    pub fn count(&self) -> usize {
        self.ntotal
    }

    pub fn has_raw_data(&self) -> bool {
        false
    }

    pub fn size(&self) -> usize {
        let centroids_size = self.centroids.len() * std::mem::size_of::<f32>();
        let ids_size: usize = self
            .clusters
            .iter()
            .map(|cluster| cluster.ids.len() * std::mem::size_of::<i64>())
            .sum();
        let codes_and_factors_size: usize = self
            .clusters
            .iter()
            .flat_map(|cluster| cluster.encoded.iter())
            .map(|entry| {
                entry.short_code.len()
                    + entry.long_code.len()
                    + std::mem::size_of::<ExFactor>()
                    + std::mem::size_of::<ExShortFactors>()
                    + 2 * std::mem::size_of::<f32>()
            })
            .sum();

        centroids_size + ids_size + codes_and_factors_size
    }

    pub fn set_nprobe(&mut self, nprobe: usize) {
        self.config.nprobe = nprobe.max(1).min(self.config.nlist);
    }

    pub fn train(&mut self, data: &[f32]) -> Result<()> {
        if self.config.metric_type != MetricType::L2 {
            return Err(KnowhereError::InvalidArg(
                "IvfExRaBitqIndex currently supports MetricType::L2 only".to_string(),
            ));
        }

        let n = data.len() / self.config.dim;
        if self.config.dim == 0 || n * self.config.dim != data.len() {
            return Err(KnowhereError::InvalidArg(
                "training data dimension mismatch".to_string(),
            ));
        }
        if n < self.config.nlist {
            return Err(KnowhereError::InvalidArg(format!(
                "training data too small: {n} < {}",
                self.config.nlist
            )));
        }

        let mut km = KMeans::new(self.config.nlist, self.config.dim);
        km.train(data);
        self.centroids = km.centroids().to_vec();
        self.rotated_centroids = self.compute_rotated_centroids(&self.centroids);
        self.clusters = (0..self.config.nlist)
            .map(|_| ClusterState::new())
            .collect();
        self.trained = true;
        Ok(())
    }

    pub fn add(&mut self, data: &[f32], ids: Option<&[i64]>) -> Result<usize> {
        if !self.trained {
            return Err(KnowhereError::InvalidArg(
                "index not trained, call train() first".to_string(),
            ));
        }

        let n = data.len() / self.config.dim;
        if n * self.config.dim != data.len() {
            return Err(KnowhereError::InvalidArg(
                "add data dimension mismatch".to_string(),
            ));
        }
        if let Some(ids) = ids {
            if ids.len() != n {
                return Err(KnowhereError::InvalidArg(
                    "ids length does not match vector count".to_string(),
                ));
            }
        }

        let padded_dim = self.quantizer.config().padded_dim();
        let encoded_vectors: Vec<(usize, i64, EncodedVector)> = data
            .par_chunks_exact(self.config.dim)
            .enumerate()
            .map_init(
                || (vec![0.0f32; padded_dim], vec![0.0f32; padded_dim]),
                |(padded, rotated), (i, vector)| {
                    let cluster = self.find_best_centroid(vector);
                    let id = ids
                        .map(|values| values[i])
                        .unwrap_or((self.ntotal + i) as i64);
                    let rotated_centroid = &self.rotated_centroids
                        [cluster * padded_dim..(cluster + 1) * padded_dim];
                    let encoded = self.quantizer.encode_with_rotated_centroid_into(
                        vector,
                        rotated_centroid,
                        padded,
                        rotated,
                    );
                    (cluster, id, encoded)
                },
            )
            .collect();

        let mut pending: Vec<Vec<(i64, EncodedVector)>> =
            (0..self.config.nlist).map(|_| Vec::new()).collect();
        for (cluster, id, encoded) in encoded_vectors {
            pending[cluster].push((id, encoded));
        }

        let mut touched_clusters = Vec::new();

        for (cluster_idx, additions) in pending.into_iter().enumerate() {
            if additions.is_empty() {
                continue;
            }
            touched_clusters.push(cluster_idx);
            let state = &mut self.clusters[cluster_idx];
            state.ids.reserve(additions.len());
            state.encoded.reserve(additions.len());
            for (id, encoded) in additions {
                state.ids.push(id);
                state.encoded.push(encoded);
            }
        }

        let rebuilt_layouts: Vec<(usize, ExRaBitQLayout)> = touched_clusters
            .par_iter()
            .map(|&cluster_idx| {
                let state = &self.clusters[cluster_idx];
                (
                    cluster_idx,
                    ExRaBitQLayout::build(self.quantizer.config(), &state.encoded, &state.ids),
                )
            })
            .collect();
        for (cluster_idx, layout) in rebuilt_layouts {
            self.clusters[cluster_idx].layout = Some(layout);
        }

        self.ntotal += n;
        Ok(n)
    }

    pub fn search(&self, query: &[f32], req: &SearchRequest) -> Result<SearchResult> {
        if !self.trained {
            return Err(KnowhereError::InvalidArg("index not trained".to_string()));
        }

        let nq = query.len() / self.config.dim;
        if nq == 0 || nq * self.config.dim != query.len() {
            return Err(KnowhereError::InvalidArg(
                "query dimension mismatch".to_string(),
            ));
        }

        let top_k = req.top_k.max(1);
        let nprobe = req.nprobe.max(1).min(self.config.nlist);
        let shortlist = self.config.rerank_k.max(top_k);

        let mut ids = Vec::with_capacity(nq * top_k);
        let mut distances = Vec::with_capacity(nq * top_k);

        for q in query.chunks_exact(self.config.dim) {
            let mut batch = self.search_single(q, top_k, shortlist, nprobe);
            batch.truncate(top_k);
            while batch.len() < top_k {
                batch.push((-1, f32::INFINITY));
            }
            for (id, dist) in batch {
                ids.push(id);
                distances.push(dist);
            }
        }

        Ok(SearchResult::new(ids, distances, 0.0))
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let snapshot = StoredSnapshot {
            config: StoredConfig {
                dim: self.config.dim,
                nlist: self.config.nlist,
                nprobe: self.config.nprobe,
                metric_type: self.config.metric_type,
                bits_per_dim: self.config.bits_per_dim,
                use_high_accuracy_scan: self.config.use_high_accuracy_scan,
                rotation_seed: self.config.rotation_seed,
                rerank_k: self.config.rerank_k,
            },
            centroids: self.centroids.clone(),
            clusters: self
                .clusters
                .iter()
                .map(|cluster| StoredCluster {
                    ids: cluster.ids.clone(),
                    encoded: cluster
                        .encoded
                        .iter()
                        .map(|entry| StoredEncodedVector {
                            short_code: entry.short_code.clone(),
                            long_code: entry.long_code.clone(),
                            xipnorm: entry.factor.xipnorm,
                            short_ip: entry.short_factors.ip,
                            short_sum_xb: entry.short_factors.sum_xb,
                            short_err: entry.short_factors.err,
                            x_norm: entry.x_norm,
                            x2: entry.x2,
                        })
                        .collect(),
                })
                .collect(),
            trained: self.trained,
            ntotal: self.ntotal,
        };

        let payload = bincode::serialize(&snapshot)
            .map_err(|e| KnowhereError::Codec(format!("serialize exrabitq snapshot: {e}")))?;
        let mut bytes = Vec::with_capacity(EXRABITQ_MAGIC.len() + 8 + payload.len());
        bytes.extend_from_slice(EXRABITQ_MAGIC);
        bytes.extend_from_slice(&EXRABITQ_VERSION.to_le_bytes());
        bytes.extend_from_slice(&(self.config.dim as u32).to_le_bytes());
        bytes.extend_from_slice(&payload);
        let mut file = File::create(path)?;
        file.write_all(&bytes)?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self> {
        let mut bytes = Vec::new();
        File::open(path)?.read_to_end(&mut bytes)?;
        let payload = if bytes.starts_with(EXRABITQ_MAGIC) {
            bytes.get(16..).ok_or_else(|| {
                KnowhereError::Codec("exrabitq snapshot header too short".to_string())
            })?
        } else {
            bytes.as_slice()
        };
        let snapshot: StoredSnapshot = bincode::deserialize(payload)
            .map_err(|e| KnowhereError::Codec(format!("deserialize exrabitq snapshot: {e}")))?;

        let config = IvfExRaBitqConfig {
            dim: snapshot.config.dim,
            nlist: snapshot.config.nlist,
            nprobe: snapshot.config.nprobe,
            metric_type: snapshot.config.metric_type,
            bits_per_dim: snapshot.config.bits_per_dim,
            use_high_accuracy_scan: snapshot.config.use_high_accuracy_scan,
            rotation_seed: snapshot.config.rotation_seed,
            rerank_k: snapshot.config.rerank_k,
        };

        let quantizer = ExRaBitQQuantizer::new(
            ExRaBitQConfig::new(config.dim, config.bits_per_dim)
                .expect("valid config")
                .with_rotation_seed(config.rotation_seed),
        )
        .expect("quantizer");

        let clusters: Vec<ClusterState> = snapshot
            .clusters
            .into_iter()
            .map(|cluster| {
                let encoded: Vec<EncodedVector> = cluster
                    .encoded
                    .into_iter()
                    .map(|entry| EncodedVector {
                        short_code: entry.short_code,
                        long_code: entry.long_code,
                        factor: ExFactor {
                            xipnorm: entry.xipnorm,
                        },
                        short_factors: ExShortFactors {
                            ip: entry.short_ip,
                            sum_xb: entry.short_sum_xb,
                            err: entry.short_err,
                        },
                        x_norm: entry.x_norm,
                        x2: entry.x2,
                    })
                    .collect();
                let layout = if encoded.is_empty() {
                    None
                } else {
                    Some(ExRaBitQLayout::build(
                        quantizer.config(),
                        &encoded,
                        &cluster.ids,
                    ))
                };
                ClusterState {
                    ids: cluster.ids,
                    encoded,
                    layout,
                }
            })
            .collect();

        Ok(Self {
            config,
            centroids: snapshot.centroids,
            rotated_centroids: Vec::new(),
            clusters,
            quantizer,
            trained: snapshot.trained,
            ntotal: snapshot.ntotal,
        }
        .with_rebuilt_rotated_centroids())
    }

    fn search_single(
        &self,
        query: &[f32],
        top_k: usize,
        shortlist: usize,
        nprobe: usize,
    ) -> Vec<(i64, f32)> {
        let probes = self.find_top_centroids(query, nprobe);
        let mut candidates = Vec::new();
        let q_padded = self.quantizer.pad_vector(query);
        let q_rotated = self.quantizer.rotate_padded(&q_padded);
        let padded_dim = self.quantizer.config().padded_dim();
        let mut residual = vec![0.0f32; padded_dim];
        let mut state: Option<ExRaBitQFastScanState> = None;

        for cluster_idx in probes {
            let Some(layout) = self.clusters[cluster_idx].layout.as_ref() else {
                continue;
            };
            let rotated_centroid = &self.rotated_centroids
                [cluster_idx * padded_dim..(cluster_idx + 1) * padded_dim];
            let mut y2 = 0.0f32;
            for ((dst, &q), &c) in residual
                .iter_mut()
                .zip(q_rotated.iter())
                .zip(rotated_centroid.iter())
            {
                let value = q - c;
                *dst = value;
                y2 += value * value;
            }

            let scan_state = if let Some(state) = state.as_mut() {
                state.reset(&residual, y2);
                state
            } else {
                state.insert(if self.config.use_high_accuracy_scan {
                    ExRaBitQFastScanState::new_high_accuracy(&residual, y2)
                } else {
                    ExRaBitQFastScanState::new(&residual, y2)
                })
            };
            let mut reranked = scan_and_rerank(
                &self.quantizer,
                layout,
                scan_state,
                shortlist.min(layout.len()),
                shortlist,
            );
            candidates.append(&mut reranked);
        }

        let candidate_limit = top_k.max(shortlist.min(candidates.len()));
        Self::truncate_top_candidates(&mut candidates, candidate_limit);
        candidates
    }

    fn find_best_centroid(&self, vector: &[f32]) -> usize {
        self.find_top_centroids(vector, 1)[0]
    }

    fn find_top_centroids(&self, vector: &[f32], count: usize) -> Vec<usize> {
        let mut scored = Vec::with_capacity(self.config.nlist);
        for cluster_idx in 0..self.config.nlist {
            let centroid =
                &self.centroids[cluster_idx * self.config.dim..(cluster_idx + 1) * self.config.dim];
            let dist = l2_distance(vector, centroid);
            scored.push((cluster_idx, dist));
        }
        Self::select_top_centroids(scored, count)
            .into_iter()
            .map(|(idx, _)| idx)
            .collect()
    }

    fn select_top_centroids(mut scored: Vec<(usize, f32)>, count: usize) -> Vec<(usize, f32)> {
        if scored.is_empty() || count == 0 {
            return Vec::new();
        }
        if count >= scored.len() {
            scored.sort_by(Self::compare_centroid_score);
            return scored;
        }

        let nth = count - 1;
        scored.select_nth_unstable_by(nth, Self::compare_centroid_score);
        scored.truncate(count);
        scored.sort_by(Self::compare_centroid_score);
        scored
    }

    fn truncate_top_candidates(candidates: &mut Vec<(i64, f32)>, count: usize) {
        if candidates.is_empty() || count == 0 {
            candidates.clear();
            return;
        }
        if count >= candidates.len() {
            candidates.sort_by(Self::compare_candidate_score);
            return;
        }

        let nth = count - 1;
        candidates.select_nth_unstable_by(nth, Self::compare_candidate_score);
        candidates.truncate(count);
        candidates.sort_by(Self::compare_candidate_score);
    }

    fn compute_rotated_centroids(&self, centroids: &[f32]) -> Vec<f32> {
        let padded_dim = self.quantizer.config().padded_dim();
        let mut rotated = vec![0.0f32; self.config.nlist * padded_dim];
        rotated
            .par_chunks_exact_mut(padded_dim)
            .zip(centroids.par_chunks_exact(self.config.dim))
            .for_each_init(
                || vec![0.0f32; padded_dim],
                |padded, (rotated_centroid, centroid)| {
                    self.quantizer.pad_vector_into(centroid, padded);
                    self.quantizer.rotate_padded_into(padded, rotated_centroid);
                },
            );
        rotated
    }

    fn with_rebuilt_rotated_centroids(mut self) -> Self {
        if self.centroids.is_empty() {
            self.rotated_centroids.clear();
        } else {
            self.rotated_centroids = self.compute_rotated_centroids(&self.centroids);
        }
        self
    }
}

fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum::<f32>()
}

impl IndexTrait for IvfExRaBitqIndex {
    fn index_type(&self) -> &str {
        "IVF-ExRaBitQ"
    }

    fn dim(&self) -> usize {
        self.config.dim
    }

    fn count(&self) -> usize {
        self.ntotal
    }

    fn is_trained(&self) -> bool {
        self.trained
    }

    fn train(&mut self, dataset: &Dataset) -> std::result::Result<(), IndexError> {
        self.train(dataset.vectors())
            .map_err(|e| IndexError::Unsupported(e.to_string()))
    }

    fn add(&mut self, dataset: &Dataset) -> std::result::Result<usize, IndexError> {
        self.add(dataset.vectors(), dataset.ids())
            .map_err(|e| IndexError::Unsupported(e.to_string()))
    }

    fn search(
        &self,
        query: &Dataset,
        top_k: usize,
    ) -> std::result::Result<IndexSearchResult, IndexError> {
        let req = SearchRequest {
            top_k,
            nprobe: self.config.nprobe,
            filter: None,
            params: None,
            radius: None,
        };
        let result = self
            .search(query.vectors(), &req)
            .map_err(|e| IndexError::Unsupported(e.to_string()))?;
        Ok(IndexSearchResult::new(
            result.ids,
            result.distances,
            result.elapsed_ms,
        ))
    }

    fn search_with_bitset(
        &self,
        query: &Dataset,
        top_k: usize,
        bitset: &BitsetView,
    ) -> std::result::Result<IndexSearchResult, IndexError> {
        let req = SearchRequest {
            top_k,
            nprobe: self.config.nprobe,
            filter: None,
            params: None,
            radius: None,
        };
        let result = self
            .search(query.vectors(), &req)
            .map_err(|e| IndexError::Unsupported(e.to_string()))?;

        let mut ids = Vec::new();
        let mut distances = Vec::new();
        for (id, distance) in result.ids.into_iter().zip(result.distances.into_iter()) {
            let idx = id as usize;
            if idx >= bitset.len() || !bitset.get(idx) {
                ids.push(id);
                distances.push(distance);
            }
        }
        Ok(IndexSearchResult::new(ids, distances, result.elapsed_ms))
    }

    fn save(&self, path: &str) -> std::result::Result<(), IndexError> {
        self.save(Path::new(path))
            .map_err(|e| IndexError::Unsupported(e.to_string()))
    }

    fn load(&mut self, path: &str) -> std::result::Result<(), IndexError> {
        *self = Self::load(Path::new(path)).map_err(|e| IndexError::Unsupported(e.to_string()))?;
        Ok(())
    }

    fn has_raw_data(&self) -> bool {
        false
    }

    fn get_vector_by_ids(&self, _ids: &[i64]) -> std::result::Result<Vec<f32>, IndexError> {
        Err(IndexError::Unsupported(
            "get_vector_by_ids not supported for ExRaBitQ (lossy compression)".into(),
        ))
    }

    fn create_ann_iterator(
        &self,
        query: &Dataset,
        _bitset: Option<&BitsetView>,
    ) -> std::result::Result<Box<dyn AnnIterator>, IndexError> {
        let top_k = self.ntotal.max(1000);
        let req = SearchRequest {
            top_k,
            nprobe: self.config.nprobe,
            filter: None,
            params: None,
            radius: None,
        };
        let result = self
            .search(query.vectors(), &req)
            .map_err(|e| IndexError::Unsupported(e.to_string()))?;
        let pairs = result.ids.into_iter().zip(result.distances).collect();
        Ok(Box::new(IvfExRaBitqAnnIterator::new(pairs)))
    }
}

pub struct IvfExRaBitqAnnIterator {
    results: Vec<(i64, f32)>,
    pos: usize,
}

impl IvfExRaBitqAnnIterator {
    pub fn new(results: Vec<(i64, f32)>) -> Self {
        Self { results, pos: 0 }
    }
}

impl AnnIterator for IvfExRaBitqAnnIterator {
    fn next(&mut self) -> Option<(i64, f32)> {
        if self.pos >= self.results.len() {
            return None;
        }
        let item = self.results[self.pos];
        self.pos += 1;
        Some(item)
    }

    fn buffer_size(&self) -> usize {
        self.results.len().saturating_sub(self.pos)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_select_top_centroids_matches_full_sort_order() {
        let scored = vec![
            (0usize, 9.0f32),
            (1, 1.0),
            (2, 3.0),
            (3, 1.0),
            (4, 0.5),
        ];

        let top3 = IvfExRaBitqIndex::select_top_centroids(scored.clone(), 3);
        assert_eq!(top3, vec![(4, 0.5), (1, 1.0), (3, 1.0)]);

        let all = IvfExRaBitqIndex::select_top_centroids(scored, 99);
        assert_eq!(all, vec![(4, 0.5), (1, 1.0), (3, 1.0), (2, 3.0), (0, 9.0)]);
    }
}
