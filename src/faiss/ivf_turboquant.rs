use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

use crate::api::{KnowhereError, MetricType, Predicate, Result, SearchRequest, SearchResult};
use crate::quantization::{KMeans, TurboQuantConfig, TurboQuantMse};

#[derive(Clone, Debug)]
pub struct IvfTurboQuantConfig {
    pub dim: usize,
    pub nlist: usize,
    pub nprobe: usize,
    pub metric_type: MetricType,
    pub bits_per_dim: u8,
    pub rotation_seed: u64,
    pub hadamard: bool,
}

impl IvfTurboQuantConfig {
    pub fn new(dim: usize, nlist: usize, bits_per_dim: u8) -> Self {
        Self {
            dim,
            nlist,
            nprobe: 1,
            metric_type: MetricType::L2,
            bits_per_dim,
            rotation_seed: 42,
            hadamard: true,
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

    pub fn with_hadamard(mut self, hadamard: bool) -> Self {
        self.hadamard = hadamard;
        self
    }
}

pub type IvfTurboQuantEntry = (i64, Vec<u8>);

pub struct IvfTurboQuantIndex {
    config: IvfTurboQuantConfig,
    centroids: Vec<f32>,
    inverted_lists: Arc<RwLock<HashMap<usize, Vec<IvfTurboQuantEntry>>>>,
    quantizer: TurboQuantMse,
    trained: bool,
    ntotal: usize,
}

impl IvfTurboQuantIndex {
    pub fn new(config: IvfTurboQuantConfig) -> Self {
        let quantizer = TurboQuantMse::new(Self::build_quantizer_config(&config));
        Self {
            config,
            centroids: Vec::new(),
            inverted_lists: Arc::new(RwLock::new(HashMap::new())),
            quantizer,
            trained: false,
            ntotal: 0,
        }
    }

    pub fn train(&mut self, data: &[f32]) -> Result<()> {
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

        let training = self.preprocess_dataset(data);
        let mut km = KMeans::new(self.config.nlist, self.config.dim);
        km.train(&training);
        self.centroids = km.centroids().to_vec();
        self.quantizer = TurboQuantMse::new(Self::build_quantizer_config(&self.config));
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

        let mut lists = self.inverted_lists.write();
        for i in 0..n {
            let raw = &data[i * self.config.dim..(i + 1) * self.config.dim];
            let vector = self.preprocess_vector(raw);
            let cluster = self.find_best_centroid(&vector);
            let code = self.quantizer.encode(&vector);
            let id = ids
                .map(|values| values[i])
                .unwrap_or((self.ntotal + i) as i64);
            lists.entry(cluster).or_default().push((id, code));
        }

        self.ntotal += n;
        Ok(n)
    }

    pub fn search(&self, query: &[f32], req: &SearchRequest) -> Result<SearchResult> {
        if !self.trained {
            return Err(KnowhereError::InvalidArg("index not trained".to_string()));
        }
        let nq = query.len() / self.config.dim;
        if nq * self.config.dim != query.len() {
            return Err(KnowhereError::InvalidArg(
                "query dimension mismatch".to_string(),
            ));
        }

        let top_k = req.top_k.max(1);
        let nprobe = req.nprobe.max(1).min(self.config.nlist);
        let filter_ref = req.filter.as_ref().map(|f| f.as_ref());

        let mut ids = Vec::with_capacity(nq * top_k);
        let mut distances = Vec::with_capacity(nq * top_k);
        for query_vec in query.chunks_exact(self.config.dim) {
            let mut batch = self.search_single(query_vec, top_k, nprobe, filter_ref);
            batch.truncate(top_k);
            while batch.len() < top_k {
                batch.push((-1, f32::INFINITY));
            }
            for (id, distance) in batch {
                ids.push(id);
                distances.push(distance);
            }
        }

        Ok(SearchResult::new(ids, distances, 0.0))
    }

    pub fn count(&self) -> usize {
        self.ntotal
    }

    fn build_quantizer_config(ivf_config: &IvfTurboQuantConfig) -> TurboQuantConfig {
        let base = TurboQuantConfig::new(ivf_config.dim, ivf_config.bits_per_dim)
            .with_rotation_seed(ivf_config.rotation_seed);
        if ivf_config.hadamard {
            base.with_hadamard()
        } else {
            base.with_dense_rotation()
        }
    }

    fn search_single(
        &self,
        query: &[f32],
        top_k: usize,
        nprobe: usize,
        filter: Option<&dyn Predicate>,
    ) -> Vec<(i64, f32)> {
        let processed_query = self.preprocess_vector(query);
        let coarse = self.rank_centroids(&processed_query, nprobe);
        // Pre-rotate query once outside the cluster loop (O(d log d) with FWHT)
        let query_rotated = self.quantizer.rotate_query(&processed_query);

        let mut candidates = Vec::new();
        let lists = self.inverted_lists.read();
        for cluster in coarse {
            if let Some(list) = lists.get(&cluster) {
                for (id, code) in list {
                    if let Some(predicate) = filter {
                        if !predicate.evaluate(*id) {
                            continue;
                        }
                    }

                    // Encode full vectors (not residuals): score_ip(Π·q, TQ(v)) ≈ q·v
                    let distance = match self.config.metric_type {
                        MetricType::L2 => {
                            let decoded = self.quantizer.decode(code);
                            l2_distance(&processed_query, &decoded)
                        }
                        MetricType::Ip => -self.quantizer.score_ip(&query_rotated, code),
                        MetricType::Cosine => 1.0 - self.quantizer.score_ip(&query_rotated, code),
                        MetricType::Hamming => continue,
                    };

                    candidates.push((*id, distance));
                }
            }
        }

        candidates.sort_by(|left, right| left.1.total_cmp(&right.1));
        candidates.truncate(top_k);
        candidates
    }

    fn rank_centroids(&self, query: &[f32], nprobe: usize) -> Vec<usize> {
        let mut ranked: Vec<(usize, f32)> = self
            .centroids
            .chunks_exact(self.config.dim)
            .enumerate()
            .map(|(idx, centroid)| {
                let score = match self.config.metric_type {
                    MetricType::L2 => l2_distance(query, centroid),
                    MetricType::Ip | MetricType::Cosine => -dot_product(query, centroid),
                    MetricType::Hamming => f32::INFINITY,
                };
                (idx, score)
            })
            .collect();
        ranked.sort_by(|left, right| left.1.total_cmp(&right.1));
        ranked
            .into_iter()
            .take(nprobe.min(self.config.nlist))
            .map(|(idx, _)| idx)
            .collect()
    }

    fn find_best_centroid(&self, vector: &[f32]) -> usize {
        let mut best_idx = 0usize;
        let mut best_score = f32::INFINITY;

        for (idx, centroid) in self.centroids.chunks_exact(self.config.dim).enumerate() {
            let score = match self.config.metric_type {
                MetricType::L2 => l2_distance(vector, centroid),
                MetricType::Ip | MetricType::Cosine => -dot_product(vector, centroid),
                MetricType::Hamming => f32::INFINITY,
            };
            if score < best_score {
                best_score = score;
                best_idx = idx;
            }
        }

        best_idx
    }

    fn preprocess_dataset(&self, data: &[f32]) -> Vec<f32> {
        if self.config.metric_type != MetricType::Cosine {
            return data.to_vec();
        }
        let mut processed = Vec::with_capacity(data.len());
        for chunk in data.chunks_exact(self.config.dim) {
            processed.extend_from_slice(&self.preprocess_vector(chunk));
        }
        processed
    }

    fn preprocess_vector(&self, vector: &[f32]) -> Vec<f32> {
        if self.config.metric_type != MetricType::Cosine {
            return vector.to_vec();
        }
        let norm = vector.iter().map(|&x| x * x).sum::<f32>().sqrt();
        if norm <= 1e-12 {
            return vector.to_vec();
        }
        vector.iter().map(|&x| x / norm).collect()
    }
}

fn dot_product(left: &[f32], right: &[f32]) -> f32 {
    left.iter().zip(right.iter()).map(|(&l, &r)| l * r).sum()
}

fn l2_distance(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right.iter())
        .map(|(&l, &r)| {
            let diff = l - r;
            diff * diff
        })
        .sum()
}
