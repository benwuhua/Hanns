use parking_lot::RwLock;
use std::collections::HashMap;

use crate::api::{KnowhereError, MetricType, Predicate, Result, SearchRequest, SearchResult};
use crate::quantization::{HvqConfig, HvqQuantizer, KMeans};

const HVQ_ENCODE_REFINE: usize = 6;

#[derive(Clone, Debug)]
pub struct IvfHvqConfig {
    pub dim: usize,
    pub nlist: usize,
    pub nprobe: usize,
    pub nbits: u8,
    pub metric_type: MetricType,
    pub seed: u64,
}

impl IvfHvqConfig {
    pub fn new(dim: usize, nlist: usize, nbits: u8) -> Self {
        Self {
            dim,
            nlist,
            nprobe: 1,
            nbits,
            metric_type: MetricType::L2,
            seed: 42,
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

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

pub struct IvfHvqEntry {
    pub id: i64,
    pub packed_bits: Vec<u8>,
    pub norm_o: f32,
    pub vmax: f32,
    pub base_quant_dist: f32,
    pub base_norm_sq: f32,
}

pub struct IvfHvqIndex {
    config: IvfHvqConfig,
    centroids: Vec<f32>,
    inverted_lists: RwLock<HashMap<usize, Vec<IvfHvqEntry>>>,
    quantizer: HvqQuantizer,
    trained: bool,
    ntotal: usize,
}

impl IvfHvqIndex {
    pub fn new(config: IvfHvqConfig) -> Self {
        let quantizer = HvqQuantizer::new(
            HvqConfig {
                dim: config.dim,
                nbits: config.nbits,
            },
            config.seed,
        );

        Self {
            config,
            centroids: Vec::new(),
            inverted_lists: RwLock::new(HashMap::new()),
            quantizer,
            trained: false,
            ntotal: 0,
        }
    }

    pub fn train(&mut self, data: &[f32]) -> Result<()> {
        self.validate_metric()?;

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
        if matches!(self.config.metric_type, MetricType::Ip | MetricType::Cosine) {
            km = km.with_metric(crate::quantization::kmeans::KMeansMetric::InnerProduct);
        }
        km.train(data);
        self.centroids = km.centroids().to_vec();

        self.quantizer = HvqQuantizer::new(
            HvqConfig {
                dim: self.config.dim,
                nbits: self.config.nbits,
            },
            self.config.seed,
        );
        self.quantizer.train(n, data);
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
        for (i, vector) in data.chunks_exact(self.config.dim).enumerate() {
            let cluster = self.find_best_centroid(vector);
            let code = self.quantizer.encode(vector, HVQ_ENCODE_REFINE);
            let norm_o = f32::from_le_bytes(code[0..4].try_into().unwrap());
            let vmax = f32::from_le_bytes(code[4..8].try_into().unwrap());
            let base_quant_dist = f32::from_le_bytes(code[8..12].try_into().unwrap());
            let packed_bits = code[12..].to_vec();
            let base_norm_sq = vector.iter().map(|x| x * x).sum::<f32>();
            let id = ids
                .map(|values| values[i])
                .unwrap_or((self.ntotal + i) as i64);
            lists.entry(cluster).or_default().push(IvfHvqEntry {
                id,
                packed_bits,
                norm_o,
                vmax,
                base_quant_dist,
                base_norm_sq,
            });
        }

        self.ntotal += n;
        Ok(n)
    }

    pub fn search(&self, query: &[f32], req: &SearchRequest) -> Result<SearchResult> {
        if !self.trained {
            return Err(KnowhereError::InvalidArg("index not trained".to_string()));
        }
        self.validate_metric()?;

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

    fn search_single(
        &self,
        query: &[f32],
        top_k: usize,
        nprobe: usize,
        filter: Option<&dyn Predicate>,
    ) -> Vec<(i64, f32)> {
        let coarse = self.rank_centroids(query, nprobe);
        let q_rot = self.quantizer.rotate_query(query);
        let state = self.quantizer.precompute_query_state(&q_rot);

        let q_norm_sq: f32 = query.iter().map(|x| x * x).sum();
        let use_l2 = matches!(self.config.metric_type, MetricType::L2);

        let mut candidates = Vec::new();
        let lists = self.inverted_lists.read();
        for cluster in coarse {
            if let Some(list) = lists.get(&cluster) {
                for entry in list {
                    if let Some(predicate) = filter {
                        if !predicate.evaluate(entry.id) {
                            continue;
                        }
                    }

                    let ip_score = self.quantizer.score_code_with_meta(
                        &state,
                        entry.norm_o,
                        entry.vmax,
                        entry.base_quant_dist,
                        &entry.packed_bits,
                    );

                    let distance = if use_l2 {
                        q_norm_sq + entry.base_norm_sq - 2.0 * ip_score
                    } else {
                        -ip_score
                    };
                    candidates.push((entry.id, distance));
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
            .map(|(idx, centroid)| (idx, l2_distance(query, centroid)))
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
        let mut best_dist = f32::INFINITY;

        for (idx, centroid) in self.centroids.chunks_exact(self.config.dim).enumerate() {
            let dist = l2_distance(vector, centroid);
            if dist < best_dist {
                best_dist = dist;
                best_idx = idx;
            }
        }

        best_idx
    }

    fn validate_metric(&self) -> Result<()> {
        if matches!(self.config.metric_type, MetricType::L2 | MetricType::Ip) {
            Ok(())
        } else {
            Err(KnowhereError::InvalidArg(
                "IVF-HVQ supports only L2 and IP metrics".to_string(),
            ))
        }
    }
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
