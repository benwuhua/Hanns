use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

use crate::api::{KnowhereError, MetricType, Predicate, Result, SearchRequest, SearchResult};
use crate::quantization::{KMeans, TurboQuantConfig, TurboQuantMse, TurboQuantProd};

/// How TurboQuant encodes vectors within each IVF list.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TurboQuantEncoding {
    /// Encode raw residual v-c. Centroid scale mismatch when ||v-c|| ≠ 1.
    Residual,
    /// Encode full vector v (no centroid term in scoring). Works when ||v|| ≈ 1.
    FullVector,
    /// Normalize residual to unit norm before encoding; store ||v-c|| per vector.
    /// Paper-recommended: codebook matches unit sphere, q·c is exact, error ∝ ||v-c||²/4^b.
    NormalizedResidual,
    /// Full-vector TurboQuantProd: (b-1)-bit MSE code + 1-bit QJL residual correction.
    Prod,
}

#[derive(Clone, Debug)]
pub struct IvfTurboQuantConfig {
    pub dim: usize,
    pub nlist: usize,
    pub nprobe: usize,
    pub metric_type: MetricType,
    pub bits_per_dim: u8,
    pub rotation_seed: u64,
    pub hadamard: bool,
    pub encoding: TurboQuantEncoding,
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
            encoding: TurboQuantEncoding::NormalizedResidual,
        }
    }

    pub fn with_encoding(mut self, encoding: TurboQuantEncoding) -> Self {
        self.encoding = encoding;
        self
    }

    /// Convenience: set full_vector mode (backward compat with benchmark).
    pub fn with_full_vector(mut self, full_vector: bool) -> Self {
        self.encoding = if full_vector {
            TurboQuantEncoding::FullVector
        } else {
            TurboQuantEncoding::NormalizedResidual
        };
        self
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

/// (id, TQ code, residual_norm). norm=1.0 for FullVector/Residual modes.
pub type IvfTurboQuantEntry = (i64, Vec<u8>, f32);

pub struct IvfTurboQuantIndex {
    config: IvfTurboQuantConfig,
    centroids: Vec<f32>,
    inverted_lists: Arc<RwLock<HashMap<usize, Vec<IvfTurboQuantEntry>>>>,
    quantizer: TurboQuantMse,
    prod_quantizer: Option<TurboQuantProd>,
    trained: bool,
    ntotal: usize,
}

impl IvfTurboQuantIndex {
    pub fn new(config: IvfTurboQuantConfig) -> Self {
        let quantizer = TurboQuantMse::new(Self::build_quantizer_config(&config));
        let prod_quantizer = if config.encoding == TurboQuantEncoding::Prod {
            Some(TurboQuantProd::new(Self::build_quantizer_config(&config)))
        } else {
            None
        };
        Self {
            config,
            centroids: Vec::new(),
            inverted_lists: Arc::new(RwLock::new(HashMap::new())),
            quantizer,
            prod_quantizer,
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
        self.prod_quantizer = if self.config.encoding == TurboQuantEncoding::Prod {
            Some(TurboQuantProd::new(Self::build_quantizer_config(&self.config)))
        } else {
            None
        };
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
            let centroid =
                &self.centroids[cluster * self.config.dim..(cluster + 1) * self.config.dim];

            let (code, r_norm) = match self.config.encoding {
                TurboQuantEncoding::FullVector => {
                    // TQ(Π·v): codebook matches ||v||≈1 directly.
                    (self.quantizer.encode(&vector), 1.0f32)
                }
                TurboQuantEncoding::Prod => {
                    let prod = self.prod_quantizer.as_ref().expect("prod");
                    prod.encode(&vector)
                }
                TurboQuantEncoding::Residual => {
                    // TQ(Π·(v-c)): raw residual, no normalization.
                    let residual = subtract_slices(&vector, centroid);
                    (self.quantizer.encode(&residual), 1.0f32)
                }
                TurboQuantEncoding::NormalizedResidual => {
                    // TQ(Π·(r/||r||)): normalize residual to unit sphere, store ||r||.
                    // Codebook matches unit sphere perfectly. Paper-recommended approach.
                    let residual = subtract_slices(&vector, centroid);
                    let norm = residual.iter().map(|&x| x * x).sum::<f32>().sqrt();
                    if norm <= 1e-12 {
                        // Zero residual → vector equals centroid exactly.
                        (self.quantizer.encode(&residual), 0.0f32)
                    } else {
                        let unit: Vec<f32> = residual.iter().map(|&x| x / norm).collect();
                        (self.quantizer.encode(&unit), norm)
                    }
                }
            };

            let id = ids
                .map(|values| values[i])
                .unwrap_or((self.ntotal + i) as i64);
            lists.entry(cluster).or_default().push((id, code, r_norm));
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
        let query_rotated = if self.config.encoding == TurboQuantEncoding::Prod {
            None
        } else {
            Some(self.quantizer.rotate_query(&processed_query))
        };
        let (prod_y_rotated, prod_s_y) = if let Some(prod) = &self.prod_quantizer {
            (
                Some(prod.rotate_query(&processed_query)),
                Some(prod.qjl_project_query(&processed_query)),
            )
        } else {
            (None, None)
        };

        let mut candidates = Vec::new();
        let lists = self.inverted_lists.read();
        for cluster in coarse {
            let centroid =
                &self.centroids[cluster * self.config.dim..(cluster + 1) * self.config.dim];
            let centroid_score = if matches!(
                self.config.encoding,
                TurboQuantEncoding::Residual | TurboQuantEncoding::NormalizedResidual
            ) && matches!(self.config.metric_type, MetricType::Ip | MetricType::Cosine)
            {
                dot_product(&processed_query, centroid)
            } else {
                0.0
            };

            // For L2 residual modes, pre-rotate (q-c) per cluster.
            let residual_query_rotated = if self.config.metric_type == MetricType::L2
                && matches!(
                    self.config.encoding,
                    TurboQuantEncoding::Residual | TurboQuantEncoding::NormalizedResidual
                )
            {
                let rq = subtract_slices(&processed_query, centroid);
                Some(self.quantizer.rotate_query(&rq))
            } else {
                None
            };

            if let Some(list) = lists.get(&cluster) {
                for (id, code, r_norm) in list {
                    if let Some(predicate) = filter {
                        if !predicate.evaluate(*id) {
                            continue;
                        }
                    }

                    let distance = match self.config.metric_type {
                        MetricType::L2 => match self.config.encoding {
                            TurboQuantEncoding::FullVector => {
                                // ||Π·q - TQ(Π·v)||² ≈ ||q-v||²
                                self.quantizer
                                    .score_l2(query_rotated.as_ref().expect("full"), code)
                            }
                            TurboQuantEncoding::Prod => {
                                let prod = self.prod_quantizer.as_ref().expect("prod");
                                let decoded = prod.decode_mse(code);
                                l2_distance(&processed_query, &decoded)
                            }
                            TurboQuantEncoding::Residual => {
                                // ||Π·(q-c) - TQ(Π·(v-c))||² ≈ ||q-v||²
                                let q_rot =
                                    residual_query_rotated.as_ref().expect("l2 residual");
                                self.quantizer.score_l2(q_rot, code)
                            }
                            TurboQuantEncoding::NormalizedResidual => {
                                // code = TQ(Π·(r/||r||)), need to scale back.
                                // ||q-v||² = ||q-c||² - 2·(q-c)·r + ||r||²
                                // ≈ ||q-c||² - 2·||r||·score_ip(Π·(q-c), code) + ||r||²
                                let q_rot =
                                    residual_query_rotated.as_ref().expect("l2 residual");
                                let qc_l2 = l2_distance(&processed_query, centroid);
                                let ip_approx = self.quantizer.score_ip(q_rot, code);
                                qc_l2 - 2.0 * r_norm * ip_approx + r_norm * r_norm
                            }
                        },
                        MetricType::Ip => match self.config.encoding {
                            TurboQuantEncoding::FullVector => {
                                -self
                                    .quantizer
                                    .score_ip(query_rotated.as_ref().expect("full"), code)
                            }
                            TurboQuantEncoding::Prod => {
                                let prod = self.prod_quantizer.as_ref().expect("prod");
                                -prod.score_ip(
                                    prod_y_rotated.as_ref().expect("prod"),
                                    prod_s_y.as_ref().expect("prod"),
                                    code,
                                    *r_norm,
                                )
                            }
                            TurboQuantEncoding::Residual => {
                                -(centroid_score
                                    + self.quantizer.score_ip(
                                        query_rotated.as_ref().expect("residual"),
                                        code,
                                    ))
                            }
                            TurboQuantEncoding::NormalizedResidual => {
                                // q·v = q·c + ||r||·score_ip(Π·q, TQ(Π·r̂)) where r̂=r/||r||
                                -(centroid_score
                                    + r_norm
                                        * self.quantizer.score_ip(
                                            query_rotated.as_ref().expect("residual"),
                                            code,
                                        ))
                            }
                        },
                        MetricType::Cosine => match self.config.encoding {
                            TurboQuantEncoding::FullVector => {
                                1.0
                                    - self
                                        .quantizer
                                        .score_ip(query_rotated.as_ref().expect("full"), code)
                            }
                            TurboQuantEncoding::Prod => {
                                let prod = self.prod_quantizer.as_ref().expect("prod");
                                1.0
                                    - prod.score_ip(
                                        prod_y_rotated.as_ref().expect("prod"),
                                        prod_s_y.as_ref().expect("prod"),
                                        code,
                                        *r_norm,
                                    )
                            }
                            TurboQuantEncoding::Residual => {
                                1.0 - (centroid_score
                                    + self.quantizer.score_ip(
                                        query_rotated.as_ref().expect("residual"),
                                        code,
                                    ))
                            }
                            TurboQuantEncoding::NormalizedResidual => {
                                1.0 - (centroid_score
                                    + r_norm
                                        * self.quantizer.score_ip(
                                            query_rotated.as_ref().expect("residual"),
                                            code,
                                        ))
                            }
                        },
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

fn subtract_slices(left: &[f32], right: &[f32]) -> Vec<f32> {
    left.iter().zip(right.iter()).map(|(&l, &r)| l - r).collect()
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
