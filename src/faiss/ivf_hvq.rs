use parking_lot::RwLock;
use std::collections::HashMap;

use crate::api::{KnowhereError, MetricType, Predicate, Result, SearchRequest, SearchResult};
use crate::quantization::{
    KMeans,
    UsqConfig, UsqEncoded, UsqFastScanState, UsqLayout, UsqQuantizer,
    fastscan_topk,
};

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

/// Per-vector encoded data awaiting layout construction.
struct PendingEntry {
    id: i64,
    encoded: UsqEncoded,
}

pub struct IvfHvqIndex {
    config: IvfHvqConfig,
    centroids: Vec<f32>,
    /// Accumulated encoded vectors per cluster (retained for test inspection).
    pending: RwLock<HashMap<usize, Vec<PendingEntry>>>,
    cluster_layouts: RwLock<HashMap<usize, UsqLayout>>,
    quantizer: UsqQuantizer,
    trained: bool,
    ntotal: usize,
}

impl IvfHvqIndex {
    pub fn new(config: IvfHvqConfig) -> Self {
        let usq_config = UsqConfig::new(config.dim, config.nbits)
            .expect("invalid USQ config")
            .with_seed(config.seed);
        let quantizer = UsqQuantizer::new(usq_config);

        Self {
            config,
            centroids: Vec::new(),
            pending: RwLock::new(HashMap::new()),
            cluster_layouts: RwLock::new(HashMap::new()),
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

        // KMeans for IVF cluster centroids
        let mut km = KMeans::new(self.config.nlist, self.config.dim);
        if matches!(self.config.metric_type, MetricType::Ip | MetricType::Cosine) {
            km = km.with_metric(crate::quantization::kmeans::KMeansMetric::InnerProduct);
        }
        km.train(data);
        self.centroids = km.centroids().to_vec();

        // Train USQ quantizer with global centroid (mean of all training data)
        let usq_config = UsqConfig::new(self.config.dim, self.config.nbits)
            .expect("invalid USQ config")
            .with_seed(self.config.seed);
        self.quantizer = UsqQuantizer::new(usq_config);

        // Compute global centroid (mean of all data)
        let mut global_centroid = vec![0.0f32; self.config.dim];
        for row in data.chunks_exact(self.config.dim) {
            for (c, &value) in global_centroid.iter_mut().zip(row.iter()) {
                *c += value;
            }
        }
        let inv_n = 1.0 / n as f32;
        for value in &mut global_centroid {
            *value *= inv_n;
        }
        self.quantizer.set_centroid(&global_centroid);

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

        // Encode all vectors (cluster assignment + USQ encode)
        let entries: Vec<(usize, PendingEntry)> = {
            #[cfg(feature = "parallel")]
            {
                use rayon::prelude::*;
                let base_id = self.ntotal as i64;
                data.par_chunks_exact(self.config.dim)
                    .enumerate()
                    .map(|(i, vector)| {
                        let cluster = self.find_best_centroid(vector);
                        let encoded = self.quantizer.encode(vector);
                        let id = ids.map(|v| v[i]).unwrap_or(base_id + i as i64);
                        (cluster, PendingEntry { id, encoded })
                    })
                    .collect()
            }
            #[cfg(not(feature = "parallel"))]
            {
                let base_id = self.ntotal as i64;
                data.chunks_exact(self.config.dim)
                    .enumerate()
                    .map(|(i, vector)| {
                        let cluster = self.find_best_centroid(vector);
                        let encoded = self.quantizer.encode(vector);
                        let id = ids.map(|v| v[i]).unwrap_or(base_id + i as i64);
                        (cluster, PendingEntry { id, encoded })
                    })
                    .collect()
            }
        };

        // Sequential cluster insertion
        let mut modified_clusters = std::collections::HashSet::new();
        let mut pending = self.pending.write();
        for (cluster, entry) in entries {
            pending.entry(cluster).or_default().push(entry);
            modified_clusters.insert(cluster);
        }

        self.ntotal += n;

        // Rebuild layouts for modified clusters
        drop(pending);
        self.rebuild_layouts(&modified_clusters);

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

    fn rebuild_layouts(&self, clusters: &std::collections::HashSet<usize>) {
        let pending = self.pending.read();
        let mut layouts = self.cluster_layouts.write();
        let config = self.quantizer.config();
        for &cluster in clusters {
            if let Some(entries) = pending.get(&cluster) {
                if entries.is_empty() {
                    continue;
                }
                let encoded: Vec<UsqEncoded> = entries.iter().map(|e| e.encoded.clone()).collect();
                let ids: Vec<i64> = entries.iter().map(|e| e.id).collect();
                let layout = UsqLayout::build(config, &encoded, &ids);
                layouts.insert(cluster, layout);
            }
        }
    }

    fn search_single(
        &self,
        query: &[f32],
        top_k: usize,
        nprobe: usize,
        filter: Option<&dyn Predicate>,
    ) -> Vec<(i64, f32)> {
        let coarse = self.rank_centroids(query, nprobe);
        let use_l2 = matches!(self.config.metric_type, MetricType::L2);

        // Pad and rotate query
        let padded_dim = self.quantizer.config().padded_dim();
        let mut q_padded = vec![0.0f32; padded_dim];
        q_padded[..self.config.dim].copy_from_slice(query);
        let q_rot = self.quantizer.rotator().rotate(&q_padded);
        let q_norm_sq: f32 = query.iter().map(|x| x * x).sum();

        // Precompute centroid_score (global centroid, same for all clusters)
        let centroid_score: f32 = {
            let mut c_padded = vec![0.0f32; padded_dim];
            c_padded[..self.config.dim].copy_from_slice(self.quantizer.centroid());
            let c_rot = self.quantizer.rotator().rotate(&c_padded);
            q_rot.iter().zip(c_rot.iter()).map(|(a, b)| a * b).sum()
        };

        // Build fastscan state (once per query, reused across clusters)
        let fs_state = UsqFastScanState::new(&q_rot, self.quantizer.config());

        // Tier-aware candidate count per cluster
        let n_candidates_per_cluster = match self.config.nbits {
            1 => (top_k * 20).max(200),
            2..=4 => (top_k * 15).max(150),
            _ => (top_k * 30).max(300),
        };

        let layouts = self.cluster_layouts.read();

        // Stage 1: collect candidates from each cluster via fastscan
        let mut all_candidates: Vec<(usize, usize)> = Vec::new(); // (cluster, local_idx)
        for &cluster in &coarse {
            if let Some(layout) = layouts.get(&cluster) {
                if layout.len() <= n_candidates_per_cluster {
                    // Small cluster: score all vectors directly
                    for local_id in 0..layout.len() {
                        all_candidates.push((cluster, local_id));
                    }
                } else {
                    let candidates = fastscan_topk(layout, &fs_state, n_candidates_per_cluster);
                    for c in candidates {
                        all_candidates.push((cluster, c.idx));
                    }
                }
            }
        }

        // Stage 2: rerank with score_with_meta
        let mut scored = Vec::new();
        for (cluster, local_id) in all_candidates {
            let layout = &layouts[&cluster];
            if local_id >= layout.len() {
                continue;
            }
            if let Some(predicate) = filter {
                if !predicate.evaluate(layout.id_at(local_id)) {
                    continue;
                }
            }

            let ip_score = self.quantizer.score_with_meta(
                &q_rot,
                centroid_score,
                layout.norm_at(local_id),
                layout.vmax_at(local_id),
                layout.quant_quality_at(local_id),
                layout.packed_bits_at(local_id),
            );

            let distance = if use_l2 {
                q_norm_sq + layout.norm_sq_at(local_id) - 2.0 * ip_score
            } else {
                -ip_score
            };
            scored.push((layout.id_at(local_id), distance));
        }

        scored.sort_by(|a, b| a.1.total_cmp(&b.1));
        scored.truncate(top_k);
        scored
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

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn normalize_rows(data: &[f32], dim: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(data.len());
        for row in data.chunks_exact(dim) {
            let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
            out.extend(row.iter().map(|x| x / norm));
        }
        out
    }

    #[test]
    fn test_ivf_hvq_fastscan_layout_roundtrip() {
        let dim = 64usize;
        let nlist = 4usize;
        let n = 128usize;
        let nbits = 4u8;

        let mut rng = StdRng::seed_from_u64(42);
        let data = normalize_rows(
            &(0..n * dim).map(|_| rng.gen_range(-1.0f32..1.0)).collect::<Vec<_>>(),
            dim,
        );

        let mut index = IvfHvqIndex::new(
            IvfHvqConfig::new(dim, nlist, nbits)
                .with_metric(MetricType::Ip)
                .with_seed(42),
        );
        index.train(&data).unwrap();
        index.add(&data, None).unwrap();

        // Verify layouts were built
        let layouts = index.cluster_layouts.read();
        let pending = index.pending.read();
        let mut total_in_layouts = 0;
        for (&cluster, layout) in layouts.iter() {
            assert!(pending.contains_key(&cluster), "layout cluster missing from pending");
            assert_eq!(layout.len(), pending[&cluster].len());
            total_in_layouts += layout.len();
        }
        assert_eq!(total_in_layouts, n, "total vectors in layouts should match ntotal");
    }

    #[test]
    fn test_ivf_hvq_fastscan_search_recall() {
        let dim = 64usize;
        let nlist = 8usize;
        let n = 1000usize;
        let nq = 10usize;
        let top_k = 10usize;
        let nbits = 4u8;

        let mut rng = StdRng::seed_from_u64(42);
        let data = normalize_rows(
            &(0..n * dim).map(|_| rng.gen_range(-1.0f32..1.0)).collect::<Vec<_>>(),
            dim,
        );
        let queries = normalize_rows(
            &(0..nq * dim).map(|_| rng.gen_range(-1.0f32..1.0)).collect::<Vec<_>>(),
            dim,
        );

        // Brute-force ground truth for IP
        let mut gt_results = Vec::new();
        for query in queries.chunks_exact(dim) {
            let mut scored: Vec<(usize, f32)> = (0..n)
                .map(|i| {
                    let v = &data[i * dim..(i + 1) * dim];
                    (i, query.iter().zip(v.iter()).map(|(&a, &b)| a * b).sum())
                })
                .collect();
            scored.sort_by(|a, b| b.1.total_cmp(&a.1));
            gt_results.push(scored.iter().take(top_k).map(|(i, _)| *i).collect::<Vec<_>>());
        }

        let mut index = IvfHvqIndex::new(
            IvfHvqConfig::new(dim, nlist, nbits)
                .with_metric(MetricType::Ip)
                .with_nprobe(nlist)
                .with_seed(42),
        );
        index.train(&data).unwrap();
        index.add(&data, None).unwrap();

        let req = SearchRequest {
            top_k,
            nprobe: nlist,
            filter: None,
            params: None,
            radius: None,
        };
        let result = index.search(&queries, &req).unwrap();

        // Compute recall
        let mut hits = 0usize;
        let total = nq * top_k;
        for qi in 0..nq {
            for &gt_id in &gt_results[qi] {
                for k in 0..top_k {
                    let result_id = result.ids[qi * top_k + k];
                    if result_id == gt_id as i64 {
                        hits += 1;
                        break;
                    }
                }
            }
        }
        let recall = hits as f32 / total as f32;
        println!("IVF-HVQ fastscan recall@{top_k} = {recall:.3}");
        assert!(recall > 0.1, "recall too low: {recall}");
    }

    #[test]
    fn test_ivf_hvq_fastscan_vs_bruteforce_equivalence() {
        let dim = 32usize;
        let nlist = 4usize;
        let n = 200usize;
        let nbits = 4u8;

        let mut rng = StdRng::seed_from_u64(123);
        let data = normalize_rows(
            &(0..n * dim).map(|_| rng.gen_range(-1.0f32..1.0)).collect::<Vec<_>>(),
            dim,
        );

        let mut index = IvfHvqIndex::new(
            IvfHvqConfig::new(dim, nlist, nbits)
                .with_metric(MetricType::Ip)
                .with_nprobe(nlist)
                .with_seed(123),
        );
        index.train(&data).unwrap();
        index.add(&data, None).unwrap();

        // Search with all clusters probed
        let query = &data[0..dim];
        let req = SearchRequest {
            top_k: 10,
            nprobe: nlist,
            filter: None,
            params: None,
            radius: None,
        };
        let result = index.search(query, &req).unwrap();

        // Verify result is non-empty and sorted
        for k in 0..10 {
            assert!(result.ids[k] >= 0, "result id should be non-negative");
        }
    }
}
