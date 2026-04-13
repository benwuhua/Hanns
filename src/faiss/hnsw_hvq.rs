//! HNSW-HVQ Index Implementation
//!
//! Minimal HNSW graph search backed by USQ compressed storage and USQ coarse
//! scoring. This path is IP/Cosine-oriented: higher score is better.

use rand::Rng;
use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

use crate::api::MetricType;
use crate::quantization::usq::{UsqConfig, UsqEncoded, UsqQuantizer, UsqQueryState};

const MAX_LAYERS: usize = 16;
const USQ_TRAIN_SEED: u64 = 42;
const HNSWHVQ_MAGIC: &[u8; 8] = b"HNSWHVQ0";

#[derive(Clone, Debug)]
pub struct HnswHvqConfig {
    pub dim: usize,
    pub m: usize,
    pub m_max0: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
    pub nbits: u8,
    pub metric_type: MetricType,
}

impl Default for HnswHvqConfig {
    fn default() -> Self {
        Self {
            dim: 0,
            m: 16,
            m_max0: 32,
            ef_construction: 100,
            ef_search: 64,
            nbits: 4,
            metric_type: MetricType::Ip,
        }
    }
}

impl HnswHvqConfig {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            ..Default::default()
        }
    }
}

#[derive(Clone, Debug)]
struct UsqNodeInfo {
    max_layer: usize,
    layer_neighbors: Vec<Vec<usize>>,
    layer_neighbor_scores: Vec<Vec<f32>>,
    encoded: UsqEncoded,
}

impl UsqNodeInfo {
    fn new(max_layer: usize, encoded: UsqEncoded, m: usize) -> Self {
        let mut layer_neighbors = Vec::with_capacity(max_layer + 1);
        let mut layer_neighbor_scores = Vec::with_capacity(max_layer + 1);
        for layer in 0..=max_layer {
            let capacity = if layer == 0 { m * 2 } else { m };
            layer_neighbors.push(Vec::with_capacity(capacity));
            layer_neighbor_scores.push(Vec::with_capacity(capacity));
        }
        Self {
            max_layer,
            layer_neighbors,
            layer_neighbor_scores,
            encoded,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct ScoreOrd(f32);

impl Eq for ScoreOrd {}

impl PartialOrd for ScoreOrd {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScoreOrd {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

pub struct HnswHvqIndex {
    config: HnswHvqConfig,
    quantizer: UsqQuantizer,
    node_info: Vec<UsqNodeInfo>,
    ids: Vec<i64>,
    id_to_idx: HashMap<i64, usize>,
    entry_point: Option<usize>,
    max_level: usize,
    trained: bool,
    level_multiplier: f64,
    next_id: i64,
}

impl HnswHvqIndex {
    pub fn new(config: HnswHvqConfig) -> Self {
        assert!(config.dim > 0, "dimension must be > 0");
        assert!(config.m > 0, "m must be > 0");
        assert!(config.m_max0 >= config.m, "m_max0 must be >= m");
        assert!(
            matches!(config.metric_type, MetricType::Ip | MetricType::Cosine),
            "HNSW-HVQ currently supports only IP/Cosine scoring"
        );

        let usq_config = UsqConfig::new(config.dim, config.nbits)
            .expect("invalid USQ config")
            .with_seed(USQ_TRAIN_SEED);
        let quantizer = UsqQuantizer::new(usq_config);

        Self {
            config,
            quantizer,
            node_info: Vec::new(),
            ids: Vec::new(),
            id_to_idx: HashMap::new(),
            entry_point: None,
            max_level: 0,
            trained: false,
            level_multiplier: 1.0 / (16.0f64).ln(),
            next_id: 0,
        }
        .with_level_multiplier()
    }

    fn with_level_multiplier(mut self) -> Self {
        self.level_multiplier = 1.0 / (self.config.m as f64).ln().max(1.0);
        self
    }

    #[inline]
    fn max_connections_for_layer(&self, layer: usize) -> usize {
        if layer == 0 {
            self.config.m_max0
        } else {
            self.config.m
        }
    }

    fn random_level(&self) -> usize {
        let mut rng = rand::thread_rng();
        let r = rng.gen::<f64>().max(f64::MIN_POSITIVE);
        let level = (-r.ln() * self.level_multiplier) as usize;
        level.min(MAX_LAYERS - 1)
    }

    /// Prepare a query: pad, rotate, and compute centroid score.
    fn prepare_query_state(&self, query: &[f32]) -> UsqQueryState {
        self.quantizer.precompute_query_state(query)
    }

    pub fn train(&mut self, data: &[f32], n: usize) {
        assert_eq!(
            data.len(),
            n * self.config.dim,
            "expected {} floats, got {}",
            n * self.config.dim,
            data.len()
        );
        assert!(n > 0, "training data must be non-empty");

        // Compute centroid.
        let dim = self.config.dim;
        let mut centroid = vec![0.0f32; dim];
        for row in data.chunks_exact(dim) {
            for (c, &value) in centroid.iter_mut().zip(row.iter()) {
                *c += value;
            }
        }
        let inv_n = 1.0 / n as f32;
        for value in &mut centroid {
            *value *= inv_n;
        }
        self.quantizer.set_centroid(&centroid);
        self.trained = true;
    }

    pub fn add(&mut self, data: &[f32], n: usize) {
        assert_eq!(
            data.len(),
            n * self.config.dim,
            "expected {} floats, got {}",
            n * self.config.dim,
            data.len()
        );
        if n == 0 {
            return;
        }

        if !self.trained {
            self.train(data, n);
        }

        self.node_info.reserve(n);
        self.ids.reserve(n);
        self.id_to_idx.reserve(n);

        for row in data.chunks_exact(self.config.dim) {
            let node_level = self.random_level();
            let encoded = self.quantizer.encode(row);
            let idx = self.ids.len();
            let id = self.next_id;
            self.next_id += 1;

            self.node_info
                .push(UsqNodeInfo::new(node_level, encoded, self.config.m));
            self.ids.push(id);
            self.id_to_idx.insert(id, idx);

            if self.entry_point.is_none() {
                self.entry_point = Some(idx);
                self.max_level = node_level;
                continue;
            }

            self.insert_node(idx, row, node_level);

            if node_level > self.max_level {
                self.max_level = node_level;
                self.entry_point = Some(idx);
            }
        }
    }

    pub fn search(&self, query: &[f32], top_k: usize) -> Vec<(i64, f32)> {
        assert_eq!(query.len(), self.config.dim);
        if !self.trained || self.node_info.is_empty() || top_k == 0 {
            return Vec::new();
        }

        let state = self.prepare_query_state(query);

        let mut current = self.entry_point.unwrap();
        for layer in (1..=self.max_level).rev() {
            current = self.search_at_layer_greedy(&state, current, layer);
        }

        let ef = self.config.ef_search.max(top_k);
        let mut candidates = self.search_layer_ef(&state, current, 0, ef);
        candidates.truncate(top_k);
        candidates
            .into_iter()
            .map(|(idx, score)| (self.ids[idx], score))
            .collect()
    }

    pub fn count(&self) -> usize {
        self.ids.len()
    }

    pub fn is_trained(&self) -> bool {
        self.trained
    }

    pub fn save(&self, path: &Path) -> crate::api::Result<()> {
        let mut file = File::create(path)?;

        file.write_all(HNSWHVQ_MAGIC)?;
        file.write_all(&(self.config.dim as u32).to_le_bytes())?;
        file.write_all(&(self.config.m as u32).to_le_bytes())?;
        file.write_all(&(self.config.m_max0 as u32).to_le_bytes())?;
        file.write_all(&(self.config.ef_construction as u32).to_le_bytes())?;
        file.write_all(&(self.config.ef_search as u32).to_le_bytes())?;
        file.write_all(&[self.config.nbits])?;
        let metric_tag = match self.config.metric_type {
            MetricType::L2 => 0u8,
            MetricType::Ip => 1u8,
            MetricType::Cosine => 2u8,
            MetricType::Hamming => 3u8,
        };
        file.write_all(&[metric_tag])?;
        file.write_all(&(self.entry_point.unwrap_or(usize::MAX) as u64).to_le_bytes())?;
        file.write_all(&(self.max_level as u32).to_le_bytes())?;
        file.write_all(&self.next_id.to_le_bytes())?;
        file.write_all(&self.level_multiplier.to_le_bytes())?;
        file.write_all(&[u8::from(self.trained)])?;

        let centroid = self.quantizer.centroid();
        file.write_all(&(centroid.len() as u64).to_le_bytes())?;
        for &v in centroid {
            file.write_all(&v.to_le_bytes())?;
        }

        file.write_all(&(self.node_info.len() as u64).to_le_bytes())?;
        for node in &self.node_info {
            file.write_all(&(node.max_layer as u32).to_le_bytes())?;
            file.write_all(&(node.layer_neighbors.len() as u32).to_le_bytes())?;
            for (neighbors, scores) in node
                .layer_neighbors
                .iter()
                .zip(node.layer_neighbor_scores.iter())
            {
                file.write_all(&(neighbors.len() as u32).to_le_bytes())?;
                for (&neighbor, &score) in neighbors.iter().zip(scores.iter()) {
                    file.write_all(&(neighbor as u64).to_le_bytes())?;
                    file.write_all(&score.to_le_bytes())?;
                }
            }

            file.write_all(&(node.encoded.packed_bits.len() as u32).to_le_bytes())?;
            file.write_all(&node.encoded.packed_bits)?;
            file.write_all(&(node.encoded.sign_bits.len() as u32).to_le_bytes())?;
            file.write_all(&node.encoded.sign_bits)?;
            file.write_all(&node.encoded.norm.to_le_bytes())?;
            file.write_all(&node.encoded.norm_sq.to_le_bytes())?;
            file.write_all(&node.encoded.vmax.to_le_bytes())?;
            file.write_all(&node.encoded.quant_quality.to_le_bytes())?;
        }

        file.write_all(&(self.ids.len() as u64).to_le_bytes())?;
        for &id in &self.ids {
            file.write_all(&id.to_le_bytes())?;
        }

        Ok(())
    }

    pub fn load(path: &Path) -> crate::api::Result<Self> {
        let mut file = File::open(path)?;

        let mut magic = [0u8; 8];
        file.read_exact(&mut magic)?;
        if &magic != HNSWHVQ_MAGIC {
            return Err(crate::api::KnowhereError::Codec(
                "invalid HNSWHVQ magic".to_string(),
            ));
        }

        let mut u32_buf = [0u8; 4];
        let mut u64_buf = [0u8; 8];
        let mut i64_buf = [0u8; 8];
        let mut f32_buf = [0u8; 4];
        let mut f64_buf = [0u8; 8];

        file.read_exact(&mut u32_buf)?;
        let dim = u32::from_le_bytes(u32_buf) as usize;
        file.read_exact(&mut u32_buf)?;
        let m = u32::from_le_bytes(u32_buf) as usize;
        file.read_exact(&mut u32_buf)?;
        let m_max0 = u32::from_le_bytes(u32_buf) as usize;
        file.read_exact(&mut u32_buf)?;
        let ef_construction = u32::from_le_bytes(u32_buf) as usize;
        file.read_exact(&mut u32_buf)?;
        let ef_search = u32::from_le_bytes(u32_buf) as usize;

        let mut one = [0u8; 1];
        file.read_exact(&mut one)?;
        let nbits = one[0];
        file.read_exact(&mut one)?;
        let metric_type = match one[0] {
            0 => MetricType::L2,
            1 => MetricType::Ip,
            2 => MetricType::Cosine,
            3 => MetricType::Hamming,
            other => {
                return Err(crate::api::KnowhereError::Codec(format!(
                    "unknown HNSWHVQ metric type tag {other}"
                )))
            }
        };

        file.read_exact(&mut u64_buf)?;
        let entry_raw = u64::from_le_bytes(u64_buf) as usize;
        let entry_point = if entry_raw == usize::MAX {
            None
        } else {
            Some(entry_raw)
        };

        file.read_exact(&mut u32_buf)?;
        let max_level = u32::from_le_bytes(u32_buf) as usize;
        file.read_exact(&mut i64_buf)?;
        let next_id = i64::from_le_bytes(i64_buf);
        file.read_exact(&mut f64_buf)?;
        let level_multiplier = f64::from_le_bytes(f64_buf);
        file.read_exact(&mut one)?;
        let trained = one[0] != 0;

        file.read_exact(&mut u64_buf)?;
        let centroid_len = u64::from_le_bytes(u64_buf) as usize;
        let mut centroid = vec![0.0f32; centroid_len];
        for value in &mut centroid {
            file.read_exact(&mut f32_buf)?;
            *value = f32::from_le_bytes(f32_buf);
        }

        let config = HnswHvqConfig {
            dim,
            m,
            m_max0,
            ef_construction,
            ef_search,
            nbits,
            metric_type,
        };

        let mut quantizer = UsqQuantizer::new(
            UsqConfig::new(config.dim, config.nbits)
                .expect("valid config")
                .with_seed(USQ_TRAIN_SEED),
        );
        if centroid_len == dim {
            quantizer.set_centroid(&centroid);
        }

        file.read_exact(&mut u64_buf)?;
        let node_count = u64::from_le_bytes(u64_buf) as usize;
        let mut node_info = Vec::with_capacity(node_count);
        for _ in 0..node_count {
            file.read_exact(&mut u32_buf)?;
            let max_layer_node = u32::from_le_bytes(u32_buf) as usize;
            file.read_exact(&mut u32_buf)?;
            let layer_count = u32::from_le_bytes(u32_buf) as usize;
            let mut layer_neighbors = Vec::with_capacity(layer_count);
            let mut layer_neighbor_scores = Vec::with_capacity(layer_count);
            for _ in 0..layer_count {
                file.read_exact(&mut u32_buf)?;
                let neighbor_count = u32::from_le_bytes(u32_buf) as usize;
                let mut neighbors = Vec::with_capacity(neighbor_count);
                let mut scores = Vec::with_capacity(neighbor_count);
                for _ in 0..neighbor_count {
                    file.read_exact(&mut u64_buf)?;
                    let neighbor = u64::from_le_bytes(u64_buf) as usize;
                    file.read_exact(&mut f32_buf)?;
                    let score = f32::from_le_bytes(f32_buf);
                    neighbors.push(neighbor);
                    scores.push(score);
                }
                layer_neighbors.push(neighbors);
                layer_neighbor_scores.push(scores);
            }

            file.read_exact(&mut u32_buf)?;
            let packed_len = u32::from_le_bytes(u32_buf) as usize;
            let mut packed_bits = vec![0u8; packed_len];
            file.read_exact(&mut packed_bits)?;

            file.read_exact(&mut u32_buf)?;
            let sign_len = u32::from_le_bytes(u32_buf) as usize;
            let mut sign_bits = vec![0u8; sign_len];
            file.read_exact(&mut sign_bits)?;

            file.read_exact(&mut f32_buf)?;
            let norm = f32::from_le_bytes(f32_buf);
            file.read_exact(&mut f32_buf)?;
            let norm_sq = f32::from_le_bytes(f32_buf);
            file.read_exact(&mut f32_buf)?;
            let vmax = f32::from_le_bytes(f32_buf);
            file.read_exact(&mut f32_buf)?;
            let quant_quality = f32::from_le_bytes(f32_buf);

            node_info.push(UsqNodeInfo {
                max_layer: max_layer_node,
                layer_neighbors,
                layer_neighbor_scores,
                encoded: UsqEncoded {
                    packed_bits,
                    sign_bits,
                    norm,
                    norm_sq,
                    vmax,
                    quant_quality,
                },
            });
        }

        file.read_exact(&mut u64_buf)?;
        let ids_len = u64::from_le_bytes(u64_buf) as usize;
        let mut ids = vec![0i64; ids_len];
        for id in &mut ids {
            file.read_exact(&mut i64_buf)?;
            *id = i64::from_le_bytes(i64_buf);
        }
        let id_to_idx = ids
            .iter()
            .enumerate()
            .map(|(idx, &id)| (id, idx))
            .collect::<HashMap<_, _>>();

        Ok(Self {
            config,
            quantizer,
            node_info,
            ids,
            id_to_idx,
            entry_point,
            max_level,
            trained,
            level_multiplier,
            next_id,
        })
    }

    fn insert_node(&mut self, idx: usize, vector: &[f32], node_level: usize) {
        let state = self.prepare_query_state(vector);
        let mut entry = self.entry_point.unwrap();

        if self.max_level > node_level {
            for layer in ((node_level + 1)..=self.max_level).rev() {
                entry = self.search_at_layer_greedy(&state, entry, layer);
            }
        }

        let connect_top = node_level.min(self.max_level);
        for layer in (0..=connect_top).rev() {
            let ef = self
                .config
                .ef_construction
                .max(self.max_connections_for_layer(layer));
            let candidates = self.search_layer_ef(&state, entry, layer, ef);
            let neighbors = self.select_neighbors(candidates, idx, layer);

            for (neighbor_idx, score) in neighbors {
                self.add_directed_edge(idx, neighbor_idx, layer, score);
                self.add_directed_edge(neighbor_idx, idx, layer, score);
            }

            if let Some(&next_entry) = self.node_info[idx].layer_neighbors[layer].first() {
                entry = next_entry;
            }
        }
    }

    fn select_neighbors(
        &self,
        mut candidates: Vec<(usize, f32)>,
        exclude_idx: usize,
        layer: usize,
    ) -> Vec<(usize, f32)> {
        let max_m = self.max_connections_for_layer(layer);
        candidates.sort_by(|a, b| b.1.total_cmp(&a.1));

        let mut seen = HashSet::new();
        let mut selected = Vec::with_capacity(max_m);
        for (idx, score) in candidates {
            if idx == exclude_idx || !seen.insert(idx) {
                continue;
            }
            selected.push((idx, score));
            if selected.len() >= max_m {
                break;
            }
        }
        selected
    }

    fn add_directed_edge(&mut self, src: usize, dst: usize, layer: usize, score_hint: f32) {
        if layer > self.node_info[src].max_layer {
            return;
        }

        let cap = self.max_connections_for_layer(layer);
        let node = &mut self.node_info[src];
        let neighbors = &mut node.layer_neighbors[layer];
        let scores = &mut node.layer_neighbor_scores[layer];

        if let Some(pos) = neighbors.iter().position(|&existing| existing == dst) {
            if score_hint > scores[pos] {
                scores[pos] = score_hint;
            }
            return;
        }

        if neighbors.len() < cap {
            neighbors.push(dst);
            scores.push(score_hint);
            return;
        }

        if let Some((worst_pos, worst_score)) = scores
            .iter()
            .copied()
            .enumerate()
            .min_by(|a, b| a.1.total_cmp(&b.1))
        {
            if score_hint > worst_score {
                neighbors[worst_pos] = dst;
                scores[worst_pos] = score_hint;
            }
        }
    }

    fn score_node(&self, state: &UsqQueryState, idx: usize) -> f32 {
        match self.config.metric_type {
            MetricType::Ip | MetricType::Cosine => {
                let enc = &self.node_info[idx].encoded;
                self.quantizer.score_with_meta(
                    state,
                    enc.norm,
                    enc.vmax,
                    enc.quant_quality,
                    &enc.packed_bits,
                )
            }
            _ => unreachable!("metric was validated at construction"),
        }
    }

    fn search_at_layer_greedy(&self, state: &UsqQueryState, entry: usize, layer: usize) -> usize {
        let mut current = entry;
        let mut current_score = self.score_node(state, current);

        loop {
            let mut improved = false;
            if layer <= self.node_info[current].max_layer {
                for &neighbor in &self.node_info[current].layer_neighbors[layer] {
                    let score = self.score_node(state, neighbor);
                    if score > current_score + 1e-6 {
                        current = neighbor;
                        current_score = score;
                        improved = true;
                    }
                }
            }
            if !improved {
                break;
            }
        }

        current
    }

    fn search_layer_ef(
        &self,
        state: &UsqQueryState,
        entry: usize,
        layer: usize,
        ef: usize,
    ) -> Vec<(usize, f32)> {
        let mut visited = HashSet::new();
        let mut candidates: BinaryHeap<(ScoreOrd, usize)> = BinaryHeap::new();
        let mut results: BinaryHeap<(Reverse<ScoreOrd>, usize)> = BinaryHeap::new();

        let entry_score = self.score_node(state, entry);
        visited.insert(entry);
        candidates.push((ScoreOrd(entry_score), entry));
        results.push((Reverse(ScoreOrd(entry_score)), entry));

        while let Some((ScoreOrd(candidate_score), candidate_idx)) = candidates.pop() {
            if results.len() >= ef {
                if let Some(&(Reverse(ScoreOrd(worst_score)), _)) = results.peek() {
                    if candidate_score < worst_score {
                        break;
                    }
                }
            }

            if layer > self.node_info[candidate_idx].max_layer {
                continue;
            }

            for &neighbor_idx in &self.node_info[candidate_idx].layer_neighbors[layer] {
                if !visited.insert(neighbor_idx) {
                    continue;
                }

                let score = self.score_node(state, neighbor_idx);
                if results.len() < ef {
                    candidates.push((ScoreOrd(score), neighbor_idx));
                    results.push((Reverse(ScoreOrd(score)), neighbor_idx));
                    continue;
                }

                if let Some(&(Reverse(ScoreOrd(worst_score)), _)) = results.peek() {
                    if score > worst_score {
                        results.pop();
                        results.push((Reverse(ScoreOrd(score)), neighbor_idx));
                        candidates.push((ScoreOrd(score), neighbor_idx));
                    }
                }
            }
        }

        let mut out: Vec<(usize, f32)> = results
            .into_iter()
            .map(|(Reverse(ScoreOrd(score)), idx)| (idx, score))
            .collect();
        out.sort_by(|a, b| b.1.total_cmp(&a.1));
        out
    }
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

    fn brute_force_topk_ip(data: &[f32], query: &[f32], dim: usize, top_k: usize) -> Vec<usize> {
        let mut scored: Vec<(usize, f32)> = data
            .chunks_exact(dim)
            .enumerate()
            .map(|(idx, row)| {
                (
                    idx,
                    query
                        .iter()
                        .zip(row.iter())
                        .map(|(&a, &b)| a * b)
                        .sum::<f32>(),
                )
            })
            .collect();
        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
        scored.into_iter().take(top_k).map(|(idx, _)| idx).collect()
    }

    #[test]
    fn test_hnsw_hvq_basic() {
        let dim = 32usize;
        let n = 500usize;
        let nq = 10usize;
        let top_k = 10usize;

        let mut rng = StdRng::seed_from_u64(7);
        let data = normalize_rows(
            &(0..n * dim)
                .map(|_| rng.gen_range(-1.0f32..1.0))
                .collect::<Vec<_>>(),
            dim,
        );
        let queries = normalize_rows(
            &(0..nq * dim)
                .map(|_| rng.gen_range(-1.0f32..1.0))
                .collect::<Vec<_>>(),
            dim,
        );

        let config = HnswHvqConfig {
            dim,
            m: 16,
            m_max0: 32,
            ef_construction: 100,
            ef_search: 64,
            nbits: 4,
            metric_type: MetricType::Ip,
        };

        let mut index = HnswHvqIndex::new(config);
        index.train(&data, n);
        index.add(&data, n);

        assert!(index.is_trained());
        assert_eq!(index.count(), n);

        for query in queries.chunks_exact(dim) {
            let results = index.search(query, top_k);
            assert_eq!(results.len(), top_k);
            for pair in results.windows(2) {
                assert!(
                    pair[0].1 >= pair[1].1,
                    "scores must be returned in descending order"
                );
            }
        }
    }

    #[test]
    fn test_hnsw_hvq_recall() {
        let dim = 32usize;
        let n = 500usize;
        let nq = 20usize;
        let top_k = 10usize;

        let mut rng = StdRng::seed_from_u64(19);
        let data = normalize_rows(
            &(0..n * dim)
                .map(|_| rng.gen_range(-1.0f32..1.0))
                .collect::<Vec<_>>(),
            dim,
        );
        let queries = normalize_rows(
            &(0..nq * dim)
                .map(|_| rng.gen_range(-1.0f32..1.0))
                .collect::<Vec<_>>(),
            dim,
        );

        let config = HnswHvqConfig {
            dim,
            m: 16,
            m_max0: 32,
            ef_construction: 100,
            ef_search: 80,
            nbits: 4,
            metric_type: MetricType::Ip,
        };

        let mut index = HnswHvqIndex::new(config);
        index.train(&data, n);
        index.add(&data, n);

        let mut hits = 0usize;
        let mut total = 0usize;
        for query in queries.chunks_exact(dim) {
            let truth = brute_force_topk_ip(&data, query, dim, top_k);
            let results = index.search(query, top_k);
            let result_ids: Vec<usize> = results.iter().map(|(id, _)| *id as usize).collect();
            for gt in truth {
                total += 1;
                if result_ids.contains(&gt) {
                    hits += 1;
                }
            }
        }

        let recall = hits as f32 / total.max(1) as f32;
        assert!(recall >= 0.5, "expected recall >= 0.5, got {recall}");
    }

    #[test]
    fn test_hnsw_hvq_save_load_roundtrip() {
        let dim = 16usize;
        let n = 100usize;
        let top_k = 5usize;

        let mut rng = StdRng::seed_from_u64(23);
        let data = normalize_rows(
            &(0..n * dim)
                .map(|_| rng.gen_range(-1.0f32..1.0))
                .collect::<Vec<_>>(),
            dim,
        );
        let query = normalize_rows(
            &(0..dim)
                .map(|_| rng.gen_range(-1.0f32..1.0))
                .collect::<Vec<_>>(),
            dim,
        );

        let config = HnswHvqConfig {
            dim,
            m: 16,
            m_max0: 32,
            ef_construction: 100,
            ef_search: 64,
            nbits: 4,
            metric_type: MetricType::Ip,
        };

        let mut index = HnswHvqIndex::new(config);
        index.train(&data, n);
        index.add(&data, n);
        let before = index.search(&query, top_k);

        let temp = tempfile::tempdir().expect("tempdir");
        let path = temp.path().join("hnsw_hvq.bin");
        index.save(&path).expect("save hnsw_hvq");

        let loaded = HnswHvqIndex::load(&path).expect("load hnsw_hvq");
        let after = loaded.search(&query, top_k);

        assert_eq!(after.len(), top_k);
        assert_eq!(
            before.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
            after.iter().map(|(id, _)| *id).collect::<Vec<_>>()
        );
    }
}
