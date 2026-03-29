//! HNSW-HVQ Index Implementation
//!
//! Minimal HNSW graph search backed by HVQ compressed storage and HVQ coarse
//! scoring. This path is IP/Cosine-oriented: higher score is better.

use rand::Rng;
use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap, HashSet};

use crate::api::MetricType;
use crate::quantization::hvq::{HvqConfig, HvqQuantizer, HvqQueryState};

const MAX_LAYERS: usize = 16;
const HVQ_TRAIN_SEED: u64 = 42;
const HVQ_ENCODE_REFINE: usize = 6;

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
struct HvqNodeInfo {
    max_layer: usize,
    layer_neighbors: Vec<Vec<usize>>,
    layer_neighbor_scores: Vec<Vec<f32>>,
    code: Vec<u8>,
}

impl HvqNodeInfo {
    fn new(max_layer: usize, code: Vec<u8>, m: usize) -> Self {
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
            code,
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
    quantizer: HvqQuantizer,
    node_info: Vec<HvqNodeInfo>,
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

        let quantizer = HvqQuantizer::new(
            HvqConfig {
                dim: config.dim,
                nbits: config.nbits,
            },
            HVQ_TRAIN_SEED,
        );

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

    pub fn train(&mut self, data: &[f32], n: usize) {
        assert_eq!(
            data.len(),
            n * self.config.dim,
            "expected {} floats, got {}",
            n * self.config.dim,
            data.len()
        );
        assert!(n > 0, "training data must be non-empty");
        self.quantizer.train(n, data);
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
            let code = self.quantizer.encode(row, HVQ_ENCODE_REFINE);
            let idx = self.ids.len();
            let id = self.next_id;
            self.next_id += 1;

            self.node_info
                .push(HvqNodeInfo::new(node_level, code, self.config.m));
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

        let q_rot = self.quantizer.rotate_query(query);
        let state = self.quantizer.precompute_query_state(&q_rot);

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

    fn insert_node(&mut self, idx: usize, vector: &[f32], node_level: usize) {
        let q_rot = self.quantizer.rotate_query(vector);
        let state = self.quantizer.precompute_query_state(&q_rot);
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

    fn score_node(&self, state: &HvqQueryState, idx: usize) -> f32 {
        match self.config.metric_type {
            MetricType::Ip | MetricType::Cosine => {
                self.quantizer.score_code(state, &self.node_info[idx].code)
            }
            _ => unreachable!("metric was validated at construction"),
        }
    }

    fn search_at_layer_greedy(&self, state: &HvqQueryState, entry: usize, layer: usize) -> usize {
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
        state: &HvqQueryState,
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
}
