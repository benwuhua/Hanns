//! Sparse Index - 稀疏向量索引
//!
//! 支持稀疏向量 (TF-IDF, BM25 等场景)
//! 使用倒排索引实现 TAAT / WAND / MaxScore 搜索

use std::cmp::Ordering;
use std::collections::HashMap;

const BLOCK_SIZE: usize = 32;

/// 稀疏向量 (元素为 (index, value))
#[derive(Clone, Debug, Default)]
pub struct SparseVector {
    pub indices: Vec<u32>, // 非零元素的索引
    pub values: Vec<f32>,  // 对应的值
}

impl SparseVector {
    pub fn new() -> Self {
        Self::default()
    }

    /// 从密集向量创建稀疏向量
    pub fn from_dense(dense: &[f32], threshold: f32) -> Self {
        let mut indices = Vec::new();
        let mut values = Vec::new();

        for (i, &v) in dense.iter().enumerate() {
            if v.abs() > threshold {
                indices.push(i as u32);
                values.push(v);
            }
        }

        let mut sparse = Self { indices, values };
        sparse.sort_by_index();
        sparse
    }

    /// L2 范数
    pub fn norm(&self) -> f32 {
        self.values.iter().map(|v| v * v).sum::<f32>().sqrt()
    }

    /// 元素绝对值和
    pub fn sum_abs(&self) -> f32 {
        self.values.iter().map(|v| v.abs()).sum()
    }

    /// 按索引排序
    pub fn sort_by_index(&mut self) {
        let mut pairs: Vec<(u32, f32)> = self
            .indices
            .iter()
            .copied()
            .zip(self.values.iter().copied())
            .filter(|(_, v)| *v != 0.0)
            .collect();
        pairs.sort_by_key(|(idx, _)| *idx);
        self.indices = pairs.iter().map(|(idx, _)| *idx).collect();
        self.values = pairs.iter().map(|(_, val)| *val).collect();
    }

    /// 点积
    pub fn dot(&self, other: &SparseVector) -> f32 {
        let mut sum = 0.0f32;
        let mut i = 0;
        let mut j = 0;

        while i < self.indices.len() && j < other.indices.len() {
            match self.indices[i].cmp(&other.indices[j]) {
                Ordering::Equal => {
                    sum += self.values[i] * other.values[j];
                    i += 1;
                    j += 1;
                }
                Ordering::Less => i += 1,
                Ordering::Greater => j += 1,
            }
        }

        sum
    }

    /// Cosine 相似度
    pub fn cosine(&self, other: &SparseVector) -> f32 {
        let dot = self.dot(other);
        let norm = self.norm() * other.norm();
        if norm <= f32::EPSILON {
            0.0
        } else {
            dot / norm
        }
    }

    /// 转成密集向量
    pub fn to_dense(&self, dim: usize) -> Vec<f32> {
        let mut dense = vec![0.0f32; dim];
        for (i, &idx) in self.indices.iter().enumerate() {
            if (idx as usize) < dim {
                dense[idx as usize] = self.values[i];
            }
        }
        dense
    }

    fn sorted_pairs(&self) -> Vec<(u32, f32)> {
        let mut pairs: Vec<(u32, f32)> = self
            .indices
            .iter()
            .copied()
            .zip(self.values.iter().copied())
            .filter(|(_, v)| *v != 0.0)
            .collect();
        pairs.sort_by_key(|(idx, _)| *idx);
        pairs
    }
}

/// 搜索算法
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[derive(Default)]
pub enum InvertedIndexAlgo {
    #[default]
    Taat,
    Wand,
    MaxScore,
}


/// 稀疏检索度量
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[derive(Default)]
pub enum SparseMetricType {
    #[default]
    Cosine,
    Bm25,
}


/// BM25 参数
#[derive(Clone, Debug)]
pub struct Bm25Params {
    pub k1: f32,
    pub b: f32,
}

impl Default for Bm25Params {
    fn default() -> Self {
        Self { k1: 1.2, b: 0.75 }
    }
}

impl Bm25Params {
    pub fn score_tf(&self, tf: f32, doc_len: f32, avg_doc_len: f32) -> f32 {
        if tf <= 0.0 {
            return 0.0;
        }
        let avg_doc_len = avg_doc_len.max(1e-6);
        let norm = 1.0 - self.b + self.b * (doc_len / avg_doc_len);
        let denom = tf + self.k1 * norm.max(1e-6);
        if denom <= 0.0 {
            0.0
        } else {
            tf * (self.k1 + 1.0) / denom
        }
    }
}

/// 搜索配置
#[derive(Clone, Debug)]
pub struct SparseSearchParams {
    pub algorithm: InvertedIndexAlgo,
    pub drop_ratio: f32,
    pub refine_factor: usize,
}

impl Default for SparseSearchParams {
    fn default() -> Self {
        Self {
            algorithm: InvertedIndexAlgo::Taat,
            drop_ratio: 0.0,
            refine_factor: 1,
        }
    }
}

#[derive(Clone, Debug)]
struct PostingEntry {
    doc_id: i64,
    value: f32,
    normalized_value: f32,
}

#[derive(Clone, Debug)]
struct PostingBlock {
    start: usize,
    end: usize,
    last_doc_id: i64,
    max_normalized_score: f32,
}

#[derive(Clone, Debug, Default)]
struct PostingList {
    entries: Vec<PostingEntry>,
    blocks: Vec<PostingBlock>,
    max_normalized_score: f32,
}

impl PostingList {
    fn insert(&mut self, entry: PostingEntry) {
        let insert_at = self
            .entries
            .binary_search_by_key(&entry.doc_id, |current| current.doc_id)
            .unwrap_or_else(|pos| pos);
        self.entries.insert(insert_at, entry);
        self.rebuild_blocks();
    }

    fn rebuild_blocks(&mut self) {
        self.blocks.clear();
        self.max_normalized_score = 0.0;

        for block_start in (0..self.entries.len()).step_by(BLOCK_SIZE) {
            let block_end = (block_start + BLOCK_SIZE).min(self.entries.len());
            let slice = &self.entries[block_start..block_end];
            let last_doc_id = slice.last().map(|entry| entry.doc_id).unwrap_or(-1);
            let max_normalized_score = slice
                .iter()
                .map(|entry| entry.normalized_value.abs())
                .fold(0.0f32, f32::max);
            self.max_normalized_score = self.max_normalized_score.max(max_normalized_score);
            self.blocks.push(PostingBlock {
                start: block_start,
                end: block_end,
                last_doc_id,
                max_normalized_score,
            });
        }
    }
}

#[derive(Debug)]
struct TopKHeap {
    capacity: usize,
    data: Vec<(i64, f32)>,
}

impl TopKHeap {
    fn new(capacity: usize) -> Self {
        Self {
            capacity,
            data: Vec::with_capacity(capacity),
        }
    }

    fn push(&mut self, doc_id: i64, score: f32) {
        if self.capacity == 0 {
            return;
        }

        if self.data.len() < self.capacity {
            self.data.push((doc_id, score));
            self.data
                .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
            return;
        }

        if let Some((min_doc, min_score)) = self.data.first_mut() {
            if score > *min_score {
                *min_doc = doc_id;
                *min_score = score;
                self.data
                    .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
            }
        }
    }

    fn min_score(&self) -> f32 {
        if self.data.len() < self.capacity {
            0.0
        } else {
            self.data.first().map(|(_, score)| *score).unwrap_or(0.0)
        }
    }

    fn is_full(&self) -> bool {
        self.data.len() >= self.capacity
    }

    fn into_sorted_vec(mut self) -> Vec<(i64, f32)> {
        self.data
            .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        self.data
    }
}

#[derive(Clone, Debug)]
struct QueryTermState<'a> {
    postings: &'a PostingList,
    query_weight: f32,
    bm25_idf: f32,
    term_upper_bound: f32,
    block_upper_bounds: Vec<f32>,
    cursor: usize,
    block_cursor: usize,
}

impl<'a> QueryTermState<'a> {
    fn is_exhausted(&self) -> bool {
        self.cursor >= self.postings.entries.len()
    }

    fn current_doc_id(&self) -> Option<i64> {
        self.postings.entries.get(self.cursor).map(|entry| entry.doc_id)
    }

    fn current_block_upper_bound(&self) -> f32 {
        if self.is_exhausted() {
            0.0
        } else {
            self.block_upper_bounds
                .get(self.block_cursor)
                .copied()
                .unwrap_or(0.0)
        }
    }

    fn score_current(
        &self,
        index: &SparseIndex,
        metric: SparseMetricType,
        bm25: &Bm25Params,
    ) -> f32 {
        let Some(entry) = self.postings.entries.get(self.cursor) else {
            return 0.0;
        };

        match metric {
            SparseMetricType::Cosine => self.query_weight * entry.normalized_value,
            SparseMetricType::Bm25 => {
                let doc_len = index.doc_lengths.get(&entry.doc_id).copied().unwrap_or(0.0);
                let tf_score = bm25.score_tf(entry.value, doc_len, index.avg_doc_len());
                self.query_weight * self.bm25_idf * tf_score
            }
        }
    }

    fn advance(&mut self) {
        self.cursor += 1;
        self.refresh_block_cursor();
    }

    fn advance_to(&mut self, target_doc_id: i64) {
        if self.is_exhausted() {
            return;
        }

        while self.block_cursor < self.postings.blocks.len() {
            let block = &self.postings.blocks[self.block_cursor];
            if block.last_doc_id >= target_doc_id {
                break;
            }
            self.block_cursor += 1;
            self.cursor = self
                .postings
                .blocks
                .get(self.block_cursor)
                .map(|next| next.start)
                .unwrap_or(self.postings.entries.len());
        }

        while self.cursor < self.postings.entries.len() {
            if self.postings.entries[self.cursor].doc_id >= target_doc_id {
                break;
            }
            self.cursor += 1;
        }

        self.refresh_block_cursor();
    }

    fn refresh_block_cursor(&mut self) {
        while self.block_cursor < self.postings.blocks.len()
            && self.cursor >= self.postings.blocks[self.block_cursor].end
        {
            self.block_cursor += 1;
        }

        if self.block_cursor >= self.postings.blocks.len() {
            self.cursor = self.postings.entries.len();
        } else if self.cursor < self.postings.blocks[self.block_cursor].start {
            self.cursor = self.postings.blocks[self.block_cursor].start;
        }
    }
}

/// 稀疏向量索引
#[allow(dead_code)]
pub struct SparseIndex {
    dim: usize,
    vectors: Vec<SparseVector>,
    ids: Vec<i64>,
    next_id: i64,
    id_to_pos: HashMap<i64, usize>,
    vector_norms: HashMap<i64, f32>,
    doc_lengths: HashMap<i64, f32>,
    total_doc_length: f32,
    metric_type: SparseMetricType,
    bm25_params: Bm25Params,
    search_params: SparseSearchParams,
    inverted_index: HashMap<u32, PostingList>,
}

impl SparseIndex {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            vectors: Vec::new(),
            ids: Vec::new(),
            next_id: 0,
            id_to_pos: HashMap::new(),
            vector_norms: HashMap::new(),
            doc_lengths: HashMap::new(),
            total_doc_length: 0.0,
            metric_type: SparseMetricType::Cosine,
            bm25_params: Bm25Params::default(),
            search_params: SparseSearchParams::default(),
            inverted_index: HashMap::new(),
        }
    }

    pub fn set_algorithm(&mut self, algorithm: InvertedIndexAlgo) {
        self.search_params.algorithm = algorithm;
    }

    pub fn set_search_params(&mut self, params: SparseSearchParams) {
        self.search_params = params;
    }

    pub fn search_params(&self) -> &SparseSearchParams {
        &self.search_params
    }

    pub fn set_metric_type(&mut self, metric_type: SparseMetricType) {
        self.metric_type = metric_type;
    }

    pub fn set_bm25_params(&mut self, k1: f32, b: f32) {
        self.metric_type = SparseMetricType::Bm25;
        self.bm25_params = Bm25Params { k1, b };
    }

    pub fn avg_doc_len(&self) -> f32 {
        if self.vectors.is_empty() {
            1.0
        } else {
            self.total_doc_length / self.vectors.len() as f32
        }
    }

    /// 添加向量
    pub fn add(&mut self, sparse: SparseVector, id: Option<i64>) -> usize {
        let id = id.unwrap_or(self.next_id);
        self.next_id = self.next_id.max(id + 1);

        let mut sparse = sparse;
        sparse.sort_by_index();

        let doc_norm = sparse.norm();
        let doc_len = sparse.sum_abs();
        let position = self.vectors.len();

        self.id_to_pos.insert(id, position);
        self.ids.push(id);
        self.vector_norms.insert(id, doc_norm);
        self.doc_lengths.insert(id, doc_len);
        self.total_doc_length += doc_len;
        self.vectors.push(sparse.clone());

        for (&idx, &val) in sparse.indices.iter().zip(&sparse.values) {
            let normalized_value = if doc_norm > 0.0 { val / doc_norm } else { 0.0 };
            self.inverted_index
                .entry(idx)
                .or_default()
                .insert(PostingEntry {
                    doc_id: id,
                    value: val,
                    normalized_value,
                });
        }

        1
    }

    /// 添加密集向量
    pub fn add_dense(&mut self, dense: &[f32], id: Option<i64>) -> usize {
        let sparse = SparseVector::from_dense(dense, 0.0);
        self.add(sparse, id)
    }

    /// 搜索
    pub fn search(&self, query: &SparseVector, k: usize) -> Vec<(i64, f32)> {
        self.search_with_params(query, k, &self.search_params)
    }

    pub fn search_with_algo(
        &self,
        query: &SparseVector,
        k: usize,
        algorithm: InvertedIndexAlgo,
    ) -> Vec<(i64, f32)> {
        let mut params = self.search_params.clone();
        params.algorithm = algorithm;
        self.search_with_params(query, k, &params)
    }

    pub fn search_with_params(
        &self,
        query: &SparseVector,
        k: usize,
        params: &SparseSearchParams,
    ) -> Vec<(i64, f32)> {
        if k == 0 || self.vectors.is_empty() {
            return Vec::new();
        }

        let query_terms = self.prepare_query_terms(query, params.drop_ratio);
        if query_terms.is_empty() {
            return Vec::new();
        }

        let internal_k = k.saturating_mul(params.refine_factor.max(1));
        let mut results = match params.algorithm {
            InvertedIndexAlgo::Taat => self.search_taat(query, &query_terms, internal_k),
            InvertedIndexAlgo::Wand => self.search_wand(&query_terms, internal_k),
            InvertedIndexAlgo::MaxScore => self.search_maxscore(&query_terms, internal_k),
        };

        results.truncate(k);
        results
    }

    /// 暴力搜索 (用于验证)
    pub fn search_brute(&self, query: &SparseVector, k: usize) -> Vec<(i64, f32)> {
        let mut results: Vec<(i64, f32)> = self
            .vectors
            .iter()
            .zip(self.ids.iter())
            .map(|(v, &id)| (id, self.score_pair(query, v, id)))
            .filter(|(_, score)| *score > 0.0)
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        results.truncate(k);
        results
    }

    fn score_pair(&self, query: &SparseVector, doc: &SparseVector, doc_id: i64) -> f32 {
        match self.metric_type {
            SparseMetricType::Cosine => query.cosine(doc),
            SparseMetricType::Bm25 => {
                let query_pairs = query.sorted_pairs();
                let doc_pairs = doc.sorted_pairs();
                let doc_len = self.doc_lengths.get(&doc_id).copied().unwrap_or(0.0);
                let mut score = 0.0;
                let mut i = 0;
                let mut j = 0;
                while i < query_pairs.len() && j < doc_pairs.len() {
                    match query_pairs[i].0.cmp(&doc_pairs[j].0) {
                        Ordering::Equal => {
                            let idf = self.bm25_idf(query_pairs[i].0);
                            let tf_score =
                                self.bm25_params
                                    .score_tf(doc_pairs[j].1, doc_len, self.avg_doc_len());
                            score += query_pairs[i].1 * idf * tf_score;
                            i += 1;
                            j += 1;
                        }
                        Ordering::Less => i += 1,
                        Ordering::Greater => j += 1,
                    }
                }
                score
            }
        }
    }

    fn search_taat(&self, _query: &SparseVector, query_terms: &[QueryTermState<'_>], k: usize) -> Vec<(i64, f32)> {
        let mut candidates: HashMap<i64, f32> = HashMap::new();

        for term in query_terms {
            for entry in &term.postings.entries {
                let contribution = match self.metric_type {
                    SparseMetricType::Cosine => term.query_weight * entry.normalized_value,
                    SparseMetricType::Bm25 => {
                        let doc_len = self.doc_lengths.get(&entry.doc_id).copied().unwrap_or(0.0);
                        let tf_score = self
                            .bm25_params
                            .score_tf(entry.value, doc_len, self.avg_doc_len());
                        term.query_weight * term.bm25_idf * tf_score
                    }
                };
                if contribution != 0.0 {
                    *candidates.entry(entry.doc_id).or_insert(0.0) += contribution;
                }
            }
        }

        self.collect_topk_from_scores(candidates.into_iter(), k)
    }

    fn search_wand(&self, query_terms: &[QueryTermState<'_>], k: usize) -> Vec<(i64, f32)> {
        let mut states = query_terms.to_vec();
        let mut heap = TopKHeap::new(k);
        let mut threshold = 0.0f32;

        loop {
            if heap.is_full() {
                let total_upper_bound: f32 = states
                    .iter()
                    .filter(|state| !state.is_exhausted())
                    .map(|state| state.current_block_upper_bound())
                    .sum();
                if total_upper_bound <= threshold {
                    break;
                }
            }

            let mut active: Vec<(usize, i64)> = states
                .iter()
                .enumerate()
                .filter_map(|(idx, state)| state.current_doc_id().map(|doc_id| (idx, doc_id)))
                .collect();
            if active.is_empty() {
                break;
            }
            active.sort_by_key(|(_, doc_id)| *doc_id);

            let mut prefix_upper_bound = 0.0f32;
            let mut pivot_doc_id = None;

            for &(idx, doc_id) in &active {
                prefix_upper_bound += states[idx].current_block_upper_bound();
                if prefix_upper_bound > threshold {
                    pivot_doc_id = Some(doc_id);
                    break;
                }
            }

            let Some(pivot_doc_id) = pivot_doc_id else {
                break;
            };

            let min_doc_id = active[0].1;
            if min_doc_id == pivot_doc_id {
                let mut score = 0.0f32;
                let mut matched_terms = Vec::new();

                for (idx, state) in states.iter_mut().enumerate() {
                    state.advance_to(pivot_doc_id);
                    if state.current_doc_id() == Some(pivot_doc_id) {
                        score += state.score_current(self, self.metric_type, &self.bm25_params);
                        matched_terms.push(idx);
                    }
                }

                if score > threshold {
                    heap.push(pivot_doc_id, score);
                    threshold = heap.min_score();
                }

                for idx in matched_terms {
                    states[idx].advance();
                }
            } else {
                for (idx, doc_id) in active {
                    if doc_id >= pivot_doc_id {
                        break;
                    }
                    states[idx].advance_to(pivot_doc_id);
                }
            }
        }

        heap.into_sorted_vec()
    }

    fn search_maxscore(&self, query_terms: &[QueryTermState<'_>], k: usize) -> Vec<(i64, f32)> {
        let mut states = query_terms.to_vec();
        states.sort_by(|a, b| {
            a.term_upper_bound
                .partial_cmp(&b.term_upper_bound)
                .unwrap_or(Ordering::Equal)
        });

        let mut heap = TopKHeap::new(k);
        let mut threshold = 0.0f32;

        loop {
            let total_upper_bound: f32 = states
                .iter()
                .filter(|state| !state.is_exhausted())
                .map(|state| state.term_upper_bound)
                .sum();
            if total_upper_bound <= threshold {
                break;
            }

            let mut optional_upper_bound = 0.0f32;
            let mut first_essential = 0usize;
            while first_essential < states.len() {
                let next = optional_upper_bound + states[first_essential].term_upper_bound;
                if next <= threshold {
                    optional_upper_bound = next;
                    first_essential += 1;
                } else {
                    break;
                }
            }

            let candidate_doc_id = states[first_essential..]
                .iter()
                .filter_map(|state| state.current_doc_id())
                .min();
            let Some(candidate_doc_id) = candidate_doc_id else {
                break;
            };

            let mut score = 0.0f32;
            let mut matched_terms = Vec::new();
            for (idx, state) in states.iter_mut().enumerate().skip(first_essential) {
                state.advance_to(candidate_doc_id);
                if state.current_doc_id() == Some(candidate_doc_id) {
                    score += state.score_current(self, self.metric_type, &self.bm25_params);
                    matched_terms.push(idx);
                }
            }

            if score + optional_upper_bound > threshold {
                for (idx, state) in states.iter_mut().enumerate().take(first_essential) {
                    if state.current_block_upper_bound() == 0.0 {
                        continue;
                    }
                    state.advance_to(candidate_doc_id);
                    if state.current_doc_id() == Some(candidate_doc_id) {
                        score += state.score_current(self, self.metric_type, &self.bm25_params);
                        matched_terms.push(idx);
                    }
                }
            }

            if score > threshold {
                heap.push(candidate_doc_id, score);
                threshold = heap.min_score();
            }

            for idx in matched_terms {
                states[idx].advance();
            }
        }

        heap.into_sorted_vec()
    }

    fn collect_topk_from_scores<I>(&self, iter: I, k: usize) -> Vec<(i64, f32)>
    where
        I: Iterator<Item = (i64, f32)>,
    {
        let mut heap = TopKHeap::new(k);
        for (doc_id, score) in iter {
            if score > 0.0 {
                heap.push(doc_id, score);
            }
        }
        heap.into_sorted_vec()
    }

    fn prepare_query_terms<'a>(
        &'a self,
        query: &SparseVector,
        drop_ratio: f32,
    ) -> Vec<QueryTermState<'a>> {
        let filtered = self.filter_query_terms(query, drop_ratio);
        if filtered.is_empty() {
            return Vec::new();
        }

        let query_norm = filtered.iter().map(|(_, val)| val * val).sum::<f32>().sqrt();

        filtered
            .into_iter()
            .filter_map(|(idx, raw_qval)| {
                let postings = self.inverted_index.get(&idx)?;
                let query_weight = match self.metric_type {
                    SparseMetricType::Cosine => {
                        if query_norm > 0.0 {
                            raw_qval / query_norm
                        } else {
                            0.0
                        }
                    }
                    SparseMetricType::Bm25 => raw_qval,
                };
                let bm25_idf = if self.metric_type == SparseMetricType::Bm25 {
                    self.bm25_idf(idx)
                } else {
                    1.0
                };
                let block_upper_bounds = self.compute_block_upper_bounds(postings, query_weight, bm25_idf);
                let term_upper_bound = block_upper_bounds.iter().copied().fold(0.0, f32::max);

                if term_upper_bound <= 0.0 {
                    return None;
                }

                Some(QueryTermState {
                    postings,
                    query_weight,
                    bm25_idf,
                    term_upper_bound,
                    block_upper_bounds,
                    cursor: 0,
                    block_cursor: 0,
                })
            })
            .collect()
    }

    fn compute_block_upper_bounds(
        &self,
        postings: &PostingList,
        query_weight: f32,
        bm25_idf: f32,
    ) -> Vec<f32> {
        match self.metric_type {
            SparseMetricType::Cosine => postings
                .blocks
                .iter()
                .map(|block| query_weight.abs() * block.max_normalized_score)
                .collect(),
            SparseMetricType::Bm25 => postings
                .blocks
                .iter()
                .map(|block| {
                    postings.entries[block.start..block.end]
                        .iter()
                        .map(|entry| {
                            let doc_len = self.doc_lengths.get(&entry.doc_id).copied().unwrap_or(0.0);
                            let tf_score = self
                                .bm25_params
                                .score_tf(entry.value, doc_len, self.avg_doc_len());
                            query_weight.abs() * bm25_idf * tf_score
                        })
                        .fold(0.0f32, f32::max)
                })
                .collect(),
        }
    }

    fn filter_query_terms(&self, query: &SparseVector, drop_ratio: f32) -> Vec<(u32, f32)> {
        let mut pairs = query.sorted_pairs();
        if pairs.is_empty() {
            return Vec::new();
        }

        let clamped = drop_ratio.clamp(0.0, 0.95);
        if clamped > 0.0 && pairs.len() > 1 {
            let keep = ((pairs.len() as f32) * (1.0 - clamped)).ceil() as usize;
            let keep = keep.max(1).min(pairs.len());
            pairs.sort_by(|a, b| {
                b.1.abs()
                    .partial_cmp(&a.1.abs())
                    .unwrap_or(Ordering::Equal)
                    .then_with(|| a.0.cmp(&b.0))
            });
            pairs.truncate(keep);
            pairs.sort_by_key(|(idx, _)| *idx);
        }

        pairs
    }

    fn bm25_idf(&self, term_idx: u32) -> f32 {
        let df = self
            .inverted_index
            .get(&term_idx)
            .map(|posting| posting.entries.len() as f32)
            .unwrap_or(0.0);
        if df <= 0.0 {
            return 0.0;
        }
        let n = self.vectors.len() as f32;
        ((n - df + 0.5) / (df + 0.5) + 1.0).ln()
    }

    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }
}

/// 稀疏向量批量编码 (TF-IDF 风格)
pub struct TfidfEncoder {
    idf: Vec<f32>,
}

impl Default for TfidfEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl TfidfEncoder {
    pub fn new() -> Self {
        Self { idf: Vec::new() }
    }

    /// 从语料库训练 IDF
    pub fn train(&mut self, corpus: &[Vec<u32>]) {
        let n = corpus.len();
        if n == 0 {
            return;
        }

        let mut doc_freq: HashMap<u32, usize> = HashMap::new();
        for doc in corpus {
            let mut unique_terms = doc.clone();
            unique_terms.sort_unstable();
            unique_terms.dedup();
            for word in unique_terms {
                *doc_freq.entry(word).or_insert(0) += 1;
            }
        }

        let mut max_word = 0u32;
        for &word in doc_freq.keys() {
            max_word = max_word.max(word);
        }

        self.idf = vec![0.0f32; (max_word + 1) as usize];
        for (word, &df) in &doc_freq {
            self.idf[*word as usize] = ((n as f32 + 1.0) / (df as f32 + 1.0)).ln() + 1.0;
        }
    }

    /// 编码单个文档
    pub fn encode(&self, doc: &[u32]) -> SparseVector {
        let mut tf: HashMap<u32, f32> = HashMap::new();
        for &word in doc {
            *tf.entry(word).or_insert(0.0) += 1.0;
        }

        let mut pairs: Vec<(u32, f32)> = tf
            .into_iter()
            .map(|(word, tf_val)| {
                let idf = self.idf.get(word as usize).copied().unwrap_or(1.0);
                (word, tf_val * idf)
            })
            .collect();
        pairs.sort_by_key(|(word, _)| *word);

        SparseVector {
            indices: pairs.iter().map(|(word, _)| *word).collect(),
            values: pairs.iter().map(|(_, value)| *value).collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_query() -> SparseVector {
        SparseVector {
            indices: vec![0, 1, 2],
            values: vec![1.0, 1.0, 1.0],
        }
    }

    #[test]
    fn test_sparse_vector() {
        let v1 = SparseVector {
            indices: vec![0, 2, 4],
            values: vec![1.0, 2.0, 3.0],
        };

        let v2 = SparseVector {
            indices: vec![1, 2, 3],
            values: vec![1.0, 2.0, 3.0],
        };

        assert_eq!(v1.dot(&v2), 4.0);
    }

    #[test]
    fn test_sparse_algorithms_match_bruteforce() {
        let mut index = SparseIndex::new(10);

        index.add(
            SparseVector {
                indices: vec![0, 1],
                values: vec![1.0, 1.0],
            },
            Some(0),
        );
        index.add(
            SparseVector {
                indices: vec![1, 2],
                values: vec![1.0, 2.0],
            },
            Some(1),
        );
        index.add(
            SparseVector {
                indices: vec![0, 2],
                values: vec![0.5, 2.0],
            },
            Some(2),
        );
        index.add(
            SparseVector {
                indices: vec![0, 1, 2],
                values: vec![1.0, 1.0, 1.0],
            },
            Some(3),
        );

        let query = sample_query();
        let brute = index.search_brute(&query, 3);
        let taat = index.search_with_algo(&query, 3, InvertedIndexAlgo::Taat);
        let wand = index.search_with_algo(&query, 3, InvertedIndexAlgo::Wand);
        let maxscore = index.search_with_algo(&query, 3, InvertedIndexAlgo::MaxScore);

        for (lhs, rhs) in taat.iter().zip(brute.iter()) {
            assert_eq!(lhs.0, rhs.0);
            assert!((lhs.1 - rhs.1).abs() < 1e-5);
        }
        for (lhs, rhs) in wand.iter().zip(brute.iter()) {
            assert_eq!(lhs.0, rhs.0);
            assert!((lhs.1 - rhs.1).abs() < 1e-5);
        }
        for (lhs, rhs) in maxscore.iter().zip(brute.iter()) {
            assert_eq!(lhs.0, rhs.0);
            assert!((lhs.1 - rhs.1).abs() < 1e-5);
        }
        assert_eq!(taat[0].0, 3);
    }

    #[test]
    fn test_sparse_drop_ratio_and_refine_factor() {
        let mut index = SparseIndex::new(16);
        for doc_id in 0..8 {
            index.add(
                SparseVector {
                    indices: vec![0, 1, 2, 3],
                    values: vec![
                        1.0 + doc_id as f32 * 0.1,
                        0.4,
                        if doc_id % 2 == 0 { 0.2 } else { 0.0 },
                        if doc_id == 7 { 5.0 } else { 0.1 },
                    ],
                },
                Some(doc_id),
            );
        }

        let query = SparseVector {
            indices: vec![0, 1, 2, 3],
            values: vec![4.0, 0.01, 0.02, 3.0],
        };

        let params = SparseSearchParams {
            algorithm: InvertedIndexAlgo::Wand,
            drop_ratio: 0.5,
            refine_factor: 2,
        };
        let results = index.search_with_params(&query, 2, &params);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 7);
    }

    #[test]
    fn test_sparse_bm25_prefers_shorter_documents() {
        let mut index = SparseIndex::new(8);
        index.set_bm25_params(1.5, 0.75);

        index.add(
            SparseVector {
                indices: vec![1],
                values: vec![2.0],
            },
            Some(0),
        );
        index.add(
            SparseVector {
                indices: vec![1, 5, 6, 7],
                values: vec![2.0, 4.0, 4.0, 4.0],
            },
            Some(1),
        );

        let query = SparseVector {
            indices: vec![1],
            values: vec![1.0],
        };

        let taat = index.search_with_algo(&query, 2, InvertedIndexAlgo::Taat);
        let wand = index.search_with_algo(&query, 2, InvertedIndexAlgo::Wand);
        let maxscore = index.search_with_algo(&query, 2, InvertedIndexAlgo::MaxScore);

        assert_eq!(taat[0].0, 0);
        assert_eq!(wand, taat);
        assert_eq!(maxscore, taat);
        assert!(taat[0].1 > taat[1].1);
    }

    #[test]
    fn test_tfidf() {
        let mut encoder = TfidfEncoder::new();
        let corpus = vec![vec![0, 1, 2], vec![1, 2, 3], vec![0, 2, 4]];

        encoder.train(&corpus);

        let encoded = encoder.encode(&[0, 1]);
        assert!(!encoded.indices.is_empty());
        assert_eq!(encoded.indices.len(), encoded.values.len());
    }
}
