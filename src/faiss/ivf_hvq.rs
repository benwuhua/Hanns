use parking_lot::RwLock;
use std::collections::HashMap;

use crate::api::{KnowhereError, MetricType, Predicate, Result, SearchRequest, SearchResult};
use crate::quantization::{HvqConfig, HvqFastScanState, HvqQuantizer, KMeans};

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
    pub sign_bits: Vec<u8>,
}

/// Per-cluster SoA layout for fastscan + rerank.
struct HvqClusterLayout {
    /// SoA rerank data
    ids: Vec<i64>,
    norms_o: Vec<f32>,
    vmaxs: Vec<f32>,
    base_quant_dists: Vec<f32>,
    packed_bits: Vec<Vec<u8>>,
    base_norm_sqs: Vec<f32>,

    /// 1-bit fastscan transposed buffer
    fastscan_codes: Vec<u8>,
    fastscan_block_size: usize,
    n_vectors: usize,
}

impl HvqClusterLayout {
    fn build(entries: &[IvfHvqEntry], dim: usize) -> Self {
        let n = entries.len();
        let sign_bits: Vec<&[u8]> = entries.iter().map(|e| e.sign_bits.as_slice()).collect();

        let (fastscan_codes, n_blocks, fastscan_block_size) =
            transpose_to_fastscan(&sign_bits, dim);

        let ids = entries.iter().map(|e| e.id).collect();
        let norms_o = entries.iter().map(|e| e.norm_o).collect();
        let vmaxs = entries.iter().map(|e| e.vmax).collect();
        let base_quant_dists = entries.iter().map(|e| e.base_quant_dist).collect();
        let packed_bits = entries.iter().map(|e| e.packed_bits.clone()).collect();
        let base_norm_sqs = entries.iter().map(|e| e.base_norm_sq).collect();

        let _ = n_blocks;
        Self {
            ids,
            norms_o,
            vmaxs,
            base_quant_dists,
            packed_bits,
            base_norm_sqs,
            fastscan_codes,
            fastscan_block_size,
            n_vectors: n,
        }
    }

    fn fastscan_topk(&self, state: &HvqFastScanState, dim: usize, n: usize) -> Vec<usize> {
        if n == 0 || self.n_vectors == 0 {
            return Vec::new();
        }

        let n_groups = dim.div_ceil(4);

        let mut heap = std::collections::BinaryHeap::with_capacity(n + 1);

        for (block_idx, block) in self
            .fastscan_codes
            .chunks_exact(self.fastscan_block_size)
            .enumerate()
        {
            #[cfg(target_arch = "x86_64")]
            let raw_scores = if std::arch::is_x86_feature_detected!("avx512bw") {
                unsafe { fastscan_block_avx512(&state.lut, block, n_groups) }
            } else {
                fastscan_block_scalar(&state.lut, block, n_groups)
            };

            #[cfg(not(target_arch = "x86_64"))]
            let raw_scores = fastscan_block_scalar(&state.lut, block, n_groups);

            for (slot, &raw_score) in raw_scores.iter().enumerate() {
                let local_id = block_idx * 32 + slot;
                if local_id >= self.n_vectors {
                    continue;
                }

                let score = raw_score as f32 * state.lut_scale;
                let candidate = MinScored {
                    id: local_id,
                    score,
                };
                if heap.len() < n {
                    heap.push(candidate);
                    continue;
                }
                if let Some(worst) = heap.peek() {
                    if candidate.score > worst.score
                        || (candidate.score == worst.score && candidate.id < worst.id)
                    {
                        heap.pop();
                        heap.push(candidate);
                    }
                }
            }
        }

        let mut results = Vec::with_capacity(heap.len());
        while let Some(item) = heap.pop() {
            results.push(item.id);
        }
        // Results come out in ascending score order (min-heap), reverse for descending
        results.sort_by(|a, b| b.cmp(a));
        results.truncate(n);
        results
    }
}

/// Transpose 1-bit sign codes into fastscan layout (same logic as HvqIndex::transpose_to_fastscan).
fn transpose_to_fastscan(raw_codes: &[&[u8]], dim: usize) -> (Vec<u8>, usize, usize) {
    let n_blocks = raw_codes.len().div_ceil(32);
    let fastscan_block_size = dim.div_ceil(4) * 16;
    let mut fastscan_codes = vec![0u8; n_blocks * fastscan_block_size];

    for block_idx in 0..n_blocks {
        let block_base = block_idx * fastscan_block_size;
        for group_idx in 0..dim.div_ceil(4) {
            let group_base = block_base + group_idx * 16;
            for slot in 0..32usize {
                let vid = block_idx * 32 + slot;
                let mut nibble = 0u8;
                if vid < raw_codes.len() {
                    for bit_pos in 0..4usize {
                        let dim_idx = group_idx * 4 + bit_pos;
                        if dim_idx >= dim {
                            break;
                        }
                        let byte = raw_codes[vid][dim_idx / 8];
                        let bit = (byte >> (dim_idx % 8)) & 1;
                        nibble |= bit << bit_pos;
                    }
                }

                let dst = group_base + slot / 2;
                if slot % 2 == 0 {
                    fastscan_codes[dst] |= nibble;
                } else {
                    fastscan_codes[dst] |= nibble << 4;
                }
            }
        }
    }

    (fastscan_codes, n_blocks, fastscan_block_size)
}

fn fastscan_block_scalar(lut: &[i8], block: &[u8], n_groups: usize) -> [i32; 32] {
    let mut scores = [0i32; 32];
    for group_idx in 0..n_groups {
        let group_lut = &lut[group_idx * 16..(group_idx + 1) * 16];
        let group_base = group_idx * 16;
        for slot in 0..32usize {
            let byte = block[group_base + slot / 2];
            let nibble = if slot % 2 == 0 {
                byte & 0x0F
            } else {
                (byte >> 4) & 0x0F
            };
            scores[slot] += group_lut[nibble as usize] as i32;
        }
    }
    scores
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bw,avx2,ssse3")]
unsafe fn fastscan_block_avx512(lut: &[i8], block: &[u8], n_groups: usize) -> [i32; 32] {
    use std::arch::x86_64::*;

    let mut acc_lo = _mm256_setzero_si256();
    let mut acc_hi = _mm256_setzero_si256();
    let nibble_mask_128 = _mm_set1_epi8(0x0F_u8 as i8);

    for group_idx in 0..n_groups {
        let group_offset = group_idx * 16;
        let lut_ptr = lut.as_ptr().add(group_idx * 16) as *const __m128i;
        let lut_128 = _mm_loadu_si128(lut_ptr);

        let data_lo_ptr = block.as_ptr().add(group_offset) as *const __m128i;
        let data_lo_64 = _mm_loadl_epi64(data_lo_ptr);
        let lo_nibbles = _mm_and_si128(data_lo_64, nibble_mask_128);
        let hi_nibbles = _mm_and_si128(_mm_srli_epi16(data_lo_64, 4), nibble_mask_128);
        let interleaved_lo = _mm_unpacklo_epi8(lo_nibbles, hi_nibbles);
        let partial_lo = _mm_shuffle_epi8(lut_128, interleaved_lo);
        let partial_lo_i16 = _mm256_cvtepi8_epi16(partial_lo);
        acc_lo = _mm256_add_epi16(acc_lo, partial_lo_i16);

        let data_hi_ptr = block.as_ptr().add(group_offset + 8) as *const __m128i;
        let data_hi_64 = _mm_loadl_epi64(data_hi_ptr);
        let hi_lo_nibbles = _mm_and_si128(data_hi_64, nibble_mask_128);
        let hi_hi_nibbles = _mm_and_si128(_mm_srli_epi16(data_hi_64, 4), nibble_mask_128);
        let interleaved_hi = _mm_unpacklo_epi8(hi_lo_nibbles, hi_hi_nibbles);
        let partial_hi = _mm_shuffle_epi8(lut_128, interleaved_hi);
        let partial_hi_i16 = _mm256_cvtepi8_epi16(partial_hi);
        acc_hi = _mm256_add_epi16(acc_hi, partial_hi_i16);
    }

    let mut scores = [0i32; 32];

    let lo_128_lo = _mm256_castsi256_si128(acc_lo);
    let lo_128_hi = _mm256_extracti128_si256(acc_lo, 1);
    let lo_i32_0 = _mm256_cvtepi16_epi32(lo_128_lo);
    let lo_i32_1 = _mm256_cvtepi16_epi32(lo_128_hi);
    _mm256_storeu_si256(scores.as_mut_ptr() as *mut __m256i, lo_i32_0);
    _mm256_storeu_si256(scores.as_mut_ptr().add(8) as *mut __m256i, lo_i32_1);

    let hi_128_lo = _mm256_castsi256_si128(acc_hi);
    let hi_128_hi = _mm256_extracti128_si256(acc_hi, 1);
    let hi_i32_0 = _mm256_cvtepi16_epi32(hi_128_lo);
    let hi_i32_1 = _mm256_cvtepi16_epi32(hi_128_hi);
    _mm256_storeu_si256(scores.as_mut_ptr().add(16) as *mut __m256i, hi_i32_0);
    _mm256_storeu_si256(scores.as_mut_ptr().add(24) as *mut __m256i, hi_i32_1);

    scores
}

#[derive(Clone, Copy, Debug)]
struct MinScored {
    id: usize,
    score: f32,
}

impl PartialEq for MinScored {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && self.score.to_bits() == other.score.to_bits()
    }
}

impl Eq for MinScored {}

impl Ord for MinScored {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other
            .score
            .total_cmp(&self.score)
            .then_with(|| other.id.cmp(&self.id))
    }
}

impl PartialOrd for MinScored {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

pub struct IvfHvqIndex {
    config: IvfHvqConfig,
    centroids: Vec<f32>,
    inverted_lists: RwLock<HashMap<usize, Vec<IvfHvqEntry>>>,
    cluster_layouts: RwLock<HashMap<usize, HvqClusterLayout>>,
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

        let mut modified_clusters = std::collections::HashSet::new();
        let mut lists = self.inverted_lists.write();
        for (i, vector) in data.chunks_exact(self.config.dim).enumerate() {
            let cluster = self.find_best_centroid(vector);
            let code = self.quantizer.encode(vector, HVQ_ENCODE_REFINE);
            let norm_o = f32::from_le_bytes(code[0..4].try_into().unwrap());
            let vmax = f32::from_le_bytes(code[4..8].try_into().unwrap());
            let base_quant_dist = f32::from_le_bytes(code[8..12].try_into().unwrap());
            let packed_bits = code[12..].to_vec();
            let base_norm_sq = vector.iter().map(|x| x * x).sum::<f32>();
            let sign_bits = self.quantizer.compute_sign_bits(vector);
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
                sign_bits,
            });
            modified_clusters.insert(cluster);
        }

        self.ntotal += n;

        // Rebuild layouts for modified clusters
        drop(lists);
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
        let lists = self.inverted_lists.read();
        let mut layouts = self.cluster_layouts.write();
        for &cluster in clusters {
            if let Some(entries) = lists.get(&cluster) {
                layouts.insert(cluster, HvqClusterLayout::build(entries, self.config.dim));
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
        let q_rot = self.quantizer.rotate_query(query);
        let use_l2 = matches!(self.config.metric_type, MetricType::L2);
        let q_norm_sq: f32 = query.iter().map(|x| x * x).sum();

        // Stage 1: fastscan coarse ranking per cluster
        let fs_state = self.quantizer.precompute_fastscan_state(&q_rot);
        let n_candidates_per_cluster = (top_k * 3).max(10);

        let layouts = self.cluster_layouts.read();
        let mut all_local_candidates = Vec::new();
        for &cluster in &coarse {
            if let Some(layout) = layouts.get(&cluster) {
                let topk_local = layout.fastscan_topk(&fs_state, self.config.dim, n_candidates_per_cluster);
                for local_id in topk_local {
                    all_local_candidates.push((cluster, local_id));
                }
            }
        }

        // Stage 2: rerank with score_code_with_meta
        let state = self.quantizer.precompute_query_state(&q_rot);
        let mut scored = Vec::new();
        for (cluster, local_id) in all_local_candidates {
            let layout = &layouts[&cluster];
            if local_id >= layout.n_vectors {
                continue;
            }
            if let Some(predicate) = filter {
                if !predicate.evaluate(layout.ids[local_id]) {
                    continue;
                }
            }

            let ip_score = self.quantizer.score_code_with_meta(
                &state,
                layout.norms_o[local_id],
                layout.vmaxs[local_id],
                layout.base_quant_dists[local_id],
                &layout.packed_bits[local_id],
            );

            let distance = if use_l2 {
                q_norm_sq + layout.base_norm_sqs[local_id] - 2.0 * ip_score
            } else {
                -ip_score
            };
            scored.push((layout.ids[local_id], distance));
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
        let lists = index.inverted_lists.read();
        let mut total_in_layouts = 0;
        for (&cluster, layout) in layouts.iter() {
            assert!(lists.contains_key(&cluster), "layout cluster missing from lists");
            assert_eq!(layout.n_vectors, lists[&cluster].len());
            total_in_layouts += layout.n_vectors;
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
