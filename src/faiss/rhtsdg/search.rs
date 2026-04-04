use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};
use std::fs::File;
use std::io::{Cursor, Read, Seek, Write};
use std::path::Path;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::api::{
    IndexConfig, IndexType, KnowhereError, MetricType, Result, SearchRequest,
    SearchResult as ApiSearchResult,
};
use crate::bitset::BitsetView;
use crate::codec::IndexCodec;
use crate::dataset::Dataset;
use crate::faiss::rhtsdg::config::RhtsdgConfig;
use crate::faiss::rhtsdg::tsdg::{stage1_prune_neighbors, stage2_filter_neighbors, DistanceMatrix};
use crate::faiss::rhtsdg::xndescent::{XNDescentBuilder, XNDescentConfig};
use crate::index::{
    AnnIterator, Index as IndexTrait, IndexError, SearchResult as IndexSearchResult,
};
use crate::search::{with_visited, VisitedList};

#[derive(Debug)]
pub struct RhtsdgIndex {
    config: IndexConfig,
    build_config: RhtsdgConfig,
    vectors: Vec<f32>,
    ids: Vec<i64>,
    layer_graphs: Vec<Vec<Vec<u32>>>,
    entry_point: u32,
    trained: bool,
    build_kind: &'static str,
}

#[derive(Debug, Clone)]
pub struct ScreenSummary {
    pub recall_at_10: f32,
    pub query_count: usize,
    pub build_kind: &'static str,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StoredRhtsdgSnapshot {
    trained: bool,
    vectors: Vec<f32>,
    ids: Vec<i64>,
    layer_graphs: Vec<Vec<Vec<u32>>>,
    entry_point: u32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct FrontierEntry {
    id: u32,
    dist: f32,
}

impl Eq for FrontierEntry {}

impl Ord for FrontierEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .dist
            .total_cmp(&self.dist)
            .then_with(|| other.id.cmp(&self.id))
    }
}

impl PartialOrd for FrontierEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct ResultEntry {
    id: u32,
    dist: f32,
}

impl Eq for ResultEntry {}

impl Ord for ResultEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist
            .total_cmp(&other.dist)
            .then_with(|| self.id.cmp(&other.id))
    }
}

impl PartialOrd for ResultEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub struct RhtsdgAnnIterator {
    results: Vec<(i64, f32)>,
    pos: usize,
}

impl RhtsdgAnnIterator {
    fn new(results: Vec<(i64, f32)>) -> Self {
        Self { results, pos: 0 }
    }
}

impl AnnIterator for RhtsdgAnnIterator {
    fn next(&mut self) -> Option<(i64, f32)> {
        if self.pos >= self.results.len() {
            return None;
        }
        let result = self.results[self.pos];
        self.pos += 1;
        Some(result)
    }

    fn buffer_size(&self) -> usize {
        self.results.len().saturating_sub(self.pos)
    }
}

impl RhtsdgIndex {
    pub fn new(config: &IndexConfig) -> Result<Self> {
        if config.index_type != IndexType::Rhtsdg {
            return Err(KnowhereError::InvalidArg(format!(
                "rhtsdg index requires IndexType::Rhtsdg, got {:?}",
                config.index_type
            )));
        }
        if config.dim == 0 {
            return Err(KnowhereError::InvalidArg(
                "dimension must be greater than zero".to_string(),
            ));
        }
        config.validate().map_err(KnowhereError::InvalidArg)?;

        Ok(Self {
            config: config.clone(),
            build_config: RhtsdgConfig::from_index_config(config),
            vectors: Vec::new(),
            ids: Vec::new(),
            layer_graphs: vec![Vec::new()],
            entry_point: 0,
            trained: false,
            build_kind: "runtime_empty",
        })
    }

    pub fn config(&self) -> &IndexConfig {
        &self.config
    }

    pub fn train(&mut self, vectors: &[f32]) -> Result<()> {
        self.validate_buffer(vectors, "training")?;
        self.trained = true;
        Ok(())
    }

    pub fn add(&mut self, vectors: &[f32], ids: Option<&[i64]>) -> Result<usize> {
        self.validate_buffer(vectors, "add")?;
        let vector_count = vectors.len() / self.config.dim;

        if let Some(ids) = ids {
            if ids.len() != vector_count {
                return Err(KnowhereError::InvalidArg(format!(
                    "ids count {} does not match vector count {}",
                    ids.len(),
                    vector_count
                )));
            }
        }

        if vector_count == 0 {
            self.trained = true;
            return Ok(0);
        }

        self.ensure_unique_ids(ids)?;

        let starting_count = self.ids.len();
        self.vectors.extend_from_slice(vectors);
        if let Some(ids) = ids {
            self.ids.extend_from_slice(ids);
        } else {
            self.ids
                .extend((starting_count..starting_count + vector_count).map(|id| id as i64));
        }

        self.rebuild_graph();
        self.trained = true;
        self.build_kind = "xndescent_tsdg_runtime";
        Ok(vector_count)
    }

    pub fn search(&self, query: &[f32], req: &SearchRequest) -> Result<ApiSearchResult> {
        self.search_internal(query, req, None)
    }

    pub fn search_with_bitset(
        &self,
        query: &[f32],
        req: &SearchRequest,
        bitset: &BitsetView,
    ) -> Result<ApiSearchResult> {
        self.search_internal(query, req, Some(bitset))
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let mut file = File::create(path)?;
        self.write_to(&mut file)
    }

    pub fn load(&mut self, path: &Path) -> Result<()> {
        let mut file = File::open(path)?;
        self.read_from(&mut file)
    }

    pub fn new_for_tests(
        dim: usize,
        vectors: Vec<f32>,
        layer0_graph: Vec<Vec<u32>>,
        entry_point: u32,
    ) -> Self {
        assert!(dim > 0, "dim must be positive");
        assert_eq!(
            vectors.len() % dim,
            0,
            "vector buffer length must be divisible by dim"
        );
        assert_eq!(
            vectors.len() / dim,
            layer0_graph.len(),
            "graph node count must match vector count"
        );

        let config = IndexConfig::new(IndexType::Rhtsdg, MetricType::L2, dim);
        let node_count = vectors.len() / dim;
        Self {
            build_config: RhtsdgConfig::from_index_config(&config),
            config,
            ids: (0..node_count).map(|id| id as i64).collect(),
            vectors,
            layer_graphs: vec![layer0_graph],
            entry_point,
            trained: true,
            build_kind: "manual_fixture",
        }
    }

    pub fn build_for_tests(dim: usize, vectors: Vec<f32>) -> Self {
        assert!(dim > 0, "dim must be positive");
        assert_eq!(
            vectors.len() % dim,
            0,
            "vector buffer length must be divisible by dim"
        );

        let num_points = vectors.len() / dim;
        let knn_k = num_points.saturating_sub(1).min(16).max(1);
        let mut config = IndexConfig::new(IndexType::Rhtsdg, MetricType::L2, dim);
        config.params.rhtsdg_knn_k = Some(knn_k);
        config.params.rhtsdg_sample_count = Some(knn_k.min(8).max(1));
        config.params.rhtsdg_iter_count = Some(4);

        let layer0_graph = build_layer0_graph(
            dim,
            &vectors,
            MetricType::L2,
            &RhtsdgConfig::from_index_config(&config),
        );

        Self {
            build_config: RhtsdgConfig::from_index_config(&config),
            config,
            ids: (0..num_points).map(|id| id as i64).collect(),
            vectors,
            layer_graphs: vec![layer0_graph],
            entry_point: 0,
            trained: true,
            build_kind: "xndescent_tsdg_fixture",
        }
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn build_kind_for_test(&self) -> &'static str {
        self.build_kind
    }

    pub fn search_batch_for_test(&self, queries: &[f32], k: usize, ef: usize) -> Vec<Vec<u32>> {
        assert_eq!(
            queries.len() % self.config.dim,
            0,
            "query buffer length must be divisible by dim"
        );

        queries
            .chunks(self.config.dim)
            .map(|query| {
                self.search_positions(query, k, ef, None)
                    .into_iter()
                    .map(|(id, _)| self.ids[id as usize] as u32)
                    .collect()
            })
            .collect()
    }

    fn validate_buffer(&self, vectors: &[f32], operation: &str) -> Result<()> {
        if vectors.len() % self.config.dim != 0 {
            return Err(KnowhereError::InvalidArg(format!(
                "{operation} buffer length {} is not divisible by dim {}",
                vectors.len(),
                self.config.dim
            )));
        }
        Ok(())
    }

    fn ensure_unique_ids(&self, ids: Option<&[i64]>) -> Result<()> {
        let Some(ids) = ids else {
            return Ok(());
        };

        let mut seen = HashSet::with_capacity(ids.len() + self.ids.len());
        for &id in &self.ids {
            seen.insert(id);
        }
        for &id in ids {
            if !seen.insert(id) {
                return Err(KnowhereError::InvalidArg(format!(
                    "duplicate id {id} is not supported"
                )));
            }
        }
        Ok(())
    }

    fn rebuild_graph(&mut self) {
        if self.ids.is_empty() {
            self.layer_graphs = vec![Vec::new()];
            self.entry_point = 0;
            return;
        }

        let layer0_graph = build_layer0_graph(
            self.config.dim,
            &self.vectors,
            self.config.metric_type,
            &self.build_config,
        );
        self.layer_graphs = vec![layer0_graph];
        self.entry_point = 0;
    }

    fn search_internal(
        &self,
        query: &[f32],
        req: &SearchRequest,
        bitset: Option<&BitsetView>,
    ) -> Result<ApiSearchResult> {
        if !self.trained {
            return Err(KnowhereError::IndexNotTrained(
                "rhtsdg index must be trained before search".to_string(),
            ));
        }
        if self.ids.is_empty() {
            return Err(KnowhereError::InvalidArg("index is empty".to_string()));
        }
        self.validate_buffer(query, "query")?;

        let start = Instant::now();
        let top_k = req.top_k.max(1);
        let ef = self.search_ef(top_k);
        let mut ids = Vec::new();
        let mut distances = Vec::new();

        for query_vector in query.chunks(self.config.dim) {
            let mut hits = self.search_positions(query_vector, top_k, ef, bitset);
            if let Some(filter) = req.filter.as_ref() {
                hits.retain(|(internal_id, _)| filter.evaluate(self.ids[*internal_id as usize]));
                hits.truncate(top_k);
            }
            for (internal_id, dist) in hits {
                ids.push(self.ids[internal_id as usize]);
                distances.push(dist);
            }
        }

        Ok(ApiSearchResult::new(
            ids,
            distances,
            start.elapsed().as_secs_f64() * 1000.0,
        ))
    }

    fn search_ef(&self, top_k: usize) -> usize {
        self.build_config.knn_k.max(top_k).max(1)
    }

    fn search_positions(
        &self,
        query: &[f32],
        top_k: usize,
        ef: usize,
        bitset: Option<&BitsetView>,
    ) -> Vec<(u32, f32)> {
        if self.layer_graphs.is_empty() || self.ids.is_empty() || top_k == 0 {
            return Vec::new();
        }

        let ef = ef.max(top_k).max(1);
        let mut current = self.entry_point;
        for layer in (1..self.layer_graphs.len()).rev() {
            if let Some((best, _)) = self
                .search_layer(query, &[current], 1, layer, bitset)
                .into_iter()
                .next()
            {
                current = best;
            }
        }

        self.search_layer(query, &[current], ef, 0, bitset)
            .into_iter()
            .take(top_k)
            .collect()
    }

    fn search_layer(
        &self,
        query: &[f32],
        entry_points: &[u32],
        ef: usize,
        layer: usize,
        bitset: Option<&BitsetView>,
    ) -> Vec<(u32, f32)> {
        with_visited(self.len(), |visited| {
            self.search_layer_inner(query, entry_points, ef, layer, bitset, visited)
        })
    }

    fn search_layer_inner(
        &self,
        query: &[f32],
        entry_points: &[u32],
        ef: usize,
        layer: usize,
        bitset: Option<&BitsetView>,
        visited: &mut VisitedList,
    ) -> Vec<(u32, f32)> {
        let mut frontier = BinaryHeap::new();
        let mut results = BinaryHeap::new();

        for &entry in entry_points {
            if visited.is_visited(entry) {
                continue;
            }
            visited.mark(entry);
            let dist = self.distance_to_node(query, entry as usize);
            frontier.push(FrontierEntry { id: entry, dist });
            if accepts_result(bitset, entry as usize) {
                push_result(&mut results, ResultEntry { id: entry, dist }, ef);
            }
        }

        while let Some(candidate) = frontier.pop() {
            if results.len() >= ef && candidate.dist > worst_result_dist(&results) {
                break;
            }

            for &neighbor in &self.layer_graphs[layer][candidate.id as usize] {
                if visited.is_visited(neighbor) {
                    continue;
                }
                visited.mark(neighbor);

                let dist = self.distance_to_node(query, neighbor as usize);
                frontier.push(FrontierEntry { id: neighbor, dist });
                if accepts_result(bitset, neighbor as usize)
                    && (results.len() < ef || dist < worst_result_dist(&results))
                {
                    push_result(&mut results, ResultEntry { id: neighbor, dist }, ef);
                }
            }
        }

        let mut ordered = results.into_vec();
        ordered.sort_by(|lhs, rhs| {
            lhs.dist
                .total_cmp(&rhs.dist)
                .then_with(|| lhs.id.cmp(&rhs.id))
        });
        ordered
            .into_iter()
            .map(|entry| (entry.id, entry.dist))
            .collect()
    }

    fn distance_to_node(&self, query: &[f32], node: usize) -> f32 {
        let start = node * self.config.dim;
        let vector = &self.vectors[start..start + self.config.dim];
        metric_distance(self.config.metric_type, query, vector)
    }

    fn position_of_id(&self, id: i64) -> Option<usize> {
        self.ids.iter().position(|candidate| *candidate == id)
    }

    fn snapshot(&self) -> StoredRhtsdgSnapshot {
        StoredRhtsdgSnapshot {
            trained: self.trained,
            vectors: self.vectors.clone(),
            ids: self.ids.clone(),
            layer_graphs: self.layer_graphs.clone(),
            entry_point: self.entry_point,
        }
    }

    fn apply_snapshot(&mut self, snapshot: StoredRhtsdgSnapshot) {
        self.build_config = RhtsdgConfig::from_index_config(&self.config);
        self.trained = snapshot.trained;
        self.vectors = snapshot.vectors;
        self.ids = snapshot.ids;
        self.layer_graphs = snapshot.layer_graphs;
        if self.layer_graphs.is_empty() {
            self.layer_graphs = vec![Vec::new()];
        }
        self.entry_point = snapshot.entry_point;
        self.build_kind = "loaded";
    }

    fn write_to<W: Write + Seek>(&self, writer: &mut W) -> Result<()> {
        IndexCodec::write_config(&self.config, writer)?;
        let payload = bincode::serialize(&self.snapshot())
            .map_err(|e| KnowhereError::Codec(format!("serialize rhtsdg snapshot: {e}")))?;
        writer.write_all(&(payload.len() as u64).to_le_bytes())?;
        writer.write_all(&payload)?;
        Ok(())
    }

    fn read_from<R: Read + Seek>(&mut self, reader: &mut R) -> Result<()> {
        let stored_config = IndexCodec::read_config(reader)?;
        self.validate_loaded_header(&stored_config)?;

        let mut payload_len = [0u8; 8];
        reader.read_exact(&mut payload_len)?;
        let payload_len = u64::from_le_bytes(payload_len) as usize;
        let mut payload = vec![0u8; payload_len];
        reader.read_exact(&mut payload)?;

        let snapshot: StoredRhtsdgSnapshot = bincode::deserialize(&payload)
            .map_err(|e| KnowhereError::Codec(format!("deserialize rhtsdg snapshot: {e}")))?;
        self.apply_snapshot(snapshot);
        Ok(())
    }

    fn validate_loaded_header(&self, stored: &IndexConfig) -> Result<()> {
        if stored.index_type != IndexType::Rhtsdg {
            return Err(KnowhereError::Codec(format!(
                "expected rhtsdg payload, got {:?}",
                stored.index_type
            )));
        }
        if stored.metric_type != self.config.metric_type {
            return Err(KnowhereError::Codec(format!(
                "metric mismatch: file {:?}, target {:?}",
                stored.metric_type, self.config.metric_type
            )));
        }
        if stored.dim != self.config.dim {
            return Err(KnowhereError::Codec(format!(
                "dimension mismatch: file {}, target {}",
                stored.dim, self.config.dim
            )));
        }
        Ok(())
    }
}

impl IndexTrait for RhtsdgIndex {
    fn index_type(&self) -> &str {
        "RHTSDG"
    }

    fn dim(&self) -> usize {
        self.config.dim
    }

    fn count(&self) -> usize {
        self.ids.len()
    }

    fn is_trained(&self) -> bool {
        self.trained
    }

    fn train(&mut self, dataset: &Dataset) -> std::result::Result<(), IndexError> {
        RhtsdgIndex::train(self, dataset.vectors()).map_err(api_error_to_index_error)
    }

    fn add(&mut self, dataset: &Dataset) -> std::result::Result<usize, IndexError> {
        RhtsdgIndex::add(self, dataset.vectors(), dataset.ids()).map_err(api_error_to_index_error)
    }

    fn search(
        &self,
        query: &Dataset,
        top_k: usize,
    ) -> std::result::Result<IndexSearchResult, IndexError> {
        let req = SearchRequest {
            top_k,
            ..Default::default()
        };
        let api_result =
            RhtsdgIndex::search(self, query.vectors(), &req).map_err(api_error_to_index_error)?;
        Ok(IndexSearchResult::new(
            api_result.ids,
            api_result.distances,
            api_result.elapsed_ms,
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
            ..Default::default()
        };
        let api_result = RhtsdgIndex::search_with_bitset(self, query.vectors(), &req, bitset)
            .map_err(api_error_to_index_error)?;
        Ok(IndexSearchResult::new(
            api_result.ids,
            api_result.distances,
            api_result.elapsed_ms,
        ))
    }

    fn get_vector_by_ids(&self, ids: &[i64]) -> std::result::Result<Vec<f32>, IndexError> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        if self.vectors.is_empty() || self.ids.is_empty() {
            return Err(IndexError::Empty);
        }

        let mut vectors = Vec::with_capacity(ids.len() * self.config.dim);
        for &id in ids {
            let position = self.position_of_id(id).ok_or_else(|| {
                IndexError::Unsupported(format!("ID {id} not found in rhtsdg index"))
            })?;
            let start = position * self.config.dim;
            let end = start + self.config.dim;
            vectors.extend_from_slice(&self.vectors[start..end]);
        }
        Ok(vectors)
    }

    fn create_ann_iterator(
        &self,
        query: &Dataset,
        bitset: Option<&BitsetView>,
    ) -> std::result::Result<Box<dyn AnnIterator>, IndexError> {
        let req = SearchRequest {
            top_k: self.count().max(1),
            ..Default::default()
        };
        let api_result = if let Some(bitset) = bitset {
            RhtsdgIndex::search_with_bitset(self, query.vectors(), &req, bitset)
        } else {
            RhtsdgIndex::search(self, query.vectors(), &req)
        }
        .map_err(api_error_to_index_error)?;

        let results = api_result
            .ids
            .into_iter()
            .zip(api_result.distances)
            .collect();
        Ok(Box::new(RhtsdgAnnIterator::new(results)))
    }

    fn serialize_to_memory(&self) -> std::result::Result<Vec<u8>, IndexError> {
        let mut buffer = Cursor::new(Vec::new());
        self.write_to(&mut buffer)
            .map_err(api_error_to_index_error)?;
        Ok(buffer.into_inner())
    }

    fn deserialize_from_memory(&mut self, data: &[u8]) -> std::result::Result<(), IndexError> {
        let mut cursor = Cursor::new(data);
        self.read_from(&mut cursor)
            .map_err(api_error_to_index_error)
    }

    fn save(&self, path: &str) -> std::result::Result<(), IndexError> {
        RhtsdgIndex::save(self, Path::new(path)).map_err(api_error_to_index_error)
    }

    fn load(&mut self, path: &str) -> std::result::Result<(), IndexError> {
        RhtsdgIndex::load(self, Path::new(path)).map_err(api_error_to_index_error)
    }

    fn has_raw_data(&self) -> bool {
        true
    }
}

pub fn run_local_screen_fixture_for_test() -> ScreenSummary {
    let dim = 2;
    let width = 16usize;
    let mut vectors = Vec::with_capacity(width * width * dim);
    for y in 0..width {
        for x in 0..width {
            vectors.push(x as f32);
            vectors.push(y as f32);
        }
    }

    let index = RhtsdgIndex::build_for_tests(dim, vectors.clone());
    let queries = vectors.clone();
    let expected = brute_force_topk(&vectors, dim, &queries, 10);
    let actual = index.search_batch_for_test(&queries, 10, 32);

    let mut total_hits = 0usize;
    let mut total_expected = 0usize;
    for (truth, got) in expected.iter().zip(actual.iter()) {
        total_expected += truth.len();
        total_hits += got.iter().filter(|id| truth.contains(id)).count();
    }

    ScreenSummary {
        recall_at_10: total_hits as f32 / total_expected as f32,
        query_count: queries.len() / dim,
        build_kind: index.build_kind,
    }
}

fn api_error_to_index_error(error: KnowhereError) -> IndexError {
    match error {
        KnowhereError::IndexNotTrained(_) => IndexError::NotTrained,
        KnowhereError::InvalidArg(msg) if msg.contains("empty") => IndexError::Empty,
        other => IndexError::Unsupported(other.to_string()),
    }
}

fn accepts_result(bitset: Option<&BitsetView>, internal_id: usize) -> bool {
    bitset.map(|bits| !bits.test(internal_id)).unwrap_or(true)
}

fn push_result(results: &mut BinaryHeap<ResultEntry>, entry: ResultEntry, ef: usize) {
    results.push(entry);
    if results.len() > ef {
        results.pop();
    }
}

fn worst_result_dist(results: &BinaryHeap<ResultEntry>) -> f32 {
    results
        .peek()
        .map(|entry| entry.dist)
        .unwrap_or(f32::INFINITY)
}

fn metric_distance(metric: MetricType, lhs: &[f32], rhs: &[f32]) -> f32 {
    match metric {
        MetricType::L2 => lhs
            .iter()
            .zip(rhs.iter())
            .map(|(a, b)| {
                let delta = a - b;
                delta * delta
            })
            .sum::<f32>()
            .sqrt(),
        MetricType::Ip => -lhs.iter().zip(rhs.iter()).map(|(a, b)| a * b).sum::<f32>(),
        MetricType::Cosine => {
            let dot = lhs.iter().zip(rhs.iter()).map(|(a, b)| a * b).sum::<f32>();
            let lhs_norm = lhs.iter().map(|v| v * v).sum::<f32>().sqrt();
            let rhs_norm = rhs.iter().map(|v| v * v).sum::<f32>().sqrt();
            if lhs_norm == 0.0 || rhs_norm == 0.0 {
                1.0
            } else {
                1.0 - dot / (lhs_norm * rhs_norm)
            }
        }
        MetricType::Hamming => unreachable!("rhtsdg does not support hamming"),
    }
}

fn brute_force_topk(vectors: &[f32], dim: usize, queries: &[f32], k: usize) -> Vec<Vec<u32>> {
    queries
        .chunks(dim)
        .map(|query| {
            let mut distances: Vec<(u32, f32)> = vectors
                .chunks(dim)
                .enumerate()
                .map(|(idx, vector)| (idx as u32, metric_distance(MetricType::L2, query, vector)))
                .collect();
            distances.sort_by(|lhs, rhs| lhs.1.total_cmp(&rhs.1).then_with(|| lhs.0.cmp(&rhs.0)));
            distances.into_iter().take(k).map(|(id, _)| id).collect()
        })
        .collect()
}

fn build_layer0_graph(
    dim: usize,
    vectors: &[f32],
    metric: MetricType,
    build_config: &RhtsdgConfig,
) -> Vec<Vec<u32>> {
    let num_points = vectors.len() / dim;
    if num_points == 0 {
        return Vec::new();
    }
    if num_points == 1 {
        return vec![Vec::new()];
    }

    let knn_k = build_config.knn_k.min(num_points - 1).max(1);
    let nndescent = XNDescentConfig {
        k: build_config.nndescent.k.min(num_points - 1).max(1),
        sample_count: build_config.nndescent.sample_count.min(knn_k).max(1),
        iter_count: build_config.nndescent.iter_count.max(1),
        reverse_count: build_config.nndescent.reverse_count.max(1),
        use_shortcut: build_config.nndescent.use_shortcut,
    };
    let base_graph = XNDescentBuilder::build(dim, vectors.to_vec(), metric, nndescent);
    diversify_graph(
        dim,
        vectors,
        metric,
        &base_graph,
        build_config.alpha,
        build_config.occ_threshold,
        knn_k,
    )
}

fn diversify_graph(
    dim: usize,
    vectors: &[f32],
    metric: MetricType,
    base_graph: &[Vec<u32>],
    alpha: f32,
    occ_threshold: u32,
    max_k: usize,
) -> Vec<Vec<u32>> {
    let distance = DistanceMatrix::from_points_with_metric(dim, vectors, metric);
    let reverse = collect_reverse_edges(base_graph);
    let mut graph = Vec::with_capacity(base_graph.len());

    for center in 0..base_graph.len() {
        let (alive, occs) = stage1_prune_neighbors(center, &base_graph[center], &distance, alpha);
        let kept = stage2_filter_neighbors(
            center,
            &alive,
            &occs,
            &reverse[center],
            &distance,
            occ_threshold,
            max_k,
        );
        graph.push(kept);
    }

    graph
}

fn collect_reverse_edges(base_graph: &[Vec<u32>]) -> Vec<Vec<u32>> {
    let mut reverse = vec![Vec::new(); base_graph.len()];
    for (node, neighbors) in base_graph.iter().enumerate() {
        for &neighbor in neighbors {
            let bucket = &mut reverse[neighbor as usize];
            if !bucket.contains(&(node as u32)) {
                bucket.push(node as u32);
            }
        }
    }
    reverse
}
