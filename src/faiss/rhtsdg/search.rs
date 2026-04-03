use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::search::{with_visited, VisitedList};

pub struct RhtsdgIndex {
    dim: usize,
    vectors: Vec<f32>,
    layer_graphs: Vec<Vec<Vec<u32>>>,
    entry_point: u32,
}

#[derive(Debug, Clone)]
pub struct ScreenSummary {
    pub recall_at_10: f32,
    pub query_count: usize,
    pub build_kind: &'static str,
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

impl RhtsdgIndex {
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

        Self {
            dim,
            vectors,
            layer_graphs: vec![layer0_graph],
            entry_point,
        }
    }

    pub fn len(&self) -> usize {
        self.vectors.len() / self.dim
    }

    pub fn search_batch_for_test(&self, queries: &[f32], k: usize, ef: usize) -> Vec<Vec<u32>> {
        assert_eq!(
            queries.len() % self.dim,
            0,
            "query buffer length must be divisible by dim"
        );

        queries
            .chunks(self.dim)
            .map(|query| {
                self.search_single_layer_for_test(query, k, ef)
                    .into_iter()
                    .map(|(id, _)| id)
                    .collect()
            })
            .collect()
    }

    fn search_single_layer_for_test(&self, query: &[f32], k: usize, ef: usize) -> Vec<(u32, f32)> {
        let ef = ef.max(k).max(1);
        let results = self.search_layer(query, &[self.entry_point], ef, 0);
        results.into_iter().take(k).collect()
    }

    fn search_layer(
        &self,
        query: &[f32],
        entry_points: &[u32],
        ef: usize,
        layer: usize,
    ) -> Vec<(u32, f32)> {
        with_visited(self.len(), |visited| self.search_layer_inner(query, entry_points, ef, layer, visited))
    }

    fn search_layer_inner(
        &self,
        query: &[f32],
        entry_points: &[u32],
        ef: usize,
        layer: usize,
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
            push_result(&mut results, ResultEntry { id: entry, dist }, ef);
        }

        while let Some(candidate) = frontier.pop() {
            let Some(worst) = results.peek().copied() else {
                continue;
            };

            if results.len() >= ef && candidate.dist > worst.dist {
                break;
            }

            for &neighbor in &self.layer_graphs[layer][candidate.id as usize] {
                if visited.is_visited(neighbor) {
                    continue;
                }
                visited.mark(neighbor);

                let dist = self.distance_to_node(query, neighbor as usize);
                if results.len() < ef || dist < worst_result_dist(&results) {
                    frontier.push(FrontierEntry { id: neighbor, dist });
                    push_result(&mut results, ResultEntry { id: neighbor, dist }, ef);
                }
            }
        }

        let mut ordered = results.into_vec();
        ordered.sort_by(|lhs, rhs| lhs.dist.total_cmp(&rhs.dist).then_with(|| lhs.id.cmp(&rhs.id)));
        ordered.into_iter().map(|entry| (entry.id, entry.dist)).collect()
    }

    fn distance_to_node(&self, query: &[f32], node: usize) -> f32 {
        let start = node * self.dim;
        let vector = &self.vectors[start..start + self.dim];
        query
            .iter()
            .zip(vector.iter())
            .map(|(lhs, rhs)| {
                let delta = lhs - rhs;
                delta * delta
            })
            .sum::<f32>()
            .sqrt()
    }
}

fn push_result(results: &mut BinaryHeap<ResultEntry>, entry: ResultEntry, ef: usize) {
    results.push(entry);
    if results.len() > ef {
        results.pop();
    }
}

fn worst_result_dist(results: &BinaryHeap<ResultEntry>) -> f32 {
    results.peek().map(|entry| entry.dist).unwrap_or(f32::INFINITY)
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

    let layer0 = exact_knn_graph(&vectors, dim, width * width - 1);
    let index = RhtsdgIndex::new_for_tests(dim, vectors.clone(), layer0, 0);
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
        build_kind: "exact_knn_fixture",
    }
}

fn exact_knn_graph(vectors: &[f32], dim: usize, degree: usize) -> Vec<Vec<u32>> {
    let num_points = vectors.len() / dim;
    let mut graph = Vec::with_capacity(num_points);

    for center in 0..num_points {
        let mut neighbors: Vec<(u32, f32)> = (0..num_points)
            .filter(|&other| other != center)
            .map(|other| {
                (
                    other as u32,
                    l2_distance(
                        &vectors[center * dim..(center + 1) * dim],
                        &vectors[other * dim..(other + 1) * dim],
                    ),
                )
            })
            .collect();
        neighbors.sort_by(|lhs, rhs| lhs.1.total_cmp(&rhs.1).then_with(|| lhs.0.cmp(&rhs.0)));
        graph.push(
            neighbors
                .into_iter()
                .take(degree)
                .map(|(id, _)| id)
                .collect(),
        );
    }

    graph
}

fn brute_force_topk(vectors: &[f32], dim: usize, queries: &[f32], k: usize) -> Vec<Vec<u32>> {
    queries
        .chunks(dim)
        .map(|query| {
            let mut distances: Vec<(u32, f32)> = vectors
                .chunks(dim)
                .enumerate()
                .map(|(idx, vector)| (idx as u32, l2_distance(query, vector)))
                .collect();
            distances.sort_by(|lhs, rhs| lhs.1.total_cmp(&rhs.1).then_with(|| lhs.0.cmp(&rhs.0)));
            distances.into_iter().take(k).map(|(id, _)| id).collect()
        })
        .collect()
}

fn l2_distance(lhs: &[f32], rhs: &[f32]) -> f32 {
    lhs.iter()
        .zip(rhs.iter())
        .map(|(a, b)| {
            let delta = a - b;
            delta * delta
        })
        .sum::<f32>()
        .sqrt()
}
