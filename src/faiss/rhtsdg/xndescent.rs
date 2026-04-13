use crate::api::MetricType;
use crate::faiss::rhtsdg::neighbor::{Neighbor, NeighborStatus, Neighborhood};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

#[derive(Debug, Clone)]
pub struct XNDescentConfig {
    pub k: usize,
    pub sample_count: usize,
    pub iter_count: usize,
    pub reverse_count: usize,
    pub use_shortcut: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct XNDescentTrace {
    pub iterations: usize,
}

pub struct XNDescentBuilder {
    dim: usize,
    vectors: Vec<f32>,
    metric: MetricType,
    config: XNDescentConfig,
    graph: Vec<Neighborhood>,
}

impl XNDescentBuilder {
    pub fn new(dim: usize, vectors: Vec<f32>, metric: MetricType, config: XNDescentConfig) -> Self {
        assert!(dim > 0, "dim must be positive");
        assert_eq!(
            vectors.len() % dim,
            0,
            "vector buffer length must be divisible by dim"
        );

        let num_points = vectors.len() / dim;
        let graph = (0..num_points)
            .map(|_| Neighborhood::new(config.k.max(1)))
            .collect();

        Self {
            dim,
            vectors,
            metric,
            config,
            graph,
        }
    }

    pub fn build(
        dim: usize,
        vectors: Vec<f32>,
        metric: MetricType,
        config: XNDescentConfig,
    ) -> Vec<Vec<u32>> {
        let builder = Self::new(dim, vectors, metric, config);
        builder.run()
    }

    pub fn new_for_tests(dim: usize, vectors: Vec<f32>, config: XNDescentConfig) -> Self {
        Self::new(dim, vectors, MetricType::L2, config)
    }

    pub fn build_for_tests(
        dim: usize,
        vectors: Vec<f32>,
        config: XNDescentConfig,
    ) -> Vec<Vec<u32>> {
        Self::build(dim, vectors, MetricType::L2, config)
    }

    pub fn seed_neighbors_for_test(&self, node: usize, neighbors: &[Neighbor]) {
        self.graph[node].seed_for_test(neighbors);
    }

    pub fn local_join_once_for_test(&mut self) -> usize {
        self.update_sample_neighbors();
        self.local_join()
    }

    pub fn run(&self) -> Vec<Vec<u32>> {
        self.run_with_trace().0
    }

    pub fn run_with_trace(&self) -> (Vec<Vec<u32>>, XNDescentTrace) {
        let mut trace = XNDescentTrace::default();
        self.init_ring_graph();

        for _ in 0..self.config.iter_count.max(1) {
            trace.iterations += 1;
            self.update_sample_neighbors();
            let updates = self.local_join();
            self.promote_new_to_old();
            if updates == 0 {
                break;
            }
        }

        (self.extract_neighbors(), trace)
    }

    pub fn vectors(&self) -> &[f32] {
        &self.vectors
    }

    pub fn has_edge_for_test(&self, node: usize, neighbor: u32) -> bool {
        self.graph[node].contains(neighbor)
    }

    fn update_sample_neighbors(&self) {
        self.graph.par_iter().for_each(|neighborhood| {
            neighborhood.rebuild_samples(self.config.sample_count.max(1));
        });

        if self.config.reverse_count == 0 {
            self.graph.par_iter().for_each(|neighborhood| {
                neighborhood.set_reverse_samples(Vec::new(), Vec::new());
            });
            return;
        }

        let mut reverse_new = vec![Vec::new(); self.graph.len()];
        let mut reverse_old = vec![Vec::new(); self.graph.len()];

        for (node, neighborhood) in self.graph.iter().enumerate() {
            let (nn_new, nn_old) = neighborhood.sample_lists();
            for neighbor in nn_new {
                let dist = self.distance(node, neighbor as usize);
                push_reverse_candidate(&mut reverse_new[neighbor as usize], node as u32, dist);
            }
            for neighbor in nn_old {
                let dist = self.distance(node, neighbor as usize);
                push_reverse_candidate(&mut reverse_old[neighbor as usize], node as u32, dist);
            }
        }

        for (node, neighborhood) in self.graph.iter().enumerate() {
            reverse_new[node]
                .sort_by(|lhs, rhs| lhs.1.total_cmp(&rhs.1).then_with(|| lhs.0.cmp(&rhs.0)));
            reverse_new[node].truncate(self.config.reverse_count);
            reverse_old[node]
                .sort_by(|lhs, rhs| lhs.1.total_cmp(&rhs.1).then_with(|| lhs.0.cmp(&rhs.0)));
            reverse_old[node].truncate(self.config.reverse_count);
            neighborhood.set_reverse_samples(
                reverse_new[node].iter().map(|(id, _)| *id).collect(),
                reverse_old[node].iter().map(|(id, _)| *id).collect(),
            );
        }
    }

    fn init_ring_graph(&self) {
        let num_points = self.graph.len();
        if num_points <= 1 {
            return;
        }

        let seed_degree = self.config.k.min(num_points - 1).max(1);
        let candidate_budget = seed_degree.saturating_mul(8).max(32).min(num_points - 1);
        self.graph
            .par_iter()
            .enumerate()
            .for_each(|(node, neighborhood)| {
                let candidates = if candidate_budget >= num_points - 1 {
                    (0..num_points)
                        .filter(|&candidate| candidate != node)
                        .map(|candidate| candidate as u32)
                        .collect::<Vec<_>>()
                } else {
                    sample_candidate_ids(node, num_points, candidate_budget)
                };
                let mut ranked = candidates
                    .into_iter()
                    .map(|neighbor| {
                        let dist = self.distance(node, neighbor as usize);
                        (neighbor, dist)
                    })
                    .collect::<Vec<_>>();
                ranked.sort_by(|lhs, rhs| lhs.1.total_cmp(&rhs.1).then_with(|| lhs.0.cmp(&rhs.0)));

                for (neighbor, dist) in ranked.into_iter().take(seed_degree) {
                    neighborhood.insert(neighbor, dist, NeighborStatus::New);
                }
            });
    }

    fn local_join(&self) -> usize {
        (0..self.graph.len())
            .into_par_iter()
            .map(|node| {
                let mut updates = 0usize;
                let (nn_new, nn_old) = self.graph[node].join_candidate_lists();

                for i in 0..nn_new.len() {
                    let id_i = nn_new[i] as usize;

                    for j in (i + 1)..nn_new.len() {
                        let id_j = nn_new[j] as usize;
                        let dist = self.distance(id_i, id_j);

                        if self.graph[id_i].insert(id_j as u32, dist, NeighborStatus::New) {
                            updates += 1;
                        }
                        if self.graph[id_j].insert(id_i as u32, dist, NeighborStatus::New) {
                            updates += 1;
                        }
                    }

                    for &old_id in &nn_old {
                        let old_idx = old_id as usize;
                        let dist = self.distance(id_i, old_idx);

                        if self.graph[id_i].insert(old_id, dist, NeighborStatus::New) {
                            updates += 1;
                        }
                        if self.graph[old_idx].insert(id_i as u32, dist, NeighborStatus::New) {
                            updates += 1;
                        }
                    }
                }

                updates
            })
            .sum()
    }

    fn promote_new_to_old(&self) {
        self.graph.par_iter().for_each(|neighborhood| {
            neighborhood.promote_new_to_old();
        });
    }

    fn extract_neighbors(&self) -> Vec<Vec<u32>> {
        self.graph.iter().map(Neighborhood::snapshot_ids).collect()
    }

    fn distance(&self, lhs: usize, rhs: usize) -> f32 {
        let lhs_start = lhs * self.dim;
        let rhs_start = rhs * self.dim;
        let lhs_slice = &self.vectors[lhs_start..lhs_start + self.dim];
        let rhs_slice = &self.vectors[rhs_start..rhs_start + self.dim];
        metric_distance(self.metric, lhs_slice, rhs_slice)
    }
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
            .sum::<f32>(),
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

fn push_reverse_candidate(bucket: &mut Vec<(u32, f32)>, id: u32, distance: f32) {
    if let Some(existing) = bucket
        .iter_mut()
        .find(|(existing_id, _)| *existing_id == id)
    {
        if distance < existing.1 {
            existing.1 = distance;
        }
        return;
    }
    bucket.push((id, distance));
}

fn sample_candidate_ids(node: usize, num_points: usize, budget: usize) -> Vec<u32> {
    let mut rng = StdRng::seed_from_u64(0x9E37_79B9_7F4A_7C15u64 ^ node as u64);
    let mut candidates = Vec::with_capacity(budget);

    while candidates.len() < budget {
        let candidate = rng.gen_range(0..num_points);
        if candidate == node {
            continue;
        }
        let candidate = candidate as u32;
        if !candidates.contains(&candidate) {
            candidates.push(candidate);
        }
    }

    candidates
}
