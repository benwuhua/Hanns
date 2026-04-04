use crate::api::MetricType;
use crate::faiss::rhtsdg::neighbor::{Neighbor, NeighborStatus, Neighborhood};

#[derive(Debug, Clone)]
pub struct XNDescentConfig {
    pub k: usize,
    pub sample_count: usize,
    pub iter_count: usize,
    pub reverse_count: usize,
    pub use_shortcut: bool,
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
        builder.init_ring_graph();

        for _ in 0..builder.config.iter_count.max(1) {
            builder.update_sample_neighbors();
            let updates = builder.local_join();
            builder.promote_new_to_old();
            if updates == 0 {
                break;
            }
        }

        builder.extract_neighbors()
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

    pub fn has_edge_for_test(&self, node: usize, neighbor: u32) -> bool {
        self.graph[node].contains(neighbor)
    }

    fn update_sample_neighbors(&self) {
        for neighborhood in &self.graph {
            neighborhood.rebuild_samples(self.config.sample_count.max(1));
        }
    }

    fn init_ring_graph(&self) {
        let num_points = self.graph.len();
        if num_points <= 1 {
            return;
        }

        let seed_degree = self.config.k.min(num_points - 1).max(1);
        for node in 0..num_points {
            for offset in 1..=seed_degree {
                let neighbor = ((node + offset) % num_points) as u32;
                let dist = self.distance(node, neighbor as usize);
                self.graph[node].insert(neighbor, dist, NeighborStatus::New);
            }
        }
    }

    fn local_join(&self) -> usize {
        let mut updates = 0usize;

        for node in 0..self.graph.len() {
            let (nn_new, nn_old) = self.graph[node].sample_lists();

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
        }

        updates
    }

    fn promote_new_to_old(&self) {
        for neighborhood in &self.graph {
            neighborhood.promote_new_to_old();
        }
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
