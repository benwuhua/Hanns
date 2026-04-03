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
    config: XNDescentConfig,
    graph: Vec<Neighborhood>,
}

impl XNDescentBuilder {
    pub fn new_for_tests(dim: usize, vectors: Vec<f32>, config: XNDescentConfig) -> Self {
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
            config,
            graph,
        }
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

    fn distance(&self, lhs: usize, rhs: usize) -> f32 {
        let lhs_start = lhs * self.dim;
        let rhs_start = rhs * self.dim;
        let lhs_slice = &self.vectors[lhs_start..lhs_start + self.dim];
        let rhs_slice = &self.vectors[rhs_start..rhs_start + self.dim];
        lhs_slice
            .iter()
            .zip(rhs_slice.iter())
            .map(|(a, b)| {
                let delta = a - b;
                delta * delta
            })
            .sum::<f32>()
            .sqrt()
    }
}
