use parking_lot::Mutex;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Neighbor {
    pub id: u32,
    pub distance: f32,
    pub status: NeighborStatus,
}

impl Neighbor {
    pub fn new(id: u32, distance: f32, status: NeighborStatus) -> Self {
        Self {
            id,
            distance,
            status,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NeighborStatus {
    New = 0,
    Old = 1,
}

#[derive(Debug, Default)]
struct NeighborhoodState {
    pool: Vec<Neighbor>,
    nn_new: Vec<u32>,
    nn_old: Vec<u32>,
    rnn_new: Vec<u32>,
    rnn_old: Vec<u32>,
}

#[derive(Debug)]
pub struct Neighborhood {
    capacity: usize,
    inner: Mutex<NeighborhoodState>,
}

impl Neighborhood {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            inner: Mutex::new(NeighborhoodState::default()),
        }
    }

    pub fn insert(&self, id: u32, distance: f32, status: NeighborStatus) -> bool {
        let mut inner = self.inner.lock();

        if let Some(existing) = inner.pool.iter_mut().find(|neighbor| neighbor.id == id) {
            if distance < existing.distance {
                existing.distance = distance;
                existing.status = status;
                return true;
            }
            return false;
        }

        if inner.pool.len() < self.capacity {
            inner.pool.push(Neighbor::new(id, distance, status));
            return true;
        }

        let Some((worst_idx, worst_neighbor)) = inner.pool.iter().enumerate().max_by(|lhs, rhs| {
            lhs.1
                .distance
                .total_cmp(&rhs.1.distance)
                .then_with(|| lhs.1.id.cmp(&rhs.1.id))
        }) else {
            return false;
        };

        if distance >= worst_neighbor.distance {
            return false;
        }

        inner.pool[worst_idx] = Neighbor::new(id, distance, status);
        true
    }

    pub fn snapshot(&self) -> Vec<Neighbor> {
        let mut snapshot = self.inner.lock().pool.clone();
        snapshot.sort_by(|lhs, rhs| {
            lhs.distance
                .total_cmp(&rhs.distance)
                .then_with(|| lhs.id.cmp(&rhs.id))
        });
        snapshot
    }

    pub fn contains(&self, id: u32) -> bool {
        self.inner
            .lock()
            .pool
            .iter()
            .any(|neighbor| neighbor.id == id)
    }

    pub fn seed_for_test(&self, neighbors: &[Neighbor]) {
        let mut inner = self.inner.lock();
        inner.pool = neighbors.iter().copied().take(self.capacity).collect();
    }

    pub fn rebuild_samples(&self, sample_count: usize) {
        let mut inner = self.inner.lock();
        let mut ordered = inner.pool.clone();
        ordered.sort_by(|lhs, rhs| {
            lhs.distance
                .total_cmp(&rhs.distance)
                .then_with(|| lhs.id.cmp(&rhs.id))
        });

        inner.nn_new.clear();
        inner.nn_old.clear();
        inner.rnn_new.clear();
        inner.rnn_old.clear();

        let mut new_seen = 0usize;
        for neighbor in ordered {
            match neighbor.status {
                NeighborStatus::New if new_seen < sample_count => {
                    inner.nn_new.push(neighbor.id);
                    new_seen += 1;
                }
                NeighborStatus::New | NeighborStatus::Old => inner.nn_old.push(neighbor.id),
            }
        }
    }

    pub fn sample_lists(&self) -> (Vec<u32>, Vec<u32>) {
        let inner = self.inner.lock();
        (inner.nn_new.clone(), inner.nn_old.clone())
    }

    pub fn promote_new_to_old(&self) {
        let mut inner = self.inner.lock();
        for neighbor in &mut inner.pool {
            if neighbor.status == NeighborStatus::New {
                neighbor.status = NeighborStatus::Old;
            }
        }
    }

    pub fn snapshot_ids(&self) -> Vec<u32> {
        self.snapshot()
            .into_iter()
            .map(|neighbor| neighbor.id)
            .collect()
    }
}
