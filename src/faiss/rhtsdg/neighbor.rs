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

        if let Some(existing_idx) = inner.pool.iter().position(|neighbor| neighbor.id == id) {
            if distance < inner.pool[existing_idx].distance {
                inner.pool.remove(existing_idx);
                insert_sorted(&mut inner.pool, Neighbor::new(id, distance, status));
                return true;
            }
            return false;
        }

        if inner.pool.len() >= self.capacity {
            let Some(worst_neighbor) = inner.pool.last() else {
                return false;
            };
            if is_not_better_candidate(id, distance, worst_neighbor) {
                return false;
            }
        }

        insert_sorted(&mut inner.pool, Neighbor::new(id, distance, status));
        if inner.pool.len() > self.capacity {
            inner.pool.pop();
        }
        true
    }

    pub fn snapshot(&self) -> Vec<Neighbor> {
        self.inner.lock().pool.clone()
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

        inner.nn_new.clear();
        inner.nn_old.clear();
        inner.rnn_new.clear();
        inner.rnn_old.clear();

        let mut new_seen = 0usize;
        for idx in 0..inner.pool.len() {
            let neighbor = inner.pool[idx];
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

    pub fn set_reverse_samples(&self, new_ids: Vec<u32>, old_ids: Vec<u32>) {
        let mut inner = self.inner.lock();
        inner.rnn_new = new_ids;
        inner.rnn_old = old_ids;
    }

    pub fn join_candidate_lists(&self) -> (Vec<u32>, Vec<u32>) {
        let inner = self.inner.lock();
        let mut nn_new = inner.nn_new.clone();
        let mut nn_old = inner.nn_old.clone();

        append_unique(&mut nn_new, &inner.rnn_new);
        append_unique_excluding(&mut nn_old, &inner.rnn_old, &nn_new);

        (nn_new, nn_old)
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

fn insert_sorted(pool: &mut Vec<Neighbor>, neighbor: Neighbor) {
    let insert_idx = pool
        .iter()
        .position(|existing| cmp_neighbor(&neighbor, existing).is_lt())
        .unwrap_or(pool.len());
    pool.insert(insert_idx, neighbor);
}

fn cmp_neighbor(lhs: &Neighbor, rhs: &Neighbor) -> std::cmp::Ordering {
    lhs.distance
        .total_cmp(&rhs.distance)
        .then_with(|| lhs.id.cmp(&rhs.id))
}

fn is_not_better_candidate(id: u32, distance: f32, worst_neighbor: &Neighbor) -> bool {
    match distance.total_cmp(&worst_neighbor.distance) {
        std::cmp::Ordering::Greater => true,
        std::cmp::Ordering::Equal => id >= worst_neighbor.id,
        std::cmp::Ordering::Less => false,
    }
}

fn append_unique(target: &mut Vec<u32>, source: &[u32]) {
    for &id in source {
        if !target.contains(&id) {
            target.push(id);
        }
    }
}

fn append_unique_excluding(target: &mut Vec<u32>, source: &[u32], excluded: &[u32]) {
    for &id in source {
        if excluded.contains(&id) || target.contains(&id) {
            continue;
        }
        target.push(id);
    }
}
