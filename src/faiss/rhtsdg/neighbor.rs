#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Neighbor {
    pub id: u32,
    pub distance: f32,
    pub status: NeighborStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NeighborStatus {
    New = 0,
    Old = 1,
}

#[derive(Debug, Default, Clone)]
pub struct Neighborhood {
    pool: Vec<Neighbor>,
}

impl Neighborhood {
    pub fn new() -> Self {
        Self { pool: Vec::new() }
    }

    pub fn pool(&self) -> &[Neighbor] {
        &self.pool
    }
}
