use crate::api::Result;

#[derive(Debug, Clone, PartialEq)]
pub struct SelectedPartitions {
    pub ids: Vec<u32>,
    pub distances: Vec<f32>,
}

pub trait IvfPartitionSelector: Send + Sync {
    fn select_partitions(&self, query: &[f32], nprobe: usize) -> SelectedPartitions;
}

pub trait IvfListScanner: Send + Sync {
    fn scan_list(
        &self,
        query: &[f32],
        list_id: u32,
        ids_out: &mut [i64],
        dists_out: &mut [f32],
    ) -> Result<usize>;
}
