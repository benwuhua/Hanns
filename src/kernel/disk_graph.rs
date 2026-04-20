use crate::api::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DiskGraphRuntimeConfig {
    pub beam_width: usize,
    pub search_list_size: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NodeRecord<'a> {
    pub node_id: u32,
    pub neighbor_ids: &'a [u32],
    pub code: Option<&'a [u8]>,
    pub raw_vector: Option<&'a [f32]>,
}

pub trait NodeReader: Send + Sync {
    fn read_node(&self, node_id: u32) -> Result<NodeRecord<'_>>;

    fn prefetch_nodes(&self, _node_ids: &[u32]) {}
}
